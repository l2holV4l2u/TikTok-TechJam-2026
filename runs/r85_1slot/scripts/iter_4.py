import os
import gc
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 19427
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 32768
EPOCHS = 7
HALF_LIFE_DAYS = 5.0

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
    "video_type",
    "user_active_degree",
    "onehot_feat3",
]

ENTITY_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "upload_type",
]

PAIR_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "upload_type",
]

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))


def load_arrays(split_name, with_labels):
    s = load(split_name)
    x = np.column_stack([s.X[f] for f in FIELDS]).astype(np.int64, copy=False)
    users = np.asarray(s.user_id, dtype=np.int64)
    dates = np.asarray(s.date, dtype=np.int32)
    field_arrays = {
        f: np.asarray(s.X[f], dtype=np.int64) for f in ENTITY_FIELDS
    }
    if with_labels:
        y = np.asarray(s.y, dtype=np.float32)
        del s
        gc.collect()
        return x, y, users, dates, field_arrays
    del s
    gc.collect()
    return x, users, dates, field_arrays


def yyyymmdd_to_day(dates):
    dates = np.asarray(dates, dtype=np.int64)
    years = dates // 10000
    months = (dates // 100) % 100
    days = dates % 100

    # All supplied dates are in April/May 2022. This monotone mapping is
    # sufficient for exact day differences over the benchmark interval.
    return (years - 2022) * 365 + (months - 4) * 30 + days


def training_weights(users, dates):
    day = yyyymmdd_to_day(dates)
    age = np.max(day) - day
    recency = np.exp2(-age.astype(np.float64) / HALF_LIFE_DAYS)

    counts = np.bincount(users)
    activity = 1.0 / np.sqrt(np.maximum(counts[users], 1))
    w = recency * activity
    w /= np.mean(w)
    return w.astype(np.float32)


cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets_np = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))
n_fields = len(FIELDS)


class NFM(nn.Module):
    """Wide model plus nonlinear bi-interaction pooling."""

    def __init__(self, base_logit, rank=16):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.factors = nn.Embedding(total_cardinality, rank)
        self.mlp = nn.Sequential(
            nn.Linear(rank, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.register_buffer(
            "offsets", torch.as_tensor(offsets_np, dtype=torch.long)
        )
        self.register_buffer(
            "base_logit", torch.tensor(float(base_logit), dtype=torch.float32)
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.02)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        ids = x + self.offsets
        wide = self.linear(ids).squeeze(-1).sum(dim=1)
        v = self.factors(ids)
        pooled = 0.5 * (
            v.sum(dim=1).square() - v.square().sum(dim=1)
        )
        nonlinear = self.mlp(pooled).squeeze(1)
        return self.base_logit + wide + nonlinear


class DCN(nn.Module):
    """Embedding concatenation followed by explicit vector cross layers."""

    def __init__(self, base_logit, rank=8, cross_layers=3):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, rank)
        dim = n_fields * rank
        self.cross_w = nn.ParameterList(
            [nn.Parameter(torch.empty(dim)) for _ in range(cross_layers)]
        )
        self.cross_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for _ in range(cross_layers)]
        )
        self.deep = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.output = nn.Linear(dim + 32, 1)
        self.register_buffer(
            "offsets", torch.as_tensor(offsets_np, dtype=torch.long)
        )
        self.register_buffer(
            "base_logit", torch.tensor(float(base_logit), dtype=torch.float32)
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for w in self.cross_w:
            nn.init.normal_(w, mean=0.0, std=0.02)
        for layer in self.deep:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        ids = x + self.offsets
        wide = self.linear(ids).squeeze(-1).sum(dim=1)
        x0 = self.embedding(ids).reshape(x.shape[0], -1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scalar = torch.sum(cross * w, dim=1, keepdim=True)
            cross = x0 * scalar + b + cross
        deep = self.deep(x0)
        nonlinear = self.output(torch.cat([cross, deep], dim=1)).squeeze(1)
        return self.base_logit + wide + nonlinear


def make_model(family, y):
    p = float(np.clip(np.mean(y), 1e-5, 1.0 - 1e-5))
    base_logit = np.log(p / (1.0 - p))
    if family == "nfm":
        return NFM(base_logit)
    if family == "dcn":
        return DCN(base_logit)
    raise ValueError(family)


def fit_neural(family, x, y, users, dates):
    seed = SEED + (101 if family == "nfm" else 211)
    torch.manual_seed(seed)
    model = make_model(family, y)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=1e-6
    )

    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    wt = torch.from_numpy(training_weights(users, dates))
    generator = torch.Generator()
    generator.manual_seed(seed)

    for _ in range(EPOCHS):
        permutation = torch.randperm(x.shape[0], generator=generator)
        model.train()
        for start in range(0, x.shape[0], BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            logits = model(xt[idx])
            losses = F.binary_cross_entropy_with_logits(
                logits, yt[idx], reduction="none"
            )
            loss = torch.mean(losses * wt[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.no_grad()
def predict_neural(model, x):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float32)
    for start in range(0, x.shape[0], PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, x.shape[0])
        result[start:end] = model(
            torch.from_numpy(x[start:end])
        ).cpu().numpy()
    return result


def clipped_logit(p):
    p = np.clip(p, 0.02, 0.98)
    return np.log(p) - np.log1p(-p)


def aggregate_lookup(keys_train, y, weights, keys_query, alpha, prior):
    order = np.argsort(keys_train, kind="mergesort")
    sorted_keys = keys_train[order]
    sorted_yw = y[order].astype(np.float64) * weights[order]
    sorted_w = weights[order].astype(np.float64)

    unique_keys, starts = np.unique(sorted_keys, return_index=True)
    sums = np.add.reduceat(sorted_yw, starts)
    counts = np.add.reduceat(sorted_w, starts)

    idx = np.searchsorted(unique_keys, keys_query)
    matched = idx < unique_keys.size
    clipped_idx = np.minimum(idx, unique_keys.size - 1)
    matched &= unique_keys[clipped_idx] == keys_query

    out_sums = np.zeros(keys_query.shape[0], dtype=np.float64)
    out_counts = np.zeros(keys_query.shape[0], dtype=np.float64)
    out_sums[matched] = sums[clipped_idx[matched]]
    out_counts[matched] = counts[clipped_idx[matched]]

    return (out_sums + alpha * prior) / (out_counts + alpha)


def empirical_bayes_scores(
    train_users,
    train_fields,
    y,
    train_dates,
    query_users,
    query_fields,
):
    weights = training_weights(train_users, train_dates).astype(np.float64)
    global_rate = float(
        np.sum(weights * y.astype(np.float64)) / np.sum(weights)
    )

    entity_logits = []
    pair_logits = []

    for field in ENTITY_FIELDS:
        tr_values = train_fields[field]
        qu_values = query_fields[field]

        entity_rate = aggregate_lookup(
            tr_values,
            y,
            weights,
            qu_values,
            alpha=30.0,
            prior=global_rate,
        )
        entity_logits.append(clipped_logit(entity_rate))

        cardinality = int(FEATURE_CARDINALITIES[field])
        tr_pair = (
            train_users.astype(np.int64) * np.int64(cardinality)
            + tr_values.astype(np.int64)
        )
        qu_pair = (
            query_users.astype(np.int64) * np.int64(cardinality)
            + qu_values.astype(np.int64)
        )
        pair_rate = aggregate_lookup(
            tr_pair,
            y,
            weights,
            qu_pair,
            alpha=6.0,
            prior=global_rate,
        )
        pair_logits.append(clipped_logit(pair_rate))

    global_score = np.mean(np.vstack(entity_logits), axis=0)
    personal_score = (
        0.45 * global_score
        + 0.55 * np.mean(np.vstack(pair_logits), axis=0)
    )
    return global_score.astype(np.float64), personal_score.astype(np.float64)


x_train, y_train, train_users, train_dates, train_fields = load_arrays(
    "train", with_labels=True
)
x_valid, y_valid, valid_users, valid_dates, valid_fields = load_arrays(
    "valid", with_labels=True
)
valid_labels = y_valid.astype(np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path):
    raise FileNotFoundError("Trusted incumbent validation scores are missing")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if inc_valid.shape[0] != y_valid.shape[0]:
    raise ValueError("Incumbent validation length mismatch")
inc_valid_scale = max(float(np.std(inc_valid)), 1e-8)
inc_valid_norm = inc_valid / inc_valid_scale

raw_predictions = {}
models = {}

eb_global, eb_personal = empirical_bayes_scores(
    train_users,
    train_fields,
    y_train,
    train_dates,
    valid_users,
    valid_fields,
)
raw_predictions["eb_global"] = eb_global
raw_predictions["eb_personal"] = eb_personal

for family in ["nfm", "dcn"]:
    model = fit_neural(
        family, x_train, y_train, train_users, train_dates
    )
    raw_predictions[family] = predict_neural(
        model, x_valid
    ).astype(np.float64)
    models[family] = model

candidate_scores = {}
candidate_metrics = {}
candidate_blend_weights = {}
candidate_scales = {}

best_name = None
best_base_family = None
best_weight = 1.0
best_scores = None
best_raw = None
best_metrics = None

for family, raw in raw_predictions.items():
    metrics_raw = evaluate(valid_users, valid_labels, raw)
    candidate_scores[f"{family}_raw"] = float(metrics_raw["primary"])
    candidate_metrics[f"{family}_raw"] = metrics_raw

    scale = max(float(np.std(raw)), 1e-8)
    candidate_scales[family] = scale
    raw_norm = raw / scale

    local_best_metrics = metrics_raw
    local_best_scores = raw
    local_best_weight = 1.0

    for w in np.linspace(0.0, 1.0, 21):
        blended = w * raw_norm + (1.0 - w) * inc_valid_norm
        metrics = evaluate(valid_users, valid_labels, blended)
        if float(metrics["primary"]) > float(local_best_metrics["primary"]):
            local_best_metrics = metrics
            local_best_scores = blended.copy()
            local_best_weight = float(w)

    candidate_scores[f"{family}_blend"] = float(
        local_best_metrics["primary"]
    )
    candidate_blend_weights[family] = local_best_weight

    if best_metrics is None or float(local_best_metrics["primary"]) > float(
        best_metrics["primary"]
    ):
        best_name = (
            f"{family}_raw"
            if local_best_weight == 1.0
            else f"{family}_blend"
        )
        best_base_family = family
        best_weight = local_best_weight
        best_scores = np.asarray(local_best_scores, dtype=np.float64)
        best_raw = raw.copy()
        best_metrics = local_best_metrics

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "selected": best_name,
            "selected_family": best_base_family,
            "own_weight": best_weight,
            "blend_weights": candidate_blend_weights,
            "half_life_days": HALF_LIFE_DAYS,
            "activity_weight": "inverse_sqrt_user_rows",
        },
        sort_keys=True,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_weight < 1.0 - 1e-12:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

# Release non-selected validation models before the required refit.
for family in list(models.keys()):
    del models[family]
models.clear()
gc.collect()

# Refit the selected recipe on train + validation, then score test.
x_refit = np.concatenate([x_train, x_valid], axis=0)
y_refit = np.concatenate([y_train, y_valid], axis=0)
users_refit = np.concatenate([train_users, valid_users], axis=0)
dates_refit = np.concatenate([train_dates, valid_dates], axis=0)
fields_refit = {
    field: np.concatenate(
        [train_fields[field], valid_fields[field]], axis=0
    )
    for field in ENTITY_FIELDS
}

x_test, test_users, test_dates, test_fields = load_arrays(
    "test", with_labels=False
)

if best_base_family in ("nfm", "dcn"):
    refit_model = fit_neural(
        best_base_family,
        x_refit,
        y_refit,
        users_refit,
        dates_refit,
    )
    own_test = predict_neural(refit_model, x_test).astype(np.float64)
    del refit_model
elif best_base_family in ("eb_global", "eb_personal"):
    test_eb_global, test_eb_personal = empirical_bayes_scores(
        users_refit,
        fields_refit,
        y_refit,
        dates_refit,
        test_users,
        test_fields,
    )
    own_test = (
        test_eb_global
        if best_base_family == "eb_global"
        else test_eb_personal
    )
else:
    raise ValueError("Unknown selected family")

if best_weight < 1.0 - 1e-12:
    if not os.path.exists(inc_test_path):
        raise FileNotFoundError("Trusted incumbent test scores are missing")
    inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
    if inc_test.shape[0] != own_test.shape[0]:
        raise ValueError("Incumbent test length mismatch")

    own_scale = candidate_scales[best_base_family]
    test_scores = (
        best_weight * (own_test / own_scale)
        + (1.0 - best_weight) * (inc_test / inc_valid_scale)
    )
else:
    test_scores = own_test

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)