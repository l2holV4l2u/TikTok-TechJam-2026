import os
import time
import json
import random
import gc

import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 271828
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

N_THREADS = min(16, os.cpu_count() or 8)
torch.set_num_threads(N_THREADS)
try:
    torch.set_num_interop_threads(min(4, N_THREADS))
except RuntimeError:
    pass


FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "music_type",
    "upload_type",
    "user_active_degree",
    "fans_user_num_range",
    "onehot_feat3",
    "onehot_feat8",
]

EB_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "music_type",
    "upload_type",
    "tab",
    "onehot_feat3",
    "onehot_feat8",
]

EB_COEFFICIENTS = {
    "video_id": 1.00,
    "author_id": 0.45,
    "tag": 0.35,
    "duration_bucket": 0.30,
    "music_type": 0.20,
    "upload_type": 0.15,
    "tab": 0.25,
    "onehot_feat3": 0.25,
    "onehot_feat8": 0.25,
}

CARDS = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1], dtype=np.int64)
TOTAL_CARDINALITY = int(sum(CARDS))
N_FIELDS = len(FIELDS)


def categorical_matrix(split):
    return np.column_stack([
        np.asarray(split.X[name], dtype=np.int64)
        for name in FIELDS
    ])


def recency_weights(dates, half_life=3.0):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    date_to_age = {
        int(d): len(unique_dates) - 1 - i
        for i, d in enumerate(unique_dates)
    }
    ages = np.fromiter(
        (date_to_age[int(d)] for d in dates),
        dtype=np.float32,
        count=len(dates),
    )
    weights = np.exp2(-ages / np.float32(half_life)).astype(np.float32)
    weights /= weights.mean()
    return weights


def signed_log1p(x):
    x = np.asarray(x, dtype=np.float32)
    return np.sign(x) * np.log1p(np.abs(x))


def collect_raw_numeric(split_name, split):
    columns = []
    names = []

    for name in sorted(split.num):
        value = np.asarray(split.num[name], dtype=np.float32)
        columns.append(value)
        names.append("num_" + name)
        columns.append(np.isnan(value).astype(np.float32))
        names.append("missing_" + name)

    for key in ("video_id", "author_id"):
        hist = historical_features(split_name, key=key)
        for name in sorted(hist):
            value = np.asarray(hist[name], dtype=np.float32)
            if len(value) != len(split.user_id):
                raise ValueError("Historical feature length mismatch")
            columns.append(value)
            names.append("{}_{}".format(key, name))

    if not columns:
        return np.empty((len(split.user_id), 0), dtype=np.float32), names

    return np.column_stack(columns).astype(np.float32), names


def fit_numeric_transform(raw):
    transformed = raw.copy()
    for j in range(transformed.shape[1]):
        column = transformed[:, j]
        finite = np.isfinite(column)
        fill = float(np.median(column[finite])) if finite.any() else 0.0
        column[~finite] = fill
        transformed[:, j] = signed_log1p(column)

    mean = transformed.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = transformed.std(axis=0, dtype=np.float64).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-5)] = 1.0
    transformed = (transformed - mean) / std
    np.clip(transformed, -8.0, 8.0, out=transformed)
    return transformed.astype(np.float32), mean, std


def apply_numeric_transform(raw, mean, std):
    transformed = raw.copy()
    for j in range(transformed.shape[1]):
        column = transformed[:, j]
        finite = np.isfinite(column)
        # A standardized value of zero corresponds to the training center.
        # Before transformation, zero is a conservative train-independent fill.
        column[~finite] = 0.0
        transformed[:, j] = signed_log1p(column)
    transformed = (transformed - mean) / std
    np.clip(transformed, -8.0, 8.0, out=transformed)
    return transformed.astype(np.float32)


def score_scale(scores):
    scale = float(np.std(np.asarray(scores, dtype=np.float64)))
    if not np.isfinite(scale) or scale < 1e-7:
        return 1.0
    return scale


def empirical_bayes_tables(train, labels, weights, smoothing=45.0):
    global_rate = float(np.sum(weights * labels) / np.sum(weights))
    global_rate = np.clip(global_rate, 1e-5, 1.0 - 1e-5)
    global_logit = np.log(global_rate / (1.0 - global_rate))

    tables = {}
    for name in EB_FIELDS:
        ids = np.asarray(train.X[name], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[name])
        counts = np.bincount(ids, weights=weights, minlength=card)
        positives = np.bincount(
            ids,
            weights=weights * labels,
            minlength=card,
        )
        rate = (
            positives + smoothing * global_rate
        ) / (
            counts + smoothing
        )
        rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
        tables[name] = np.log(rate / (1.0 - rate)) - global_logit

    return global_logit, tables


def empirical_bayes_predict(split, global_logit, tables):
    score = np.full(len(split.user_id), global_logit, dtype=np.float64)
    coefficient_sum = 0.0
    contribution = np.zeros_like(score)

    for name in EB_FIELDS:
        coefficient = EB_COEFFICIENTS[name]
        ids = np.asarray(split.X[name], dtype=np.int64)
        contribution += coefficient * tables[name][ids]
        coefficient_sum += coefficient

    score += contribution / max(coefficient_sum, 1e-8)
    return score


class MMoE(nn.Module):
    def __init__(
        self,
        total_cardinality,
        offsets,
        n_fields,
        numeric_dim,
        n_tasks,
        embedding_dim=8,
        n_experts=4,
        expert_dim=80,
    ):
        super().__init__()
        self.register_buffer(
            "offsets",
            torch.as_tensor(offsets, dtype=torch.long),
        )
        self.embedding = nn.Embedding(total_cardinality, embedding_dim)
        self.linear = nn.Embedding(total_cardinality, 1)

        input_dim = n_fields * embedding_dim + numeric_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.06),
                nn.Linear(128, expert_dim),
                nn.ReLU(),
            )
            for _ in range(n_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, n_experts)
            for _ in range(n_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, 48),
                nn.ReLU(),
                nn.Linear(48, 1),
            )
            for _ in range(n_tasks)
        ])
        self.n_tasks = n_tasks
        self.biases = nn.Parameter(torch.zeros(n_tasks))

        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.linear.weight, std=0.01)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, categorical, numeric):
        indices = categorical + self.offsets
        embedding = self.embedding(indices).flatten(start_dim=1)
        shared_input = torch.cat([embedding, numeric], dim=1)

        expert_outputs = torch.stack(
            [expert(shared_input) for expert in self.experts],
            dim=1,
        )

        wide = self.linear(indices).sum(dim=1).squeeze(1)
        outputs = []
        for task in range(self.n_tasks):
            gate = torch.softmax(self.gates[task](shared_input), dim=1)
            representation = (
                expert_outputs * gate.unsqueeze(2)
            ).sum(dim=1)
            logit = self.towers[task](representation).squeeze(1)
            outputs.append(logit + wide + self.biases[task])

        return torch.stack(outputs, dim=1)


def fit_mmoe(model, x_cat, x_num, targets, weights, epochs=3):
    cat_tensor = torch.from_numpy(x_cat)
    num_tensor = torch.from_numpy(x_num)
    target_tensor = torch.from_numpy(targets)
    weight_tensor = torch.from_numpy(weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0e-3,
        weight_decay=3e-6,
    )
    generator = torch.Generator()
    generator.manual_seed(SEED)
    batch_size = 8192
    n_rows = len(x_cat)

    task_weights = torch.ones(targets.shape[1], dtype=torch.float32)
    if targets.shape[1] > 1:
        task_weights[1:] = 0.22
    task_weights /= task_weights.sum()

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_rows, generator=generator)
        accumulated = 0.0
        denominator = 0.0

        for start in range(0, n_rows, batch_size):
            idx = permutation[start:start + batch_size]
            logits = model(cat_tensor[idx], num_tensor[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits,
                target_tensor[idx],
                reduction="none",
            )
            row_loss = (losses * task_weights).sum(dim=1)
            wb = weight_tensor[idx]
            loss = (row_loss * wb).sum() / wb.sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            accumulated += float((row_loss.detach() * wb).sum())
            denominator += float(wb.sum())

        print(
            "FINDINGS family=mmoe epoch={} weighted_loss={:.6f}".format(
                epoch + 1,
                accumulated / max(denominator, 1.0),
            ),
            flush=True,
        )

    return model


@torch.inference_mode()
def predict_mmoe(model, x_cat, x_num, batch_size=32768):
    model.eval()
    output = np.empty(len(x_cat), dtype=np.float64)
    for start in range(0, len(x_cat), batch_size):
        end = min(start + batch_size, len(x_cat))
        categorical = torch.from_numpy(x_cat[start:end])
        numeric = torch.from_numpy(x_num[start:end])
        output[start:end] = (
            model(categorical, numeric)[:, 0].cpu().numpy()
        )
    return output


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
weights = recency_weights(train.date, half_life=3.0)

x_train_cat = categorical_matrix(train)
x_valid_cat = categorical_matrix(valid)

raw_train_num, numeric_names = collect_raw_numeric("train", train)
raw_valid_num, valid_numeric_names = collect_raw_numeric("valid", valid)
if numeric_names != valid_numeric_names:
    raise ValueError("Numeric/history feature schema mismatch")

x_train_num, numeric_mean, numeric_std = fit_numeric_transform(raw_train_num)
x_valid_num = apply_numeric_transform(
    raw_valid_num,
    numeric_mean,
    numeric_std,
)

print(
    "FINDINGS recency_half_life=3 effective_rows={:.0f} numeric_features={} weight_range={:.4f}/{:.4f}".format(
        float(weights.sum() ** 2 / np.square(weights).sum()),
        x_train_num.shape[1],
        float(weights.min()),
        float(weights.max()),
    ),
    flush=True,
)


# ---------------------------------------------------------------------------
# Family 1: empirical Bayes target statistics.
# Stable, smoothed entity/content propensities form the prediction directly.
# ---------------------------------------------------------------------------
eb_global_logit, eb_tables = empirical_bayes_tables(
    train,
    y_train,
    weights,
    smoothing=45.0,
)
eb_valid = empirical_bayes_predict(
    valid,
    eb_global_logit,
    eb_tables,
)


# ---------------------------------------------------------------------------
# Family 2: binary LightGBM.
# It models nonlinear thresholds and interactions among the same categorical
# inputs and train-only historical/numeric summaries.
# ---------------------------------------------------------------------------
x_train_lgb = np.concatenate(
    [x_train_cat.astype(np.float32), x_train_num],
    axis=1,
)
x_valid_lgb = np.concatenate(
    [x_valid_cat.astype(np.float32), x_valid_num],
    axis=1,
)

lgb_train = lgb.Dataset(
    x_train_lgb,
    label=y_train,
    weight=weights,
    categorical_feature=list(range(N_FIELDS)),
    free_raw_data=True,
)

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.045,
    "num_leaves": 63,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.88,
    "bagging_fraction": 0.88,
    "bagging_freq": 1,
    "lambda_l1": 0.08,
    "lambda_l2": 3.0,
    "max_bin": 127,
    "cat_smooth": 30.0,
    "cat_l2": 15.0,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "num_threads": N_THREADS,
    "verbose": -1,
}

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=260,
)
lgb_valid = lgb_model.predict(
    x_valid_lgb,
    raw_score=True,
).astype(np.float64)

del lgb_train, x_train_lgb
gc.collect()


# ---------------------------------------------------------------------------
# Family 3: MMoE.
# Auxiliary train outcomes are targets only, never row inputs. Shared experts
# can learn engagement structure while task gates retain a long-view-specific
# predictor.
# ---------------------------------------------------------------------------
auxiliary_names = [
    name for name in (
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
    )
    if name in train.aux
][:2]

target_columns = [y_train]
for name in auxiliary_names:
    value = np.asarray(train.aux[name], dtype=np.float32)
    value = np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
    value = (value > 0).astype(np.float32)
    target_columns.append(value)

mmoe_targets = np.column_stack(target_columns).astype(np.float32)
print(
    "FINDINGS mmoe_tasks={}".format(
        json.dumps(["long_view"] + auxiliary_names)
    ),
    flush=True,
)

mmoe_model = MMoE(
    TOTAL_CARDINALITY,
    OFFSETS,
    N_FIELDS,
    x_train_num.shape[1],
    mmoe_targets.shape[1],
)
fit_mmoe(
    mmoe_model,
    x_train_cat,
    x_train_num,
    mmoe_targets,
    weights,
    epochs=3,
)
mmoe_valid = predict_mmoe(
    mmoe_model,
    x_valid_cat,
    x_valid_num,
)


# Select family and incumbent blend using validation only.
shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are required")

inc_valid = np.load(inc_valid_path).astype(np.float64)
if len(inc_valid) != len(y_valid):
    raise ValueError("Incumbent validation prediction length mismatch")

families_valid = {
    "empirical_bayes": eb_valid,
    "lightgbm_binary": lgb_valid,
    "mmoe": mmoe_valid,
}

candidate_results = {}
best_primary = -np.inf
best_metrics = None
best_name = None
best_family = None
best_weight = None
best_valid_scores = None
best_raw_valid = None

inc_scale = score_scale(inc_valid)
normalized_inc_valid = inc_valid / inc_scale

for family_name, raw_valid in families_valid.items():
    raw_metrics = evaluate(valid.user_id, y_valid, raw_valid)
    candidate_results[family_name] = float(raw_metrics["primary"])

    if float(raw_metrics["primary"]) > best_primary:
        best_primary = float(raw_metrics["primary"])
        best_metrics = raw_metrics
        best_name = family_name
        best_family = family_name
        best_weight = 1.0
        best_valid_scores = raw_valid.copy()
        best_raw_valid = None

    own_scale = score_scale(raw_valid)
    normalized_own_valid = raw_valid / own_scale

    for own_weight in (0.10, 0.20, 0.35, 0.50, 0.65, 0.80):
        blended = (
            own_weight * normalized_own_valid
            + (1.0 - own_weight) * normalized_inc_valid
        )
        metrics = evaluate(valid.user_id, y_valid, blended)
        name = "{}_inc_blend_{:.2f}".format(
            family_name,
            own_weight,
        )
        candidate_results[name] = float(metrics["primary"])

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_metrics = metrics
            best_name = name
            best_family = family_name
            best_weight = own_weight
            best_valid_scores = blended.copy()
            best_raw_valid = raw_valid.copy()


# Load test only after all validation selection is complete. No test labels or
# auxiliary outcomes are accessed.
test = load("test")
x_test_cat = categorical_matrix(test)
raw_test_num, test_numeric_names = collect_raw_numeric("test", test)
if numeric_names != test_numeric_names:
    raise ValueError("Test numeric/history feature schema mismatch")
x_test_num = apply_numeric_transform(
    raw_test_num,
    numeric_mean,
    numeric_std,
)

if best_family == "empirical_bayes":
    own_test = empirical_bayes_predict(
        test,
        eb_global_logit,
        eb_tables,
    )
elif best_family == "lightgbm_binary":
    x_test_lgb = np.concatenate(
        [x_test_cat.astype(np.float32), x_test_num],
        axis=1,
    )
    own_test = lgb_model.predict(
        x_test_lgb,
        raw_score=True,
    ).astype(np.float64)
elif best_family == "mmoe":
    own_test = predict_mmoe(
        mmoe_model,
        x_test_cat,
        x_test_num,
    )
else:
    raise RuntimeError("Unknown selected family")

if best_weight == 1.0:
    best_test_scores = own_test
else:
    inc_test = np.load(inc_test_path).astype(np.float64)
    if len(inc_test) != len(test.user_id):
        raise ValueError("Incumbent test prediction length mismatch")

    selected_valid_raw = families_valid[best_family]
    own_scale = score_scale(selected_valid_raw)
    best_test_scores = (
        best_weight * (own_test / own_scale)
        + (1.0 - best_weight) * (inc_test / inc_scale)
    )


out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test_scores, dtype=np.float64),
    )
    if best_raw_valid is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

print(
    "FINDINGS selected={} family={} own_weight={:.2f}".format(
        best_name,
        best_family,
        best_weight,
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(
        candidate_results,
        sort_keys=True,
    ),
    flush=True,
)

elapsed = time.time() - START
print(
    'METRICS {{"primary": {:.10f}, "gauc": {:.10f}, "ndcg@5": {:.10f}, "gpu_seconds": {:.4f}}}'.format(
        float(best_metrics["primary"]),
        float(best_metrics["gauc"]),
        float(best_metrics["ndcg@5"]),
        elapsed,
    )
)