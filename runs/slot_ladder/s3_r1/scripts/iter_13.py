import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 8675309
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat8",
]
NUMERIC_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1], dtype=np.int64)
TOTAL_CARDINALITY = int(sum(CARDS))
N_FIELDS = len(FIELDS)

BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768
EPOCHS = 3
LR = 0.0022


def make_categorical(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[field], dtype=np.int64) + offset
            for field, offset in zip(FIELDS, OFFSETS)
        ]),
        dtype=np.int64,
    )


def fit_numeric_transform(split):
    centers = []
    scales = []
    for field in NUMERIC_FIELDS:
        x = np.asarray(split.num[field], dtype=np.float64)
        finite = np.isfinite(x)
        clean = np.zeros_like(x)
        clean[finite] = np.sign(x[finite]) * np.log1p(np.abs(x[finite]))
        center = float(np.mean(clean[finite])) if finite.any() else 0.0
        scale = float(np.std(clean[finite])) if finite.any() else 1.0
        centers.append(center)
        scales.append(max(scale, 1e-6))
    return np.asarray(centers), np.asarray(scales)


def make_numeric(split, centers, scales):
    columns = []
    for j, field in enumerate(NUMERIC_FIELDS):
        x = np.asarray(split.num[field], dtype=np.float64)
        finite = np.isfinite(x)
        z = np.zeros_like(x)
        z[finite] = np.sign(x[finite]) * np.log1p(np.abs(x[finite]))
        z[finite] = (z[finite] - centers[j]) / scales[j]
        z = np.clip(z, -8.0, 8.0)
        columns.append(z.astype(np.float32))
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def standardize_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    if not finite.all():
        replacement = float(np.median(scores[finite])) if finite.any() else 0.0
        scores = np.nan_to_num(
            scores, nan=replacement, posinf=replacement, neginf=replacement
        )
    scale = float(np.std(scores))
    if scale < 1e-12:
        return scores - float(np.mean(scores))
    return (scores - float(np.mean(scores))) / scale


class AdditiveGAM(nn.Module):
    """A stationary generalized additive model over categorical fields."""

    def __init__(self):
        super().__init__()
        self.effects = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.numeric = nn.Linear(len(NUMERIC_FIELDS), 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.effects.weight)
        nn.init.zeros_(self.numeric.weight)

    def forward(self, x, numeric, regime=None):
        return (
            self.effects(x).sum(dim=1).squeeze(1)
            + self.numeric(numeric).squeeze(1)
            + self.bias
        )


class ThirdOrderFM(nn.Module):
    """Explicit elementary-symmetric interactions through order three."""

    def __init__(self, dim=14):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, dim)
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.numeric = nn.Linear(len(NUMERIC_FIELDS), 1, bias=False)
        self.order_scale = nn.Parameter(torch.tensor([1.0, 1.0]))
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.035)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.numeric.weight)

    def forward(self, x, numeric, regime=None):
        e = self.embedding(x)
        s1 = e.sum(dim=1)
        s2 = e.square().sum(dim=1)
        s3 = e.pow(3).sum(dim=1)

        order2 = 0.5 * (s1.square() - s2)
        order3 = (
            s1.pow(3) - 3.0 * s1 * s2 + 2.0 * s3
        ) / 6.0

        interaction = (
            self.order_scale[0] * order2.sum(dim=1)
            + self.order_scale[1] * order3.sum(dim=1)
        )
        return (
            self.linear(x).sum(dim=1).squeeze(1)
            + self.numeric(numeric).squeeze(1)
            + interaction
            + self.bias
        )


class TemporalExpertModel(nn.Module):
    """
    Shared representation with separate early/middle/late scoring functions.
    Evaluation extrapolates the late-training expert.
    """

    def __init__(self, dim=8):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, dim)
        input_dim = N_FIELDS * dim + len(NUMERIC_FIELDS)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 72),
                nn.ReLU(),
                nn.Linear(72, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(3)
        ])
        self.wide = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.expert_bias = nn.Parameter(torch.zeros(3))
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.03)
        nn.init.zeros_(self.wide.weight)

    def forward(self, x, numeric, regime=None):
        e = self.embedding(x)
        features = torch.cat([e.flatten(1), numeric], dim=1)
        all_outputs = torch.cat(
            [expert(features) for expert in self.experts], dim=1
        )

        if regime is None:
            selected = all_outputs[:, 2] + self.expert_bias[2]
        else:
            selected = all_outputs.gather(1, regime[:, None]).squeeze(1)
            selected = selected + self.expert_bias.index_select(0, regime)

        return self.wide(x).sum(dim=1).squeeze(1) + selected


def train_models(models, x_np, numeric_np, y_np, regimes_np):
    x = torch.from_numpy(x_np)
    numeric = torch.from_numpy(numeric_np)
    labels = torch.from_numpy(y_np)
    regimes = torch.from_numpy(regimes_np)

    parameters = []
    for model in models.values():
        parameters.extend(model.parameters())

    optimizer = torch.optim.AdamW(
        parameters, lr=LR, weight_decay=2e-6
    )
    rng = np.random.default_rng(SEED + 101)
    n = x_np.shape[0]

    for epoch in range(EPOCHS):
        permutation = rng.permutation(n)
        losses = {name: 0.0 for name in models}
        n_batches = 0

        for start in range(0, n, BATCH_SIZE):
            idx_np = permutation[start:start + BATCH_SIZE]
            idx = torch.from_numpy(idx_np)
            xb = x.index_select(0, idx)
            nb = numeric.index_select(0, idx)
            yb = labels.index_select(0, idx)
            rb = regimes.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            batch_losses = {}

            for name, model in models.items():
                if name == "temporal_experts":
                    logits = model(xb, nb, rb)
                else:
                    logits = model(xb, nb, None)

                loss = F.binary_cross_entropy_with_logits(logits, yb)
                batch_losses[name] = loss
                total_loss = total_loss + loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()

            for name, loss in batch_losses.items():
                losses[name] += float(loss.detach())
            n_batches += 1

        print(
            "FINDINGS "
            + json.dumps({
                "epoch": epoch + 1,
                "mean_losses": {
                    name: losses[name] / max(n_batches, 1)
                    for name in models
                },
            }, sort_keys=True)
        )


def predict(model, x_np, numeric_np):
    result = np.empty(x_np.shape[0], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, x_np.shape[0], PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, x_np.shape[0])
            xb = torch.from_numpy(x_np[start:end])
            nb = torch.from_numpy(numeric_np[start:end])
            result[start:end] = model(xb, nb, None).cpu().numpy()
    return result


train = load("train")
x_train = make_categorical(train)
numeric_centers, numeric_scales = fit_numeric_transform(train)
numeric_train = make_numeric(train, numeric_centers, numeric_scales)
y_train = np.asarray(train.y, dtype=np.float32)

dates = np.asarray(train.date, dtype=np.int64)
unique_dates = np.sort(np.unique(dates))
date_index = np.searchsorted(unique_dates, dates)

# Thirteen dates become 4 early, 4 middle, and 5 late dates.
regimes_train = np.where(
    date_index < 4, 0, np.where(date_index < 8, 1, 2)
).astype(np.int64)

print(
    "FINDINGS "
    + json.dumps({
        "fields": FIELDS,
        "regime_rows": [
            int(np.sum(regimes_train == i)) for i in range(3)
        ],
        "regime_positive_rates": [
            float(y_train[regimes_train == i].mean()) for i in range(3)
        ],
    }, sort_keys=True)
)

models = {
    "stationary_additive_gam": AdditiveGAM(),
    "third_order_fm": ThirdOrderFM(),
    "temporal_experts": TemporalExpertModel(),
}

train_models(
    models,
    x_train,
    numeric_train,
    y_train,
    regimes_train,
)

valid = load("valid")
x_valid = make_categorical(valid)
numeric_valid = make_numeric(valid, numeric_centers, numeric_scales)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are unavailable")

inc_valid_raw = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_valid = standardize_scores(inc_valid_raw)

valid_raw_predictions = {}
candidate_scores = {}
candidate_metrics = {}
blend_alphas = [0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0]

best_primary = -np.inf
best_name = None
best_family = None
best_alpha = None
best_valid_scores = None

inc_metrics = evaluate(valid_users, y_valid, inc_valid)
candidate_scores["trusted_incumbent"] = float(inc_metrics["primary"])

for family, model in models.items():
    raw = predict(model, x_valid, numeric_valid).astype(np.float64)
    valid_raw_predictions[family] = raw
    own = standardize_scores(raw)

    own_metrics = evaluate(valid_users, y_valid, own)
    candidate_scores[family] = float(own_metrics["primary"])

    for alpha in blend_alphas:
        blended = alpha * own + (1.0 - alpha) * inc_valid
        metrics = evaluate(valid_users, y_valid, blended)
        name = family + "_blend_" + str(alpha)
        candidate_scores[name] = float(metrics["primary"])
        candidate_metrics[name] = metrics

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_name = name
            best_family = family
            best_alpha = float(alpha)
            best_valid_scores = blended.copy()

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

best_metrics = evaluate(valid_users, y_valid, best_valid_scores)
print(
    "FINDINGS "
    + json.dumps({
        "selected": best_name,
        "selected_family": best_family,
        "own_weight": best_alpha,
        "own_primary": candidate_scores[best_family],
        "incumbent_primary": candidate_scores["trusted_incumbent"],
    }, sort_keys=True)
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(valid_raw_predictions[best_family], dtype=np.float64),
    )

test = load("test")
x_test = make_categorical(test)
numeric_test = make_numeric(test, numeric_centers, numeric_scales)

own_test_raw = predict(
    models[best_family], x_test, numeric_test
).astype(np.float64)
own_test = standardize_scores(own_test_raw)

inc_test_raw = np.asarray(np.load(inc_test_path), dtype=np.float64)
inc_test = standardize_scores(inc_test_raw)
test_scores = best_alpha * own_test + (1.0 - best_alpha) * inc_test

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)