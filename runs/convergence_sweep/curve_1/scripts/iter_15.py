import os
import time
import json
import random
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 8675309
THREADS = min(8, os.cpu_count() or 8)
BATCH_SIZE = 16384

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

FIELDS = [
    "author_id", "duration_bucket", "fans_user_num_range",
    "follow_user_num_range", "friend_user_num_range", "hour",
    "is_live_streamer", "is_lowactive_period", "is_video_author",
    "music_type", "onehot_feat0", "onehot_feat1", "onehot_feat10",
    "onehot_feat11", "onehot_feat12", "onehot_feat13",
    "onehot_feat14", "onehot_feat15", "onehot_feat16",
    "onehot_feat17", "onehot_feat2", "onehot_feat3",
    "onehot_feat4", "onehot_feat5", "onehot_feat6",
    "onehot_feat7", "onehot_feat8", "onehot_feat9",
    "register_days_bucket", "register_days_range", "tab", "tag",
    "upload_type", "user_active_degree", "user_id", "video_id",
    "video_type",
]

NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]


def recency_weights(dates, half_life=3.0):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    date_to_age = {int(d): len(unique_dates) - 1 - i
                   for i, d in enumerate(unique_dates)}
    age = np.fromiter(
        (date_to_age[int(d)] for d in dates),
        dtype=np.float32,
        count=len(dates),
    )
    weights = np.exp2(-age / np.float32(half_life))
    return (weights / weights.mean()).astype(np.float32)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    group_starts = np.repeat(starts, lengths)
    group_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.float64) - group_starts

    values = np.full(n, 0.5, dtype=np.float64)
    multi = group_lengths > 1
    values[multi] = positions[multi] / (group_lengths[multi] - 1.0)

    result = np.empty(n, dtype=np.float64)
    result[order] = values
    return result


class TargetFeatureBuilder:
    def __init__(self, fields, smoothings=(6.0, 25.0, 100.0)):
        self.fields = list(fields)
        self.smoothings = tuple(float(x) for x in smoothings)
        self.tables = {}
        self.prior = 0.0
        self.numeric_medians = {}

    def fit(self, split, labels, weights):
        labels = np.asarray(labels, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        self.prior = float(np.sum(labels * weights) / np.sum(weights))

        for field in self.fields:
            ids = np.asarray(split.X[field], dtype=np.int64)
            cardinality = FEATURE_CARDINALITIES[field]
            count = np.bincount(
                ids, weights=weights, minlength=cardinality
            ).astype(np.float64)
            positive = np.bincount(
                ids, weights=weights * labels, minlength=cardinality
            ).astype(np.float64)
            self.tables[field] = (count, positive)

        for field in NUM_FIELDS:
            raw = np.asarray(split.num[field], dtype=np.float32)
            finite = np.isfinite(raw)
            self.numeric_medians[field] = (
                float(np.median(raw[finite])) if np.any(finite) else 0.0
            )
        return self

    def transform(self, split, labels=None, weights=None):
        train_mode = labels is not None
        if train_mode:
            labels = np.asarray(labels, dtype=np.float64)
            weights = np.asarray(weights, dtype=np.float64)

        columns = []
        for field in self.fields:
            ids = np.asarray(split.X[field], dtype=np.int64)
            count_table, positive_table = self.tables[field]
            count = count_table[ids].copy()
            positive = positive_table[ids].copy()

            if train_mode:
                count -= weights
                positive -= weights * labels

            count = np.maximum(count, 0.0)
            for smoothing in self.smoothings:
                rate = (
                    positive + smoothing * self.prior
                ) / np.maximum(count + smoothing, 1e-8)
                columns.append(rate.astype(np.float32))
            columns.append(np.log1p(count).astype(np.float32))

        for field in NUM_FIELDS:
            raw = np.asarray(split.num[field], dtype=np.float32)
            missing = ~np.isfinite(raw)
            clean = np.where(missing, self.numeric_medians[field], raw)
            signed_log = np.sign(clean) * np.log1p(np.abs(clean))
            columns.append(signed_log.astype(np.float32))
            columns.append(missing.astype(np.float32))

        hour = np.asarray(split.X["hour"], dtype=np.float32)
        columns.append(np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32))
        columns.append(np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32))

        return np.column_stack(columns).astype(np.float32, copy=False)


def history_matrix(split_name):
    columns = []
    schema = []
    split = None

    for entity in ("video_id", "author_id"):
        values = historical_features(split_name, key=entity)
        for name in sorted(values):
            x = np.asarray(values[name], dtype=np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            x = np.sign(x) * np.log1p(np.abs(x))
            columns.append(x.astype(np.float32))
            schema.append(entity + ":" + name)

    if not columns:
        split = load(split_name)
        return np.empty((len(split.user_id), 0), dtype=np.float32), schema
    return np.column_stack(columns).astype(np.float32, copy=False), schema


def build_matrix(split, split_name, builder, labels=None, weights=None):
    base = builder.transform(split, labels=labels, weights=weights)
    hist, schema = history_matrix(split_name)
    matrix = np.concatenate([base, hist], axis=1).astype(
        np.float32, copy=False
    )
    matrix = np.nan_to_num(
        matrix, nan=0.0, posinf=20.0, neginf=-20.0
    ).astype(np.float32, copy=False)
    return matrix, schema


class LinearGLM(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear = nn.Linear(dimension, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


class FourierKernelClassifier(nn.Module):
    def __init__(self, input_dim, fourier_dim=80):
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(SEED + 101)
        projection = torch.randn(
            input_dim, fourier_dim, generator=generator
        ) / np.sqrt(max(input_dim, 1))
        phase = 2.0 * np.pi * torch.rand(
            fourier_dim, generator=generator
        )
        self.register_buffer("projection", projection)
        self.register_buffer("phase", phase)
        self.output = nn.Linear(fourier_dim * 2, 1)

    def forward(self, x):
        z = x @ self.projection + self.phase
        features = torch.cat([torch.cos(z), torch.sin(z)], dim=1)
        features = features / np.sqrt(self.projection.shape[1])
        return self.output(features).squeeze(1)


def fit_torch_model(model, x, labels, weights, mean, std,
                    epochs, learning_rate, name):
    labels = np.asarray(labels, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    rng = np.random.RandomState(SEED + len(name) * 17)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=2e-5
    )

    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(labels))
        weighted_loss = 0.0
        total_rows = 0

        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb = torch.from_numpy(
                ((x[idx] - mean) / std).astype(np.float32, copy=False)
            )
            yb = torch.from_numpy(labels[idx])
            wb = torch.from_numpy(weights[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = torch.sum(losses * wb) / torch.sum(wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            weighted_loss += float(loss.detach()) * len(idx)
            total_rows += len(idx)

        print(
            "FINDINGS {} epoch={} loss={:.6f}".format(
                name, epoch + 1, weighted_loss / max(total_rows, 1)
            ),
            flush=True,
        )
    return model


@torch.no_grad()
def predict_torch(model, x, mean, std):
    model.eval()
    output = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), 32768):
        end = min(start + 32768, len(x))
        xb = torch.from_numpy(
            ((x[start:end] - mean) / std).astype(np.float32, copy=False)
        )
        output[start:end] = model(xb).numpy().astype(np.float64)
    return output


def load_incumbent():
    shared = os.environ.get("SHARED_ARTIFACTS", "")
    valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
    test_path = os.path.join(shared, "incumbent_test_scores.npy")
    if not os.path.exists(valid_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Trusted incumbent predictions are unavailable")
    return (
        np.load(valid_path).astype(np.float64),
        np.load(test_path).astype(np.float64),
    )


def score_primary(users, labels, scores):
    return float(evaluate(users, labels, scores)["primary"])


train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
weights = recency_weights(train.date, half_life=3.0)

builder = TargetFeatureBuilder(FIELDS).fit(train, train_y, weights)
x_train, train_schema = build_matrix(
    train, "train", builder, labels=train_y, weights=weights
)
x_valid, valid_schema = build_matrix(valid, "valid", builder)

if train_schema != valid_schema:
    raise RuntimeError("Historical feature schemas do not match")

print(
    "FINDINGS feature_dimension={} history_dimension={} recency_weight_min={:.4f}".format(
        x_train.shape[1], len(train_schema), float(weights.min())
    ),
    flush=True,
)

mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
std = np.maximum(std, np.float32(1e-3))

# Family 1: globally linear generalized linear model. It cannot memorize
# arbitrary high-order crosses and therefore acts as a stationarity-biased model.
linear_model = fit_torch_model(
    LinearGLM(x_train.shape[1]),
    x_train, train_y, weights, mean, std,
    epochs=3, learning_rate=2.5e-3, name="linear_glm",
)
linear_valid = predict_torch(linear_model, x_valid, mean, std)

# Family 2: random Fourier features approximate a stationary RBF kernel,
# producing smooth nonlinear interactions unlike either a tree or deep CTR net.
kernel_model = fit_torch_model(
    FourierKernelClassifier(x_train.shape[1], fourier_dim=80),
    x_train, train_y, weights, mean, std,
    epochs=3, learning_rate=3.0e-3, name="fourier_kernel",
)
kernel_valid = predict_torch(kernel_model, x_valid, mean, std)

# Family 3: depth-one gradient boosting is an additive boosted GAM. Each tree
# forms a univariate response and cannot create non-stationary identity crosses.
gam_train = lgb.Dataset(
    x_train, label=train_y, weight=weights, free_raw_data=False
)
gam_model = lgb.train(
    {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.045,
        "num_leaves": 2,
        "max_depth": 1,
        "min_data_in_leaf": 900,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 3.0,
        "max_bin": 127,
        "num_threads": THREADS,
        "seed": SEED,
        "verbose": -1,
    },
    gam_train,
    num_boost_round=260,
)
gam_valid = gam_model.predict(x_valid).astype(np.float64)

inc_valid, inc_test = load_incumbent()
if len(inc_valid) != len(valid_y):
    raise RuntimeError("Incumbent validation length mismatch")

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
family_valid = {
    "linear_glm": linear_valid,
    "fourier_kernel": kernel_valid,
    "boosted_gam": gam_valid,
}

candidate_scores = {
    "trusted_incumbent": score_primary(
        valid.user_id, valid_y, inc_valid_rank
    )
}
best_primary = candidate_scores["trusted_incumbent"]
best_family_name = "linear_glm"
best_alpha = 0.0
best_valid = inc_valid_rank.copy()
best_raw_valid = within_user_rank(valid.user_id, linear_valid)

alphas = [0.10, 0.20, 0.35, 0.50, 0.70, 1.00]

for name, raw_scores in family_valid.items():
    ranked = within_user_rank(valid.user_id, raw_scores)
    standalone = score_primary(valid.user_id, valid_y, ranked)
    candidate_scores[name] = standalone

    if standalone > best_primary:
        best_primary = standalone
        best_family_name = name
        best_alpha = 1.0
        best_valid = ranked.copy()
        best_raw_valid = ranked.copy()

    best_family_blend = -1.0
    for alpha in alphas:
        blended = (1.0 - alpha) * inc_valid_rank + alpha * ranked
        primary = score_primary(valid.user_id, valid_y, blended)
        best_family_blend = max(best_family_blend, primary)

        if primary > best_primary:
            best_primary = primary
            best_family_name = name
            best_alpha = alpha
            best_valid = blended.copy()
            best_raw_valid = ranked.copy()

    candidate_scores[name + "_best_incumbent_blend"] = best_family_blend

print(
    "CANDIDATES " + json.dumps(
        {k: round(float(v), 8) for k, v in candidate_scores.items()},
        sort_keys=True,
    ),
    flush=True,
)
print(
    "FINDINGS selected_family={} own_weight={:.2f}".format(
        best_family_name, best_alpha
    ),
    flush=True,
)

metrics = evaluate(valid.user_id, valid_y, best_valid)

# Build test features only after all model/blend selection has been completed
# on validation. No test outcomes are accessed.
test = load("test")
x_test, test_schema = build_matrix(test, "test", builder)
if test_schema != train_schema:
    raise RuntimeError("Historical feature schemas do not match on test")

if best_family_name == "linear_glm":
    raw_test = predict_torch(linear_model, x_test, mean, std)
elif best_family_name == "fourier_kernel":
    raw_test = predict_torch(kernel_model, x_test, mean, std)
elif best_family_name == "boosted_gam":
    raw_test = gam_model.predict(x_test).astype(np.float64)
else:
    raise RuntimeError("Unknown selected family")

raw_test_rank = within_user_rank(test.user_id, raw_test)
inc_test_rank = within_user_rank(test.user_id, inc_test)
best_test = (
    (1.0 - best_alpha) * inc_test_rank
    + best_alpha * raw_test_rank
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )

del x_train, x_valid, x_test
gc.collect()

elapsed = time.time() - START
print(
    "METRICS " + json.dumps({
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    }),
    flush=True,
)