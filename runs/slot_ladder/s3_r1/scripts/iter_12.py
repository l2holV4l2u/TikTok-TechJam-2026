import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 271828
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

CAT_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "video_type",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "user_active_degree",
    "is_video_author",
]
HALF_LIFE_DAYS = 4.0


def assign_group_features(order, boundary):
    n = order.size
    sorted_boundary = boundary[order]
    start_mask = np.empty(n, dtype=bool)
    start_mask[0] = True
    start_mask[1:] = sorted_boundary[1:] != sorted_boundary[:-1]
    starts = np.flatnonzero(start_mask)
    ends = np.r_[starts[1:], n]
    counts = ends - starts

    rank_sorted = np.arange(n, dtype=np.int64) - np.repeat(starts, counts)
    count_sorted = np.repeat(counts, counts)

    rank = np.empty(n, dtype=np.float32)
    count = np.empty(n, dtype=np.float32)
    rank[order] = rank_sorted.astype(np.float32)
    count[order] = count_sorted.astype(np.float32)
    return rank, count


def temporal_features(split):
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    dates = np.asarray(split.date, dtype=np.int64)
    rows = np.arange(users.size, dtype=np.int64)

    chronological = np.lexsort((rows, times, users))
    user_rank, user_count = assign_group_features(chronological, users)

    user_day_key = users * np.int64(100000000) + dates
    day_order = np.lexsort((rows, times, user_day_key))
    day_rank, day_count = assign_group_features(day_order, user_day_key)

    time_min = times.min()
    batch_key = (
        users.astype(np.int64) * np.int64(10000000000000)
        + (times - time_min)
    )
    batch_order = np.lexsort((rows, batch_key))
    batch_rank, batch_count = assign_group_features(batch_order, batch_key)

    sorted_users = users[chronological]
    sorted_times = times[chronological]
    same_prev = np.r_[False, sorted_users[1:] == sorted_users[:-1]]
    same_next = np.r_[sorted_users[:-1] == sorted_users[1:], False]

    prev_gap_sorted = np.zeros(users.size, dtype=np.float64)
    next_gap_sorted = np.zeros(users.size, dtype=np.float64)
    prev_gap_sorted[same_prev] = (
        sorted_times[same_prev] - sorted_times[np.flatnonzero(same_prev) - 1]
    ) / 60000.0
    next_indices = np.flatnonzero(same_next)
    next_gap_sorted[same_next] = (
        sorted_times[next_indices + 1] - sorted_times[same_next]
    ) / 60000.0

    prev_gap = np.empty(users.size, dtype=np.float32)
    next_gap = np.empty(users.size, dtype=np.float32)
    prev_gap[chronological] = np.log1p(
        np.clip(prev_gap_sorted, 0.0, 24.0 * 60.0)
    ).astype(np.float32)
    next_gap[chronological] = np.log1p(
        np.clip(next_gap_sorted, 0.0, 24.0 * 60.0)
    ).astype(np.float32)

    first_mask = np.empty(users.size, dtype=bool)
    first_mask[0] = True
    first_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(first_mask)
    ends = np.r_[starts[1:], users.size]
    counts = ends - starts

    first_time_sorted = np.repeat(sorted_times[starts], counts)
    last_time_sorted = np.repeat(sorted_times[ends - 1], counts)
    elapsed_sorted = (sorted_times - first_time_sorted) / 3600000.0
    remaining_sorted = (last_time_sorted - sorted_times) / 3600000.0

    elapsed = np.empty(users.size, dtype=np.float32)
    remaining = np.empty(users.size, dtype=np.float32)
    elapsed[chronological] = np.log1p(
        np.clip(elapsed_sorted, 0.0, 24.0 * 31.0)
    ).astype(np.float32)
    remaining[chronological] = np.log1p(
        np.clip(remaining_sorted, 0.0, 24.0 * 31.0)
    ).astype(np.float32)

    hour = np.asarray(split.X["hour"], dtype=np.float32)
    hour_angle = 2.0 * np.pi * hour / 24.0

    continuous = np.column_stack([
        np.log1p(user_rank),
        np.log1p(np.maximum(user_count - 1.0 - user_rank, 0.0)),
        user_rank / np.maximum(user_count - 1.0, 1.0),
        np.log1p(user_count),
        np.log1p(day_rank),
        np.log1p(np.maximum(day_count - 1.0 - day_rank, 0.0)),
        day_rank / np.maximum(day_count - 1.0, 1.0),
        np.log1p(day_count),
        np.log1p(batch_rank),
        batch_rank / np.maximum(batch_count - 1.0, 1.0),
        np.log1p(batch_count),
        prev_gap,
        next_gap,
        elapsed,
        remaining,
        np.sin(hour_angle),
        np.cos(hour_angle),
        (prev_gap == 0.0).astype(np.float32),
        (next_gap == 0.0).astype(np.float32),
    ]).astype(np.float32)

    categorical = np.column_stack([
        np.asarray(split.X[name], dtype=np.int32) for name in CAT_FIELDS
    ]).astype(np.float32)

    matrix = np.ascontiguousarray(
        np.column_stack([continuous, categorical]), dtype=np.float32
    )
    return matrix, continuous, categorical.astype(np.int64)


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int64)
    unique_dates = np.sort(np.unique(dates))
    age_lookup = {
        int(date): len(unique_dates) - 1 - index
        for index, date in enumerate(unique_dates)
    }
    ages = np.fromiter(
        (age_lookup[int(date)] for date in dates),
        dtype=np.float32,
        count=dates.size,
    )
    weights = np.exp2(-ages / HALF_LIFE_DAYS).astype(np.float32)
    weights /= np.mean(weights)
    return weights


def grouped_order(user_ids):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[order]
    boundaries = np.r_[
        0,
        np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1,
        sorted_users.size,
    ]
    groups = np.diff(boundaries).astype(np.int32)
    return order, groups


def standardized(x):
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.all():
        x = np.where(finite, x, 0.0)
    scale = float(np.std(x))
    if scale < 1e-12:
        scale = 1.0
    return (x - float(np.mean(x))) / scale


def blend_score(incumbent, candidate, alpha):
    return (1.0 - alpha) * standardized(incumbent) + alpha * standardized(candidate)


class AdditiveTemporalModel(nn.Module):
    def __init__(self, n_continuous):
        super().__init__()
        self.offsets = np.cumsum(
            [0] + [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS[:-1]],
            dtype=np.int64,
        )
        total = int(sum(int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS))
        self.cat_linear = nn.Embedding(total, 1)
        self.cont_linear = nn.Linear(n_continuous, 1)
        self.spline = nn.Sequential(
            nn.Linear(n_continuous, 48),
            nn.Tanh(),
            nn.Linear(48, 1),
        )
        nn.init.zeros_(self.cat_linear.weight)
        nn.init.zeros_(self.cont_linear.weight)
        nn.init.zeros_(self.cont_linear.bias)

    def forward(self, continuous, categorical):
        offsets = torch.as_tensor(
            self.offsets, dtype=torch.long, device=categorical.device
        )
        cat_score = self.cat_linear(categorical + offsets).sum(dim=1)
        return (
            cat_score
            + self.cont_linear(continuous)
            + 0.25 * self.spline(continuous)
        ).squeeze(1)


def fit_additive(cont, cat, labels, weights):
    centers = cont.mean(axis=0, dtype=np.float64).astype(np.float32)
    scales = cont.std(axis=0, dtype=np.float64).astype(np.float32)
    scales[scales < 1e-5] = 1.0
    cont_scaled = np.ascontiguousarray(
        (cont - centers) / scales, dtype=np.float32
    )

    model = AdditiveTemporalModel(cont.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-6)
    rng = np.random.default_rng(SEED + 44)
    batch_size = 32768

    model.train()
    for epoch in range(3):
        order = rng.permutation(labels.size)
        total_loss = 0.0
        steps = 0
        for start in range(0, labels.size, batch_size):
            idx = order[start:start + batch_size]
            cb = torch.from_numpy(cont_scaled[idx])
            xb = torch.from_numpy(cat[idx])
            yb = torch.from_numpy(labels[idx])
            wb = torch.from_numpy(weights[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(cb, xb)
            losses = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = torch.sum(losses * wb) / torch.sum(wb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            steps += 1

        print("FINDINGS " + json.dumps({
            "additive_epoch": epoch + 1,
            "loss": total_loss / max(steps, 1),
        }))

    return model, centers, scales


def predict_additive(model, centers, scales, cont, cat):
    cont_scaled = np.ascontiguousarray(
        (cont - centers) / scales, dtype=np.float32
    )
    output = np.empty(cont.shape[0], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, cont.shape[0], 65536):
            end = min(start + 65536, cont.shape[0])
            output[start:end] = model(
                torch.from_numpy(cont_scaled[start:end]),
                torch.from_numpy(cat[start:end]),
            ).cpu().numpy()
    return output


train = load("train")
valid = load("valid")

x_train, cont_train, cat_train = temporal_features(train)
x_valid, cont_valid, cat_valid = temporal_features(valid)
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
weights = recency_weights(train.date)

n_cont = cont_train.shape[1]
categorical_indices = list(range(n_cont, n_cont + len(CAT_FIELDS)))

print("FINDINGS " + json.dumps({
    "continuous_temporal_features": n_cont,
    "categorical_features": len(CAT_FIELDS),
    "recency_weight_min": float(weights.min()),
    "recency_weight_max": float(weights.max()),
    "train_rows": int(y_train.size),
}))

binary_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "min_data_in_leaf": 400,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 1.0,
    "max_bin": 127,
    "num_threads": min(8, max(1, os.cpu_count() or 1)),
    "seed": SEED,
    "verbose": -1,
}
binary_dataset = lgb.Dataset(
    x_train,
    label=y_train,
    weight=weights,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
binary_model = lgb.train(
    binary_params,
    binary_dataset,
    num_boost_round=210,
)
pred_binary = binary_model.predict(x_valid).astype(np.float64)

rank_order, rank_groups = grouped_order(train.user_id)
rank_params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "label_gain": [0, 1],
    "learning_rate": 0.045,
    "num_leaves": 63,
    "min_data_in_leaf": 350,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 1.5,
    "max_bin": 127,
    "lambdarank_truncation_level": 10,
    "num_threads": min(8, max(1, os.cpu_count() or 1)),
    "seed": SEED + 1,
    "verbose": -1,
}
rank_dataset = lgb.Dataset(
    x_train[rank_order],
    label=y_train[rank_order],
    weight=weights[rank_order],
    group=rank_groups,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
rank_model = lgb.train(
    rank_params,
    rank_dataset,
    num_boost_round=190,
)
pred_rank = rank_model.predict(x_valid).astype(np.float64)

additive_model, add_centers, add_scales = fit_additive(
    cont_train, cat_train, y_train, weights
)
pred_additive = predict_additive(
    additive_model, add_centers, add_scales, cont_valid, cat_valid
).astype(np.float64)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.load(inc_valid_path).astype(np.float64)

raw_predictions = {
    "temporal_binary": pred_binary,
    "temporal_lambdamart": pred_rank,
    "temporal_additive": pred_additive,
}
models = {
    "temporal_binary": binary_model,
    "temporal_lambdamart": rank_model,
    "temporal_additive": additive_model,
}

candidate_scores = {}
for name, prediction in raw_predictions.items():
    metric = evaluate(valid.user_id, y_valid, prediction)
    candidate_scores[name + "_raw"] = float(metric["primary"])

alphas = [0.10, 0.20, 0.35, 0.50, 0.70]
best_name = "incumbent"
best_alpha = 0.0
best_scores = inc_valid.copy()
best_metric = evaluate(valid.user_id, y_valid, best_scores)
candidate_scores["incumbent"] = float(best_metric["primary"])

for name, prediction in raw_predictions.items():
    for alpha in alphas:
        scores = blend_score(inc_valid, prediction, alpha)
        metric = evaluate(valid.user_id, y_valid, scores)
        key = name + "_blend_" + str(alpha)
        candidate_scores[key] = float(metric["primary"])
        if metric["primary"] > best_metric["primary"]:
            best_name = name
            best_alpha = alpha
            best_scores = scores
            best_metric = metric

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print("FINDINGS " + json.dumps({
    "selected_family": best_name,
    "selected_candidate_weight": best_alpha,
    "selected_primary": float(best_metric["primary"]),
}))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_name != "incumbent":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_predictions[best_name], dtype=np.float64),
        )

test = load("test")
x_test, cont_test, cat_test = temporal_features(test)
inc_test = np.load(inc_test_path).astype(np.float64)

if best_name == "temporal_binary":
    raw_test = binary_model.predict(x_test).astype(np.float64)
    test_scores = blend_score(inc_test, raw_test, best_alpha)
elif best_name == "temporal_lambdamart":
    raw_test = rank_model.predict(x_test).astype(np.float64)
    test_scores = blend_score(inc_test, raw_test, best_alpha)
elif best_name == "temporal_additive":
    raw_test = predict_additive(
        additive_model, add_centers, add_scales, cont_test, cat_test
    ).astype(np.float64)
    test_scores = blend_score(inc_test, raw_test, best_alpha)
else:
    test_scores = inc_test

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metric["primary"]),
    "gauc": float(best_metric["gauc"]),
    "ndcg@5": float(best_metric["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))