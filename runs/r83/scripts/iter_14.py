import os
import time
import json
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

from pipeline.data import load
from pipeline.evaluate import evaluate

START = time.time()
SEED = 41873
HALF_LIFE = 7.0
BATCH_SIZE = 16384
PRED_BATCH = 65536

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))


def recency_weights(dates):
    dates = np.asarray(dates)
    unique_dates = np.unique(dates)
    day_index = np.searchsorted(unique_dates, dates).astype(np.float32)
    age = float(day_index.max()) - day_index
    weights = np.exp2(-age / HALF_LIFE).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def group_positions(sorted_new_group):
    n = len(sorted_new_group)
    starts = np.flatnonzero(sorted_new_group)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts
    position = np.arange(n, dtype=np.int64) - np.repeat(starts, lengths)
    size = np.repeat(lengths, lengths)
    return position, size


def weekday_values(dates):
    dates = np.asarray(dates, dtype=np.int64)
    unique_dates = np.unique(dates)
    values = np.empty(len(unique_dates), dtype=np.int64)
    for i, value in enumerate(unique_dates):
        value = int(value)
        year = value // 10000
        month = (value // 100) % 100
        day = value % 100
        values[i] = datetime.date(year, month, day).weekday()
    return values[np.searchsorted(unique_dates, dates)]


def temporal_features(split):
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    dates = np.asarray(split.date, dtype=np.int64)
    hours = np.asarray(split.X["hour"], dtype=np.int64)
    n = len(users)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, times, users))
    sorted_users = users[order]
    sorted_times = times[order]
    sorted_dates = dates[order]

    user_change = np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    time_gap = np.r_[0, sorted_times[1:] - sorted_times[:-1]]
    date_change = np.r_[True, sorted_dates[1:] != sorted_dates[:-1]]

    session_new = user_change | date_change | (time_gap > 30 * 60 * 1000)
    session_pos_sorted, session_size_sorted = group_positions(session_new)

    day_new = user_change | date_change
    day_pos_sorted, day_size_sorted = group_positions(day_new)

    batch_new = (
        user_change
        | np.r_[True, sorted_times[1:] != sorted_times[:-1]]
    )
    batch_pos_sorted, batch_size_sorted = group_positions(batch_new)

    prev_gap_sorted = np.zeros(n, dtype=np.float64)
    valid_prev = ~session_new
    prev_gap_sorted[valid_prev] = (
        time_gap[valid_prev].astype(np.float64) / 1000.0
    )

    next_gap_sorted = np.zeros(n, dtype=np.float64)
    same_next = np.r_[
        (~session_new[1:]),
        False,
    ]
    next_gap_sorted[same_next] = (
        (sorted_times[1:] - sorted_times[:-1])[same_next[:-1]]
        .astype(np.float64) / 1000.0
    )

    session_starts = np.flatnonzero(session_new)
    session_lengths = np.diff(np.r_[session_starts, n])
    session_start_time = np.repeat(
        sorted_times[session_starts], session_lengths
    )
    elapsed_sorted = (
        sorted_times.astype(np.float64)
        - session_start_time.astype(np.float64)
    ) / 1000.0

    def unsort(values):
        result = np.empty_like(values)
        result[order] = values
        return result

    session_pos = unsort(session_pos_sorted)
    session_size = unsort(session_size_sorted)
    day_pos = unsort(day_pos_sorted)
    day_size = unsort(day_size_sorted)
    batch_pos = unsort(batch_pos_sorted)
    batch_size = unsort(batch_size_sorted)
    prev_gap = unsort(prev_gap_sorted)
    next_gap = unsort(next_gap_sorted)
    elapsed = unsort(elapsed_sorted)

    session_reverse = session_size - session_pos - 1
    day_reverse = day_size - day_pos - 1

    session_fraction = session_pos / np.maximum(session_size - 1, 1)
    day_fraction = day_pos / np.maximum(day_size - 1, 1)
    batch_fraction = batch_pos / np.maximum(batch_size - 1, 1)

    weekdays = weekday_values(dates)
    hour_angle = 2.0 * np.pi * (hours.astype(np.float64) % 24) / 24.0
    weekday_angle = 2.0 * np.pi * weekdays.astype(np.float64) / 7.0

    X = np.column_stack([
        np.log1p(session_pos),
        np.log1p(session_reverse),
        np.log1p(session_size),
        session_fraction,
        np.log1p(day_pos),
        np.log1p(day_reverse),
        np.log1p(day_size),
        day_fraction,
        np.log1p(batch_pos),
        np.log1p(batch_size),
        batch_fraction,
        np.log1p(np.minimum(prev_gap, 3600.0)),
        np.log1p(np.minimum(next_gap, 3600.0)),
        np.log1p(np.minimum(elapsed, 6.0 * 3600.0)),
        np.sin(hour_angle),
        np.cos(hour_angle),
        np.sin(weekday_angle),
        np.cos(weekday_angle),
        (session_pos == 0).astype(np.float64),
        (session_reverse == 0).astype(np.float64),
    ]).astype(np.float32)

    gap_bin = np.minimum(
        np.floor(np.log2(1.0 + prev_gap / 5.0)).astype(np.int64), 15
    )
    elapsed_bin = np.minimum(
        np.floor(np.log2(1.0 + elapsed / 10.0)).astype(np.int64), 15
    )

    bins = np.column_stack([
        np.minimum(session_pos, 20),
        np.minimum(session_reverse, 20),
        np.minimum(session_size, 24),
        np.minimum(day_pos, 30),
        np.minimum(day_reverse, 30),
        np.minimum(day_size, 35),
        np.minimum(batch_pos, 10),
        np.minimum(batch_size, 12),
        hours % 24,
        weekdays,
        gap_bin,
        elapsed_bin,
    ]).astype(np.int64)

    return np.ascontiguousarray(X), np.ascontiguousarray(bins)


def within_user_rank(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts
    positions = np.arange(n) - np.repeat(starts, lengths)
    denominators = np.repeat(np.maximum(lengths - 1, 1), lengths)
    ranked = positions.astype(np.float64) / denominators
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std())
    if std < 1e-12:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def combine_scores(raw, incumbent, users, mode, alpha):
    raw = np.asarray(raw, dtype=np.float64)
    incumbent = np.asarray(incumbent, dtype=np.float64)
    if mode == "raw":
        return raw
    if mode == "zblend":
        return alpha * zscore(incumbent) + (1.0 - alpha) * zscore(raw)
    if mode == "rankblend":
        return (
            alpha * within_user_rank(users, incumbent)
            + (1.0 - alpha) * within_user_rank(users, raw)
        )
    raise ValueError(mode)


class EmpiricalBayesPosition:
    def __init__(self, smoothing=60.0):
        self.smoothing = float(smoothing)
        self.base_logit = 0.0
        self.effects = []

    def fit(self, bins, y, weights):
        y = np.asarray(y, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        base = float(np.sum(weights * y) / np.sum(weights))
        base = np.clip(base, 1e-5, 1.0 - 1e-5)
        self.base_logit = np.log(base / (1.0 - base))
        self.effects = []

        for j in range(bins.shape[1]):
            values = bins[:, j]
            cardinality = int(values.max()) + 1
            total = np.bincount(
                values, weights=weights, minlength=cardinality
            )
            positive = np.bincount(
                values, weights=weights * y, minlength=cardinality
            )
            rate = (
                positive + self.smoothing * base
            ) / (total + self.smoothing)
            rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
            effect = np.log(rate / (1.0 - rate)) - self.base_logit
            self.effects.append(effect.astype(np.float64))
        return self

    def predict(self, bins):
        score = np.full(len(bins), self.base_logit, dtype=np.float64)
        scale = 1.0 / np.sqrt(max(len(self.effects), 1))
        for j, effect in enumerate(self.effects):
            index = np.minimum(bins[:, j], len(effect) - 1)
            score += scale * effect[index]
        return score


class TemporalMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.SiLU(),
            nn.Linear(48, 24),
            nn.SiLU(),
            nn.Linear(24, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def fit_mlp(X, y, weights, epochs=3):
    torch.manual_seed(SEED + 71)
    rng = np.random.default_rng(SEED + 71)

    mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    Xn = np.ascontiguousarray((X - mean) / std, dtype=np.float32)

    xt = torch.from_numpy(Xn)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    wt = torch.from_numpy(np.asarray(weights, dtype=np.float32))

    model = TemporalMLP(X.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0025, weight_decay=2e-4
    )

    for _ in range(epochs):
        order = rng.permutation(len(X))
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            idx = torch.from_numpy(order[start:start + BATCH_SIZE])
            logits = model(xt[idx])
            losses = F.binary_cross_entropy_with_logits(
                logits, yt[idx], reduction="none"
            )
            batch_weights = wt[idx]
            loss = (
                losses * batch_weights
            ).sum() / batch_weights.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model, mean, std


@torch.inference_mode()
def predict_mlp(model, mean, std, X):
    Xn = np.ascontiguousarray((X - mean) / std, dtype=np.float32)
    xt = torch.from_numpy(Xn)
    result = np.empty(len(X), dtype=np.float32)
    model.eval()
    for start in range(0, len(X), PRED_BATCH):
        end = min(start + PRED_BATCH, len(X))
        result[start:end] = model(xt[start:end]).cpu().numpy()
    return result.astype(np.float64)


def fit_lgbm(X, y, weights, rounds):
    dataset = lgb.Dataset(
        X,
        label=np.asarray(y, dtype=np.float32),
        weight=np.asarray(weights, dtype=np.float32),
        free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.045,
        "num_leaves": 31,
        "max_depth": 7,
        "min_data_in_leaf": 1500,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "num_threads": min(8, os.cpu_count() or 1),
        "seed": SEED + 113,
        "feature_fraction_seed": SEED + 127,
        "bagging_seed": SEED + 131,
        "verbose": -1,
    }
    return lgb.train(params, dataset, num_boost_round=rounds)


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)
train_weights = recency_weights(train.date)

X_train, bins_train = temporal_features(train)
X_valid, bins_valid = temporal_features(valid)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

raw_predictions = {}

eb_model = EmpiricalBayesPosition(smoothing=60.0).fit(
    bins_train, y_train, train_weights
)
raw_predictions["empirical_bayes"] = eb_model.predict(bins_valid)

lgb_rounds = 180
lgb_model = fit_lgbm(X_train, y_train, train_weights, lgb_rounds)
raw_predictions["lightgbm_temporal"] = lgb_model.predict(
    X_valid, num_iteration=lgb_rounds
).astype(np.float64)

mlp_model, mlp_mean, mlp_std = fit_mlp(
    X_train, y_train, train_weights, epochs=3
)
raw_predictions["mlp_temporal"] = predict_mlp(
    mlp_model, mlp_mean, mlp_std, X_valid
)

candidate_predictions = {
    "trusted_incumbent": inc_valid,
}
candidate_specs = {
    "trusted_incumbent": ("incumbent", "raw", 1.0),
}

for family, raw in raw_predictions.items():
    raw_name = family + "_raw"
    candidate_predictions[raw_name] = raw
    candidate_specs[raw_name] = (family, "raw", 0.0)

    for alpha in (0.70, 0.80, 0.90, 0.95):
        name = family + "_zblend_inc%.2f" % alpha
        candidate_predictions[name] = combine_scores(
            raw, inc_valid, valid_users, "zblend", alpha
        )
        candidate_specs[name] = (family, "zblend", alpha)

        name = family + "_rankblend_inc%.2f" % alpha
        candidate_predictions[name] = combine_scores(
            raw, inc_valid, valid_users, "rankblend", alpha
        )
        candidate_specs[name] = (family, "rankblend", alpha)

candidate_metrics = {}
best_name = None
best_result = None

for name, scores in candidate_predictions.items():
    result = evaluate(valid_users, y_valid, scores)
    candidate_metrics[name] = float(result["primary"])
    if best_result is None or result["primary"] > best_result["primary"]:
        best_name = name
        best_result = result

valid_scores = np.asarray(
    candidate_predictions[best_name], dtype=np.float64
)
winning_family, winning_mode, winning_alpha = candidate_specs[best_name]

print("CANDIDATES " + json.dumps(candidate_metrics, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps({
        "winner": best_name,
        "raw_primary": {
            name: candidate_metrics[name + "_raw"]
            for name in raw_predictions
        },
        "incumbent_primary": candidate_metrics["trusted_incumbent"],
        "temporal_features": int(X_train.shape[1]),
        "validation_aux_read": False,
        "test_aux_read": False,
    }, sort_keys=True)
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        valid_scores.astype(np.float64),
    )

test = load("test")
test_users = np.asarray(test.user_id)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if winning_family == "incumbent":
    test_scores = inc_test
else:
    X_test, bins_test = temporal_features(test)

    y_combined = np.concatenate([
        y_train,
        np.asarray(valid.y, dtype=np.float32),
    ])
    combined_dates = np.concatenate([
        np.asarray(train.date),
        np.asarray(valid.date),
    ])
    combined_weights = recency_weights(combined_dates)

    if winning_family == "empirical_bayes":
        bins_combined = np.concatenate(
            [bins_train, bins_valid], axis=0
        )
        final_model = EmpiricalBayesPosition(smoothing=60.0).fit(
            bins_combined, y_combined, combined_weights
        )
        raw_test = final_model.predict(bins_test)

    elif winning_family == "lightgbm_temporal":
        X_combined = np.concatenate([X_train, X_valid], axis=0)
        final_model = fit_lgbm(
            X_combined, y_combined, combined_weights, lgb_rounds
        )
        raw_test = final_model.predict(
            X_test, num_iteration=lgb_rounds
        ).astype(np.float64)

    elif winning_family == "mlp_temporal":
        X_combined = np.concatenate([X_train, X_valid], axis=0)
        final_model, final_mean, final_std = fit_mlp(
            X_combined,
            y_combined,
            combined_weights,
            epochs=3,
        )
        raw_test = predict_mlp(
            final_model, final_mean, final_std, X_test
        )

    else:
        raise ValueError(winning_family)

    test_scores = combine_scores(
        raw_test,
        inc_test,
        test_users,
        winning_mode,
        winning_alpha,
    )

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(best_result["primary"]),
        "gauc": float(best_result["gauc"]),
        "ndcg@5": float(best_result["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)