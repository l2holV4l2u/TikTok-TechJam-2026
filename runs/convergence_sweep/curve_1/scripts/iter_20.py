import os
import time
import json
import random

import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 8675309
THREADS = min(8, os.cpu_count() or 8)

random.seed(SEED)
np.random.seed(SEED)


def recency_weights(dates, half_life=3.0):
    dates = np.asarray(dates, dtype=np.int32)
    unique = np.unique(dates)
    positions = np.searchsorted(unique, dates)
    ages = len(unique) - 1 - positions
    weights = np.exp2(-ages.astype(np.float32) / np.float32(half_life))
    weights /= np.mean(weights)
    return weights.astype(np.float32)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores.copy()

    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    result_sorted = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    result_sorted[multi] = (
        positions[multi] /
        (repeated_lengths[multi].astype(np.float64) - 1.0)
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = result_sorted
    return result


def ordered_context(split):
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    dates = np.asarray(split.date, dtype=np.int32)
    n = len(users)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, times, users))
    su = users[order]
    st = times[order]
    sd = dates[order]

    new_user = np.empty(n, dtype=bool)
    new_user[0] = True
    new_user[1:] = su[1:] != su[:-1]

    new_day = new_user.copy()
    new_day[1:] |= sd[1:] != sd[:-1]

    new_session = new_user.copy()
    if n > 1:
        gap = st[1:] - st[:-1]
        new_session[1:] |= gap > 30 * 60 * 1000
        new_session[1:] |= gap < 0

    new_batch = new_user.copy()
    if n > 1:
        new_batch[1:] |= st[1:] != st[:-1]

    def group_geometry(starts_mask):
        starts = np.flatnonzero(starts_mask)
        ends = np.r_[starts[1:], n]
        lengths = ends - starts
        repeated_starts = np.repeat(starts, lengths)
        repeated_lengths = np.repeat(lengths, lengths)
        pos = np.arange(n, dtype=np.int64) - repeated_starts
        reverse = repeated_lengths - 1 - pos
        fraction = (pos.astype(np.float32) + 0.5) / repeated_lengths
        return pos, reverse, repeated_lengths, fraction

    day_pos, day_reverse, day_size, day_fraction = group_geometry(new_day)
    ses_pos, ses_reverse, ses_size, ses_fraction = group_geometry(new_session)
    batch_pos, batch_reverse, batch_size, batch_fraction = group_geometry(new_batch)

    gap_prev = np.zeros(n, dtype=np.float32)
    if n > 1:
        good = ~new_session[1:]
        gap_values = np.maximum(st[1:] - st[:-1], 0).astype(np.float64)
        gap_prev[1:][good] = np.log1p(gap_values[good] / 1000.0).astype(
            np.float32
        )

    gap_next = np.zeros(n, dtype=np.float32)
    if n > 1:
        same_next = ~new_session[1:]
        gap_values = np.maximum(st[1:] - st[:-1], 0).astype(np.float64)
        gap_next[:-1][same_next] = np.log1p(
            gap_values[same_next] / 1000.0
        ).astype(np.float32)

    previous = np.full(n, -1, dtype=np.int64)
    following = np.full(n, -1, dtype=np.int64)
    if n > 1:
        same = ~new_session[1:]
        left_rows = order[:-1][same]
        right_rows = order[1:][same]
        previous[right_rows] = left_rows
        following[left_rows] = right_rows

    features_sorted = np.column_stack([
        np.minimum(day_pos, 63),
        np.minimum(day_reverse, 63),
        np.log1p(day_size),
        day_fraction,
        np.minimum(ses_pos, 63),
        np.minimum(ses_reverse, 63),
        np.log1p(ses_size),
        ses_fraction,
        np.minimum(batch_pos, 31),
        np.minimum(batch_reverse, 31),
        np.log1p(batch_size),
        batch_fraction,
        gap_prev,
        gap_next,
    ]).astype(np.float32)

    features = np.empty_like(features_sorted)
    features[order] = features_sorted

    metadata = {
        "session_pos": np.empty(n, dtype=np.int32),
        "session_fraction": np.empty(n, dtype=np.float32),
        "batch_pos": np.empty(n, dtype=np.int32),
        "day_fraction": np.empty(n, dtype=np.float32),
        "previous": previous,
        "following": following,
    }
    metadata["session_pos"][order] = ses_pos.astype(np.int32)
    metadata["session_fraction"][order] = ses_fraction.astype(np.float32)
    metadata["batch_pos"][order] = batch_pos.astype(np.int32)
    metadata["day_fraction"][order] = day_fraction.astype(np.float32)

    return features, metadata


def safe_numeric(values):
    values = np.asarray(values, dtype=np.float32)
    transformed = np.sign(values) * np.log1p(np.abs(values))
    transformed[~np.isfinite(transformed)] = 0.0
    return transformed.astype(np.float32)


def build_matrix(split_name, split, temporal_features):
    pieces = []
    categorical_indices = []

    field_names = sorted(split.X.keys())
    for name in field_names:
        categorical_indices.append(len(pieces))
        pieces.append(np.asarray(split.X[name], dtype=np.float32))

    for name in sorted(split.num.keys()):
        raw = np.asarray(split.num[name], dtype=np.float32)
        pieces.append(safe_numeric(raw))
        pieces.append(np.isfinite(raw).astype(np.float32))

    for key in ("video_id", "author_id"):
        histories = historical_features(split_name, key=key)
        for name in sorted(histories.keys()):
            value = np.asarray(histories[name])
            if value.ndim == 1 and len(value) == len(split.user_id):
                value = value.astype(np.float32, copy=False)
                value = np.nan_to_num(
                    value, nan=0.0, posinf=0.0, neginf=0.0
                )
                pieces.append(value)

    for column in range(temporal_features.shape[1]):
        pieces.append(temporal_features[:, column])

    matrix = np.column_stack(pieces).astype(np.float32, copy=False)
    return matrix, categorical_indices


def fit_hazard(metadata, labels, weights):
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    definitions = [
        np.minimum(metadata["session_pos"], 31),
        np.minimum(
            (metadata["session_fraction"] * 10.0).astype(np.int32), 9
        ),
        np.minimum(metadata["batch_pos"], 15),
        np.minimum(
            (metadata["day_fraction"] * 10.0).astype(np.int32), 9
        ),
    ]
    sizes = [32, 10, 16, 10]

    global_rate = np.sum(weights * labels) / np.sum(weights)
    prior_strength = 300.0
    tables = []

    for bins, size in zip(definitions, sizes):
        total = np.bincount(bins, weights=weights, minlength=size)
        positive = np.bincount(
            bins, weights=weights * labels, minlength=size
        )
        rate = (
            positive + prior_strength * global_rate
        ) / (total + prior_strength)
        rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
        tables.append(np.log(rate / (1.0 - rate)))

    return tables, float(np.log(global_rate / (1.0 - global_rate)))


def hazard_predict(metadata, tables, global_logit):
    definitions = [
        np.minimum(metadata["session_pos"], 31),
        np.minimum(
            (metadata["session_fraction"] * 10.0).astype(np.int32), 9
        ),
        np.minimum(metadata["batch_pos"], 15),
        np.minimum(
            (metadata["day_fraction"] * 10.0).astype(np.int32), 9
        ),
    ]

    score = np.zeros(len(metadata["session_pos"]), dtype=np.float64)
    for bins, table in zip(definitions, tables):
        score += table[bins] - global_logit
    return score


def neighbor_smooth(user_ids, scores, metadata, beta):
    ranks = within_user_rank(user_ids, scores)
    probabilities = np.clip(0.02 + 0.96 * ranks, 1e-5, 1.0 - 1e-5)
    logits = np.log(probabilities / (1.0 - probabilities))

    previous = metadata["previous"]
    following = metadata["following"]
    neighbor_sum = np.zeros(len(logits), dtype=np.float64)
    neighbor_count = np.zeros(len(logits), dtype=np.float64)

    valid = previous >= 0
    neighbor_sum[valid] += logits[previous[valid]]
    neighbor_count[valid] += 1.0

    valid = following >= 0
    neighbor_sum[valid] += logits[following[valid]]
    neighbor_count[valid] += 1.0

    neighbor_mean = np.zeros(len(logits), dtype=np.float64)
    available = neighbor_count > 0
    neighbor_mean[available] = (
        neighbor_sum[available] / neighbor_count[available]
    )
    return logits + beta * neighbor_mean


def rank_blend(user_ids, incumbent, candidate, alpha):
    incumbent_rank = within_user_rank(user_ids, incumbent)
    candidate_rank = within_user_rank(user_ids, candidate)
    return (1.0 - alpha) * incumbent_rank + alpha * candidate_rank


train = load("train")
valid = load("valid")

train_temporal, train_meta = ordered_context(train)
valid_temporal, valid_meta = ordered_context(valid)

X_train, categorical_indices = build_matrix(
    "train", train, train_temporal
)
X_valid, _ = build_matrix("valid", valid, valid_temporal)

train_labels = np.asarray(train.y, dtype=np.float32)
train_weights = recency_weights(train.date, half_life=3.0)

dataset = lgb.Dataset(
    X_train,
    label=train_labels,
    weight=train_weights,
    categorical_feature=categorical_indices,
    free_raw_data=True,
)

params = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 250,
    "max_bin": 127,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.80,
    "bagging_freq": 1,
    "lambda_l1": 0.4,
    "lambda_l2": 2.0,
    "min_gain_to_split": 0.001,
    "max_cat_threshold": 32,
    "cat_smooth": 30.0,
    "cat_l2": 15.0,
    "num_threads": THREADS,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "data_random_seed": SEED + 3,
    "verbose": -1,
}

model = lgb.train(
    params,
    dataset,
    num_boost_round=230,
)

gbdt_valid = model.predict(X_valid).astype(np.float64)

hazard_tables, hazard_global = fit_hazard(
    train_meta, train_labels, train_weights
)
hazard_valid = hazard_predict(
    valid_meta, hazard_tables, hazard_global
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
incumbent_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
incumbent_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)
incumbent_valid = np.load(incumbent_valid_path).astype(np.float64)

valid_users = np.asarray(valid.user_id, dtype=np.int64)
valid_labels = np.asarray(valid.y, dtype=np.int8)

candidate_scores = {}
candidate_metrics = {}
candidate_specs = {}

def register(name, scores, spec):
    metrics = evaluate(valid_users, valid_labels, scores)
    candidate_scores[name] = np.asarray(scores, dtype=np.float64)
    candidate_metrics[name] = metrics
    candidate_specs[name] = spec


register("incumbent", incumbent_valid, ("incumbent",))

register("temporal_gbdt", gbdt_valid, ("gbdt",))
register("temporal_hazard", hazard_valid, ("hazard",))

for alpha in (0.10, 0.20, 0.35, 0.50, 0.70):
    register(
        "gbdt_blend_%.2f" % alpha,
        rank_blend(valid_users, incumbent_valid, gbdt_valid, alpha),
        ("gbdt_blend", alpha),
    )
    register(
        "hazard_blend_%.2f" % alpha,
        rank_blend(valid_users, incumbent_valid, hazard_valid, alpha),
        ("hazard_blend", alpha),
    )

for beta in (-0.35, -0.15, 0.15, 0.35, 0.60):
    smoothed_incumbent = neighbor_smooth(
        valid_users, incumbent_valid, valid_meta, beta
    )
    register(
        "neighbor_incumbent_%+.2f" % beta,
        smoothed_incumbent,
        ("neighbor_incumbent", beta),
    )

    smoothed_gbdt = neighbor_smooth(
        valid_users, gbdt_valid, valid_meta, beta
    )
    for alpha in (0.15, 0.30, 0.50):
        register(
            "neighbor_gbdt_%+.2f_blend_%.2f" % (beta, alpha),
            rank_blend(
                valid_users, incumbent_valid, smoothed_gbdt, alpha
            ),
            ("neighbor_gbdt_blend", beta, alpha),
        )

winner = max(
    candidate_metrics,
    key=lambda name: candidate_metrics[name]["primary"]
)
valid_scores = candidate_scores[winner]
metrics = candidate_metrics[winner]
winner_spec = candidate_specs[winner]

compact_candidates = {
    name: round(float(value["primary"]), 6)
    for name, value in candidate_metrics.items()
}
print("CANDIDATES " + json.dumps(compact_candidates, sort_keys=True))
print(
    "FINDINGS winner=%s spec=%s gbdt=%.6f hazard=%.6f incumbent=%.6f"
    % (
        winner,
        repr(winner_spec),
        candidate_metrics["temporal_gbdt"]["primary"],
        candidate_metrics["temporal_hazard"]["primary"],
        candidate_metrics["incumbent"]["primary"],
    )
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(gbdt_valid, dtype=np.float64),
    )

# Test features and predictions are constructed only after all validation
# choices have been fixed. No test labels are accessed.
test = load("test")
test_temporal, test_meta = ordered_context(test)
X_test, _ = build_matrix("test", test, test_temporal)
gbdt_test = model.predict(X_test).astype(np.float64)
hazard_test = hazard_predict(test_meta, hazard_tables, hazard_global)
incumbent_test = np.load(incumbent_test_path).astype(np.float64)
test_users = np.asarray(test.user_id, dtype=np.int64)

kind = winner_spec[0]
if kind == "incumbent":
    test_scores = incumbent_test
elif kind == "gbdt":
    test_scores = gbdt_test
elif kind == "hazard":
    test_scores = hazard_test
elif kind == "gbdt_blend":
    alpha = winner_spec[1]
    test_scores = rank_blend(
        test_users, incumbent_test, gbdt_test, alpha
    )
elif kind == "hazard_blend":
    alpha = winner_spec[1]
    test_scores = rank_blend(
        test_users, incumbent_test, hazard_test, alpha
    )
elif kind == "neighbor_incumbent":
    beta = winner_spec[1]
    test_scores = neighbor_smooth(
        test_users, incumbent_test, test_meta, beta
    )
elif kind == "neighbor_gbdt_blend":
    beta, alpha = winner_spec[1], winner_spec[2]
    smoothed = neighbor_smooth(
        test_users, gbdt_test, test_meta, beta
    )
    test_scores = rank_blend(
        test_users, incumbent_test, smoothed, alpha
    )
else:
    raise RuntimeError("Unknown winner specification: %r" % (winner_spec,))

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)