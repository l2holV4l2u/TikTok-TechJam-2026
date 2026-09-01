import json
import os

import lightgbm as lgb
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
NUM_BOOST_ROUND = 300
EARLY_STOPPING_ROUNDS = 30
THREADS = min(12, os.cpu_count() or 1)

RAW_FIELDS = list(FEATURE_CARDINALITIES.keys())
SINGLE_STAT_FIELDS = [
    ("video_id", 20.0),
    ("author_id", 30.0),
    ("tag", 80.0),
    ("onehot_feat3", 40.0),
    ("duration_bucket", 80.0),
]


def stable_group_order(user_ids):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[order]
    if sorted_users.size == 0:
        return order, np.empty(0, dtype=np.int32)
    boundaries = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1], True]
    )
    groups = np.diff(boundaries).astype(np.int32)
    return order, groups


def fit_single_stats(train, field):
    ids = np.asarray(train.X[field], dtype=np.int64)
    y = np.asarray(train.y, dtype=np.float64)
    cardinality = int(FEATURE_CARDINALITIES[field])
    counts = np.bincount(ids, minlength=cardinality).astype(np.float64)
    sums = np.bincount(ids, weights=y, minlength=cardinality).astype(np.float64)
    return counts, sums


def apply_single_stats(split, field, counts, sums, prior, alpha, is_train):
    ids = np.asarray(split.X[field], dtype=np.int64)
    if is_train:
        y = np.asarray(split.y, dtype=np.float64)
        effective_count = counts[ids] - 1.0
        effective_sum = sums[ids] - y
    else:
        effective_count = counts[ids]
        effective_sum = sums[ids]

    rate = (
        effective_sum + alpha * prior
    ) / (
        effective_count + alpha
    )
    log_count = np.log1p(np.maximum(effective_count, 0.0))
    return rate.astype(np.float32), log_count.astype(np.float32)


def make_pair_keys(split, right_field):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    right = np.asarray(split.X[right_field], dtype=np.int64)
    right_cardinality = int(FEATURE_CARDINALITIES[right_field])
    return users * np.int64(right_cardinality) + right


def fit_pair_stats(train, right_field):
    keys = make_pair_keys(train, right_field)
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    y = np.asarray(train.y, dtype=np.float64)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.bincount(inverse, weights=y).astype(np.float64)
    return unique_keys, inverse, counts, sums


def lookup_pair_stats(keys, unique_keys, counts, sums):
    positions = np.searchsorted(unique_keys, keys)
    safe_positions = np.minimum(positions, unique_keys.size - 1)
    found = (
        (positions < unique_keys.size)
        & (unique_keys[safe_positions] == keys)
    )

    out_counts = np.zeros(keys.shape[0], dtype=np.float64)
    out_sums = np.zeros(keys.shape[0], dtype=np.float64)
    out_counts[found] = counts[safe_positions[found]]
    out_sums[found] = sums[safe_positions[found]]
    return out_counts, out_sums


def apply_pair_stats(
    split,
    right_field,
    unique_keys,
    train_inverse,
    counts,
    sums,
    prior,
    alpha,
    is_train,
):
    if is_train:
        y = np.asarray(split.y, dtype=np.float64)
        effective_count = counts[train_inverse] - 1.0
        effective_sum = sums[train_inverse] - y
    else:
        keys = make_pair_keys(split, right_field)
        effective_count, effective_sum = lookup_pair_stats(
            keys, unique_keys, counts, sums
        )

    rate = (
        effective_sum + alpha * prior
    ) / (
        effective_count + alpha
    )
    log_count = np.log1p(np.maximum(effective_count, 0.0))
    return rate.astype(np.float32), log_count.astype(np.float32)


train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.int8)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
prior = float(train_y.mean())

single_stats = {}
for field, alpha in SINGLE_STAT_FIELDS:
    counts, sums = fit_single_stats(train, field)
    single_stats[field] = {
        "counts": counts,
        "sums": sums,
        "alpha": alpha,
    }

pair_stats = {}
for right_field, alpha in [("author_id", 8.0), ("tag", 20.0)]:
    unique_keys, inverse, counts, sums = fit_pair_stats(train, right_field)
    pair_stats[right_field] = {
        "unique_keys": unique_keys,
        "train_inverse": inverse,
        "counts": counts,
        "sums": sums,
        "alpha": alpha,
    }


feature_names = list(RAW_FIELDS)
categorical_indices = list(range(len(RAW_FIELDS)))

for field, _ in SINGLE_STAT_FIELDS:
    feature_names.extend([
        f"{field}_train_rate",
        f"{field}_train_log_count",
    ])

for right_field in pair_stats:
    feature_names.extend([
        f"user_{right_field}_train_rate",
        f"user_{right_field}_train_log_count",
    ])


def make_matrix(split, is_train=False):
    columns = [
        np.asarray(split.X[field], dtype=np.float32)
        for field in RAW_FIELDS
    ]

    for field, _ in SINGLE_STAT_FIELDS:
        stat = single_stats[field]
        rate, log_count = apply_single_stats(
            split=split,
            field=field,
            counts=stat["counts"],
            sums=stat["sums"],
            prior=prior,
            alpha=stat["alpha"],
            is_train=is_train,
        )
        columns.extend([rate, log_count])

    for right_field, stat in pair_stats.items():
        rate, log_count = apply_pair_stats(
            split=split,
            right_field=right_field,
            unique_keys=stat["unique_keys"],
            train_inverse=stat["train_inverse"],
            counts=stat["counts"],
            sums=stat["sums"],
            prior=prior,
            alpha=stat["alpha"],
            is_train=is_train,
        )
        columns.extend([rate, log_count])

    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


x_train = make_matrix(train, is_train=True)
x_valid = make_matrix(valid, is_train=False)

train_order, train_groups = stable_group_order(train.user_id)
valid_order, valid_groups = stable_group_order(valid.user_id)

x_train_grouped = np.ascontiguousarray(x_train[train_order])
y_train_grouped = np.ascontiguousarray(train_y[train_order])
x_valid_grouped = np.ascontiguousarray(x_valid[valid_order])
y_valid_grouped = np.ascontiguousarray(valid_y[valid_order])

del x_train
del x_valid

train_dataset = lgb.Dataset(
    x_train_grouped,
    label=y_train_grouped,
    group=train_groups,
    feature_name=feature_names,
    categorical_feature=categorical_indices,
    free_raw_data=True,
)

valid_dataset = lgb.Dataset(
    x_valid_grouped,
    label=y_valid_grouped,
    group=valid_groups,
    feature_name=feature_names,
    categorical_feature=categorical_indices,
    reference=train_dataset,
    free_raw_data=True,
)

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "lambdarank_truncation_level": 10,
    "label_gain": [0, 1],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 120,
    "min_sum_hessian_in_leaf": 1e-3,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 1.0,
    "max_cat_threshold": 64,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "max_bin": 127,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "data_random_seed": SEED,
    "num_threads": THREADS,
    "verbose": -1,
}

model = lgb.train(
    params,
    train_dataset,
    num_boost_round=NUM_BOOST_ROUND,
    valid_sets=[valid_dataset],
    valid_names=["valid"],
    callbacks=[
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
    ],
)

valid_scores_grouped = model.predict(
    x_valid_grouped,
    num_iteration=model.best_iteration,
)
valid_scores = np.empty(valid_scores_grouped.shape[0], dtype=np.float64)
valid_scores[valid_order] = valid_scores_grouped

metrics = evaluate(valid_users, valid_y, valid_scores)
metrics = {key: float(value) for key, value in metrics.items()}

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    test = load("test")
    x_test = make_matrix(test, is_train=False)
    test_scores = model.predict(
        x_test,
        num_iteration=model.best_iteration,
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

final_metrics = {
    "primary": metrics["primary"],
    "gauc": metrics["gauc"],
    "ndcg@5": metrics["ndcg@5"],
    "gpu_seconds": 0.0,
}
print("METRICS " + json.dumps(final_metrics))