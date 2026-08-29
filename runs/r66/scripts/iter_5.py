import os
import gc
import json
import time
import random

import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 20260828

random.seed(SEED)
np.random.seed(SEED)

try:
    lgb.register_logger(None)
except Exception:
    pass


PROFILE_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "music_type",
    "video_type",
    "upload_type",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat2",
]


def safe_rate(sums, counts, prior, strength):
    return (
        np.asarray(sums, dtype=np.float64)
        + float(strength) * np.asarray(prior, dtype=np.float64)
    ) / (
        np.asarray(counts, dtype=np.float64) + float(strength)
    )


def lookup_sorted(keys, query):
    """Return indices and a mask for exact matches in a sorted key array."""
    query = np.asarray(query, dtype=np.int64)
    pos = np.searchsorted(keys, query)
    clipped = np.minimum(pos, max(len(keys) - 1, 0))
    if len(keys) == 0:
        return np.zeros(len(query), dtype=np.int64), np.zeros(len(query), dtype=bool)
    found = (pos < len(keys)) & (keys[clipped] == query)
    return clipped, found


def build_user_statistics(train_user_ids, labels, recent_mask, user_cardinality):
    labels_f = np.asarray(labels, dtype=np.float64)
    users = np.asarray(train_user_ids, dtype=np.int64)
    recent_f = np.asarray(recent_mask, dtype=np.float64)

    counts = np.bincount(users, minlength=user_cardinality).astype(np.float64)
    sums = np.bincount(
        users, weights=labels_f, minlength=user_cardinality
    ).astype(np.float64)

    recent_counts = np.bincount(
        users, weights=recent_f, minlength=user_cardinality
    ).astype(np.float64)
    recent_sums = np.bincount(
        users, weights=labels_f * recent_f, minlength=user_cardinality
    ).astype(np.float64)

    global_rate = float(labels_f.mean())
    if recent_f.sum() > 0:
        recent_global_rate = float(
            np.sum(labels_f * recent_f) / np.sum(recent_f)
        )
    else:
        recent_global_rate = global_rate

    return {
        "counts": counts,
        "sums": sums,
        "recent_counts": recent_counts,
        "recent_sums": recent_sums,
        "global_rate": global_rate,
        "recent_global_rate": recent_global_rate,
    }


def user_features_train(users, labels, recent_mask, stats):
    users = np.asarray(users, dtype=np.int64)
    y = np.asarray(labels, dtype=np.float64)
    recent = np.asarray(recent_mask, dtype=np.float64)

    count_loo = np.maximum(stats["counts"][users] - 1.0, 0.0)
    sum_loo = stats["sums"][users] - y
    rate_loo = safe_rate(
        sum_loo, count_loo, stats["global_rate"], strength=12.0
    )

    recent_count_loo = np.maximum(
        stats["recent_counts"][users] - recent, 0.0
    )
    recent_sum_loo = stats["recent_sums"][users] - y * recent
    recent_rate_loo = safe_rate(
        recent_sum_loo,
        recent_count_loo,
        stats["recent_global_rate"],
        strength=10.0,
    )

    return (
        rate_loo,
        recent_rate_loo,
        count_loo,
        recent_count_loo,
    )


def user_features_inference(users, stats):
    users = np.asarray(users, dtype=np.int64)
    valid_user = (users >= 0) & (users < len(stats["counts"]))
    clipped = np.clip(users, 0, len(stats["counts"]) - 1)

    count = np.where(valid_user, stats["counts"][clipped], 0.0)
    sums = np.where(valid_user, stats["sums"][clipped], 0.0)
    rate = safe_rate(sums, count, stats["global_rate"], strength=12.0)

    recent_count = np.where(
        valid_user, stats["recent_counts"][clipped], 0.0
    )
    recent_sums = np.where(
        valid_user, stats["recent_sums"][clipped], 0.0
    )
    recent_rate = safe_rate(
        recent_sums,
        recent_count,
        stats["recent_global_rate"],
        strength=10.0,
    )
    return rate, recent_rate, count, recent_count


def aggregate_field(
    users,
    values,
    labels,
    recent_mask,
    cardinality,
):
    users = np.asarray(users, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64)
    labels_f = np.asarray(labels, dtype=np.float64)
    recent_f = np.asarray(recent_mask, dtype=np.float64)

    pair_keys = users * np.int64(cardinality) + values
    unique_keys, inverse, counts = np.unique(
        pair_keys, return_inverse=True, return_counts=True
    )
    counts = counts.astype(np.float64)
    sums = np.bincount(
        inverse, weights=labels_f, minlength=len(unique_keys)
    ).astype(np.float64)
    recent_counts = np.bincount(
        inverse, weights=recent_f, minlength=len(unique_keys)
    ).astype(np.float64)
    recent_sums = np.bincount(
        inverse,
        weights=labels_f * recent_f,
        minlength=len(unique_keys),
    ).astype(np.float64)

    value_counts = np.bincount(
        values, minlength=cardinality
    ).astype(np.float64)
    value_sums = np.bincount(
        values, weights=labels_f, minlength=cardinality
    ).astype(np.float64)
    value_recent_counts = np.bincount(
        values, weights=recent_f, minlength=cardinality
    ).astype(np.float64)
    value_recent_sums = np.bincount(
        values,
        weights=labels_f * recent_f,
        minlength=cardinality,
    ).astype(np.float64)

    return {
        "keys": unique_keys,
        "inverse": inverse.astype(np.int32, copy=False),
        "counts": counts,
        "sums": sums,
        "recent_counts": recent_counts,
        "recent_sums": recent_sums,
        "value_counts": value_counts,
        "value_sums": value_sums,
        "value_recent_counts": value_recent_counts,
        "value_recent_sums": value_recent_sums,
        "cardinality": int(cardinality),
    }


def field_features_train(
    table,
    values,
    labels,
    recent_mask,
    user_rate,
    recent_user_rate,
    global_rate,
    recent_global_rate,
):
    inv = table["inverse"]
    values = np.asarray(values, dtype=np.int64)
    y = np.asarray(labels, dtype=np.float64)
    recent = np.asarray(recent_mask, dtype=np.float64)

    pair_count = np.maximum(table["counts"][inv] - 1.0, 0.0)
    pair_sum = table["sums"][inv] - y

    value_count = np.maximum(table["value_counts"][values] - 1.0, 0.0)
    value_sum = table["value_sums"][values] - y
    value_rate = safe_rate(
        value_sum, value_count, global_rate, strength=30.0
    )

    pair_prior = 0.55 * user_rate + 0.45 * value_rate
    pair_rate_fast = safe_rate(
        pair_sum, pair_count, pair_prior, strength=3.0
    )
    pair_rate_stable = safe_rate(
        pair_sum, pair_count, pair_prior, strength=12.0
    )

    recent_pair_count = np.maximum(
        table["recent_counts"][inv] - recent, 0.0
    )
    recent_pair_sum = table["recent_sums"][inv] - y * recent

    recent_value_count = np.maximum(
        table["value_recent_counts"][values] - recent, 0.0
    )
    recent_value_sum = (
        table["value_recent_sums"][values] - y * recent
    )
    recent_value_rate = safe_rate(
        recent_value_sum,
        recent_value_count,
        recent_global_rate,
        strength=24.0,
    )

    recent_prior = 0.55 * recent_user_rate + 0.45 * recent_value_rate
    recent_pair_rate = safe_rate(
        recent_pair_sum,
        recent_pair_count,
        recent_prior,
        strength=5.0,
    )

    return np.column_stack(
        [
            np.log1p(pair_count),
            pair_rate_fast,
            pair_rate_stable - user_rate,
            pair_rate_stable - value_rate,
            value_rate,
            np.log1p(recent_pair_count),
            recent_pair_rate - recent_user_rate,
            recent_pair_rate - recent_value_rate,
            recent_pair_rate - pair_rate_stable,
        ]
    ).astype(np.float32, copy=False)


def field_features_inference(
    table,
    users,
    values,
    user_rate,
    recent_user_rate,
    global_rate,
    recent_global_rate,
):
    users = np.asarray(users, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64)
    cardinality = table["cardinality"]

    pair_keys = users * np.int64(cardinality) + values
    positions, found = lookup_sorted(table["keys"], pair_keys)

    pair_count = np.zeros(len(users), dtype=np.float64)
    pair_sum = np.zeros(len(users), dtype=np.float64)
    recent_pair_count = np.zeros(len(users), dtype=np.float64)
    recent_pair_sum = np.zeros(len(users), dtype=np.float64)

    pair_count[found] = table["counts"][positions[found]]
    pair_sum[found] = table["sums"][positions[found]]
    recent_pair_count[found] = table["recent_counts"][positions[found]]
    recent_pair_sum[found] = table["recent_sums"][positions[found]]

    valid_value = (values >= 0) & (values < cardinality)
    clipped_values = np.clip(values, 0, cardinality - 1)

    value_count = np.where(
        valid_value, table["value_counts"][clipped_values], 0.0
    )
    value_sum = np.where(
        valid_value, table["value_sums"][clipped_values], 0.0
    )
    value_rate = safe_rate(
        value_sum, value_count, global_rate, strength=30.0
    )

    pair_prior = 0.55 * user_rate + 0.45 * value_rate
    pair_rate_fast = safe_rate(
        pair_sum, pair_count, pair_prior, strength=3.0
    )
    pair_rate_stable = safe_rate(
        pair_sum, pair_count, pair_prior, strength=12.0
    )

    recent_value_count = np.where(
        valid_value,
        table["value_recent_counts"][clipped_values],
        0.0,
    )
    recent_value_sum = np.where(
        valid_value,
        table["value_recent_sums"][clipped_values],
        0.0,
    )
    recent_value_rate = safe_rate(
        recent_value_sum,
        recent_value_count,
        recent_global_rate,
        strength=24.0,
    )

    recent_prior = 0.55 * recent_user_rate + 0.45 * recent_value_rate
    recent_pair_rate = safe_rate(
        recent_pair_sum,
        recent_pair_count,
        recent_prior,
        strength=5.0,
    )

    return np.column_stack(
        [
            np.log1p(pair_count),
            pair_rate_fast,
            pair_rate_stable - user_rate,
            pair_rate_stable - value_rate,
            value_rate,
            np.log1p(recent_pair_count),
            recent_pair_rate - recent_user_rate,
            recent_pair_rate - recent_value_rate,
            recent_pair_rate - pair_rate_stable,
        ]
    ).astype(np.float32, copy=False)


def standardize_valid(scores):
    scores = np.asarray(scores, dtype=np.float64)
    mean = float(np.mean(scores))
    std = max(float(np.std(scores)), 1e-8)
    return (scores - mean) / std, mean, std


def apply_standardization(scores, mean, std):
    return (np.asarray(scores, dtype=np.float64) - mean) / std


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)

train_users_x = np.asarray(train.X["user_id"], dtype=np.int64)
valid_users_x = np.asarray(valid.X["user_id"], dtype=np.int64)
valid_eval_users = np.asarray(valid.user_id, dtype=np.int64)

max_train_date = int(np.max(np.asarray(train.date, dtype=np.int64)))
unique_train_dates = np.unique(np.asarray(train.date, dtype=np.int64))
recent_dates = unique_train_dates[-7:]
recent_mask = np.isin(
    np.asarray(train.date, dtype=np.int64), recent_dates
)

user_cardinality = int(FEATURE_CARDINALITIES["user_id"])
user_stats = build_user_statistics(
    train_users_x,
    y_train,
    recent_mask,
    user_cardinality,
)

(
    train_user_rate,
    train_recent_user_rate,
    train_user_count,
    train_recent_user_count,
) = user_features_train(
    train_users_x, y_train, recent_mask, user_stats
)

(
    valid_user_rate,
    valid_recent_user_rate,
    valid_user_count,
    valid_recent_user_count,
) = user_features_inference(valid_users_x, user_stats)

train_blocks = [
    np.column_stack(
        [
            train_user_rate,
            train_recent_user_rate,
            train_recent_user_rate - train_user_rate,
            np.log1p(train_user_count),
            np.log1p(train_recent_user_count),
        ]
    ).astype(np.float32)
]
valid_blocks = [
    np.column_stack(
        [
            valid_user_rate,
            valid_recent_user_rate,
            valid_recent_user_rate - valid_user_rate,
            np.log1p(valid_user_count),
            np.log1p(valid_recent_user_count),
        ]
    ).astype(np.float32)
]

feature_names = [
    "user_rate",
    "recent_user_rate",
    "recent_minus_full_user_rate",
    "log_user_history_count",
    "log_recent_user_history_count",
]

tables = {}

suffixes = [
    "log_pair_count",
    "pair_rate_fast",
    "pair_lift_user",
    "pair_lift_value",
    "value_rate",
    "log_recent_pair_count",
    "recent_pair_lift_user",
    "recent_pair_lift_value",
    "recent_minus_full_pair_rate",
]

for field in PROFILE_FIELDS:
    cardinality = int(FEATURE_CARDINALITIES[field])
    train_values = np.asarray(train.X[field], dtype=np.int64)
    valid_values = np.asarray(valid.X[field], dtype=np.int64)

    table = aggregate_field(
        train_users_x,
        train_values,
        y_train,
        recent_mask,
        cardinality,
    )

    train_block = field_features_train(
        table,
        train_values,
        y_train,
        recent_mask,
        train_user_rate,
        train_recent_user_rate,
        user_stats["global_rate"],
        user_stats["recent_global_rate"],
    )
    valid_block = field_features_inference(
        table,
        valid_users_x,
        valid_values,
        valid_user_rate,
        valid_recent_user_rate,
        user_stats["global_rate"],
        user_stats["recent_global_rate"],
    )

    train_blocks.append(train_block)
    valid_blocks.append(valid_block)
    feature_names.extend([field + "__" + suffix for suffix in suffixes])

    # The training-only inverse map is no longer needed after constructing
    # leave-one-out features, substantially reducing persistent memory.
    del table["inverse"]
    tables[field] = table
    gc.collect()

x_train = np.ascontiguousarray(
    np.column_stack(train_blocks), dtype=np.float32
)
x_valid = np.ascontiguousarray(
    np.column_stack(valid_blocks), dtype=np.float32
)

del train_blocks, valid_blocks
gc.collect()

dtrain = lgb.Dataset(
    x_train,
    label=y_train,
    feature_name=feature_names,
    free_raw_data=True,
)
dvalid = lgb.Dataset(
    x_valid,
    label=y_valid,
    reference=dtrain,
    feature_name=feature_names,
    free_raw_data=True,
)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.035,
    "num_leaves": 31,
    "max_depth": 7,
    "min_data_in_leaf": 700,
    "min_sum_hessian_in_leaf": 2.0,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.88,
    "bagging_freq": 1,
    "lambda_l1": 0.25,
    "lambda_l2": 10.0,
    "max_bin": 127,
    "verbosity": -1,
    "verbose": -1,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "data_random_seed": SEED + 3,
    "num_threads": min(16, os.cpu_count() or 1),
    "force_col_wise": True,
}

model = lgb.train(
    params,
    dtrain,
    num_boost_round=550,
    valid_sets=[dvalid],
    valid_names=["valid"],
    callbacks=[
        lgb.early_stopping(50, first_metric_only=True, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

profile_valid = np.asarray(
    model.predict(x_valid, num_iteration=model.best_iteration),
    dtype=np.float64,
)

artifacts = os.environ.get("RUN_ARTIFACTS")
if not artifacts:
    raise RuntimeError("RUN_ARTIFACTS is required")

inc_valid_path = os.path.join(
    artifacts, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    artifacts, "incumbent_test_scores.npy"
)
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are missing")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if len(inc_valid) != len(valid_eval_users):
    raise ValueError("Incumbent validation score length mismatch")

inc_valid_z, inc_mean, inc_std = standardize_valid(inc_valid)
profile_valid_z, profile_mean, profile_std = standardize_valid(profile_valid)

candidate_primary = {}
best_primary = -np.inf
best_beta = None
best_metrics = None
best_valid_scores = None

betas = np.asarray(
    [
        -0.20,
        -0.10,
        -0.05,
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        0.65,
        0.80,
        1.00,
    ],
    dtype=np.float64,
)

profile_only_metrics = evaluate(
    valid_eval_users, y_valid, profile_valid_z
)
candidate_primary["profile_only"] = float(
    profile_only_metrics["primary"]
)

for beta in betas:
    candidate_scores = inc_valid_z + float(beta) * profile_valid_z
    metrics = evaluate(valid_eval_users, y_valid, candidate_scores)
    name = "incumbent_plus_profile_%+.2f" % float(beta)
    candidate_primary[name] = float(metrics["primary"])

    if float(metrics["primary"]) > best_primary:
        best_primary = float(metrics["primary"])
        best_beta = float(beta)
        best_metrics = metrics
        best_valid_scores = np.asarray(
            candidate_scores, dtype=np.float64
        ).copy()

if best_valid_scores is None:
    raise RuntimeError("No blend candidate was selected")

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        best_valid_scores,
    )

print(
    "FINDINGS "
    + json.dumps(
        {
            "profile_only_primary": float(
                profile_only_metrics["primary"]
            ),
            "profile_only_gauc": float(profile_only_metrics["gauc"]),
            "profile_only_ndcg@5": float(
                profile_only_metrics["ndcg@5"]
            ),
            "selected_beta": float(best_beta),
            "best_iteration": int(model.best_iteration),
            "recent_train_days": [
                int(x) for x in recent_dates.tolist()
            ],
            "valid_seen_user_fraction": float(
                np.mean(valid_user_count > 0)
            ),
            "valid_seen_video_pair_fraction": float(
                np.mean(
                    field_features_inference(
                        tables["video_id"],
                        valid_users_x,
                        np.asarray(valid.X["video_id"], dtype=np.int64),
                        valid_user_rate,
                        valid_recent_user_rate,
                        user_stats["global_rate"],
                        user_stats["recent_global_rate"],
                    )[:, 0] > 0
                )
            ),
        },
        sort_keys=True,
    )
)
print(
    "CANDIDATES "
    + json.dumps(candidate_primary, sort_keys=True)
)

# Free training matrices before constructing test features.
del x_train, x_valid, dtrain, dvalid
del train_user_rate, train_recent_user_rate
del train_user_count, train_recent_user_count
gc.collect()

test = load("test")
test_users_x = np.asarray(test.X["user_id"], dtype=np.int64)

(
    test_user_rate,
    test_recent_user_rate,
    test_user_count,
    test_recent_user_count,
) = user_features_inference(test_users_x, user_stats)

test_blocks = [
    np.column_stack(
        [
            test_user_rate,
            test_recent_user_rate,
            test_recent_user_rate - test_user_rate,
            np.log1p(test_user_count),
            np.log1p(test_recent_user_count),
        ]
    ).astype(np.float32)
]

for field in PROFILE_FIELDS:
    test_values = np.asarray(test.X[field], dtype=np.int64)
    test_blocks.append(
        field_features_inference(
            tables[field],
            test_users_x,
            test_values,
            test_user_rate,
            test_recent_user_rate,
            user_stats["global_rate"],
            user_stats["recent_global_rate"],
        )
    )

x_test = np.ascontiguousarray(
    np.column_stack(test_blocks), dtype=np.float32
)
del test_blocks
gc.collect()

profile_test = np.asarray(
    model.predict(x_test, num_iteration=model.best_iteration),
    dtype=np.float64,
)
profile_test_z = apply_standardization(
    profile_test, profile_mean, profile_std
)

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test) != len(test.user_id):
    raise ValueError("Incumbent test score length mismatch")
inc_test_z = apply_standardization(inc_test, inc_mean, inc_std)

test_scores = inc_test_z + best_beta * profile_test_z

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

try:
    model.save_model(
        os.path.join(artifacts, "personalized_profile_lgbm.txt"),
        num_iteration=model.best_iteration,
    )
    with open(
        os.path.join(artifacts, "personalized_profile_lgbm.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "seed": SEED,
                "profile_fields": PROFILE_FIELDS,
                "best_iteration": int(model.best_iteration),
                "selected_beta": float(best_beta),
                "profile_valid_primary": float(
                    profile_only_metrics["primary"]
                ),
                "blended_valid_primary": float(
                    best_metrics["primary"]
                ),
                "recent_dates": [
                    int(x) for x in recent_dates.tolist()
                ],
                "max_train_date": max_train_date,
            },
            f,
            indent=2,
            sort_keys=True,
        )
except Exception:
    pass

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