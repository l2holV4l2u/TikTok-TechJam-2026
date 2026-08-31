import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

START = time.time()
SEED = 20260831
rng = np.random.default_rng(SEED)

train = load("train")
valid = load("valid")
test = load("test")

y = np.asarray(train.y, dtype=np.float32)
yv = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)


def per_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    n = len(values)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, values, users))
    sorted_users = users[order]

    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = sorted_users[1:] != sorted_users[:-1]

    starts = np.where(starts_flag, np.arange(n), 0)
    starts = np.maximum.accumulate(starts)
    within = np.arange(n) - starts

    group_starts = np.flatnonzero(starts_flag)
    group_ends = np.r_[group_starts[1:], n]
    sizes = group_ends - group_starts
    repeated_sizes = np.repeat(sizes, sizes)

    ranks = np.where(
        repeated_sizes > 1,
        within / np.maximum(repeated_sizes - 1, 1),
        0.5,
    ).astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranks
    return result


def finite_float(x):
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0)


# Exclude user_id because it is constant within each evaluated user's candidate
# set and encourages temporally fragile memorization. The retained fields cover
# item identity, content, context, and user state.
categorical_fields = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "video_type",
    "music_type",
    "hour",
    "is_live_streamer",
    "is_video_author",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_bucket",
    "register_days_range",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat4",
    "onehot_feat5",
    "onehot_feat6",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat9",
    "onehot_feat10",
    "onehot_feat11",
    "onehot_feat12",
]

numeric_fields = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

# Organizer-provided histories are train-only. Train values are leave-one-out;
# validation and test values use the complete train split.
hist_tr_video = historical_features("train", key="video_id")
hist_va_video = historical_features("valid", key="video_id")
hist_te_video = historical_features("test", key="video_id")

hist_tr_author = historical_features("train", key="author_id")
hist_va_author = historical_features("valid", key="author_id")
hist_te_author = historical_features("test", key="author_id")

history_keys_video = sorted(hist_tr_video.keys())
history_keys_author = sorted(hist_tr_author.keys())


def make_matrix(split, hv, ha):
    columns = []

    for field in categorical_fields:
        columns.append(np.asarray(split.X[field], dtype=np.float32))

    for field in numeric_fields:
        raw = finite_float(split.num[field])
        columns.append(np.log1p(np.maximum(raw, 0.0)).astype(np.float32))

    for key in history_keys_video:
        columns.append(finite_float(hv[key]))

    for key in history_keys_author:
        columns.append(finite_float(ha[key]))

    return np.column_stack(columns).astype(np.float32, copy=False)


Xtr = make_matrix(train, hist_tr_video, hist_tr_author)
Xva = make_matrix(valid, hist_va_video, hist_va_author)
Xte = make_matrix(test, hist_te_video, hist_te_author)

n_categorical = len(categorical_fields)
categorical_indices = list(range(n_categorical))
feature_names = (
    categorical_fields
    + ["log_" + x for x in numeric_fields]
    + ["video_" + x for x in history_keys_video]
    + ["author_" + x for x in history_keys_author]
)

dates = np.asarray(train.date, dtype=np.int32)
last_train_date = int(np.max(dates))
day_age = (last_train_date - dates).astype(np.float32)

# Moderate decay avoids discarding the high-volume early days while shifting
# the fitted response surface toward the split boundary.
train_weight = np.exp(
    -np.log(2.0) * day_age / 7.0
).astype(np.float32)
train_weight /= max(float(np.mean(train_weight)), 1e-6)

print(
    "FINDINGS matrix_shape=%s categorical=%d numeric_history=%d "
    "weight_range=[%.4f,%.4f]"
    % (
        str(Xtr.shape),
        n_categorical,
        Xtr.shape[1] - n_categorical,
        float(train_weight.min()),
        float(train_weight.max()),
    )
)

# ----------------------------------------------------------------------
# Family 1: pointwise histogram GBDT.
#
# Trees discover nonlinear threshold effects and high-order interactions
# between content identities, user state, and leakage-safe entity histories.
# ----------------------------------------------------------------------

binary_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 350,
    "min_sum_hessian_in_leaf": 5.0,
    "feature_fraction": 0.78,
    "bagging_fraction": 0.82,
    "bagging_freq": 1,
    "lambda_l1": 0.15,
    "lambda_l2": 8.0,
    "max_bin": 127,
    "cat_smooth": 25.0,
    "cat_l2": 12.0,
    "max_cat_to_onehot": 16,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "num_threads": -1,
    "verbose": -1,
}

binary_dataset = lgb.Dataset(
    Xtr,
    label=y,
    weight=train_weight,
    categorical_feature=categorical_indices,
    feature_name=feature_names,
    free_raw_data=False,
)

binary_model = lgb.train(
    binary_params,
    binary_dataset,
    num_boost_round=170,
)

binary_valid = binary_model.predict(
    Xva, num_iteration=binary_model.current_iteration()
).astype(np.float64)
binary_test = binary_model.predict(
    Xte, num_iteration=binary_model.current_iteration()
).astype(np.float64)

del binary_model, binary_dataset
gc.collect()

# ----------------------------------------------------------------------
# Family 2: LambdaRank GBDT.
#
# Sorting training impressions into user groups changes the optimization
# target from calibrated row probability to within-user ordering, with the
# largest gradients concentrated near the nDCG cutoff.
# ----------------------------------------------------------------------

train_users = np.asarray(train.user_id, dtype=np.int64)
row_index = np.arange(len(y), dtype=np.int64)
rank_order = np.lexsort((row_index, train_users))
sorted_users = train_users[rank_order]

group_starts = np.r_[
    0,
    np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1,
]
group_sizes = np.diff(np.r_[group_starts, len(sorted_users)]).astype(np.int32)

Xrank = np.ascontiguousarray(Xtr[rank_order])
yrank = y[rank_order]
wrank = train_weight[rank_order]

rank_params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "label_gain": [0, 1],
    "lambdarank_truncation_level": 10,
    "learning_rate": 0.045,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 350,
    "min_sum_hessian_in_leaf": 5.0,
    "feature_fraction": 0.78,
    "bagging_fraction": 0.82,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 10.0,
    "max_bin": 127,
    "cat_smooth": 25.0,
    "cat_l2": 12.0,
    "max_cat_to_onehot": 16,
    "seed": SEED + 10,
    "feature_fraction_seed": SEED + 11,
    "bagging_seed": SEED + 12,
    "num_threads": -1,
    "verbose": -1,
}

rank_dataset = lgb.Dataset(
    Xrank,
    label=yrank,
    weight=wrank,
    group=group_sizes,
    categorical_feature=categorical_indices,
    feature_name=feature_names,
    free_raw_data=False,
)

rank_model = lgb.train(
    rank_params,
    rank_dataset,
    num_boost_round=145,
)

lambda_valid = rank_model.predict(
    Xva, num_iteration=rank_model.current_iteration()
).astype(np.float64)
lambda_test = rank_model.predict(
    Xte, num_iteration=rank_model.current_iteration()
).astype(np.float64)

del rank_model, rank_dataset, Xrank, yrank, wrank, rank_order
gc.collect()

# The large dense training matrix is no longer needed.
del Xtr
gc.collect()

# ----------------------------------------------------------------------
# Family 3: random-intersection empirical kernel.
#
# Each random categorical conjunction defines a partition of impressions.
# Shrunk response rates from many different partitions are averaged in logit
# space. Unlike a tree, this is an exchangeable matching kernel: two rows
# influence each other exactly when they repeatedly share content/context
# intersections. Randomized partitions reduce dependence on any one drifting
# identity field.
# ----------------------------------------------------------------------

kernel_fields = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "video_type",
    "music_type",
    "hour",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_bucket",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

kernel_valid_sum = np.zeros(len(valid.user_id), dtype=np.float64)
kernel_test_sum = np.zeros(len(test.user_id), dtype=np.float64)

weighted_global = float(
    np.sum(train_weight.astype(np.float64) * y.astype(np.float64))
    / np.sum(train_weight.astype(np.float64))
)
global_logit = np.log(
    np.clip(weighted_global, 1e-5, 1.0 - 1e-5)
    / np.clip(1.0 - weighted_global, 1e-5, 1.0)
)

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
HASH_CONST = np.uint64(0x9E3779B97F4A7C15)


def conjunction_hash(split, fields):
    h = np.full(len(split.user_id), HASH_CONST, dtype=np.uint64)
    for j, field in enumerate(fields):
        x = np.asarray(split.X[field], dtype=np.uint64)
        salt = np.uint64(
            (0xBF58476D1CE4E5B9 + j * 0x1F123BB5) & 0xFFFFFFFFFFFFFFFF
        )
        z = (x + salt) & MASK64
        z ^= z >> np.uint64(30)
        z = (z * np.uint64(0xBF58476D1CE4E5B9)) & MASK64
        z ^= z >> np.uint64(27)
        z = (z * np.uint64(0x94D049BB133111EB)) & MASK64
        z ^= z >> np.uint64(31)
        h ^= z
        h = (
            h * np.uint64(0x9E3779B185EBCA87)
            + np.uint64(0xD1B54A32D192ED03)
        ) & MASK64
    return h


def lookup_rates(query_keys, unique_keys, rates, fallback):
    pos = np.searchsorted(unique_keys, query_keys)
    safe = np.minimum(pos, len(unique_keys) - 1)
    found = (
        (pos < len(unique_keys))
        & (unique_keys[safe] == query_keys)
    )
    out = np.full(len(query_keys), fallback, dtype=np.float64)
    out[found] = rates[safe[found]]
    return out


n_partitions = 14
partition_descriptions = []

for p in range(n_partitions):
    width = 2 + (p % 3)
    # Identity fields are available but not forced into every partition.
    selected = rng.choice(
        kernel_fields, size=width, replace=False
    ).tolist()
    partition_descriptions.append("+".join(selected))

    htr = conjunction_hash(train, selected)
    hva = conjunction_hash(valid, selected)
    hte = conjunction_hash(test, selected)

    unique_keys, inverse = np.unique(htr, return_inverse=True)
    cell_weight = np.bincount(
        inverse, weights=train_weight
    ).astype(np.float64)
    cell_positive = np.bincount(
        inverse, weights=train_weight * y
    ).astype(np.float64)

    # Wider intersections are sparser and receive stronger global shrinkage.
    prior = 18.0 + 10.0 * (width - 2)
    cell_rate = (
        cell_positive + prior * weighted_global
    ) / np.maximum(cell_weight + prior, 1e-8)

    va_rate = lookup_rates(
        hva, unique_keys, cell_rate, weighted_global
    )
    te_rate = lookup_rates(
        hte, unique_keys, cell_rate, weighted_global
    )

    va_logit = np.log(
        np.clip(va_rate, 1e-5, 1.0 - 1e-5)
        / np.clip(1.0 - va_rate, 1e-5, 1.0)
    )
    te_logit = np.log(
        np.clip(te_rate, 1e-5, 1.0 - 1e-5)
        / np.clip(1.0 - te_rate, 1e-5, 1.0)
    )

    kernel_valid_sum += va_logit - global_logit
    kernel_test_sum += te_logit - global_logit

    del htr, hva, hte, unique_keys, inverse
    del cell_weight, cell_positive, cell_rate
    gc.collect()

kernel_valid = kernel_valid_sum / n_partitions
kernel_test = kernel_test_sum / n_partitions

print(
    "FINDINGS kernel_partitions=%s"
    % "|".join(partition_descriptions)
)

# ----------------------------------------------------------------------
# Compare each standalone family and all incumbent blends. Rank-percentile
# aggregation is invariant to differing score calibration and preserves the
# within-user nature of both benchmark metrics.
# ----------------------------------------------------------------------

inc_valid_rank = per_user_rank(valid.user_id, inc_valid)
inc_test_rank = per_user_rank(test.user_id, inc_test)

families = {
    "pointwise_gbdt": (binary_valid, binary_test),
    "lambdarank_gbdt": (lambda_valid, lambda_test),
    "random_intersection_kernel": (kernel_valid, kernel_test),
}

blend_weights = [0.0, 0.05, 0.10, 0.16, 0.24, 0.34, 0.46, 0.60]

candidate_log = {}
best_primary = -np.inf
best_valid_scores = None
best_test_scores = None
best_raw_valid = None
best_name = None

for family_name, (raw_valid, raw_test) in families.items():
    family_valid_rank = per_user_rank(valid.user_id, raw_valid)
    family_test_rank = per_user_rank(test.user_id, raw_test)

    raw_metrics = evaluate(valid.user_id, yv, family_valid_rank)
    candidate_log[family_name + "_raw"] = float(raw_metrics["primary"])

    for own_weight in blend_weights:
        candidate_valid = (
            (1.0 - own_weight) * inc_valid_rank
            + own_weight * family_valid_rank
        )
        candidate_test = (
            (1.0 - own_weight) * inc_test_rank
            + own_weight * family_test_rank
        )

        metrics = evaluate(valid.user_id, yv, candidate_valid)
        name = "%s_blend_%.2f" % (family_name, own_weight)
        candidate_log[name] = float(metrics["primary"])

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_valid_scores = candidate_valid.copy()
            best_test_scores = candidate_test.copy()
            best_raw_valid = family_valid_rank.copy()
            best_name = name

final_metrics = evaluate(valid.user_id, yv, best_valid_scores)

print(
    "FINDINGS selected=%s primary=%.6f"
    % (best_name, float(final_metrics["primary"]))
)
print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(final_metrics["primary"]),
            "gauc": float(final_metrics["gauc"]),
            "ndcg@5": float(final_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)