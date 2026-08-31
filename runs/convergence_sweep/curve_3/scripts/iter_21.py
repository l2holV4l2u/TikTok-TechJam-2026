import os
import time
import json
import gc

import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 20260831
THREADS = max(1, min(12, os.cpu_count() or 1))


def safe_logit(probability):
    probability = np.clip(
        np.asarray(probability, dtype=np.float64),
        1e-5,
        1.0 - 1e-5,
    )
    return np.log(probability / (1.0 - probability))


def within_user_rank(scores, users):
    scores = np.nan_to_num(
        np.asarray(scores, dtype=np.float64),
        nan=0.0,
        posinf=1e20,
        neginf=-1e20,
    )
    users = np.asarray(users, dtype=np.int64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]
    positions = np.arange(n, dtype=np.int64)

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.maximum.accumulate(
        np.where(starts_mask, positions, 0)
    )

    ends_mask = np.empty(n, dtype=bool)
    ends_mask[-1] = True
    ends_mask[:-1] = sorted_users[:-1] != sorted_users[1:]
    ends = np.minimum.accumulate(
        np.where(ends_mask, positions, n - 1)[::-1]
    )[::-1]

    denominator = np.maximum(ends - starts, 1)
    normalized_rank = (positions - starts) / denominator

    result = np.empty(n, dtype=np.float64)
    result[order] = normalized_rank
    return result


train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)

train_users = np.asarray(train.user_id, dtype=np.int64)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)

train_dates = np.asarray(train.date, dtype=np.int32)
last_train_date = int(np.max(train_dates))
day_age = (last_train_date - train_dates).astype(np.float64)

# Emphasize behavior near the date boundary while retaining enough effective
# sample size for rare entities.
sample_weights = np.power(0.5, day_age / 5.0).astype(np.float32)
sample_weights /= np.mean(sample_weights)

weighted_rate = float(
    np.sum(sample_weights * y_train) / np.sum(sample_weights)
)
weighted_logit = float(safe_logit(weighted_rate))

TE_FIELDS = [
    "video_id",
    "author_id",
    "user_id",
    "tab",
    "tag",
    "onehot_feat3",
    "upload_type",
    "onehot_feat8",
    "duration_bucket",
    "onehot_feat1",
    "music_type",
    "onehot_feat7",
    "user_active_degree",
    "register_days_bucket",
    "register_days_range",
    "onehot_feat0",
    "onehot_feat12",
    "fans_user_num_range",
    "onehot_feat11",
    "onehot_feat6",
    "onehot_feat2",
    "friend_user_num_range",
    "hour",
    "is_live_streamer",
    "onehot_feat9",
    "onehot_feat16",
    "follow_user_num_range",
    "onehot_feat4",
    "onehot_feat5",
    "is_video_author",
    "onehot_feat10",
    "onehot_feat15",
    "onehot_feat14",
    "onehot_feat13",
    "video_type",
    "onehot_feat17",
    "is_lowactive_period",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

TE_STRENGTH = {
    "video_id": 35.0,
    "author_id": 40.0,
    "user_id": 50.0,
    "tab": 150.0,
    "tag": 100.0,
    "onehot_feat3": 70.0,
    "upload_type": 130.0,
    "onehot_feat8": 80.0,
}
DEFAULT_STRENGTH = 180.0


def field_values(split, field):
    if field == "video_id":
        return np.asarray(split.video_id, dtype=np.int64)
    if field == "user_id":
        return np.asarray(split.user_id, dtype=np.int64)
    return np.asarray(split.X[field], dtype=np.int64)


target_tables = {}
for field in TE_FIELDS:
    values = field_values(train, field)
    cardinality = int(FEATURE_CARDINALITIES[field])
    counts = np.bincount(
        values,
        weights=sample_weights,
        minlength=cardinality,
    ).astype(np.float64)
    positives = np.bincount(
        values,
        weights=sample_weights * y_train,
        minlength=cardinality,
    ).astype(np.float64)
    target_tables[field] = (counts, positives)


def target_encode(split, field, leave_one_out):
    values = field_values(split, field)
    counts, positives = target_tables[field]
    safe_values = np.clip(values, 0, len(counts) - 1)

    local_counts = counts[safe_values].copy()
    local_positives = positives[safe_values].copy()

    if leave_one_out:
        local_counts -= sample_weights
        local_positives -= sample_weights * y_train

    strength = TE_STRENGTH.get(field, DEFAULT_STRENGTH)
    rates = (
        local_positives + strength * weighted_rate
    ) / np.maximum(local_counts + strength, 1e-8)

    encoded = safe_logit(rates) - weighted_logit
    reliability = local_counts / np.maximum(local_counts + strength, 1e-8)
    return encoded.astype(np.float32), reliability.astype(np.float32)


def get_histories(split_name):
    histories = {}
    histories.update(historical_features(split_name, key="video_id"))
    histories.update(historical_features(split_name, key="author_id"))
    return histories


hist_train = get_histories("train")
hist_valid = get_histories("valid")
hist_test = get_histories("test")

history_keys = sorted(
    set(hist_train.keys())
    & set(hist_valid.keys())
    & set(hist_test.keys())
)


def build_features(split, histories, leave_one_out):
    columns = []

    for field in TE_FIELDS:
        encoded, reliability = target_encode(split, field, leave_one_out)
        columns.append(encoded)
        if field in (
            "video_id",
            "author_id",
            "user_id",
            "tab",
            "tag",
            "onehot_feat3",
        ):
            columns.append(reliability)

    for field in NUM_FIELDS:
        values = np.asarray(split.num[field], dtype=np.float32)
        missing = ~np.isfinite(values)
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        values = np.log1p(np.maximum(values, 0.0)).astype(np.float32)
        columns.append(values)
        if np.any(missing):
            columns.append(missing.astype(np.float32))

    for key in history_keys:
        values = np.asarray(histories[key], dtype=np.float32)
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        columns.append(values)

    hour = np.asarray(split.X["hour"], dtype=np.float32)
    columns.append(np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32))
    columns.append(np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32))

    return np.column_stack(columns).astype(np.float32, copy=False)


X_train = build_features(train, hist_train, True)
X_valid = build_features(valid, hist_valid, False)
X_test = build_features(test, hist_test, False)

del hist_train, hist_valid, hist_test
gc.collect()

# Robust scaling is fit strictly on train.
feature_mean = np.average(
    X_train.astype(np.float64),
    axis=0,
    weights=sample_weights,
)
feature_variance = np.average(
    (X_train.astype(np.float64) - feature_mean) ** 2,
    axis=0,
    weights=sample_weights,
)
feature_scale = np.sqrt(np.maximum(feature_variance, 1e-5))

X_train = np.clip(
    (X_train - feature_mean) / feature_scale,
    -10.0,
    10.0,
).astype(np.float32)
X_valid = np.clip(
    (X_valid - feature_mean) / feature_scale,
    -10.0,
    10.0,
).astype(np.float32)
X_test = np.clip(
    (X_test - feature_mean) / feature_scale,
    -10.0,
    10.0,
).astype(np.float32)

aux_keys = set(train.aux.keys())
if "is_click" not in aux_keys or "play_time_ms" not in aux_keys:
    raise RuntimeError(
        "Required train-only auxiliary targets is_click/play_time_ms are absent"
    )

click_train = np.nan_to_num(
    np.asarray(train.aux["is_click"], dtype=np.float32),
    nan=0.0,
    posinf=1.0,
    neginf=0.0,
)
click_train = (click_train > 0.5).astype(np.int8)

play_time = np.nan_to_num(
    np.asarray(train.aux["play_time_ms"], dtype=np.float64),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)
duration = np.nan_to_num(
    np.asarray(train.num["duration_ms"], dtype=np.float64),
    nan=1.0,
    posinf=1.0,
    neginf=1.0,
)
duration = np.maximum(duration, 1000.0)

# The capped ratio suppresses extreme background-play outliers. Its logarithm
# remains continuous for the regression family.
completion_ratio = np.clip(play_time / duration, 0.0, 2.0)
completion_target = np.log1p(completion_ratio).astype(np.float32)

print(
    "FINDINGS "
    + json.dumps(
        {
            "train_click_rate": float(np.mean(click_train)),
            "train_completion_mean": float(np.mean(completion_ratio)),
            "train_completion_longview_corr": float(
                np.corrcoef(completion_ratio, y_train)[0, 1]
            ),
            "feature_dimension": int(X_train.shape[1]),
        },
        sort_keys=True,
    )
)

common_params = {
    "verbosity": -1,
    "num_threads": THREADS,
    "learning_rate": 0.055,
    "num_leaves": 40,
    "max_depth": -1,
    "min_data_in_leaf": 900,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.80,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 7.0,
    "max_bin": 127,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
}

candidate_valid = {}
candidate_test = {}

# Family 1: joint-state multiclass modeling. Unlike independent binary
# prediction, this learns the full click/long-view state distribution.
joint_target = (2 * y_train + click_train).astype(np.int32)
joint_params = dict(common_params)
joint_params.update({
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_class": 4,
})

joint_model = lgb.train(
    joint_params,
    lgb.Dataset(
        X_train,
        label=joint_target,
        weight=sample_weights,
        free_raw_data=True,
    ),
    num_boost_round=180,
)
joint_valid_probability = joint_model.predict(X_valid)
joint_test_probability = joint_model.predict(X_test)

candidate_valid["joint_state_multiclass"] = (
    joint_valid_probability[:, 2] + joint_valid_probability[:, 3]
).astype(np.float32)
candidate_test["joint_state_multiclass"] = (
    joint_test_probability[:, 2] + joint_test_probability[:, 3]
).astype(np.float32)

del joint_model, joint_valid_probability, joint_test_probability
gc.collect()

# Family 2: continuous watch-completion regression. It forms predictions from
# a train-only dense engagement surrogate rather than the binary scored label.
regression_params = dict(common_params)
regression_params.update({
    "objective": "huber",
    "metric": "huber",
    "alpha": 0.88,
    "seed": SEED + 10,
    "feature_fraction_seed": SEED + 11,
    "bagging_seed": SEED + 12,
})

completion_model = lgb.train(
    regression_params,
    lgb.Dataset(
        X_train,
        label=completion_target,
        weight=sample_weights,
        free_raw_data=True,
    ),
    num_boost_round=180,
)
candidate_valid["completion_regression"] = completion_model.predict(
    X_valid
).astype(np.float32)
candidate_test["completion_regression"] = completion_model.predict(
    X_test
).astype(np.float32)

del completion_model
gc.collect()

# Family 3: auxiliary ordinal LambdaRank. Relevance is derived only from
# train watch completion and click, while groups directly represent users.
ordinal_relevance = np.digitize(
    completion_ratio,
    bins=np.asarray([0.12, 0.35, 0.70, 1.00], dtype=np.float64),
).astype(np.int32)
ordinal_relevance = np.maximum(
    ordinal_relevance,
    click_train.astype(np.int32),
)

user_order = np.argsort(train_users, kind="stable")
sorted_users = train_users[user_order]
group_starts = np.r_[
    0,
    np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1,
    len(sorted_users),
]
group_sizes = np.diff(group_starts).astype(np.int32)

rank_params = dict(common_params)
rank_params.update({
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "label_gain": [0, 1, 3, 7, 15],
    "lambdarank_truncation_level": 10,
    "learning_rate": 0.045,
    "num_leaves": 32,
    "seed": SEED + 20,
    "feature_fraction_seed": SEED + 21,
    "bagging_seed": SEED + 22,
})

rank_model = lgb.train(
    rank_params,
    lgb.Dataset(
        X_train[user_order],
        label=ordinal_relevance[user_order],
        weight=sample_weights[user_order],
        group=group_sizes,
        free_raw_data=True,
    ),
    num_boost_round=150,
)
candidate_valid["auxiliary_ordinal_lambdarank"] = rank_model.predict(
    X_valid
).astype(np.float32)
candidate_test["auxiliary_ordinal_lambdarank"] = rank_model.predict(
    X_test
).astype(np.float32)

del rank_model, user_order, sorted_users, group_starts, group_sizes
gc.collect()

# A cross-family rank aggregate is included because the three outputs estimate
# different notions of relevance and have incomparable numerical scales.
family_names = list(candidate_valid.keys())
family_valid_ranks = [
    within_user_rank(candidate_valid[name], valid_users)
    for name in family_names
]
family_test_ranks = [
    within_user_rank(candidate_test[name], test_users)
    for name in family_names
]
candidate_valid["cross_family_rank_mean"] = np.mean(
    np.column_stack(family_valid_ranks),
    axis=1,
).astype(np.float32)
candidate_test["cross_family_rank_mean"] = np.mean(
    np.column_stack(family_test_ranks),
    axis=1,
).astype(np.float32)

shared = os.environ.get("SHARED_ARTIFACTS")
incumbent_valid_path = (
    os.path.join(shared, "incumbent_valid_scores.npy") if shared else ""
)
incumbent_test_path = (
    os.path.join(shared, "incumbent_test_scores.npy") if shared else ""
)

if not (
    incumbent_valid_path
    and incumbent_test_path
    and os.path.exists(incumbent_valid_path)
    and os.path.exists(incumbent_test_path)
):
    raise RuntimeError("Trusted incumbent prediction files are unavailable")

incumbent_valid = np.asarray(
    np.load(incumbent_valid_path),
    dtype=np.float64,
)
incumbent_test = np.asarray(
    np.load(incumbent_test_path),
    dtype=np.float64,
)

incumbent_valid_rank = within_user_rank(incumbent_valid, valid_users)
incumbent_test_rank = within_user_rank(incumbent_test, test_users)

candidate_scores = {}
best_primary = -np.inf
best_name = None
best_valid_scores = None
best_test_scores = None
best_raw_valid = None
best_alpha = None

# Score every standalone family and a conservative family/incumbent blend.
# Alpha zero is included so weak auxiliary supervision cannot damage the run.
blend_alphas = [0.0, 0.04, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50]

for name in list(candidate_valid.keys()):
    raw_valid = np.asarray(candidate_valid[name], dtype=np.float64)
    raw_test = np.asarray(candidate_test[name], dtype=np.float64)
    raw_metrics = evaluate(valid_users, y_valid, raw_valid)
    candidate_scores[name] = float(raw_metrics["primary"])

    valid_rank = within_user_rank(raw_valid, valid_users)
    test_rank = within_user_rank(raw_test, test_users)

    local_best_primary = -np.inf
    local_best_alpha = None

    for alpha in blend_alphas:
        blended_valid = (
            (1.0 - alpha) * incumbent_valid_rank + alpha * valid_rank
        )
        metrics = evaluate(valid_users, y_valid, blended_valid)
        primary = float(metrics["primary"])

        if primary > local_best_primary:
            local_best_primary = primary
            local_best_alpha = alpha

        if primary > best_primary:
            best_primary = primary
            best_name = name
            best_alpha = alpha
            best_valid_scores = blended_valid.copy()
            best_test_scores = (
                (1.0 - alpha) * incumbent_test_rank + alpha * test_rank
            )
            best_raw_valid = raw_valid.copy()

    candidate_scores[name + "_best_incumbent_blend"] = float(
        local_best_primary
    )
    candidate_scores[name + "_blend_alpha"] = float(local_best_alpha)

final_metrics = evaluate(valid_users, y_valid, best_valid_scores)

print(
    "FINDINGS "
    + json.dumps(
        {
            "selected_family": best_name,
            "selected_candidate_weight": float(best_alpha),
            "selected_primary": float(final_metrics["primary"]),
        },
        sort_keys=True,
    )
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
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