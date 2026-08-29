import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2024
NUM_BOOST_ROUND = 260

AFFINITY_FIELDS = [
    "duration_bucket",
    "tag",
    "tab",
    "upload_type",
    "author_id",
    "video_id",
    "onehot_feat3",
    "onehot_feat8",
    "hour",
]

SMOOTHING_VALUES = (5.0, 20.0)


def safe_logit(p):
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


def build_affinity_features(reference, target=None, leave_one_out=False):
    """
    Construct user-by-context empirical-Bayes features.

    If target is None, features are produced for the reference rows.
    With leave_one_out=True, each reference row's own label is removed
    from all statistics used for that row.
    """
    y_ref = np.asarray(reference.y, dtype=np.float64)
    user_ref = np.asarray(reference.X["user_id"], dtype=np.int64)
    n_ref = len(y_ref)

    if target is None:
        target = reference
        same_rows = True
    else:
        same_rows = False

    user_target = np.asarray(target.X["user_id"], dtype=np.int64)
    n_target = len(user_target)
    global_rate = float(np.mean(y_ref))
    global_logit = float(safe_logit(global_rate))

    columns = []

    for field in AFFINITY_FIELDS:
        context_ref = np.asarray(reference.X[field], dtype=np.int64)
        context_target = np.asarray(target.X[field], dtype=np.int64)
        cardinality = int(FEATURE_CARDINALITIES[field])

        context_count = np.bincount(
            context_ref,
            minlength=cardinality,
        ).astype(np.float64)
        context_positive = np.bincount(
            context_ref,
            weights=y_ref,
            minlength=cardinality,
        ).astype(np.float64)

        pair_key_ref = user_ref * np.int64(cardinality) + context_ref
        unique_keys, inverse, pair_count_unique = np.unique(
            pair_key_ref,
            return_inverse=True,
            return_counts=True,
        )
        pair_positive_unique = np.bincount(
            inverse,
            weights=y_ref,
            minlength=len(unique_keys),
        ).astype(np.float64)
        pair_count_unique = pair_count_unique.astype(np.float64)

        if same_rows:
            pair_count = pair_count_unique[inverse]
            pair_positive = pair_positive_unique[inverse]
            ctx_count = context_count[context_ref]
            ctx_positive = context_positive[context_ref]

            if leave_one_out:
                pair_count = pair_count - 1.0
                pair_positive = pair_positive - y_ref
                ctx_count = ctx_count - 1.0
                ctx_positive = ctx_positive - y_ref
        else:
            pair_key_target = (
                user_target * np.int64(cardinality) + context_target
            )
            locations = np.searchsorted(unique_keys, pair_key_target)
            found = locations < len(unique_keys)
            safe_locations = np.minimum(
                locations, max(len(unique_keys) - 1, 0)
            )
            if len(unique_keys):
                found &= unique_keys[safe_locations] == pair_key_target
            else:
                found[:] = False

            pair_count = np.zeros(n_target, dtype=np.float64)
            pair_positive = np.zeros(n_target, dtype=np.float64)
            if np.any(found):
                matched = safe_locations[found]
                pair_count[found] = pair_count_unique[matched]
                pair_positive[found] = pair_positive_unique[matched]

            ctx_count = context_count[context_target]
            ctx_positive = context_positive[context_target]

        context_rate = (
            ctx_positive + 20.0 * global_rate
        ) / (ctx_count + 20.0)
        context_logit = safe_logit(context_rate)

        columns.append(
            np.asarray(
                np.clip(context_logit - global_logit, -4.0, 4.0),
                dtype=np.float32,
            )
        )

        for smoothing in SMOOTHING_VALUES:
            pair_rate = (
                pair_positive + smoothing * context_rate
            ) / (pair_count + smoothing)
            affinity = safe_logit(pair_rate) - context_logit
            columns.append(
                np.asarray(
                    np.clip(affinity, -5.0, 5.0),
                    dtype=np.float32,
                )
            )

        columns.append(
            np.asarray(np.log1p(pair_count), dtype=np.float32)
        )

        del (
            context_ref,
            context_target,
            context_count,
            context_positive,
            pair_key_ref,
            unique_keys,
            inverse,
            pair_count_unique,
            pair_positive_unique,
            pair_count,
            pair_positive,
            ctx_count,
            ctx_positive,
            context_rate,
            context_logit,
        )
        gc.collect()

    # Continuous duration supplies finer granularity than duration_bucket.
    duration = np.asarray(target.num["duration_ms"], dtype=np.float64)
    duration = np.nan_to_num(
        duration,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    columns.append(
        np.asarray(np.log1p(np.maximum(duration, 0.0)), dtype=np.float32)
    )

    # Account-side quantities can modulate whether sparse affinities are
    # reliable, but they do not contain row outcomes.
    for name in [
        "user_fans_user_num",
        "user_follow_user_num",
        "user_friend_user_num",
        "user_register_days",
    ]:
        values = np.asarray(target.num[name], dtype=np.float64)
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        columns.append(
            np.asarray(
                np.log1p(np.maximum(values, 0.0)),
                dtype=np.float32,
            )
        )

    matrix = np.ascontiguousarray(
        np.column_stack(columns),
        dtype=np.float32,
    )
    return matrix


def train_booster(X, y, rounds):
    dataset = lgb.Dataset(
        X,
        label=np.asarray(y, dtype=np.float32),
        free_raw_data=True,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "max_depth": 8,
        "min_data_in_leaf": 800,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 4.0,
        "max_bin": 127,
        "num_threads": max(1, min(12, os.cpu_count() or 1)),
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "data_random_seed": SEED + 3,
        "verbose": -1,
    }
    return lgb.train(
        params,
        dataset,
        num_boost_round=rounds,
    )


def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return (values - mean) / std


def load_incumbent(valid_length):
    artifact_dir = os.environ.get("RUN_ARTIFACTS")
    if not artifact_dir:
        return None, None

    valid_path = os.path.join(
        artifact_dir, "incumbent_valid_scores.npy"
    )
    test_path = os.path.join(
        artifact_dir, "incumbent_test_scores.npy"
    )
    if not (
        os.path.exists(valid_path)
        and os.path.exists(test_path)
    ):
        return None, None

    valid_scores = np.asarray(np.load(valid_path), dtype=np.float64)
    test_scores = np.asarray(np.load(test_path), dtype=np.float64)
    if len(valid_scores) != valid_length:
        return None, None
    return valid_scores, test_scores


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)

print(
    "FINDINGS "
    + json.dumps(
        {
            "phase": "feature_construction",
            "fields": AFFINITY_FIELDS,
            "smoothing": list(SMOOTHING_VALUES),
        }
    )
)

X_train = build_affinity_features(
    train,
    target=None,
    leave_one_out=True,
)
X_valid = build_affinity_features(
    train,
    target=valid,
    leave_one_out=False,
)

print(
    "FINDINGS "
    + json.dumps(
        {
            "train_feature_shape": list(X_train.shape),
            "valid_feature_shape": list(X_valid.shape),
            "train_feature_finite": bool(np.isfinite(X_train).all()),
            "valid_feature_finite": bool(np.isfinite(X_valid).all()),
        }
    )
)

booster = train_booster(
    X_train,
    y_train,
    NUM_BOOST_ROUND,
)
personal_valid_raw = booster.predict(
    X_valid,
    num_iteration=NUM_BOOST_ROUND,
)
personal_valid_logit = safe_logit(personal_valid_raw)
personal_valid_z = zscore(personal_valid_logit)

# A transparent affinity-only score tests whether the personalized
# statistics themselves rank usefully without the tree model.
affinity_column_indices = []
for field_index in range(len(AFFINITY_FIELDS)):
    base = field_index * 4
    affinity_column_indices.extend([base + 1, base + 2])

direct_valid = np.mean(
    X_valid[:, affinity_column_indices],
    axis=1,
).astype(np.float64)
direct_valid_z = zscore(direct_valid)

incumbent_valid, incumbent_test = load_incumbent(len(valid.y))

candidate_scores = {}
candidate_metadata = {}

candidate_scores["personal_lgbm"] = personal_valid_z
candidate_metadata["personal_lgbm"] = ("lgbm", 1.0)

candidate_scores["direct_affinity"] = direct_valid_z
candidate_metadata["direct_affinity"] = ("direct", 1.0)

if incumbent_valid is not None:
    incumbent_valid_z = zscore(incumbent_valid)

    # Include incumbent alone so validation selection cannot make the
    # reported result worse merely because the new signal is unhelpful.
    candidate_scores["incumbent"] = incumbent_valid_z
    candidate_metadata["incumbent"] = ("incumbent", 0.0)

    for alpha in np.linspace(0.10, 0.90, 9):
        name = "incumbent_personal_{:.2f}".format(alpha)
        candidate_scores[name] = (
            (1.0 - alpha) * incumbent_valid_z
            + alpha * personal_valid_z
        )
        candidate_metadata[name] = ("lgbm", float(alpha))

    for alpha in (0.10, 0.20, 0.30, 0.40):
        name = "incumbent_direct_{:.2f}".format(alpha)
        candidate_scores[name] = (
            (1.0 - alpha) * incumbent_valid_z
            + alpha * direct_valid_z
        )
        candidate_metadata[name] = ("direct", float(alpha))

candidate_metrics = {
    name: evaluate(valid.user_id, valid.y, scores)
    for name, scores in candidate_scores.items()
}
best_name = max(
    candidate_metrics,
    key=lambda name: candidate_metrics[name]["primary"],
)
best_kind, best_alpha = candidate_metadata[best_name]
valid_scores = candidate_scores[best_name]
metrics = candidate_metrics[best_name]

print(
    "CANDIDATES "
    + json.dumps(
        {
            name: float(result["primary"])
            for name, result in candidate_metrics.items()
        },
        sort_keys=True,
    )
)
print(
    "FINDINGS "
    + json.dumps(
        {
            "selected": best_name,
            "selected_kind": best_kind,
            "selected_new_model_weight": best_alpha,
            "personal_lgbm_primary": float(
                candidate_metrics["personal_lgbm"]["primary"]
            ),
            "direct_affinity_primary": float(
                candidate_metrics["direct_affinity"]["primary"]
            ),
            "selected_primary": float(metrics["primary"]),
        }
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Refit the identical feature/model recipe on train + validation, then
# produce test scores without reading test labels.
test = load("test")

combined_y = np.concatenate(
    [
        np.asarray(train.y, dtype=np.float32),
        np.asarray(valid.y, dtype=np.float32),
    ]
)
combined_user_id = np.concatenate(
    [
        np.asarray(train.user_id, dtype=np.int64),
        np.asarray(valid.user_id, dtype=np.int64),
    ]
)
combined_video_id = np.concatenate(
    [
        np.asarray(train.video_id, dtype=np.int64),
        np.asarray(valid.video_id, dtype=np.int64),
    ]
)
combined_date = np.concatenate(
    [
        np.asarray(train.date, dtype=np.int32),
        np.asarray(valid.date, dtype=np.int32),
    ]
)
combined_time_ms = np.concatenate(
    [
        np.asarray(train.time_ms, dtype=np.int64),
        np.asarray(valid.time_ms, dtype=np.int64),
    ]
)

class CombinedSplit:
    pass


combined = CombinedSplit()
combined.y = combined_y
combined.user_id = combined_user_id
combined.video_id = combined_video_id
combined.date = combined_date
combined.time_ms = combined_time_ms
combined.X = {
    name: np.concatenate(
        [
            np.asarray(train.X[name], dtype=np.int64),
            np.asarray(valid.X[name], dtype=np.int64),
        ]
    )
    for name in set(AFFINITY_FIELDS + ["user_id"])
}
combined.num = {
    name: np.concatenate(
        [
            np.asarray(train.num[name], dtype=np.float32),
            np.asarray(valid.num[name], dtype=np.float32),
        ]
    )
    for name in [
        "duration_ms",
        "user_fans_user_num",
        "user_follow_user_num",
        "user_friend_user_num",
        "user_register_days",
    ]
}

del X_train, X_valid, booster
gc.collect()

X_combined = build_affinity_features(
    combined,
    target=None,
    leave_one_out=True,
)
X_test = build_affinity_features(
    combined,
    target=test,
    leave_one_out=False,
)

combined_booster = train_booster(
    X_combined,
    combined_y,
    NUM_BOOST_ROUND,
)
personal_test_raw = combined_booster.predict(
    X_test,
    num_iteration=NUM_BOOST_ROUND,
)
personal_test_z = zscore(safe_logit(personal_test_raw))

direct_test = np.mean(
    X_test[:, affinity_column_indices],
    axis=1,
).astype(np.float64)
direct_test_z = zscore(direct_test)

if best_kind == "incumbent" and incumbent_test is not None:
    test_scores = zscore(incumbent_test)
elif (
    best_kind == "lgbm"
    and incumbent_test is not None
    and len(incumbent_test) == len(personal_test_z)
    and best_alpha < 1.0
):
    test_scores = (
        (1.0 - best_alpha) * zscore(incumbent_test)
        + best_alpha * personal_test_z
    )
elif (
    best_kind == "direct"
    and incumbent_test is not None
    and len(incumbent_test) == len(direct_test_z)
    and best_alpha < 1.0
):
    test_scores = (
        (1.0 - best_alpha) * zscore(incumbent_test)
        + best_alpha * direct_test_z
    )
elif best_kind == "direct":
    test_scores = direct_test_z
else:
    test_scores = personal_test_z

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
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