import os
import time
import json
import random
import gc

import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2026

random.seed(SEED)
np.random.seed(SEED)

try:
    lgb.register_logger(None)
except Exception:
    pass


CATEGORICAL_FIELDS = [
    "author_id",
    "duration_bucket",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "hour",
    "is_live_streamer",
    "is_lowactive_period",
    "is_video_author",
    "music_type",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat10",
    "onehot_feat11",
    "onehot_feat12",
    "onehot_feat13",
    "onehot_feat14",
    "onehot_feat15",
    "onehot_feat16",
    "onehot_feat17",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat4",
    "onehot_feat5",
    "onehot_feat6",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat9",
    "register_days_bucket",
    "register_days_range",
    "tab",
    "tag",
    "upload_type",
    "user_active_degree",
    "user_id",
    "video_id",
    "video_type",
]

NUMERIC_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def discover_history_specs(train_split):
    specs = []
    for key in ("video_id", "author_id"):
        hist = historical_features("train", key=key)
        for name in sorted(hist):
            arr = np.asarray(hist[name])
            if arr.ndim == 1 and len(arr) == len(train_split.user_id):
                specs.append((key, name))
    return specs


def numeric_transform(arr):
    x = np.asarray(arr, dtype=np.float32).copy()
    finite = np.isfinite(x)
    if finite.any():
        observed = x[finite]
        if float(np.min(observed)) >= 0.0:
            x[finite] = np.log1p(observed)
    return x


def history_transform(name, arr):
    x = np.asarray(arr, dtype=np.float32).copy()
    finite = np.isfinite(x)
    if not finite.any():
        return x

    observed = x[finite]
    low = name.lower()

    looks_like_count = (
        "count" in low
        or "cnt" in low
        or "impression" in low
        or "exposure" in low
        or "frequency" in low
        or "freq" in low
        or "number" in low
        or low.endswith("_n")
    )
    bounded_rate = (
        float(np.min(observed)) >= 0.0
        and float(np.max(observed)) <= 1.0001
    )

    if looks_like_count and not bounded_rate and float(np.min(observed)) >= 0.0:
        x[finite] = np.log1p(observed)
    return x


def build_matrix(split_name, split, history_specs):
    n = len(split.user_id)
    n_cols = len(CATEGORICAL_FIELDS) + len(NUMERIC_FIELDS) + len(history_specs)
    matrix = np.empty((n, n_cols), dtype=np.float32)
    feature_names = []
    col = 0

    for field in CATEGORICAL_FIELDS:
        matrix[:, col] = np.asarray(split.X[field], dtype=np.float32)
        feature_names.append("cat_" + field)
        col += 1

    for field in NUMERIC_FIELDS:
        matrix[:, col] = numeric_transform(split.num[field])
        feature_names.append("num_" + field)
        col += 1

    history_cache = {}
    for key in ("video_id", "author_id"):
        if any(spec_key == key for spec_key, _ in history_specs):
            history_cache[key] = historical_features(split_name, key=key)

    for key, name in history_specs:
        if name not in history_cache[key]:
            raise KeyError("Missing historical feature %s:%s" % (key, name))
        arr = np.asarray(history_cache[key][name])
        if arr.ndim != 1 or len(arr) != n:
            raise ValueError("Invalid historical feature %s:%s" % (key, name))
        matrix[:, col] = history_transform(name, arr)
        feature_names.append("hist_" + key + "_" + name)
        col += 1

    return matrix, feature_names


def stable_group_order(user_ids):
    users = np.asarray(user_ids)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    if len(sorted_users) == 0:
        return order, np.empty(0, dtype=np.int32)

    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], len(sorted_users)]
    groups = (ends - starts).astype(np.int32)
    return order, groups


def restore_original_order(sorted_predictions, order):
    result = np.empty(len(order), dtype=np.float64)
    result[order] = np.asarray(sorted_predictions, dtype=np.float64)
    return result


def standardized(scores, reference_mean, reference_std):
    return (np.asarray(scores, dtype=np.float64) - reference_mean) / reference_std


train = load("train")
valid = load("valid")

history_specs = discover_history_specs(train)

x_train, feature_names = build_matrix("train", train, history_specs)
x_valid, valid_feature_names = build_matrix("valid", valid, history_specs)

if feature_names != valid_feature_names:
    raise RuntimeError("Train and validation feature layouts differ")

train_order, train_groups = stable_group_order(train.user_id)
valid_order, valid_groups = stable_group_order(valid.user_id)

x_train = np.ascontiguousarray(x_train[train_order])
x_valid = np.ascontiguousarray(x_valid[valid_order])

y_train = np.ascontiguousarray(
    np.asarray(train.y, dtype=np.int8)[train_order]
)
y_valid_sorted = np.ascontiguousarray(
    np.asarray(valid.y, dtype=np.int8)[valid_order]
)

categorical_indices = list(range(len(CATEGORICAL_FIELDS)))

dtrain = lgb.Dataset(
    x_train,
    label=y_train,
    group=train_groups,
    feature_name=feature_names,
    categorical_feature=categorical_indices,
    free_raw_data=True,
)

dvalid = lgb.Dataset(
    x_valid,
    label=y_valid_sorted,
    group=valid_groups,
    feature_name=feature_names,
    categorical_feature=categorical_indices,
    reference=dtrain,
    free_raw_data=True,
)

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "lambdarank_truncation_level": 15,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 250,
    "min_sum_hessian_in_leaf": 1e-3,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "min_data_per_group": 100,
    "max_cat_threshold": 32,
    "cat_l2": 10.0,
    "cat_smooth": 20.0,
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
    num_boost_round=450,
    valid_sets=[dvalid],
    valid_names=["valid"],
    callbacks=[
        lgb.early_stopping(45, first_metric_only=True, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

tree_valid_sorted = model.predict(
    x_valid,
    num_iteration=model.best_iteration,
)
tree_valid = restore_original_order(tree_valid_sorted, valid_order)

valid_users = np.asarray(valid.user_id)
valid_labels = np.asarray(valid.y)

artifacts = os.environ.get("RUN_ARTIFACTS")
if not artifacts:
    raise RuntimeError("RUN_ARTIFACTS is required to access the trusted incumbent")

inc_valid_path = os.path.join(artifacts, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(artifacts, "incumbent_test_scores.npy")

if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are missing")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if len(inc_valid) != len(valid_users):
    raise ValueError("Incumbent validation prediction length mismatch")

tree_mean = float(np.mean(tree_valid))
tree_std = float(np.std(tree_valid))
inc_mean = float(np.mean(inc_valid))
inc_std = float(np.std(inc_valid))

tree_std = max(tree_std, 1e-8)
inc_std = max(inc_std, 1e-8)

tree_valid_z = standardized(tree_valid, tree_mean, tree_std)
inc_valid_z = standardized(inc_valid, inc_mean, inc_std)

alphas = np.linspace(0.0, 1.0, 11)
candidate_metrics = {}
candidate_scores = {}

best_primary = -np.inf
best_alpha = None
best_metrics = None
best_valid_scores = None

for alpha in alphas:
    scores = (
        float(alpha) * tree_valid_z
        + (1.0 - float(alpha)) * inc_valid_z
    )
    metrics = evaluate(valid_users, valid_labels, scores)
    name = "tree_weight_%.1f" % float(alpha)
    candidate_metrics[name] = float(metrics["primary"])
    candidate_scores[name] = {
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
    }

    if float(metrics["primary"]) > best_primary:
        best_primary = float(metrics["primary"])
        best_alpha = float(alpha)
        best_metrics = metrics
        best_valid_scores = np.asarray(scores, dtype=np.float64).copy()

if best_valid_scores is None:
    raise RuntimeError("No blend candidate was evaluated")

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )

# Release training-side arrays before constructing the test feature matrix.
del x_train, x_valid, y_train, y_valid_sorted, dtrain, dvalid
gc.collect()

test = load("test")
x_test, test_feature_names = build_matrix("test", test, history_specs)

if feature_names != test_feature_names:
    raise RuntimeError("Train and test feature layouts differ")

tree_test = np.asarray(
    model.predict(x_test, num_iteration=model.best_iteration),
    dtype=np.float64,
)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if len(inc_test) != len(test.user_id):
    raise ValueError("Incumbent test prediction length mismatch")

# Validation-derived scales are applied unchanged to test. Centering constants
# do not affect within-user ranks, but using the same scales preserves the
# validation-selected relative contribution of the two models.
tree_test_z = standardized(tree_test, tree_mean, tree_std)
inc_test_z = standardized(inc_test, inc_mean, inc_std)
test_scores = (
    best_alpha * tree_test_z
    + (1.0 - best_alpha) * inc_test_z
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

try:
    model.save_model(
        os.path.join(artifacts, "lambdarank_all_features_history.txt"),
        num_iteration=model.best_iteration,
    )
    with open(
        os.path.join(artifacts, "lambdarank_all_features_history.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "seed": SEED,
                "best_iteration": int(model.best_iteration),
                "best_alpha": float(best_alpha),
                "tree_mean_valid": tree_mean,
                "tree_std_valid": tree_std,
                "incumbent_mean_valid": inc_mean,
                "incumbent_std_valid": inc_std,
                "categorical_fields": CATEGORICAL_FIELDS,
                "numeric_fields": NUMERIC_FIELDS,
                "history_specs": history_specs,
                "candidate_metrics": candidate_scores,
            },
            f,
        )
except Exception as exc:
    print("FINDINGS " + json.dumps({"artifact_save_warning": str(exc)}))

elapsed = time.time() - START

print(
    "FINDINGS "
    + json.dumps(
        {
            "model": "lightgbm_lambdarank",
            "best_iteration": int(model.best_iteration),
            "n_features": int(len(feature_names)),
            "n_history_features": int(len(history_specs)),
            "n_train_groups": int(len(train_groups)),
            "n_valid_groups": int(len(valid_groups)),
            "tree_only_primary": float(candidate_metrics["tree_weight_1.0"]),
            "incumbent_primary": float(candidate_metrics["tree_weight_0.0"]),
            "selected_tree_weight": float(best_alpha),
        }
    )
)
print("CANDIDATES " + json.dumps(candidate_metrics))
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