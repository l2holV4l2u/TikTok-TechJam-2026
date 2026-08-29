import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 2026
NUM_THREADS = max(1, min(16, os.cpu_count() or 1))

CAT_FIELDS = [
    "user_id", "video_id", "tab", "hour", "user_active_degree",
    "is_lowactive_period", "is_live_streamer", "is_video_author",
    "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
    "onehot_feat0", "onehot_feat1", "onehot_feat2", "onehot_feat3",
    "onehot_feat4", "onehot_feat5", "onehot_feat6", "onehot_feat7",
    "onehot_feat8", "onehot_feat9", "onehot_feat10", "onehot_feat11",
    "onehot_feat12", "onehot_feat13", "onehot_feat14", "onehot_feat15",
    "onehot_feat16", "onehot_feat17", "author_id", "video_type",
    "upload_type", "music_type", "tag", "duration_bucket",
    "register_days_bucket",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]
ENTITY_FIELDS = ["video_id", "author_id", "tag"]
PAIR_FIELDS = [
    ("tag", 6.0),
    ("duration_bucket", 6.0),
    ("upload_type", 6.0),
    ("author_id", 10.0),
    ("video_id", 12.0),
]


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    ex = np.exp(x[~positive])
    out[~positive] = ex / (1.0 + ex)
    return out


def global_rank(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    result = np.empty(len(x), dtype=np.float64)
    result[order] = (np.arange(len(x), dtype=np.float64) + 0.5) / len(x)
    return result


def make_base_matrix(split):
    n = len(split.user_id)
    result = np.empty((n, len(CAT_FIELDS) + len(NUM_FIELDS)), dtype=np.float32)

    for j, field in enumerate(CAT_FIELDS):
        result[:, j] = np.asarray(split.X[field], dtype=np.float32)

    offset = len(CAT_FIELDS)
    for j, field in enumerate(NUM_FIELDS):
        values = np.asarray(split.num[field], dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = np.sign(values) * np.log1p(np.abs(values))
        result[:, offset + j] = values

    return result


def entity_history(fit_ids, pred_ids, y, smoothing=20.0):
    fit_ids = np.asarray(fit_ids, dtype=np.int64)
    pred_ids = np.asarray(pred_ids, dtype=np.int64)
    y64 = np.asarray(y, dtype=np.float64)
    prior = float(y64.mean())

    size = int(max(fit_ids.max(initial=0), pred_ids.max(initial=0))) + 1
    counts = np.bincount(fit_ids, minlength=size).astype(np.float64)
    sums = np.bincount(fit_ids, weights=y64, minlength=size).astype(np.float64)

    fit_count = np.maximum(counts[fit_ids] - 1.0, 0.0)
    fit_sum = sums[fit_ids] - y64
    fit_rate = (fit_sum + smoothing * prior) / (fit_count + smoothing)

    pred_count = counts[pred_ids]
    pred_rate = (sums[pred_ids] + smoothing * prior) / (
        pred_count + smoothing
    )

    fit_features = np.column_stack([
        np.log1p(fit_count),
        fit_rate,
    ]).astype(np.float32)
    pred_features = np.column_stack([
        np.log1p(pred_count),
        pred_rate,
    ]).astype(np.float32)
    return fit_features, pred_features


def pair_history(fit_users, fit_values, pred_users, pred_values, y, smoothing):
    fit_users = np.asarray(fit_users, dtype=np.int64)
    fit_values = np.asarray(fit_values, dtype=np.int64)
    pred_users = np.asarray(pred_users, dtype=np.int64)
    pred_values = np.asarray(pred_values, dtype=np.int64)
    y64 = np.asarray(y, dtype=np.float64)
    prior = float(y64.mean())

    value_base = int(max(fit_values.max(initial=0),
                         pred_values.max(initial=0))) + 1
    fit_keys = fit_users * value_base + fit_values
    pred_keys = pred_users * value_base + pred_values

    unique_keys, inverse, counts = np.unique(
        fit_keys, return_inverse=True, return_counts=True
    )
    sums = np.bincount(inverse, weights=y64, minlength=len(unique_keys))

    fit_count = counts[inverse].astype(np.float64) - 1.0
    fit_sum = sums[inverse] - y64
    fit_rate = (fit_sum + smoothing * prior) / (
        fit_count + smoothing
    )

    positions = np.searchsorted(unique_keys, pred_keys)
    present = positions < len(unique_keys)
    safe_positions = np.minimum(positions, len(unique_keys) - 1)
    present &= unique_keys[safe_positions] == pred_keys

    pred_count = np.zeros(len(pred_keys), dtype=np.float64)
    pred_sum = np.zeros(len(pred_keys), dtype=np.float64)
    pred_count[present] = counts[safe_positions[present]]
    pred_sum[present] = sums[safe_positions[present]]
    pred_rate = (pred_sum + smoothing * prior) / (
        pred_count + smoothing
    )

    fit_features = np.column_stack([
        np.log1p(np.maximum(fit_count, 0.0)),
        fit_rate,
    ]).astype(np.float32)
    pred_features = np.column_stack([
        np.log1p(pred_count),
        pred_rate,
    ]).astype(np.float32)
    return fit_features, pred_features


def make_fit_and_prediction_matrices(fit_split, y_fit, pred_split):
    fit_parts = [make_base_matrix(fit_split)]
    pred_parts = [make_base_matrix(pred_split)]

    for field in ENTITY_FIELDS:
        fit_extra, pred_extra = entity_history(
            fit_split.X[field],
            pred_split.X[field],
            y_fit,
            smoothing=20.0,
        )
        fit_parts.append(fit_extra)
        pred_parts.append(pred_extra)

    for field, smoothing in PAIR_FIELDS:
        fit_extra, pred_extra = pair_history(
            fit_split.user_id,
            fit_split.X[field],
            pred_split.user_id,
            pred_split.X[field],
            y_fit,
            smoothing,
        )
        fit_parts.append(fit_extra)
        pred_parts.append(pred_extra)

    fit_matrix = np.ascontiguousarray(
        np.concatenate(fit_parts, axis=1), dtype=np.float32
    )
    pred_matrix = np.ascontiguousarray(
        np.concatenate(pred_parts, axis=1), dtype=np.float32
    )
    return fit_matrix, pred_matrix


class CombinedSplit:
    def __init__(self, first, second):
        self.user_id = np.concatenate([
            np.asarray(first.user_id, dtype=np.int64),
            np.asarray(second.user_id, dtype=np.int64),
        ])
        self.X = {}
        for field in CAT_FIELDS:
            self.X[field] = np.concatenate([
                np.asarray(first.X[field], dtype=np.int64),
                np.asarray(second.X[field], dtype=np.int64),
            ])
        self.num = {}
        for field in NUM_FIELDS:
            self.num[field] = np.concatenate([
                np.asarray(first.num[field], dtype=np.float32),
                np.asarray(second.num[field], dtype=np.float32),
            ])


params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.04,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 180,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "max_cat_to_onehot": 16,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "data_random_seed": SEED,
    "num_threads": NUM_THREADS,
    "verbose": -1,
}


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

X_train, X_valid = make_fit_and_prediction_matrices(
    train, y_train, valid
)

categorical_indices = list(range(len(CAT_FIELDS)))
train_dataset = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
valid_dataset = lgb.Dataset(
    X_valid,
    label=y_valid,
    categorical_feature=categorical_indices,
    reference=train_dataset,
    free_raw_data=False,
)

valid_model = lgb.train(
    params,
    train_dataset,
    num_boost_round=700,
    valid_sets=[valid_dataset],
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)
best_iteration = int(valid_model.best_iteration)
lgb_valid_raw = valid_model.predict(
    X_valid, num_iteration=best_iteration, raw_score=True
)
lgb_valid_prob = sigmoid(lgb_valid_raw)

artifacts = os.environ["RUN_ARTIFACTS"]
inc_valid = np.asarray(
    np.load(os.path.join(artifacts, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test_path = os.path.join(artifacts, "incumbent_test_scores.npy")

if len(inc_valid) != len(y_valid):
    raise RuntimeError("Incumbent validation prediction length mismatch")

inc_valid_prob = sigmoid(inc_valid)
inc_valid_rank = global_rank(inc_valid)
lgb_valid_rank = global_rank(lgb_valid_raw)

candidates = {}
candidate_specs = []
best_metrics = None
best_scores = None
best_spec = None

def consider(name, scores, spec):
    global best_metrics, best_scores, best_spec
    result = evaluate(valid.user_id, y_valid, scores)
    candidates[name] = float(result["primary"])
    candidate_specs.append((name, spec))
    if best_metrics is None or result["primary"] > best_metrics["primary"]:
        best_metrics = result
        best_scores = np.asarray(scores, dtype=np.float64).copy()
        best_spec = spec


consider("incumbent", inc_valid, ("incumbent", 1.0))
consider("lightgbm", lgb_valid_raw, ("lightgbm", 0.0))

for alpha in [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]:
    raw_blend = alpha * inc_valid + (1.0 - alpha) * lgb_valid_raw
    consider(
        "raw_inc_{:.2f}".format(alpha),
        raw_blend,
        ("raw", alpha),
    )

    probability_blend = (
        alpha * inc_valid_prob + (1.0 - alpha) * lgb_valid_prob
    )
    consider(
        "prob_inc_{:.2f}".format(alpha),
        probability_blend,
        ("prob", alpha),
    )

    rank_blend = (
        alpha * inc_valid_rank + (1.0 - alpha) * lgb_valid_rank
    )
    consider(
        "rank_inc_{:.2f}".format(alpha),
        rank_blend,
        ("rank", alpha),
    )

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )

print("CANDIDATES " + json.dumps(candidates, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps({
        "selected": best_spec,
        "best_iteration": best_iteration,
        "n_features": int(X_train.shape[1]),
        "lgb_primary": candidates["lightgbm"],
        "incumbent_primary": candidates["incumbent"],
    }, sort_keys=True)
)

# Release validation model and matrices before the allowed train+validation refit.
del valid_model, train_dataset, valid_dataset
del X_train, X_valid
gc.collect()

combined = CombinedSplit(train, valid)
y_combined = np.ascontiguousarray(
    np.concatenate([y_train, y_valid.astype(np.float32)]),
    dtype=np.float32,
)

test = load("test")
X_combined, X_test = make_fit_and_prediction_matrices(
    combined, y_combined, test
)

combined_dataset = lgb.Dataset(
    X_combined,
    label=y_combined,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
test_model = lgb.train(
    params,
    combined_dataset,
    num_boost_round=best_iteration,
    callbacks=[lgb.log_evaluation(period=0)],
)

lgb_test_raw = test_model.predict(
    X_test, num_iteration=best_iteration, raw_score=True
)

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test) != len(test.user_id):
    raise RuntimeError("Incumbent test prediction length mismatch")

mode, alpha = best_spec
if mode == "incumbent":
    test_scores = inc_test
elif mode == "lightgbm":
    test_scores = lgb_test_raw
elif mode == "raw":
    test_scores = alpha * inc_test + (1.0 - alpha) * lgb_test_raw
elif mode == "prob":
    test_scores = (
        alpha * sigmoid(inc_test)
        + (1.0 - alpha) * sigmoid(lgb_test_raw)
    )
elif mode == "rank":
    test_scores = (
        alpha * global_rank(inc_test)
        + (1.0 - alpha) * global_rank(lgb_test_raw)
    )
else:
    raise RuntimeError("Unknown selected blend mode")

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START_TIME
print(
    "METRICS "
    + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)