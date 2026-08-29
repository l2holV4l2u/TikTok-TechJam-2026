import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

_start_time = time.time()

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


SEED = 2026
NUM_BOOST_ROUND = 650
EARLY_STOPPING_ROUNDS = 50

CAT_FIELDS = [
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

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def get_history_names():
    probe = historical_features("train", key="video_id")
    video_names = sorted(probe.keys())
    del probe
    probe = historical_features("train", key="author_id")
    author_names = sorted(probe.keys())
    del probe
    return video_names, author_names


VIDEO_HISTORY_NAMES, AUTHOR_HISTORY_NAMES = get_history_names()


def make_matrix(split_name, split):
    columns = []

    for name in CAT_FIELDS:
        columns.append(
            np.asarray(split.X[name], dtype=np.float32).reshape(-1, 1)
        )

    for name in NUM_FIELDS:
        raw = np.asarray(split.num[name], dtype=np.float32)
        missing = ~np.isfinite(raw)
        safe = np.where(missing, 0.0, np.maximum(raw, 0.0))
        transformed = np.log1p(safe).astype(np.float32, copy=False)
        columns.append(transformed.reshape(-1, 1))
        columns.append(missing.astype(np.float32).reshape(-1, 1))

    video_history = historical_features(split_name, key="video_id")
    for name in VIDEO_HISTORY_NAMES:
        value = np.asarray(video_history[name], dtype=np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(value.reshape(-1, 1))
    del video_history

    author_history = historical_features(split_name, key="author_id")
    for name in AUTHOR_HISTORY_NAMES:
        value = np.asarray(author_history[name], dtype=np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(value.reshape(-1, 1))
    del author_history

    matrix = np.ascontiguousarray(np.concatenate(columns, axis=1))
    del columns
    return matrix


def stable_zscore(reference, values):
    mean = float(np.mean(reference))
    std = float(np.std(reference))
    std = max(std, 1e-8)
    return (np.asarray(values, dtype=np.float64) - mean) / std


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row_id = np.arange(n, dtype=np.int64)

    order = np.lexsort((row_id, scores, user_ids))
    sorted_users = user_ids[order]

    group_start_mask = np.empty(n, dtype=bool)
    group_start_mask[0] = True
    group_start_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(group_start_mask)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    positions_sorted = (
        np.arange(n, dtype=np.float64) - np.repeat(starts, lengths)
    )
    denominators = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    ranks_sorted = positions_sorted / denominators

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


train = load("train")
valid = load("valid")

X_train = make_matrix("train", train)
X_valid = make_matrix("valid", valid)

categorical_indices = list(range(len(CAT_FIELDS)))

# Mild recency weighting adapts the pointwise learner to the immediately
# following validation period without discarding the earlier training days.
train_day = np.asarray(train.date, dtype=np.int32)
days_old = np.maximum(int(train_day.max()) - train_day, 0).astype(np.float32)
train_weight = np.exp(-0.035 * days_old).astype(np.float32)

dtrain = lgb.Dataset(
    X_train,
    label=np.asarray(train.y, dtype=np.float32),
    weight=train_weight,
    categorical_feature=categorical_indices,
    free_raw_data=True,
)
dvalid = lgb.Dataset(
    X_valid,
    label=np.asarray(valid.y, dtype=np.float32),
    categorical_feature=categorical_indices,
    reference=dtrain,
    free_raw_data=True,
)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.045,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 350,
    "min_sum_hessian_in_leaf": 1e-3,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.82,
    "bagging_freq": 1,
    "lambda_l1": 0.08,
    "lambda_l2": 1.2,
    "max_bin": 127,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "max_cat_to_onehot": 16,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "data_random_seed": SEED + 3,
    "num_threads": min(16, max(1, os.cpu_count() or 1)),
    "verbose": -1,
}

model = lgb.train(
    params,
    dtrain,
    num_boost_round=NUM_BOOST_ROUND,
    valid_sets=[dvalid],
    valid_names=["valid"],
    callbacks=[
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

best_iteration = int(model.best_iteration or NUM_BOOST_ROUND)
lgb_valid_probability = model.predict(
    X_valid, num_iteration=best_iteration
)
lgb_valid_probability = np.clip(
    np.asarray(lgb_valid_probability, dtype=np.float64),
    1e-7,
    1.0 - 1e-7,
)
lgb_valid_logit = np.log(
    lgb_valid_probability / (1.0 - lgb_valid_probability)
)

candidate_metrics = {}
selection = []

lgb_metrics = evaluate(valid.user_id, valid.y, lgb_valid_logit)
candidate_metrics["lgb_only"] = float(lgb_metrics["primary"])
selection.append(
    (
        float(lgb_metrics["primary"]),
        "lgb_only",
        1.0,
        lgb_valid_logit,
        lgb_metrics,
    )
)

artifacts = os.environ.get("RUN_ARTIFACTS", "")
incumbent_valid_path = os.path.join(
    artifacts, "incumbent_valid_scores.npy"
)
incumbent_test_path = os.path.join(
    artifacts, "incumbent_test_scores.npy"
)

have_incumbent = (
    os.path.isfile(incumbent_valid_path)
    and os.path.isfile(incumbent_test_path)
)

if have_incumbent:
    incumbent_valid = np.asarray(
        np.load(incumbent_valid_path), dtype=np.float64
    )
    if len(incumbent_valid) != len(valid.y):
        raise ValueError("Incumbent validation prediction length mismatch")

    incumbent_metrics = evaluate(
        valid.user_id, valid.y, incumbent_valid
    )
    candidate_metrics["incumbent"] = float(
        incumbent_metrics["primary"]
    )
    selection.append(
        (
            float(incumbent_metrics["primary"]),
            "incumbent",
            0.0,
            incumbent_valid,
            incumbent_metrics,
        )
    )

    lgb_z = stable_zscore(lgb_valid_logit, lgb_valid_logit)
    incumbent_z = stable_zscore(incumbent_valid, incumbent_valid)

    lgb_rank = within_user_rank(valid.user_id, lgb_valid_logit)
    incumbent_rank = within_user_rank(
        valid.user_id, incumbent_valid
    )

    blend_alphas = np.arange(0.10, 0.91, 0.05)
    for alpha in blend_alphas:
        alpha = float(alpha)

        z_blend = alpha * lgb_z + (1.0 - alpha) * incumbent_z
        z_metrics = evaluate(valid.user_id, valid.y, z_blend)
        z_name = "zblend_%.2f" % alpha
        candidate_metrics[z_name] = float(z_metrics["primary"])
        selection.append(
            (
                float(z_metrics["primary"]),
                "zblend",
                alpha,
                z_blend,
                z_metrics,
            )
        )

        rank_blend = (
            alpha * lgb_rank + (1.0 - alpha) * incumbent_rank
        )
        rank_metrics = evaluate(
            valid.user_id, valid.y, rank_blend
        )
        rank_name = "rankblend_%.2f" % alpha
        candidate_metrics[rank_name] = float(
            rank_metrics["primary"]
        )
        selection.append(
            (
                float(rank_metrics["primary"]),
                "rankblend",
                alpha,
                rank_blend,
                rank_metrics,
            )
        )

selection.sort(key=lambda item: item[0], reverse=True)
best_primary, best_kind, best_alpha, valid_scores, best_metrics = selection[0]
valid_scores = np.asarray(valid_scores, dtype=np.float64).copy()

print(
    "FINDINGS best_iteration=%d features=%d histories=%d selected=%s alpha=%.2f"
    % (
        best_iteration,
        X_train.shape[1],
        len(VIDEO_HISTORY_NAMES) + len(AUTHOR_HISTORY_NAMES),
        best_kind,
        best_alpha,
    )
)

top_candidates = sorted(
    candidate_metrics.items(), key=lambda item: item[1], reverse=True
)[:10]
print(
    "CANDIDATES "
    + json.dumps(
        {name: score for name, score in top_candidates},
        separators=(", ", ": "),
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        valid_scores.astype(np.float64, copy=False),
    )

# Retain the fitted booster but release the large training matrix before
# constructing test features.
del X_train, X_valid, dtrain, dvalid
gc.collect()

test = load("test")
X_test = make_matrix("test", test)

lgb_test_probability = model.predict(
    X_test, num_iteration=best_iteration
)
lgb_test_probability = np.clip(
    np.asarray(lgb_test_probability, dtype=np.float64),
    1e-7,
    1.0 - 1e-7,
)
lgb_test_logit = np.log(
    lgb_test_probability / (1.0 - lgb_test_probability)
)

if best_kind == "lgb_only":
    test_scores = lgb_test_logit
elif best_kind == "incumbent":
    test_scores = np.asarray(
        np.load(incumbent_test_path), dtype=np.float64
    )
else:
    incumbent_test = np.asarray(
        np.load(incumbent_test_path), dtype=np.float64
    )
    if len(incumbent_test) != len(test.user_id):
        raise ValueError("Incumbent test prediction length mismatch")

    if best_kind == "zblend":
        lgb_test_z = stable_zscore(
            lgb_valid_logit, lgb_test_logit
        )
        incumbent_test_z = stable_zscore(
            incumbent_valid, incumbent_test
        )
        test_scores = (
            best_alpha * lgb_test_z
            + (1.0 - best_alpha) * incumbent_test_z
        )
    elif best_kind == "rankblend":
        lgb_test_rank = within_user_rank(
            test.user_id, lgb_test_logit
        )
        incumbent_test_rank = within_user_rank(
            test.user_id, incumbent_test
        )
        test_scores = (
            best_alpha * lgb_test_rank
            + (1.0 - best_alpha) * incumbent_test_rank
        )
    else:
        raise RuntimeError("Unknown selected prediction kind")

test_scores = np.asarray(test_scores, dtype=np.float64)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        test_scores,
    )

elapsed = time.time() - _start_time
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result, separators=(", ", ": ")))