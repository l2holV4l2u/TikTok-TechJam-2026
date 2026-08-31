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
SEED = 2024
NTHREAD = min(8, max(1, os.cpu_count() or 1))

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "duration_bucket",
    "hour",
    "user_active_degree",
    "video_type",
    "music_type",
    "is_video_author",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

HISTORY_KEYS = [
    "train_count_log1p",
    "long_view_rate",
    "is_click_rate",
    "play_time_ms_logmean",
    "is_like_rate",
    "is_profile_enter_rate",
]


def get_histories(split_name):
    return {
        "video_id": historical_features(split_name, key="video_id"),
        "author_id": historical_features(split_name, key="author_id"),
    }


def make_features(split, histories):
    columns = []

    for name in CAT_FIELDS:
        columns.append(np.asarray(split.X[name], dtype=np.float32))

    for name in NUM_FIELDS:
        a = np.asarray(split.num[name], dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(a, 0.0)).astype(np.float32))

    for entity in ("video_id", "author_id"):
        h = histories[entity]
        for suffix in HISTORY_KEYS:
            name = entity + "_" + suffix
            a = np.asarray(h[name], dtype=np.float32)
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            columns.append(a)

    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def empirical_history_score(histories):
    vh = histories["video_id"]
    ah = histories["author_id"]

    v_lv = np.nan_to_num(
        np.asarray(vh["video_id_long_view_rate"], dtype=np.float64),
        nan=0.3366,
    )
    a_lv = np.nan_to_num(
        np.asarray(ah["author_id_long_view_rate"], dtype=np.float64),
        nan=0.3366,
    )
    v_click = np.nan_to_num(
        np.asarray(vh["video_id_is_click_rate"], dtype=np.float64),
        nan=0.5,
    )
    a_click = np.nan_to_num(
        np.asarray(ah["author_id_is_click_rate"], dtype=np.float64),
        nan=0.5,
    )
    v_count = np.nan_to_num(
        np.asarray(vh["video_id_train_count_log1p"], dtype=np.float64),
        nan=0.0,
    )
    a_count = np.nan_to_num(
        np.asarray(ah["author_id_train_count_log1p"], dtype=np.float64),
        nan=0.0,
    )

    # More frequent entities receive slightly more of their entity-specific
    # estimate, while sparse entities regress toward the other entity.
    v_reliability = np.clip(v_count / 8.0, 0.15, 1.0)
    a_reliability = np.clip(a_count / 8.0, 0.15, 1.0)

    score = (
        0.48 * v_reliability * v_lv
        + 0.27 * a_reliability * a_lv
        + 0.15 * v_click
        + 0.10 * a_click
    )
    score /= (
        0.48 * v_reliability
        + 0.27 * a_reliability
        + 0.25
    )
    score = np.clip(score, 1e-5, 1.0 - 1e-5)
    return np.log(score / (1.0 - score))


def group_order_and_sizes(user_ids):
    user_ids = np.asarray(user_ids)
    order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[order]
    if len(sorted_users) == 0:
        return order, np.empty(0, dtype=np.int32)

    boundaries = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1], True]
    )
    sizes = np.diff(boundaries).astype(np.int32)
    return order, sizes


def predict_ranker_original_order(model, X, user_ids):
    order, _ = group_order_and_sizes(user_ids)
    sorted_pred = model.predict(
        X[order],
        num_iteration=model.best_iteration,
        raw_score=True,
    )
    pred = np.empty(len(order), dtype=np.float64)
    pred[order] = sorted_pred
    return pred


def center_by_user(scores, user_ids):
    scores = np.asarray(scores, dtype=np.float64)
    _, inv = np.unique(np.asarray(user_ids), return_inverse=True)
    sums = np.bincount(inv, weights=scores)
    counts = np.bincount(inv)
    means = sums / np.maximum(counts, 1)
    return scores - means[inv]


def normalized_component(scores, user_ids, scale=None):
    centered = center_by_user(scores, user_ids)
    if scale is None:
        scale = float(np.std(centered))
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
    return centered / scale, scale


def metric_primary(user_ids, labels, scores):
    return float(evaluate(user_ids, labels, scores)["primary"])


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

hist_train = get_histories("train")
hist_valid = get_histories("valid")

X_train = make_features(train, hist_train)
X_valid = make_features(valid, hist_valid)

categorical_indices = list(range(len(CAT_FIELDS)))

# Family 1: pointwise boosted trees. Recent training days are upweighted to
# reduce mismatch with the later validation/test exposure distributions.
last_train_date = int(np.max(np.asarray(train.date)))
days_old = last_train_date - np.asarray(train.date, dtype=np.int64)
recency_weight = np.exp2(-days_old / 5.0).astype(np.float32)
recency_weight /= np.mean(recency_weight)

binary_train = lgb.Dataset(
    X_train,
    label=y_train,
    weight=recency_weight,
    categorical_feature=categorical_indices,
    free_raw_data=True,
)
binary_valid = lgb.Dataset(
    X_valid,
    label=y_valid,
    categorical_feature=categorical_indices,
    reference=binary_train,
    free_raw_data=True,
)

binary_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 400,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "max_cat_to_onehot": 16,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "num_threads": NTHREAD,
    "verbose": -1,
}

binary_model = lgb.train(
    binary_params,
    binary_train,
    num_boost_round=420,
    valid_sets=[binary_valid],
    callbacks=[lgb.early_stopping(35, verbose=False)],
)

binary_valid_scores = binary_model.predict(
    X_valid,
    num_iteration=binary_model.best_iteration,
    raw_score=True,
).astype(np.float64)

del binary_train, binary_valid
gc.collect()

# Family 2: LambdaRank. Sorting only changes training/evaluation storage order;
# predictions are restored to the original row order.
train_order, train_groups = group_order_and_sizes(train.user_id)
valid_order, valid_groups = group_order_and_sizes(valid.user_id)

rank_train = lgb.Dataset(
    X_train[train_order],
    label=y_train[train_order],
    group=train_groups,
    categorical_feature=categorical_indices,
    free_raw_data=True,
)
rank_valid = lgb.Dataset(
    X_valid[valid_order],
    label=y_valid[valid_order],
    group=valid_groups,
    categorical_feature=categorical_indices,
    reference=rank_train,
    free_raw_data=True,
)

rank_params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5],
    "label_gain": [0, 1],
    "lambdarank_truncation_level": 10,
    "learning_rate": 0.045,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 350,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "max_cat_to_onehot": 16,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "seed": SEED + 1,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 1,
    "num_threads": NTHREAD,
    "verbose": -1,
}

rank_model = lgb.train(
    rank_params,
    rank_train,
    num_boost_round=340,
    valid_sets=[rank_valid],
    callbacks=[lgb.early_stopping(35, verbose=False)],
)

rank_sorted_scores = rank_model.predict(
    X_valid[valid_order],
    num_iteration=rank_model.best_iteration,
    raw_score=True,
)
rank_valid_scores = np.empty(len(valid_order), dtype=np.float64)
rank_valid_scores[valid_order] = rank_sorted_scores

del rank_train, rank_valid, train_order, valid_order
gc.collect()

# Family 3: non-parametric empirical Bayes entity histories.
emp_valid_scores = empirical_history_score(hist_valid).astype(np.float64)

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required for incumbent blending")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)

families = {
    "recency_binary_gbdt": binary_valid_scores,
    "lambdarank_gbdt": rank_valid_scores,
    "empirical_bayes_history": emp_valid_scores,
}

candidate_scores = {}
family_blend_choice = {}

inc_norm_valid, inc_scale = normalized_component(
    inc_valid, valid.user_id
)

blend_grid = [0.20, 0.35, 0.50, 0.65, 0.80]

winner_name = None
winner_scores = None
winner_raw = None
winner_family = None
winner_alpha = None
winner_primary = -np.inf
winner_family_scale = None

for family_name, raw_scores in families.items():
    raw_primary = metric_primary(valid.user_id, y_valid, raw_scores)
    candidate_scores[family_name] = raw_primary

    if raw_primary > winner_primary:
        winner_primary = raw_primary
        winner_name = family_name
        winner_scores = raw_scores.copy()
        winner_raw = raw_scores.copy()
        winner_family = family_name
        winner_alpha = None
        winner_family_scale = None

    family_norm, family_scale = normalized_component(
        raw_scores, valid.user_id
    )

    best_blend_primary = -np.inf
    best_blend_alpha = None
    best_blend_scores = None

    for alpha in blend_grid:
        blended = alpha * family_norm + (1.0 - alpha) * inc_norm_valid
        p = metric_primary(valid.user_id, y_valid, blended)
        if p > best_blend_primary:
            best_blend_primary = p
            best_blend_alpha = alpha
            best_blend_scores = blended.copy()

    blend_name = family_name + "_incumbent_blend"
    candidate_scores[blend_name] = best_blend_primary
    family_blend_choice[family_name] = (
        best_blend_alpha,
        family_scale,
    )

    if best_blend_primary > winner_primary:
        winner_primary = best_blend_primary
        winner_name = blend_name
        winner_scores = best_blend_scores
        winner_raw = raw_scores.copy()
        winner_family = family_name
        winner_alpha = best_blend_alpha
        winner_family_scale = family_scale

metrics = evaluate(valid.user_id, y_valid, winner_scores)

print(
    "FINDINGS binary_best_iteration=%d rank_best_iteration=%d winner=%s"
    % (
        int(binary_model.best_iteration),
        int(rank_model.best_iteration),
        winner_name,
    )
)
print(
    "FINDINGS selected_blend_alpha=%s"
    % ("none" if winner_alpha is None else ("%.2f" % winner_alpha))
)
print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_scores.items()},
        sort_keys=True,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(winner_scores, dtype=np.float64),
    )
    if winner_alpha is not None:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(winner_raw, dtype=np.float64),
        )

# Test is read only after all model and blend selection is complete.
test = load("test")
hist_test = get_histories("test")
X_test = make_features(test, hist_test)

if winner_family == "recency_binary_gbdt":
    own_test = binary_model.predict(
        X_test,
        num_iteration=binary_model.best_iteration,
        raw_score=True,
    ).astype(np.float64)
elif winner_family == "lambdarank_gbdt":
    own_test = rank_model.predict(
        X_test,
        num_iteration=rank_model.best_iteration,
        raw_score=True,
    ).astype(np.float64)
elif winner_family == "empirical_bayes_history":
    own_test = empirical_history_score(hist_test).astype(np.float64)
else:
    raise RuntimeError("Unknown winning family")

if winner_alpha is None:
    test_scores = own_test
else:
    inc_test = np.load(
        os.path.join(shared, "incumbent_test_scores.npy")
    ).astype(np.float64)

    own_test_norm, _ = normalized_component(
        own_test,
        test.user_id,
        scale=winner_family_scale,
    )
    inc_test_norm, _ = normalized_component(
        inc_test,
        test.user_id,
        scale=inc_scale,
    )
    test_scores = (
        winner_alpha * own_test_norm
        + (1.0 - winner_alpha) * inc_test_norm
    )

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