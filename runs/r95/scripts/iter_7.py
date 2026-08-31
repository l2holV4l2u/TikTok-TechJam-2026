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
SEED = 2026

train = load("train")
valid = load("valid")
test = load("test")

# Exclude user identity to make the complementary model less dependent on
# user activity patterns that shift sharply after the date boundary. Video
# and author identities are represented both directly and by train-only
# smoothed histories.
CAT_FIELDS = [
    "author_id", "duration_bucket", "fans_user_num_range",
    "follow_user_num_range", "friend_user_num_range", "hour",
    "is_live_streamer", "is_lowactive_period", "is_video_author",
    "music_type", "onehot_feat0", "onehot_feat1", "onehot_feat10",
    "onehot_feat11", "onehot_feat12", "onehot_feat13", "onehot_feat14",
    "onehot_feat15", "onehot_feat16", "onehot_feat17", "onehot_feat2",
    "onehot_feat3", "onehot_feat4", "onehot_feat5", "onehot_feat6",
    "onehot_feat7", "onehot_feat8", "onehot_feat9",
    "register_days_bucket", "register_days_range", "tab", "tag",
    "upload_type", "user_active_degree", "video_id", "video_type",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]


def get_histories(split_name):
    result = {}
    for entity in ("video_id", "author_id"):
        h = historical_features(split_name, key=entity)
        for name, values in sorted(h.items()):
            result[entity + "__" + name] = np.asarray(values, dtype=np.float32)
    return result


htr = get_histories("train")
hva = get_histories("valid")
hte = get_histories("test")
hist_names = sorted(set(htr).intersection(hva).intersection(hte))

print(
    "FINDINGS history_features=%d names=%s"
    % (len(hist_names), ",".join(hist_names)),
    flush=True,
)


def numeric_transform(a):
    a = np.asarray(a, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    # All exposed raw quantities are nonnegative and heavy-tailed.
    return np.log1p(np.maximum(a, 0.0)).astype(np.float32)


def build_matrix(split, histories):
    columns = []
    for f in CAT_FIELDS:
        columns.append(np.asarray(split.X[f], dtype=np.float32))
    for f in NUM_FIELDS:
        columns.append(numeric_transform(split.num[f]))
    for f in hist_names:
        a = np.asarray(histories[f], dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(a)
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


Xtr = build_matrix(train, htr)
Xva = build_matrix(valid, hva)
Xte = build_matrix(test, hte)
ytr = np.asarray(train.y, dtype=np.float32)
yva = np.asarray(valid.y, dtype=np.int8)

del htr, hva, hte
gc.collect()

# Weight the main predictor toward the end of train. Unlike a side-component
# experiment, this changes every tree split and leaf estimate.
last_date = int(np.max(np.asarray(train.date, dtype=np.int64)))
age = last_date - np.asarray(train.date, dtype=np.int64)
sample_weight = np.exp2(-age.astype(np.float32) / 4.0)
sample_weight /= np.mean(sample_weight)

categorical_indices = list(range(len(CAT_FIELDS)))
dtrain = lgb.Dataset(
    Xtr,
    label=ytr,
    weight=sample_weight,
    categorical_feature=categorical_indices,
    free_raw_data=True,
)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.045,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 400,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.82,
    "bagging_freq": 1,
    "lambda_l1": 0.15,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "cat_smooth": 40.0,
    "cat_l2": 12.0,
    "min_gain_to_split": 1e-4,
    "num_threads": max(1, min(12, os.cpu_count() or 1)),
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "data_random_seed": SEED + 3,
    "verbose": -1,
}

model = lgb.train(params, dtrain, num_boost_round=260)
tree_va = model.predict(Xva, num_iteration=260).astype(np.float64)
tree_te = model.predict(Xte, num_iteration=260).astype(np.float64)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_va = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_te = np.asarray(
    np.load(os.path.join(shared, "incumbent_test_scores.npy")),
    dtype=np.float64,
)


def within_user_ranks(user_ids, scores):
    """Ascending query-local percentile ranks, vectorized over all rows."""
    users = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    # Row position makes equal-score ordering deterministic.
    order = np.lexsort((row, scores, users))
    sorted_users = users[order]

    starts = np.empty(n, dtype=np.int64)
    starts[0] = 0
    boundary = np.empty(n, dtype=bool)
    boundary[0] = True
    boundary[1:] = sorted_users[1:] != sorted_users[:-1]
    starts[boundary] = np.flatnonzero(boundary)
    np.maximum.accumulate(starts, out=starts)

    local_position = np.arange(n, dtype=np.int64) - starts

    ends = np.empty(n, dtype=np.int64)
    ends[-1] = n
    end_boundary = np.empty(n, dtype=bool)
    end_boundary[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_boundary[-1] = True
    end_positions = np.flatnonzero(end_boundary)
    ends[end_positions] = end_positions + 1
    # Fill each group with its ending position.
    ends_rev = ends[::-1].copy()
    np.minimum.accumulate(ends_rev, out=ends_rev)
    ends = ends_rev[::-1]
    group_size = ends - starts

    ranked_sorted = np.full(n, 0.5, dtype=np.float64)
    multi = group_size > 1
    ranked_sorted[multi] = (
        local_position[multi].astype(np.float64)
        / (group_size[multi].astype(np.float64) - 1.0)
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


tree_rank_va = within_user_ranks(valid.user_id, tree_va)
tree_rank_te = within_user_ranks(test.user_id, tree_te)
inc_rank_va = within_user_ranks(valid.user_id, inc_va)
inc_rank_te = within_user_ranks(test.user_id, inc_te)

candidate_valid = {
    "recency_history_lgbm_raw": tree_va,
    "incumbent": inc_va,
}
candidate_test = {
    "recency_history_lgbm_raw": tree_te,
    "incumbent": inc_te,
}
candidate_raw = {
    "recency_history_lgbm_raw": tree_va,
    "incumbent": tree_va,
}

# The trusted-incumbent contract explicitly permits choosing blend weight on
# validation and applying the identical fixed weight on test.
for own_weight in (0.10, 0.20, 0.30, 0.40, 0.50, 0.65):
    name = "query_rank_blend_tree_%.2f" % own_weight
    candidate_valid[name] = (
        own_weight * tree_rank_va + (1.0 - own_weight) * inc_rank_va
    )
    candidate_test[name] = (
        own_weight * tree_rank_te + (1.0 - own_weight) * inc_rank_te
    )
    candidate_raw[name] = tree_va

# Also compare ordinary score blending after converting the tree probability
# to a logit. This separates the effect of complementary predictions from the
# effect of query-local calibration.
eps = 1e-6
tree_logit_va = np.log(
    np.clip(tree_va, eps, 1.0 - eps)
    / np.clip(1.0 - tree_va, eps, 1.0 - eps)
)
tree_logit_te = np.log(
    np.clip(tree_te, eps, 1.0 - eps)
    / np.clip(1.0 - tree_te, eps, 1.0 - eps)
)

for own_weight in (0.15, 0.30, 0.45):
    name = "logit_blend_tree_%.2f" % own_weight
    candidate_valid[name] = (
        own_weight * tree_logit_va + (1.0 - own_weight) * inc_va
    )
    candidate_test[name] = (
        own_weight * tree_logit_te + (1.0 - own_weight) * inc_te
    )
    candidate_raw[name] = tree_va

metrics_by_name = {}
for name, scores in candidate_valid.items():
    metrics_by_name[name] = evaluate(valid.user_id, yva, scores)

best_name = max(
    metrics_by_name,
    key=lambda n: (
        metrics_by_name[n]["primary"],
        metrics_by_name[n]["gauc"],
    ),
)
best_metrics = metrics_by_name[best_name]
best_va = candidate_valid[best_name]
best_te = candidate_test[best_name]

print(
    "CANDIDATES "
    + json.dumps(
        {
            name: round(float(m["primary"]), 7)
            for name, m in metrics_by_name.items()
        },
        sort_keys=True,
    ),
    flush=True,
)

tree_metric = metrics_by_name["recency_history_lgbm_raw"]
rank_corr = np.corrcoef(tree_rank_va, inc_rank_va)[0, 1]
print(
    "FINDINGS tree_primary=%.6f tree_gauc=%.6f tree_ndcg5=%.6f "
    "query_rank_corr=%.6f winner=%s"
    % (
        tree_metric["primary"],
        tree_metric["gauc"],
        tree_metric["ndcg@5"],
        rank_corr,
        best_name,
    ),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_va, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_te, dtype=np.float64),
    )
    if best_name != "recency_history_lgbm_raw":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(tree_va, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.6f}'
    % (
        best_metrics["primary"],
        best_metrics["gauc"],
        best_metrics["ndcg@5"],
        elapsed,
    )
)