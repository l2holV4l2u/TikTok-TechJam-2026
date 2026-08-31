import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 20260831
THREADS = min(8, os.cpu_count() or 8)

np.random.seed(SEED)

train = load("train")
valid = load("valid")
test = load("test")

y = np.asarray(train.y, dtype=np.float32)
yv = np.asarray(valid.y, dtype=np.int8)
uv = np.asarray(valid.user_id, dtype=np.int64)

ntr = len(y)
nva = len(yv)
nte = len(test.user_id)

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "duration_bucket", "upload_type", "hour",
    "user_active_degree", "register_days_bucket",
    "fans_user_num_range", "follow_user_num_range",
    "friend_user_num_range", "music_type", "video_type",
    "onehot_feat2", "onehot_feat3", "onehot_feat7", "onehot_feat8",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
STAT_FIELDS = ["user_id", "video_id", "author_id", "tag"]


def row_recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates, dtype=np.int64)
    unique_dates = np.unique(dates)
    day = np.searchsorted(unique_dates, dates)
    age = unique_dates.size - 1 - day
    w = np.exp2(-age.astype(np.float32) / half_life).astype(np.float32)
    return w / np.maximum(w.mean(), 1.0e-8)


def numeric_matrix(sample, rows=None):
    cols = []
    for name in NUM_FIELDS:
        a = np.asarray(sample.num[name], dtype=np.float32)
        if rows is not None:
            a = a[rows]
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(np.log1p(np.maximum(a, 0.0)))
    return np.column_stack(cols).astype(np.float32, copy=False)


def entity_statistics(fit_sample, fit_rows, fit_y, pred_sample, pred_rows=None,
                      leave_one_out=False):
    """
    All sufficient statistics are computed from fit_rows only. For fitting
    rows, leave-one-out values prevent direct target leakage.
    """
    result = []
    prior = float(np.mean(fit_y))
    fit_rows = np.asarray(fit_rows, dtype=np.int64)

    for field in STAT_FIELDS:
        card = int(FEATURE_CARDINALITIES[field])
        fit_ids = np.asarray(fit_sample.X[field], dtype=np.int64)[fit_rows]
        counts = np.bincount(fit_ids, minlength=card).astype(np.float32)
        positives = np.bincount(
            fit_ids, weights=fit_y, minlength=card
        ).astype(np.float32)

        pred_ids_all = np.asarray(pred_sample.X[field], dtype=np.int64)
        pred_ids = pred_ids_all if pred_rows is None else pred_ids_all[pred_rows]

        c = counts[pred_ids].astype(np.float32)
        p = positives[pred_ids].astype(np.float32)

        if leave_one_out:
            # This mode is used only when prediction rows equal fit_rows.
            c = np.maximum(c - 1.0, 0.0)
            p = np.maximum(
                p - np.asarray(fit_y, dtype=np.float32), 0.0
            )

        smooth = 18.0 if field == "user_id" else 35.0
        rate = (p + smooth * prior) / (c + smooth)
        result.append(np.log1p(c))
        result.append(rate.astype(np.float32))

    return np.column_stack(result).astype(np.float32, copy=False)


def build_matrix(sample, rows, stats):
    cat = np.column_stack([
        np.asarray(sample.X[f], dtype=np.float32)[rows]
        for f in CAT_FIELDS
    ])
    num = numeric_matrix(sample, rows)
    return np.column_stack([cat, num, stats]).astype(np.float32, copy=False)


def group_order(sample, rows, user_day=True):
    rows = np.asarray(rows, dtype=np.int64)
    users = np.asarray(sample.user_id, dtype=np.int64)[rows]
    dates = np.asarray(sample.date, dtype=np.int64)[rows]
    times = np.asarray(sample.time_ms, dtype=np.int64)[rows]

    if user_day:
        order_local = np.lexsort(
            (rows, times, dates, users)
        )
        ou = users[order_local]
        od = dates[order_local]
        boundary = np.r_[
            True,
            (ou[1:] != ou[:-1]) | (od[1:] != od[:-1])
        ]
    else:
        order_local = np.lexsort((rows, times, users))
        ou = users[order_local]
        boundary = np.r_[True, ou[1:] != ou[:-1]]

    starts = np.flatnonzero(boundary)
    sizes = np.diff(np.r_[starts, len(rows)]).astype(np.int32)
    return order_local, sizes


def user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    rows = np.arange(len(scores), dtype=np.int64)
    order = np.lexsort((rows, scores, users))
    su = users[order]

    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    ends = np.r_[starts[1:], len(order)]
    lengths = ends - starts
    positions = np.arange(len(order), dtype=np.float64) - np.repeat(
        starts, lengths
    )
    denominators = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    ranked = positions / denominators

    out = np.empty(len(scores), dtype=np.float64)
    out[order] = ranked
    return out


def slate_sizes(users):
    users = np.asarray(users, dtype=np.int64)
    _, inverse, counts = np.unique(
        users, return_inverse=True, return_counts=True
    )
    return counts[inverse].astype(np.float32)


def empirical_score(stats):
    # Stats are [log_count, rate] pairs for user, video, author, tag.
    user_rate = stats[:, 1]
    video_rate = stats[:, 3]
    author_rate = stats[:, 5]
    tag_rate = stats[:, 7]
    video_count = np.expm1(stats[:, 2])
    author_count = np.expm1(stats[:, 4])

    item_reliability = video_count / (video_count + 40.0)
    score = (
        0.16 * user_rate
        + 0.38 * (
            item_reliability * video_rate
            + (1.0 - item_reliability) * author_rate
        )
        + 0.28 * author_rate
        + 0.18 * tag_rate
        + 0.006 * np.log1p(video_count)
        + 0.004 * np.log1p(author_count)
    )
    return score.astype(np.float64)


def gate_matrix(point, ranker, empirical, stats, users):
    point = np.asarray(point, dtype=np.float32)
    ranker = np.asarray(ranker, dtype=np.float32)
    empirical = np.asarray(empirical, dtype=np.float32)

    rp = user_rank(users, point).astype(np.float32)
    rr = user_rank(users, ranker).astype(np.float32)
    re = user_rank(users, empirical).astype(np.float32)

    size = np.log1p(slate_sizes(users)).astype(np.float32)
    user_count = stats[:, 0]
    video_count = stats[:, 2]
    author_count = stats[:, 4]

    return np.column_stack([
        point, ranker, empirical,
        rp, rr, re,
        rp - rr, rp - re, rr - re,
        np.abs(rp - rr), np.abs(rp - re), np.abs(rr - re),
        size, user_count, video_count, author_count,
        stats[:, 1], stats[:, 3], stats[:, 5], stats[:, 7],
    ]).astype(np.float32, copy=False)


categorical_indices = list(range(len(CAT_FIELDS)))

point_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.055,
    "num_leaves": 47,
    "max_depth": 9,
    "min_data_in_leaf": 650,
    "lambda_l1": 0.2,
    "lambda_l2": 9.0,
    "feature_fraction": 0.88,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "max_bin": 127,
    "max_cat_threshold": 32,
    "cat_smooth": 35.0,
    "verbose": -1,
    "num_threads": THREADS,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
}

rank_params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5],
    "label_gain": [0, 1],
    "lambdarank_truncation_level": 12,
    "learning_rate": 0.05,
    "num_leaves": 39,
    "max_depth": 9,
    "min_data_in_leaf": 650,
    "lambda_l2": 9.0,
    "feature_fraction": 0.88,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "max_bin": 127,
    "max_cat_threshold": 32,
    "cat_smooth": 35.0,
    "verbose": -1,
    "num_threads": THREADS,
    "seed": SEED + 10,
    "feature_fraction_seed": SEED + 11,
    "bagging_seed": SEED + 12,
}

# ----------------------------------------------------------------------
# Train-only temporal holdout for learning the conditional gate.
# ----------------------------------------------------------------------
train_dates = np.asarray(train.date, dtype=np.int64)
unique_dates = np.unique(train_dates)
holdout_dates = unique_dates[-3:]

early_rows = np.flatnonzero(train_dates < holdout_dates[0])
hold_rows = np.flatnonzero(train_dates >= holdout_dates[0])

y_early = y[early_rows]
y_hold = y[hold_rows]

early_stats = entity_statistics(
    train, early_rows, y_early, train, early_rows, leave_one_out=True
)
hold_stats = entity_statistics(
    train, early_rows, y_early, train, hold_rows, leave_one_out=False
)

x_early = build_matrix(train, early_rows, early_stats)
x_hold = build_matrix(train, hold_rows, hold_stats)

early_weights = row_recency_weights(train_dates[early_rows], half_life=4.0)

dpoint_early = lgb.Dataset(
    x_early,
    label=y_early,
    weight=early_weights,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
point_early_model = lgb.train(
    point_params, dpoint_early, num_boost_round=135
)
hold_point = point_early_model.predict(x_hold)

early_order, early_groups = group_order(train, early_rows, user_day=True)
drank_early = lgb.Dataset(
    x_early[early_order],
    label=y_early[early_order],
    weight=early_weights[early_order],
    group=early_groups,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
rank_early_model = lgb.train(
    rank_params, drank_early, num_boost_round=145
)
hold_rank = rank_early_model.predict(x_hold)
hold_empirical = empirical_score(hold_stats)

hold_users = np.asarray(train.user_id, dtype=np.int64)[hold_rows]
x_gate_hold = gate_matrix(
    hold_point, hold_rank, hold_empirical, hold_stats, hold_users
)

# The gate itself is listwise and shallow. It can condition on slate size,
# disagreement, and history reliability without becoming a large fourth
# standalone predictor.
gate_order, gate_groups = group_order(train, hold_rows, user_day=False)
gate_params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5],
    "label_gain": [0, 1],
    "lambdarank_truncation_level": 10,
    "learning_rate": 0.045,
    "num_leaves": 11,
    "max_depth": 4,
    "min_data_in_leaf": 350,
    "lambda_l2": 18.0,
    "feature_fraction": 0.92,
    "bagging_fraction": 0.92,
    "bagging_freq": 1,
    "max_bin": 63,
    "verbose": -1,
    "num_threads": THREADS,
    "seed": SEED + 20,
}
dgate = lgb.Dataset(
    x_gate_hold[gate_order],
    label=y_hold[gate_order],
    group=gate_groups,
    free_raw_data=True,
)
gate_model = lgb.train(gate_params, dgate, num_boost_round=75)

hold_gate = gate_model.predict(x_gate_hold)
hold_metrics = {
    "point": evaluate(hold_users, y_hold.astype(np.int8), hold_point)["primary"],
    "rank": evaluate(hold_users, y_hold.astype(np.int8), hold_rank)["primary"],
    "empirical": evaluate(
        hold_users, y_hold.astype(np.int8), hold_empirical
    )["primary"],
    "gate": evaluate(hold_users, y_hold.astype(np.int8), hold_gate)["primary"],
}
print("FINDINGS temporal_holdout=" + json.dumps(hold_metrics, sort_keys=True))

del (
    dpoint_early, drank_early, x_early, x_hold, early_stats,
    point_early_model, rank_early_model
)
gc.collect()

# ----------------------------------------------------------------------
# Refit the three base families on all allowed training rows.
# ----------------------------------------------------------------------
all_rows = np.arange(ntr, dtype=np.int64)

full_train_stats = entity_statistics(
    train, all_rows, y, train, all_rows, leave_one_out=True
)
valid_stats = entity_statistics(
    train, all_rows, y, valid, None, leave_one_out=False
)
test_stats = entity_statistics(
    train, all_rows, y, test, None, leave_one_out=False
)

x_train = build_matrix(train, all_rows, full_train_stats)
x_valid = build_matrix(valid, np.arange(nva), valid_stats)
x_test = build_matrix(test, np.arange(nte), test_stats)

full_weights = row_recency_weights(train_dates, half_life=4.0)

dpoint = lgb.Dataset(
    x_train,
    label=y,
    weight=full_weights,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
point_model = lgb.train(point_params, dpoint, num_boost_round=135)
valid_point = point_model.predict(x_valid)
test_point = point_model.predict(x_test)

full_order, full_groups = group_order(train, all_rows, user_day=True)
drank = lgb.Dataset(
    x_train[full_order],
    label=y[full_order],
    weight=full_weights[full_order],
    group=full_groups,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
rank_model = lgb.train(rank_params, drank, num_boost_round=145)
valid_rank = rank_model.predict(x_valid)
test_rank = rank_model.predict(x_test)

valid_empirical = empirical_score(valid_stats)
test_empirical = empirical_score(test_stats)

valid_gate_features = gate_matrix(
    valid_point, valid_rank, valid_empirical, valid_stats,
    np.asarray(valid.user_id, dtype=np.int64)
)
test_gate_features = gate_matrix(
    test_point, test_rank, test_empirical, test_stats,
    np.asarray(test.user_id, dtype=np.int64)
)

valid_gate = gate_model.predict(valid_gate_features)
test_gate = gate_model.predict(test_gate_features)

# ----------------------------------------------------------------------
# Compare all structurally different families and their permitted blends
# with the trusted incumbent in within-user rank space.
# ----------------------------------------------------------------------
shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

candidate_valid = {
    "pointwise_boost": valid_point,
    "lambda_rank": valid_rank,
    "empirical_bayes": valid_empirical,
    "conditional_gate": valid_gate,
}
candidate_test = {
    "pointwise_boost": test_point,
    "lambda_rank": test_rank,
    "empirical_bayes": test_empirical,
    "conditional_gate": test_gate,
}

candidate_scores = {}
best_name = None
best_primary = -np.inf
best_valid = None
best_test = None
best_raw = valid_gate
best_is_combination = False

for name in candidate_valid:
    metric = evaluate(uv, yv, candidate_valid[name])
    candidate_scores[name] = float(metric["primary"])
    if metric["primary"] > best_primary:
        best_primary = metric["primary"]
        best_name = name
        best_valid = candidate_valid[name]
        best_test = candidate_test[name]
        best_raw = candidate_valid[name]
        best_is_combination = False

if os.path.exists(inc_valid_path) and os.path.exists(inc_test_path):
    incumbent_valid = np.load(inc_valid_path)
    incumbent_test = np.load(inc_test_path)

    inc_v_rank = user_rank(valid.user_id, incumbent_valid)
    inc_t_rank = user_rank(test.user_id, incumbent_test)

    # A coarse grid limits validation adaptivity while exercising every
    # genuinely distinct family against the same trusted reference.
    for family in candidate_valid:
        own_v_rank = user_rank(valid.user_id, candidate_valid[family])
        own_t_rank = user_rank(test.user_id, candidate_test[family])

        for alpha in (0.20, 0.35, 0.50, 0.65, 0.80):
            blended_valid = (
                alpha * own_v_rank + (1.0 - alpha) * inc_v_rank
            )
            metric = evaluate(uv, yv, blended_valid)
            name = "%s_inc_a%.2f" % (family, alpha)
            candidate_scores[name] = float(metric["primary"])

            if metric["primary"] > best_primary:
                best_primary = metric["primary"]
                best_name = name
                best_valid = blended_valid
                best_test = (
                    alpha * own_t_rank + (1.0 - alpha) * inc_t_rank
                )
                best_raw = candidate_valid[family]
                best_is_combination = True

final_metric = evaluate(uv, yv, best_valid)

print("FINDINGS selected=%s" % best_name)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    if best_is_combination:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.4f}'
    % (
        final_metric["primary"],
        final_metric["gauc"],
        final_metric["ndcg@5"],
        elapsed,
    )
)