import os
import gc
import json
import time
import random
import numpy as np
import lightgbm as lgb
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2718
random.seed(SEED)
np.random.seed(SEED)

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "video_type",
]
NUM_FIELDS = [
    "duration_ms",
    "user_follow_user_num",
    "user_fans_user_num",
    "user_friend_user_num",
    "user_register_days",
]
CAT_INDICES = list(range(len(CAT_FIELDS)))


def day_indices(dates):
    d = np.asarray(dates, dtype=np.int64)
    unique_days = np.unique(d)
    return np.searchsorted(unique_days, d).astype(np.float32)


def recency_weights(dates, half_life=4.0):
    idx = day_indices(dates)
    age = float(idx.max()) - idx
    w = np.exp2(-age / float(half_life)).astype(np.float32)
    w /= max(float(w.mean()), 1e-8)
    return w


def make_lgb_features(split):
    columns = []
    for f in CAT_FIELDS:
        columns.append(np.asarray(split.X[f], dtype=np.float32))

    for f in NUM_FIELDS:
        x = np.asarray(split.num[f], dtype=np.float32)
        finite = np.isfinite(x)
        clean = np.zeros_like(x, dtype=np.float32)
        clean[finite] = np.log1p(np.maximum(x[finite], 0.0))
        columns.append(clean)
        columns.append((~finite).astype(np.float32))

    return np.column_stack(columns).astype(np.float32, copy=False)


def fit_binary(train_split, x_train, predict_x):
    y = np.asarray(train_split.y, dtype=np.float32)
    weights = recency_weights(train_split.date, half_life=4.0)

    dtrain = lgb.Dataset(
        x_train,
        label=y,
        weight=weights,
        categorical_feature=CAT_INDICES,
        free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "metric": "None",
        "learning_rate": 0.055,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 250,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "num_threads": max(1, min(8, os.cpu_count() or 1)),
        "verbose": -1,
    }
    model = lgb.train(params, dtrain, num_boost_round=180)
    pred = model.predict(predict_x, num_iteration=model.current_iteration())
    return np.asarray(pred, dtype=np.float64)


def user_sorted_order_and_groups(user_ids):
    u = np.asarray(user_ids, dtype=np.int64)
    order = np.argsort(u, kind="stable")
    sorted_u = u[order]
    if len(sorted_u) == 0:
        return order, np.empty(0, dtype=np.int32)

    starts = np.r_[0, np.flatnonzero(sorted_u[1:] != sorted_u[:-1]) + 1]
    ends = np.r_[starts[1:], len(sorted_u)]
    groups = (ends - starts).astype(np.int32)
    return order, groups


def fit_lambdarank(train_split, x_train, predict_x):
    y = np.asarray(train_split.y, dtype=np.float32)
    weights = recency_weights(train_split.date, half_life=4.0)
    order, groups = user_sorted_order_and_groups(train_split.user_id)

    dtrain = lgb.Dataset(
        x_train[order],
        label=y[order],
        weight=weights[order],
        group=groups,
        categorical_feature=CAT_INDICES,
        free_raw_data=False,
    )
    params = {
        "objective": "lambdarank",
        "metric": "None",
        "lambdarank_truncation_level": 10,
        "learning_rate": 0.055,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "seed": SEED + 100,
        "feature_fraction_seed": SEED + 101,
        "bagging_seed": SEED + 102,
        "num_threads": max(1, min(8, os.cpu_count() or 1)),
        "verbose": -1,
    }
    model = lgb.train(params, dtrain, num_boost_round=150)
    pred = model.predict(predict_x, num_iteration=model.current_iteration())
    return np.asarray(pred, dtype=np.float64)


def fit_svd(train_split, predict_split, rank=24):
    n_users = int(FEATURE_CARDINALITIES["user_id"])
    n_videos = int(FEATURE_CARDINALITIES["video_id"])

    users = np.asarray(train_split.user_id, dtype=np.int64)
    videos = np.asarray(train_split.video_id, dtype=np.int64)
    labels = np.asarray(train_split.y, dtype=np.float32)
    weights = recency_weights(train_split.date, half_life=4.0)

    positive = labels > 0
    rows = users[positive]
    cols = videos[positive]
    vals = weights[positive]

    matrix = sparse.coo_matrix(
        (vals, (rows, cols)),
        shape=(n_users, n_videos),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()

    u, s, vt = svds(
        matrix,
        k=rank,
        which="LM",
        return_singular_vectors=True,
        random_state=SEED,
    )
    order = np.argsort(s)[::-1]
    s = s[order].astype(np.float32)
    u = u[:, order].astype(np.float32)
    vt = vt[order].astype(np.float32)

    user_factors = u * np.sqrt(s)[None, :]
    video_factors = vt.T * np.sqrt(s)[None, :]

    pu = np.asarray(predict_split.user_id, dtype=np.int64)
    pv = np.asarray(predict_split.video_id, dtype=np.int64)
    pred = np.sum(user_factors[pu] * video_factors[pv], axis=1)

    item_count = np.asarray(matrix.getnnz(axis=0), dtype=np.float32)
    popularity = np.log1p(item_count)
    popularity /= max(float(popularity.std()), 1e-6)
    pred = pred.astype(np.float64) + 0.05 * popularity[pv].astype(np.float64)
    return pred


def smoothed_rate(ids, labels, weights, cardinality, prior, strength):
    count = np.bincount(
        ids,
        weights=weights,
        minlength=cardinality,
    ).astype(np.float64)
    positive = np.bincount(
        ids,
        weights=weights * labels,
        minlength=cardinality,
    ).astype(np.float64)
    return (positive + strength * prior) / (count + strength)


def logit(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(x) - np.log1p(-x)


def fit_empirical_bayes(train_split, predict_split):
    y = np.asarray(train_split.y, dtype=np.float64)
    w = recency_weights(train_split.date, half_life=4.0).astype(np.float64)
    prior = float(np.sum(w * y) / np.sum(w))

    tr_video = np.asarray(train_split.video_id, dtype=np.int64)
    tr_author = np.asarray(train_split.X["author_id"], dtype=np.int64)
    tr_user = np.asarray(train_split.user_id, dtype=np.int64)
    tr_tag = np.asarray(train_split.X["tag"], dtype=np.int64)
    tr_duration = np.asarray(train_split.X["duration_bucket"], dtype=np.int64)

    te_video = np.asarray(predict_split.video_id, dtype=np.int64)
    te_author = np.asarray(predict_split.X["author_id"], dtype=np.int64)
    te_user = np.asarray(predict_split.user_id, dtype=np.int64)
    te_tag = np.asarray(predict_split.X["tag"], dtype=np.int64)
    te_duration = np.asarray(
        predict_split.X["duration_bucket"], dtype=np.int64
    )

    video_rate = smoothed_rate(
        tr_video, y, w, int(FEATURE_CARDINALITIES["video_id"]), prior, 20.0
    )
    author_rate = smoothed_rate(
        tr_author, y, w, int(FEATURE_CARDINALITIES["author_id"]), prior, 35.0
    )

    tag_card = int(FEATURE_CARDINALITIES["tag"])
    dur_card = int(FEATURE_CARDINALITIES["duration_bucket"])
    user_card = int(FEATURE_CARDINALITIES["user_id"])

    tr_user_tag = tr_user * tag_card + tr_tag
    te_user_tag = te_user * tag_card + te_tag
    user_tag_rate = smoothed_rate(
        tr_user_tag,
        y,
        w,
        user_card * tag_card,
        prior,
        8.0,
    )

    tr_user_duration = tr_user * dur_card + tr_duration
    te_user_duration = te_user * dur_card + te_duration
    user_duration_rate = smoothed_rate(
        tr_user_duration,
        y,
        w,
        user_card * dur_card,
        prior,
        10.0,
    )

    score = (
        0.42 * logit(video_rate[te_video])
        + 0.25 * logit(author_rate[te_author])
        + 0.23 * logit(user_tag_rate[te_user_tag])
        + 0.10 * logit(user_duration_rate[te_user_duration])
    )
    return np.asarray(score, dtype=np.float64)


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    std = max(float(np.std(x)), 1e-8)
    return (x - float(np.mean(x))) / std


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    base = np.arange(n, dtype=np.int64)
    order = np.lexsort((base, scores, users))
    sorted_users = users[order]

    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], n]

    group_start = np.repeat(starts, ends - starts)
    group_size = np.repeat(ends - starts, ends - starts)
    positions = np.arange(n, dtype=np.float64) - group_start

    ranks_sorted = (positions + 0.5) / group_size
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def combine_scores(user_ids, incumbent, family, alpha, mode):
    if mode == "z":
        return (1.0 - alpha) * zscore(incumbent) + alpha * zscore(family)
    if mode == "rank":
        ir = within_user_rank(user_ids, incumbent)
        fr = within_user_rank(user_ids, family)
        return (1.0 - alpha) * ir + alpha * fr
    raise ValueError(mode)


train = load("train")
valid = load("valid")

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path):
    raise RuntimeError("Trusted incumbent validation predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if len(inc_valid) != len(valid.user_id):
    raise RuntimeError("Incumbent validation prediction length mismatch")

x_train = make_lgb_features(train)
x_valid = make_lgb_features(valid)

family_predictions = {}

family_predictions["lgb_binary_recency4"] = fit_binary(
    train, x_train, x_valid
)
gc.collect()

family_predictions["lambdarank_recency4"] = fit_lambdarank(
    train, x_train, x_valid
)
gc.collect()

family_predictions["svd_positive"] = fit_svd(train, valid, rank=24)
gc.collect()

family_predictions["empirical_bayes"] = fit_empirical_bayes(train, valid)
gc.collect()

candidate_predictions = {"incumbent": inc_valid}
candidate_recipes = {
    "incumbent": {
        "family": None,
        "alpha": 0.0,
        "mode": "incumbent",
    }
}
candidate_scores = {
    "incumbent": float(
        evaluate(valid.user_id, valid.y, inc_valid)["primary"]
    )
}

for family_name, pred in family_predictions.items():
    standalone = evaluate(valid.user_id, valid.y, pred)
    candidate_predictions[family_name] = pred
    candidate_recipes[family_name] = {
        "family": family_name,
        "alpha": 1.0,
        "mode": "standalone",
    }
    candidate_scores[family_name] = float(standalone["primary"])

    for mode in ("z", "rank"):
        for alpha in (0.25, 0.50, 0.75):
            name = "%s_%s_inc%.2f" % (
                family_name,
                mode,
                1.0 - alpha,
            )
            blended = combine_scores(
                valid.user_id,
                inc_valid,
                pred,
                alpha,
                mode,
            )
            candidate_predictions[name] = blended
            candidate_recipes[name] = {
                "family": family_name,
                "alpha": alpha,
                "mode": mode,
            }
            candidate_scores[name] = float(
                evaluate(valid.user_id, valid.y, blended)["primary"]
            )

winner = max(candidate_scores, key=candidate_scores.get)
winner_recipe = candidate_recipes[winner]
valid_scores = np.asarray(candidate_predictions[winner], dtype=np.float64)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS winner=%s binary=%.6f ranker=%.6f svd=%.6f eb=%.6f"
    % (
        winner,
        candidate_scores["lgb_binary_recency4"],
        candidate_scores["lambdarank_recency4"],
        candidate_scores["svd_positive"],
        candidate_scores["empirical_bayes"],
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

test = load("test")
if not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent test predictions are unavailable")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if winner_recipe["family"] is None:
    test_scores = inc_test
else:
    family = winner_recipe["family"]

    class CombinedSplit:
        pass

    combined = CombinedSplit()
    combined.X = {
        key: np.concatenate(
            [
                np.asarray(train.X[key]),
                np.asarray(valid.X[key]),
            ]
        )
        for key in train.X.keys()
    }
    combined.num = {
        key: np.concatenate(
            [
                np.asarray(train.num[key]),
                np.asarray(valid.num[key]),
            ]
        )
        for key in train.num.keys()
    }
    combined.y = np.concatenate(
        [
            np.asarray(train.y),
            np.asarray(valid.y),
        ]
    )
    combined.user_id = np.concatenate(
        [
            np.asarray(train.user_id),
            np.asarray(valid.user_id),
        ]
    )
    combined.video_id = np.concatenate(
        [
            np.asarray(train.video_id),
            np.asarray(valid.video_id),
        ]
    )
    combined.date = np.concatenate(
        [
            np.asarray(train.date),
            np.asarray(valid.date),
        ]
    )

    if family in ("lgb_binary_recency4", "lambdarank_recency4"):
        x_combined = np.vstack([x_train, x_valid]).astype(
            np.float32, copy=False
        )
        x_test = make_lgb_features(test)

        if family == "lgb_binary_recency4":
            family_test = fit_binary(combined, x_combined, x_test)
        else:
            family_test = fit_lambdarank(combined, x_combined, x_test)

    elif family == "svd_positive":
        family_test = fit_svd(combined, test, rank=24)

    elif family == "empirical_bayes":
        family_test = fit_empirical_bayes(combined, test)

    else:
        raise ValueError(family)

    if winner_recipe["mode"] == "standalone":
        test_scores = family_test
    else:
        test_scores = combine_scores(
            test.user_id,
            inc_test,
            family_test,
            float(winner_recipe["alpha"]),
            winner_recipe["mode"],
        )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.6f}'
    % (
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        float(elapsed),
    )
)