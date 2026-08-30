import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
THREADS = max(1, min(8, os.cpu_count() or 1))
NUM_ROUNDS = 190

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "hour",
    "user_active_degree",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "onehot_feat1",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

STAT_FIELDS = [
    ("video_id", 25.0),
    ("author_id", 40.0),
    ("tag", 100.0),
    ("duration_bucket", 160.0),
]


class Combined:
    pass


def merge_splits(a, b):
    c = Combined()
    needed = set(CAT_FIELDS + [x[0] for x in STAT_FIELDS])
    c.X = {
        name: np.concatenate([a.X[name], b.X[name]])
        for name in needed
    }
    c.num = {
        name: np.concatenate([a.num[name], b.num[name]])
        for name in NUM_FIELDS
    }
    c.user_id = np.concatenate([a.user_id, b.user_id])
    c.video_id = np.concatenate([a.video_id, b.video_id])
    c.date = np.concatenate([a.date, b.date])
    return c


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / max(float(x.std()), 1e-8)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, users))
    su = users[order]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    ends = np.r_[starts[1:], n]
    sizes = ends - starts
    position = np.arange(n) - np.repeat(starts, sizes)
    denom = np.maximum(np.repeat(sizes, sizes) - 1, 1)
    ranked_sorted = position.astype(np.float64) / denom
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


def recency_weights(dates, half_life):
    if half_life is None:
        return np.ones(len(dates), dtype=np.float32)
    dates = np.asarray(dates, dtype=np.int64)
    age = dates.max() - dates
    w = np.exp2(-age.astype(np.float32) / float(half_life))
    w /= max(float(w.mean()), 1e-8)
    return w.astype(np.float32)


def build_stat_tables(fit_split, y):
    y = np.asarray(y, dtype=np.float64)
    prior = float(y.mean())
    tables = {}
    for field, smoothing in STAT_FIELDS:
        ids = np.asarray(fit_split.X[field], dtype=np.int64)
        card = FEATURE_CARDINALITIES[field]
        count = np.bincount(ids, minlength=card).astype(np.float64)
        positive = np.bincount(ids, weights=y, minlength=card)
        rate = (positive + smoothing * prior) / (count + smoothing)
        tables[field] = {
            "count": count,
            "positive": positive,
            "rate": rate,
            "smoothing": smoothing,
        }
    return prior, tables


def make_features(fit_split, y, eval_split):
    y64 = np.asarray(y, dtype=np.float64)
    prior, tables = build_stat_tables(fit_split, y64)

    train_columns = []
    eval_columns = []

    for field in CAT_FIELDS:
        train_columns.append(
            np.asarray(fit_split.X[field], dtype=np.float32)
        )
        eval_columns.append(
            np.asarray(eval_split.X[field], dtype=np.float32)
        )

    for field in NUM_FIELDS:
        tr = np.asarray(fit_split.num[field], dtype=np.float64)
        ev = np.asarray(eval_split.num[field], dtype=np.float64)
        finite = np.isfinite(tr)
        median = float(np.median(tr[finite])) if finite.any() else 0.0
        tr = np.where(np.isfinite(tr), tr, median)
        ev = np.where(np.isfinite(ev), ev, median)
        train_columns.append(np.log1p(np.maximum(tr, 0)).astype(np.float32))
        eval_columns.append(np.log1p(np.maximum(ev, 0)).astype(np.float32))

    for field, _ in STAT_FIELDS:
        ids_tr = np.asarray(fit_split.X[field], dtype=np.int64)
        ids_ev = np.asarray(eval_split.X[field], dtype=np.int64)
        table = tables[field]
        count = table["count"]
        positive = table["positive"]
        smoothing = table["smoothing"]

        loo_count = np.maximum(count[ids_tr] - 1.0, 0.0)
        loo_positive = positive[ids_tr] - y64
        loo_rate = (
            loo_positive + smoothing * prior
        ) / (loo_count + smoothing)

        full_rate = table["rate"][ids_ev]
        full_count = count[ids_ev]

        train_columns.append(logit(loo_rate).astype(np.float32))
        train_columns.append(np.log1p(loo_count).astype(np.float32))
        eval_columns.append(logit(full_rate).astype(np.float32))
        eval_columns.append(np.log1p(full_count).astype(np.float32))

    x_train = np.ascontiguousarray(
        np.column_stack(train_columns), dtype=np.float32
    )
    x_eval = np.ascontiguousarray(
        np.column_stack(eval_columns), dtype=np.float32
    )
    return x_train, x_eval, tables, prior


def empirical_predict(split, tables, prior):
    score = np.zeros(len(split.user_id), dtype=np.float64)
    coefficients = {
        "video_id": 0.55,
        "author_id": 0.30,
        "tag": 0.10,
        "duration_bucket": 0.05,
    }
    for field, coefficient in coefficients.items():
        ids = np.asarray(split.X[field], dtype=np.int64)
        score += coefficient * logit(tables[field]["rate"][ids])
    return score.astype(np.float32)


def fit_binary(x, y, weights):
    params = {
        "objective": "binary",
        "metric": "None",
        "learning_rate": 0.055,
        "num_leaves": 96,
        "max_depth": -1,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "cat_smooth": 20.0,
        "cat_l2": 8.0,
        "num_threads": THREADS,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "verbose": -1,
    }
    ds = lgb.Dataset(
        x,
        label=np.asarray(y, dtype=np.float32),
        weight=weights,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )
    return lgb.train(params, ds, num_boost_round=NUM_ROUNDS)


def ranking_order_and_groups(users, dates, max_group=5000):
    users = np.asarray(users, dtype=np.int64)
    dates = np.asarray(dates, dtype=np.int64)
    row = np.arange(len(users), dtype=np.int64)
    order = np.lexsort((row, dates, users))
    su = users[order]
    boundaries = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    ends = np.r_[boundaries[1:], len(order)]

    groups = []
    for start, end in zip(boundaries, ends):
        size = int(end - start)
        while size > max_group:
            groups.append(max_group)
            size -= max_group
        if size:
            groups.append(size)
    return order, np.asarray(groups, dtype=np.int32)


def fit_lambdarank(x, y, users, dates, weights):
    order, groups = ranking_order_and_groups(users, dates)
    params = {
        "objective": "lambdarank",
        "metric": "None",
        "learning_rate": 0.045,
        "num_leaves": 80,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.88,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 2.5,
        "max_bin": 127,
        "lambdarank_truncation_level": 10,
        "label_gain": [0, 1],
        "num_threads": THREADS,
        "seed": SEED + 10,
        "verbose": -1,
    }
    ds = lgb.Dataset(
        x[order],
        label=np.asarray(y, dtype=np.int8)[order],
        weight=np.asarray(weights, dtype=np.float32)[order],
        group=groups,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )
    return lgb.train(params, ds, num_boost_round=NUM_ROUNDS)


def fit_svd(fit_split, y, rank=24):
    users = np.asarray(fit_split.user_id, dtype=np.int64)
    videos = np.asarray(fit_split.video_id, dtype=np.int64)
    y = np.asarray(y, dtype=np.float32)

    n_users = FEATURE_CARDINALITIES["user_id"]
    n_videos = FEATURE_CARDINALITIES["video_id"]

    values = 0.15 + y
    matrix = sparse.coo_matrix(
        (values, (users, videos)),
        shape=(n_users, n_videos),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()

    model = TruncatedSVD(
        n_components=rank,
        n_iter=3,
        random_state=SEED,
        algorithm="randomized",
    )
    user_vectors = model.fit_transform(matrix).astype(np.float32)
    video_vectors = model.components_.T.astype(np.float32)
    return user_vectors, video_vectors


def predict_svd(model, split):
    user_vectors, video_vectors = model
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    return np.einsum(
        "ij,ij->i", user_vectors[users], video_vectors[videos]
    ).astype(np.float32)


def select_blends(valid, y_valid, incumbent, predictions):
    incumbent = np.asarray(incumbent, dtype=np.float64)
    incumbent_z = zscore(incumbent)
    incumbent_rank = within_user_rank(valid.user_id, incumbent)
    alphas = [0.15, 0.30, 0.45, 0.60, 0.75]

    candidates = {
        "trusted_incumbent": float(
            evaluate(valid.user_id, y_valid, incumbent)["primary"]
        )
    }
    best_score = candidates["trusted_incumbent"]
    best_descriptor = ("trusted_incumbent", "standalone", 1.0)
    best_prediction = incumbent.copy()

    for name, pred in predictions.items():
        pred = np.asarray(pred, dtype=np.float64)
        standalone = float(
            evaluate(valid.user_id, y_valid, pred)["primary"]
        )
        candidates[name] = standalone
        if standalone > best_score:
            best_score = standalone
            best_descriptor = (name, "standalone", 1.0)
            best_prediction = pred.copy()

        pred_z = zscore(pred)
        pred_rank = within_user_rank(valid.user_id, pred)

        local_raw = (-np.inf, None, None)
        local_rank = (-np.inf, None, None)

        for alpha in alphas:
            raw = alpha * pred_z + (1.0 - alpha) * incumbent_z
            score_raw = float(
                evaluate(valid.user_id, y_valid, raw)["primary"]
            )
            if score_raw > local_raw[0]:
                local_raw = (score_raw, alpha, raw)

            ranked = (
                alpha * pred_rank
                + (1.0 - alpha) * incumbent_rank
            )
            score_rank = float(
                evaluate(valid.user_id, y_valid, ranked)["primary"]
            )
            if score_rank > local_rank[0]:
                local_rank = (score_rank, alpha, ranked)

        candidates[name + "_rawblend"] = float(local_raw[0])
        candidates[name + "_rankblend"] = float(local_rank[0])

        if local_raw[0] > best_score:
            best_score = local_raw[0]
            best_descriptor = (name, "rawblend", local_raw[1])
            best_prediction = np.asarray(local_raw[2], dtype=np.float64)

        if local_rank[0] > best_score:
            best_score = local_rank[0]
            best_descriptor = (name, "rankblend", local_rank[1])
            best_prediction = np.asarray(local_rank[2], dtype=np.float64)

    return candidates, best_descriptor, best_prediction


def apply_blend(mode, alpha, new_scores, incumbent_scores, user_ids):
    if mode == "standalone":
        return np.asarray(new_scores, dtype=np.float64)
    if mode == "rawblend":
        return (
            alpha * zscore(new_scores)
            + (1.0 - alpha) * zscore(incumbent_scores)
        )
    return (
        alpha * within_user_rank(user_ids, new_scores)
        + (1.0 - alpha)
        * within_user_rank(user_ids, incumbent_scores)
    )


def main():
    train = load("train")
    valid = load("valid")
    y_train = np.asarray(train.y, dtype=np.float32)
    y_valid = np.asarray(valid.y, dtype=np.int8)

    shared = os.environ.get("SHARED_ARTIFACTS", "")
    incumbent_valid = np.load(
        os.path.join(shared, "incumbent_valid_scores.npy")
    ).astype(np.float64)

    x_train, x_valid, tables, prior = make_features(
        train, y_train, valid
    )

    predictions = {}

    uniform_weights = recency_weights(train.date, None)
    recent_weights = recency_weights(train.date, 4.0)

    binary_uniform = fit_binary(
        x_train, y_train, uniform_weights
    )
    predictions["gbdt_binary_uniform"] = binary_uniform.predict(
        x_valid
    ).astype(np.float32)
    del binary_uniform
    gc.collect()

    binary_recent = fit_binary(
        x_train, y_train, recent_weights
    )
    predictions["gbdt_binary_recency_h4"] = binary_recent.predict(
        x_valid
    ).astype(np.float32)
    del binary_recent
    gc.collect()

    rank_recent = fit_lambdarank(
        x_train,
        y_train,
        train.user_id,
        train.date,
        recent_weights,
    )
    predictions["lambdarank_recency_h4"] = rank_recent.predict(
        x_valid
    ).astype(np.float32)
    del rank_recent
    gc.collect()

    predictions["empirical_bayes"] = empirical_predict(
        valid, tables, prior
    )

    svd_model = fit_svd(train, y_train)
    predictions["latent_svd"] = predict_svd(svd_model, valid)
    del svd_model
    gc.collect()

    candidates, descriptor, valid_scores = select_blends(
        valid, y_valid, incumbent_valid, predictions
    )
    metrics = evaluate(valid.user_id, y_valid, valid_scores)

    out_dir = os.environ.get("ITER_OUT")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(
            os.path.join(out_dir, "scores_valid.npy"),
            np.asarray(valid_scores, dtype=np.float64),
        )

    selected_name, blend_mode, alpha = descriptor

    test = load("test")
    incumbent_test = np.load(
        os.path.join(shared, "incumbent_test_scores.npy")
    ).astype(np.float64)

    if selected_name == "trusted_incumbent":
        test_scores = incumbent_test
    else:
        combined = merge_splits(train, valid)
        y_combined = np.concatenate(
            [y_train, y_valid.astype(np.float32)]
        )

        if selected_name == "latent_svd":
            refit = fit_svd(combined, y_combined)
            new_test = predict_svd(refit, test)
        elif selected_name == "empirical_bayes":
            _, refit_tables = build_stat_tables(combined, y_combined)
            new_test = empirical_predict(
                test, refit_tables, float(y_combined.mean())
            )
        else:
            x_refit, x_test, _, _ = make_features(
                combined, y_combined, test
            )
            if selected_name == "gbdt_binary_uniform":
                refit_weights = recency_weights(combined.date, None)
                refit = fit_binary(
                    x_refit, y_combined, refit_weights
                )
            elif selected_name == "gbdt_binary_recency_h4":
                refit_weights = recency_weights(combined.date, 4.0)
                refit = fit_binary(
                    x_refit, y_combined, refit_weights
                )
            else:
                refit_weights = recency_weights(combined.date, 4.0)
                refit = fit_lambdarank(
                    x_refit,
                    y_combined,
                    combined.user_id,
                    combined.date,
                    refit_weights,
                )
            new_test = refit.predict(x_test).astype(np.float32)

        test_scores = apply_blend(
            blend_mode,
            float(alpha),
            new_test,
            incumbent_test,
            test.user_id,
        )

    if out_dir:
        np.save(
            os.path.join(out_dir, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

    print(
        "FINDINGS "
        + json.dumps(
            {
                "winner": selected_name,
                "blend_mode": blend_mode,
                "new_model_weight": float(alpha),
                "binary_recency_minus_uniform": float(
                    candidates["gbdt_binary_recency_h4"]
                    - candidates["gbdt_binary_uniform"]
                ),
            },
            separators=(",", ":"),
        )
    )
    print(
        "CANDIDATES "
        + json.dumps(candidates, sort_keys=True, separators=(",", ":"))
    )

    payload = {
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(time.time() - START),
    }
    print("METRICS " + json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()