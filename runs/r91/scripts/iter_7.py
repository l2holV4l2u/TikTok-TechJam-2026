import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2026

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "hour",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "video_type",
    "is_video_author",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "register_days_bucket",
]

NB_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "hour",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "video_type",
    "is_video_author",
]

ENTITY_FIELDS = [
    ("video_id", 25.0),
    ("author_id", 35.0),
    ("tag", 90.0),
    ("duration_bucket", 140.0),
    ("upload_type", 100.0),
    ("music_type", 120.0),
]

PAIR_FIELDS = [
    ("author_id", 18.0),
    ("tag", 25.0),
    ("duration_bucket", 35.0),
    ("upload_type", 35.0),
    ("music_type", 40.0),
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / max(float(x.std()), 1e-8)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, user_ids))
    su = user_ids[order]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    ends = np.r_[starts[1:], n]
    sizes = ends - starts
    pos = np.arange(n, dtype=np.float64) - np.repeat(starts, sizes)
    denom = np.maximum(np.repeat(sizes, sizes) - 1, 1)
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = pos / denom
    return ranked


def make_combined(a, b):
    class Combined:
        pass

    c = Combined()
    all_fields = sorted(set(CAT_FIELDS + NB_FIELDS +
                            [x[0] for x in ENTITY_FIELDS] +
                            [x[0] for x in PAIR_FIELDS]))
    c.X = {
        f: np.concatenate([
            np.asarray(a.X[f], dtype=np.int64),
            np.asarray(b.X[f], dtype=np.int64),
        ])
        for f in all_fields
    }
    c.num = {
        f: np.concatenate([
            np.asarray(a.num[f], dtype=np.float32),
            np.asarray(b.num[f], dtype=np.float32),
        ])
        for f in NUM_FIELDS
    }
    c.user_id = np.concatenate([
        np.asarray(a.user_id, dtype=np.int64),
        np.asarray(b.user_id, dtype=np.int64),
    ])
    c.date = np.concatenate([
        np.asarray(a.date, dtype=np.int32),
        np.asarray(b.date, dtype=np.int32),
    ])
    return c


def fit_statistics(split, y):
    y = np.asarray(y, dtype=np.float64)
    prior = float(y.mean())
    model = {
        "prior": prior,
        "entity": {},
        "pairs": {},
        "nb": {},
    }

    for field, smoothing in ENTITY_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        card = FEATURE_CARDINALITIES[field]
        cnt = np.bincount(ids, minlength=card).astype(np.float64)
        pos = np.bincount(ids, weights=y, minlength=card)
        rate = (pos + smoothing * prior) / (cnt + smoothing)
        model["entity"][field] = {
            "count": cnt,
            "pos": pos,
            "rate": rate.astype(np.float32),
            "smoothing": float(smoothing),
        }

    users = np.asarray(split.user_id, dtype=np.int64)
    for field, smoothing in PAIR_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[field])
        keys = users * card + ids
        uniq, inv, cnt = np.unique(
            keys, return_inverse=True, return_counts=True
        )
        pos = np.bincount(inv, weights=y, minlength=len(uniq))
        model["pairs"][field] = {
            "keys": uniq.astype(np.int64),
            "count": cnt.astype(np.float32),
            "pos": pos.astype(np.float32),
            "card": card,
            "smoothing": float(smoothing),
        }

    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    for field in NB_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        card = FEATURE_CARDINALITIES[field]
        c1 = np.bincount(ids, weights=y, minlength=card).astype(np.float64)
        ct = np.bincount(ids, minlength=card).astype(np.float64)
        c0 = ct - c1
        alpha = 2.0
        p1 = (c1 + alpha) / (n1 + alpha * card)
        p0 = (c0 + alpha) / (n0 + alpha * card)
        model["nb"][field] = np.log(p1 / p0).astype(np.float32)

    return model


def entity_rate(model, fit_split, y_fit, target, field, training):
    info = model["entity"][field]
    ids = np.asarray(target.X[field], dtype=np.int64)
    if not training:
        return info["rate"][ids].astype(np.float32)

    y_fit = np.asarray(y_fit, dtype=np.float64)
    cnt = info["count"][ids] - 1.0
    pos = info["pos"][ids] - y_fit
    a = info["smoothing"]
    return ((pos + a * model["prior"]) / (cnt + a)).astype(np.float32)


def pair_rate(model, y_fit, target, field, training):
    info = model["pairs"][field]
    ids = np.asarray(target.X[field], dtype=np.int64)
    users = np.asarray(target.user_id, dtype=np.int64)
    keys = users * info["card"] + ids

    loc = np.searchsorted(info["keys"], keys)
    ok = loc < len(info["keys"])
    safe = np.minimum(loc, len(info["keys"]) - 1)
    ok &= info["keys"][safe] == keys

    cnt = np.zeros(len(keys), dtype=np.float64)
    pos = np.zeros(len(keys), dtype=np.float64)
    cnt[ok] = info["count"][safe[ok]]
    pos[ok] = info["pos"][safe[ok]]

    if training:
        y_fit = np.asarray(y_fit, dtype=np.float64)
        cnt -= 1.0
        pos -= y_fit

    a = info["smoothing"]
    return ((pos + a * model["prior"]) / (cnt + a)).astype(np.float32)


def predict_empirical(model, fit_split, y_fit, target, training=False):
    coefficients = {
        "video_id": 0.55,
        "author_id": 0.32,
        "tag": 0.16,
        "duration_bucket": 0.13,
        "upload_type": 0.08,
        "music_type": 0.06,
    }
    pair_coefficients = {
        "author_id": 0.75,
        "tag": 0.48,
        "duration_bucket": 0.30,
        "upload_type": 0.22,
        "music_type": 0.16,
    }

    score = np.zeros(len(target.user_id), dtype=np.float64)
    prior_logit = float(logit(model["prior"]))

    for field, _ in ENTITY_FIELDS:
        r = entity_rate(
            model, fit_split, y_fit, target, field, training
        )
        score += coefficients[field] * (logit(r) - prior_logit)

    for field, _ in PAIR_FIELDS:
        r = pair_rate(model, y_fit, target, field, training)
        score += pair_coefficients[field] * (logit(r) - prior_logit)

    return score.astype(np.float32)


def predict_naive_bayes(model, target):
    score = np.full(
        len(target.user_id),
        float(logit(model["prior"])),
        dtype=np.float64,
    )
    for field in NB_FIELDS:
        ids = np.asarray(target.X[field], dtype=np.int64)
        score += model["nb"][field][ids]
    return score.astype(np.float32)


def make_lgb_features(model, fit_split, y_fit, target, training):
    columns = []

    for field in CAT_FIELDS:
        columns.append(np.asarray(target.X[field], dtype=np.float32))

    for field in NUM_FIELDS:
        v = np.asarray(target.num[field], dtype=np.float64)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(v, 0.0)).astype(np.float32))

    max_date = int(np.max(fit_split.date))
    age = max_date - np.asarray(target.date, dtype=np.int64)
    columns.append(age.astype(np.float32))

    for field, _ in ENTITY_FIELDS:
        rate = entity_rate(
            model, fit_split, y_fit, target, field, training
        )
        columns.append(rate)

    for field, _ in PAIR_FIELDS:
        rate = pair_rate(model, y_fit, target, field, training)
        columns.append(rate)

    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def recency_weight(dates, half_life=4.0):
    d = np.asarray(dates, dtype=np.int64)
    age = int(d.max()) - d
    w = np.exp2(-age.astype(np.float32) / float(half_life))
    w /= max(float(w.mean()), 1e-8)
    return w.astype(np.float32)


def fit_lgb_binary(X, y, dates):
    cat_idx = list(range(len(CAT_FIELDS)))
    ds = lgb.Dataset(
        X,
        label=np.asarray(y, dtype=np.float32),
        weight=recency_weight(dates, 4.0),
        categorical_feature=cat_idx,
        free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.06,
        "num_leaves": 63,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.78,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "max_cat_threshold": 32,
        "num_threads": max(1, min(8, os.cpu_count() or 1)),
        "seed": SEED,
        "verbose": -1,
    }
    return lgb.train(params, ds, num_boost_round=220)


def fit_lgb_ranker(X, y, users, dates):
    users = np.asarray(users, dtype=np.int64)
    row = np.arange(len(users), dtype=np.int64)
    order = np.lexsort((row, users))
    sorted_users = users[order]
    _, group = np.unique(sorted_users, return_counts=True)

    ds = lgb.Dataset(
        X[order],
        label=np.asarray(y, dtype=np.float32)[order],
        weight=recency_weight(dates, 4.0)[order],
        group=group.astype(np.int32),
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "learning_rate": 0.055,
        "num_leaves": 63,
        "min_data_in_leaf": 250,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.88,
        "bagging_freq": 1,
        "lambda_l2": 2.5,
        "max_bin": 127,
        "max_cat_threshold": 32,
        "lambdarank_truncation_level": 10,
        "num_threads": max(1, min(8, os.cpu_count() or 1)),
        "seed": SEED + 17,
        "verbose": -1,
    }
    return lgb.train(params, ds, num_boost_round=180)


def select_candidates(valid, y_valid, incumbent, predictions):
    candidates = {}
    incumbent = np.asarray(incumbent, dtype=np.float64)
    inc_z = zscore(incumbent)
    inc_rank = within_user_rank(valid.user_id, incumbent)

    inc_metric = evaluate(valid.user_id, y_valid, incumbent)
    best_primary = float(inc_metric["primary"])
    best_scores = incumbent.copy()
    best_desc = ("trusted_incumbent", "standalone", 0.0)
    candidates["trusted_incumbent"] = best_primary

    alphas = [0.15, 0.25, 0.35, 0.45, 0.55, 0.70, 0.85]

    for name, pred in predictions.items():
        pred = np.asarray(pred, dtype=np.float64)
        standalone = float(
            evaluate(valid.user_id, y_valid, pred)["primary"]
        )
        candidates[name] = standalone
        if standalone > best_primary:
            best_primary = standalone
            best_scores = pred.copy()
            best_desc = (name, "standalone", 1.0)

        pz = zscore(pred)
        pr = within_user_rank(valid.user_id, pred)

        local_raw = (-np.inf, None)
        local_rank = (-np.inf, None)
        local_raw_scores = None
        local_rank_scores = None

        for alpha in alphas:
            raw = alpha * pz + (1.0 - alpha) * inc_z
            value = float(
                evaluate(valid.user_id, y_valid, raw)["primary"]
            )
            if value > local_raw[0]:
                local_raw = (value, alpha)
                local_raw_scores = raw

            ranked = alpha * pr + (1.0 - alpha) * inc_rank
            value = float(
                evaluate(valid.user_id, y_valid, ranked)["primary"]
            )
            if value > local_rank[0]:
                local_rank = (value, alpha)
                local_rank_scores = ranked

        candidates[name + "_rawblend"] = float(local_raw[0])
        candidates[name + "_rankblend"] = float(local_rank[0])

        if local_raw[0] > best_primary:
            best_primary = float(local_raw[0])
            best_scores = np.asarray(local_raw_scores, dtype=np.float64)
            best_desc = (name, "rawblend", float(local_raw[1]))

        if local_rank[0] > best_primary:
            best_primary = float(local_rank[0])
            best_scores = np.asarray(local_rank_scores, dtype=np.float64)
            best_desc = (name, "rankblend", float(local_rank[1]))

    return best_scores, best_desc, candidates


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

    stats = fit_statistics(train, y_train)

    predictions = {}
    predictions["empirical_bayes_pairs"] = predict_empirical(
        stats, train, y_train, valid, training=False
    )
    predictions["naive_bayes_generative"] = predict_naive_bayes(
        stats, valid
    )

    X_train = make_lgb_features(
        stats, train, y_train, train, training=True
    )
    X_valid = make_lgb_features(
        stats, train, y_train, valid, training=False
    )

    binary_model = fit_lgb_binary(
        X_train, y_train, train.date
    )
    predictions["lightgbm_binary_recency"] = binary_model.predict(
        X_valid
    ).astype(np.float32)

    rank_model = fit_lgb_ranker(
        X_train, y_train, train.user_id, train.date
    )
    predictions["lightgbm_lambdarank_recency"] = rank_model.predict(
        X_valid
    ).astype(np.float32)

    best_valid, descriptor, candidates = select_candidates(
        valid, y_valid, incumbent_valid, predictions
    )
    final_metrics = evaluate(valid.user_id, y_valid, best_valid)

    out = os.environ.get("ITER_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        np.save(
            os.path.join(out, "scores_valid.npy"),
            np.asarray(best_valid, dtype=np.float64),
        )

    winner, blend_mode, alpha = descriptor

    test = load("test")
    incumbent_test = np.load(
        os.path.join(shared, "incumbent_test_scores.npy")
    ).astype(np.float64)

    if winner == "trusted_incumbent":
        test_scores = incumbent_test
    else:
        combined = make_combined(train, valid)
        y_combined = np.concatenate([
            y_train,
            y_valid.astype(np.float32),
        ])
        combined_stats = fit_statistics(combined, y_combined)

        if winner == "empirical_bayes_pairs":
            new_test = predict_empirical(
                combined_stats,
                combined,
                y_combined,
                test,
                training=False,
            )
        elif winner == "naive_bayes_generative":
            new_test = predict_naive_bayes(combined_stats, test)
        else:
            del binary_model, rank_model, X_train, X_valid
            gc.collect()

            X_combined = make_lgb_features(
                combined_stats,
                combined,
                y_combined,
                combined,
                training=True,
            )
            X_test = make_lgb_features(
                combined_stats,
                combined,
                y_combined,
                test,
                training=False,
            )

            if winner == "lightgbm_binary_recency":
                refit = fit_lgb_binary(
                    X_combined, y_combined, combined.date
                )
            else:
                refit = fit_lgb_ranker(
                    X_combined,
                    y_combined,
                    combined.user_id,
                    combined.date,
                )
            new_test = refit.predict(X_test).astype(np.float32)

        test_scores = apply_blend(
            blend_mode,
            alpha,
            new_test,
            incumbent_test,
            test.user_id,
        )

    if out:
        np.save(
            os.path.join(out, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

    print(
        "FINDINGS "
        + json.dumps(
            {
                "winner": winner,
                "blend_mode": blend_mode,
                "new_family_weight": float(alpha),
            },
            separators=(",", ":"),
        )
    )
    print(
        "CANDIDATES "
        + json.dumps(
            candidates,
            sort_keys=True,
            separators=(",", ":"),
        )
    )

    payload = {
        "primary": float(final_metrics["primary"]),
        "gauc": float(final_metrics["gauc"]),
        "ndcg@5": float(final_metrics["ndcg@5"]),
        "gpu_seconds": float(time.time() - START),
    }
    print("METRICS " + json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()