import os
import time
import json
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.float64)
utr = np.asarray(train.user_id, dtype=np.int64)
uva = np.asarray(valid.user_id, dtype=np.int64)
ute = np.asarray(test.user_id, dtype=np.int64)
yva = np.asarray(valid.y, dtype=np.int8)

train_dates = np.asarray(train.date, dtype=np.int64)
valid_dates = np.asarray(valid.date, dtype=np.int64)
test_dates = np.asarray(test.date, dtype=np.int64)

last_train_date = int(train_dates.max())
unique_dates = np.unique(train_dates)
date_to_index = {int(d): i for i, d in enumerate(unique_dates)}
train_day = np.fromiter(
    (date_to_index[int(d)] for d in train_dates),
    dtype=np.int64,
    count=len(train_dates),
)
n_days = len(unique_dates)


def future_day_index(dates):
    # Calendar dates in this benchmark are consecutive. Convert YYYYMMDD using
    # numpy datetime64 so the extrapolation also remains valid across months.
    base = np.datetime64(
        "%04d-%02d-%02d" %
        (last_train_date // 10000, (last_train_date // 100) % 100,
         last_train_date % 100)
    )
    strings = np.asarray([
        "%04d-%02d-%02d" % (int(d) // 10000, (int(d) // 100) % 100, int(d) % 100)
        for d in np.asarray(dates)
    ])
    delta = (
        strings.astype("datetime64[D]") - base
    ).astype(np.int64)
    return (n_days - 1 + delta).astype(np.float64)


valid_future_day = future_day_index(valid_dates)
test_future_day = future_day_index(test_dates)

age = (n_days - 1 - train_day).astype(np.float64)
w4 = np.exp2(-age / 4.0)
w8 = np.exp2(-age / 8.0)


def clip_logit(p):
    p = np.clip(p, 1.0e-4, 1.0 - 1.0e-4)
    return np.log(p) - np.log1p(-p)


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    # Row index is a deterministic final tie breaker.
    row = np.arange(len(scores), dtype=np.int64)
    order = np.lexsort((row, scores, users))
    su = users[order]
    n = len(order)

    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    position = np.arange(n, dtype=np.float64) - np.repeat(starts, lengths)
    denom = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    ranked_sorted = position / denom

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


# ---------------------------------------------------------------------------
# Family 1: temporally decayed empirical-Bayes marginal relevance.
# ---------------------------------------------------------------------------

MARGINAL_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "tab",
    "hour",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

MARGINAL_IMPORTANCE = {
    "video_id": 1.4,
    "author_id": 1.2,
    "tag": 0.9,
    "duration_bucket": 1.0,
    "upload_type": 0.7,
    "tab": 0.8,
    "hour": 0.4,
    "onehot_feat3": 0.6,
    "onehot_feat7": 0.5,
    "onehot_feat8": 0.6,
}


def empirical_bayes_scores(sample_weight, strength):
    prior = float(np.sum(sample_weight * ytr) / np.sum(sample_weight))
    va_score = np.zeros(len(valid_dates), dtype=np.float64)
    te_score = np.zeros(len(test_dates), dtype=np.float64)
    total_importance = 0.0

    for field in MARGINAL_FIELDS:
        k = int(FEATURE_CARDINALITIES[field])
        ids = np.asarray(train.X[field], dtype=np.int64)
        cnt = np.bincount(ids, weights=sample_weight, minlength=k)
        pos = np.bincount(
            ids, weights=sample_weight * ytr, minlength=k
        )
        rate = (pos + strength * prior) / (cnt + strength)
        effect = clip_logit(rate) - clip_logit(prior)

        importance = MARGINAL_IMPORTANCE[field]
        va_score += importance * effect[
            np.asarray(valid.X[field], dtype=np.int64)
        ]
        te_score += importance * effect[
            np.asarray(test.X[field], dtype=np.int64)
        ]
        total_importance += importance

    return va_score / total_importance, te_score / total_importance


eb4_valid, eb4_test = empirical_bayes_scores(w4, strength=18.0)
eb8_valid, eb8_test = empirical_bayes_scores(w8, strength=25.0)


# ---------------------------------------------------------------------------
# Family 2: a generative categorical Naive Bayes model. Unlike empirical rate
# averaging, it estimates class-conditional feature likelihood ratios.
# ---------------------------------------------------------------------------

NB_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "tab",
    "hour",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]


def naive_bayes_scores(sample_weight, alpha=5.0):
    pos_total = float(np.sum(sample_weight * ytr))
    neg_total = float(np.sum(sample_weight * (1.0 - ytr)))
    prior_log_odds = np.log(pos_total / neg_total)

    va_score = np.full(len(valid_dates), prior_log_odds, dtype=np.float64)
    te_score = np.full(len(test_dates), prior_log_odds, dtype=np.float64)

    # Divide contributions to temper the conditional-independence assumption;
    # positive scaling does not change standalone order but stabilizes blending.
    scale = 1.0 / np.sqrt(len(NB_FIELDS))

    for field in NB_FIELDS:
        k = int(FEATURE_CARDINALITIES[field])
        ids = np.asarray(train.X[field], dtype=np.int64)

        pos = np.bincount(
            ids, weights=sample_weight * ytr, minlength=k
        )
        neg = np.bincount(
            ids, weights=sample_weight * (1.0 - ytr), minlength=k
        )

        log_ratio = (
            np.log(pos + alpha)
            - np.log(pos_total + alpha * k)
            - np.log(neg + alpha)
            + np.log(neg_total + alpha * k)
        )

        va_score += scale * log_ratio[
            np.asarray(valid.X[field], dtype=np.int64)
        ]
        te_score += scale * log_ratio[
            np.asarray(test.X[field], dtype=np.int64)
        ]

    return va_score, te_score


nb_valid, nb_test = naive_bayes_scores(w4, alpha=6.0)


# ---------------------------------------------------------------------------
# Family 3: entity-level temporal extrapolation. A regularized weighted linear
# trend is estimated directly from all binary rows for each entity. Prediction
# uses each evaluation row's future date, rather than assuming stationarity.
# ---------------------------------------------------------------------------

TREND_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "tab",
]


def temporal_trend_scores():
    t = train_day.astype(np.float64)
    va_score = np.zeros(len(valid_dates), dtype=np.float64)
    te_score = np.zeros(len(test_dates), dtype=np.float64)

    for field in TREND_FIELDS:
        k = int(FEATURE_CARDINALITIES[field])
        ids = np.asarray(train.X[field], dtype=np.int64)

        n = np.bincount(ids, minlength=k).astype(np.float64)
        sy = np.bincount(ids, weights=ytr, minlength=k)
        st = np.bincount(ids, weights=t, minlength=k)
        st2 = np.bincount(ids, weights=t * t, minlength=k)
        sty = np.bincount(ids, weights=t * ytr, minlength=k)

        safe_n = np.maximum(n, 1.0)
        mean_t = st / safe_n
        mean_y = sy / safe_n

        covariance_numerator = sty - st * sy / safe_n
        variance_numerator = st2 - st * st / safe_n

        # The ridge term strongly shrinks noisy trends for rare entities.
        slope = covariance_numerator / (variance_numerator + 80.0)
        slope = np.clip(slope, -0.025, 0.025)

        # Smooth the entity intercept toward the global recent rate.
        recent_prior = float(np.sum(w4 * ytr) / np.sum(w4))
        base = (sy + 20.0 * recent_prior) / (n + 20.0)

        va_ids = np.asarray(valid.X[field], dtype=np.int64)
        te_ids = np.asarray(test.X[field], dtype=np.int64)

        va_prob = (
            base[va_ids]
            + slope[va_ids] * (valid_future_day - mean_t[va_ids])
        )
        te_prob = (
            base[te_ids]
            + slope[te_ids] * (test_future_day - mean_t[te_ids])
        )

        va_score += clip_logit(np.clip(va_prob, 0.01, 0.99))
        te_score += clip_logit(np.clip(te_prob, 0.01, 0.99))

    va_score /= len(TREND_FIELDS)
    te_score /= len(TREND_FIELDS)
    return va_score, te_score


trend_valid, trend_test = temporal_trend_scores()


# ---------------------------------------------------------------------------
# Family 4: personalized user-content empirical affinity. Each user's train
# history estimates preference for low-cardinality content attributes. This
# forms user x content interactions without learning identity embeddings.
# ---------------------------------------------------------------------------

AFFINITY_FIELDS = [
    "tag",
    "duration_bucket",
    "upload_type",
    "tab",
    "onehot_feat2",
    "onehot_feat7",
]

user_card = int(FEATURE_CARDINALITIES["user_id"])


def user_affinity_scores(sample_weight, strength=4.0):
    user_cnt = np.bincount(
        utr, weights=sample_weight, minlength=user_card
    )
    user_pos = np.bincount(
        utr, weights=sample_weight * ytr, minlength=user_card
    )
    global_prior = float(
        np.sum(sample_weight * ytr) / np.sum(sample_weight)
    )
    user_prior = (user_pos + 15.0 * global_prior) / (user_cnt + 15.0)

    va_score = np.zeros(len(valid_dates), dtype=np.float64)
    te_score = np.zeros(len(test_dates), dtype=np.float64)

    for field in AFFINITY_FIELDS:
        k = int(FEATURE_CARDINALITIES[field])
        tr_cat = np.asarray(train.X[field], dtype=np.int64)
        key = utr * k + tr_cat
        size = user_card * k

        cnt = np.bincount(
            key, weights=sample_weight, minlength=size
        )
        pos = np.bincount(
            key, weights=sample_weight * ytr, minlength=size
        )

        prior_flat = np.repeat(user_prior, k)
        rate = (pos + strength * prior_flat) / (cnt + strength)
        effect = clip_logit(rate) - clip_logit(prior_flat)

        va_key = (
            np.asarray(valid.user_id, dtype=np.int64) * k
            + np.asarray(valid.X[field], dtype=np.int64)
        )
        te_key = (
            np.asarray(test.user_id, dtype=np.int64) * k
            + np.asarray(test.X[field], dtype=np.int64)
        )

        va_score += effect[va_key]
        te_score += effect[te_key]

        del cnt, pos, rate, effect, prior_flat

    return va_score / len(AFFINITY_FIELDS), te_score / len(AFFINITY_FIELDS)


affinity_valid, affinity_test = user_affinity_scores(w8, strength=4.0)


# ---------------------------------------------------------------------------
# Rank aggregation with the trusted incumbent. The benchmark only depends on
# within-user order, so ordinal ranks avoid arbitrary cross-family scales.
# ---------------------------------------------------------------------------

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(shared, "incumbent_test_scores.npy")),
    dtype=np.float64,
)

families_valid = {
    "eb_half4": eb4_valid,
    "eb_half8": eb8_valid,
    "naive_bayes": nb_valid,
    "temporal_trend": trend_valid,
    "user_affinity": affinity_valid,
}
families_test = {
    "eb_half4": eb4_test,
    "eb_half8": eb8_test,
    "naive_bayes": nb_test,
    "temporal_trend": trend_test,
    "user_affinity": affinity_test,
}

rank_valid = {
    name: within_user_rank(uva, score)
    for name, score in families_valid.items()
}
rank_test = {
    name: within_user_rank(ute, score)
    for name, score in families_test.items()
}

# Cross-family Borda ensembles are themselves an aggregation family and can
# exploit complementary temporal, marginal, generative, and personalized order.
rank_valid["content_ensemble"] = np.mean(
    np.stack([
        rank_valid["eb_half4"],
        rank_valid["naive_bayes"],
        rank_valid["temporal_trend"],
    ], axis=0),
    axis=0,
)
rank_test["content_ensemble"] = np.mean(
    np.stack([
        rank_test["eb_half4"],
        rank_test["naive_bayes"],
        rank_test["temporal_trend"],
    ], axis=0),
    axis=0,
)

rank_valid["all_ensemble"] = np.mean(
    np.stack([
        rank_valid["eb_half4"],
        rank_valid["naive_bayes"],
        rank_valid["temporal_trend"],
        rank_valid["user_affinity"],
    ], axis=0),
    axis=0,
)
rank_test["all_ensemble"] = np.mean(
    np.stack([
        rank_test["eb_half4"],
        rank_test["naive_bayes"],
        rank_test["temporal_trend"],
        rank_test["user_affinity"],
    ], axis=0),
    axis=0,
)

inc_rank_valid = within_user_rank(uva, inc_valid)
inc_rank_test = within_user_rank(ute, inc_test)

candidate_scores = {}
best_primary = -np.inf
best_metrics = None
best_valid = None
best_test = None
best_raw = None
best_name = None

alphas = [0.10, 0.20, 0.30, 0.40, 0.60, 1.00]

for name in rank_valid:
    own_va = rank_valid[name]
    own_te = rank_test[name]

    standalone_metrics = evaluate(uva, yva, own_va)
    candidate_scores[name + "_standalone"] = float(
        standalone_metrics["primary"]
    )

    for alpha in alphas:
        va_score = (1.0 - alpha) * inc_rank_valid + alpha * own_va
        te_score = (1.0 - alpha) * inc_rank_test + alpha * own_te
        metrics = evaluate(uva, yva, va_score)
        primary = float(metrics["primary"])
        cname = name + "_blend_%.2f" % alpha
        candidate_scores[cname] = primary

        if primary > best_primary:
            best_primary = primary
            best_metrics = metrics
            best_valid = va_score.copy()
            best_test = te_score.copy()
            best_raw = own_va.copy()
            best_name = cname

print(
    "FINDINGS selected=%s families=%d" %
    (best_name, len(rank_valid)),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_scores, sort_keys=True),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.6f}' %
    (
        float(best_metrics["primary"]),
        float(best_metrics["gauc"]),
        float(best_metrics["ndcg@5"]),
        elapsed,
    )
)