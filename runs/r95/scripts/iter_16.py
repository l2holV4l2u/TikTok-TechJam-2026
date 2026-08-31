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
yva = np.asarray(valid.y, dtype=np.int8)

utr = np.asarray(train.user_id, dtype=np.int64)
uva = np.asarray(valid.user_id, dtype=np.int64)
ute = np.asarray(test.user_id, dtype=np.int64)

train_dates = np.asarray(train.date, dtype=np.int64)
last_date = int(train_dates.max())
unique_dates = np.unique(train_dates)
date_index = {int(d): i for i, d in enumerate(unique_dates)}
train_day = np.fromiter(
    (date_index[int(d)] for d in train_dates),
    dtype=np.int64,
    count=len(train_dates),
)
age = (len(unique_dates) - 1 - train_day).astype(np.float64)

# Recent observations matter more across the date split, but an eight-day
# half-life retains enough history for sparse users and user-attribute cells.
w4 = np.exp2(-age / 4.0)
w8 = np.exp2(-age / 8.0)

user_card = int(FEATURE_CARDINALITIES["user_id"])
user_count = np.bincount(utr, minlength=user_card).astype(np.float64)
user_recent_count = np.bincount(
    utr, weights=w8, minlength=user_card
).astype(np.float64)


def safe_user_lookup(values, users):
    users = np.asarray(users, dtype=np.int64)
    result = np.zeros(len(users), dtype=np.float64)
    good = (users >= 0) & (users < len(values))
    result[good] = values[users[good]]
    return result


valid_count = safe_user_lookup(user_count, uva)
test_count = safe_user_lookup(user_count, ute)
valid_recent_count = safe_user_lookup(user_recent_count, uva)
test_recent_count = safe_user_lookup(user_recent_count, ute)


def clip_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1.0e-4, 1.0 - 1.0e-4)
    return np.log(p) - np.log1p(-p)


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    row = np.arange(len(scores), dtype=np.int64)

    # Ascending score gives rank zero to the lowest-scored row. Row position is
    # a deterministic tie breaker matching the logged row order.
    order = np.lexsort((row, scores, users))
    sorted_users = users[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], len(order)]
    lengths = ends - starts

    positions = (
        np.arange(len(order), dtype=np.float64)
        - np.repeat(starts, lengths)
    )
    denominators = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    ranked_sorted = positions / denominators

    result = np.empty(len(scores), dtype=np.float64)
    result[order] = ranked_sorted
    return result


# -------------------------------------------------------------------------
# Family A: stationary, recency-weighted empirical-Bayes content model.
# It excludes user identity and therefore remains defined for sparse users.
# -------------------------------------------------------------------------

CONTENT_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "tab",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

CONTENT_WEIGHTS = {
    "video_id": 1.4,
    "author_id": 1.2,
    "tag": 1.0,
    "duration_bucket": 1.0,
    "upload_type": 0.7,
    "tab": 0.8,
    "onehot_feat3": 0.6,
    "onehot_feat7": 0.5,
    "onehot_feat8": 0.7,
}


def empirical_bayes_content(sample_weight, strength):
    prior = float(np.sum(sample_weight * ytr) / np.sum(sample_weight))
    va = np.zeros(len(uva), dtype=np.float64)
    te = np.zeros(len(ute), dtype=np.float64)
    total_weight = 0.0

    for field in CONTENT_FIELDS:
        cardinality = int(FEATURE_CARDINALITIES[field])
        tr_ids = np.asarray(train.X[field], dtype=np.int64)

        count = np.bincount(
            tr_ids, weights=sample_weight, minlength=cardinality
        )
        positive = np.bincount(
            tr_ids, weights=sample_weight * ytr, minlength=cardinality
        )
        rate = (positive + strength * prior) / (count + strength)
        effect = clip_logit(rate) - clip_logit(prior)

        importance = CONTENT_WEIGHTS[field]
        va += importance * effect[
            np.asarray(valid.X[field], dtype=np.int64)
        ]
        te += importance * effect[
            np.asarray(test.X[field], dtype=np.int64)
        ]
        total_weight += importance

    return va / total_weight, te / total_weight


eb_valid, eb_test = empirical_bayes_content(w4, strength=20.0)


# -------------------------------------------------------------------------
# Family B: personalized user-by-content empirical affinity.
# This forms predictions through explicit user preference tables rather than
# entity popularity. Shrinkage makes missing cells exactly neutral.
# -------------------------------------------------------------------------

AFFINITY_FIELDS = [
    "tag",
    "duration_bucket",
    "upload_type",
    "tab",
    "onehot_feat2",
    "onehot_feat7",
]


def user_affinity(sample_weight, strength=5.0):
    global_prior = float(
        np.sum(sample_weight * ytr) / np.sum(sample_weight)
    )
    ucnt = np.bincount(
        utr, weights=sample_weight, minlength=user_card
    )
    upos = np.bincount(
        utr, weights=sample_weight * ytr, minlength=user_card
    )
    user_prior = (upos + 18.0 * global_prior) / (ucnt + 18.0)

    va = np.zeros(len(uva), dtype=np.float64)
    te = np.zeros(len(ute), dtype=np.float64)

    for field in AFFINITY_FIELDS:
        cardinality = int(FEATURE_CARDINALITIES[field])
        tr_cat = np.asarray(train.X[field], dtype=np.int64)
        train_key = utr * cardinality + tr_cat
        size = user_card * cardinality

        cell_count = np.bincount(
            train_key, weights=sample_weight, minlength=size
        )
        cell_positive = np.bincount(
            train_key,
            weights=sample_weight * ytr,
            minlength=size,
        )

        repeated_prior = np.repeat(user_prior, cardinality)
        rate = (
            cell_positive + strength * repeated_prior
        ) / (cell_count + strength)
        effect = clip_logit(rate) - clip_logit(repeated_prior)

        va_key = (
            np.asarray(valid.user_id, dtype=np.int64) * cardinality
            + np.asarray(valid.X[field], dtype=np.int64)
        )
        te_key = (
            np.asarray(test.user_id, dtype=np.int64) * cardinality
            + np.asarray(test.X[field], dtype=np.int64)
        )

        va += effect[va_key]
        te += effect[te_key]

        del cell_count, cell_positive, repeated_prior, rate, effect

    return va / len(AFFINITY_FIELDS), te / len(AFFINITY_FIELDS)


aff_valid, aff_test = user_affinity(w8, strength=5.0)


# -------------------------------------------------------------------------
# Family C: reliability-controlled personalized content predictor.
# Personalized affinity is admitted only as effective user history grows.
# -------------------------------------------------------------------------

warm_valid = valid_recent_count / (valid_recent_count + 20.0)
warm_test = test_recent_count / (test_recent_count + 20.0)

personalized_valid = eb_valid + warm_valid * aff_valid
personalized_test = eb_test + warm_test * aff_test

eb_rank_valid = within_user_rank(uva, eb_valid)
eb_rank_test = within_user_rank(ute, eb_test)
aff_rank_valid = within_user_rank(uva, aff_valid)
aff_rank_test = within_user_rank(ute, aff_test)
personalized_rank_valid = within_user_rank(uva, personalized_valid)
personalized_rank_test = within_user_rank(ute, personalized_test)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(shared, "incumbent_test_scores.npy")),
    dtype=np.float64,
)
inc_rank_valid = within_user_rank(uva, inc_valid)
inc_rank_test = within_user_rank(ute, inc_test)

families_valid = {
    "empirical_bayes": eb_rank_valid,
    "user_affinity": aff_rank_valid,
    "reliability_personalized": personalized_rank_valid,
}
families_test = {
    "empirical_bayes": eb_rank_test,
    "user_affinity": aff_rank_test,
    "reliability_personalized": personalized_rank_test,
}

candidate_scores = {}
best_primary = -np.inf
best_metrics = None
best_valid = None
best_test = None
best_raw = None
best_name = None

# First compare each structurally distinct ranker by itself and by ordinary
# rank blending with the incumbent.
for name in families_valid:
    own_va = families_valid[name]
    own_te = families_test[name]

    standalone = evaluate(uva, yva, own_va)
    candidate_scores[name + "_standalone"] = float(
        standalone["primary"]
    )

    for alpha in [0.10, 0.20, 0.30, 0.40]:
        va_score = (1.0 - alpha) * inc_rank_valid + alpha * own_va
        te_score = (1.0 - alpha) * inc_rank_test + alpha * own_te
        metrics = evaluate(uva, yva, va_score)
        primary = float(metrics["primary"])
        cname = "%s_global_%.2f" % (name, alpha)
        candidate_scores[cname] = primary

        if primary > best_primary:
            best_primary = primary
            best_metrics = metrics
            best_valid = va_score.copy()
            best_test = te_score.copy()
            best_raw = own_va.copy()
            best_name = cname

# Cold-start gates put more weight on the stationary content rank as training
# history falls. A gate is constant within a user, but it changes the relative
# contribution of the two rankers for that user's slate.
for tau in [5.0, 15.0, 40.0, 100.0]:
    cold_va = tau / (valid_count + tau)
    cold_te = tau / (test_count + tau)

    for max_alpha in [0.20, 0.40, 0.60, 0.80]:
        alpha_va = max_alpha * cold_va
        alpha_te = max_alpha * cold_te

        va_score = (
            (1.0 - alpha_va) * inc_rank_valid
            + alpha_va * eb_rank_valid
        )
        te_score = (
            (1.0 - alpha_te) * inc_rank_test
            + alpha_te * eb_rank_test
        )
        metrics = evaluate(uva, yva, va_score)
        primary = float(metrics["primary"])
        cname = "cold_eb_tau%d_max%.2f" % (int(tau), max_alpha)
        candidate_scores[cname] = primary

        if primary > best_primary:
            best_primary = primary
            best_metrics = metrics
            best_valid = va_score.copy()
            best_test = te_score.copy()
            best_raw = eb_rank_valid.copy()
            best_name = cname

# A three-way reliability gate: stationary EB dominates at low history,
# incumbent dominates in the middle, and explicit affinity receives a small
# contribution only for users whose preference cells are estimable.
for tau in [15.0, 40.0, 100.0]:
    cold_va = tau / (valid_count + tau)
    cold_te = tau / (test_count + tau)
    warm_va_gate = valid_count / (valid_count + tau)
    warm_te_gate = test_count / (test_count + tau)

    for cold_scale, warm_scale in [
        (0.40, 0.10),
        (0.60, 0.10),
        (0.60, 0.20),
        (0.80, 0.15),
    ]:
        a_cold_va = cold_scale * cold_va
        a_cold_te = cold_scale * cold_te
        a_warm_va = warm_scale * warm_va_gate
        a_warm_te = warm_scale * warm_te_gate

        va_score = (
            (1.0 - a_cold_va - a_warm_va) * inc_rank_valid
            + a_cold_va * eb_rank_valid
            + a_warm_va * aff_rank_valid
        )
        te_score = (
            (1.0 - a_cold_te - a_warm_te) * inc_rank_test
            + a_cold_te * eb_rank_test
            + a_warm_te * aff_rank_test
        )
        metrics = evaluate(uva, yva, va_score)
        primary = float(metrics["primary"])
        cname = "threeway_tau%d_c%.2f_w%.2f" % (
            int(tau), cold_scale, warm_scale
        )
        candidate_scores[cname] = primary

        if primary > best_primary:
            best_primary = primary
            best_metrics = metrics
            best_valid = va_score.copy()
            best_test = te_score.copy()
            best_raw = personalized_rank_valid.copy()
            best_name = cname

quantiles = np.quantile(
    valid_count, [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
)
unseen_rows = float(np.mean(valid_count == 0))
sparse_rows = float(np.mean(valid_count <= 15))
print(
    "FINDINGS " + json.dumps(
        {
            "valid_train_history_quantiles": [
                float(x) for x in quantiles
            ],
            "valid_unseen_user_row_fraction": unseen_rows,
            "valid_history_le_15_row_fraction": sparse_rows,
            "winner": best_name,
        },
        sort_keys=True,
    )
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

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

elapsed = float(time.time() - START)
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": elapsed,
        }
    )
)