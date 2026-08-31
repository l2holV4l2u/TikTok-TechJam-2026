import os
import time
import json
import math
import gc

import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
EPS = 1e-6

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
train_dates = np.asarray(train.date, dtype=np.int32)
train_day = (train_dates % 100).astype(np.float32)

# Recency weights are determined entirely from the training split.
age = np.max(train_day) - train_day
recency_weight = np.exp(-math.log(2.0) * age / 5.0).astype(np.float32)
recency_weight /= np.mean(recency_weight)

weighted_total = float(np.sum(recency_weight))
weighted_positive_total = float(np.sum(recency_weight * y_train))
global_rate = weighted_positive_total / weighted_total


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float32), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def query_sparse(sorted_keys, values, query_keys):
    """Vectorized lookup in a sorted sparse key/value table."""
    query_keys = np.asarray(query_keys, dtype=np.int64)
    idx = np.searchsorted(sorted_keys, query_keys)
    clipped = np.minimum(idx, len(sorted_keys) - 1)
    found = (idx < len(sorted_keys)) & (
        sorted_keys[clipped] == query_keys
    )
    result = np.zeros(len(query_keys), dtype=np.float32)
    result[found] = values[clipped[found]]
    return result


def sparse_pair_statistics(user, category, cardinality):
    """
    Recency-weighted sufficient statistics for observed (user, category)
    pairs. The representation contains only observed pairs, avoiding a
    dense user-by-category table.
    """
    keys = (
        np.asarray(user, dtype=np.int64) * np.int64(cardinality)
        + np.asarray(category, dtype=np.int64)
    )
    unique_keys, inverse = np.unique(keys, return_inverse=True)

    count = np.bincount(
        inverse,
        weights=recency_weight,
        minlength=len(unique_keys),
    ).astype(np.float32)
    positive = np.bincount(
        inverse,
        weights=recency_weight * y_train,
        minlength=len(unique_keys),
    ).astype(np.float32)

    del keys, inverse
    return unique_keys, count, positive


# The fields deliberately mix coarse content, presentation context, and
# identities. Stronger smoothing is used for high-cardinality sparse pairs.
PAIR_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat8",
    "video_type",
]

PAIR_STRENGTH = {
    "video_id": 8.0,
    "author_id": 12.0,
    "tag": 18.0,
    "tab": 28.0,
    "duration_bucket": 30.0,
    "upload_type": 25.0,
    "music_type": 28.0,
    "onehot_feat3": 14.0,
    "onehot_feat8": 16.0,
    "video_type": 35.0,
}

PAIR_WEIGHT = {
    "video_id": 1.15,
    "author_id": 0.95,
    "tag": 0.65,
    "tab": 0.85,
    "duration_bucket": 0.45,
    "upload_type": 0.35,
    "music_type": 0.25,
    "onehot_feat3": 0.55,
    "onehot_feat8": 0.45,
    "video_type": 0.20,
}

NB_STRENGTH = {
    "video_id": 7.0,
    "author_id": 10.0,
    "tag": 15.0,
    "tab": 24.0,
    "duration_bucket": 28.0,
    "upload_type": 22.0,
    "music_type": 25.0,
    "onehot_feat3": 12.0,
    "onehot_feat8": 14.0,
    "video_type": 30.0,
}

NB_WEIGHT = {
    "video_id": 0.90,
    "author_id": 0.80,
    "tag": 0.55,
    "tab": 0.75,
    "duration_bucket": 0.40,
    "upload_type": 0.30,
    "music_type": 0.20,
    "onehot_feat3": 0.45,
    "onehot_feat8": 0.35,
    "video_type": 0.15,
}

TEMPORAL_STRENGTH = {
    "video_id": 30.0,
    "author_id": 45.0,
    "tag": 70.0,
    "tab": 110.0,
    "duration_bucket": 130.0,
    "upload_type": 100.0,
    "music_type": 120.0,
    "onehot_feat3": 55.0,
    "onehot_feat8": 65.0,
    "video_type": 140.0,
}

TEMPORAL_WEIGHT = {
    "video_id": 1.25,
    "author_id": 0.90,
    "tag": 0.60,
    "tab": 0.85,
    "duration_bucket": 0.40,
    "upload_type": 0.30,
    "music_type": 0.20,
    "onehot_feat3": 0.45,
    "onehot_feat8": 0.35,
    "video_type": 0.15,
}

tr_user = np.asarray(train.X["user_id"], dtype=np.int64)
va_user = np.asarray(valid.X["user_id"], dtype=np.int64)
te_user = np.asarray(test.X["user_id"], dtype=np.int64)

user_cardinality = int(FEATURE_CARDINALITIES["user_id"])
user_count = np.bincount(
    tr_user,
    weights=recency_weight,
    minlength=user_cardinality,
).astype(np.float32)
user_positive = np.bincount(
    tr_user,
    weights=recency_weight * y_train,
    minlength=user_cardinality,
).astype(np.float32)
user_negative = user_count - user_positive

# User totals only enter the generative normalization. These terms are
# constant within a user, but retaining them makes cold-user fallback sane.
user_positive_denom = user_positive + 12.0
user_negative_denom = user_negative + 12.0

pair_valid = np.zeros(len(va_user), dtype=np.float32)
pair_test = np.zeros(len(te_user), dtype=np.float32)
nb_valid = np.zeros(len(va_user), dtype=np.float32)
nb_test = np.zeros(len(te_user), dtype=np.float32)
temporal_valid = np.zeros(len(va_user), dtype=np.float32)
temporal_test = np.zeros(len(te_user), dtype=np.float32)

old_mask = train_day <= 415
recent_mask = train_day >= 416
old_center = float(np.average(train_day[old_mask]))
recent_center = float(np.average(train_day[recent_mask]))
center_separation = max(recent_center - old_center, 1.0)

old_global_rate = float(np.mean(y_train[old_mask]))
recent_global_rate = float(np.mean(y_train[recent_mask]))

valid_day = (np.asarray(valid.date, dtype=np.int32) % 100).astype(np.float32)
test_day = (np.asarray(test.date, dtype=np.int32) % 100).astype(np.float32)

for name in PAIR_FIELDS:
    cardinality = int(FEATURE_CARDINALITIES[name])
    tr_cat = np.asarray(train.X[name], dtype=np.int64)
    va_cat = np.asarray(valid.X[name], dtype=np.int64)
    te_cat = np.asarray(test.X[name], dtype=np.int64)

    # Recency-weighted global entity posterior.
    entity_count = np.bincount(
        tr_cat,
        weights=recency_weight,
        minlength=cardinality,
    ).astype(np.float32)
    entity_positive = np.bincount(
        tr_cat,
        weights=recency_weight * y_train,
        minlength=cardinality,
    ).astype(np.float32)

    entity_strength = float(TEMPORAL_STRENGTH[name])
    entity_rate = (
        entity_positive + entity_strength * global_rate
    ) / (entity_count + entity_strength)

    # Family 1: discriminative empirical-Bayes user/category posterior.
    pair_keys, pair_count, pair_positive = sparse_pair_statistics(
        tr_user, tr_cat, cardinality
    )

    va_keys = va_user * np.int64(cardinality) + va_cat
    te_keys = te_user * np.int64(cardinality) + te_cat

    va_pair_count = query_sparse(pair_keys, pair_count, va_keys)
    te_pair_count = query_sparse(pair_keys, pair_count, te_keys)
    va_pair_positive = query_sparse(pair_keys, pair_positive, va_keys)
    te_pair_positive = query_sparse(pair_keys, pair_positive, te_keys)

    strength = float(PAIR_STRENGTH[name])
    va_pair_rate = (
        va_pair_positive + strength * entity_rate[va_cat]
    ) / (va_pair_count + strength)
    te_pair_rate = (
        te_pair_positive + strength * entity_rate[te_cat]
    ) / (te_pair_count + strength)

    pair_valid += float(PAIR_WEIGHT[name]) * safe_logit(va_pair_rate)
    pair_test += float(PAIR_WEIGHT[name]) * safe_logit(te_pair_rate)

    # Family 2: generative user-profile Naive Bayes. It ranks categories
    # by their smoothed likelihood under a user's positive versus negative
    # historical profile.
    pair_negative = np.maximum(pair_count - pair_positive, 0.0)

    total_pos_by_cat = np.bincount(
        tr_cat,
        weights=recency_weight * y_train,
        minlength=cardinality,
    ).astype(np.float32)
    total_neg_by_cat = np.bincount(
        tr_cat,
        weights=recency_weight * (1.0 - y_train),
        minlength=cardinality,
    ).astype(np.float32)

    positive_category_prob = (
        total_pos_by_cat + 0.5
    ) / (np.sum(total_pos_by_cat) + 0.5 * cardinality)
    negative_category_prob = (
        total_neg_by_cat + 0.5
    ) / (np.sum(total_neg_by_cat) + 0.5 * cardinality)

    va_pair_negative = query_sparse(pair_keys, pair_negative, va_keys)
    te_pair_negative = query_sparse(pair_keys, pair_negative, te_keys)

    nb_strength = float(NB_STRENGTH[name])

    va_pos_likelihood = (
        va_pair_positive
        + nb_strength * positive_category_prob[va_cat]
    ) / user_positive_denom[
        np.minimum(va_user, user_cardinality - 1)
    ]
    va_neg_likelihood = (
        va_pair_negative
        + nb_strength * negative_category_prob[va_cat]
    ) / user_negative_denom[
        np.minimum(va_user, user_cardinality - 1)
    ]

    te_pos_likelihood = (
        te_pair_positive
        + nb_strength * positive_category_prob[te_cat]
    ) / user_positive_denom[
        np.minimum(te_user, user_cardinality - 1)
    ]
    te_neg_likelihood = (
        te_pair_negative
        + nb_strength * negative_category_prob[te_cat]
    ) / user_negative_denom[
        np.minimum(te_user, user_cardinality - 1)
    ]

    nb_valid += float(NB_WEIGHT[name]) * (
        np.log(np.maximum(va_pos_likelihood, EPS))
        - np.log(np.maximum(va_neg_likelihood, EPS))
    )
    nb_test += float(NB_WEIGHT[name]) * (
        np.log(np.maximum(te_pos_likelihood, EPS))
        - np.log(np.maximum(te_neg_likelihood, EPS))
    )

    # Family 3: two-state temporal posterior. Old and recent entity rates
    # are independently shrunk to their window means, then their log-odds
    # difference is conservatively extrapolated by evaluation date.
    old_count = np.bincount(
        tr_cat[old_mask],
        minlength=cardinality,
    ).astype(np.float32)
    old_positive = np.bincount(
        tr_cat[old_mask],
        weights=y_train[old_mask],
        minlength=cardinality,
    ).astype(np.float32)

    recent_count = np.bincount(
        tr_cat[recent_mask],
        minlength=cardinality,
    ).astype(np.float32)
    recent_positive = np.bincount(
        tr_cat[recent_mask],
        weights=y_train[recent_mask],
        minlength=cardinality,
    ).astype(np.float32)

    temporal_strength = float(TEMPORAL_STRENGTH[name])
    old_rate = (
        old_positive + temporal_strength * old_global_rate
    ) / (old_count + temporal_strength)
    recent_rate = (
        recent_positive + temporal_strength * recent_global_rate
    ) / (recent_count + temporal_strength)

    recent_logit = safe_logit(recent_rate)
    trend_per_day = (
        recent_logit - safe_logit(old_rate)
    ) / center_separation

    # Conservative damping reduces variance for sparse identities.
    va_offset = 0.30 * (valid_day - recent_center)
    te_offset = 0.30 * (test_day - recent_center)

    temporal_valid += float(TEMPORAL_WEIGHT[name]) * (
        recent_logit[va_cat] + va_offset * trend_per_day[va_cat]
    )
    temporal_test += float(TEMPORAL_WEIGHT[name]) * (
        recent_logit[te_cat] + te_offset * trend_per_day[te_cat]
    )

    del (
        tr_cat, va_cat, te_cat, entity_count, entity_positive,
        pair_keys, pair_count, pair_positive, pair_negative,
        va_keys, te_keys, va_pair_count, te_pair_count,
        va_pair_positive, te_pair_positive, va_pair_negative,
        te_pair_negative, old_count, old_positive,
        recent_count, recent_positive
    )
    gc.collect()


own_candidates = {
    "personalized_pair_bayes": (
        pair_valid.astype(np.float32),
        pair_test.astype(np.float32),
    ),
    "generative_user_naive_bayes": (
        nb_valid.astype(np.float32),
        nb_test.astype(np.float32),
    ),
    "temporal_entity_state": (
        temporal_valid.astype(np.float32),
        temporal_test.astype(np.float32),
    ),
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float32)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float32)

inc_scale = max(float(np.std(inc_valid)), 1e-6)
inc_valid_normalized = inc_valid / inc_scale
inc_test_normalized = inc_test / inc_scale

candidate_log = {}
best_metric = evaluate(valid.user_id, y_valid, inc_valid)
best_name = "incumbent"
best_valid_scores = inc_valid.astype(np.float32)
best_test_scores = inc_test.astype(np.float32)
best_raw_valid = pair_valid.astype(np.float32)
best_alpha = 0.0

candidate_log["incumbent"] = float(best_metric["primary"])

blend_alphas = [0.15, 0.30, 0.50, 0.75, 1.00]

for family_name, (own_valid, own_test) in own_candidates.items():
    own_metric = evaluate(valid.user_id, y_valid, own_valid)
    candidate_log[family_name] = float(own_metric["primary"])

    own_scale = max(float(np.std(own_valid)), 1e-6)
    own_valid_normalized = own_valid / own_scale
    own_test_normalized = own_test / own_scale

    family_best_blend = -np.inf
    family_best_alpha = None

    for alpha in blend_alphas:
        blended_valid = (
            (1.0 - alpha) * inc_valid_normalized
            + alpha * own_valid_normalized
        ).astype(np.float32)
        blended_metric = evaluate(valid.user_id, y_valid, blended_valid)

        family_best_blend = max(
            family_best_blend, float(blended_metric["primary"])
        )

        if float(blended_metric["primary"]) > float(best_metric["primary"]):
            blended_test = (
                (1.0 - alpha) * inc_test_normalized
                + alpha * own_test_normalized
            ).astype(np.float32)

            best_metric = blended_metric
            best_name = family_name + "_incumbent_blend"
            best_valid_scores = blended_valid
            best_test_scores = blended_test
            best_raw_valid = own_valid
            best_alpha = float(alpha)
            family_best_alpha = float(alpha)

    candidate_log[family_name + "_best_blend"] = float(family_best_blend)

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "selected": best_name,
            "own_weight": best_alpha,
            "old_global_rate": old_global_rate,
            "recent_global_rate": recent_global_rate,
            "recency_weighted_rate": global_rate,
        },
        sort_keys=True,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metric["primary"]),
            "gauc": float(best_metric["gauc"]),
            "ndcg@5": float(best_metric["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)