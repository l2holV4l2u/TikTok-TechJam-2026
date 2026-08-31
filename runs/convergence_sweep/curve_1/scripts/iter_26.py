import os
import time
import json
import warnings

import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
warnings.filterwarnings("ignore")
np.random.seed(20260831)

train = load("train")
valid = load("valid")
test = load("test")

train_y = np.asarray(train.y, dtype=np.float64)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.load(inc_valid_path).astype(np.float64)
inc_test = np.load(inc_test_path).astype(np.float64)

if inc_valid.shape[0] != valid_users.shape[0]:
    raise RuntimeError("Incumbent validation prediction length mismatch")
if inc_test.shape[0] != test_users.shape[0]:
    raise RuntimeError("Incumbent test prediction length mismatch")


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def recency_weights(dates, half_life=5.0):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    day_index = np.searchsorted(unique_dates, dates)
    age = unique_dates.size - 1 - day_index
    return np.exp2(-age.astype(np.float64) / float(half_life))


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]

    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_flag)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    ranks = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ranks[multi] = (
        positions[multi]
        / (repeated_lengths[multi].astype(np.float64) - 1.0)
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranks
    return result


def sparse_fit(keys, values, weights, prior, prior_values=None):
    keys = np.asarray(keys, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    counts = np.bincount(
        inverse, weights=weights, minlength=unique_keys.size
    ).astype(np.float64)
    sums = np.bincount(
        inverse, weights=weights * values, minlength=unique_keys.size
    ).astype(np.float64)

    if prior_values is None:
        prior_mean = float(np.sum(weights * values) / np.sum(weights))
        estimates = (sums + prior * prior_mean) / (counts + prior)
    else:
        prior_values = np.asarray(prior_values, dtype=np.float64)
        estimates = (sums + prior * prior_values) / (counts + prior)

    return unique_keys, estimates.astype(np.float32), counts.astype(np.float32)


def sparse_lookup(keys, fitted_keys, fitted_values, default=0.0):
    keys = np.asarray(keys, dtype=np.int64)
    positions = np.searchsorted(fitted_keys, keys)
    valid = positions < fitted_keys.size

    out = np.full(keys.size, default, dtype=np.float64)
    valid_rows = np.flatnonzero(valid)
    if valid_rows.size:
        exact = fitted_keys[positions[valid_rows]] == keys[valid_rows]
        exact_rows = valid_rows[exact]
        out[exact_rows] = fitted_values[positions[exact_rows]]
    return out


weights = recency_weights(train.date, half_life=5.0)
global_rate = float(np.sum(weights * train_y) / np.sum(weights))
global_logit = float(logit(global_rate))


# ----------------------------------------------------------------------
# Family 1: within-user/day fixed-effect residualization.
#
# Global entity target rates partly encode which users were exposed to an
# entity. Removing each user's local user-day label propensity before
# estimating entity effects targets the within-user lift that GAUC/nDCG use.
# ----------------------------------------------------------------------

def fit_fixed_effect_family(split, labels, sample_weight):
    users = np.asarray(split.user_id, dtype=np.int64)
    dates = np.asarray(split.date, dtype=np.int32)
    unique_dates = np.unique(dates)
    day_index = np.searchsorted(unique_dates, dates).astype(np.int64)

    group_key = users * unique_dates.size + day_index
    group_unique, group_inverse = np.unique(group_key, return_inverse=True)

    group_count = np.bincount(
        group_inverse, minlength=group_unique.size
    ).astype(np.float64)
    group_positive = np.bincount(
        group_inverse, weights=labels, minlength=group_unique.size
    ).astype(np.float64)

    # Modest shrinkage avoids treating one-row user-days as known propensities.
    group_mean = (
        group_positive + 2.5 * global_rate
    ) / (group_count + 2.5)
    residual = labels - group_mean[group_inverse]

    fields = (
        "video_id",
        "author_id",
        "tag",
        "duration_bucket",
        "tab",
        "onehot_feat3",
        "upload_type",
    )
    priors = {
        "video_id": 18.0,
        "author_id": 25.0,
        "tag": 35.0,
        "duration_bucket": 45.0,
        "tab": 80.0,
        "onehot_feat3": 35.0,
        "upload_type": 55.0,
    }
    coefficients = {
        "video_id": 0.58,
        "author_id": 0.26,
        "tag": 0.10,
        "duration_bucket": 0.08,
        "tab": 0.10,
        "onehot_feat3": 0.08,
        "upload_type": 0.05,
    }

    fitted = {}
    for field in fields:
        ids = np.asarray(split.X[field], dtype=np.int64)
        cardinality = FEATURE_CARDINALITIES[field]
        count = np.bincount(
            ids, weights=sample_weight, minlength=cardinality
        ).astype(np.float64)
        total = np.bincount(
            ids,
            weights=sample_weight * residual,
            minlength=cardinality,
        ).astype(np.float64)
        fitted[field] = (
            total / (count + priors[field])
        ).astype(np.float32)

    diagnostics = {
        "user_day_groups": int(group_unique.size),
        "residual_sd": float(np.std(residual)),
        "singleton_user_days": float(np.mean(group_count == 1)),
    }
    return fitted, coefficients, diagnostics


def predict_fixed_effect(split, fitted, coefficients):
    result = np.zeros(len(split.user_id), dtype=np.float64)
    for field, coefficient in coefficients.items():
        result += coefficient * fitted[field][
            np.asarray(split.X[field], dtype=np.int64)
        ]
    return result


fixed_model, fixed_coefficients, fixed_diag = fit_fixed_effect_family(
    train, train_y, weights
)
fixed_valid = predict_fixed_effect(valid, fixed_model, fixed_coefficients)
fixed_test = predict_fixed_effect(test, fixed_model, fixed_coefficients)

print("FINDINGS fixed_effect=" + json.dumps(fixed_diag, sort_keys=True))


# ----------------------------------------------------------------------
# Family 2: demographic-segment/entity hierarchical Bayes.
#
# Users with the same stable profile share a segment. Segment/entity rates
# back off to global entity quality, permitting personalized ranking for
# sparse evaluation users without requiring repeated validation impressions.
# ----------------------------------------------------------------------

SEGMENT_FIELDS = (
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "is_live_streamer",
    "onehot_feat0",
    "onehot_feat1",
)


def segment_hash(split, modulus=8192):
    h = np.zeros(len(split.user_id), dtype=np.uint64)
    for i, field in enumerate(SEGMENT_FIELDS):
        x = np.asarray(split.X[field], dtype=np.uint64)
        h ^= (
            x + np.uint64(0x9E3779B97F4A7C15)
            + (h << np.uint64(6))
            + (h >> np.uint64(2))
            + np.uint64(1315423911 * (i + 1))
        )
    return np.asarray(h % np.uint64(modulus), dtype=np.int64)


def fit_demographic_hierarchy(split, labels, sample_weight):
    train_segment = segment_hash(split)

    entity_specs = (
        ("video_id", 30.0, 0.55, 55.0, 0.45),
        ("author_id", 38.0, 0.28, 70.0, 0.28),
        ("tag", 55.0, 0.10, 90.0, 0.16),
    )

    fitted = {}
    for field, entity_prior, base_coef, cross_prior, cross_coef in entity_specs:
        ids = np.asarray(split.X[field], dtype=np.int64)
        cardinality = FEATURE_CARDINALITIES[field]

        count = np.bincount(
            ids, weights=sample_weight, minlength=cardinality
        ).astype(np.float64)
        positive = np.bincount(
            ids,
            weights=sample_weight * labels,
            minlength=cardinality,
        ).astype(np.float64)
        base_rate = (
            positive + entity_prior * global_rate
        ) / (count + entity_prior)
        base_score = logit(base_rate) - global_logit

        joint_key = train_segment * cardinality + ids
        unique_key, inverse = np.unique(joint_key, return_inverse=True)
        joint_count = np.bincount(
            inverse, weights=sample_weight, minlength=unique_key.size
        ).astype(np.float64)
        joint_positive = np.bincount(
            inverse,
            weights=sample_weight * labels,
            minlength=unique_key.size,
        ).astype(np.float64)

        entity_for_key = unique_key % cardinality
        parent_rate = base_rate[entity_for_key]
        joint_rate = (
            joint_positive + cross_prior * parent_rate
        ) / (joint_count + cross_prior)
        deviation = logit(joint_rate) - logit(parent_rate)

        fitted[field] = {
            "cardinality": cardinality,
            "base_score": base_score.astype(np.float32),
            "keys": unique_key,
            "deviation": deviation.astype(np.float32),
            "base_coef": base_coef,
            "cross_coef": cross_coef,
        }

    diagnostics = {
        "segments_observed": int(np.unique(train_segment).size),
        "segment_modulus": 8192,
    }
    return fitted, diagnostics


def predict_demographic_hierarchy(split, fitted):
    segment = segment_hash(split)
    result = np.zeros(len(split.user_id), dtype=np.float64)

    for field, model in fitted.items():
        ids = np.asarray(split.X[field], dtype=np.int64)
        result += model["base_coef"] * model["base_score"][ids]

        key = segment * model["cardinality"] + ids
        deviation = sparse_lookup(
            key, model["keys"], model["deviation"], default=0.0
        )
        result += model["cross_coef"] * deviation

    return result


demographic_model, demographic_diag = fit_demographic_hierarchy(
    train, train_y, weights
)
demographic_valid = predict_demographic_hierarchy(valid, demographic_model)
demographic_test = predict_demographic_hierarchy(test, demographic_model)

print(
    "FINDINGS demographic_hierarchy="
    + json.dumps(demographic_diag, sort_keys=True)
)


# ----------------------------------------------------------------------
# Family 3: content/context conjunction rule ensemble.
#
# Exact pair rules estimate context-specific content quality such as
# video-by-tab and author-by-duration. Independent shrinkage of each rule
# makes this a sparse rule ensemble rather than an embedding interaction
# model, and the content-based rules can remain usable across date drift.
# ----------------------------------------------------------------------

RULE_SPECS = (
    ("video_id", "tab", 35.0, 0.34),
    ("author_id", "tab", 45.0, 0.18),
    ("tag", "tab", 65.0, 0.10),
    ("author_id", "duration_bucket", 50.0, 0.15),
    ("tag", "duration_bucket", 65.0, 0.10),
    ("onehot_feat3", "duration_bucket", 55.0, 0.08),
    ("duration_bucket", "tab", 75.0, 0.08),
    ("author_id", "upload_type", 60.0, 0.08),
)


def fit_rule_ensemble(split, labels, sample_weight):
    base_fields = (
        ("video_id", 24.0, 0.48),
        ("author_id", 32.0, 0.24),
        ("tag", 48.0, 0.08),
        ("duration_bucket", 65.0, 0.07),
        ("tab", 100.0, 0.08),
    )

    base_models = {}
    for field, prior, coefficient in base_fields:
        ids = np.asarray(split.X[field], dtype=np.int64)
        cardinality = FEATURE_CARDINALITIES[field]
        count = np.bincount(
            ids, weights=sample_weight, minlength=cardinality
        ).astype(np.float64)
        positive = np.bincount(
            ids,
            weights=sample_weight * labels,
            minlength=cardinality,
        ).astype(np.float64)
        rate = (
            positive + prior * global_rate
        ) / (count + prior)
        base_models[field] = (
            (logit(rate) - global_logit).astype(np.float32),
            coefficient,
        )

    rules = []
    for left, right, prior, coefficient in RULE_SPECS:
        left_ids = np.asarray(split.X[left], dtype=np.int64)
        right_ids = np.asarray(split.X[right], dtype=np.int64)
        right_card = FEATURE_CARDINALITIES[right]
        keys = left_ids * right_card + right_ids

        unique_keys, estimates, counts = sparse_fit(
            keys, labels, sample_weight, prior=prior
        )
        centered = logit(estimates) - global_logit

        # Reliability is already induced by Bayesian shrinkage. Retain counts
        # for diagnostics and use the centered rule log-odds directly.
        rules.append(
            {
                "left": left,
                "right": right,
                "right_card": right_card,
                "keys": unique_keys,
                "scores": centered.astype(np.float32),
                "coefficient": coefficient,
                "counts": counts,
            }
        )

    diagnostics = {
        "rules": len(rules),
        "rule_cells": int(sum(rule["keys"].size for rule in rules)),
        "cells_ge20": int(
            sum(np.sum(rule["counts"] >= 20.0) for rule in rules)
        ),
    }
    return base_models, rules, diagnostics


def predict_rule_ensemble(split, base_models, rules):
    result = np.zeros(len(split.user_id), dtype=np.float64)

    for field, (scores, coefficient) in base_models.items():
        ids = np.asarray(split.X[field], dtype=np.int64)
        result += coefficient * scores[ids]

    for rule in rules:
        left_ids = np.asarray(split.X[rule["left"]], dtype=np.int64)
        right_ids = np.asarray(split.X[rule["right"]], dtype=np.int64)
        keys = left_ids * rule["right_card"] + right_ids
        values = sparse_lookup(
            keys, rule["keys"], rule["scores"], default=0.0
        )
        result += rule["coefficient"] * values

    return result


rule_base, rule_models, rule_diag = fit_rule_ensemble(
    train, train_y, weights
)
rule_valid = predict_rule_ensemble(valid, rule_base, rule_models)
rule_test = predict_rule_ensemble(test, rule_base, rule_models)

print("FINDINGS rule_ensemble=" + json.dumps(rule_diag, sort_keys=True))


# A cross-family rank aggregate lets independent mechanisms vote without
# assuming their raw score scales are calibrated.
fixed_valid_rank = within_user_rank(valid_users, fixed_valid)
demographic_valid_rank = within_user_rank(valid_users, demographic_valid)
rule_valid_rank = within_user_rank(valid_users, rule_valid)

fixed_test_rank = within_user_rank(test_users, fixed_test)
demographic_test_rank = within_user_rank(test_users, demographic_test)
rule_test_rank = within_user_rank(test_users, rule_test)

aggregate_valid = (
    fixed_valid_rank + demographic_valid_rank + rule_valid_rank
) / 3.0
aggregate_test = (
    fixed_test_rank + demographic_test_rank + rule_test_rank
) / 3.0

own_valid_candidates = {
    "fixed_effect": fixed_valid,
    "demographic_hierarchy": demographic_valid,
    "content_rule_ensemble": rule_valid,
    "cross_family_rank_aggregate": aggregate_valid,
}
own_test_candidates = {
    "fixed_effect": fixed_test,
    "demographic_hierarchy": demographic_test,
    "content_rule_ensemble": rule_test,
    "cross_family_rank_aggregate": aggregate_test,
}

inc_valid_rank = within_user_rank(valid_users, inc_valid)
inc_test_rank = within_user_rank(test_users, inc_test)

candidate_log = {}
best_primary = -np.inf
best_metrics = None
best_scores_valid = None
best_scores_test = None
best_raw_valid = None
best_name = None

# Include the trusted incumbent as the zero-weight safeguard, then score
# every standalone family and several incumbent blends.
inc_metrics = evaluate(valid_users, valid_y, inc_valid)
candidate_log["trusted_incumbent"] = float(inc_metrics["primary"])

best_primary = float(inc_metrics["primary"])
best_metrics = inc_metrics
best_scores_valid = inc_valid.copy()
best_scores_test = inc_test.copy()
best_raw_valid = aggregate_valid.copy()
best_name = "trusted_incumbent"

alphas = (0.05, 0.10, 0.20, 0.30, 0.40, 0.55, 0.75, 1.00)

for family_name, raw_valid in own_valid_candidates.items():
    raw_test = own_test_candidates[family_name]
    valid_rank = within_user_rank(valid_users, raw_valid)
    test_rank = within_user_rank(test_users, raw_test)

    standalone_metrics = evaluate(valid_users, valid_y, raw_valid)
    candidate_log[family_name + "_standalone"] = float(
        standalone_metrics["primary"]
    )

    for alpha in alphas:
        blended_valid = (
            (1.0 - alpha) * inc_valid_rank + alpha * valid_rank
        )
        metrics = evaluate(valid_users, valid_y, blended_valid)
        name = family_name + "_blend_" + ("%.2f" % alpha)
        candidate_log[name] = float(metrics["primary"])

        if float(metrics["primary"]) > best_primary:
            blended_test = (
                (1.0 - alpha) * inc_test_rank + alpha * test_rank
            )
            best_primary = float(metrics["primary"])
            best_metrics = metrics
            best_scores_valid = blended_valid.copy()
            best_scores_test = blended_test.copy()
            best_raw_valid = raw_valid.copy()
            best_name = name

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print(
    "FINDINGS selected="
    + json.dumps(
        {
            "name": best_name,
            "primary": best_primary,
            "global_train_rate": global_rate,
        },
        sort_keys=True,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_scores_test, dtype=np.float64),
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
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)