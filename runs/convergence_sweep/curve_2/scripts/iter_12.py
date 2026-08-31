import os
import time
import json
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

tr_user = np.asarray(train.X["user_id"], dtype=np.int64)
va_user = np.asarray(valid.X["user_id"], dtype=np.int64)
te_user = np.asarray(test.X["user_id"], dtype=np.int64)

tr_time = np.asarray(train.time_ms, dtype=np.int64)
va_time = np.asarray(valid.time_ms, dtype=np.int64)
te_time = np.asarray(test.time_ms, dtype=np.int64)

n_users = int(FEATURE_CARDINALITIES["user_id"])

# Recent training interactions receive more weight because evaluation begins
# immediately after the training boundary and the distribution is drifting.
train_day = (np.asarray(train.date, dtype=np.int32) % 100).astype(np.float32)
age_days = float(np.max(train_day)) - train_day
sample_weight = np.exp(-np.log(2.0) * age_days / 5.0).astype(np.float32)
sample_weight /= np.mean(sample_weight)

global_rate = float(
    np.sum(sample_weight * y_train) / np.maximum(np.sum(sample_weight), EPS)
)


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float32), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def sparse_lookup(keys, values, query):
    query = np.asarray(query, dtype=np.int64)
    idx = np.searchsorted(keys, query)
    clipped = np.minimum(idx, len(keys) - 1)
    found = (idx < len(keys)) & (keys[clipped] == query)
    result = np.zeros(len(query), dtype=np.float32)
    result[found] = values[clipped[found]]
    return result


def chronological_order(users, times):
    rows = np.arange(len(users), dtype=np.int64)
    return np.lexsort((rows, times, users))


tr_order = chronological_order(tr_user, tr_time)
va_order = chronological_order(va_user, va_time)
te_order = chronological_order(te_user, te_time)

# Index of the last training impression for each user.
last_train_index = np.full(n_users, -1, dtype=np.int64)
last_train_index[tr_user[tr_order]] = tr_order


def previous_categories(split_user, split_order, current_category,
                        train_category):
    """
    Previous visible category in the same evaluation split. For a user's
    first evaluation impression, back off to that user's final train event.
    This uses no evaluation outcomes.
    """
    result = np.zeros(len(split_user), dtype=np.int64)

    ordered_users = split_user[split_order]
    same_user = np.zeros(len(split_order), dtype=bool)
    same_user[1:] = ordered_users[1:] == ordered_users[:-1]

    ordered_result = np.zeros(len(split_order), dtype=np.int64)
    ordered_current = current_category[split_order]
    ordered_result[1:][same_user[1:]] = ordered_current[:-1][same_user[1:]]

    first_positions = ~same_user
    first_users = ordered_users[first_positions]
    train_idx = last_train_index[np.minimum(first_users, n_users - 1)]
    has_train = train_idx >= 0

    first_values = np.zeros(len(first_users), dtype=np.int64)
    first_values[has_train] = train_category[train_idx[has_train]]
    ordered_result[first_positions] = first_values

    result[split_order] = ordered_result
    return result


# Consecutive train events define the transition supervision. Very long gaps
# are excluded because they are not plausibly part of the same preference
# state or feed session.
ordered_tr_user = tr_user[tr_order]
consecutive = np.zeros(len(tr_order), dtype=bool)
consecutive[1:] = ordered_tr_user[1:] == ordered_tr_user[:-1]

gap_ms = np.zeros(len(tr_order), dtype=np.int64)
gap_ms[1:] = tr_time[tr_order[1:]] - tr_time[tr_order[:-1]]
consecutive &= (gap_ms >= 0) & (gap_ms <= 2 * 24 * 3600 * 1000)

current_indices = tr_order[consecutive]
previous_indices = np.empty(len(current_indices), dtype=np.int64)
ordered_positions = np.flatnonzero(consecutive)
previous_indices[:] = tr_order[ordered_positions - 1]

TRANSITION_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "tab",
    "onehot_feat3",
    "duration_bucket",
]

FIELD_WEIGHT = {
    "video_id": 1.30,
    "author_id": 1.10,
    "tag": 0.80,
    "tab": 0.85,
    "onehot_feat3": 0.70,
    "duration_bucket": 0.45,
}

TRANSITION_STRENGTH = {
    "video_id": 12.0,
    "author_id": 16.0,
    "tag": 24.0,
    "tab": 35.0,
    "onehot_feat3": 22.0,
    "duration_bucket": 40.0,
}

BACKOFF_THRESHOLD = {
    "video_id": 2.0,
    "author_id": 3.0,
    "tag": 5.0,
    "tab": 8.0,
    "onehot_feat3": 5.0,
    "duration_bucket": 10.0,
}

# Last-response hazards are deliberately shorter-memory than the empirical
# Bayes mechanisms tried previously: the latest positive and latest negative
# event compete directly rather than being averaged into a stationary rate.
HAZARD_HALF_LIFE = {
    "video_id": 5.0,
    "author_id": 6.0,
    "tag": 8.0,
    "tab": 9.0,
    "onehot_feat3": 7.0,
    "duration_bucket": 10.0,
}

additive_valid = np.zeros(len(valid.user_id), dtype=np.float32)
additive_test = np.zeros(len(test.user_id), dtype=np.float32)

backoff_valid = np.zeros(len(valid.user_id), dtype=np.float32)
backoff_test = np.zeros(len(test.user_id), dtype=np.float32)
backoff_valid_assigned = np.zeros(len(valid.user_id), dtype=bool)
backoff_test_assigned = np.zeros(len(test.user_id), dtype=bool)

hazard_valid = np.zeros(len(valid.user_id), dtype=np.float32)
hazard_test = np.zeros(len(test.user_id), dtype=np.float32)

train_end_ms = int(np.max(tr_time))
day_ms = float(24 * 3600 * 1000)

for name in TRANSITION_FIELDS:
    cardinality = int(FEATURE_CARDINALITIES[name])
    tr_cat = np.asarray(train.X[name], dtype=np.int64)
    va_cat = np.asarray(valid.X[name], dtype=np.int64)
    te_cat = np.asarray(test.X[name], dtype=np.int64)

    va_prev = previous_categories(va_user, va_order, va_cat, tr_cat)
    te_prev = previous_categories(te_user, te_order, te_cat, tr_cat)

    # Smoothed current-category prior.
    entity_count = np.bincount(
        tr_cat,
        weights=sample_weight,
        minlength=cardinality
    ).astype(np.float32)
    entity_positive = np.bincount(
        tr_cat,
        weights=sample_weight * y_train,
        minlength=cardinality
    ).astype(np.float32)
    entity_rate = (
        entity_positive + 30.0 * global_rate
    ) / (entity_count + 30.0)

    # Sparse first-order transition sufficient statistics.
    transition_key = (
        tr_cat[previous_indices] * np.int64(cardinality)
        + tr_cat[current_indices]
    )
    unique_key, inverse = np.unique(transition_key, return_inverse=True)

    trans_weight = sample_weight[current_indices]
    trans_count = np.bincount(
        inverse, weights=trans_weight, minlength=len(unique_key)
    ).astype(np.float32)
    trans_positive = np.bincount(
        inverse,
        weights=trans_weight * y_train[current_indices],
        minlength=len(unique_key)
    ).astype(np.float32)

    va_key = va_prev * np.int64(cardinality) + va_cat
    te_key = te_prev * np.int64(cardinality) + te_cat

    va_count = sparse_lookup(unique_key, trans_count, va_key)
    te_count = sparse_lookup(unique_key, trans_count, te_key)
    va_positive = sparse_lookup(unique_key, trans_positive, va_key)
    te_positive = sparse_lookup(unique_key, trans_positive, te_key)

    strength = float(TRANSITION_STRENGTH[name])
    va_rate = (
        va_positive + strength * entity_rate[va_cat]
    ) / (va_count + strength)
    te_rate = (
        te_positive + strength * entity_rate[te_cat]
    ) / (te_count + strength)

    va_transition_score = safe_logit(va_rate)
    te_transition_score = safe_logit(te_rate)

    additive_valid += float(FIELD_WEIGHT[name]) * va_transition_score
    additive_test += float(FIELD_WEIGHT[name]) * te_transition_score

    # Variable-order backoff: use the first sufficiently supported transition
    # in specificity order, rather than summing correlated transition tables.
    threshold = float(BACKOFF_THRESHOLD[name])
    take_va = (~backoff_valid_assigned) & (va_count >= threshold)
    take_te = (~backoff_test_assigned) & (te_count >= threshold)
    backoff_valid[take_va] = va_transition_score[take_va]
    backoff_test[take_te] = te_transition_score[take_te]
    backoff_valid_assigned[take_va] = True
    backoff_test_assigned[take_te] = True

    # Latest positive and negative timestamps for each observed user/category.
    pair_key = tr_user * np.int64(cardinality) + tr_cat
    pair_unique, pair_inverse = np.unique(pair_key, return_inverse=True)

    last_positive = np.zeros(len(pair_unique), dtype=np.int64)
    last_negative = np.zeros(len(pair_unique), dtype=np.int64)

    positive_rows = y_train > 0.5
    negative_rows = ~positive_rows
    np.maximum.at(
        last_positive,
        pair_inverse[positive_rows],
        tr_time[positive_rows]
    )
    np.maximum.at(
        last_negative,
        pair_inverse[negative_rows],
        tr_time[negative_rows]
    )

    va_pair_key = va_user * np.int64(cardinality) + va_cat
    te_pair_key = te_user * np.int64(cardinality) + te_cat

    va_last_pos = sparse_lookup(
        pair_unique, last_positive.astype(np.float64), va_pair_key
    ).astype(np.float64)
    va_last_neg = sparse_lookup(
        pair_unique, last_negative.astype(np.float64), va_pair_key
    ).astype(np.float64)
    te_last_pos = sparse_lookup(
        pair_unique, last_positive.astype(np.float64), te_pair_key
    ).astype(np.float64)
    te_last_neg = sparse_lookup(
        pair_unique, last_negative.astype(np.float64), te_pair_key
    ).astype(np.float64)

    half_life = float(HAZARD_HALF_LIFE[name])

    def decayed_presence(last_timestamp):
        present = last_timestamp > 0
        age = np.maximum(
            (train_end_ms - last_timestamp) / day_ms, 0.0
        )
        value = np.zeros(len(last_timestamp), dtype=np.float32)
        value[present] = np.exp(
            -np.log(2.0) * age[present] / half_life
        ).astype(np.float32)
        return value

    va_hazard = decayed_presence(va_last_pos) - decayed_presence(va_last_neg)
    te_hazard = decayed_presence(te_last_pos) - decayed_presence(te_last_neg)

    hazard_valid += float(FIELD_WEIGHT[name]) * va_hazard
    hazard_test += float(FIELD_WEIGHT[name]) * te_hazard

    del (
        tr_cat, va_cat, te_cat, va_prev, te_prev,
        entity_count, entity_positive, entity_rate,
        transition_key, unique_key, inverse, trans_count, trans_positive,
        va_key, te_key, va_count, te_count, va_positive, te_positive,
        va_rate, te_rate, pair_key, pair_unique, pair_inverse,
        last_positive, last_negative, va_pair_key, te_pair_key,
        va_last_pos, va_last_neg, te_last_pos, te_last_neg
    )
    gc.collect()

# Unresolved backoff rows use the least specific current-item prior represented
# by the additive transition model's average scale.
backoff_valid[~backoff_valid_assigned] = (
    additive_valid[~backoff_valid_assigned]
    / max(sum(FIELD_WEIGHT.values()), EPS)
)
backoff_test[~backoff_test_assigned] = (
    additive_test[~backoff_test_assigned]
    / max(sum(FIELD_WEIGHT.values()), EPS)
)

own_families = {
    "additive_first_order_markov": (
        additive_valid.astype(np.float32),
        additive_test.astype(np.float32),
    ),
    "hierarchical_markov_backoff": (
        backoff_valid.astype(np.float32),
        backoff_test.astype(np.float32),
    ),
    "last_response_hazard": (
        hazard_valid.astype(np.float32),
        hazard_test.astype(np.float32),
    ),
}

# A temporal-state ensemble is also tested. Its components are standardized
# using validation only; the same fixed transformations are applied to test.
ensemble_valid = np.zeros(len(valid.user_id), dtype=np.float32)
ensemble_test = np.zeros(len(test.user_id), dtype=np.float32)
for va_score, te_score in own_families.values():
    center = float(np.mean(va_score))
    scale = max(float(np.std(va_score)), 1e-6)
    ensemble_valid += (va_score - center) / scale
    ensemble_test += (te_score - center) / scale
ensemble_valid /= float(len(own_families))
ensemble_test /= float(len(own_families))
own_families["temporal_state_ensemble"] = (
    ensemble_valid.astype(np.float32),
    ensemble_test.astype(np.float32),
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float32
)
inc_test = np.asarray(
    np.load(os.path.join(shared, "incumbent_test_scores.npy")),
    dtype=np.float32
)

inc_center = float(np.mean(inc_valid))
inc_scale = max(float(np.std(inc_valid)), 1e-6)
inc_valid_z = (inc_valid - inc_center) / inc_scale
inc_test_z = (inc_test - inc_center) / inc_scale

best_metric = evaluate(valid.user_id, y_valid, inc_valid)
best_valid = inc_valid.copy()
best_test = inc_test.copy()
best_raw = additive_valid.copy()
best_name = "incumbent"
best_alpha = 0.0

candidate_log = {"incumbent": float(best_metric["primary"])}
blend_alphas = [0.05, 0.10, 0.20, 0.35, 0.50]

for family_name, (own_valid, own_test) in own_families.items():
    raw_metric = evaluate(valid.user_id, y_valid, own_valid)
    candidate_log[family_name] = float(raw_metric["primary"])

    center = float(np.mean(own_valid))
    scale = max(float(np.std(own_valid)), 1e-6)
    own_valid_z = (own_valid - center) / scale
    own_test_z = (own_test - center) / scale

    family_best = -np.inf
    family_best_alpha = 0.0

    for alpha in blend_alphas:
        blended_valid = (
            (1.0 - alpha) * inc_valid_z + alpha * own_valid_z
        ).astype(np.float32)
        metric = evaluate(valid.user_id, y_valid, blended_valid)

        if float(metric["primary"]) > family_best:
            family_best = float(metric["primary"])
            family_best_alpha = float(alpha)

        if float(metric["primary"]) > float(best_metric["primary"]):
            best_metric = metric
            best_valid = blended_valid
            best_test = (
                (1.0 - alpha) * inc_test_z + alpha * own_test_z
            ).astype(np.float32)
            best_raw = own_valid.copy()
            best_name = family_name + "_incumbent_blend"
            best_alpha = float(alpha)

    candidate_log[family_name + "_best_blend"] = family_best
    candidate_log[family_name + "_blend_alpha"] = family_best_alpha

candidate_log["selected_alpha"] = best_alpha

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print(
    "FINDINGS selected=%s alpha=%.3f transition_coverage_valid=%.4f "
    "transition_coverage_test=%.4f"
    % (
        best_name,
        best_alpha,
        float(np.mean(backoff_valid_assigned)),
        float(np.mean(backoff_test_assigned)),
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64)
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64)
    )
    if best_name != "incumbent":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64)
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