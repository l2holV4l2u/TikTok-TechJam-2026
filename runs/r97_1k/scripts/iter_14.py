import os
import gc
import json
import time
import warnings
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")

START = time.time()
OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")

if OUT:
    os.makedirs(OUT, exist_ok=True)


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float32), 1e-4, 1.0 - 1e-4)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    start_pos = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local_rank = (
        np.arange(n, dtype=np.float32) - start_pos.astype(np.float32)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_pos = np.flatnonzero(ends)
    group_sizes = np.diff(np.r_[-1, end_pos]).astype(np.float32)

    group_index = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(group_sizes[group_index] - 1.0, 1.0)

    result = np.empty(n, dtype=np.float32)
    result[order] = local_rank / denom
    return result


def fit_global_table(code, y, weights, cardinality, alpha, prior):
    code = np.asarray(code, dtype=np.int64)
    sw = np.bincount(
        code, weights=weights, minlength=cardinality
    ).astype(np.float32)
    sy = np.bincount(
        code, weights=weights * y, minlength=cardinality
    ).astype(np.float32)

    rate = (sy + float(alpha) * prior) / np.maximum(
        sw + float(alpha), 1e-6
    )
    return safe_logit(rate), sw


def apply_global(code, table, fallback):
    code = np.asarray(code, dtype=np.int64)
    result = np.full(len(code), fallback, dtype=np.float32)
    ok = (code >= 0) & (code < len(table))
    result[ok] = table[code[ok]]
    return result


def fit_hierarchical_pair(
    group,
    category,
    y,
    weights,
    group_cardinality,
    category_cardinality,
    category_logit,
    alpha,
):
    group = np.asarray(group, dtype=np.int64)
    category = np.asarray(category, dtype=np.int64)

    key = group * int(category_cardinality) + category
    size = int(group_cardinality) * int(category_cardinality)

    sw = np.bincount(
        key, weights=weights, minlength=size
    ).astype(np.float32)
    sy = np.bincount(
        key, weights=weights * y, minlength=size
    ).astype(np.float32)

    category_rate = (
        1.0 / (1.0 + np.exp(-np.asarray(category_logit, dtype=np.float32)))
    )
    prior_for_key = np.tile(
        category_rate, int(group_cardinality)
    ).astype(np.float32)

    rate = (
        sy + float(alpha) * prior_for_key
    ) / np.maximum(sw + float(alpha), 1e-6)

    delta = safe_logit(rate) - safe_logit(prior_for_key)
    confidence = sw / np.maximum(sw + float(alpha), 1e-6)

    # Group zero is unseen/unknown and should contribute no personal signal.
    delta[:category_cardinality] = 0.0
    confidence[:category_cardinality] = 0.0

    return delta.astype(np.float32), confidence.astype(np.float32)


def apply_pair(group, category, table_info, category_cardinality):
    delta_table, confidence_table = table_info
    group = np.asarray(group, dtype=np.int64)
    category = np.asarray(category, dtype=np.int64)
    key = group * int(category_cardinality) + category

    delta = np.zeros(len(group), dtype=np.float32)
    confidence = np.zeros(len(group), dtype=np.float32)
    ok = (key >= 0) & (key < len(delta_table)) & (group != 0)

    delta[ok] = delta_table[key[ok]]
    confidence[ok] = confidence_table[key[ok]]
    return delta, confidence


COHORT_FIELDS = (
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
)


def cohort_code(split):
    result = np.zeros(len(split.user_id), dtype=np.int64)
    multiplier = 1

    for name in COHORT_FIELDS:
        card = int(FEATURE_CARDINALITIES[name])
        values = np.asarray(split.X[name], dtype=np.int64)
        values = np.clip(values, 0, card - 1)
        result += multiplier * values
        multiplier *= card

    return result, multiplier


train = load("train")
train_y = np.asarray(train.y, dtype=np.float32)
train_date = np.asarray(train.date, dtype=np.int32)

max_train_date = int(np.max(train_date))
day_age = (max_train_date - train_date).astype(np.float32)

# This is applied to the main preference estimator rather than merely to a
# weak side feature. Four days was independently motivated by the measured
# drift and does not use validation labels to construct any statistic.
half_life = 4.0
train_weight = np.power(0.5, day_age / half_life).astype(np.float32)
train_weight /= np.mean(train_weight)

prior = float(
    np.sum(train_weight * train_y) /
    np.maximum(np.sum(train_weight), 1e-6)
)
prior_logit = float(safe_logit(np.asarray([prior]))[0])

print(
    "FINDINGS preference_half_life=%.1f weighted_prior=%.6f "
    "weight_q01_q50_q99=%.4f,%.4f,%.4f"
    % (
        half_life,
        prior,
        float(np.quantile(train_weight, 0.01)),
        float(np.quantile(train_weight, 0.50)),
        float(np.quantile(train_weight, 0.99)),
    ),
    flush=True,
)

PERSONAL_FIELDS = (
    "tag",
    "duration_bucket",
    "tab",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
)

GLOBAL_EXTRA_FIELDS = (
    "author_id",
    "video_id",
)

global_tables = {}
global_counts = {}

for name in PERSONAL_FIELDS + GLOBAL_EXTRA_FIELDS:
    card = int(FEATURE_CARDINALITIES[name])
    alpha = 100.0 if name in GLOBAL_EXTRA_FIELDS else 300.0
    table, counts = fit_global_table(
        train.X[name],
        train_y,
        train_weight,
        card,
        alpha,
        prior,
    )
    global_tables[name] = table
    global_counts[name] = counts

user_card = int(FEATURE_CARDINALITIES["user_id"])
train_uid = np.asarray(train.user_id, dtype=np.int64)

pair_alphas = {
    "tag": 24.0,
    "duration_bucket": 30.0,
    "tab": 35.0,
    "upload_type": 28.0,
    "onehot_feat3": 18.0,
    "onehot_feat8": 18.0,
}

user_pair_tables = {}
for name in PERSONAL_FIELDS:
    card = int(FEATURE_CARDINALITIES[name])
    user_pair_tables[name] = fit_hierarchical_pair(
        train_uid,
        train.X[name],
        train_y,
        train_weight,
        user_card,
        card,
        global_tables[name],
        pair_alphas[name],
    )

train_cohort, cohort_card = cohort_code(train)

COHORT_PREFERENCE_FIELDS = (
    "tag",
    "duration_bucket",
    "onehot_feat3",
    "onehot_feat8",
)

cohort_pair_tables = {}
for name in COHORT_PREFERENCE_FIELDS:
    card = int(FEATURE_CARDINALITIES[name])
    cohort_pair_tables[name] = fit_hierarchical_pair(
        train_cohort,
        train.X[name],
        train_y,
        train_weight,
        cohort_card,
        card,
        global_tables[name],
        80.0,
    )

print(
    "FINDINGS user_pair_cells=%d cohort_cardinality=%d"
    % (
        int(sum(
            user_card * int(FEATURE_CARDINALITIES[x])
            for x in PERSONAL_FIELDS
        )),
        int(cohort_card),
    ),
    flush=True,
)

del train_uid, train_cohort, train_date, day_age
del train_weight, train_y, train
gc.collect()


def score_preference_families(split):
    n = len(split.user_id)
    uid = np.asarray(split.user_id, dtype=np.int64)

    # Stable item/content evidence, deliberately downweighting brittle video
    # identity relative to author and coarse content descriptors.
    global_score = np.zeros(n, dtype=np.float32)
    global_weight = 0.0

    global_specs = (
        ("author_id", 1.00),
        ("video_id", 0.45),
        ("tag", 0.75),
        ("duration_bucket", 0.75),
        ("tab", 0.55),
        ("upload_type", 0.45),
        ("onehot_feat3", 0.65),
        ("onehot_feat8", 0.65),
    )

    for name, weight in global_specs:
        value = apply_global(
            split.X[name], global_tables[name], prior_logit
        )
        global_score += float(weight) * (value - prior_logit)
        global_weight += abs(float(weight))

    global_score /= max(global_weight, 1e-6)

    personal_deltas = []
    personal_confidences = []

    for name in PERSONAL_FIELDS:
        card = int(FEATURE_CARDINALITIES[name])
        delta, confidence = apply_pair(
            uid,
            split.X[name],
            user_pair_tables[name],
            card,
        )
        personal_deltas.append(delta)
        personal_confidences.append(confidence)

    delta_matrix = np.stack(personal_deltas, axis=0)
    conf_matrix = np.stack(personal_confidences, axis=0)

    # Family 1: hierarchical additive empirical Bayes. Every field contributes
    # in proportion to the evidence available for that user/category pair.
    weighted_sum = np.sum(
        delta_matrix * conf_matrix, axis=0, dtype=np.float32
    )
    confidence_sum = np.sum(conf_matrix, axis=0, dtype=np.float32)
    additive = weighted_sum / np.maximum(confidence_sum, 0.35)

    # Family 2: mixture-of-experts. Only the most reliable personal preference
    # expert is used on each impression, avoiding cancellation and noisy sums.
    best_field = np.argmax(conf_matrix, axis=0)
    column = np.arange(n, dtype=np.int64)
    gated = delta_matrix[best_field, column]
    gated *= conf_matrix[best_field, column]

    # Family 3: robust product-of-experts. The median personal log-odds lift is
    # insensitive to one stale or spuriously extreme category estimate.
    adjusted = delta_matrix * np.sqrt(
        np.maximum(conf_matrix, 0.0)
    )
    robust = np.median(adjusted, axis=0).astype(np.float32)

    # Family 4: demographic cohort preferences transfer to unseen and
    # low-evidence users rather than relying on their individual identity.
    cohort, _ = cohort_code(split)
    cohort_sum = np.zeros(n, dtype=np.float32)
    cohort_conf = np.zeros(n, dtype=np.float32)

    for name in COHORT_PREFERENCE_FIELDS:
        card = int(FEATURE_CARDINALITIES[name])
        delta, confidence = apply_pair(
            cohort,
            split.X[name],
            cohort_pair_tables[name],
            card,
        )
        cohort_sum += delta * confidence
        cohort_conf += confidence

    cohort_score = cohort_sum / np.maximum(cohort_conf, 0.35)

    outputs = {
        "global_content": global_score.astype(np.float32),
        "hierarchical_additive": (
            0.35 * global_score + 0.65 * additive
        ).astype(np.float32),
        "confidence_gated_expert": (
            0.30 * global_score + 0.70 * gated
        ).astype(np.float32),
        "robust_preference_median": (
            0.40 * global_score + 0.60 * robust
        ).astype(np.float32),
        "demographic_cohort": (
            0.35 * global_score + 0.65 * cohort_score
        ).astype(np.float32),
        "personal_cohort_hybrid": (
            0.25 * global_score +
            0.50 * additive +
            0.25 * cohort_score
        ).astype(np.float32),
    }

    del personal_deltas, personal_confidences
    del delta_matrix, conf_matrix, adjusted
    del cohort, cohort_sum, cohort_conf
    gc.collect()
    return outputs


inc_valid_path = (
    os.path.join(SHARED, "incumbent_valid_scores.npy")
    if SHARED else ""
)
inc_test_path = (
    os.path.join(SHARED, "incumbent_test_scores.npy")
    if SHARED else ""
)

if not (
    inc_valid_path
    and inc_test_path
    and os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise RuntimeError("Trusted incumbent artifacts are required")

valid = load("valid")
valid_uid = np.asarray(valid.user_id, dtype=np.int64)
valid_y = np.asarray(valid.y)

inc_valid = np.load(inc_valid_path, mmap_mode="r")
inc_valid_rank = within_user_rank(valid_uid, inc_valid)
inc_metrics = evaluate(valid_uid, valid_y, inc_valid_rank)

valid_families = score_preference_families(valid)

candidate_scores = {
    "trusted_incumbent": float(inc_metrics["primary"])
}

best_name = "trusted_incumbent"
best_family = None
best_alpha = 0.0
best_valid = inc_valid_rank.copy()
best_raw_valid = None
best_primary = float(inc_metrics["primary"])

# Rank-space fusion is invariant to incompatible probability calibration.
# Negative weights are included because a target statistic can identify
# exposure bias whose inverse is the useful correction.
ALPHAS = (
    -0.30, -0.20, -0.12, -0.08, -0.05, -0.03,
     0.02,  0.03,  0.05,  0.08,  0.12,  0.18,
     0.25,  0.35,  0.50,  0.70,
)

for family, raw_score in valid_families.items():
    raw_metrics = evaluate(valid_uid, valid_y, raw_score)
    raw_primary = float(raw_metrics["primary"])
    candidate_scores[family + "_standalone"] = raw_primary

    family_rank = within_user_rank(valid_uid, raw_score)
    rank_corr = float(np.corrcoef(inc_valid_rank, family_rank)[0, 1])

    local_best = -np.inf
    local_alpha = 0.0

    for alpha in ALPHAS:
        blended = (
            inc_valid_rank + float(alpha) * (family_rank - 0.5)
        ).astype(np.float32)
        metrics = evaluate(valid_uid, valid_y, blended)
        primary = float(metrics["primary"])

        if primary > local_best:
            local_best = primary
            local_alpha = float(alpha)

        if primary > best_primary:
            best_primary = primary
            best_name = "%s_blend_%+.2f" % (family, alpha)
            best_family = family
            best_alpha = float(alpha)
            best_valid = blended.copy()
            best_raw_valid = raw_score.copy()

    candidate_scores[family + "_best_blend"] = float(local_best)

    print(
        "FINDINGS family=%s standalone=%.6f corr_incumbent=%.6f "
        "best_alpha=%+.2f best_blend=%.6f"
        % (
            family,
            raw_primary,
            rank_corr,
            local_alpha,
            local_best,
        ),
        flush=True,
    )

final_metrics = evaluate(valid_uid, valid_y, best_valid)

print(
    "FINDINGS winner=%s alpha=%+.2f"
    % (best_name, best_alpha),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_scores, sort_keys=True),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    if best_family is not None:
        np.save(
            os.path.join(OUT, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

del valid_families, valid_y, valid, best_valid, best_raw_valid
del inc_valid, inc_valid_rank
gc.collect()

test = load("test")
inc_test = np.load(inc_test_path, mmap_mode="r")
inc_test_rank = within_user_rank(test.user_id, inc_test)

if best_family is None:
    test_scores = inc_test_rank
else:
    test_families = score_preference_families(test)
    raw_test = test_families[best_family]
    family_test_rank = within_user_rank(test.user_id, raw_test)
    test_scores = (
        inc_test_rank +
        best_alpha * (family_test_rank - 0.5)
    ).astype(np.float32)
    del test_families, raw_test, family_test_rank

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print(
    "METRICS " + json.dumps({
        "primary": float(final_metrics["primary"]),
        "gauc": float(final_metrics["gauc"]),
        "ndcg@5": float(final_metrics["ndcg@5"]),
        "gpu_seconds": elapsed,
    })
)