import os
import time
import json
import gc
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 20260831
np.random.seed(SEED)

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.float64)
yva = np.asarray(valid.y, dtype=np.int8)
utr = np.asarray(train.user_id, dtype=np.int64)
uva = np.asarray(valid.user_id, dtype=np.int64)
ute = np.asarray(test.user_id, dtype=np.int64)

n_users = int(FEATURE_CARDINALITIES["user_id"])

# Recency weighting is applied to the main train-only statistics. A four-day
# half-life emphasizes behavior nearest the deployment boundary.
dates = np.asarray(train.date, dtype=np.int64)
unique_dates = np.unique(dates)
day_index = np.searchsorted(unique_dates, dates)
age = (len(unique_dates) - 1 - day_index).astype(np.float64)
weights = np.exp2(-age / 4.0)
weights /= weights.mean()

global_prior = float(np.sum(weights * ytr) / np.sum(weights))
eps = 1.0e-6


def logit(p):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    rows = np.arange(len(scores), dtype=np.int64)

    order = np.lexsort((rows, scores, users))
    su = users[order]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    ends = np.r_[starts[1:], len(order)]
    lengths = ends - starts

    positions = (
        np.arange(len(order), dtype=np.float64)
        - np.repeat(starts, lengths)
    )
    denominators = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    ranked = positions / denominators

    result = np.empty(len(scores), dtype=np.float64)
    result[order] = ranked
    return result


# User support controls deterministic warm/sparse routing.
user_rows = np.bincount(
    utr, weights=weights, minlength=n_users
).astype(np.float64)
user_pos = np.bincount(
    utr, weights=weights * ytr, minlength=n_users
).astype(np.float64)

valid_support = user_rows[np.minimum(uva, n_users - 1)]
test_support = user_rows[np.minimum(ute, n_users - 1)]

# Smooth routing rather than a validation-selected threshold.
valid_gate = valid_support / (valid_support + 20.0)
test_gate = test_support / (test_support + 20.0)


GLOBAL_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

PERSONAL_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
]


def global_field_rate(field, strength=60.0):
    card = int(FEATURE_CARDINALITIES[field])
    x = np.asarray(train.X[field], dtype=np.int64)
    total = np.bincount(x, weights=weights, minlength=card)
    positive = np.bincount(x, weights=weights * ytr, minlength=card)
    rate = (positive + strength * global_prior) / (total + strength)
    return np.clip(rate, eps, 1.0 - eps), total, positive


global_rates = {}
global_totals = {}
global_positives = {}

for field in GLOBAL_FIELDS:
    r, c, p = global_field_rate(field)
    global_rates[field] = r
    global_totals[field] = c
    global_positives[field] = p


def global_score(sample):
    score = np.zeros(len(sample.user_id), dtype=np.float64)

    # Stable content fields receive most weight; exact video identity is
    # deliberately damped to reduce sensitivity to temporal identity drift.
    coefficients = {
        "video_id": 0.35,
        "author_id": 0.65,
        "tag": 0.80,
        "duration_bucket": 0.70,
        "upload_type": 0.40,
        "music_type": 0.30,
        "onehot_feat3": 0.45,
        "onehot_feat7": 0.30,
        "onehot_feat8": 0.40,
    }

    for field in GLOBAL_FIELDS:
        x = np.asarray(sample.X[field], dtype=np.int64)
        score += coefficients[field] * logit(global_rates[field][x])

    return score


global_valid = global_score(valid)
global_test = global_score(test)


# -------------------------------------------------------------------------
# Family 1: hierarchical user-content empirical Bayes.
#
# For each user and content value, estimate a recency-weighted long-view rate
# and shrink it toward that value's global rate. Only the personalized
# deviation is added to the global content model.
# -------------------------------------------------------------------------
def fit_cross_statistics(field):
    card = int(FEATURE_CARDINALITIES[field])
    x = np.asarray(train.X[field], dtype=np.int64)
    keys = utr.astype(np.int64) * np.int64(card) + x

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    totals = np.bincount(inverse, weights=weights)
    positives = np.bincount(inverse, weights=weights * ytr)

    return card, unique_keys, totals, positives


def lookup_cross_deviation(
    sample, field, fitted, prior_strength=12.0
):
    card, keys, totals, positives = fitted
    users = np.asarray(sample.user_id, dtype=np.int64)
    values = np.asarray(sample.X[field], dtype=np.int64)
    query = users * np.int64(card) + values

    positions = np.searchsorted(keys, query)
    found = positions < len(keys)
    safe_positions = np.minimum(positions, len(keys) - 1)
    found &= keys[safe_positions] == query

    prior = global_rates[field][values]
    posterior = prior.copy()

    if np.any(found):
        idx = safe_positions[found]
        posterior[found] = (
            positives[idx] + prior_strength * prior[found]
        ) / (totals[idx] + prior_strength)

    deviation = logit(posterior) - logit(prior)
    return np.clip(deviation, -3.5, 3.5)


hier_valid = global_valid.copy()
hier_test = global_test.copy()

hier_coefficients = {
    "video_id": 0.40,
    "author_id": 0.75,
    "tag": 1.00,
    "duration_bucket": 0.85,
    "upload_type": 0.55,
}

for field in PERSONAL_FIELDS:
    fitted = fit_cross_statistics(field)
    dv = lookup_cross_deviation(valid, field, fitted)
    dt = lookup_cross_deviation(test, field, fitted)

    # Sparse users remain close to global content evidence. Warm users can
    # express their own preferences.
    hier_valid += (
        valid_gate * hier_coefficients[field] * dv
    )
    hier_test += (
        test_gate * hier_coefficients[field] * dt
    )

    del fitted, dv, dt
    gc.collect()


# -------------------------------------------------------------------------
# Family 2: positive-profile likelihood ratio.
#
# This ignores exposure failures when forming a user's taste profile. It asks
# whether an attribute occurs among that user's positive history more often
# than expected from the global positive distribution. This differs from the
# outcome-rate estimator above and is robust when logged negatives are
# policy-dependent.
# -------------------------------------------------------------------------
PROFILE_FIELDS = [
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
]

profile_valid = 0.20 * global_valid
profile_test = 0.20 * global_test

total_weighted_positives = max(float(np.sum(weights * ytr)), eps)


def fit_positive_profile(field):
    card = int(FEATURE_CARDINALITIES[field])
    x = np.asarray(train.X[field], dtype=np.int64)
    keys = utr.astype(np.int64) * np.int64(card) + x

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    positive_counts = np.bincount(
        inverse, weights=weights * ytr
    )

    global_positive = np.bincount(
        x, weights=weights * ytr, minlength=card
    )
    global_positive_prob = (
        global_positive + 0.5
    ) / (
        total_weighted_positives + 0.5 * card
    )

    return card, unique_keys, positive_counts, global_positive_prob


def positive_profile_score(sample, fitted):
    card, keys, positive_counts, global_positive_prob = fitted
    users = np.asarray(sample.user_id, dtype=np.int64)
    values = None

    # Recover the field from the caller-supplied sample separately.
    return card, keys, positive_counts, global_positive_prob


profile_coefficients = {
    "author_id": 0.80,
    "tag": 1.10,
    "duration_bucket": 0.90,
    "upload_type": 0.55,
    "onehot_feat3": 0.50,
}

for field in PROFILE_FIELDS:
    card, keys, pos_counts, global_pos_prob = fit_positive_profile(field)

    def score_profile_for(sample):
        users = np.asarray(sample.user_id, dtype=np.int64)
        values = np.asarray(sample.X[field], dtype=np.int64)
        query = users * np.int64(card) + values

        positions = np.searchsorted(keys, query)
        found = positions < len(keys)
        safe_positions = np.minimum(positions, len(keys) - 1)
        found &= keys[safe_positions] == query

        observed = np.zeros(len(users), dtype=np.float64)
        observed[found] = pos_counts[safe_positions[found]]

        up = user_pos[np.minimum(users, n_users - 1)]
        expected = up * global_pos_prob[values]

        # Additive smoothing prevents rare positive coincidences from
        # dominating while retaining a genuine enrichment ratio.
        affinity = np.log(
            (observed + 0.75) / (expected + 0.75)
        )
        return np.clip(affinity, -2.5, 3.5)

    profile_valid += (
        valid_gate
        * profile_coefficients[field]
        * score_profile_for(valid)
    )
    profile_test += (
        test_gate
        * profile_coefficients[field]
        * score_profile_for(test)
    )

    del keys, pos_counts, global_pos_prob
    gc.collect()


# -------------------------------------------------------------------------
# Family 3: recent-positive nearest-profile scoring.
#
# Keep each user's five most recent positive train impressions and reward
# exact candidate matches at several semantic resolutions. This is a
# non-parametric nearest-history rule rather than an averaged target rate.
# -------------------------------------------------------------------------
RECENT_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
]

positive_rows = np.flatnonzero(ytr > 0.5)
positive_order = positive_rows[
    np.lexsort((
        positive_rows,
        np.asarray(train.time_ms, dtype=np.int64)[positive_rows],
        utr[positive_rows],
    ))
]

sorted_positive_users = utr[positive_order]
starts = np.flatnonzero(
    np.r_[True, sorted_positive_users[1:] != sorted_positive_users[:-1]]
)
ends = np.r_[starts[1:], len(positive_order)]
group_users = sorted_positive_users[starts]

recent_tables = {}
history_depth = 5

for field in RECENT_FIELDS:
    table = np.full(
        (n_users, history_depth),
        -1,
        dtype=np.int32,
    )
    values = np.asarray(train.X[field], dtype=np.int32)

    for depth in range(history_depth):
        idx = ends - 1 - depth
        valid_idx = idx >= starts
        if np.any(valid_idx):
            rows = positive_order[idx[valid_idx]]
            table[group_users[valid_idx], depth] = values[rows]

    recent_tables[field] = table


recent_valid = 0.30 * global_valid
recent_test = 0.30 * global_test

recent_coefficients = {
    "video_id": 1.10,
    "author_id": 0.85,
    "tag": 0.65,
    "duration_bucket": 0.35,
}
depth_weights = np.asarray(
    [1.00, 0.72, 0.52, 0.38, 0.28],
    dtype=np.float64,
)


def recent_match_score(sample):
    users = np.asarray(sample.user_id, dtype=np.int64)
    safe_users = np.minimum(users, n_users - 1)
    result = np.zeros(len(users), dtype=np.float64)

    for field in RECENT_FIELDS:
        candidate = np.asarray(sample.X[field], dtype=np.int32)
        history = recent_tables[field][safe_users]
        matches = history == candidate[:, None]
        result += recent_coefficients[field] * (
            matches * depth_weights[None, :]
        ).sum(axis=1)

    return result


recent_valid += valid_gate * recent_match_score(valid)
recent_test += test_gate * recent_match_score(test)

del recent_tables, positive_order, positive_rows
gc.collect()


# Rank-space ensembles prevent calibration differences between these
# structurally different score generators from controlling the mixture.
rank_global_valid = within_user_rank(uva, global_valid)
rank_hier_valid = within_user_rank(uva, hier_valid)
rank_profile_valid = within_user_rank(uva, profile_valid)
rank_recent_valid = within_user_rank(uva, recent_valid)

rank_global_test = within_user_rank(ute, global_test)
rank_hier_test = within_user_rank(ute, hier_test)
rank_profile_test = within_user_rank(ute, profile_test)
rank_recent_test = within_user_rank(ute, recent_test)

own_ensemble_valid = (
    0.15 * rank_global_valid
    + 0.40 * rank_hier_valid
    + 0.30 * rank_profile_valid
    + 0.15 * rank_recent_valid
)
own_ensemble_test = (
    0.15 * rank_global_test
    + 0.40 * rank_hier_test
    + 0.30 * rank_profile_test
    + 0.15 * rank_recent_test
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)

candidate_valid = {
    "global_content": rank_global_valid,
    "hierarchical_user_content": rank_hier_valid,
    "positive_profile_lr": rank_profile_valid,
    "recent_positive_profile": rank_recent_valid,
    "own_rank_ensemble": own_ensemble_valid,
}
candidate_test = {
    "global_content": rank_global_test,
    "hierarchical_user_content": rank_hier_test,
    "positive_profile_lr": rank_profile_test,
    "recent_positive_profile": rank_recent_test,
    "own_rank_ensemble": own_ensemble_test,
}

uses_incumbent = (
    os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
)

if uses_incumbent:
    incumbent_valid_raw = np.load(inc_valid_path).astype(np.float64)
    incumbent_test_raw = np.load(inc_test_path).astype(np.float64)

    incumbent_valid = within_user_rank(uva, incumbent_valid_raw)
    incumbent_test = within_user_rank(ute, incumbent_test_raw)

    candidate_valid["incumbent"] = incumbent_valid
    candidate_test["incumbent"] = incumbent_test

    # Compare each family with the trusted incumbent. The same selected
    # interpolation is applied unchanged to test.
    own_names = [
        "hierarchical_user_content",
        "positive_profile_lr",
        "recent_positive_profile",
        "own_rank_ensemble",
    ]
    blend_alphas = [0.15, 0.25, 0.35, 0.50]

    for name in own_names:
        for alpha in blend_alphas:
            blend_name = "%s_blend_%.2f" % (name, alpha)
            candidate_valid[blend_name] = (
                (1.0 - alpha) * incumbent_valid
                + alpha * candidate_valid[name]
            )
            candidate_test[blend_name] = (
                (1.0 - alpha) * incumbent_test
                + alpha * candidate_test[name]
            )


candidate_metrics = {}
for name, scores in candidate_valid.items():
    candidate_metrics[name] = evaluate(
        uva, yva, scores
    )

best_name = max(
    candidate_metrics,
    key=lambda name: candidate_metrics[name]["primary"],
)
best_valid = candidate_valid[best_name]
best_test = candidate_test[best_name]
best_metrics = candidate_metrics[best_name]

candidate_summary = {
    name: float(metrics["primary"])
    for name, metrics in candidate_metrics.items()
}
print("CANDIDATES " + json.dumps(
    candidate_summary, sort_keys=True
))

print(
    "FINDINGS selected=%s mean_user_support=%.4f "
    "sparse_fraction_lt5=%.4f warm_fraction_ge20=%.4f"
    % (
        best_name,
        float(valid_support.mean()),
        float(np.mean(valid_support < 5.0)),
        float(np.mean(valid_support >= 20.0)),
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )

    # The raw output records this script's own strongest combined estimator
    # whenever the reported winner incorporates trusted incumbent scores.
    if uses_incumbent and (
        best_name == "incumbent"
        or "_blend_" in best_name
    ):
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(own_ensemble_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, '
    '"ndcg@5": %.10f, "gpu_seconds": %.6f}'
    % (
        float(best_metrics["primary"]),
        float(best_metrics["gauc"]),
        float(best_metrics["ndcg@5"]),
        elapsed,
    )
)