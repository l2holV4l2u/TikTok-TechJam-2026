import os
import time
import json
import gc
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
EPS = 1e-8
HALF_LIFE = 4.0

PROFILE_FIELDS = [
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "video_type",
    "tab",
    "hour",
    "onehot_feat2",
    "onehot_feat7",
    "onehot_feat8",
    "is_video_author",
    "user_active_degree",
]


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates, dtype=np.int32)
    age = dates.max() - dates
    w = np.power(0.5, age.astype(np.float32) / float(half_life))
    w /= max(float(w.mean()), 1e-8)
    return w.astype(np.float64)


def tied_rank_percentile(user_ids, scores):
    """Within-user percentile ranks with exact ties receiving equal ranks."""
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores.copy()

    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, user_ids))
    u = user_ids[order]
    s = scores[order]

    user_start = np.empty(n, dtype=bool)
    user_start[0] = True
    user_start[1:] = u[1:] != u[:-1]

    user_start_idx = np.maximum.accumulate(
        np.where(user_start, np.arange(n, dtype=np.int64), 0)
    )

    user_end = np.empty(n, dtype=bool)
    user_end[-1] = True
    user_end[:-1] = u[:-1] != u[1:]
    user_end_idx = np.minimum.accumulate(
        np.where(
            user_end,
            np.arange(n, dtype=np.int64),
            n - 1,
        )[::-1]
    )[::-1]
    user_size = user_end_idx - user_start_idx + 1

    tie_start = np.empty(n, dtype=bool)
    tie_start[0] = True
    tie_start[1:] = (u[1:] != u[:-1]) | (s[1:] != s[:-1])
    tie_start_idx = np.maximum.accumulate(
        np.where(tie_start, np.arange(n, dtype=np.int64), 0)
    )

    tie_end = np.empty(n, dtype=bool)
    tie_end[-1] = True
    tie_end[:-1] = (u[:-1] != u[1:]) | (s[:-1] != s[1:])
    tie_end_idx = np.minimum.accumulate(
        np.where(
            tie_end,
            np.arange(n, dtype=np.int64),
            n - 1,
        )[::-1]
    )[::-1]

    midpoint = 0.5 * (
        (tie_start_idx - user_start_idx).astype(np.float64)
        + (tie_end_idx - user_start_idx).astype(np.float64)
    )
    ranked = (midpoint + 0.5) / user_size.astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def lookup_dense(values, keys, valid_mask):
    out = np.zeros(len(keys), dtype=np.float64)
    out[valid_mask] = values[keys[valid_mask]]
    return out


train = load("train")
valid = load("valid")
test = load("test")

y = np.asarray(train.y, dtype=np.float64)
weights = recency_weights(train.date, HALF_LIFE)
positive_weights = weights * y
negative_weights = weights * (1.0 - y)

train_users = np.asarray(train.user_id, dtype=np.int64)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)

n_users = max(
    int(FEATURE_CARDINALITIES.get("user_id", 0)),
    int(train_users.max(initial=0)) + 1,
    int(valid_users.max(initial=0)) + 1,
    int(test_users.max(initial=0)) + 1,
)

user_total = np.bincount(
    train_users, weights=weights, minlength=n_users
).astype(np.float64)
user_positive = np.bincount(
    train_users, weights=positive_weights, minlength=n_users
).astype(np.float64)
user_negative = np.bincount(
    train_users, weights=negative_weights, minlength=n_users
).astype(np.float64)

global_prior = float(positive_weights.sum() / max(weights.sum(), EPS))
user_prior = (user_positive + 20.0 * global_prior) / (
    user_total + 20.0
)

scores_valid = {
    "content_likelihood_ratio": np.zeros(len(valid_users), dtype=np.float64),
    "content_centered_rate": np.zeros(len(valid_users), dtype=np.float64),
    "content_signed_residual": np.zeros(len(valid_users), dtype=np.float64),
}
scores_test = {
    name: np.zeros(len(test_users), dtype=np.float64)
    for name in scores_valid
}

field_diagnostics = {}

for field in PROFILE_FIELDS:
    card = int(FEATURE_CARDINALITIES[field])
    tr_x = np.asarray(train.X[field], dtype=np.int64)
    va_x = np.asarray(valid.X[field], dtype=np.int64)
    te_x = np.asarray(test.X[field], dtype=np.int64)

    tr_known = (tr_x >= 0) & (tr_x < card)
    va_known = (
        (va_x >= 0) & (va_x < card)
        & (valid_users >= 0) & (valid_users < n_users)
    )
    te_known = (
        (te_x >= 0) & (te_x < card)
        & (test_users >= 0) & (test_users < n_users)
    )

    tr_key = train_users[tr_known] * card + tr_x[tr_known]
    table_size = n_users * card

    exposure = np.bincount(
        tr_key,
        weights=weights[tr_known],
        minlength=table_size,
    ).astype(np.float64)
    positive = np.bincount(
        tr_key,
        weights=positive_weights[tr_known],
        minlength=table_size,
    ).astype(np.float64)
    negative = exposure - positive

    global_exposure = np.bincount(
        tr_x[tr_known],
        weights=weights[tr_known],
        minlength=card,
    ).astype(np.float64)
    global_positive = np.bincount(
        tr_x[tr_known],
        weights=positive_weights[tr_known],
        minlength=card,
    ).astype(np.float64)
    global_negative = global_exposure - global_positive

    global_rate = (
        global_positive + 30.0 * global_prior
    ) / (global_exposure + 30.0)

    positive_distribution = (
        global_positive + 0.5
    ) / (global_positive.sum() + 0.5 * card)
    negative_distribution = (
        global_negative + 0.5
    ) / (global_negative.sum() + 0.5 * card)

    frequency = global_exposure / max(global_exposure.sum(), EPS)
    idf = np.log1p(1.0 / np.maximum(frequency, 1e-7))
    idf /= max(float(np.mean(idf)), EPS)

    def add_query(split_x, split_users, known, destination):
        keys = split_users * card + split_x

        count_q = lookup_dense(exposure, keys, known)
        pos_q = lookup_dense(positive, keys, known)
        neg_q = lookup_dense(negative, keys, known)

        safe_x = np.clip(split_x, 0, card - 1)
        gp = positive_distribution[safe_x]
        gn = negative_distribution[safe_x]
        gr = global_rate[safe_x]
        idf_q = idf[safe_x]

        valid_user = (
            (split_users >= 0) & (split_users < n_users)
        )
        safe_user = np.clip(split_users, 0, n_users - 1)

        up_total = np.where(
            valid_user, user_positive[safe_user], 0.0
        )
        un_total = np.where(
            valid_user, user_negative[safe_user], 0.0
        )
        local_prior = np.where(
            valid_user, user_prior[safe_user], global_prior
        )

        # Family 1: a user-specific positive-versus-negative content
        # likelihood ratio with global feature distributions as priors.
        alpha = 8.0
        pos_probability = (pos_q + alpha * gp) / (
            up_total + alpha
        )
        neg_probability = (neg_q + alpha * gn) / (
            un_total + alpha
        )
        likelihood = np.log(pos_probability + EPS) - np.log(
            neg_probability + EPS
        )
        reliability = np.sqrt(count_q / (count_q + 3.0))
        destination["content_likelihood_ratio"] += np.where(
            known, likelihood * (0.35 + 0.65 * reliability), 0.0
        )

        # Family 2: empirical-Bayes user-value response residual, centered
        # against the value's population response rate.
        smooth = 12.0
        posterior_rate = (pos_q + smooth * gr) / (
            count_q + smooth
        )
        centered = safe_logit(posterior_rate) - safe_logit(gr)
        centered *= np.sqrt(count_q / (count_q + smooth))
        destination["content_centered_rate"] += np.where(
            known, centered, 0.0
        )

        # Family 3: signed preference residual. It treats each user's
        # expected positives as a baseline and emphasizes rare attributes.
        residual = (pos_q - local_prior * count_q) / np.sqrt(
            count_q + 4.0
        )
        destination["content_signed_residual"] += np.where(
            known, residual * idf_q, 0.0
        )

    add_query(va_x, valid_users, va_known, scores_valid)
    add_query(te_x, test_users, te_known, scores_test)

    field_diagnostics[field] = {
        "cardinality": card,
        "valid_known_fraction": float(va_known.mean()),
        "mean_user_value_evidence": float(
            exposure[exposure > 0].mean()
        ) if np.any(exposure > 0) else 0.0,
    }

    del exposure, positive, negative
    gc.collect()


# A fourth, genuinely metric content model: compare the candidate's log
# duration to each user's train-period positive and negative prototypes.
tr_duration = np.log1p(np.maximum(
    np.nan_to_num(
        np.asarray(train.num["duration_ms"], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ),
    0.0,
))
va_duration = np.log1p(np.maximum(
    np.nan_to_num(
        np.asarray(valid.num["duration_ms"], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ),
    0.0,
))
te_duration = np.log1p(np.maximum(
    np.nan_to_num(
        np.asarray(test.num["duration_ms"], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ),
    0.0,
))

positive_duration_sum = np.bincount(
    train_users,
    weights=positive_weights * tr_duration,
    minlength=n_users,
).astype(np.float64)
negative_duration_sum = np.bincount(
    train_users,
    weights=negative_weights * tr_duration,
    minlength=n_users,
).astype(np.float64)

global_positive_duration = float(
    np.sum(positive_weights * tr_duration)
    / max(np.sum(positive_weights), EPS)
)
global_negative_duration = float(
    np.sum(negative_weights * tr_duration)
    / max(np.sum(negative_weights), EPS)
)

positive_duration_mean = (
    positive_duration_sum + 5.0 * global_positive_duration
) / (user_positive + 5.0)
negative_duration_mean = (
    negative_duration_sum + 5.0 * global_negative_duration
) / (user_negative + 5.0)

global_duration_scale = max(float(np.std(tr_duration)), 0.25)


def duration_prototype(query_users, query_duration):
    known_user = (query_users >= 0) & (query_users < n_users)
    safe_user = np.clip(query_users, 0, n_users - 1)
    pos_mean = np.where(
        known_user,
        positive_duration_mean[safe_user],
        global_positive_duration,
    )
    neg_mean = np.where(
        known_user,
        negative_duration_mean[safe_user],
        global_negative_duration,
    )
    evidence = np.where(
        known_user,
        np.sqrt(
            user_total[safe_user] / (user_total[safe_user] + 15.0)
        ),
        0.0,
    )
    return evidence * (
        np.abs(query_duration - neg_mean)
        - np.abs(query_duration - pos_mean)
    ) / global_duration_scale


scores_valid["duration_prototype_metric"] = duration_prototype(
    valid_users, va_duration
)
scores_test["duration_prototype_metric"] = duration_prototype(
    test_users, te_duration
)

# Combine the complementary categorical formulations before incumbent fusion.
profile_names = [
    "content_likelihood_ratio",
    "content_centered_rate",
    "content_signed_residual",
    "duration_prototype_metric",
]

valid_ranks = {
    name: tied_rank_percentile(valid_users, scores_valid[name])
    for name in profile_names
}
test_ranks = {
    name: tied_rank_percentile(test_users, scores_test[name])
    for name in profile_names
}

scores_valid["categorical_profile_ensemble"] = (
    0.45 * valid_ranks["content_likelihood_ratio"]
    + 0.30 * valid_ranks["content_centered_rate"]
    + 0.20 * valid_ranks["content_signed_residual"]
    + 0.05 * valid_ranks["duration_prototype_metric"]
)
scores_test["categorical_profile_ensemble"] = (
    0.45 * test_ranks["content_likelihood_ratio"]
    + 0.30 * test_ranks["content_centered_rate"]
    + 0.20 * test_ranks["content_signed_residual"]
    + 0.05 * test_ranks["duration_prototype_metric"]
)

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

if len(inc_valid) != len(valid_users) or len(inc_test) != len(test_users):
    raise RuntimeError("Incumbent prediction length mismatch")

inc_valid_rank = tied_rank_percentile(valid_users, inc_valid)
inc_test_rank = tied_rank_percentile(test_users, inc_test)

candidate_valid = {"trusted_incumbent": inc_valid}
candidate_test = {"trusted_incumbent": inc_test}
candidate_raw_family = {"trusted_incumbent": "categorical_profile_ensemble"}

all_own_names = profile_names + ["categorical_profile_ensemble"]

for name in all_own_names:
    own_va = np.asarray(scores_valid[name], dtype=np.float64)
    own_te = np.asarray(scores_test[name], dtype=np.float64)

    candidate_valid[name + "_standalone"] = own_va
    candidate_test[name + "_standalone"] = own_te
    candidate_raw_family[name + "_standalone"] = name

    own_va_rank = tied_rank_percentile(valid_users, own_va)
    own_te_rank = tied_rank_percentile(test_users, own_te)

    for alpha in (0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.40):
        candidate_name = f"{name}_incumbent_blend_{alpha:.2f}"
        candidate_valid[candidate_name] = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * own_va_rank
        )
        candidate_test[candidate_name] = (
            (1.0 - alpha) * inc_test_rank
            + alpha * own_te_rank
        )
        candidate_raw_family[candidate_name] = name

candidate_metrics = {}
for name, score in candidate_valid.items():
    candidate_metrics[name] = evaluate(
        valid.user_id, valid.y, score
    )

best_name = max(
    candidate_metrics,
    key=lambda n: float(candidate_metrics[n]["primary"]),
)
best_metrics = candidate_metrics[best_name]
best_valid = np.asarray(candidate_valid[best_name], dtype=np.float64)
best_test = np.asarray(candidate_test[best_name], dtype=np.float64)
raw_name = candidate_raw_family[best_name]
raw_valid = np.asarray(scores_valid[raw_name], dtype=np.float64)

print("CANDIDATES " + json.dumps(
    {
        name: float(metrics["primary"])
        for name, metrics in candidate_metrics.items()
    },
    sort_keys=True,
))
print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "best_raw_family": raw_name,
    "half_life_days": HALF_LIFE,
    "profile_fields": PROFILE_FIELDS,
    "global_train_prior": global_prior,
    "field_diagnostics": field_diagnostics,
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        best_valid,
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        best_test,
    )
    if best_name == "trusted_incumbent" or "blend" in best_name:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            raw_valid,
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))