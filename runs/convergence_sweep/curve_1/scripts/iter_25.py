import os
import time
import json
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.spatial import cKDTree

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
warnings.filterwarnings("ignore")
np.random.seed(314159)

train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.float64)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.load(inc_valid_path).astype(np.float64)
inc_test = np.load(inc_test_path).astype(np.float64)


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

    group_starts = np.repeat(starts, lengths)
    group_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.float64) - group_starts

    ranks = np.full(n, 0.5, dtype=np.float64)
    multi = group_lengths > 1
    ranks[multi] = (
        positions[multi]
        / (group_lengths[multi].astype(np.float64) - 1.0)
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranks
    return result


def rank_blend(user_ids, incumbent, candidate, alpha):
    return (
        (1.0 - alpha) * within_user_rank(user_ids, incumbent)
        + alpha * within_user_rank(user_ids, candidate)
    )


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def recency_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    ages = unique_dates.size - 1 - np.searchsorted(unique_dates, dates)
    return np.exp2(-ages.astype(np.float64) / float(half_life))


# ---------------------------------------------------------------------
# Family 1: robust across-day empirical Bayes.
#
# Each entity receives one shrunk estimate per training day. Prediction
# uses the recency-weighted center of daily log-odds and penalizes entities
# whose apparent quality changes strongly between days. This deliberately
# favors stable video/content signals across the date boundary.
# ---------------------------------------------------------------------

ROBUST_FIELDS = (
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "onehot_feat3",
    "tab",
)
ROBUST_COEFFICIENTS = {
    "video_id": 0.52,
    "author_id": 0.25,
    "tag": 0.09,
    "duration_bucket": 0.08,
    "onehot_feat3": 0.06,
    "tab": 0.10,
}
ROBUST_PRIORS = {
    "video_id": 22.0,
    "author_id": 28.0,
    "tag": 45.0,
    "duration_bucket": 55.0,
    "onehot_feat3": 40.0,
    "tab": 100.0,
}


def fit_robust_daily_bayes(split, labels):
    labels = np.asarray(labels, dtype=np.float64)
    dates = np.asarray(split.date, dtype=np.int32)
    unique_dates = np.unique(dates)
    n_days = unique_dates.size
    day_index = np.searchsorted(unique_dates, dates)

    day_count = np.bincount(day_index, minlength=n_days).astype(np.float64)
    day_positive = np.bincount(
        day_index, weights=labels, minlength=n_days
    ).astype(np.float64)
    day_rate = (day_positive + 10.0) / (day_count + 20.0)

    recent_day_weight = np.exp2(
        -(n_days - 1 - np.arange(n_days, dtype=np.float64)) / 4.0
    )
    recent_day_weight /= recent_day_weight.sum()

    fitted = {}
    stability_summary = {}

    for field in ROBUST_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        cardinality = FEATURE_CARDINALITIES[field]
        prior = ROBUST_PRIORS[field]

        joint = day_index.astype(np.int64) * cardinality + ids
        size = n_days * cardinality

        counts = np.bincount(joint, minlength=size).astype(np.float64)
        positives = np.bincount(
            joint, weights=labels, minlength=size
        ).astype(np.float64)

        counts = counts.reshape(n_days, cardinality)
        positives = positives.reshape(n_days, cardinality)

        rates = (
            positives + prior * day_rate[:, None]
        ) / (counts + prior)
        daily_logits = logit(rates)

        center = np.sum(
            recent_day_weight[:, None] * daily_logits, axis=0
        )
        variance = np.sum(
            recent_day_weight[:, None]
            * (daily_logits - center[None, :]) ** 2,
            axis=0,
        )
        instability = np.sqrt(np.maximum(variance, 0.0))

        total_count = counts.sum(axis=0)
        reliability = total_count / (total_count + 35.0)

        # Shrink unstable entities toward zero log-odds contribution.
        stable_score = center / (1.0 + 0.65 * instability)
        stable_score *= reliability
        fitted[field] = stable_score.astype(np.float32)

        observed = total_count >= 20
        stability_summary[field] = float(
            instability[observed].mean()
        ) if np.any(observed) else 0.0

    return fitted, stability_summary


def predict_robust_daily(split, fitted):
    n = len(split.user_id)
    result = np.zeros(n, dtype=np.float64)
    for field in ROBUST_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        result += ROBUST_COEFFICIENTS[field] * fitted[field][ids]
    return result


robust_model, stability = fit_robust_daily_bayes(train, train_y)
robust_valid = predict_robust_daily(valid, robust_model)

print(
    "FINDINGS robust_daily_instability="
    + json.dumps(stability, sort_keys=True)
)


# ---------------------------------------------------------------------
# Family 2: propensity-standardized entity quality.
#
# Tab strongly changes both exposure and label rate. For each row, estimate
# p(tab) / p(tab | video), clipped for variance control, then estimate video
# and author outcomes under the common marginal tab mixture. This targets
# quality rather than quality confounded with a drifting exposure channel.
# ---------------------------------------------------------------------

def fit_propensity_standardized(split, labels):
    labels = np.asarray(labels, dtype=np.float64)
    videos = np.asarray(split.X["video_id"], dtype=np.int64)
    authors = np.asarray(split.X["author_id"], dtype=np.int64)
    tabs = np.asarray(split.X["tab"], dtype=np.int64)
    tags = np.asarray(split.X["tag"], dtype=np.int64)

    nv = FEATURE_CARDINALITIES["video_id"]
    na = FEATURE_CARDINALITIES["author_id"]
    nt = FEATURE_CARDINALITIES["tab"]
    ng = FEATURE_CARDINALITIES["tag"]

    base_weight = recency_weights(split.date, half_life=5.0)

    tab_count = np.bincount(
        tabs, weights=base_weight, minlength=nt
    ).astype(np.float64)
    tab_prob = (tab_count + 1.0) / (tab_count.sum() + nt)

    video_count = np.bincount(
        videos, weights=base_weight, minlength=nv
    ).astype(np.float64)
    tab_video_joint = tabs.astype(np.int64) * nv + videos
    tab_video_count = np.bincount(
        tab_video_joint,
        weights=base_weight,
        minlength=nt * nv,
    ).reshape(nt, nv)

    # Smoothed p(tab | video), backed off to the marginal tab mixture.
    conditional_tab = (
        tab_video_count + 8.0 * tab_prob[:, None]
    ) / (video_count[None, :] + 8.0)

    propensity_weight = (
        tab_prob[tabs] / np.maximum(conditional_tab[tabs, videos], 1e-6)
    )
    propensity_weight = np.clip(propensity_weight, 0.20, 5.0)
    weights = base_weight * propensity_weight

    global_rate = float(np.sum(weights * labels) / np.sum(weights))

    def rate_for(ids, cardinality, prior):
        count = np.bincount(
            ids, weights=weights, minlength=cardinality
        ).astype(np.float64)
        positive = np.bincount(
            ids, weights=weights * labels, minlength=cardinality
        ).astype(np.float64)
        rate = (positive + prior * global_rate) / (count + prior)
        return logit(rate).astype(np.float32), count

    video_score, video_effective_count = rate_for(videos, nv, 35.0)
    author_score, _ = rate_for(authors, na, 45.0)
    tag_score, _ = rate_for(tags, ng, 65.0)

    diagnostics = {
        "ips_mean": float(propensity_weight.mean()),
        "ips_p99": float(np.quantile(propensity_weight, 0.99)),
        "videos_effective_ge20": float(
            np.mean(video_effective_count >= 20.0)
        ),
    }
    return {
        "video": video_score,
        "author": author_score,
        "tag": tag_score,
        "diagnostics": diagnostics,
    }


def predict_propensity(split, model):
    videos = np.asarray(split.X["video_id"], dtype=np.int64)
    authors = np.asarray(split.X["author_id"], dtype=np.int64)
    tags = np.asarray(split.X["tag"], dtype=np.int64)
    return (
        0.62 * model["video"][videos]
        + 0.28 * model["author"][authors]
        + 0.10 * model["tag"][tags]
    ).astype(np.float64)


propensity_model = fit_propensity_standardized(train, train_y)
propensity_valid = predict_propensity(valid, propensity_model)

print(
    "FINDINGS propensity_standardization="
    + json.dumps(propensity_model["diagnostics"], sort_keys=True)
)


# ---------------------------------------------------------------------
# Family 3: user-user neighborhood collaborative filtering.
#
# Positive histories are projected into a low-dimensional latent space only
# to find nearby users. Prediction itself is non-parametric: it aggregates
# the actual recency-weighted positive videos of neighboring users. This can
# transfer preferences to sparse evaluation users without imposing one
# global user-video bilinear scoring rule.
# ---------------------------------------------------------------------

def fit_user_neighbor_cf(split, labels, rank=24, neighbors=24):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)

    nu = FEATURE_CARDINALITIES["user_id"]
    nv = FEATURE_CARDINALITIES["video_id"]

    positive = labels > 0.5
    recency = recency_weights(split.date, half_life=6.0)
    values = recency[positive].astype(np.float32)

    interaction = sp.coo_matrix(
        (values, (users[positive], videos[positive])),
        shape=(nu, nv),
        dtype=np.float32,
    ).tocsr()
    interaction.sum_duplicates()

    row_norm = np.sqrt(
        np.asarray(interaction.multiply(interaction).sum(axis=1)).ravel()
    )
    inverse_norm = np.zeros_like(row_norm, dtype=np.float32)
    nonzero = row_norm > 0
    inverse_norm[nonzero] = 1.0 / row_norm[nonzero]
    normalized = sp.diags(inverse_norm).dot(interaction).tocsr()

    u, singular, _ = svds(
        normalized.astype(np.float64),
        k=rank,
        which="LM",
        return_singular_vectors=True,
        random_state=314159,
    )
    order = np.argsort(singular)[::-1]
    latent = u[:, order] * singular[order][None, :]

    latent_norm = np.linalg.norm(latent, axis=1)
    usable = latent_norm > 1e-10
    latent[usable] /= latent_norm[usable, None]

    usable_rows = np.flatnonzero(usable)
    tree = cKDTree(latent[usable_rows])
    query_k = min(neighbors + 1, usable_rows.size)
    distances, local_neighbors = tree.query(
        latent[usable_rows], k=query_k, workers=-1
    )

    if query_k == 1:
        distances = distances[:, None]
        local_neighbors = local_neighbors[:, None]

    neighbor_ids = usable_rows[local_neighbors]
    similarities = 1.0 - 0.5 * np.square(distances)
    similarities = np.maximum(similarities, 0.0)

    # Remove the query user itself regardless of where ties place it.
    self_mask = neighbor_ids == usable_rows[:, None]
    similarities[self_mask] = 0.0
    similarities = similarities[:, :neighbors]
    neighbor_ids = neighbor_ids[:, :neighbors]

    row_sums = similarities.sum(axis=1)
    valid_sum = row_sums > 0
    similarities[valid_sum] /= row_sums[valid_sum, None]

    rows = np.repeat(usable_rows, similarities.shape[1])
    cols = neighbor_ids.reshape(-1)
    vals = similarities.reshape(-1)
    keep = vals > 0

    neighbor_matrix = sp.coo_matrix(
        (vals[keep], (rows[keep], cols[keep])),
        shape=(nu, nu),
        dtype=np.float32,
    ).tocsr()

    # Sparse multiplication keeps only videos present in neighbor histories.
    neighbor_video_score = neighbor_matrix.dot(interaction).tocsr()

    diagnostics = {
        "users_with_positive_history": int(usable.sum()),
        "neighbor_edges": int(neighbor_matrix.nnz),
        "predicted_user_video_pairs": int(neighbor_video_score.nnz),
    }
    return neighbor_video_score, diagnostics


def predict_user_neighbor(split, score_matrix):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    return np.asarray(score_matrix[users, videos]).reshape(-1).astype(
        np.float64
    )


neighbor_matrix, neighbor_diagnostics = fit_user_neighbor_cf(
    train, train_y, rank=24, neighbors=24
)
neighbor_valid = predict_user_neighbor(valid, neighbor_matrix)

print(
    "FINDINGS user_neighbor_cf="
    + json.dumps(neighbor_diagnostics, sort_keys=True)
)


# ---------------------------------------------------------------------
# Compare every standalone family and its rank blend with the incumbent.
# ---------------------------------------------------------------------

valid_raw_candidates = {
    "robust_daily_bayes": robust_valid,
    "propensity_standardized": propensity_valid,
    "user_neighbor_cf": neighbor_valid,
}

candidate_log = {}
standalone_metrics = {}

for name, scores in valid_raw_candidates.items():
    metrics = evaluate(valid_users, valid_y, scores)
    standalone_metrics[name] = metrics
    candidate_log[name + "_standalone"] = float(metrics["primary"])

blend_alphas = np.array(
    [0.0, 0.04, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.70, 1.0],
    dtype=np.float64,
)

best_name = None
best_alpha = None
best_valid_scores = None
best_metrics = None
best_primary = -np.inf

for name, raw_scores in valid_raw_candidates.items():
    family_best = -np.inf
    family_best_alpha = 0.0

    for alpha in blend_alphas:
        blended = rank_blend(
            valid_users, inc_valid, raw_scores, float(alpha)
        )
        metrics = evaluate(valid_users, valid_y, blended)
        primary = float(metrics["primary"])

        if primary > family_best:
            family_best = primary
            family_best_alpha = float(alpha)

        if primary > best_primary:
            best_primary = primary
            best_name = name
            best_alpha = float(alpha)
            best_valid_scores = blended
            best_metrics = metrics

    candidate_log[name + "_best_blend"] = float(family_best)
    candidate_log[name + "_best_alpha"] = float(family_best_alpha)

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print(
    "FINDINGS selected_family=%s selected_candidate_weight=%.3f"
    % (best_name, best_alpha)
)


# Load test only after all fitting and validation selection is complete.
test = load("test")
test_users = np.asarray(test.user_id, dtype=np.int64)

if best_name == "robust_daily_bayes":
    best_test_raw = predict_robust_daily(test, robust_model)
elif best_name == "propensity_standardized":
    best_test_raw = predict_propensity(test, propensity_model)
elif best_name == "user_neighbor_cf":
    best_test_raw = predict_user_neighbor(test, neighbor_matrix)
else:
    raise RuntimeError("Unknown selected family")

best_test_scores = rank_blend(
    test_users, inc_test, best_test_raw, best_alpha
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(valid_raw_candidates[best_name], dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test_scores, dtype=np.float64),
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