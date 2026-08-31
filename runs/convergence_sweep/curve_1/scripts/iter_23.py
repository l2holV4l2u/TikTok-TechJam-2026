import os
import time
import json
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
warnings.filterwarnings("ignore")
np.random.seed(314159)

train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.int8)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.load(inc_valid_path).astype(np.float64)


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

    ranked = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ranked[multi] = (
        positions[multi]
        / (repeated_lengths[multi].astype(np.float64) - 1.0)
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def rank_blend(user_ids, left, right, alpha):
    return (
        (1.0 - alpha) * within_user_rank(user_ids, left)
        + alpha * within_user_rank(user_ids, right)
    )


def date_recency_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    age = unique_dates.size - 1 - np.searchsorted(unique_dates, dates)
    return np.exp2(-age.astype(np.float64) / half_life)


def fit_smoothed_rate(ids, labels, weights, cardinality, prior_strength):
    ids = np.asarray(ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    total = np.bincount(ids, weights=weights, minlength=cardinality)
    positive = np.bincount(
        ids, weights=weights * labels, minlength=cardinality
    )

    global_rate = float(np.sum(weights * labels) / np.sum(weights))
    rate = (
        positive + prior_strength * global_rate
    ) / (total + prior_strength)

    return np.log(
        np.clip(rate, 1e-5, 1.0 - 1e-5)
        / np.clip(1.0 - rate, 1e-5, 1.0)
    ).astype(np.float32)


BAYES_FIELDS = [
    ("video_id", 0.44, 25.0),
    ("author_id", 0.26, 35.0),
    ("tag", 0.10, 45.0),
    ("duration_bucket", 0.10, 55.0),
    ("tab", 0.10, 80.0),
]


def fit_bayes_model(split, labels, row_mask, half_life):
    dates = np.asarray(split.date, dtype=np.int32)[row_mask]
    weights = date_recency_weights(dates, half_life)
    labels = np.asarray(labels, dtype=np.int8)[row_mask]

    tables = {}
    for field, _, prior in BAYES_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)[row_mask]
        tables[field] = fit_smoothed_rate(
            ids,
            labels,
            weights,
            FEATURE_CARDINALITIES[field],
            prior,
        )
    return tables


def predict_bayes(split, tables):
    result = np.zeros(len(split.user_id), dtype=np.float64)
    for field, coefficient, _ in BAYES_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        table = tables[field]
        safe = np.minimum(ids, table.size - 1)
        result += coefficient * table[safe]
    return result


# Select temporal weighting on a train-only proxy window. The last three
# training days imitate a later date-split without fitting on validation.
train_dates = np.asarray(train.date, dtype=np.int32)
unique_train_dates = np.unique(train_dates)
proxy_dates = unique_train_dates[-3:]
proxy_mask = np.isin(train_dates, proxy_dates)
proxy_fit_mask = ~proxy_mask
proxy_users = np.asarray(train.user_id, dtype=np.int64)[proxy_mask]
proxy_labels = train_y[proxy_mask]

half_lives = [1.5, 3.0, 6.0, 12.0]
proxy_results = {}

for half_life in half_lives:
    model = fit_bayes_model(
        train, train_y, proxy_fit_mask, half_life
    )
    proxy_scores = predict_bayes(train, model)[proxy_mask]
    proxy_metric = evaluate(proxy_users, proxy_labels, proxy_scores)
    proxy_results[half_life] = float(proxy_metric["primary"])

chosen_half_life = max(proxy_results, key=proxy_results.get)
print(
    "FINDINGS proxy_half_lives="
    + json.dumps(
        {str(k): round(v, 6) for k, v in proxy_results.items()},
        sort_keys=True,
    )
)
print(
    "FINDINGS selected_half_life=%.1f proxy_primary=%.6f"
    % (chosen_half_life, proxy_results[chosen_half_life])
)

full_mask = np.ones(train_y.size, dtype=bool)
bayes_model = fit_bayes_model(
    train, train_y, full_mask, chosen_half_life
)
bayes_valid = predict_bayes(valid, bayes_model)


def fit_supervised_svd(split, labels, rank=48):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float32)

    weights = date_recency_weights(split.date, half_life=6.0).astype(
        np.float32
    )

    # Positives carry greater magnitude, while unsuccessful exposures supply
    # explicit negative evidence. Duplicate user-video rows are summed by CSR.
    values = weights * np.where(labels > 0.5, 1.0, -0.32).astype(
        np.float32
    )

    matrix = sp.coo_matrix(
        (values, (users, videos)),
        shape=(
            FEATURE_CARDINALITIES["user_id"],
            FEATURE_CARDINALITIES["video_id"],
        ),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()

    # Remove each user's average observed preference so factors emphasize
    # item ordering rather than activity or prevalence.
    binary = matrix.copy()
    binary.data = np.ones_like(binary.data)
    row_n = np.asarray(binary.sum(axis=1)).reshape(-1)
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    row_mean = row_sum / np.maximum(row_n, 1.0)

    centered = matrix.copy()
    centered.data -= np.repeat(row_mean, np.diff(centered.indptr)).astype(
        np.float32
    )
    centered.eliminate_zeros()

    actual_rank = min(rank, min(centered.shape) - 1)
    try:
        left, singular, right_t = svds(
            centered.astype(np.float64),
            k=actual_rank,
            which="LM",
            return_singular_vectors=True,
            random_state=314159,
            maxiter=700,
            tol=2e-3,
        )
        order = np.argsort(singular)[::-1]
        singular = singular[order]
        left = left[:, order]
        right_t = right_t[order]

        root = np.sqrt(np.maximum(singular, 0.0))
        user_factors = left * root[None, :]
        item_factors = right_t.T * root[None, :]
    except Exception as exc:
        print("FINDINGS svd_fallback=" + repr(exc))
        user_factors = np.zeros(
            (centered.shape[0], 1), dtype=np.float64
        )
        item_factors = np.zeros(
            (centered.shape[1], 1), dtype=np.float64
        )

    return (
        user_factors.astype(np.float32),
        item_factors.astype(np.float32),
    )


def predict_svd(split, user_factors, item_factors):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)

    safe_users = np.minimum(users, user_factors.shape[0] - 1)
    safe_videos = np.minimum(videos, item_factors.shape[0] - 1)

    result = np.einsum(
        "ij,ij->i",
        user_factors[safe_users],
        item_factors[safe_videos],
        optimize=True,
    ).astype(np.float64)

    cold = users >= user_factors.shape[0]
    result[cold] = 0.0
    return result


svd_users, svd_items = fit_supervised_svd(train, train_y, rank=48)
svd_valid = predict_svd(valid, svd_users, svd_items)


def grouped_mean(keys, values):
    keys = np.asarray(keys, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)

    unique_keys, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True
    )
    del unique_keys
    totals = np.bincount(inverse, weights=values)
    return totals[inverse] / counts[inverse]


def exposure_set_graph_score(split, base_scores):
    users = np.asarray(split.user_id, dtype=np.int64)
    base_rank = within_user_rank(users, base_scores)

    author = np.asarray(split.X["author_id"], dtype=np.int64)
    video = np.asarray(split.X["video_id"], dtype=np.int64)
    tag = np.asarray(split.X["tag"], dtype=np.int64)
    duration = np.asarray(split.X["duration_bucket"], dtype=np.int64)

    author_key = (
        users * FEATURE_CARDINALITIES["author_id"] + author
    )
    video_key = users * FEATURE_CARDINALITIES["video_id"] + video
    tag_key = users * FEATURE_CARDINALITIES["tag"] + tag
    duration_key = (
        users * FEATURE_CARDINALITIES["duration_bucket"] + duration
    )

    author_consensus = grouped_mean(author_key, base_rank)
    video_consensus = grouped_mean(video_key, base_rank)
    tag_consensus = grouped_mean(tag_key, base_rank)
    duration_consensus = grouped_mean(duration_key, base_rank)

    # Message passing over repeated entities in the current logged slate.
    return (
        0.52 * base_rank
        + 0.22 * author_consensus
        + 0.12 * video_consensus
        + 0.08 * tag_consensus
        + 0.06 * duration_consensus
    )


graph_valid = exposure_set_graph_score(valid, inc_valid)

# A cross-family aggregate can cancel unstable entity popularity with smoother
# latent affinities before blending against the trusted incumbent.
bayes_svd_valid = (
    0.55 * within_user_rank(valid_users, bayes_valid)
    + 0.45 * within_user_rank(valid_users, svd_valid)
)

raw_families_valid = {
    "bayes_drift": bayes_valid,
    "supervised_svd": svd_valid,
    "exposure_graph": graph_valid,
    "bayes_svd_cross_family": bayes_svd_valid,
}

candidate_scores = {}
candidate_arrays = {}

for family_name, family_scores in raw_families_valid.items():
    raw_metric = evaluate(valid_users, valid_y, family_scores)
    raw_name = family_name + "_raw"
    candidate_scores[raw_name] = float(raw_metric["primary"])
    candidate_arrays[raw_name] = family_scores

    for alpha in (0.10, 0.20, 0.35, 0.50, 0.70):
        blended = rank_blend(
            valid_users, inc_valid, family_scores, alpha
        )
        metric = evaluate(valid_users, valid_y, blended)
        name = "%s_incblend_%.2f" % (family_name, alpha)
        candidate_scores[name] = float(metric["primary"])
        candidate_arrays[name] = blended

inc_metric = evaluate(valid_users, valid_y, inc_valid)
candidate_scores["trusted_incumbent"] = float(inc_metric["primary"])
candidate_arrays["trusted_incumbent"] = inc_valid

winner_name = max(candidate_scores, key=candidate_scores.get)
winner_valid = candidate_arrays[winner_name]
winner_metrics = evaluate(valid_users, valid_y, winner_valid)

raw_names = [name for name in candidate_scores if name.endswith("_raw")]
best_raw_name = max(raw_names, key=lambda name: candidate_scores[name])
best_raw_valid = candidate_arrays[best_raw_name]

print(
    "CANDIDATES "
    + json.dumps(
        {
            name: round(score, 6)
            for name, score in sorted(candidate_scores.items())
        },
        sort_keys=True,
    )
)
print(
    "FINDINGS winner=%s best_raw=%s"
    % (winner_name, best_raw_name)
)

# Test features and incumbent predictions are read only after all validation
# model and blend selection has completed.
test = load("test")
test_users = np.asarray(test.user_id, dtype=np.int64)
inc_test = np.load(inc_test_path).astype(np.float64)

bayes_test = predict_bayes(test, bayes_model)
svd_test = predict_svd(test, svd_users, svd_items)
graph_test = exposure_set_graph_score(test, inc_test)
bayes_svd_test = (
    0.55 * within_user_rank(test_users, bayes_test)
    + 0.45 * within_user_rank(test_users, svd_test)
)

raw_families_test = {
    "bayes_drift": bayes_test,
    "supervised_svd": svd_test,
    "exposure_graph": graph_test,
    "bayes_svd_cross_family": bayes_svd_test,
}

if winner_name == "trusted_incumbent":
    winner_test = inc_test
elif winner_name.endswith("_raw"):
    family = winner_name[:-4]
    winner_test = raw_families_test[family]
else:
    marker = "_incblend_"
    family, alpha_text = winner_name.split(marker)
    alpha = float(alpha_text)
    winner_test = rank_blend(
        test_users, inc_test, raw_families_test[family], alpha
    )

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(winner_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(winner_test, dtype=np.float64),
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
            "primary": float(winner_metrics["primary"]),
            "gauc": float(winner_metrics["gauc"]),
            "ndcg@5": float(winner_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)