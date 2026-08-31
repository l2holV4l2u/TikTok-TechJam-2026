import os
import time
import json
import gc

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()


def within_user_rank(scores, users):
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users, dtype=np.int64)
    n = len(scores)
    if n == 0:
        return scores.copy()

    scores = np.nan_to_num(
        scores, nan=0.0, posinf=1e20, neginf=-1e20
    )
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, scores, users))
    su = users[order]
    ss = scores[order]
    positions = np.arange(n, dtype=np.int64)

    user_start_flag = np.empty(n, dtype=bool)
    user_start_flag[0] = True
    user_start_flag[1:] = su[1:] != su[:-1]
    user_starts = np.maximum.accumulate(
        np.where(user_start_flag, positions, 0)
    )

    user_end_flag = np.empty(n, dtype=bool)
    user_end_flag[-1] = True
    user_end_flag[:-1] = su[:-1] != su[1:]
    user_ends = np.minimum.accumulate(
        np.where(user_end_flag, positions, n - 1)[::-1]
    )[::-1]

    tie_start_flag = np.empty(n, dtype=bool)
    tie_start_flag[0] = True
    tie_start_flag[1:] = (
        (su[1:] != su[:-1]) | (ss[1:] != ss[:-1])
    )
    tie_starts = np.maximum.accumulate(
        np.where(tie_start_flag, positions, 0)
    )

    tie_end_flag = np.empty(n, dtype=bool)
    tie_end_flag[-1] = True
    tie_end_flag[:-1] = (
        (su[:-1] != su[1:]) | (ss[:-1] != ss[1:])
    )
    tie_ends = np.minimum.accumulate(
        np.where(tie_end_flag, positions, n - 1)[::-1]
    )[::-1]

    local_average = 0.5 * (tie_starts + tie_ends) - user_starts
    denominator = np.maximum(user_ends - user_starts, 1)
    ranked_sorted = local_average / denominator

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


def paired_sparse_lookup(matrix, rows, cols):
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    return np.asarray(matrix[rows, cols]).ravel().astype(
        np.float32, copy=False
    )


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def recency_weights(dates, half_life=5.0):
    dates = np.asarray(dates, dtype=np.int32)
    age = np.max(dates) - dates
    return np.power(0.5, age.astype(np.float64) / half_life).astype(
        np.float32
    )


def fit_signed_content_profiles(train, fields, weights, shrinkage=10.0):
    users = np.asarray(train.user_id, dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.float32)
    n_users = int(FEATURE_CARDINALITIES["user_id"])
    models = {}

    for field in fields:
        values = np.asarray(train.X[field], dtype=np.int64)
        cardinality = int(FEATURE_CARDINALITIES[field])

        total_value = np.bincount(
            values, weights=weights, minlength=cardinality
        ).astype(np.float64)
        positive_value = np.bincount(
            values, weights=weights * labels, minlength=cardinality
        ).astype(np.float64)

        global_mean = float(
            np.sum(weights * labels) / np.maximum(np.sum(weights), 1e-8)
        )
        value_rate = (
            positive_value + 20.0 * global_mean
        ) / np.maximum(total_value + 20.0, 1e-8)

        residual = weights * (labels - value_rate[values]).astype(
            np.float32
        )

        numerator = sp.coo_matrix(
            (
                residual,
                (users.astype(np.int32), values.astype(np.int32)),
            ),
            shape=(n_users, cardinality),
            dtype=np.float32,
        ).tocsr()
        numerator.sum_duplicates()

        denominator = sp.coo_matrix(
            (
                weights,
                (users.astype(np.int32), values.astype(np.int32)),
            ),
            shape=(n_users, cardinality),
            dtype=np.float32,
        ).tocsr()
        denominator.sum_duplicates()

        frequency_scale = np.log1p(
            len(values) / np.maximum(total_value, 1.0)
        )
        frequency_scale /= max(
            float(np.mean(frequency_scale[total_value > 0])), 1e-8
        )
        frequency_scale = np.clip(
            frequency_scale, 0.3, 3.0
        ).astype(np.float32)

        models[field] = (
            numerator,
            denominator,
            value_rate.astype(np.float32),
            frequency_scale,
            float(shrinkage),
        )

    return models


def signed_content_scores(split, models):
    users = np.asarray(split.user_id, dtype=np.int64)
    score = np.zeros(len(users), dtype=np.float32)

    field_weights = {
        "video_id": 0.55,
        "author_id": 0.70,
        "tag": 0.55,
        "onehot_feat3": 0.35,
        "upload_type": 0.22,
        "tab": 0.30,
        "duration_bucket": 0.30,
        "onehot_feat8": 0.22,
    }

    for field, model in models.items():
        numerator, denominator, value_rate, idf, shrinkage = model
        values = np.asarray(split.X[field], dtype=np.int64)

        valid_users = np.minimum(
            users, numerator.shape[0] - 1
        )
        valid_values = np.minimum(
            values, numerator.shape[1] - 1
        )

        num = paired_sparse_lookup(
            numerator, valid_users, valid_values
        )
        den = paired_sparse_lookup(
            denominator, valid_users, valid_values
        )
        residual = num / (den + shrinkage)

        # Include a conservative global component so the family remains a
        # meaningful standalone ranker while its personalized residual remains
        # the principal complementary signal.
        base = safe_logit(value_rate[valid_values]).astype(np.float32)
        base -= np.float32(np.mean(base))
        component = 0.28 * base + idf[valid_values] * residual
        score += field_weights[field] * component

    return score


def duration_bins_from_train(duration, n_bins=32):
    x = np.log1p(
        np.maximum(np.nan_to_num(duration, nan=0.0), 0.0)
    )
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(x, quantiles))
    return edges.astype(np.float32)


def assign_duration_bins(duration, edges):
    x = np.log1p(
        np.maximum(np.nan_to_num(duration, nan=0.0), 0.0)
    )
    return np.searchsorted(edges, x, side="right").astype(np.int64)


def fit_duration_kernel(train, weights, n_bins=32):
    users = np.asarray(train.user_id, dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.float32)
    duration = np.asarray(train.num["duration_ms"], dtype=np.float32)
    n_users = int(FEATURE_CARDINALITIES["user_id"])

    edges = duration_bins_from_train(duration, n_bins=n_bins)
    bins = assign_duration_bins(duration, edges)
    actual_bins = len(edges) + 1

    flat = users * actual_bins + bins
    total = np.bincount(
        flat,
        weights=weights,
        minlength=n_users * actual_bins,
    ).reshape(n_users, actual_bins).astype(np.float32)
    positive = np.bincount(
        flat,
        weights=weights * labels,
        minlength=n_users * actual_bins,
    ).reshape(n_users, actual_bins).astype(np.float32)

    global_total = np.sum(total, axis=0).astype(np.float64)
    global_positive = np.sum(positive, axis=0).astype(np.float64)
    global_mean = float(
        np.sum(global_positive) / np.maximum(np.sum(global_total), 1e-8)
    )
    global_rate = (
        global_positive + 30.0 * global_mean
    ) / np.maximum(global_total + 30.0, 1e-8)

    residual_numerator = (
        positive - total * global_rate[None, :]
    ).astype(np.float32)

    grid = np.arange(actual_bins, dtype=np.float32)
    distance = grid[:, None] - grid[None, :]
    kernel = np.exp(-0.5 * (distance / 2.0) ** 2).astype(np.float32)
    kernel /= np.maximum(kernel.sum(axis=1, keepdims=True), 1e-8)

    smooth_num = residual_numerator @ kernel.T
    smooth_den = total @ kernel.T

    return (
        edges,
        smooth_num.astype(np.float32),
        smooth_den.astype(np.float32),
        global_rate.astype(np.float32),
    )


def duration_kernel_scores(split, model):
    edges, smooth_num, smooth_den, global_rate = model
    users = np.asarray(split.user_id, dtype=np.int64)
    bins = assign_duration_bins(
        np.asarray(split.num["duration_ms"], dtype=np.float32),
        edges,
    )
    safe_users = np.minimum(users, smooth_num.shape[0] - 1)
    bins = np.minimum(bins, smooth_num.shape[1] - 1)

    residual = (
        smooth_num[safe_users, bins]
        / (smooth_den[safe_users, bins] + 8.0)
    )
    base = safe_logit(global_rate[bins]).astype(np.float32)
    return (base + 2.0 * residual).astype(np.float32)


def fit_signed_residual_spectral(train, weights, rank=24):
    users = np.asarray(train.user_id, dtype=np.int64)
    videos = np.asarray(train.video_id, dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.float32)

    n_users = int(FEATURE_CARDINALITIES["user_id"])
    n_videos = int(FEATURE_CARDINALITIES["video_id"])

    item_total = np.bincount(
        videos, weights=weights, minlength=n_videos
    ).astype(np.float64)
    item_positive = np.bincount(
        videos, weights=weights * labels, minlength=n_videos
    ).astype(np.float64)
    global_mean = float(
        np.sum(weights * labels) / np.maximum(np.sum(weights), 1e-8)
    )
    item_rate = (
        item_positive + 25.0 * global_mean
    ) / np.maximum(item_total + 25.0, 1e-8)

    residual = (
        weights * (labels - item_rate[videos]).astype(np.float32)
    ).astype(np.float32)

    matrix = sp.coo_matrix(
        (
            residual,
            (users.astype(np.int32), videos.astype(np.int32)),
        ),
        shape=(n_users, n_videos),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()

    row_energy = np.sqrt(
        np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    ).astype(np.float32)
    col_energy = np.sqrt(
        np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel()
    ).astype(np.float32)

    row_scale = 1.0 / np.sqrt(np.maximum(row_energy, 0.2))
    col_scale = 1.0 / np.sqrt(np.maximum(col_energy, 0.2))
    normalized = (
        sp.diags(row_scale) @ matrix @ sp.diags(col_scale)
    ).tocsr()

    k = min(rank, min(normalized.shape) - 2)
    u, singular, vt = svds(
        normalized,
        k=k,
        which="LM",
        return_singular_vectors=True,
        random_state=2026,
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order].astype(np.float32)
    u = u[:, order].astype(np.float32)
    vt = vt[order].astype(np.float32)

    user_factors = (
        u * singular[None, :] * row_scale[:, None]
    ).astype(np.float32)
    item_factors = (
        vt.T * col_scale[:, None]
    ).astype(np.float32)

    return (
        user_factors,
        item_factors,
        item_rate.astype(np.float32),
    )


def spectral_scores(split, model):
    user_factors, item_factors, item_rate = model
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)

    safe_users = np.minimum(users, user_factors.shape[0] - 1)
    safe_videos = np.minimum(videos, item_factors.shape[0] - 1)

    latent = np.sum(
        user_factors[safe_users] * item_factors[safe_videos],
        axis=1,
    )
    latent_std = max(float(np.std(latent)), 1e-6)
    latent = latent / latent_std

    base = safe_logit(item_rate[safe_videos])
    base = (base - np.mean(base)) / max(float(np.std(base)), 1e-6)
    return (base + 0.65 * latent).astype(np.float32)


def primary(users, labels, scores):
    return float(evaluate(users, labels, scores)["primary"])


train = load("train")
valid = load("valid")
test = load("test")

train_weights = recency_weights(train.date, half_life=5.0)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
valid_y = np.asarray(valid.y, dtype=np.int8)
test_users = np.asarray(test.user_id, dtype=np.int64)

profile_fields = [
    "video_id",
    "author_id",
    "tag",
    "onehot_feat3",
    "upload_type",
    "tab",
    "duration_bucket",
    "onehot_feat8",
]

profile_model = fit_signed_content_profiles(
    train, profile_fields, train_weights, shrinkage=10.0
)
profile_valid = signed_content_scores(valid, profile_model)
profile_test = signed_content_scores(test, profile_model)

duration_model = fit_duration_kernel(
    train, train_weights, n_bins=32
)
duration_valid = duration_kernel_scores(valid, duration_model)
duration_test = duration_kernel_scores(test, duration_model)

spectral_model = fit_signed_residual_spectral(
    train, train_weights, rank=24
)
spectral_valid = spectral_scores(valid, spectral_model)
spectral_test = spectral_scores(test, spectral_model)

del profile_model, duration_model, spectral_model
gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)
if not (
    os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise FileNotFoundError("Trusted incumbent predictions unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

valid_raw_models = {
    "signed_content_profile": np.asarray(profile_valid, dtype=np.float64),
    "duration_kernel": np.asarray(duration_valid, dtype=np.float64),
    "signed_residual_spectral": np.asarray(
        spectral_valid, dtype=np.float64
    ),
}
test_raw_models = {
    "signed_content_profile": np.asarray(profile_test, dtype=np.float64),
    "duration_kernel": np.asarray(duration_test, dtype=np.float64),
    "signed_residual_spectral": np.asarray(
        spectral_test, dtype=np.float64
    ),
}

valid_ranks = {
    name: within_user_rank(score, valid_users)
    for name, score in valid_raw_models.items()
}
test_ranks = {
    name: within_user_rank(score, test_users)
    for name, score in test_raw_models.items()
}

# A cross-family aggregate is itself a rank aggregation family, combining
# personalized content residuals, smooth duration preference, and latent
# residual structure without allowing any one score scale to dominate.
valid_ranks["cross_family_borda"] = np.mean(
    np.stack(
        [
            valid_ranks["signed_content_profile"],
            valid_ranks["duration_kernel"],
            valid_ranks["signed_residual_spectral"],
        ],
        axis=0,
    ),
    axis=0,
)
test_ranks["cross_family_borda"] = np.mean(
    np.stack(
        [
            test_ranks["signed_content_profile"],
            test_ranks["duration_kernel"],
            test_ranks["signed_residual_spectral"],
        ],
        axis=0,
    ),
    axis=0,
)

valid_raw_models["cross_family_borda"] = valid_ranks[
    "cross_family_borda"
]
test_raw_models["cross_family_borda"] = test_ranks[
    "cross_family_borda"
]

inc_valid_rank = within_user_rank(inc_valid, valid_users)
inc_test_rank = within_user_rank(inc_test, test_users)

candidate_log = {
    "trusted_incumbent": primary(
        valid_users, valid_y, inc_valid_rank
    )
}

for name in valid_ranks:
    candidate_log[name + "_standalone"] = primary(
        valid_users, valid_y, valid_ranks[name]
    )

alphas = np.asarray(
    [0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25],
    dtype=np.float64,
)

best_name = "trusted_incumbent"
best_alpha = 0.0
best_score = candidate_log["trusted_incumbent"]
best_valid = inc_valid_rank
best_test = inc_test_rank
best_own_valid = None

for name in valid_ranks:
    family_best = -np.inf
    family_best_alpha = 0.0

    for alpha in alphas:
        blended_valid = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * valid_ranks[name]
        )
        score = primary(valid_users, valid_y, blended_valid)

        if score > family_best:
            family_best = score
            family_best_alpha = float(alpha)

        if score > best_score:
            best_score = score
            best_name = name
            best_alpha = float(alpha)
            best_valid = blended_valid.copy()
            best_test = (
                (1.0 - alpha) * inc_test_rank
                + alpha * test_ranks[name]
            )
            best_own_valid = valid_raw_models[name]

    candidate_log[
        name + "_best_incumbent_blend"
    ] = float(family_best)
    candidate_log[
        name + "_best_alpha"
    ] = float(family_best_alpha)

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "selected_family": best_name,
            "selected_alpha": best_alpha,
            "spectral_rank": 24,
            "recency_half_life_days": 5.0,
        },
        sort_keys=True,
    )
)

metrics = evaluate(valid_users, valid_y, best_valid)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    if best_own_valid is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_own_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)