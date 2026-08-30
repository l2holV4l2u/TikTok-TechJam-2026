import os
import time
import json
import gc
import numpy as np
import scipy.sparse as sp

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()

USER_CARD = int(FEATURE_CARDINALITIES["user_id"])
VIDEO_CARD = int(FEATURE_CARDINALITIES["video_id"])


def build_positive_matrix(user_ids, video_ids, labels):
    labels = np.asarray(labels, dtype=np.int8)
    keep = labels > 0
    rows = np.asarray(user_ids, dtype=np.int64)[keep]
    cols = np.asarray(video_ids, dtype=np.int64)[keep]
    data = np.ones(rows.size, dtype=np.float32)

    mat = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(USER_CARD, VIDEO_CARD),
        dtype=np.float32,
    ).tocsr()
    mat.sum_duplicates()
    mat.data[:] = 1.0
    mat.eliminate_zeros()
    return mat


def zero_diagonal_and_clean(mat):
    mat = mat.tocsr().astype(np.float32)
    mat.setdiag(0.0)
    mat.eliminate_zeros()
    mat.sort_indices()
    return mat


def make_association_matrices(profile):
    user_degree = np.asarray(profile.sum(axis=1)).ravel().astype(np.float32)
    item_degree = np.asarray(profile.sum(axis=0)).ravel().astype(np.float32)

    cooc = (profile.T @ profile).tocsr().astype(np.float32)
    cooc = zero_diagonal_and_clean(cooc)

    inv_sqrt_item = np.zeros(VIDEO_CARD, dtype=np.float32)
    nz_item = item_degree > 0
    inv_sqrt_item[nz_item] = 1.0 / np.sqrt(item_degree[nz_item])

    cosine = cooc.multiply(inv_sqrt_item[:, None])
    cosine = cosine.multiply(inv_sqrt_item[None, :])
    cosine = zero_diagonal_and_clean(cosine)

    inv_user_degree = np.zeros(USER_CARD, dtype=np.float32)
    nz_user = user_degree > 0
    inv_user_degree[nz_user] = 1.0 / np.maximum(user_degree[nz_user], 1.0)
    weighted_profile = profile.multiply(inv_user_degree[:, None])
    resource = (profile.T @ weighted_profile).tocsr().astype(np.float32)
    resource = zero_diagonal_and_clean(resource)

    # Positive PMI on binary user-item incidence. This emphasizes unexpectedly
    # strong item associations rather than raw or cosine-normalized overlap.
    active_users = float(max(np.count_nonzero(user_degree), 1))
    pmi = cooc.tocoo(copy=True)
    denom = (
        item_degree[pmi.row].astype(np.float64)
        * item_degree[pmi.col].astype(np.float64)
    )
    ratio = (
        pmi.data.astype(np.float64) * active_users
        / np.maximum(denom, 1e-12)
    )
    pmi_data = np.maximum(np.log(np.maximum(ratio, 1e-12)), 0.0)
    keep = pmi_data > 0.0
    pmi = sp.coo_matrix(
        (
            pmi_data[keep].astype(np.float32),
            (pmi.row[keep], pmi.col[keep]),
        ),
        shape=(VIDEO_CARD, VIDEO_CARD),
        dtype=np.float32,
    ).tocsr()
    pmi = zero_diagonal_and_clean(pmi)

    popularity = np.log1p(item_degree).astype(np.float32)

    return {
        "knn_cosine": cosine,
        "resource_allocation": resource,
        "positive_pmi": pmi,
    }, popularity, {
        "profile_nnz": int(profile.nnz),
        "cooc_nnz": int(cooc.nnz),
        "cosine_nnz": int(cosine.nnz),
        "resource_nnz": int(resource.nnz),
        "pmi_nnz": int(pmi.nnz),
    }


def score_from_profile(profile, association, eval_users, eval_videos, popularity):
    users = np.asarray(eval_users, dtype=np.int64)
    videos = np.asarray(eval_videos, dtype=np.int64)
    n = users.size

    counts = np.diff(profile.indptr)[users].astype(np.int64)
    total = int(counts.sum())

    if total == 0:
        return (1e-6 * popularity[videos]).astype(np.float64)

    repeated_rows = np.repeat(np.arange(n, dtype=np.int64), counts)
    repeated_starts = np.repeat(profile.indptr[users], counts)

    cumulative_before = np.repeat(
        np.cumsum(counts, dtype=np.int64) - counts,
        counts,
    )
    local_position = np.arange(total, dtype=np.int64) - cumulative_before
    history_positions = repeated_starts + local_position
    history_items = profile.indices[history_positions]
    candidate_items = videos[repeated_rows]

    values = np.asarray(
        association[history_items, candidate_items]
    ).ravel().astype(np.float64)

    scores = np.bincount(
        repeated_rows,
        weights=values,
        minlength=n,
    ).astype(np.float64)

    norm = np.sqrt(np.maximum(counts.astype(np.float64), 1.0))
    scores /= norm

    # Meaningful deterministic fallback for unseen associations.
    scores += 1e-6 * popularity[videos].astype(np.float64)
    return scores


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, users))
    sorted_users = users[order]
    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(first)
    counts = np.diff(np.r_[starts, n])

    positions = (
        np.arange(n, dtype=np.float64)
        - np.repeat(starts, counts).astype(np.float64)
    )
    denominators = np.maximum(np.repeat(counts, counts) - 1, 1)
    ranks = positions / denominators
    ranks[np.repeat(counts, counts) == 1] = 0.5

    result = np.empty(n, dtype=np.float64)
    result[order] = ranks
    return result


def fit_memory_models(user_ids, video_ids, labels):
    profile = build_positive_matrix(user_ids, video_ids, labels)
    associations, popularity, stats = make_association_matrices(profile)
    return profile, associations, popularity, stats


train = load("train")
valid = load("valid")

train_users_model = np.asarray(train.X["user_id"], dtype=np.int64)
train_videos = np.asarray(train.X["video_id"], dtype=np.int64)
train_y = np.asarray(train.y, dtype=np.int8)

valid_users_model = np.asarray(valid.X["user_id"], dtype=np.int64)
valid_videos = np.asarray(valid.X["video_id"], dtype=np.int64)
valid_users_eval = np.asarray(valid.user_id, dtype=np.int64)
valid_y = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(shared, "incumbent_test_scores.npy")),
    dtype=np.float64,
)

profile, associations, popularity, stats = fit_memory_models(
    train_users_model,
    train_videos,
    train_y,
)
print("FINDINGS " + json.dumps(stats, sort_keys=True))

inc_metrics = evaluate(valid_users_eval, valid_y, inc_valid)
inc_rank = within_user_rank(valid_users_eval, inc_valid)

candidate_values = {
    "incumbent": float(inc_metrics["primary"]),
}
raw_predictions = {}
model_ranks = {}

best_name = "incumbent"
best_family = "knn_cosine"
best_alpha = 0.0
best_scores = inc_valid.copy()
best_metrics = inc_metrics
best_primary = float(inc_metrics["primary"])

blend_weights = (0.05, 0.10, 0.18, 0.26, 0.34, 0.44)

for family, association in associations.items():
    pred = score_from_profile(
        profile,
        association,
        valid_users_model,
        valid_videos,
        popularity,
    )
    raw_predictions[family] = pred
    pred_rank = within_user_rank(valid_users_eval, pred)
    model_ranks[family] = pred_rank

    metrics_raw = evaluate(valid_users_eval, valid_y, pred)
    raw_primary = float(metrics_raw["primary"])
    candidate_values[family] = raw_primary

    if raw_primary > best_primary:
        best_primary = raw_primary
        best_name = family
        best_family = family
        best_alpha = 1.0
        best_scores = pred.copy()
        best_metrics = metrics_raw

    for alpha in blend_weights:
        blend = (1.0 - alpha) * inc_rank + alpha * pred_rank
        metrics_blend = evaluate(valid_users_eval, valid_y, blend)
        primary = float(metrics_blend["primary"])
        name = f"{family}_rankblend_{alpha:.2f}"
        candidate_values[name] = primary

        if primary > best_primary:
            best_primary = primary
            best_name = name
            best_family = family
            best_alpha = float(alpha)
            best_scores = blend.copy()
            best_metrics = metrics_blend

print("CANDIDATES " + json.dumps(candidate_values, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "selected": best_name,
            "selected_family": best_family,
            "selected_alpha": best_alpha,
            "incumbent_primary": float(inc_metrics["primary"]),
        },
        sort_keys=True,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(raw_predictions[best_family], dtype=np.float64),
    )

# Refit the identical selected memory-based recipe on train + validation.
# Validation labels are allowed here because this refit is used only for test scoring.
del associations
gc.collect()

all_users = np.concatenate([train_users_model, valid_users_model])
all_videos = np.concatenate([train_videos, valid_videos])
all_y = np.concatenate([train_y, valid_y])

profile_full, associations_full, popularity_full, refit_stats = fit_memory_models(
    all_users,
    all_videos,
    all_y,
)
print("FINDINGS " + json.dumps(
    {"refit_" + k: v for k, v in refit_stats.items()},
    sort_keys=True,
))

test = load("test")
test_users_model = np.asarray(test.X["user_id"], dtype=np.int64)
test_videos = np.asarray(test.X["video_id"], dtype=np.int64)
test_users_eval = np.asarray(test.user_id, dtype=np.int64)

test_raw = score_from_profile(
    profile_full,
    associations_full[best_family],
    test_users_model,
    test_videos,
    popularity_full,
)

if best_alpha <= 0.0:
    test_scores = inc_test.copy()
elif best_alpha >= 1.0:
    test_scores = test_raw
else:
    test_model_rank = within_user_rank(test_users_eval, test_raw)
    test_inc_rank = within_user_rank(test_users_eval, inc_test)
    test_scores = (
        (1.0 - best_alpha) * test_inc_rank
        + best_alpha * test_model_rank
    )

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.4f}'
    % (
        float(best_metrics["primary"]),
        float(best_metrics["gauc"]),
        float(best_metrics["ndcg@5"]),
        elapsed,
    )
)