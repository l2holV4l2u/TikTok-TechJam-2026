import os
import time
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 48371
RANK = 24
HALF_LIFE_DAYS = 7.0

np.random.seed(SEED)

N_USERS = int(FEATURE_CARDINALITIES["user_id"])
N_VIDEOS = int(FEATURE_CARDINALITIES["video_id"])


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int64)
    unique_dates = np.unique(dates)
    day_index = np.searchsorted(unique_dates, dates).astype(np.float64)
    age = float(day_index.max()) - day_index
    weights = np.exp2(-age / HALF_LIFE_DAYS)
    weights /= max(float(weights.mean()), 1e-12)
    return weights.astype(np.float64)


def stable_svds(matrix, rank=RANK):
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    maximum_rank = min(matrix.shape) - 1
    k = min(int(rank), maximum_rank)
    if k < 1 or matrix.nnz == 0:
        return (
            np.zeros((matrix.shape[0], 1), dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros((1, matrix.shape[1]), dtype=np.float64),
        )

    v0_size = min(matrix.shape)
    v0 = np.linspace(0.5, 1.5, v0_size, dtype=np.float64)
    v0 /= np.linalg.norm(v0)

    try:
        u, singular_values, vt = svds(
            matrix,
            k=k,
            which="LM",
            solver="arpack",
            v0=v0,
            tol=2e-3,
            maxiter=700,
            return_singular_vectors=True,
        )
    except Exception:
        u, singular_values, vt = svds(
            matrix,
            k=k,
            which="LM",
            solver="lobpcg",
            v0=v0,
            tol=5e-3,
            maxiter=300,
            return_singular_vectors=True,
        )

    order = np.argsort(singular_values)[::-1]
    return u[:, order], singular_values[order], vt[order]


def make_interaction_matrix(user_ids, video_ids, values):
    matrix = sparse.coo_matrix(
        (
            np.asarray(values, dtype=np.float64),
            (
                np.asarray(user_ids, dtype=np.int64),
                np.asarray(video_ids, dtype=np.int64),
            ),
        ),
        shape=(N_USERS, N_VIDEOS),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def fit_spectral_model(user_ids, video_ids, labels, dates, method):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    video_ids = np.asarray(video_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    weights = recency_weights(dates)

    if method == "puresvd_positive":
        matrix = make_interaction_matrix(
            user_ids, video_ids, weights * labels
        )
        u, singular_values, vt = stable_svds(matrix)
        return {
            "method": method,
            "user_factors": u * singular_values[None, :],
            "item_factors": vt.T,
        }

    if method == "signed_residual_svd":
        weighted_mean = float(
            np.sum(weights * labels) / max(np.sum(weights), 1e-12)
        )
        residual = weights * (labels - weighted_mean)
        matrix = make_interaction_matrix(user_ids, video_ids, residual)

        row_energy = np.sqrt(
            np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
        )
        col_energy = np.sqrt(
            np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel()
        )
        row_scale = 1.0 / np.sqrt(np.maximum(row_energy, 1.0))
        col_scale = 1.0 / np.sqrt(np.maximum(col_energy, 1.0))
        normalized = sparse.diags(row_scale) @ matrix @ sparse.diags(col_scale)

        u, singular_values, vt = stable_svds(normalized)
        return {
            "method": method,
            "user_factors": u * singular_values[None, :],
            "item_factors": vt.T,
        }

    if method == "normalized_positive_graph":
        matrix = make_interaction_matrix(
            user_ids, video_ids, weights * labels
        )
        user_degree = np.asarray(matrix.sum(axis=1)).ravel()
        item_degree = np.asarray(matrix.sum(axis=0)).ravel()

        user_scale = 1.0 / np.sqrt(np.maximum(user_degree, 1.0))
        item_scale = 1.0 / np.sqrt(np.maximum(item_degree, 1.0))
        normalized = sparse.diags(user_scale) @ matrix @ sparse.diags(item_scale)

        u, singular_values, vt = stable_svds(normalized)
        return {
            "method": method,
            "user_factors": u * singular_values[None, :],
            "item_factors": vt.T,
        }

    raise ValueError(method)


def fit_transition_model(user_ids, video_ids, labels, dates, time_ms):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    video_ids = np.asarray(video_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    time_ms = np.asarray(time_ms, dtype=np.int64)
    weights = recency_weights(dates)

    row_position = np.arange(len(user_ids), dtype=np.int64)
    order = np.lexsort((row_position, time_ms, user_ids))

    ordered_users = user_ids[order]
    left = order[:-1]
    right = order[1:]
    same_user = ordered_users[:-1] == ordered_users[1:]

    left = left[same_user]
    right = right[same_user]

    transition_weight = (
        np.sqrt(weights[left] * weights[right])
        * labels[left]
        * labels[right]
    )
    keep = transition_weight > 0.0
    left = left[keep]
    right = right[keep]
    transition_weight = transition_weight[keep]

    src = video_ids[left]
    dst = video_ids[right]

    transition = sparse.coo_matrix(
        (
            np.concatenate([transition_weight, transition_weight]),
            (
                np.concatenate([src, dst]),
                np.concatenate([dst, src]),
            ),
        ),
        shape=(N_VIDEOS, N_VIDEOS),
        dtype=np.float64,
    ).tocsr()
    transition.sum_duplicates()
    transition.eliminate_zeros()

    degree = np.asarray(transition.sum(axis=1)).ravel()
    scale = 1.0 / np.sqrt(np.maximum(degree, 1.0))
    normalized = sparse.diags(scale) @ transition @ sparse.diags(scale)

    _, singular_values, vt = stable_svds(normalized)
    item_factors = vt.T * np.sqrt(
        np.maximum(singular_values, 1e-12)
    )[None, :]

    positive_profile = make_interaction_matrix(
        user_ids, video_ids, weights * labels
    )
    profile_mass = np.asarray(positive_profile.sum(axis=1)).ravel()
    user_factors = positive_profile @ item_factors
    user_factors = np.asarray(user_factors, dtype=np.float64)
    user_factors /= np.maximum(profile_mass[:, None], 1.0)

    user_norm = np.linalg.norm(user_factors, axis=1, keepdims=True)
    item_norm = np.linalg.norm(item_factors, axis=1, keepdims=True)
    user_factors /= np.maximum(user_norm, 1e-8)
    item_factors /= np.maximum(item_norm, 1e-8)

    return {
        "method": "transition_item_embedding",
        "user_factors": user_factors,
        "item_factors": item_factors,
        "n_positive_transitions": int(len(transition_weight)),
    }


def predict_latent(model, user_ids, video_ids):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    video_ids = np.asarray(video_ids, dtype=np.int64)
    user_factors = model["user_factors"]
    item_factors = model["item_factors"]

    scores = np.einsum(
        "ij,ij->i",
        user_factors[user_ids],
        item_factors[video_ids],
        optimize=True,
    )
    return np.nan_to_num(
        np.asarray(scores, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    ordered_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_users[1:] != ordered_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    positions = (
        np.arange(n, dtype=np.int64) - np.repeat(starts, lengths)
    )
    denominators = np.repeat(np.maximum(lengths - 1, 1), lengths)
    ranked = positions.astype(np.float64) / denominators

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    standard_deviation = float(values.std())
    if standard_deviation < 1e-12:
        return np.zeros_like(values)
    return (values - float(values.mean())) / standard_deviation


def combine_scores(raw, incumbent, users, mode, incumbent_weight):
    if mode == "raw":
        return np.asarray(raw, dtype=np.float64)
    if mode == "rank":
        return (
            incumbent_weight * within_user_rank(users, incumbent)
            + (1.0 - incumbent_weight) * within_user_rank(users, raw)
        )
    if mode == "zscore":
        return (
            incumbent_weight * zscore(incumbent)
            + (1.0 - incumbent_weight) * zscore(raw)
        )
    raise ValueError(mode)


train = load("train")
valid = load("valid")

train_users = np.asarray(train.user_id, dtype=np.int64)
train_videos = np.asarray(train.video_id, dtype=np.int64)
train_labels = np.asarray(train.y, dtype=np.float64)

valid_users = np.asarray(valid.user_id, dtype=np.int64)
valid_videos = np.asarray(valid.video_id, dtype=np.int64)
valid_labels = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

METHODS = [
    "puresvd_positive",
    "signed_residual_svd",
    "normalized_positive_graph",
    "transition_item_embedding",
]

raw_valid = {}
model_findings = {}

for method in METHODS:
    if method == "transition_item_embedding":
        model = fit_transition_model(
            train_users,
            train_videos,
            train_labels,
            train.date,
            train.time_ms,
        )
        model_findings[method] = {
            "n_positive_transitions": model["n_positive_transitions"]
        }
    else:
        model = fit_spectral_model(
            train_users,
            train_videos,
            train_labels,
            train.date,
            method,
        )

    raw_valid[method] = predict_latent(
        model, valid_users, valid_videos
    )
    del model

candidate_predictions = {
    "trusted_incumbent": inc_valid.copy()
}
candidate_specs = {
    "trusted_incumbent": ("incumbent", "raw", 1.0)
}

for method in METHODS:
    candidate_predictions[method + "_raw"] = raw_valid[method]
    candidate_specs[method + "_raw"] = (method, "raw", 0.0)

    for alpha in (0.10, 0.25, 0.50, 0.75):
        rank_name = method + "_rank_inc%.2f" % alpha
        candidate_predictions[rank_name] = combine_scores(
            raw_valid[method],
            inc_valid,
            valid_users,
            "rank",
            alpha,
        )
        candidate_specs[rank_name] = (method, "rank", alpha)

        z_name = method + "_z_inc%.2f" % alpha
        candidate_predictions[z_name] = combine_scores(
            raw_valid[method],
            inc_valid,
            valid_users,
            "zscore",
            alpha,
        )
        candidate_specs[z_name] = (method, "zscore", alpha)

candidate_metrics = {}
best_name = None
best_result = None

for name, scores in candidate_predictions.items():
    result = evaluate(valid_users, valid_labels, scores)
    candidate_metrics[name] = float(result["primary"])
    if (
        best_result is None
        or float(result["primary"]) > float(best_result["primary"])
    ):
        best_name = name
        best_result = result

valid_scores = np.asarray(
    candidate_predictions[best_name], dtype=np.float64
)
winning_method, winning_mode, winning_alpha = candidate_specs[best_name]

print("CANDIDATES " + json.dumps(candidate_metrics, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": best_name,
            "raw_primary": {
                method: candidate_metrics[method + "_raw"]
                for method in METHODS
            },
            "model_findings": model_findings,
            "validation_aux_read": False,
            "test_labels_read": False,
        },
        sort_keys=True,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        valid_scores.astype(np.float64),
    )

test = load("test")
test_users = np.asarray(test.user_id, dtype=np.int64)
test_videos = np.asarray(test.video_id, dtype=np.int64)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if winning_method == "incumbent":
    test_scores = inc_test.copy()
else:
    combined_users = np.concatenate([
        train_users,
        valid_users,
    ])
    combined_videos = np.concatenate([
        train_videos,
        valid_videos,
    ])
    combined_labels = np.concatenate([
        train_labels,
        valid_labels.astype(np.float64),
    ])
    combined_dates = np.concatenate([
        np.asarray(train.date),
        np.asarray(valid.date),
    ])
    combined_times = np.concatenate([
        np.asarray(train.time_ms),
        np.asarray(valid.time_ms),
    ])

    if winning_method == "transition_item_embedding":
        final_model = fit_transition_model(
            combined_users,
            combined_videos,
            combined_labels,
            combined_dates,
            combined_times,
        )
    else:
        final_model = fit_spectral_model(
            combined_users,
            combined_videos,
            combined_labels,
            combined_dates,
            winning_method,
        )

    raw_test = predict_latent(
        final_model, test_users, test_videos
    )
    test_scores = combine_scores(
        raw_test,
        inc_test,
        test_users,
        winning_mode,
        winning_alpha,
    )

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_result["primary"]),
            "gauc": float(best_result["gauc"]),
            "ndcg@5": float(best_result["ndcg@5"]),
            "gpu_seconds": elapsed,
        }
    )
)