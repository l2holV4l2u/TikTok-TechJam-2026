import os
import time
import json
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
EPS = 1.0e-12


def temporal_weights(dates, half_life=4.0):
    dates = np.asarray(dates, dtype=np.int64)
    latest = int(dates.max())
    return np.power(
        2.0,
        (dates.astype(np.float64) - latest) / float(half_life),
    ).astype(np.float32)


def average_within_user_rank(user_ids, scores):
    """Ascending percentile ranks with exact ties receiving equal ranks."""
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    su = user_ids[order]
    ss = scores[order]

    user_start_mask = np.empty(n, dtype=bool)
    user_start_mask[0] = True
    user_start_mask[1:] = su[1:] != su[:-1]
    user_starts = np.flatnonzero(user_start_mask)
    user_group = np.cumsum(user_start_mask) - 1
    user_start_for_row = user_starts[user_group]
    user_sizes = np.diff(np.append(user_starts, n))
    size_for_row = user_sizes[user_group]

    tie_start_mask = np.empty(n, dtype=bool)
    tie_start_mask[0] = True
    tie_start_mask[1:] = (
        (su[1:] != su[:-1])
        | (ss[1:] != ss[:-1])
    )
    tie_starts = np.flatnonzero(tie_start_mask)
    tie_ends = np.append(tie_starts[1:], n) - 1
    tie_group = np.cumsum(tie_start_mask) - 1

    tie_mid_abs = 0.5 * (
        tie_starts.astype(np.float64) + tie_ends.astype(np.float64)
    )
    positions = (
        tie_mid_abs[tie_group]
        - user_start_for_row.astype(np.float64)
    )
    denominators = np.maximum(size_for_row - 1, 1).astype(np.float64)
    ranked_sorted = positions / denominators
    ranked_sorted[size_for_row == 1] = 0.5

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def keep_topk_rows(matrix, k):
    matrix = matrix.tocsr(copy=True)
    for row in range(matrix.shape[0]):
        lo = matrix.indptr[row]
        hi = matrix.indptr[row + 1]
        count = hi - lo
        if count > k:
            local = matrix.data[lo:hi]
            keep = np.argpartition(local, count - k)[count - k:]
            mask = np.ones(count, dtype=bool)
            mask[keep] = False
            local[mask] = 0.0
    matrix.eliminate_zeros()
    return matrix


def sparse_lookup(matrix, rows, cols):
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    return np.asarray(matrix[rows, cols]).reshape(-1).astype(np.float64)


def positive_profile_matrix(train, n_users, n_items):
    y = np.asarray(train.y, dtype=np.float32)
    uid = np.asarray(train.X["user_id"], dtype=np.int64)
    vid = np.asarray(train.X["video_id"], dtype=np.int64)
    decay = temporal_weights(train.date, half_life=4.0)

    mask = y > 0.5
    matrix = sp.coo_matrix(
        (
            decay[mask],
            (uid[mask], vid[mask]),
        ),
        shape=(n_users, n_items),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def build_cosine_covisitation(profile, topk=80):
    cooc = (profile.T @ profile).tocsr().astype(np.float32)
    diagonal = np.asarray(cooc.diagonal(), dtype=np.float64)
    cooc.setdiag(0.0)
    cooc.eliminate_zeros()
    cooc = keep_topk_rows(cooc, topk)

    inverse_norm = np.zeros_like(diagonal)
    positive = diagonal > 0
    inverse_norm[positive] = 1.0 / np.sqrt(diagonal[positive])

    row_ids = np.repeat(
        np.arange(cooc.shape[0], dtype=np.int64),
        np.diff(cooc.indptr),
    )
    cooc.data *= (
        inverse_norm[row_ids]
        * inverse_norm[cooc.indices]
    ).astype(np.float32)
    cooc.eliminate_zeros()
    return cooc


def build_markov(train, n_users, n_items, history_length=5, topk=80):
    y = np.asarray(train.y, dtype=np.int8)
    uid = np.asarray(train.X["user_id"], dtype=np.int64)
    vid = np.asarray(train.X["video_id"], dtype=np.int64)
    tm = np.asarray(train.time_ms, dtype=np.int64)
    dates = np.asarray(train.date, dtype=np.int64)
    rows = np.arange(uid.size, dtype=np.int64)

    positive_rows = np.flatnonzero(y > 0)
    order = positive_rows[
        np.lexsort(
            (
                rows[positive_rows],
                tm[positive_rows],
                uid[positive_rows],
            )
        )
    ]

    ou = uid[order]
    ov = vid[order]
    ot = tm[order]
    od = dates[order]

    adjacent = (
        (ou[1:] == ou[:-1])
        & ((ot[1:] - ot[:-1]) <= 3 * 86400000)
    )
    source = ov[:-1][adjacent]
    target = ov[1:][adjacent]
    target_dates = od[1:][adjacent]
    edge_weight = temporal_weights(target_dates, half_life=4.0)

    transition = sp.coo_matrix(
        (edge_weight, (source, target)),
        shape=(n_items, n_items),
        dtype=np.float32,
    ).tocsr()
    transition.sum_duplicates()
    transition.setdiag(0.0)
    transition.eliminate_zeros()
    transition = keep_topk_rows(transition, topk)

    row_sums = np.asarray(transition.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(row_sums, dtype=np.float32)
    nonzero = row_sums > 0
    inverse[nonzero] = 1.0 / row_sums[nonzero]
    transition = sp.diags(inverse) @ transition
    transition = transition.tocsr()

    # Train-only recent positive history used as the Markov state.
    starts = np.empty(order.size, dtype=bool)
    starts[0] = True
    starts[1:] = ou[1:] != ou[:-1]
    group_starts = np.flatnonzero(starts)
    group_ids = np.cumsum(starts) - 1
    group_ends = np.append(group_starts[1:], order.size)
    reverse_position = (
        group_ends[group_ids] - 1 - np.arange(order.size, dtype=np.int64)
    )
    recent = reverse_position < history_length

    history_weight = np.power(
        0.70,
        reverse_position[recent].astype(np.float32),
    )
    history = sp.coo_matrix(
        (
            history_weight,
            (ou[recent], ov[recent]),
        ),
        shape=(n_users, n_items),
        dtype=np.float32,
    ).tocsr()
    history.sum_duplicates()

    prediction = (history @ transition).tocsr()
    return prediction


def fit_svd(profile, rank=40):
    # Recency-weighted implicit reconstruction; no validation data is used.
    u, singular, vt = svds(
        profile.astype(np.float32),
        k=rank,
        which="LM",
        random_state=2026,
        return_singular_vectors=True,
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order].astype(np.float32)
    u = u[:, order].astype(np.float32)
    vt = vt[order].astype(np.float32)
    user_factor = u * singular[None, :]
    return user_factor, vt


def split_component_scores(split, covis_prediction, markov_prediction,
                           svd_user, svd_items):
    uid = np.asarray(split.X["user_id"], dtype=np.int64)
    vid = np.asarray(split.X["video_id"], dtype=np.int64)

    covis = sparse_lookup(covis_prediction, uid, vid)
    markov = sparse_lookup(markov_prediction, uid, vid)
    svd = np.sum(
        svd_user[uid] * svd_items[:, vid].T,
        axis=1,
        dtype=np.float64,
    )
    return {
        "covisitation_cosine": covis,
        "directed_markov": markov,
        "implicit_svd": svd,
    }


train = load("train")
valid = load("valid")

n_users = int(FEATURE_CARDINALITIES["user_id"])
n_items = int(FEATURE_CARDINALITIES["video_id"])

profile = positive_profile_matrix(train, n_users, n_items)

covis_similarity = build_cosine_covisitation(profile, topk=80)
covis_prediction = (profile @ covis_similarity).tocsr()

markov_prediction = build_markov(
    train,
    n_users,
    n_items,
    history_length=5,
    topk=80,
)

svd_user, svd_items = fit_svd(profile, rank=40)

valid_raw_components = split_component_scores(
    valid,
    covis_prediction,
    markov_prediction,
    svd_user,
    svd_items,
)
valid_rank_components = {
    name: average_within_user_rank(valid.user_id, score)
    for name, score in valid_raw_components.items()
}

# Cross-family aggregations test whether exact, sequential, and latent
# neighborhoods make complementary ranking errors.
valid_own_candidates = dict(valid_rank_components)
valid_own_candidates["covis_markov_mean"] = (
    0.5 * valid_rank_components["covisitation_cosine"]
    + 0.5 * valid_rank_components["directed_markov"]
)
valid_own_candidates["covis_svd_mean"] = (
    0.5 * valid_rank_components["covisitation_cosine"]
    + 0.5 * valid_rank_components["implicit_svd"]
)
valid_own_candidates["three_family_mean"] = (
    valid_rank_components["covisitation_cosine"]
    + valid_rank_components["directed_markov"]
    + valid_rank_components["implicit_svd"]
) / 3.0

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path):
    raise FileNotFoundError("Trusted incumbent validation predictions missing")
if not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent test predictions missing")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if inc_valid.size != valid.user_id.size:
    raise RuntimeError("Trusted incumbent validation length mismatch")
inc_valid_rank = average_within_user_rank(valid.user_id, inc_valid)

candidate_metrics = {}
candidate_scores = {}
candidate_specs = {}

inc_metric = evaluate(valid.user_id, valid.y, inc_valid_rank)
candidate_metrics["trusted_incumbent"] = float(inc_metric["primary"])
candidate_scores["trusted_incumbent"] = inc_valid_rank
candidate_specs["trusted_incumbent"] = ("covisitation_cosine", 0.0)

blend_weights = (0.03, 0.06, 0.10, 0.15, 0.25, 0.40)

for family, own_rank in valid_own_candidates.items():
    standalone_result = evaluate(valid.user_id, valid.y, own_rank)
    standalone_name = family + "_standalone"
    candidate_metrics[standalone_name] = float(
        standalone_result["primary"]
    )
    candidate_scores[standalone_name] = own_rank
    candidate_specs[standalone_name] = (family, 1.0)

    for alpha in blend_weights:
        blended = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * own_rank
        )
        name = f"{family}_incblend_{alpha:.2f}"
        result = evaluate(valid.user_id, valid.y, blended)
        candidate_metrics[name] = float(result["primary"])
        candidate_scores[name] = blended
        candidate_specs[name] = (family, alpha)

winner_name = max(candidate_metrics, key=candidate_metrics.get)
winner_family, winner_alpha = candidate_specs[winner_name]
valid_scores = candidate_scores[winner_name]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print(
    "FINDINGS collaborative_graph_sizes "
    + json.dumps(
        {
            "positive_profile_nnz": int(profile.nnz),
            "covisitation_nnz": int(covis_similarity.nnz),
            "covis_prediction_nnz": int(covis_prediction.nnz),
            "markov_prediction_nnz": int(markov_prediction.nnz),
        },
        sort_keys=True,
    )
)
print("CANDIDATES " + json.dumps(candidate_metrics, sort_keys=True))
print(
    "FINDINGS winner "
    + json.dumps(
        {
            "candidate": winner_name,
            "own_family": winner_family,
            "own_rank_weight": float(winner_alpha),
        },
        sort_keys=True,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(
            valid_own_candidates[winner_family],
            dtype=np.float64,
        ),
    )

test = load("test")
test_raw_components = split_component_scores(
    test,
    covis_prediction,
    markov_prediction,
    svd_user,
    svd_items,
)
test_rank_components = {
    name: average_within_user_rank(test.user_id, score)
    for name, score in test_raw_components.items()
}
test_own_candidates = dict(test_rank_components)
test_own_candidates["covis_markov_mean"] = (
    0.5 * test_rank_components["covisitation_cosine"]
    + 0.5 * test_rank_components["directed_markov"]
)
test_own_candidates["covis_svd_mean"] = (
    0.5 * test_rank_components["covisitation_cosine"]
    + 0.5 * test_rank_components["implicit_svd"]
)
test_own_candidates["three_family_mean"] = (
    test_rank_components["covisitation_cosine"]
    + test_rank_components["directed_markov"]
    + test_rank_components["implicit_svd"]
) / 3.0

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if inc_test.size != test.user_id.size:
    raise RuntimeError("Trusted incumbent test length mismatch")
inc_test_rank = average_within_user_rank(test.user_id, inc_test)

test_own_rank = test_own_candidates[winner_family]
test_scores = (
    (1.0 - winner_alpha) * inc_test_rank
    + winner_alpha * test_own_rank
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
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