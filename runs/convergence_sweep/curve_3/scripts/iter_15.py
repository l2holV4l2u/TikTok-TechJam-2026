import os
import time
import json
import gc

import numpy as np
import scipy.sparse as sp

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
K_HISTORY = 12

MEMORY_FIELDS = [
    ("video_id", 1.00),
    ("author_id", 0.75),
    ("tag", 0.42),
    ("onehot_feat3", 0.34),
    ("upload_type", 0.24),
    ("duration_bucket", 0.22),
    ("onehot_feat8", 0.18),
]


def build_positive_history(train, k):
    """Last k positive impression row indices for every categorical user id."""
    users = np.asarray(train.user_id, dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.int8)
    times = np.asarray(train.time_ms, dtype=np.int64)
    rows = np.arange(len(users), dtype=np.int64)

    positive_rows = rows[(labels == 1) & (users > 0)]
    pu = users[positive_rows]
    pt = times[positive_rows]

    order = np.lexsort((positive_rows, pt, pu))
    sorted_rows = positive_rows[order]
    sorted_users = users[sorted_rows]
    n = len(sorted_rows)

    positions = np.arange(n, dtype=np.int64)
    end_flags = np.empty(n, dtype=bool)
    end_flags[-1] = True
    end_flags[:-1] = sorted_users[:-1] != sorted_users[1:]
    group_ends = np.minimum.accumulate(
        np.where(end_flags, positions, n - 1)[::-1]
    )[::-1]
    reverse_rank = group_ends - positions

    keep = reverse_rank < k
    n_users = int(FEATURE_CARDINALITIES["user_id"])
    history = np.full((n_users, k), -1, dtype=np.int64)
    history[
        sorted_users[keep],
        reverse_rank[keep],
    ] = sorted_rows[keep]
    history[0, :] = -1
    return history, sorted_rows, sorted_users


def lookup_history_rows(history_rows, query_users):
    query_users = np.asarray(query_users, dtype=np.int64)
    valid_users = (
        (query_users > 0) & (query_users < history_rows.shape[0])
    )
    result = np.full(
        (len(query_users), history_rows.shape[1]),
        -1,
        dtype=np.int64,
    )
    result[valid_users] = history_rows[query_users[valid_users]]
    return result


def field_history(train, history_rows, field):
    values = np.asarray(train.X[field], dtype=np.int64)
    result = np.zeros(history_rows.shape, dtype=np.int64)
    mask = history_rows >= 0
    result[mask] = values[history_rows[mask]]
    return result


def paired_sparse_lookup(matrix, rows, cols, chunk=1000000):
    rows = np.asarray(rows, dtype=np.int64).ravel()
    cols = np.asarray(cols, dtype=np.int64).ravel()
    result = np.zeros(len(rows), dtype=np.float32)

    for start in range(0, len(rows), chunk):
        end = min(start + chunk, len(rows))
        block = matrix[rows[start:end], cols[start:end]]
        result[start:end] = np.asarray(block).ravel().astype(
            np.float32, copy=False
        )
    return result


def make_recent_interaction_matrix(train, history_rows):
    hist_videos = field_history(train, history_rows, "video_id")
    valid = (history_rows >= 0) & (hist_videos > 0)

    row_grid = np.broadcast_to(
        np.arange(history_rows.shape[0], dtype=np.int32)[:, None],
        history_rows.shape,
    )
    rows = row_grid[valid]
    cols = hist_videos[valid].astype(np.int32, copy=False)

    # Recent positives receive modestly decreasing mass. Duplicated user-item
    # events are later collapsed, preventing repeat exposure from dominating.
    decay = np.power(
        0.88,
        np.broadcast_to(
            np.arange(history_rows.shape[1], dtype=np.float32)[None, :],
            history_rows.shape,
        ),
    )
    data = decay[valid].astype(np.float32, copy=False)

    matrix = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(
            int(FEATURE_CARDINALITIES["user_id"]),
            int(FEATURE_CARDINALITIES["video_id"]),
        ),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    return matrix, hist_videos


def fit_coview_graph(recent_matrix):
    # Positive co-view graph. Cosine normalization suppresses globally popular
    # videos and emphasizes videos sharing a specific audience.
    binary = recent_matrix.copy()
    binary.data[:] = 1.0
    binary.eliminate_zeros()

    item_support = np.asarray(binary.sum(axis=0)).ravel().astype(np.float32)
    graph = (binary.T @ binary).tocsr().astype(np.float32)
    graph.setdiag(0.0)
    graph.eliminate_zeros()

    row_counts = np.diff(graph.indptr)
    row_ids = np.repeat(
        np.arange(graph.shape[0], dtype=np.int64), row_counts
    )
    col_ids = graph.indices
    denominator = np.sqrt(
        np.maximum(item_support[row_ids], 1.0)
        * np.maximum(item_support[col_ids], 1.0)
    )
    graph.data /= denominator.astype(np.float32)
    graph.data = np.minimum(graph.data, 1.0)
    return graph


def coview_scores(split, history_rows, hist_videos, graph):
    query_users = np.asarray(split.user_id, dtype=np.int64)
    candidate = np.asarray(split.video_id, dtype=np.int64)
    qrows = lookup_history_rows(history_rows, query_users)

    qhist = np.zeros(qrows.shape, dtype=np.int64)
    mask = qrows >= 0
    qhist[mask] = np.asarray(
        hist_videos[
            np.clip(query_users, 0, hist_videos.shape[0] - 1)
        ]
    )[mask]

    repeated_candidate = np.broadcast_to(candidate[:, None], qhist.shape)
    similarities = paired_sparse_lookup(
        graph, repeated_candidate, qhist
    ).reshape(qhist.shape)
    similarities[~mask] = 0.0

    decay = np.power(
        0.87, np.arange(qhist.shape[1], dtype=np.float32)
    )
    score = (similarities * decay[None, :]).sum(axis=1)
    return score.astype(np.float32)


def fit_transition_matrix(train, sorted_positive_rows, sorted_positive_users,
                          field):
    values = np.asarray(train.X[field], dtype=np.int64)
    previous_same_user = (
        sorted_positive_users[1:] == sorted_positive_users[:-1]
    )
    source_rows = sorted_positive_rows[:-1][previous_same_user]
    target_rows = sorted_positive_rows[1:][previous_same_user]

    source = values[source_rows]
    target = values[target_rows]
    good = (source > 0) & (target > 0)
    source = source[good]
    target = target[good]

    cardinality = int(FEATURE_CARDINALITIES[field])
    matrix = sp.coo_matrix(
        (
            np.ones(len(source), dtype=np.float32),
            (source.astype(np.int32), target.astype(np.int32)),
        ),
        shape=(cardinality, cardinality),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()

    source_mass = np.asarray(matrix.sum(axis=1)).ravel()
    target_mass = np.asarray(matrix.sum(axis=0)).ravel()
    row_counts = np.diff(matrix.indptr)
    row_ids = np.repeat(
        np.arange(cardinality, dtype=np.int64), row_counts
    )
    col_ids = matrix.indices

    denominator = np.sqrt(
        np.maximum(source_mass[row_ids], 1.0)
        * np.maximum(target_mass[col_ids], 1.0)
    )
    matrix.data = np.log1p(
        5.0 * matrix.data / denominator.astype(np.float32)
    ).astype(np.float32)
    return matrix


def transition_scores(split, train, history_rows, transition_models):
    users = np.asarray(split.user_id, dtype=np.int64)
    qrows = lookup_history_rows(history_rows, users)
    last_rows = qrows[:, 0]
    valid_history = last_rows >= 0

    score = np.zeros(len(users), dtype=np.float32)
    coefficients = {
        "video_id": 0.58,
        "author_id": 0.27,
        "tag": 0.15,
    }

    for field, matrix in transition_models.items():
        source = np.zeros(len(users), dtype=np.int64)
        source[valid_history] = np.asarray(
            train.X[field], dtype=np.int64
        )[last_rows[valid_history]]
        target = np.asarray(split.X[field], dtype=np.int64)

        component = paired_sparse_lookup(matrix, source, target)
        component[~valid_history] = 0.0
        score += coefficients[field] * component

    return score


def episodic_memory_scores(split, train, history_rows):
    users = np.asarray(split.user_id, dtype=np.int64)
    qrows = lookup_history_rows(history_rows, users)
    valid_history = qrows >= 0
    score = np.zeros(len(users), dtype=np.float32)

    slot_decay = np.power(
        0.84, np.arange(history_rows.shape[1], dtype=np.float32)
    )

    for field, coefficient in MEMORY_FIELDS:
        train_values = np.asarray(train.X[field], dtype=np.int64)
        cardinality = int(FEATURE_CARDINALITIES[field])

        hist_values = np.zeros(qrows.shape, dtype=np.int64)
        hist_values[valid_history] = train_values[
            qrows[valid_history]
        ]

        candidate = np.asarray(split.X[field], dtype=np.int64)
        frequencies = np.bincount(
            train_values,
            minlength=cardinality,
        ).astype(np.float64)

        idf = np.log1p(
            len(train_values) / np.maximum(frequencies, 1.0)
        )
        idf /= max(float(np.mean(idf[frequencies > 0])), 1e-6)
        idf = np.clip(idf, 0.25, 3.5).astype(np.float32)

        matches = (
            (hist_values == candidate[:, None])
            & valid_history
            & (candidate[:, None] > 0)
        )
        local = (
            matches.astype(np.float32) * slot_decay[None, :]
        ).sum(axis=1)
        score += coefficient * idf[
            np.minimum(candidate, len(idf) - 1)
        ] * local

    # Log compression prevents repeated exact video matches from overwhelming
    # all other evidence when blended with a calibrated incumbent.
    return np.log1p(np.maximum(score, 0.0)).astype(np.float32)


def within_user_rank(scores, users):
    """Average normalized ranks, preserving ties rather than row-ordering them."""
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users, dtype=np.int64)
    n = len(scores)
    if n == 0:
        return scores.copy()

    scores = np.nan_to_num(scores, nan=0.0, posinf=1e20, neginf=-1e20)
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

    average_local_rank = (
        0.5 * (tie_starts + tie_ends) - user_starts
    )
    denominator = np.maximum(user_ends - user_starts, 1)
    ranked_sorted = average_local_rank / denominator

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


def metric_primary(users, labels, scores):
    return float(evaluate(users, labels, scores)["primary"])


train = load("train")
valid = load("valid")
test = load("test")

valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)

history_rows, sorted_pos_rows, sorted_pos_users = build_positive_history(
    train, K_HISTORY
)

# Family 1: audience-based collaborative item graph.
recent_matrix, hist_videos = make_recent_interaction_matrix(
    train, history_rows
)
coview_graph = fit_coview_graph(recent_matrix)
coview_valid = coview_scores(
    valid, history_rows, hist_videos, coview_graph
)
coview_test = coview_scores(
    test, history_rows, hist_videos, coview_graph
)

del recent_matrix, coview_graph
gc.collect()

# Family 2: first-order sequence transition model over three resolutions.
transition_models = {
    field: fit_transition_matrix(
        train, sorted_pos_rows, sorted_pos_users, field
    )
    for field in ("video_id", "author_id", "tag")
}
transition_valid = transition_scores(
    valid, train, history_rows, transition_models
)
transition_test = transition_scores(
    test, train, history_rows, transition_models
)

del transition_models
gc.collect()

# Family 3: non-parametric episodic matching to recent positive content.
memory_valid = episodic_memory_scores(
    valid, train, history_rows
)
memory_test = episodic_memory_scores(
    test, train, history_rows
)

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

valid_models = {
    "coview_graph": np.asarray(coview_valid, dtype=np.float64),
    "sequential_transition": np.asarray(
        transition_valid, dtype=np.float64
    ),
    "episodic_content_memory": np.asarray(
        memory_valid, dtype=np.float64
    ),
}
test_models = {
    "coview_graph": np.asarray(coview_test, dtype=np.float64),
    "sequential_transition": np.asarray(
        transition_test, dtype=np.float64
    ),
    "episodic_content_memory": np.asarray(
        memory_test, dtype=np.float64
    ),
}

inc_valid_rank = within_user_rank(inc_valid, valid_users)
inc_test_rank = within_user_rank(inc_test, test_users)
valid_ranks = {
    name: within_user_rank(score, valid_users)
    for name, score in valid_models.items()
}
test_ranks = {
    name: within_user_rank(score, test_users)
    for name, score in test_models.items()
}

candidate_scores = {}
payloads = {}

inc_primary = metric_primary(valid_users, valid_y, inc_valid)
candidate_scores["trusted_incumbent"] = inc_primary
payloads["trusted_incumbent"] = (
    inc_valid,
    inc_test,
    None,
    False,
)

for name in valid_models:
    standalone_primary = metric_primary(
        valid_users, valid_y, valid_models[name]
    )
    candidate_scores[name + "_standalone"] = standalone_primary
    payloads[name + "_standalone"] = (
        valid_models[name],
        test_models[name],
        valid_models[name],
        False,
    )

blend_alphas = [0.04, 0.08, 0.12, 0.18, 0.25, 0.34]
for name in valid_models:
    for alpha in blend_alphas:
        valid_blend = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * valid_ranks[name]
        )
        test_blend = (
            (1.0 - alpha) * inc_test_rank
            + alpha * test_ranks[name]
        )
        key = f"{name}_blend_{alpha:.2f}"
        candidate_scores[key] = metric_primary(
            valid_users, valid_y, valid_blend
        )
        payloads[key] = (
            valid_blend,
            test_blend,
            valid_models[name],
            True,
        )

# A consensus of structurally distinct memories can be more robust than any
# individual sparse signal. It is still blended conservatively with incumbent.
consensus_valid = np.mean(
    np.column_stack(list(valid_ranks.values())), axis=1
)
consensus_test = np.mean(
    np.column_stack(list(test_ranks.values())), axis=1
)
for alpha in blend_alphas:
    valid_blend = (
        (1.0 - alpha) * inc_valid_rank
        + alpha * consensus_valid
    )
    test_blend = (
        (1.0 - alpha) * inc_test_rank
        + alpha * consensus_test
    )
    key = f"three_family_consensus_blend_{alpha:.2f}"
    candidate_scores[key] = metric_primary(
        valid_users, valid_y, valid_blend
    )
    payloads[key] = (
        valid_blend,
        test_blend,
        consensus_valid,
        True,
    )

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores, test_scores, raw_valid_scores, uses_external = payloads[
    winner
]

metrics = evaluate(valid_users, valid_y, valid_scores)

print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner,
            "incumbent_primary": inc_primary,
            "coview_graph_nnz_user_history": int(
                np.sum(history_rows >= 0)
            ),
            "users_with_positive_history": int(
                np.sum(history_rows[:, 0] >= 0)
            ),
        },
        sort_keys=True,
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_scores.items()},
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
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if uses_external and raw_valid_scores is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_valid_scores, dtype=np.float64),
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