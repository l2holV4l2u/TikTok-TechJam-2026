import os
import time
import json
import numpy as np
from scipy import sparse

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
HALF_LIFE = 4.0
TOPK = 64
USER_BATCH = 1500

np.random.seed(73129)


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int64)
    unique_days = np.sort(np.unique(dates))
    mapping = {int(d): i for i, d in enumerate(unique_days)}
    day_index = np.asarray([mapping[int(d)] for d in dates], dtype=np.float32)
    age = float(len(unique_days) - 1) - day_index
    return np.exp(-np.log(2.0) * age / HALF_LIFE).astype(np.float32)


def make_sparse(rows, cols, values, shape):
    mat = sparse.coo_matrix(
        (values.astype(np.float32), (rows, cols)),
        shape=shape,
        dtype=np.float32
    ).tocsr()
    mat.sum_duplicates()
    return mat


def binary_positive_matrix(users, videos, labels, n_users, n_videos):
    positive = labels > 0
    mat = make_sparse(
        users[positive],
        videos[positive],
        np.ones(int(positive.sum()), dtype=np.float32),
        (n_users, n_videos)
    )
    mat.data[:] = 1.0
    return mat


def prune_rows_topk(mat, topk):
    mat = mat.tocsr()
    indptr = mat.indptr
    indices = mat.indices
    data = mat.data

    out_indices = []
    out_data = []
    out_indptr = np.zeros(mat.shape[0] + 1, dtype=np.int64)

    total = 0
    for row in range(mat.shape[0]):
        lo, hi = indptr[row], indptr[row + 1]
        vals = data[lo:hi]
        inds = indices[lo:hi]

        if vals.size > topk:
            keep = np.argpartition(vals, -topk)[-topk:]
            vals = vals[keep]
            inds = inds[keep]

        if vals.size:
            order = np.argsort(inds)
            out_indices.append(inds[order].astype(np.int32, copy=False))
            out_data.append(vals[order].astype(np.float32, copy=False))
            total += vals.size

        out_indptr[row + 1] = total

    if total:
        out_indices = np.concatenate(out_indices)
        out_data = np.concatenate(out_data)
    else:
        out_indices = np.empty(0, dtype=np.int32)
        out_data = np.empty(0, dtype=np.float32)

    return sparse.csr_matrix(
        (out_data, out_indices, out_indptr),
        shape=mat.shape,
        dtype=np.float32
    )


def build_item_graphs(binary_positive):
    item_degree = np.asarray(binary_positive.sum(axis=0)).ravel().astype(np.float32)
    item_degree = np.maximum(item_degree, 1.0)

    cooc = (binary_positive.T @ binary_positive).tocsr().astype(np.float32)
    cooc.setdiag(0.0)
    cooc.eliminate_zeros()

    inv_sqrt = 1.0 / np.sqrt(item_degree)
    cosine = sparse.diags(inv_sqrt) @ cooc @ sparse.diags(inv_sqrt)
    cosine = prune_rows_topk(cosine.tocsr(), TOPK)

    user_degree = np.asarray(binary_positive.sum(axis=1)).ravel().astype(np.float32)
    user_degree = np.maximum(user_degree, 1.0)
    weighted_users = sparse.diags(1.0 / user_degree) @ binary_positive
    resource = (binary_positive.T @ weighted_users).tocsr().astype(np.float32)
    resource.setdiag(0.0)
    resource.eliminate_zeros()

    # Penalize globally common destinations while retaining asymmetric
    # resource-allocation transitions.
    resource = resource @ sparse.diags(1.0 / np.sqrt(item_degree))
    resource = prune_rows_topk(resource.tocsr(), TOPK)

    return cosine, resource, item_degree


def pair_scores(left_matrix, transition, target_users, target_items):
    target_users = np.asarray(target_users, dtype=np.int64)
    target_items = np.asarray(target_items, dtype=np.int64)

    unique_users, inverse = np.unique(target_users, return_inverse=True)
    output = np.zeros(target_users.size, dtype=np.float32)

    for start in range(0, unique_users.size, USER_BATCH):
        end = min(start + USER_BATCH, unique_users.size)
        block_users = unique_users[start:end]
        propagated = (left_matrix[block_users] @ transition).tocsr()

        selected = np.flatnonzero((inverse >= start) & (inverse < end))
        local_rows = inverse[selected] - start
        values = propagated[local_rows, target_items[selected]]
        output[selected] = np.asarray(values).reshape(-1).astype(np.float32)

    return output.astype(np.float64)


def direct_entity_scores(profile, target_users, target_entities):
    values = profile[
        np.asarray(target_users, dtype=np.int64),
        np.asarray(target_entities, dtype=np.int64)
    ]
    return np.asarray(values).reshape(-1).astype(np.float64)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size

    # Stable row index only resolves exact ties after the graph score has been
    # augmented by a small popularity prior.
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    lengths = ends - starts
    local = np.arange(n, dtype=np.int64) - np.repeat(starts, lengths)

    ranked_sorted = (local.astype(np.float64) + 0.5) / np.repeat(lengths, lengths)
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def build_graph_state(train):
    users = np.asarray(train.X["user_id"], dtype=np.int64)
    videos = np.asarray(train.X["video_id"], dtype=np.int64)
    authors = np.asarray(train.X["author_id"], dtype=np.int64)
    tags = np.asarray(train.X["tag"], dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.int8)
    weights = recency_weights(train.date)

    n_users = int(FEATURE_CARDINALITIES["user_id"])
    n_videos = int(FEATURE_CARDINALITIES["video_id"])
    n_authors = int(FEATURE_CARDINALITIES["author_id"])
    n_tags = int(FEATURE_CARDINALITIES["tag"])

    binary_positive = binary_positive_matrix(
        users, videos, labels, n_users, n_videos
    )
    cosine, resource, item_degree = build_item_graphs(binary_positive)

    positive = labels > 0
    positive_profile = make_sparse(
        users[positive],
        videos[positive],
        weights[positive],
        (n_users, n_videos)
    )

    signed_values = weights * np.where(labels > 0, 1.0, -0.30).astype(np.float32)
    signed_profile = make_sparse(
        users, videos, signed_values, (n_users, n_videos)
    )

    author_profile = make_sparse(
        users[positive],
        authors[positive],
        weights[positive],
        (n_users, n_authors)
    )
    author_degree = np.asarray(author_profile.sum(axis=0)).ravel().astype(np.float32)
    author_scale = np.power(np.maximum(author_degree, 1.0), -0.35)
    author_profile = author_profile @ sparse.diags(author_scale)

    tag_profile = make_sparse(
        users[positive],
        tags[positive],
        weights[positive],
        (n_users, n_tags)
    )
    tag_degree = np.asarray(tag_profile.sum(axis=0)).ravel().astype(np.float32)
    tag_scale = np.power(np.maximum(tag_degree, 1.0), -0.30)
    tag_profile = tag_profile @ sparse.diags(tag_scale)

    video_popularity = np.log1p(item_degree).astype(np.float64)

    return {
        "positive_profile": positive_profile.tocsr(),
        "signed_profile": signed_profile.tocsr(),
        "author_profile": author_profile.tocsr(),
        "tag_profile": tag_profile.tocsr(),
        "cosine": cosine,
        "resource": resource,
        "video_popularity": video_popularity,
    }


def graph_predictions(split, state):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    videos = np.asarray(split.X["video_id"], dtype=np.int64)
    authors = np.asarray(split.X["author_id"], dtype=np.int64)
    tags = np.asarray(split.X["tag"], dtype=np.int64)

    pop = state["video_popularity"][videos]
    tie_prior = 1e-6 * pop

    cosine_positive = pair_scores(
        state["positive_profile"], state["cosine"], users, videos
    ) + tie_prior
    resource_positive = pair_scores(
        state["positive_profile"], state["resource"], users, videos
    ) + tie_prior
    cosine_signed = pair_scores(
        state["signed_profile"], state["cosine"], users, videos
    ) + tie_prior

    author = direct_entity_scores(
        state["author_profile"], users, authors
    ) + tie_prior
    tag = direct_entity_scores(
        state["tag_profile"], users, tags
    ) + tie_prior

    ranks = {
        "graph_cosine_positive": within_user_rank(users, cosine_positive),
        "graph_resource_positive": within_user_rank(users, resource_positive),
        "graph_cosine_signed": within_user_rank(users, cosine_signed),
        "graph_author": within_user_rank(users, author),
        "graph_tag": within_user_rank(users, tag),
    }

    ranks["graph_item_ensemble"] = (
        ranks["graph_cosine_positive"]
        + ranks["graph_resource_positive"]
        + ranks["graph_cosine_signed"]
    ) / 3.0

    ranks["graph_heterogeneous"] = (
        0.55 * ranks["graph_item_ensemble"]
        + 0.30 * ranks["graph_author"]
        + 0.15 * ranks["graph_tag"]
    )

    return ranks


train = load("train")
valid = load("valid")

state = build_graph_state(train)
valid_graph = graph_predictions(valid, state)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.load(inc_valid_path).astype(np.float64)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
valid_labels = np.asarray(valid.y, dtype=np.int8)
inc_valid_rank = within_user_rank(valid_users, inc_valid)

candidate_scores = {}
candidate_primary = {}
candidate_raw = {}
candidate_spec = {}

for family, own_rank in valid_graph.items():
    metrics = evaluate(valid_users, valid_labels, own_rank)
    candidate_scores[family] = own_rank
    candidate_raw[family] = own_rank
    candidate_primary[family] = float(metrics["primary"])
    candidate_spec[family] = (family, None)

    for incumbent_weight in (0.50, 0.70, 0.85, 0.93):
        name = family + "_blend_" + str(incumbent_weight)
        blended = (
            incumbent_weight * inc_valid_rank
            + (1.0 - incumbent_weight) * own_rank
        )
        metrics = evaluate(valid_users, valid_labels, blended)
        candidate_scores[name] = blended
        candidate_raw[name] = own_rank
        candidate_primary[name] = float(metrics["primary"])
        candidate_spec[name] = (family, incumbent_weight)

best_name = max(candidate_primary, key=candidate_primary.get)
best_valid_scores = candidate_scores[best_name]
best_family, best_incumbent_weight = candidate_spec[best_name]
best_metrics = evaluate(valid_users, valid_labels, best_valid_scores)

print("CANDIDATES " + json.dumps(
    {k: round(v, 6) for k, v in sorted(candidate_primary.items())},
    sort_keys=True
))
print(
    "FINDINGS winner=%s graph_nnz_cosine=%d graph_nnz_resource=%d"
    % (
        best_name,
        int(state["cosine"].nnz),
        int(state["resource"].nnz),
    )
)

test = load("test")
test_graph = graph_predictions(test, state)
own_test_rank = test_graph[best_family]

if best_incumbent_weight is None:
    test_scores = own_test_rank
else:
    inc_test = np.load(inc_test_path).astype(np.float64)
    inc_test_rank = within_user_rank(np.asarray(test.user_id), inc_test)
    test_scores = (
        best_incumbent_weight * inc_test_rank
        + (1.0 - best_incumbent_weight) * own_test_rank
    )

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64)
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )
    if best_incumbent_weight is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[best_name], dtype=np.float64)
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))