import os
import time
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
LATENT_DIM = 32


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    boundary = np.empty(n, dtype=bool)
    boundary[0] = True
    boundary[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.maximum.accumulate(
        np.where(boundary, np.arange(n, dtype=np.int64), 0)
    )
    position = np.arange(n, dtype=np.int64) - starts

    _, counts = np.unique(sorted_users, return_counts=True)
    repeated_counts = np.repeat(counts, counts)
    denominator = np.maximum(repeated_counts - 1, 1)

    result = np.empty(n, dtype=np.float32)
    result[order] = (position / denominator).astype(np.float32)
    return result


def make_pair_mean_matrix(rows, cols, values, shape):
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)

    sums = sparse.coo_matrix(
        (values, (rows, cols)), shape=shape, dtype=np.float64
    ).tocsr()
    counts = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=shape,
        dtype=np.float64,
    ).tocsr()

    sums.sum_duplicates()
    counts.sum_duplicates()
    sums.sort_indices()
    counts.sort_indices()

    if (
        not np.array_equal(sums.indptr, counts.indptr)
        or not np.array_equal(sums.indices, counts.indices)
    ):
        raise RuntimeError("Sparse pair aggregation patterns differ")

    sums.data /= np.maximum(counts.data, 1.0)
    sums.eliminate_zeros()
    return sums.astype(np.float32)


def normalize_sparse(matrix, column_power=0.25):
    matrix = matrix.tocsr().astype(np.float64)

    row_norm = np.sqrt(
        np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    )
    row_scale = np.zeros_like(row_norm)
    good_rows = row_norm > 0
    row_scale[good_rows] = 1.0 / row_norm[good_rows]
    matrix = sparse.diags(row_scale).dot(matrix)

    col_frequency = np.asarray((matrix != 0).sum(axis=0)).ravel()
    col_scale = np.ones_like(col_frequency, dtype=np.float64)
    good_cols = col_frequency > 0
    col_scale[good_cols] = np.power(
        col_frequency[good_cols], -float(column_power)
    )
    matrix = matrix.dot(sparse.diags(col_scale))
    return matrix.tocsr().astype(np.float32)


def fit_latent(source, entity_name, mode):
    users = np.asarray(source.user_id, dtype=np.int64)
    if entity_name == "video_id":
        entities = np.asarray(source.video_id, dtype=np.int64)
    else:
        entities = np.asarray(source.X[entity_name], dtype=np.int64)

    y = np.asarray(source.y, dtype=np.float64)
    n_users = max(
        int(FEATURE_CARDINALITIES["user_id"]),
        int(users.max()) + 1,
    )
    n_entities = max(
        int(FEATURE_CARDINALITIES[entity_name]),
        int(entities.max()) + 1,
    )

    if mode == "positive":
        positive = y > 0
        matrix = sparse.coo_matrix(
            (
                np.ones(int(positive.sum()), dtype=np.float32),
                (users[positive], entities[positive]),
            ),
            shape=(n_users, n_entities),
            dtype=np.float32,
        ).tocsr()
        matrix.sum_duplicates()
        matrix.data = np.log1p(matrix.data)
        matrix = normalize_sparse(matrix, column_power=0.25)

    elif mode == "residual":
        prior = float(y.mean())
        residual = y - prior
        matrix = make_pair_mean_matrix(
            users, entities, residual, (n_users, n_entities)
        )
        matrix = normalize_sparse(matrix, column_power=0.10)

    else:
        raise ValueError(mode)

    k = min(LATENT_DIM, min(matrix.shape) - 1)
    if k < 2 or matrix.nnz == 0:
        return {
            "user": np.zeros((n_users, 2), dtype=np.float32),
            "entity": np.zeros((2, n_entities), dtype=np.float32),
            "entity_name": entity_name,
            "mode": mode,
        }

    # The previous attempt passed a RandomState instance, which is incompatible
    # with this SciPy/NumPy RNG transition. An integer seed is accepted.
    u, singular, vt = svds(
        matrix,
        k=k,
        which="LM",
        tol=1e-2,
        maxiter=400,
        random_state=42,
    )

    order = np.argsort(singular)[::-1]
    singular = singular[order]
    u = u[:, order]
    vt = vt[order]

    user_factors = (
        u * np.sqrt(np.maximum(singular, 0.0))[None, :]
    ).astype(np.float32)
    entity_factors = (
        np.sqrt(np.maximum(singular, 0.0))[:, None] * vt
    ).astype(np.float32)

    return {
        "user": user_factors,
        "entity": entity_factors,
        "entity_name": entity_name,
        "mode": mode,
    }


def score_latent(model, query):
    users = np.asarray(query.user_id, dtype=np.int64)
    if model["entity_name"] == "video_id":
        entities = np.asarray(query.video_id, dtype=np.int64)
    else:
        entities = np.asarray(
            query.X[model["entity_name"]], dtype=np.int64
        )

    user_factors = model["user"]
    entity_factors = model["entity"]

    valid = (
        (users >= 0)
        & (users < user_factors.shape[0])
        & (entities >= 0)
        & (entities < entity_factors.shape[1])
    )
    result = np.zeros(len(users), dtype=np.float32)
    idx = np.flatnonzero(valid)
    if len(idx):
        result[idx] = np.einsum(
            "ij,ji->i",
            user_factors[users[idx]],
            entity_factors[:, entities[idx]],
            optimize=True,
        ).astype(np.float32)
    return result


def build_transition_component(source, field):
    users = np.asarray(source.user_id, dtype=np.int64)
    times = np.asarray(source.time_ms, dtype=np.int64)
    y = np.asarray(source.y, dtype=np.int8)

    if field == "video_id":
        values = np.asarray(source.video_id, dtype=np.int64)
    else:
        values = np.asarray(source.X[field], dtype=np.int64)

    cardinality = max(
        int(FEATURE_CARDINALITIES[field]),
        int(values.max()) + 1,
    )
    n_users = max(
        int(FEATURE_CARDINALITIES["user_id"]),
        int(users.max()) + 1,
    )

    positive_rows = np.flatnonzero(y == 1)
    last_value = np.full(n_users, -1, dtype=np.int64)

    if len(positive_rows) == 0:
        return {
            "matrix": sparse.csr_matrix(
                (cardinality, cardinality), dtype=np.float32
            ),
            "last": last_value,
            "popularity": np.zeros(cardinality, dtype=np.float32),
            "field": field,
        }

    order = np.lexsort((
        positive_rows,
        times[positive_rows],
        users[positive_rows],
    ))
    rows = positive_rows[order]
    sorted_users = users[rows]
    sorted_values = values[rows]

    group_end = np.empty(len(rows), dtype=bool)
    group_end[-1] = True
    group_end[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_rows = np.flatnonzero(group_end)
    last_value[sorted_users[end_rows]] = sorted_values[end_rows]

    adjacent = sorted_users[1:] == sorted_users[:-1]
    previous = sorted_values[:-1][adjacent]
    current = sorted_values[1:][adjacent]

    transition = sparse.coo_matrix(
        (
            np.ones(len(previous), dtype=np.float32),
            (previous, current),
        ),
        shape=(cardinality, cardinality),
        dtype=np.float32,
    ).tocsr()
    transition.sum_duplicates()
    transition.data = np.log1p(transition.data)

    row_sum = np.asarray(transition.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sum)
    nonzero = row_sum > 0
    inv[nonzero] = 1.0 / row_sum[nonzero]
    transition = sparse.diags(inv).dot(transition).tocsr()

    popularity = np.bincount(
        sorted_values, minlength=cardinality
    ).astype(np.float32)
    popularity = np.log1p(popularity)
    maximum = float(popularity.max())
    if maximum > 0:
        popularity /= maximum

    return {
        "matrix": transition,
        "last": last_value,
        "popularity": popularity,
        "field": field,
    }


def score_transition_component(component, query):
    users = np.asarray(query.user_id, dtype=np.int64)
    field = component["field"]
    if field == "video_id":
        values = np.asarray(query.video_id, dtype=np.int64)
    else:
        values = np.asarray(query.X[field], dtype=np.int64)

    last = component["last"]
    matrix = component["matrix"]
    popularity = component["popularity"]

    result = np.zeros(len(users), dtype=np.float32)
    valid_users = (users >= 0) & (users < len(last))
    previous = np.full(len(users), -1, dtype=np.int64)
    previous[valid_users] = last[users[valid_users]]

    valid = (
        (previous >= 0)
        & (previous < matrix.shape[0])
        & (values >= 0)
        & (values < matrix.shape[1])
    )
    idx = np.flatnonzero(valid)
    if len(idx):
        result[idx] = np.asarray(
            matrix[previous[idx], values[idx]]
        ).ravel().astype(np.float32)

    valid_values = (values >= 0) & (values < len(popularity))
    result[valid_values] += 0.03 * popularity[values[valid_values]]
    return result


def fit_transition_model(source):
    return {
        "video_id": build_transition_component(source, "video_id"),
        "author_id": build_transition_component(source, "author_id"),
        "tag": build_transition_component(source, "tag"),
    }


def score_transition_model(model, query):
    video = score_transition_component(model["video_id"], query)
    author = score_transition_component(model["author_id"], query)
    tag = score_transition_component(model["tag"], query)
    return (
        1.00 * video + 0.75 * author + 0.45 * tag
    ).astype(np.float32)


class CombinedSplit:
    pass


def combine_splits(a, b):
    c = CombinedSplit()
    c.user_id = np.concatenate([
        np.asarray(a.user_id), np.asarray(b.user_id)
    ])
    c.video_id = np.concatenate([
        np.asarray(a.video_id), np.asarray(b.video_id)
    ])
    c.time_ms = np.concatenate([
        np.asarray(a.time_ms), np.asarray(b.time_ms)
    ])
    c.date = np.concatenate([
        np.asarray(a.date), np.asarray(b.date)
    ])
    c.y = np.concatenate([
        np.asarray(a.y), np.asarray(b.y)
    ])
    c.X = {
        name: np.concatenate([
            np.asarray(a.X[name]), np.asarray(b.X[name])
        ])
        for name in ["author_id", "tag"]
    }
    return c


train = load("train")
valid = load("valid")
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float32, copy=False)
inc_rank = within_user_rank(valid.user_id, inc_valid)
inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)

candidate_scores = {}
candidate_log = {
    "trusted_incumbent": float(inc_metrics["primary"])
}

latent_specs = [
    ("latent_positive_video", "video_id", "positive"),
    ("latent_residual_video", "video_id", "residual"),
    ("latent_positive_author", "author_id", "positive"),
]

for name, field, mode in latent_specs:
    model = fit_latent(train, field, mode)
    candidate_scores[name] = score_latent(model, valid)
    del model

transition_model = fit_transition_model(train)
candidate_scores["chronological_transition"] = score_transition_model(
    transition_model, valid
)
del transition_model

blend_grid = np.asarray(
    [0.00, 0.05, 0.10, 0.15, 0.20, 0.25,
     0.30, 0.35, 0.40, 0.50, 0.60],
    dtype=np.float64,
)

best = {
    "name": "trusted_incumbent",
    "alpha": 0.0,
    "primary": float(inc_metrics["primary"]),
    "scores": inc_valid.copy(),
}

for name, raw in candidate_scores.items():
    family_rank = within_user_rank(valid.user_id, raw)
    standalone = evaluate(valid.user_id, y_valid, family_rank)
    candidate_log[name] = float(standalone["primary"])

    local_best = -np.inf
    local_alpha = 0.0
    local_scores = None

    for alpha in blend_grid:
        blended = (
            (1.0 - float(alpha)) * inc_rank
            + float(alpha) * family_rank
        ).astype(np.float32)
        primary = float(
            evaluate(valid.user_id, y_valid, blended)["primary"]
        )
        if primary > local_best:
            local_best = primary
            local_alpha = float(alpha)
            local_scores = blended.copy()

    candidate_log[name + "_blend"] = local_best
    candidate_log[name + "_alpha"] = local_alpha

    if local_best > best["primary"]:
        best = {
            "name": name,
            "alpha": local_alpha,
            "primary": local_best,
            "scores": local_scores,
        }

valid_scores = np.asarray(best["scores"], dtype=np.float32)
metrics = evaluate(valid.user_id, y_valid, valid_scores)

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print("FINDINGS " + json.dumps({
    "selected_family": best["name"],
    "selected_alpha": best["alpha"],
    "svds_rng_fix": "integer_seed",
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

test = load("test")
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float32, copy=False)

if best["name"] == "trusted_incumbent" or best["alpha"] <= 0:
    test_scores = inc_test
else:
    combined = combine_splits(train, valid)

    if best["name"] == "chronological_transition":
        selected_model = fit_transition_model(combined)
        new_test_raw = score_transition_model(selected_model, test)
    else:
        spec_lookup = {
            name: (field, mode)
            for name, field, mode in latent_specs
        }
        field, mode = spec_lookup[best["name"]]
        selected_model = fit_latent(combined, field, mode)
        new_test_raw = score_latent(selected_model, test)

    new_test_rank = within_user_rank(test.user_id, new_test_raw)
    inc_test_rank = within_user_rank(test.user_id, inc_test)
    test_scores = (
        (1.0 - best["alpha"]) * inc_test_rank
        + best["alpha"] * new_test_rank
    ).astype(np.float32)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))