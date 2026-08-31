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
np.random.seed(271828)

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


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def recency_weights(dates, half_life=5.0):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    age = unique_dates.size - 1 - np.searchsorted(unique_dates, dates)
    return np.exp2(-age.astype(np.float64) / float(half_life))


def within_user_rank(user_ids, scores):
    """Tie-preserving normalized ranks within each logged impression set."""
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    su = user_ids[order]
    ss = scores[order]

    user_start = np.empty(n, dtype=bool)
    user_start[0] = True
    user_start[1:] = su[1:] != su[:-1]
    user_starts = np.flatnonzero(user_start)
    user_ends = np.r_[user_starts[1:], n]
    user_lengths = user_ends - user_starts

    repeated_user_start = np.repeat(user_starts, user_lengths)
    repeated_user_length = np.repeat(user_lengths, user_lengths)

    tie_start = np.empty(n, dtype=bool)
    tie_start[0] = True
    tie_start[1:] = (su[1:] != su[:-1]) | (ss[1:] != ss[:-1])
    tie_starts = np.flatnonzero(tie_start)
    tie_ends = np.r_[tie_starts[1:], n]
    tie_lengths = tie_ends - tie_starts

    tie_midpoint = 0.5 * (
        tie_starts.astype(np.float64)
        + tie_ends.astype(np.float64)
        - 1.0
    )
    sorted_rank = (
        np.repeat(tie_midpoint, tie_lengths)
        - repeated_user_start.astype(np.float64)
    )

    multi = repeated_user_length > 1
    sorted_rank[multi] /= (
        repeated_user_length[multi].astype(np.float64) - 1.0
    )
    sorted_rank[~multi] = 0.5

    result = np.empty(n, dtype=np.float64)
    result[order] = sorted_rank
    return result


def weighted_entity_logit(ids, labels, weights, cardinality, prior):
    count = np.bincount(
        ids, weights=weights, minlength=cardinality
    ).astype(np.float64)
    positive = np.bincount(
        ids, weights=weights * labels, minlength=cardinality
    ).astype(np.float64)
    global_rate = float(np.sum(weights * labels) / np.sum(weights))
    rate = (positive + prior * global_rate) / (count + prior)
    return logit(rate).astype(np.float32), count


weights = recency_weights(train.date, half_life=5.0)
weighted_global_rate = float(
    np.sum(weights * train_y) / np.sum(weights)
)
weighted_global_logit = float(logit(weighted_global_rate))


# ---------------------------------------------------------------------
# Family 1: dependency-free behavioral user-archetype co-clustering.
#
# Users are embedded from recency-weighted, reliability-shrunk preferences
# over several content vocabularies. A small Lloyd implementation replaces
# sklearn MiniBatchKMeans. Cluster-conditioned content rates then pool
# evidence across behaviorally similar users.
# ---------------------------------------------------------------------

ARCH_FIELDS = (
    "tag",
    "duration_bucket",
    "onehot_feat3",
    "tab",
    "author_id",
)


def make_user_behavior_matrix(split, labels, row_weights):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    nu = FEATURE_CARDINALITIES["user_id"]

    offsets = {}
    total_columns = 0
    for field in ARCH_FIELDS:
        offsets[field] = total_columns
        total_columns += FEATURE_CARDINALITIES[field]

    row_parts = []
    col_parts = []
    count_parts = []
    positive_parts = []

    for field in ARCH_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        cols = ids + offsets[field]

        count_matrix = sp.coo_matrix(
            (row_weights.astype(np.float32), (users, cols)),
            shape=(nu, total_columns),
            dtype=np.float32,
        ).tocsr()
        positive_matrix = sp.coo_matrix(
            ((row_weights * labels).astype(np.float32), (users, cols)),
            shape=(nu, total_columns),
            dtype=np.float32,
        ).tocsr()
        count_matrix.sum_duplicates()
        positive_matrix.sum_duplicates()

        count_coo = count_matrix.tocoo()
        positive_coo = positive_matrix.tocoo()

        row_parts.append(count_coo.row)
        col_parts.append(count_coo.col)
        count_parts.append(count_coo.data)

        # The two matrices have the same structural coordinates except for
        # entries whose weighted positive sum is exactly zero. Reindex via
        # sparse paired lookup to keep construction robust.
        pos_values = np.asarray(
            positive_matrix[count_coo.row, count_coo.col]
        ).reshape(-1)
        positive_parts.append(pos_values)

    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    counts = np.concatenate(count_parts).astype(np.float64)
    positives = np.concatenate(positive_parts).astype(np.float64)

    prior = 8.0
    local_rate = (
        positives + prior * weighted_global_rate
    ) / (counts + prior)
    residual = logit(local_rate) - weighted_global_logit
    residual *= np.sqrt(counts / (counts + 8.0))

    matrix = sp.coo_matrix(
        (residual.astype(np.float32), (rows, cols)),
        shape=(nu, total_columns),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()

    norm = np.sqrt(
        np.asarray(matrix.multiply(matrix).sum(axis=1)).reshape(-1)
    )
    inv = np.zeros_like(norm, dtype=np.float32)
    usable = norm > 1e-8
    inv[usable] = 1.0 / norm[usable]
    matrix = sp.diags(inv).dot(matrix).tocsr()
    return matrix, usable


def numpy_lloyd(features, usable, n_clusters=24, iterations=18):
    n, dim = features.shape
    usable_rows = np.flatnonzero(usable)
    if usable_rows.size < n_clusters:
        n_clusters = max(2, usable_rows.size)

    projection = features[usable_rows, 0]
    quantiles = np.linspace(0.0, 1.0, n_clusters + 2)[1:-1]
    positions = np.clip(
        (quantiles * (usable_rows.size - 1)).astype(np.int64),
        0,
        usable_rows.size - 1,
    )
    ordered = usable_rows[np.argsort(projection)]
    centers = features[ordered[positions]].copy()

    assignment = np.zeros(n, dtype=np.int32)
    usable_features = features[usable_rows]

    for iteration in range(iterations):
        x2 = np.sum(usable_features * usable_features, axis=1)[:, None]
        c2 = np.sum(centers * centers, axis=1)[None, :]
        distance = x2 + c2 - 2.0 * usable_features.dot(centers.T)
        local_assignment = np.argmin(distance, axis=1).astype(np.int32)
        assignment[usable_rows] = local_assignment

        new_centers = np.zeros_like(centers)
        cluster_count = np.bincount(
            local_assignment, minlength=n_clusters
        ).astype(np.float64)

        for dimension in range(dim):
            new_centers[:, dimension] = np.bincount(
                local_assignment,
                weights=usable_features[:, dimension],
                minlength=n_clusters,
            )

        nonempty = cluster_count > 0
        new_centers[nonempty] /= cluster_count[nonempty, None]

        if np.any(~nonempty):
            farthest = np.argsort(np.min(distance, axis=1))[::-1]
            empty_ids = np.flatnonzero(~nonempty)
            for j, cluster_id in enumerate(empty_ids):
                new_centers[cluster_id] = usable_features[
                    farthest[j % farthest.size]
                ]

        shift = float(np.mean((new_centers - centers) ** 2))
        centers = new_centers
        if shift < 1e-8:
            break

    assignment[~usable] = 0
    return assignment, centers


def fit_archetype_model(split, labels, row_weights):
    behavior, usable = make_user_behavior_matrix(
        split, labels, row_weights
    )

    rank = min(14, min(behavior.shape) - 1)
    u, singular, _ = svds(
        behavior.astype(np.float64),
        k=rank,
        which="LM",
        random_state=271828,
    )
    order = np.argsort(singular)[::-1]
    embedding = u[:, order] * singular[order][None, :]

    norm = np.linalg.norm(embedding, axis=1)
    good = usable & (norm > 1e-10)
    embedding[good] /= norm[good, None]

    user_cluster, centers = numpy_lloyd(
        embedding, good, n_clusters=24, iterations=18
    )
    n_clusters = centers.shape[0]

    users = np.asarray(split.X["user_id"], dtype=np.int64)
    row_cluster = user_cluster[users]

    tables = {}
    coefficients = {
        "tag": 0.24,
        "duration_bucket": 0.16,
        "onehot_feat3": 0.14,
        "tab": 0.12,
        "author_id": 0.34,
    }
    priors = {
        "tag": 45.0,
        "duration_bucket": 60.0,
        "onehot_feat3": 40.0,
        "tab": 100.0,
        "author_id": 30.0,
    }

    for field in ARCH_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        card = FEATURE_CARDINALITIES[field]
        joint = row_cluster.astype(np.int64) * card + ids

        count = np.bincount(
            joint,
            weights=row_weights,
            minlength=n_clusters * card,
        ).astype(np.float64)
        positive = np.bincount(
            joint,
            weights=row_weights * labels,
            minlength=n_clusters * card,
        ).astype(np.float64)

        prior = priors[field]
        rate = (
            positive + prior * weighted_global_rate
        ) / (count + prior)
        table = logit(rate) - weighted_global_logit
        tables[field] = table.reshape(n_clusters, card).astype(
            np.float32
        )

    diagnostics = {
        "usable_users": int(good.sum()),
        "clusters": int(n_clusters),
        "cluster_min": int(
            np.bincount(
                user_cluster[good], minlength=n_clusters
            ).min()
        ),
        "cluster_max": int(
            np.bincount(
                user_cluster[good], minlength=n_clusters
            ).max()
        ),
    }

    return {
        "user_cluster": user_cluster,
        "tables": tables,
        "coefficients": coefficients,
        "diagnostics": diagnostics,
    }


def predict_archetype(split, model):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    clusters = model["user_cluster"][users]
    result = np.zeros(users.size, dtype=np.float64)

    for field, coefficient in model["coefficients"].items():
        ids = np.asarray(split.X[field], dtype=np.int64)
        result += coefficient * model["tables"][field][clusters, ids]

    return result


archetype_model = fit_archetype_model(train, train_y, weights)
archetype_valid = predict_archetype(valid, archetype_model)

print(
    "FINDINGS archetype_model="
    + json.dumps(archetype_model["diagnostics"], sort_keys=True)
)


# ---------------------------------------------------------------------
# Family 2: explicit user-content preference tables.
#
# This is not a global entity-quality model: each prediction retrieves
# the target user's shrunk residual preference for the candidate's author,
# tag, duration and metadata classes. Global content quality is included
# only as a backoff when a user-content pair is unobserved.
# ---------------------------------------------------------------------

PREFERENCE_FIELDS = (
    "author_id",
    "tag",
    "duration_bucket",
    "onehot_feat3",
    "tab",
)
PREFERENCE_PRIORS = {
    "author_id": 7.0,
    "tag": 10.0,
    "duration_bucket": 14.0,
    "onehot_feat3": 10.0,
    "tab": 18.0,
}
PREFERENCE_COEFFICIENTS = {
    "author_id": 0.32,
    "tag": 0.22,
    "duration_bucket": 0.18,
    "onehot_feat3": 0.14,
    "tab": 0.14,
}


def fit_preference_tables(split, labels, row_weights):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    nu = FEATURE_CARDINALITIES["user_id"]

    tables = {}
    global_tables = {}
    nonzero = {}

    for field in PREFERENCE_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        card = FEATURE_CARDINALITIES[field]

        count = sp.coo_matrix(
            (row_weights.astype(np.float32), (users, ids)),
            shape=(nu, card),
            dtype=np.float32,
        ).tocsr()
        positive = sp.coo_matrix(
            ((row_weights * labels).astype(np.float32), (users, ids)),
            shape=(nu, card),
            dtype=np.float32,
        ).tocsr()
        count.sum_duplicates()
        positive.sum_duplicates()

        count_coo = count.tocoo()
        pos_values = np.asarray(
            positive[count_coo.row, count_coo.col]
        ).reshape(-1).astype(np.float64)
        local_count = count_coo.data.astype(np.float64)

        prior = PREFERENCE_PRIORS[field]
        local_rate = (
            pos_values + prior * weighted_global_rate
        ) / (local_count + prior)
        residual = logit(local_rate) - weighted_global_logit
        residual *= np.sqrt(local_count / (local_count + prior))

        table = sp.coo_matrix(
            (
                residual.astype(np.float32),
                (count_coo.row, count_coo.col),
            ),
            shape=(nu, card),
            dtype=np.float32,
        ).tocsr()
        tables[field] = table
        nonzero[field] = int(table.nnz)

        global_score, _ = weighted_entity_logit(
            ids, labels, row_weights, card, prior=35.0
        )
        global_tables[field] = (
            global_score.astype(np.float64) - weighted_global_logit
        ).astype(np.float32)

    return {
        "tables": tables,
        "global_tables": global_tables,
        "nonzero": nonzero,
    }


def paired_sparse_lookup(matrix, rows, cols):
    return np.asarray(matrix[rows, cols]).reshape(-1).astype(np.float64)


def predict_preference(split, model):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    result = np.zeros(users.size, dtype=np.float64)
    backoff = np.zeros(users.size, dtype=np.float64)

    for field in PREFERENCE_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        coefficient = PREFERENCE_COEFFICIENTS[field]
        personal = paired_sparse_lookup(
            model["tables"][field], users, ids
        )
        global_effect = model["global_tables"][field][ids].astype(
            np.float64
        )

        result += coefficient * personal
        backoff += coefficient * global_effect

    # Preserve personalized residuals while retaining stable candidate
    # quality for pairs absent from a sparse user's history.
    return 0.72 * result + 0.28 * backoff


preference_model = fit_preference_tables(
    train, train_y, weights
)
preference_valid = predict_preference(valid, preference_model)

print(
    "FINDINGS preference_nonzero="
    + json.dumps(preference_model["nonzero"], sort_keys=True)
)


# ---------------------------------------------------------------------
# Family 3: graph-smoothed item quality.
#
# A recency-weighted user-video residual graph produces latent video
# coordinates. Each video's direct empirical quality is then averaged with
# nearby videos in that graph. This can stabilize temporally noisy item IDs
# while retaining collaborative rather than metadata-only neighborhoods.
# ---------------------------------------------------------------------

def fit_graph_quality(split, labels, row_weights):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    videos = np.asarray(split.X["video_id"], dtype=np.int64)
    nu = FEATURE_CARDINALITIES["user_id"]
    nv = FEATURE_CARDINALITIES["video_id"]

    residual_value = (
        row_weights * (labels - weighted_global_rate)
    ).astype(np.float32)

    residual = sp.coo_matrix(
        (residual_value, (users, videos)),
        shape=(nu, nv),
        dtype=np.float32,
    ).tocsr()
    residual.sum_duplicates()

    row_norm = np.sqrt(
        np.asarray(residual.multiply(residual).sum(axis=1)).reshape(-1)
    )
    inv_row = np.zeros_like(row_norm, dtype=np.float32)
    usable_rows = row_norm > 1e-8
    inv_row[usable_rows] = 1.0 / row_norm[usable_rows]
    residual = sp.diags(inv_row).dot(residual).tocsr()

    rank = min(20, min(residual.shape) - 1)
    _, singular, vt = svds(
        residual.astype(np.float64),
        k=rank,
        which="LM",
        random_state=271828,
    )
    order = np.argsort(singular)[::-1]
    item_embedding = vt[order].T * singular[order][None, :]

    item_norm = np.linalg.norm(item_embedding, axis=1)
    usable_items = item_norm > 1e-10
    item_embedding[usable_items] /= item_norm[usable_items, None]

    direct_logit, direct_count = weighted_entity_logit(
        videos, labels, row_weights, nv, prior=24.0
    )
    direct_effect = (
        direct_logit.astype(np.float64) - weighted_global_logit
    )

    usable_ids = np.flatnonzero(usable_items)
    tree = cKDTree(item_embedding[usable_ids])
    query_k = min(21, usable_ids.size)
    distance, local_neighbor = tree.query(
        item_embedding[usable_ids],
        k=query_k,
        workers=-1,
    )

    if query_k == 1:
        distance = distance[:, None]
        local_neighbor = local_neighbor[:, None]

    neighbor_ids = usable_ids[local_neighbor]
    similarity = np.maximum(
        1.0 - 0.5 * np.square(distance), 0.0
    )

    self_mask = neighbor_ids == usable_ids[:, None]
    similarity[self_mask] = 0.0

    reliability = (
        direct_count[neighbor_ids]
        / (direct_count[neighbor_ids] + 25.0)
    )
    neighbor_weight = similarity * reliability
    denominator = neighbor_weight.sum(axis=1)

    smoothed = direct_effect.copy()
    valid_denominator = denominator > 1e-10
    neighbor_average = np.zeros(usable_ids.size, dtype=np.float64)
    neighbor_average[valid_denominator] = (
        np.sum(
            neighbor_weight[valid_denominator]
            * direct_effect[neighbor_ids[valid_denominator]],
            axis=1,
        )
        / denominator[valid_denominator]
    )

    own_reliability = (
        direct_count[usable_ids]
        / (direct_count[usable_ids] + 35.0)
    )
    smoothed[usable_ids] = (
        own_reliability * direct_effect[usable_ids]
        + (1.0 - own_reliability) * neighbor_average
    )

    content_fields = (
        "author_id",
        "tag",
        "duration_bucket",
        "tab",
    )
    content_tables = {}
    for field in content_fields:
        ids = np.asarray(split.X[field], dtype=np.int64)
        card = FEATURE_CARDINALITIES[field]
        score, _ = weighted_entity_logit(
            ids, labels, row_weights, card, prior=45.0
        )
        content_tables[field] = (
            score.astype(np.float64) - weighted_global_logit
        ).astype(np.float32)

    diagnostics = {
        "usable_items": int(usable_items.sum()),
        "mean_direct_count": float(direct_count.mean()),
        "mean_neighbor_weight": float(
            denominator[valid_denominator].mean()
        ) if np.any(valid_denominator) else 0.0,
    }

    return {
        "smoothed_video": smoothed.astype(np.float32),
        "content_tables": content_tables,
        "diagnostics": diagnostics,
    }


def predict_graph_quality(split, model):
    videos = np.asarray(split.X["video_id"], dtype=np.int64)
    result = 0.64 * model["smoothed_video"][videos].astype(
        np.float64
    )

    coefficients = {
        "author_id": 0.18,
        "tag": 0.08,
        "duration_bucket": 0.06,
        "tab": 0.04,
    }
    for field, coefficient in coefficients.items():
        ids = np.asarray(split.X[field], dtype=np.int64)
        result += (
            coefficient
            * model["content_tables"][field][ids].astype(np.float64)
        )
    return result


graph_model = fit_graph_quality(train, train_y, weights)
graph_valid = predict_graph_quality(valid, graph_model)

print(
    "FINDINGS graph_quality="
    + json.dumps(graph_model["diagnostics"], sort_keys=True)
)


# ---------------------------------------------------------------------
# Validation comparison: each standalone family, each incumbent blend,
# and rank ensembles of structurally different new families.
# ---------------------------------------------------------------------

valid_raw = {
    "archetype": archetype_valid,
    "preference_table": preference_valid,
    "graph_quality": graph_valid,
}

valid_rank = {
    name: within_user_rank(valid_users, score)
    for name, score in valid_raw.items()
}
inc_valid_rank = within_user_rank(valid_users, inc_valid)

family_recipes = {
    "archetype": {"archetype": 1.0},
    "preference_table": {"preference_table": 1.0},
    "graph_quality": {"graph_quality": 1.0},
    "archetype_preference": {
        "archetype": 0.5,
        "preference_table": 0.5,
    },
    "preference_graph": {
        "preference_table": 0.5,
        "graph_quality": 0.5,
    },
    "all_new_families": {
        "archetype": 1.0 / 3.0,
        "preference_table": 1.0 / 3.0,
        "graph_quality": 1.0 / 3.0,
    },
}

candidate_report = {}
best_primary = -np.inf
best_scores = None
best_recipe = None
best_alpha = None
best_raw_component = None

alphas = np.linspace(0.0, 1.0, 11)

for recipe_name, recipe in family_recipes.items():
    new_rank = np.zeros(valid_y.size, dtype=np.float64)
    raw_component = np.zeros(valid_y.size, dtype=np.float64)

    for family_name, coefficient in recipe.items():
        new_rank += coefficient * valid_rank[family_name]
        raw_component += coefficient * valid_raw[family_name]

    standalone_metric = evaluate(
        valid_users, valid_y, new_rank
    )["primary"]
    candidate_report[recipe_name + "_standalone"] = float(
        standalone_metric
    )

    recipe_best = -np.inf
    for alpha in alphas:
        blended = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * new_rank
        )
        metric = evaluate(
            valid_users, valid_y, blended
        )["primary"]

        if metric > recipe_best:
            recipe_best = metric

        if metric > best_primary:
            best_primary = float(metric)
            best_scores = blended.copy()
            best_recipe = dict(recipe)
            best_alpha = float(alpha)
            best_raw_component = raw_component.copy()

    candidate_report[
        recipe_name + "_best_incumbent_blend"
    ] = float(recipe_best)

print(
    "CANDIDATES "
    + json.dumps(candidate_report, sort_keys=True)
)
print(
    "FINDINGS selected="
    + json.dumps(
        {
            "recipe": best_recipe,
            "incumbent_blend_alpha": best_alpha,
        },
        sort_keys=True,
    )
)

metrics = evaluate(valid_users, valid_y, best_scores)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_component, dtype=np.float64),
    )

# Score test with the exact validation-selected family recipe and blend.
test = load("test")
test_users = np.asarray(test.user_id, dtype=np.int64)

test_raw = {}
if "archetype" in best_recipe:
    test_raw["archetype"] = predict_archetype(
        test, archetype_model
    )
if "preference_table" in best_recipe:
    test_raw["preference_table"] = predict_preference(
        test, preference_model
    )
if "graph_quality" in best_recipe:
    test_raw["graph_quality"] = predict_graph_quality(
        test, graph_model
    )

test_new_rank = np.zeros(test_users.size, dtype=np.float64)
for family_name, coefficient in best_recipe.items():
    test_new_rank += coefficient * within_user_rank(
        test_users, test_raw[family_name]
    )

test_inc_rank = within_user_rank(test_users, inc_test)
test_scores = (
    (1.0 - best_alpha) * test_inc_rank
    + best_alpha * test_new_rank
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