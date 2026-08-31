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
SEED = 28417
RNG = np.random.default_rng(SEED)

LATENT_RANK = 64
COLLECTIVE_FIELDS = ["video_id", "author_id", "tag", "duration_bucket"]
NB_FIELDS = [
    "video_id", "author_id", "tag", "duration_bucket", "upload_type",
    "music_type", "tab", "hour", "onehot_feat3", "onehot_feat8",
]


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates, dtype=np.int32)
    age = np.maximum(int(dates.max()) - dates, 0).astype(np.float64)
    return np.power(0.5, age / half_life).astype(np.float32)


def rank_percentile(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    first = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(np.concatenate((np.array([-1]), end_positions)))
    row_sizes = np.repeat(sizes, sizes)

    positions = np.arange(n, dtype=np.int64) - first
    ranked = (positions.astype(np.float64) + 0.5) / row_sizes

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def row_l2_normalize(matrix):
    matrix = matrix.tocsr().astype(np.float32)
    squared = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    scale = np.zeros_like(squared, dtype=np.float32)
    nonzero = squared > 0
    scale[nonzero] = 1.0 / np.sqrt(squared[nonzero])
    return sp.diags(scale).dot(matrix).tocsr()


def spectral_decompose(matrix, rank):
    rank = min(rank, min(matrix.shape) - 1)
    u, singular, vt = svds(
        matrix.astype(np.float32),
        k=rank,
        which="LM",
        return_singular_vectors=True,
        random_state=SEED,
        tol=2e-3,
        maxiter=700,
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order].astype(np.float32)
    u = u[:, order].astype(np.float32)
    vt = vt[order].astype(np.float32)
    user_factors = u * singular[None, :]
    return user_factors, vt


def paired_factor_score(user_factors, entity_factors, users, entities, chunk=300000):
    users = np.asarray(users, dtype=np.int64)
    entities = np.asarray(entities, dtype=np.int64)
    result = np.zeros(len(users), dtype=np.float32)
    valid = (
        (users >= 0) & (users < user_factors.shape[0]) &
        (entities >= 0) & (entities < entity_factors.shape[1])
    )
    valid_indices = np.flatnonzero(valid)
    for begin in range(0, len(valid_indices), chunk):
        idx = valid_indices[begin:begin + chunk]
        result[idx] = np.einsum(
            "ij,ji->i",
            user_factors[users[idx]],
            entity_factors[:, entities[idx]],
            optimize=True,
        )
    return result.astype(np.float64)


def build_implicit_video_svd(train, weights):
    users = np.asarray(train.X["user_id"], dtype=np.int64)
    videos = np.asarray(train.X["video_id"], dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.float32)

    n_users = int(FEATURE_CARDINALITIES["user_id"])
    n_videos = int(FEATURE_CARDINALITIES["video_id"])

    values = weights * labels
    matrix = sp.coo_matrix(
        (values, (users, videos)),
        shape=(n_users, n_videos),
        dtype=np.float32,
    ).tocsr()
    matrix.eliminate_zeros()

    # Damp globally frequent videos so the factors emphasize personalized
    # preference rather than merely reconstructing exposure popularity.
    document_frequency = np.asarray((matrix != 0).sum(axis=0)).ravel()
    idf = np.log1p(n_users / (1.0 + document_frequency)).astype(np.float32)
    matrix = matrix.dot(sp.diags(idf))
    matrix = row_l2_normalize(matrix)

    return spectral_decompose(matrix, LATENT_RANK)


def build_collective_signed_svd(train, weights):
    users = np.asarray(train.X["user_id"], dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.float32)
    n_users = int(FEATURE_CARDINALITIES["user_id"])

    prior = float(np.sum(weights * labels) / np.sum(weights))
    residual = weights * (labels - prior)

    blocks = []
    offsets = {}
    offset = 0
    for field in COLLECTIVE_FIELDS:
        entities = np.asarray(train.X[field], dtype=np.int64)
        cardinality = int(FEATURE_CARDINALITIES[field])
        offsets[field] = (offset, cardinality)

        block = sp.coo_matrix(
            (residual, (users, entities)),
            shape=(n_users, cardinality),
            dtype=np.float32,
        ).tocsr()
        block.eliminate_zeros()

        # Normalize each view independently so video_id cannot swamp the
        # lower-cardinality semantic views solely through its scale.
        block = row_l2_normalize(block)
        blocks.append(block * np.float32(1.0 / np.sqrt(len(COLLECTIVE_FIELDS))))
        offset += cardinality

    collective = sp.hstack(blocks, format="csr", dtype=np.float32)
    collective = row_l2_normalize(collective)
    user_factors, entity_factors = spectral_decompose(
        collective, LATENT_RANK
    )
    return user_factors, entity_factors, offsets


def predict_collective(split, user_factors, entity_factors, offsets):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    total = np.zeros(len(split), dtype=np.float64)

    for field in COLLECTIVE_FIELDS:
        offset, cardinality = offsets[field]
        entities = np.asarray(split.X[field], dtype=np.int64)
        shifted = offset + np.clip(entities, 0, cardinality - 1)
        total += paired_factor_score(
            user_factors, entity_factors, users, shifted
        )

    return total / np.sqrt(float(len(COLLECTIVE_FIELDS)))


def fit_naive_bayes(train, weights, smoothing=120.0):
    labels = np.asarray(train.y, dtype=np.float64)
    weights64 = np.asarray(weights, dtype=np.float64)
    positive_total = float(np.sum(weights64 * labels))
    negative_total = float(np.sum(weights64 * (1.0 - labels)))

    model = {}
    for field in NB_FIELDS:
        values = np.asarray(train.X[field], dtype=np.int64)
        cardinality = int(FEATURE_CARDINALITIES[field])

        total_count = np.bincount(
            values, weights=weights64, minlength=cardinality
        ).astype(np.float64)
        positive_count = np.bincount(
            values, weights=weights64 * labels, minlength=cardinality
        ).astype(np.float64)
        negative_count = total_count - positive_count

        frequency_prior = (total_count + 1.0) / (
            np.sum(total_count) + cardinality
        )
        positive_probability = (
            positive_count + smoothing * frequency_prior
        ) / (positive_total + smoothing)
        negative_probability = (
            negative_count + smoothing * frequency_prior
        ) / (negative_total + smoothing)

        log_ratio = np.log(
            np.maximum(positive_probability, 1e-12) /
            np.maximum(negative_probability, 1e-12)
        )
        model[field] = log_ratio.astype(np.float32)
    return model


def predict_naive_bayes(split, model):
    score = np.zeros(len(split), dtype=np.float64)
    for field, table in model.items():
        values = np.asarray(split.X[field], dtype=np.int64)
        valid = (values >= 0) & (values < len(table))
        score[valid] += table[values[valid]]
    return score / np.sqrt(float(len(model)))


train = load("train")
valid = load("valid")
test = load("test")

weights = recency_weights(train.date, half_life=4.0)

# Family 1: positive-only implicit PureSVD.
implicit_users, implicit_videos = build_implicit_video_svd(train, weights)
implicit_valid = paired_factor_score(
    implicit_users,
    implicit_videos,
    np.asarray(valid.X["user_id"], dtype=np.int64),
    np.asarray(valid.X["video_id"], dtype=np.int64),
)
implicit_test = paired_factor_score(
    implicit_users,
    implicit_videos,
    np.asarray(test.X["user_id"], dtype=np.int64),
    np.asarray(test.X["video_id"], dtype=np.int64),
)
del implicit_users, implicit_videos
gc.collect()

# Family 2: signed multi-view collective factorization.
collective_users, collective_entities, collective_offsets = (
    build_collective_signed_svd(train, weights)
)
collective_valid = predict_collective(
    valid, collective_users, collective_entities, collective_offsets
)
collective_test = predict_collective(
    test, collective_users, collective_entities, collective_offsets
)
del collective_users, collective_entities
gc.collect()

# Family 3: generative categorical evidence aggregation.
nb_model = fit_naive_bayes(train, weights)
nb_valid = predict_naive_bayes(valid, nb_model)
nb_test = predict_naive_bayes(test, nb_model)

del train, nb_model, weights
gc.collect()

family_valid = {
    "implicit_puresvd": implicit_valid,
    "signed_collective_svd": collective_valid,
    "categorical_naive_bayes": nb_valid,
}
family_test = {
    "implicit_puresvd": implicit_test,
    "signed_collective_svd": collective_test,
    "categorical_naive_bayes": nb_test,
}

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = rank_percentile(valid.user_id, inc_valid)
inc_test_rank = rank_percentile(test.user_id, inc_test)

family_valid_rank = {
    name: rank_percentile(valid.user_id, score)
    for name, score in family_valid.items()
}
family_test_rank = {
    name: rank_percentile(test.user_id, score)
    for name, score in family_test.items()
}

candidate_valid = {"incumbent": inc_valid}
candidate_test = {"incumbent": inc_test}
candidate_raw = {"incumbent": inc_valid}

for name in family_valid:
    candidate_valid[name + "_standalone"] = family_valid[name]
    candidate_test[name + "_standalone"] = family_test[name]
    candidate_raw[name + "_standalone"] = family_valid[name]

    for alpha in (0.05, 0.10, 0.20, 0.35, 0.50, 0.65):
        key = f"{name}_incblend_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank +
            alpha * family_valid_rank[name]
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank +
            alpha * family_test_rank[name]
        )
        candidate_raw[key] = family_valid[name]

# Spectral views can complement one another: positive SVD models affinity,
# whereas signed collective SVD models above/below-prior preference.
for collective_weight in (0.25, 0.50, 0.75):
    spectral_name = f"spectral_hybrid_{collective_weight:.2f}"
    spectral_valid = (
        (1.0 - collective_weight) *
        family_valid_rank["implicit_puresvd"] +
        collective_weight *
        family_valid_rank["signed_collective_svd"]
    )
    spectral_test = (
        (1.0 - collective_weight) *
        family_test_rank["implicit_puresvd"] +
        collective_weight *
        family_test_rank["signed_collective_svd"]
    )

    candidate_valid[spectral_name + "_standalone"] = spectral_valid
    candidate_test[spectral_name + "_standalone"] = spectral_test
    candidate_raw[spectral_name + "_standalone"] = spectral_valid

    for alpha in (0.10, 0.20, 0.35, 0.50):
        key = f"{spectral_name}_incblend_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank + alpha * spectral_valid
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank + alpha * spectral_test
        )
        candidate_raw[key] = spectral_valid

# Three-way composition checks whether metadata evidence repairs cold/sparse
# cases where neither latent view has enough interaction support.
threeway_valid = (
    0.42 * family_valid_rank["implicit_puresvd"] +
    0.42 * family_valid_rank["signed_collective_svd"] +
    0.16 * family_valid_rank["categorical_naive_bayes"]
)
threeway_test = (
    0.42 * family_test_rank["implicit_puresvd"] +
    0.42 * family_test_rank["signed_collective_svd"] +
    0.16 * family_test_rank["categorical_naive_bayes"]
)
candidate_valid["three_family_standalone"] = threeway_valid
candidate_test["three_family_standalone"] = threeway_test
candidate_raw["three_family_standalone"] = threeway_valid

for alpha in (0.10, 0.20, 0.35, 0.50):
    key = f"three_family_incblend_{alpha:.2f}"
    candidate_valid[key] = (
        (1.0 - alpha) * inc_valid_rank + alpha * threeway_valid
    )
    candidate_test[key] = (
        (1.0 - alpha) * inc_test_rank + alpha * threeway_test
    )
    candidate_raw[key] = threeway_valid

candidate_metrics = {
    name: evaluate(valid.user_id, valid.y, scores)
    for name, scores in candidate_valid.items()
}
best_name = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"])
)
best_metrics = candidate_metrics[best_name]
best_valid = candidate_valid[best_name]
best_test = candidate_test[best_name]

print("CANDIDATES " + json.dumps({
    name: float(metrics["primary"])
    for name, metrics in candidate_metrics.items()
}, sort_keys=True))

print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "family_rank_correlations_with_incumbent": {
        name: float(np.corrcoef(inc_valid_rank, ranks)[0, 1])
        for name, ranks in family_valid_rank.items()
    },
    "family_pairwise_rank_correlations": {
        "implicit_collective": float(np.corrcoef(
            family_valid_rank["implicit_puresvd"],
            family_valid_rank["signed_collective_svd"]
        )[0, 1]),
        "implicit_naive_bayes": float(np.corrcoef(
            family_valid_rank["implicit_puresvd"],
            family_valid_rank["categorical_naive_bayes"]
        )[0, 1]),
        "collective_naive_bayes": float(np.corrcoef(
            family_valid_rank["signed_collective_svd"],
            family_valid_rank["categorical_naive_bayes"]
        )[0, 1]),
    },
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    if best_name != "incumbent":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[best_name], dtype=np.float64),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))