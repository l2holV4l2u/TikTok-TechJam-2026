import os
import time
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
np.random.seed(2026)

GLOBAL_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "onehot_feat3",
    "onehot_feat8",
    "upload_type",
    "music_type",
    "tab",
]
GLOBAL_COEFFICIENTS = {
    "video_id": 2.00,
    "author_id": 1.50,
    "tag": 0.65,
    "duration_bucket": 0.80,
    "onehot_feat3": 0.65,
    "onehot_feat8": 0.45,
    "upload_type": 0.35,
    "music_type": 0.25,
    "tab": 0.30,
}
GLOBAL_SMOOTHING = {
    "video_id": 24.0,
    "author_id": 32.0,
    "tag": 90.0,
    "duration_bucket": 100.0,
    "onehot_feat3": 55.0,
    "onehot_feat8": 65.0,
    "upload_type": 100.0,
    "music_type": 120.0,
    "tab": 140.0,
}
AFFINITY_FIELDS = [
    ("author_id", 1.25, 10.0),
    ("tag", 0.85, 13.0),
    ("duration_bucket", 0.65, 15.0),
    ("onehot_feat3", 0.70, 11.0),
    ("onehot_feat8", 0.45, 14.0),
]


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    su = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = su[1:] != su[:-1]
    group_start = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = su[:-1] != su[1:]
    end_positions = np.flatnonzero(ends)
    group_sizes = np.diff(
        np.concatenate((np.asarray([-1], dtype=np.int64), end_positions))
    )
    repeated_sizes = np.repeat(group_sizes, group_sizes)
    positions = np.arange(n, dtype=np.int64) - group_start

    sorted_result = (positions.astype(np.float64) + 0.5) / repeated_sizes
    result = np.empty(n, dtype=np.float64)
    result[order] = sorted_result
    return result


def sigmoid_logit(rate):
    rate = np.clip(np.asarray(rate, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(rate) - np.log1p(-rate)


def temporal_weights(dates, half_life):
    if half_life is None:
        return np.ones(len(dates), dtype=np.float64)
    dates = np.asarray(dates, dtype=np.int32)
    ages = int(dates.max()) - dates
    weights = np.power(0.5, ages.astype(np.float64) / float(half_life))
    weights /= max(weights.mean(), 1e-12)
    return weights


def categorical_rate(train_ids, query_ids, y, weights, cardinality,
                     prior, smoothing):
    train_ids = np.asarray(train_ids, dtype=np.int64)
    query_ids = np.asarray(query_ids, dtype=np.int64)
    weighted_count = np.bincount(
        train_ids, weights=weights, minlength=cardinality
    ).astype(np.float64)
    weighted_positive = np.bincount(
        train_ids, weights=weights * y, minlength=cardinality
    ).astype(np.float64)
    rates = (
        weighted_positive + smoothing * prior
    ) / (weighted_count + smoothing)
    safe_query = np.clip(query_ids, 0, cardinality - 1)
    return rates, rates[safe_query]


def sparse_cross_rate(train_user, train_value, query_user, query_value,
                      value_cardinality, y, weights, prior_train,
                      prior_query, smoothing):
    train_key = (
        np.asarray(train_user, dtype=np.int64) * int(value_cardinality)
        + np.asarray(train_value, dtype=np.int64)
    )
    query_key = (
        np.asarray(query_user, dtype=np.int64) * int(value_cardinality)
        + np.asarray(query_value, dtype=np.int64)
    )

    unique_keys, inverse = np.unique(train_key, return_inverse=True)
    counts = np.bincount(inverse, weights=weights).astype(np.float64)
    positives = np.bincount(
        inverse, weights=weights * y
    ).astype(np.float64)
    prior_sums = np.bincount(
        inverse, weights=weights * prior_train
    ).astype(np.float64)
    group_prior = prior_sums / np.maximum(counts, 1e-12)
    rates = (
        positives + smoothing * group_prior
    ) / (counts + smoothing)

    positions = np.searchsorted(unique_keys, query_key)
    found = positions < len(unique_keys)
    matched = np.zeros(len(query_key), dtype=bool)
    matched[found] = unique_keys[positions[found]] == query_key[found]

    result = np.asarray(prior_query, dtype=np.float64).copy()
    result[matched] = rates[positions[matched]]
    return result


def empirical_bayes_predictions(train, valid, test, half_life,
                                personalized=False):
    y = np.asarray(train.y, dtype=np.float64)
    weights = temporal_weights(train.date, half_life)
    prior = float(np.sum(weights * y) / np.sum(weights))

    valid_score = np.zeros(len(valid.user_id), dtype=np.float64)
    test_score = np.zeros(len(test.user_id), dtype=np.float64)

    train_rates = {}
    valid_rates = {}
    test_rates = {}

    coefficient_sum = 0.0
    for field in GLOBAL_FIELDS:
        cardinality = FEATURE_CARDINALITIES[field]
        full_rates, va_rate = categorical_rate(
            train.X[field],
            valid.X[field],
            y,
            weights,
            cardinality,
            prior,
            GLOBAL_SMOOTHING[field],
        )
        te_ids = np.clip(
            np.asarray(test.X[field], dtype=np.int64), 0, cardinality - 1
        )
        tr_ids = np.clip(
            np.asarray(train.X[field], dtype=np.int64), 0, cardinality - 1
        )
        te_rate = full_rates[te_ids]
        tr_rate = full_rates[tr_ids]

        train_rates[field] = tr_rate
        valid_rates[field] = va_rate
        test_rates[field] = te_rate

        coefficient = GLOBAL_COEFFICIENTS[field]
        coefficient_sum += coefficient
        valid_score += coefficient * sigmoid_logit(va_rate)
        test_score += coefficient * sigmoid_logit(te_rate)

    valid_score /= coefficient_sum
    test_score /= coefficient_sum

    if personalized:
        for field, coefficient, smoothing in AFFINITY_FIELDS:
            cardinality = FEATURE_CARDINALITIES[field]
            va_cross = sparse_cross_rate(
                train.X["user_id"],
                train.X[field],
                valid.X["user_id"],
                valid.X[field],
                cardinality,
                y,
                weights,
                train_rates[field],
                valid_rates[field],
                smoothing,
            )
            te_cross = sparse_cross_rate(
                train.X["user_id"],
                train.X[field],
                test.X["user_id"],
                test.X[field],
                cardinality,
                y,
                weights,
                train_rates[field],
                test_rates[field],
                smoothing,
            )
            valid_score += coefficient * (
                sigmoid_logit(va_cross)
                - sigmoid_logit(valid_rates[field])
            )
            test_score += coefficient * (
                sigmoid_logit(te_cross)
                - sigmoid_logit(test_rates[field])
            )

    return valid_score, test_score


def spectral_predictions(train, valid, test, rank=40):
    user_count = FEATURE_CARDINALITIES["user_id"]
    video_count = FEATURE_CARDINALITIES["video_id"]
    author_count = FEATURE_CARDINALITIES["author_id"]

    user = np.asarray(train.X["user_id"], dtype=np.int64)
    video = np.asarray(train.X["video_id"], dtype=np.int64)
    author = np.asarray(train.X["author_id"], dtype=np.int64)
    y = np.asarray(train.y, dtype=np.float64)

    # Positive-only implicit matrices, normalized to suppress globally dominant
    # users/entities and make the factors represent preference rather than volume.
    positive = y > 0.5
    pu = user[positive]
    pv = video[positive]
    pa = author[positive]

    def factorize(row_ids, col_ids, n_cols, local_rank):
        matrix = sparse.coo_matrix(
            (
                np.ones(len(row_ids), dtype=np.float64),
                (row_ids, col_ids),
            ),
            shape=(user_count, n_cols),
        ).tocsr()
        matrix.sum_duplicates()
        matrix.data[:] = np.log1p(matrix.data)

        row_degree = np.asarray(matrix.sum(axis=1)).ravel()
        col_degree = np.asarray(matrix.sum(axis=0)).ravel()
        row_scale = 1.0 / np.sqrt(np.maximum(row_degree, 1.0))
        col_scale = 1.0 / np.sqrt(np.maximum(col_degree, 1.0))
        normalized = sparse.diags(row_scale) @ matrix @ sparse.diags(col_scale)

        u, singular, vt = svds(
            normalized,
            k=local_rank,
            which="LM",
            return_singular_vectors=True,
            random_state=2026,
        )
        order = np.argsort(singular)[::-1]
        singular = singular[order]
        u = u[:, order]
        vt = vt[order]
        user_factors = u * np.sqrt(singular)[None, :]
        entity_factors = vt.T * np.sqrt(singular)[None, :]
        return user_factors, entity_factors

    uv, vv = factorize(pu, pv, video_count, rank)
    ua, av = factorize(pu, pa, author_count, min(28, rank))

    def score(split):
        su = np.clip(
            np.asarray(split.X["user_id"], dtype=np.int64), 0, user_count - 1
        )
        sv = np.clip(
            np.asarray(split.X["video_id"], dtype=np.int64), 0, video_count - 1
        )
        sa = np.clip(
            np.asarray(split.X["author_id"], dtype=np.int64), 0, author_count - 1
        )
        video_score = np.einsum("ij,ij->i", uv[su], vv[sv])
        author_score = np.einsum("ij,ij->i", ua[su], av[sa])
        return video_score + 0.65 * author_score

    return score(valid), score(test)


train = load("train")
valid = load("valid")
test = load("test")

raw_valid = {}
raw_test = {}

# Different drift assumptions for a purely non-parametric content model.
raw_valid["eb_uniform"], raw_test["eb_uniform"] = (
    empirical_bayes_predictions(
        train, valid, test, half_life=None, personalized=False
    )
)
raw_valid["eb_recent3"], raw_test["eb_recent3"] = (
    empirical_bayes_predictions(
        train, valid, test, half_life=3.0, personalized=False
    )
)
raw_valid["eb_recent7"], raw_test["eb_recent7"] = (
    empirical_bayes_predictions(
        train, valid, test, half_life=7.0, personalized=False
    )
)

# Personalized train-only user-content tables form predictions differently
# from global entity popularity and are heavily shrunk for sparse pairs.
raw_valid["affinity_recent3"], raw_test["affinity_recent3"] = (
    empirical_bayes_predictions(
        train, valid, test, half_life=3.0, personalized=True
    )
)
raw_valid["affinity_recent7"], raw_test["affinity_recent7"] = (
    empirical_bayes_predictions(
        train, valid, test, half_life=7.0, personalized=True
    )
)

# Spectral collaborative preference is independent of target-rate tables.
raw_valid["spectral"], raw_test["spectral"] = spectral_predictions(
    train, valid, test, rank=40
)

# A fixed rank ensemble checks whether collaborative factors complement
# temporally robust personalized content affinity.
raw_valid["affinity_spectral"] = (
    0.72 * rank_percentile(
        valid.user_id, raw_valid["affinity_recent3"]
    )
    + 0.28 * rank_percentile(valid.user_id, raw_valid["spectral"])
)
raw_test["affinity_spectral"] = (
    0.72 * rank_percentile(
        test.user_id, raw_test["affinity_recent3"]
    )
    + 0.28 * rank_percentile(test.user_id, raw_test["spectral"])
)

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)
inc_valid_rank = rank_percentile(valid.user_id, inc_valid)
inc_test_rank = rank_percentile(test.user_id, inc_test)

candidate_scores = {}
candidate_test_scores = {}
candidate_metrics = {}
candidate_source = {}
candidate_is_blend = {}

for name in raw_valid:
    candidate_scores[name] = raw_valid[name]
    candidate_test_scores[name] = raw_test[name]
    candidate_metrics[name] = evaluate(
        valid.user_id, valid.y, raw_valid[name]
    )
    candidate_source[name] = name
    candidate_is_blend[name] = False

    va_rank = rank_percentile(valid.user_id, raw_valid[name])
    te_rank = rank_percentile(test.user_id, raw_test[name])
    for alpha in (0.10, 0.20, 0.35, 0.50):
        key = f"{name}_incblend_{alpha:.2f}"
        candidate_scores[key] = (
            alpha * va_rank + (1.0 - alpha) * inc_valid_rank
        )
        candidate_test_scores[key] = (
            alpha * te_rank + (1.0 - alpha) * inc_test_rank
        )
        candidate_metrics[key] = evaluate(
            valid.user_id, valid.y, candidate_scores[key]
        )
        candidate_source[key] = name
        candidate_is_blend[key] = True

best_key = max(
    candidate_metrics,
    key=lambda key: float(candidate_metrics[key]["primary"]),
)
best_metrics = candidate_metrics[best_key]
best_valid = candidate_scores[best_key]
best_test = candidate_test_scores[best_key]
best_source = candidate_source[best_key]

summary = {
    key: float(value["primary"])
    for key, value in candidate_metrics.items()
}
print("CANDIDATES " + json.dumps(summary, sort_keys=True))
print(
    "FINDINGS " + json.dumps(
        {
            "best_candidate": best_key,
            "best_raw_family": best_source,
            "raw_primaries": {
                name: float(candidate_metrics[name]["primary"])
                for name in raw_valid
            },
            "spectral_rank": 40,
            "empirical_bayes_half_lives": ["uniform", 3.0, 7.0],
        },
        sort_keys=True,
    )
)

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
    if candidate_is_blend[best_key]:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_valid[best_source], dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS " + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)