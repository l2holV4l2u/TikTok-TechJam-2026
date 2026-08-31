import os
import time
import json
import gc
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 19427
np.random.seed(SEED)

PAIR_FIELDS = ["video_id", "author_id", "tag", "duration_bucket"]
PAIR_WEIGHTS = {
    "video_id": 0.38,
    "author_id": 0.30,
    "tag": 0.20,
    "duration_bucket": 0.12,
}
ENTITY_SMOOTH = {
    "video_id": 18.0,
    "author_id": 22.0,
    "tag": 45.0,
    "duration_bucket": 90.0,
}
PAIR_SMOOTH = {
    "video_id": 3.0,
    "author_id": 5.0,
    "tag": 8.0,
    "duration_bucket": 12.0,
}


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]

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

    position = np.arange(n, dtype=np.int64) - first
    ranked = (position.astype(np.float64) + 0.5) / row_sizes
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def date_weights(dates, half_life, reference_date=None):
    dates = np.asarray(dates, dtype=np.int32)
    if reference_date is None:
        reference_date = int(dates.max())
    age = np.maximum(reference_date - dates, 0).astype(np.float64)
    if half_life is None:
        return np.ones(len(dates), dtype=np.float64)
    return np.power(0.5, age / float(half_life))


def fit_rate_map(keys, labels, weights, prior, smoothing):
    keys = np.asarray(keys, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    denominator = np.bincount(inverse, weights=weights)
    numerator = np.bincount(inverse, weights=weights * labels)
    rates = (numerator + smoothing * prior) / (denominator + smoothing)
    return unique_keys, rates.astype(np.float64), denominator.astype(np.float64)


def lookup_rate(query_keys, unique_keys, values, default):
    query_keys = np.asarray(query_keys, dtype=np.int64)
    positions = np.searchsorted(unique_keys, query_keys)
    output = np.full(len(query_keys), default, dtype=np.float64)
    valid = positions < len(unique_keys)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices):
        matched = (
            unique_keys[positions[valid_indices]]
            == query_keys[valid_indices]
        )
        use = valid_indices[matched]
        output[use] = values[positions[use]]
    return output


def joint_key(user, entity, entity_cardinality):
    user = np.asarray(user, dtype=np.int64)
    entity = np.asarray(entity, dtype=np.int64)
    return user * np.int64(entity_cardinality) + entity


def fit_temporal_statistics(train, mask, half_life):
    y = np.asarray(train.y, dtype=np.float64)[mask]
    dates = np.asarray(train.date, dtype=np.int32)[mask]
    weights = date_weights(dates, half_life)
    stationary_weights = np.ones(len(y), dtype=np.float64)

    recent_prior = float(np.sum(weights * y) / np.sum(weights))
    stationary_prior = float(y.mean())
    users = np.asarray(train.X["user_id"], dtype=np.int64)[mask]

    model = {
        "recent_prior": recent_prior,
        "stationary_prior": stationary_prior,
        "half_life": half_life,
        "fields": {},
    }

    for field in PAIR_FIELDS:
        entity = np.asarray(train.X[field], dtype=np.int64)[mask]
        card = FEATURE_CARDINALITIES[field]

        er_keys, er_rates, er_counts = fit_rate_map(
            entity, y, weights, recent_prior, ENTITY_SMOOTH[field]
        )
        es_keys, es_rates, es_counts = fit_rate_map(
            entity, y, stationary_weights, stationary_prior,
            ENTITY_SMOOTH[field]
        )

        pairs = joint_key(users, entity, card)
        pr_keys, pr_rates, pr_counts = fit_rate_map(
            pairs, y, weights, recent_prior, PAIR_SMOOTH[field]
        )

        model["fields"][field] = {
            "entity_recent": (er_keys, er_rates, er_counts),
            "entity_stationary": (es_keys, es_rates, es_counts),
            "pair_recent": (pr_keys, pr_rates, pr_counts),
        }

    return model


def predict_temporal_statistics(model, split):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    n = len(split)

    entity_recent = np.zeros(n, dtype=np.float64)
    temporal_residual = np.zeros(n, dtype=np.float64)
    personalized = np.zeros(n, dtype=np.float64)
    total_weight = 0.0

    for field in PAIR_FIELDS:
        field_weight = PAIR_WEIGHTS[field]
        total_weight += field_weight
        entity = np.asarray(split.X[field], dtype=np.int64)
        card = FEATURE_CARDINALITIES[field]
        field_model = model["fields"][field]

        erk, erv, _ = field_model["entity_recent"]
        esk, esv, _ = field_model["entity_stationary"]
        prk, prv, _ = field_model["pair_recent"]

        recent_rate = lookup_rate(
            entity, erk, erv, model["recent_prior"]
        )
        stationary_rate = lookup_rate(
            entity, esk, esv, model["stationary_prior"]
        )
        pair_rate = lookup_rate(
            joint_key(users, entity, card),
            prk,
            prv,
            model["recent_prior"],
        )

        recent_logit = logit(recent_rate)
        stationary_logit = logit(stationary_rate)
        pair_logit = logit(pair_rate)

        entity_recent += field_weight * recent_logit
        temporal_residual += field_weight * (
            recent_logit - stationary_logit
        )

        # Shrink personalized effects toward the corresponding recent entity
        # estimate by averaging in log-odds space.
        personalized += field_weight * (
            0.62 * pair_logit + 0.38 * recent_logit
        )

    entity_recent /= total_weight
    temporal_residual /= total_weight
    personalized /= total_weight

    return {
        "temporal_entity": entity_recent,
        "temporal_residual": temporal_residual,
        "personalized_rates": personalized,
        "personalized_plus_residual": personalized + 0.45 * temporal_residual,
    }


def fit_weighted_svd(train, half_life, rank=40):
    users = np.asarray(train.X["user_id"], dtype=np.int64)
    videos = np.asarray(train.X["video_id"], dtype=np.int64)
    y = np.asarray(train.y, dtype=np.int8)
    weights = date_weights(train.date, half_life)

    positive = y == 1
    rows = users[positive]
    cols = videos[positive]
    data = weights[positive].astype(np.float64)

    n_users = FEATURE_CARDINALITIES["user_id"]
    n_videos = FEATURE_CARDINALITIES["video_id"]
    matrix = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(n_users, n_videos),
        dtype=np.float64,
    ).tocsr()

    # BM25-like normalization reduces domination by highly active users and
    # globally popular videos while retaining repeated positive evidence.
    row_mass = np.asarray(matrix.sum(axis=1)).ravel()
    col_mass = np.asarray(matrix.sum(axis=0)).ravel()
    row_scale = 1.0 / np.sqrt(np.maximum(row_mass, 1.0))
    col_scale = 1.0 / np.sqrt(np.maximum(col_mass, 1.0))
    normalized = sparse.diags(row_scale) @ matrix @ sparse.diags(col_scale)

    actual_rank = min(rank, min(normalized.shape) - 1)
    u, singular, vt = svds(
        normalized,
        k=actual_rank,
        which="LM",
        random_state=SEED,
        return_singular_vectors=True,
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order]
    u = u[:, order]
    vt = vt[order]

    user_factors = u * np.sqrt(singular)[None, :]
    video_factors = vt.T * np.sqrt(singular)[None, :]
    return (
        user_factors.astype(np.float32),
        video_factors.astype(np.float32),
    )


def predict_svd(split, user_factors, video_factors):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    videos = np.asarray(split.X["video_id"], dtype=np.int64)
    return np.einsum(
        "ij,ij->i",
        user_factors[users],
        video_factors[videos],
        optimize=True,
    ).astype(np.float64)


train = load("train")
valid = load("valid")
test = load("test")

# Legal train-only temporal selection: use the final two TRAIN days as the
# holdout, and never use validation labels to choose the decay horizon.
train_dates = np.asarray(train.date, dtype=np.int32)
internal_fit = train_dates <= 20220419
internal_holdout = train_dates >= 20220420

internal_scores = {}
for half_life in (2.0, 4.0, 8.0):
    internal_model = fit_temporal_statistics(
        train, internal_fit, half_life
    )
    internal_predictions = predict_temporal_statistics(
        internal_model,
        type("TrainView", (), {
            "X": {
                key: np.asarray(value)[internal_holdout]
                for key, value in train.X.items()
            },
            "__len__": lambda self: int(np.sum(internal_holdout)),
        })()
    )
    metric = evaluate(
        np.asarray(train.user_id)[internal_holdout],
        np.asarray(train.y)[internal_holdout],
        internal_predictions["personalized_plus_residual"],
    )
    internal_scores[half_life] = float(metric["primary"])
    del internal_model, internal_predictions
    gc.collect()

selected_half_life = max(internal_scores, key=internal_scores.get)

# Refit every statistic on all and only the training split.
temporal_model = fit_temporal_statistics(
    train, np.ones(len(train), dtype=bool), selected_half_life
)
temporal_valid = predict_temporal_statistics(temporal_model, valid)
temporal_test = predict_temporal_statistics(temporal_model, test)

# Structurally different latent implicit-feedback model.
try:
    user_factors, video_factors = fit_weighted_svd(
        train, selected_half_life, rank=40
    )
    svd_valid = predict_svd(valid, user_factors, video_factors)
    svd_test = predict_svd(test, user_factors, video_factors)
    svd_ok = True
except Exception as exc:
    svd_valid = np.zeros(len(valid), dtype=np.float64)
    svd_test = np.zeros(len(test), dtype=np.float64)
    svd_ok = False
    print("FINDINGS " + json.dumps({
        "svd_failure": repr(exc)
    }, sort_keys=True))

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

raw_valid = {
    **temporal_valid,
    "weighted_implicit_svd": svd_valid,
}
raw_test = {
    **temporal_test,
    "weighted_implicit_svd": svd_test,
}

valid_ranks = {
    name: rank_percentile(valid.user_id, values)
    for name, values in raw_valid.items()
}
test_ranks = {
    name: rank_percentile(test.user_id, values)
    for name, values in raw_test.items()
}

candidate_valid = {}
candidate_test = {}
candidate_raw = {}
candidate_metrics = {}

# Standalone structurally different predictors.
for name in raw_valid:
    key = name + "_standalone"
    candidate_valid[key] = raw_valid[name]
    candidate_test[key] = raw_test[name]
    candidate_raw[key] = raw_valid[name]

# Blend every family with the trusted incumbent as requested.
for name in raw_valid:
    for alpha in (0.10, 0.20, 0.35, 0.50):
        key = f"{name}_incblend_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * valid_ranks[name]
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank
            + alpha * test_ranks[name]
        )
        candidate_raw[key] = raw_valid[name]

# Cross-family ensembles test whether latent affinity and temporal residuals
# repair different incumbent errors.
ensemble_specs = [
    ("rates_svd_equal", {
        "personalized_plus_residual": 0.50,
        "weighted_implicit_svd": 0.50,
    }),
    ("entity_pair_svd", {
        "temporal_entity": 0.20,
        "personalized_rates": 0.50,
        "weighted_implicit_svd": 0.30,
    }),
    ("residual_pair_svd", {
        "temporal_residual": 0.25,
        "personalized_rates": 0.50,
        "weighted_implicit_svd": 0.25,
    }),
]

for ensemble_name, components in ensemble_specs:
    ensemble_valid = np.zeros(len(valid), dtype=np.float64)
    ensemble_test = np.zeros(len(test), dtype=np.float64)
    for name, weight in components.items():
        ensemble_valid += weight * valid_ranks[name]
        ensemble_test += weight * test_ranks[name]

    standalone_key = ensemble_name + "_standalone"
    candidate_valid[standalone_key] = ensemble_valid
    candidate_test[standalone_key] = ensemble_test
    candidate_raw[standalone_key] = ensemble_valid

    for alpha in (0.10, 0.20, 0.35, 0.50):
        key = f"{ensemble_name}_incblend_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * ensemble_valid
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank
            + alpha * ensemble_test
        )
        candidate_raw[key] = ensemble_valid

# Include the unchanged incumbent so an unhelpful broad family search cannot
# destroy the best saved iteration.
candidate_valid["incumbent"] = inc_valid
candidate_test["incumbent"] = inc_test
candidate_raw["incumbent"] = inc_valid

for key, scores in candidate_valid.items():
    candidate_metrics[key] = evaluate(
        valid.user_id, valid.y, scores
    )

best_key = max(
    candidate_metrics,
    key=lambda key: float(candidate_metrics[key]["primary"]),
)
best_metrics = candidate_metrics[best_key]
best_valid = candidate_valid[best_key]
best_test = candidate_test[best_key]

summary = {
    key: float(value["primary"])
    for key, value in candidate_metrics.items()
}
print("CANDIDATES " + json.dumps(summary, sort_keys=True))
print("FINDINGS " + json.dumps({
    "best_candidate": best_key,
    "internal_half_life_scores": {
        str(k): v for k, v in internal_scores.items()
    },
    "selected_half_life_days": float(selected_half_life),
    "svd_ok": bool(svd_ok),
    "valid_rank_correlations_with_incumbent": {
        name: float(np.corrcoef(inc_valid_rank, ranks)[0, 1])
        for name, ranks in valid_ranks.items()
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
    if best_key != "incumbent":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[best_key], dtype=np.float64),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))