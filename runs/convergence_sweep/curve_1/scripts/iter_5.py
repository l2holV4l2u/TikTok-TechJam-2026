import os
import time
import json
import random
import gc

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
random.seed(SEED)
np.random.seed(SEED)

PAIR_SMOOTHING = {
    "video_id": 10.0,
    "author_id": 16.0,
    "tag": 22.0,
}
PAIR_COEFFICIENTS = {
    "video_id": 0.75,
    "author_id": 0.48,
    "tag": 0.28,
}
BASE_COEFFICIENTS = {
    "video_id": 0.60,
    "author_id": 0.30,
    "tag": 0.10,
}
BLEND_WEIGHTS = [0.10, 0.20, 0.35, 0.50, 0.70]


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    positions = np.searchsorted(unique_dates, dates)
    ages = unique_dates.size - 1 - positions
    weights = np.exp2(-ages.astype(np.float32) / np.float32(half_life))
    weights /= weights.mean()
    return weights.astype(np.float64)


def clipped_logit(probability):
    probability = np.clip(
        np.asarray(probability, dtype=np.float64),
        1e-5,
        1.0 - 1e-5,
    )
    return np.log(probability / (1.0 - probability))


def fit_entity_rate(ids, labels, weights, cardinality, smoothing=30.0):
    ids = np.asarray(ids, dtype=np.int64)
    counts = np.bincount(
        ids,
        weights=weights,
        minlength=cardinality,
    ).astype(np.float64)
    positives = np.bincount(
        ids,
        weights=weights * labels,
        minlength=cardinality,
    ).astype(np.float64)

    global_rate = float(np.sum(weights * labels) / np.sum(weights))
    rates = (
        positives + smoothing * global_rate
    ) / (
        counts + smoothing
    )
    return rates, counts, global_rate


def fit_pair_deviation(
    user_ids,
    entity_ids,
    labels,
    weights,
    entity_cardinality,
    entity_rates,
    smoothing,
):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    entity_ids = np.asarray(entity_ids, dtype=np.int64)
    keys = user_ids * np.int64(entity_cardinality) + entity_ids

    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
    sorted_weights = weights[order]
    sorted_positive_weights = sorted_weights * labels[order]

    starts = np.r_[0, np.flatnonzero(
        sorted_keys[1:] != sorted_keys[:-1]
    ) + 1]
    unique_keys = sorted_keys[starts]
    counts = np.add.reduceat(sorted_weights, starts)
    positives = np.add.reduceat(sorted_positive_weights, starts)

    unique_entities = (
        unique_keys % np.int64(entity_cardinality)
    ).astype(np.int64)
    priors = entity_rates[unique_entities]
    rates = (
        positives + smoothing * priors
    ) / (
        counts + smoothing
    )
    deviations = clipped_logit(rates) - clipped_logit(priors)

    return (
        unique_keys.astype(np.int64),
        deviations.astype(np.float64),
        counts.astype(np.float64),
    )


def lookup_sorted(keys, values, query_keys):
    query_keys = np.asarray(query_keys, dtype=np.int64)
    positions = np.searchsorted(keys, query_keys)
    output = np.zeros(query_keys.size, dtype=np.float64)

    valid = positions < keys.size
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size:
        matched = keys[positions[valid_indices]] == query_keys[valid_indices]
        matched_indices = valid_indices[matched]
        output[matched_indices] = values[positions[matched_indices]]
    return output


def build_previous_entity(split, entity_name, initial_last=None):
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    entities = np.asarray(split.X[entity_name], dtype=np.int64)
    rows = np.arange(users.size, dtype=np.int64)

    order = np.lexsort((rows, times, users))
    ordered_users = users[order]
    ordered_entities = entities[order]

    previous_ordered = np.zeros(users.size, dtype=np.int64)
    same_user = ordered_users[1:] == ordered_users[:-1]
    previous_ordered[1:][same_user] = ordered_entities[:-1][same_user]

    if initial_last is not None:
        first = np.r_[True, ordered_users[1:] != ordered_users[:-1]]
        first_positions = np.flatnonzero(first)
        previous_ordered[first_positions] = initial_last[
            ordered_users[first_positions]
        ]

    previous = np.empty(users.size, dtype=np.int64)
    previous[order] = previous_ordered
    return previous


def last_entity_by_user(split, entity_name, user_cardinality):
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    entities = np.asarray(split.X[entity_name], dtype=np.int64)
    rows = np.arange(users.size, dtype=np.int64)

    order = np.lexsort((rows, times, users))
    ordered_users = users[order]
    last_positions = np.r_[
        ordered_users[1:] != ordered_users[:-1],
        True,
    ]
    result = np.zeros(user_cardinality, dtype=np.int64)
    selected = order[last_positions]
    result[users[selected]] = entities[selected]
    return result


def fit_transition_deviation(
    previous_ids,
    current_ids,
    labels,
    weights,
    entity_cardinality,
    entity_rates,
    smoothing=28.0,
):
    previous_ids = np.asarray(previous_ids, dtype=np.int64)
    current_ids = np.asarray(current_ids, dtype=np.int64)

    usable = previous_ids != 0
    keys = (
        previous_ids[usable] * np.int64(entity_cardinality)
        + current_ids[usable]
    )
    local_weights = weights[usable]
    local_labels = labels[usable]

    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
    sorted_weights = local_weights[order]
    sorted_positive_weights = sorted_weights * local_labels[order]

    starts = np.r_[0, np.flatnonzero(
        sorted_keys[1:] != sorted_keys[:-1]
    ) + 1]
    unique_keys = sorted_keys[starts]
    counts = np.add.reduceat(sorted_weights, starts)
    positives = np.add.reduceat(sorted_positive_weights, starts)

    current_for_key = (
        unique_keys % np.int64(entity_cardinality)
    ).astype(np.int64)
    priors = entity_rates[current_for_key]
    rates = (
        positives + smoothing * priors
    ) / (
        counts + smoothing
    )
    deviations = clipped_logit(rates) - clipped_logit(priors)
    return unique_keys, deviations.astype(np.float64), counts


def fit_pair_model(train, labels, weights):
    model = {
        "entity_rates": {},
        "pair_tables": {},
        "global_rate": float(np.sum(weights * labels) / np.sum(weights)),
    }

    for name in ("video_id", "author_id", "tag"):
        card = int(FEATURE_CARDINALITIES[name])
        rates, counts, _ = fit_entity_rate(
            train.X[name],
            labels,
            weights,
            card,
            smoothing=35.0,
        )
        model["entity_rates"][name] = rates

        pair_keys, pair_values, pair_counts = fit_pair_deviation(
            train.user_id,
            train.X[name],
            labels,
            weights,
            card,
            rates,
            smoothing=PAIR_SMOOTHING[name],
        )
        model["pair_tables"][name] = (
            pair_keys,
            pair_values,
            pair_counts,
        )

    return model


def predict_pair_model(model, split):
    users = np.asarray(split.user_id, dtype=np.int64)
    score = np.zeros(users.size, dtype=np.float64)

    for name, coefficient in BASE_COEFFICIENTS.items():
        entity_ids = np.asarray(split.X[name], dtype=np.int64)
        score += coefficient * clipped_logit(
            model["entity_rates"][name][entity_ids]
        )

    for name, coefficient in PAIR_COEFFICIENTS.items():
        entity_ids = np.asarray(split.X[name], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[name])
        query_keys = users * np.int64(card) + entity_ids
        keys, values, _ = model["pair_tables"][name]
        score += coefficient * lookup_sorted(
            keys,
            values,
            query_keys,
        )

    return score


def fit_latent_model(train, labels, weights, rank=28):
    user_card = int(FEATURE_CARDINALITIES["user_id"])
    video_card = int(FEATURE_CARDINALITIES["video_id"])

    users = np.asarray(train.user_id, dtype=np.int64)
    videos = np.asarray(train.video_id, dtype=np.int64)

    count_matrix = sp.coo_matrix(
        (weights, (users, videos)),
        shape=(user_card, video_card),
        dtype=np.float64,
    ).tocsr()
    positive_matrix = sp.coo_matrix(
        (weights * labels, (users, videos)),
        shape=(user_card, video_card),
        dtype=np.float64,
    ).tocsr()

    item_counts = np.asarray(count_matrix.sum(axis=0)).ravel()
    item_positives = np.asarray(positive_matrix.sum(axis=0)).ravel()
    global_rate = float(np.sum(weights * labels) / np.sum(weights))
    item_rates = (
        item_positives + 35.0 * global_rate
    ) / (
        item_counts + 35.0
    )

    coo_counts = count_matrix.tocoo()
    pair_positive_values = np.asarray(
        positive_matrix[coo_counts.row, coo_counts.col]
    ).ravel()
    pair_rates = (
        pair_positive_values
        + 12.0 * item_rates[coo_counts.col]
    ) / (
        coo_counts.data + 12.0
    )

    residual_values = (
        clipped_logit(pair_rates)
        - clipped_logit(item_rates[coo_counts.col])
    )
    confidence = np.sqrt(np.minimum(coo_counts.data, 12.0))
    residual_values *= confidence

    residual_matrix = sp.coo_matrix(
        (
            residual_values,
            (coo_counts.row, coo_counts.col),
        ),
        shape=(user_card, video_card),
        dtype=np.float64,
    ).tocsr()

    u, singular_values, vt = svds(
        residual_matrix,
        k=rank,
        which="LM",
        random_state=SEED,
        tol=1e-3,
        maxiter=500,
    )
    order = np.argsort(singular_values)[::-1]
    singular_values = singular_values[order]
    u = u[:, order]
    vt = vt[order]

    user_factors = (
        u * np.sqrt(singular_values)[None, :]
    ).astype(np.float32)
    video_factors = (
        vt.T * np.sqrt(singular_values)[None, :]
    ).astype(np.float32)

    return {
        "user_factors": user_factors,
        "video_factors": video_factors,
        "item_rates": item_rates,
        "singular_values": singular_values,
    }


def predict_latent_model(model, split):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    latent = np.einsum(
        "ij,ij->i",
        model["user_factors"][users],
        model["video_factors"][videos],
        optimize=True,
    ).astype(np.float64)
    base = clipped_logit(model["item_rates"][videos])
    return base + 0.65 * latent


def predict_transition_model(
    split,
    initial_last,
    entity_rates,
    transition_keys,
    transition_values,
):
    card = int(FEATURE_CARDINALITIES["author_id"])
    previous = build_previous_entity(
        split,
        "author_id",
        initial_last=initial_last,
    )
    current = np.asarray(split.X["author_id"], dtype=np.int64)
    query_keys = previous * np.int64(card) + current
    transition = lookup_sorted(
        transition_keys,
        transition_values,
        query_keys,
    )
    return clipped_logit(entity_rates[current]) + 0.85 * transition


def standardize(scores):
    scores = np.asarray(scores, dtype=np.float64)
    center = float(np.mean(scores))
    scale = float(np.std(scores))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return (scores - center) / scale, center, scale


def apply_standardization(scores, center, scale):
    return (np.asarray(scores, dtype=np.float64) - center) / scale


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float64)
y_valid = np.asarray(valid.y, dtype=np.int8)
weights = recency_weights(train.date, half_life=4.0)

print(
    "FINDINGS recency_half_life=4 effective_rows={:.0f} weight_min={:.4f} weight_max={:.4f}".format(
        float(weights.sum() ** 2 / np.square(weights).sum()),
        float(weights.min()),
        float(weights.max()),
    ),
    flush=True,
)

# ---------------------------------------------------------------------------
# Family 1: personalized empirical-Bayes user/entity affinities.
# ---------------------------------------------------------------------------
pair_model = fit_pair_model(train, y_train, weights)
pair_valid = predict_pair_model(pair_model, valid)

pair_table_summary = {}
for name, (_, _, counts) in pair_model["pair_tables"].items():
    pair_table_summary[name] = {
        "pairs": int(counts.size),
        "repeat_weight_share": float(np.mean(counts > 1.5)),
    }
print(
    "FINDINGS pair_tables=" + json.dumps(pair_table_summary, sort_keys=True),
    flush=True,
)

# ---------------------------------------------------------------------------
# Family 2: latent matrix factorization of smoothed user-video residual rates.
# ---------------------------------------------------------------------------
latent_model = fit_latent_model(
    train,
    y_train,
    weights,
    rank=28,
)
latent_valid = predict_latent_model(latent_model, valid)

print(
    "FINDINGS latent_rank=28 singular_values_top={}".format(
        np.array2string(
            latent_model["singular_values"][:5],
            precision=3,
            separator=",",
        )
    ),
    flush=True,
)

# ---------------------------------------------------------------------------
# Family 3: sequential author-transition propensity.
# It uses the previous logged impression, ordered strictly by time_ms and row
# position. No current-row or previous-row outcome is used at prediction time.
# ---------------------------------------------------------------------------
author_card = int(FEATURE_CARDINALITIES["author_id"])
author_rates, _, _ = fit_entity_rate(
    train.X["author_id"],
    y_train,
    weights,
    author_card,
    smoothing=35.0,
)
previous_train_author = build_previous_entity(
    train,
    "author_id",
    initial_last=None,
)
transition_keys, transition_values, transition_counts = (
    fit_transition_deviation(
        previous_train_author,
        train.X["author_id"],
        y_train,
        weights,
        author_card,
        author_rates,
        smoothing=28.0,
    )
)
last_train_author = last_entity_by_user(
    train,
    "author_id",
    int(FEATURE_CARDINALITIES["user_id"]),
)
transition_valid = predict_transition_model(
    valid,
    last_train_author,
    author_rates,
    transition_keys,
    transition_values,
)

print(
    "FINDINGS transition_pairs={} repeated_share={:.4f}".format(
        int(transition_counts.size),
        float(np.mean(transition_counts > 1.5)),
    ),
    flush=True,
)

own_valid_predictions = {
    "personalized_eb": pair_valid,
    "latent_mf": latent_valid,
    "sequential_transition": transition_valid,
}

candidate_scores = {}
candidate_predictions = {}
candidate_sources = {}

for name, predictions in own_valid_predictions.items():
    result = evaluate(valid.user_id, y_valid, predictions)
    candidate_scores[name] = float(result["primary"])
    candidate_predictions[name] = predictions
    candidate_sources[name] = (name, 1.0)

    print(
        "FINDINGS standalone={} primary={:.6f} gauc={:.6f} ndcg5={:.6f}".format(
            name,
            float(result["primary"]),
            float(result["gauc"]),
            float(result["ndcg@5"]),
        ),
        flush=True,
    )

shared = os.environ.get("SHARED_ARTIFACTS")
incumbent_valid_path = (
    os.path.join(shared, "incumbent_valid_scores.npy")
    if shared else ""
)
incumbent_test_path = (
    os.path.join(shared, "incumbent_test_scores.npy")
    if shared else ""
)

incumbent_valid = None
blend_metadata = {}

if incumbent_valid_path and os.path.exists(incumbent_valid_path):
    incumbent_valid = np.asarray(
        np.load(incumbent_valid_path),
        dtype=np.float64,
    )
    if incumbent_valid.size != y_valid.size:
        raise ValueError("Trusted incumbent validation length mismatch")

    incumbent_z, incumbent_center, incumbent_scale = standardize(
        incumbent_valid
    )

    for family_name, own_predictions in own_valid_predictions.items():
        own_z, own_center, own_scale = standardize(own_predictions)

        for alpha in BLEND_WEIGHTS:
            blended = (
                (1.0 - alpha) * incumbent_z
                + alpha * own_z
            )
            candidate_name = "{}_blend_{:.2f}".format(
                family_name,
                alpha,
            )
            result = evaluate(valid.user_id, y_valid, blended)
            candidate_scores[candidate_name] = float(result["primary"])
            candidate_predictions[candidate_name] = blended
            candidate_sources[candidate_name] = (
                family_name,
                float(alpha),
            )
            blend_metadata[candidate_name] = {
                "incumbent_center": incumbent_center,
                "incumbent_scale": incumbent_scale,
                "own_center": own_center,
                "own_scale": own_scale,
            }

best_name = max(candidate_scores, key=candidate_scores.get)
valid_scores = np.asarray(
    candidate_predictions[best_name],
    dtype=np.float64,
)
selected_family, selected_alpha = candidate_sources[best_name]
own_model_valid_scores = np.asarray(
    own_valid_predictions[selected_family],
    dtype=np.float64,
)

metrics = evaluate(valid.user_id, y_valid, valid_scores)

print(
    "CANDIDATES " + json.dumps(
        {
            name: round(score, 7)
            for name, score in sorted(candidate_scores.items())
        },
        sort_keys=True,
    ),
    flush=True,
)
print(
    "FINDINGS selected={} source_family={} own_weight={:.2f}".format(
        best_name,
        selected_family,
        selected_alpha,
    ),
    flush=True,
)

# Test labels are never accessed.
test = load("test")

pair_test = predict_pair_model(pair_model, test)
latent_test = predict_latent_model(latent_model, test)
transition_test = predict_transition_model(
    test,
    last_train_author,
    author_rates,
    transition_keys,
    transition_values,
)

own_test_predictions = {
    "personalized_eb": pair_test,
    "latent_mf": latent_test,
    "sequential_transition": transition_test,
}
own_model_test_scores = np.asarray(
    own_test_predictions[selected_family],
    dtype=np.float64,
)

if selected_alpha < 1.0:
    if not incumbent_test_path or not os.path.exists(incumbent_test_path):
        raise FileNotFoundError(
            "Selected blend but trusted incumbent test scores are absent"
        )

    incumbent_test = np.asarray(
        np.load(incumbent_test_path),
        dtype=np.float64,
    )
    if incumbent_test.size != len(test.user_id):
        raise ValueError("Trusted incumbent test length mismatch")

    metadata = blend_metadata[best_name]
    incumbent_test_z = apply_standardization(
        incumbent_test,
        metadata["incumbent_center"],
        metadata["incumbent_scale"],
    )
    own_test_z = apply_standardization(
        own_model_test_scores,
        metadata["own_center"],
        metadata["own_scale"],
    )
    test_scores = (
        (1.0 - selected_alpha) * incumbent_test_z
        + selected_alpha * own_test_z
    )
else:
    test_scores = own_model_test_scores

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
    if selected_alpha < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(own_model_valid_scores, dtype=np.float64),
        )

del train, valid, test
gc.collect()

elapsed = time.time() - START
print(
    'METRICS {{"primary": {:.10f}, "gauc": {:.10f}, "ndcg@5": {:.10f}, "gpu_seconds": {:.4f}}}'.format(
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        float(elapsed),
    )
)