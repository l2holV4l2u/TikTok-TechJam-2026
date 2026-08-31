import os
import time
import json
import gc
import numpy as np
from scipy.spatial import cKDTree

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 20260831
rng = np.random.default_rng(SEED)

train = load("train")
valid = load("valid")
test = load("test")

y = np.asarray(train.y, dtype=np.float32)
tr_users = np.asarray(train.user_id, dtype=np.int64)
yv = np.asarray(valid.y, dtype=np.int8)

n_users = int(FEATURE_CARDINALITIES["user_id"])

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

# Recency statistics are computed strictly from train.
train_dates = np.asarray(train.date, dtype=np.int32)
last_date = int(np.max(train_dates))
ages = (last_date - train_dates).astype(np.float32)

# All dates are in April 2022, so YYYYMMDD subtraction equals day age here.
recency = np.exp(-np.log(2.0) * ages / 6.0).astype(np.float32)
recency /= max(float(np.mean(recency)), 1e-6)
global_rate = float(np.sum(recency * y) / np.sum(recency))


def per_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    n = len(values)

    order = np.lexsort((np.arange(n, dtype=np.int64), values, users))
    su = users[order]

    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = su[1:] != su[:-1]

    starts = np.where(starts_flag, np.arange(n), 0)
    starts = np.maximum.accumulate(starts)
    within = np.arange(n) - starts

    group_starts = np.flatnonzero(starts_flag)
    group_ends = np.r_[group_starts[1:], n]
    sizes = group_ends - group_starts
    sorted_sizes = np.repeat(sizes, sizes)

    rr = np.where(
        sorted_sizes > 1,
        within / np.maximum(sorted_sizes - 1, 1),
        0.5,
    ).astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[order] = rr
    return result


def sorted_lookup(keys, unique_keys, values):
    keys = np.asarray(keys, dtype=np.int64)
    pos = np.searchsorted(unique_keys, keys)
    safe = np.minimum(pos, len(unique_keys) - 1)
    found = (pos < len(unique_keys)) & (unique_keys[safe] == keys)
    result = np.zeros(len(keys), dtype=np.float32)
    result[found] = values[safe[found]]
    return result


inc_valid_rank = per_user_rank(valid.user_id, inc_valid)
inc_test_rank = per_user_rank(test.user_id, inc_test)

# ---------------------------------------------------------------------
# Family 1: collaborative user-neighborhood response smoothing.
#
# Each user is represented by residual response profiles over several stable
# content fields. Nearest users are found in a random-projected profile space.
# Their complete response tables are then averaged, producing a genuinely
# neighborhood-based predictor rather than an identity embedding.
# ---------------------------------------------------------------------

NEIGHBOR_FIELDS = [
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "onehot_feat7",
    "onehot_feat8",
]

user_weight = np.bincount(
    tr_users, weights=recency, minlength=n_users
).astype(np.float32)
active_users = np.flatnonzero(user_weight > 0)

response_tables = {}
signature_parts = []

for field in NEIGHBOR_FIELDS:
    cardinality = int(FEATURE_CARDINALITIES[field])
    category = np.asarray(train.X[field], dtype=np.int64)

    category_weight = np.bincount(
        category, weights=recency, minlength=cardinality
    ).astype(np.float64)
    category_positive = np.bincount(
        category, weights=recency * y, minlength=cardinality
    ).astype(np.float64)
    category_rate = (
        category_positive + 30.0 * global_rate
    ) / np.maximum(category_weight + 30.0, 1e-8)

    residual = y - category_rate[category].astype(np.float32)
    composite = tr_users * np.int64(cardinality) + category

    flat_size = n_users * cardinality
    pair_weight = np.bincount(
        composite,
        weights=recency,
        minlength=flat_size,
    ).reshape(n_users, cardinality).astype(np.float32)

    pair_residual = np.bincount(
        composite,
        weights=recency * residual,
        minlength=flat_size,
    ).reshape(n_users, cardinality).astype(np.float32)

    table = pair_residual / np.maximum(pair_weight + 8.0, 1e-6)
    response_tables[field] = (
        table,
        category_rate.astype(np.float32),
    )

    # Equalize fields before forming the neighborhood geometry.
    field_scale = np.sqrt(float(cardinality))
    signature_parts.append(table * field_scale)

signature = np.concatenate(signature_parts, axis=1).astype(np.float32)

# Random projection makes nearest-neighbor search stable and inexpensive while
# preserving distances among the 26k train-observed users.
PROJECTION_DIM = 56
projection = rng.normal(
    0.0,
    1.0 / np.sqrt(PROJECTION_DIM),
    size=(signature.shape[1], PROJECTION_DIM),
).astype(np.float32)
projected = signature @ projection

norm = np.sqrt(np.sum(projected * projected, axis=1, keepdims=True))
projected /= np.maximum(norm, 1e-5)

active_projected = projected[active_users]
tree = cKDTree(active_projected)

K = 24
distances, local_neighbors = tree.query(
    active_projected,
    k=min(K + 1, len(active_users)),
    workers=-1,
)

if distances.ndim == 1:
    distances = distances[:, None]
    local_neighbors = local_neighbors[:, None]

# The first result is the user itself. Remove it and construct inverse-distance
# weights over other train users.
distances = distances[:, 1:]
local_neighbors = local_neighbors[:, 1:]
neighbor_users = active_users[local_neighbors]

neighbor_weights = 1.0 / np.maximum(distances, 0.08)
neighbor_weights /= np.maximum(
    neighbor_weights.sum(axis=1, keepdims=True), 1e-8
)
neighbor_weights = neighbor_weights.astype(np.float32)

smoothed_tables = {}
for field in NEIGHBOR_FIELDS:
    table, category_rate = response_tables[field]
    smoothed = np.zeros_like(table, dtype=np.float32)

    # Looping over 24 neighbors, not over rows or users.
    for j in range(neighbor_users.shape[1]):
        smoothed[active_users] += (
            neighbor_weights[:, j, None] * table[neighbor_users[:, j]]
        )

    smoothed_tables[field] = (smoothed, category_rate)


def neighbor_predict(split):
    users = np.asarray(split.user_id, dtype=np.int64)
    known_user = (
        (users >= 0)
        & (users < n_users)
        & (user_weight[np.minimum(users, n_users - 1)] > 0)
    )

    result = np.zeros(len(users), dtype=np.float32)
    population = np.zeros(len(users), dtype=np.float32)

    field_weights = {
        "tag": 1.00,
        "tab": 0.90,
        "duration_bucket": 0.85,
        "upload_type": 0.65,
        "onehot_feat7": 0.65,
        "onehot_feat8": 0.70,
    }

    safe_users = np.minimum(np.maximum(users, 0), n_users - 1)

    for field in NEIGHBOR_FIELDS:
        category = np.asarray(split.X[field], dtype=np.int64)
        table, category_rate = smoothed_tables[field]
        safe_category = np.minimum(
            np.maximum(category, 0), len(category_rate) - 1
        )
        weight = field_weights[field]

        personal = table[safe_users, safe_category]
        personal[~known_user] = 0.0

        result += weight * personal
        population += 0.12 * weight * (
            category_rate[safe_category] - global_rate
        )

    return result + population


neighbor_valid = neighbor_predict(valid)
neighbor_test = neighbor_predict(test)

print(
    "FINDINGS neighbor_active_users=%d signature_dim=%d projected_dim=%d"
    % (len(active_users), signature.shape[1], PROJECTION_DIM)
)

del signature, projected, active_projected, projection, tree
gc.collect()

# ---------------------------------------------------------------------
# Family 2: sparse personalized conjunction residuals.
#
# Instead of adding independent user-category affinities, estimate response
# residuals for explicit content conjunctions such as tag x duration and
# tab x upload type. These represent high-order specialization without dense
# neural identity embeddings and back off to train-only population pair rates.
# ---------------------------------------------------------------------

PAIR_FIELDS = [
    ("tag", "duration_bucket"),
    ("tab", "tag"),
    ("upload_type", "tag"),
    ("onehot_feat7", "duration_bucket"),
    ("onehot_feat8", "tag"),
]

pair_tables = []

for field_a, field_b in PAIR_FIELDS:
    card_a = int(FEATURE_CARDINALITIES[field_a])
    card_b = int(FEATURE_CARDINALITIES[field_b])
    pair_cardinality = card_a * card_b

    a = np.asarray(train.X[field_a], dtype=np.int64)
    b = np.asarray(train.X[field_b], dtype=np.int64)
    pair = a * np.int64(card_b) + b

    pair_weight = np.bincount(
        pair, weights=recency, minlength=pair_cardinality
    ).astype(np.float64)
    pair_positive = np.bincount(
        pair, weights=recency * y, minlength=pair_cardinality
    ).astype(np.float64)
    pair_rate = (
        pair_positive + 35.0 * global_rate
    ) / np.maximum(pair_weight + 35.0, 1e-8)

    residual = y - pair_rate[pair].astype(np.float32)
    composite = tr_users * np.int64(pair_cardinality) + pair

    unique_keys, inverse = np.unique(composite, return_inverse=True)
    personal_weight = np.bincount(
        inverse, weights=recency
    ).astype(np.float64)
    personal_residual = np.bincount(
        inverse, weights=recency * residual
    ).astype(np.float64)

    # Strong shrinkage is intentional because evaluation has few rows/user.
    personal_value = (
        personal_residual / np.maximum(personal_weight + 12.0, 1e-8)
    ).astype(np.float32)

    pair_tables.append(
        (
            field_a,
            field_b,
            card_b,
            pair_cardinality,
            unique_keys.astype(np.int64),
            personal_value,
            pair_rate.astype(np.float32),
        )
    )


def conjunction_predict(split):
    users = np.asarray(split.user_id, dtype=np.int64)
    result = np.zeros(len(users), dtype=np.float32)

    for (
        field_a,
        field_b,
        card_b,
        pair_cardinality,
        unique_keys,
        personal_value,
        pair_rate,
    ) in pair_tables:
        a = np.asarray(split.X[field_a], dtype=np.int64)
        b = np.asarray(split.X[field_b], dtype=np.int64)
        pair = a * np.int64(card_b) + b
        pair = np.minimum(np.maximum(pair, 0), pair_cardinality - 1)

        composite = users * np.int64(pair_cardinality) + pair
        personal = sorted_lookup(
            composite, unique_keys, personal_value
        )
        population = pair_rate[pair] - global_rate
        result += personal + 0.16 * population

    return result


conjunction_valid = conjunction_predict(valid)
conjunction_test = conjunction_predict(test)

# ---------------------------------------------------------------------
# Family 3: setwise content-kernel propagation.
#
# The incumbent is pointwise. Within each user's actually logged candidate
# set, repeated or closely related entities provide contextual evidence.
# Propagating scores among candidates sharing video, author, tag, or tab forms
# a setwise prediction while never using evaluation labels.
# ---------------------------------------------------------------------

SETWISE_FIELDS = [
    ("video_id", 1.00),
    ("author_id", 0.85),
    ("tag", 0.55),
    ("tab", 0.40),
    ("duration_bucket", 0.30),
]


def group_mean_signal(users, categories, values, cardinality):
    keys = (
        np.asarray(users, dtype=np.int64) * np.int64(cardinality)
        + np.asarray(categories, dtype=np.int64)
    )
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    sums = np.bincount(inverse, weights=values).astype(np.float64)
    counts = np.bincount(inverse).astype(np.float64)
    means = sums / np.maximum(counts, 1.0)
    return means[inverse], counts[inverse]


def setwise_predict(split, base_rank):
    users = np.asarray(split.user_id, dtype=np.int64)
    numerator = 0.35 * np.asarray(base_rank, dtype=np.float64)
    denominator = np.full(len(users), 0.35, dtype=np.float64)

    for field, weight in SETWISE_FIELDS:
        categories = np.asarray(split.X[field], dtype=np.int64)
        cardinality = int(FEATURE_CARDINALITIES[field])
        means, counts = group_mean_signal(
            users, categories, base_rank, cardinality
        )

        # Singleton groups carry no relational information.
        reliability = np.clip((counts - 1.0) / 2.0, 0.0, 1.0)
        effective = weight * reliability
        numerator += effective * means
        denominator += effective

    return numerator / np.maximum(denominator, 1e-8)


setwise_valid = setwise_predict(valid, inc_valid_rank)
setwise_test = setwise_predict(test, inc_test_rank)

# ---------------------------------------------------------------------
# Validation-only selection among fixed prediction rules and blend weights.
# No parameter or statistic above uses validation labels.
# ---------------------------------------------------------------------

signals = {
    "collaborative_neighbor": (neighbor_valid, neighbor_test),
    "personalized_conjunction": (conjunction_valid, conjunction_test),
    "setwise_content_kernel": (setwise_valid, setwise_test),
    "neighbor_plus_conjunction": (
        0.55 * per_user_rank(valid.user_id, neighbor_valid)
        + 0.45 * per_user_rank(valid.user_id, conjunction_valid),
        0.55 * per_user_rank(test.user_id, neighbor_test)
        + 0.45 * per_user_rank(test.user_id, conjunction_test),
    ),
}

blend_weights = [0.05, 0.10, 0.15, 0.22, 0.30, 0.40]

candidate_scores = {}
best_primary = -np.inf
best_valid_scores = None
best_test_scores = None
best_raw_valid = None
best_name = None

for family_name, (raw_valid, raw_test) in signals.items():
    raw_valid_rank = per_user_rank(valid.user_id, raw_valid)
    raw_test_rank = per_user_rank(test.user_id, raw_test)

    raw_metrics = evaluate(valid.user_id, yv, raw_valid_rank)
    candidate_scores[family_name + "_raw"] = float(
        raw_metrics["primary"]
    )

    family_best = -np.inf
    family_best_alpha = None

    for alpha in blend_weights:
        candidate_valid = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * raw_valid_rank
        )
        metrics = evaluate(valid.user_id, yv, candidate_valid)
        primary = float(metrics["primary"])

        if primary > family_best:
            family_best = primary
            family_best_alpha = alpha

        if primary > best_primary:
            best_primary = primary
            best_valid_scores = candidate_valid.copy()
            best_test_scores = (
                (1.0 - alpha) * inc_test_rank
                + alpha * raw_test_rank
            )
            best_raw_valid = raw_valid_rank.copy()
            best_name = family_name + "_blend_" + str(alpha)

    candidate_scores[
        family_name + "_best_blend_" + str(family_best_alpha)
    ] = float(family_best)

# Also compare a setwise transformation after each new family has already
# been blended with the incumbent. This is a distinct final prediction rule,
# but uses the same validation-selected fixed grid.
for family_name in [
    "collaborative_neighbor",
    "personalized_conjunction",
]:
    raw_valid, raw_test = signals[family_name]
    raw_valid_rank = per_user_rank(valid.user_id, raw_valid)
    raw_test_rank = per_user_rank(test.user_id, raw_test)

    for alpha in [0.10, 0.20, 0.30]:
        preliminary_valid = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * raw_valid_rank
        )
        preliminary_test = (
            (1.0 - alpha) * inc_test_rank
            + alpha * raw_test_rank
        )

        propagated_valid = setwise_predict(valid, preliminary_valid)
        propagated_test = setwise_predict(test, preliminary_test)

        for propagation_weight in [0.15, 0.30]:
            candidate_valid = (
                (1.0 - propagation_weight) * preliminary_valid
                + propagation_weight * propagated_valid
            )
            metrics = evaluate(valid.user_id, yv, candidate_valid)
            primary = float(metrics["primary"])

            name = (
                family_name
                + "_setwise_a"
                + str(alpha)
                + "_p"
                + str(propagation_weight)
            )
            candidate_scores[name] = primary

            if primary > best_primary:
                best_primary = primary
                best_valid_scores = candidate_valid.copy()
                best_test_scores = (
                    (1.0 - propagation_weight) * preliminary_test
                    + propagation_weight * propagated_test
                )
                best_raw_valid = raw_valid_rank.copy()
                best_name = name

final_metrics = evaluate(valid.user_id, yv, best_valid_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS selected=%s primary=%.6f"
    % (best_name, float(final_metrics["primary"]))
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(final_metrics["primary"]),
            "gauc": float(final_metrics["gauc"]),
            "ndcg@5": float(final_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)