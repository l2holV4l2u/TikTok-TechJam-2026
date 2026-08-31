import os
import time
import json
import gc
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

START = time.time()
SEED = 20260831
rng = np.random.default_rng(SEED)

train = load("train")
valid = load("valid")
test = load("test")

y = np.asarray(train.y, dtype=np.float32)
yv = np.asarray(valid.y, dtype=np.int8)
tr_users = np.asarray(train.user_id, dtype=np.int64)
n_users = int(FEATURE_CARDINALITIES["user_id"])
n_videos = int(FEATURE_CARDINALITIES["video_id"])

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

# All recency weights depend only on train dates.
train_dates = np.asarray(train.date, dtype=np.int32)
last_date = int(np.max(train_dates))
unique_dates = np.unique(train_dates)

def date_to_day(d):
    s = str(int(d))
    return int(
        (
            np.datetime64(s[:4] + "-" + s[4:6] + "-" + s[6:8])
            - np.datetime64("2022-01-01")
        ).astype(int)
    )

last_day = date_to_day(last_date)
date_age_table = {
    int(d): float(last_day - date_to_day(d)) for d in unique_dates
}
ages = np.fromiter(
    (date_age_table[int(d)] for d in train_dates),
    dtype=np.float32,
    count=len(train_dates),
)
recency = np.exp(-np.log(2.0) * ages / 5.0).astype(np.float32)
recency /= max(float(np.mean(recency)), 1e-6)

global_rate = float(np.sum(recency * y) / np.sum(recency))


def per_user_rank(user_ids, scores):
    """Map scores to [0,1] ranks independently within each logged user."""
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    n = len(values)

    order = np.lexsort((np.arange(n, dtype=np.int64), values, users))
    sorted_users = users[order]

    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = sorted_users[1:] != sorted_users[:-1]

    starts = np.where(new_group, np.arange(n), 0)
    starts = np.maximum.accumulate(starts)
    within = np.arange(n) - starts

    group_starts = np.flatnonzero(new_group)
    group_ends = np.r_[group_starts[1:], n]
    group_sizes = group_ends - group_starts
    sizes_sorted = np.repeat(group_sizes, group_sizes)

    ranked_sorted = np.where(
        sizes_sorted > 1,
        within / np.maximum(sizes_sorted - 1, 1),
        0.5,
    )
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


inc_valid_rank = per_user_rank(valid.user_id, inc_valid)
inc_test_rank = per_user_rank(test.user_id, inc_test)

# ---------------------------------------------------------------------
# Family 1: supervised sequential item-transition spectral graph.
#
# Consecutive train impressions create a directed video graph. Transitions
# connecting positively received videos receive the most weight. A low-rank
# spectral representation maps each user's recency-weighted positive history
# to a vector, and candidates are scored by alignment with that vector.
# ---------------------------------------------------------------------

row_position = np.arange(len(y), dtype=np.int64)
order = np.lexsort(
    (
        row_position,
        np.asarray(train.time_ms, dtype=np.int64),
        tr_users,
    )
)
ordered_users = tr_users[order]
ordered_videos = np.asarray(train.video_id, dtype=np.int64)[order]
ordered_y = y[order]
ordered_recency = recency[order]

same_user = ordered_users[1:] == ordered_users[:-1]
src = ordered_videos[:-1][same_user]
dst = ordered_videos[1:][same_user]

previous_y = ordered_y[:-1][same_user]
current_y = ordered_y[1:][same_user]
edge_recency = np.sqrt(
    ordered_recency[:-1][same_user] * ordered_recency[1:][same_user]
)

# Positive-positive transitions dominate, but weaker exposure transitions
# preserve graph connectivity for less active users and items.
edge_weight = edge_recency * (
    0.08
    + 0.42 * previous_y
    + 0.42 * current_y
    + 1.10 * previous_y * current_y
)
transition = sp.coo_matrix(
    (edge_weight.astype(np.float32), (src, dst)),
    shape=(n_videos, n_videos),
).tocsr()
transition = transition + transition.T

degree = np.asarray(transition.sum(axis=1)).ravel().astype(np.float64)
inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-6))
normalized_graph = sp.diags(inv_sqrt_degree) @ transition @ sp.diags(
    inv_sqrt_degree
)

GRAPH_DIM = 28
try:
    eig_values, eig_vectors = svds(
        normalized_graph.astype(np.float64),
        k=GRAPH_DIM,
        which="LM",
        random_state=SEED,
    )
    descending = np.argsort(np.abs(eig_values))[::-1]
    eig_values = eig_values[descending]
    eig_vectors = eig_vectors[:, descending]
    video_graph_embedding = (
        eig_vectors * np.sqrt(np.maximum(np.abs(eig_values), 1e-8))[None, :]
    ).astype(np.float32)
except Exception as exc:
    print("FINDINGS graph_svd_fallback=%s" % repr(exc))
    video_graph_embedding = rng.normal(
        0.0, 1.0 / np.sqrt(GRAPH_DIM), size=(n_videos, GRAPH_DIM)
    ).astype(np.float32)
    video_graph_embedding[0] = 0.0

# Sparse user-positive-video matrix computes all profiles without row loops.
positive_profile_weight = (
    recency * (0.15 + 0.85 * y)
).astype(np.float32)
user_video = sp.coo_matrix(
    (
        positive_profile_weight,
        (
            tr_users,
            np.asarray(train.video_id, dtype=np.int64),
        ),
    ),
    shape=(n_users, n_videos),
).tocsr()

user_graph_embedding = (
    user_video @ video_graph_embedding
).astype(np.float32)
profile_norm = np.sqrt(
    np.sum(user_graph_embedding * user_graph_embedding, axis=1, keepdims=True)
)
user_graph_embedding /= np.maximum(profile_norm, 1e-5)

video_norm = np.sqrt(
    np.sum(video_graph_embedding * video_graph_embedding, axis=1, keepdims=True)
)
video_graph_embedding /= np.maximum(video_norm, 1e-5)


def graph_predict(split):
    u = np.asarray(split.user_id, dtype=np.int64)
    v = np.asarray(split.video_id, dtype=np.int64)
    result = np.zeros(len(u), dtype=np.float32)
    known = (u >= 0) & (u < n_users) & (v >= 0) & (v < n_videos)
    result[known] = np.sum(
        user_graph_embedding[u[known]] * video_graph_embedding[v[known]],
        axis=1,
    )
    return result


graph_valid = graph_predict(valid)
graph_test = graph_predict(test)

print(
    "FINDINGS graph_edges=%d graph_nonisolated_videos=%d"
    % (transition.nnz, int(np.sum(degree > 0)))
)

del transition, normalized_graph, user_video
gc.collect()

# ---------------------------------------------------------------------
# Family 2: hierarchical user-content residual affinity.
#
# First estimate each category's population response. Then estimate how much
# each user systematically over- or under-performs that baseline for the
# category. This removes popularity before learning personal affinities.
# ---------------------------------------------------------------------

AFFINITY_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "music_type",
    "upload_type",
]

AFFINITY_RIDGE = {
    "video_id": 10.0,
    "author_id": 9.0,
    "tag": 14.0,
    "tab": 18.0,
    "duration_bucket": 18.0,
    "onehot_feat3": 14.0,
    "onehot_feat7": 16.0,
    "onehot_feat8": 16.0,
    "music_type": 20.0,
    "upload_type": 20.0,
}


def sorted_lookup(keys, unique_keys, values):
    keys = np.asarray(keys, dtype=np.int64)
    pos = np.searchsorted(unique_keys, keys)
    safe = np.minimum(pos, len(unique_keys) - 1)
    found = (pos < len(unique_keys)) & (unique_keys[safe] == keys)
    out = np.zeros(len(keys), dtype=np.float32)
    out[found] = values[safe[found]]
    return out


affinity_tables = {}
category_rate_tables = {}

for field in AFFINITY_FIELDS:
    cardinality = int(FEATURE_CARDINALITIES[field])
    category = np.asarray(train.X[field], dtype=np.int64)

    cat_weight = np.bincount(
        category, weights=recency, minlength=cardinality
    ).astype(np.float64)
    cat_positive = np.bincount(
        category, weights=recency * y, minlength=cardinality
    ).astype(np.float64)

    category_rate = (
        cat_positive + 24.0 * global_rate
    ) / np.maximum(cat_weight + 24.0, 1e-8)
    category_rate_tables[field] = category_rate.astype(np.float32)

    residual = y - category_rate[category]
    composite = tr_users * np.int64(cardinality) + category

    unique_keys, inverse = np.unique(composite, return_inverse=True)
    pair_weight = np.bincount(
        inverse, weights=recency
    ).astype(np.float64)
    pair_residual = np.bincount(
        inverse, weights=recency * residual
    ).astype(np.float64)

    ridge = AFFINITY_RIDGE[field]
    pair_value = (
        pair_residual / np.maximum(pair_weight + ridge, 1e-8)
    ).astype(np.float32)
    affinity_tables[field] = (
        cardinality,
        unique_keys.astype(np.int64),
        pair_value,
    )


def affinity_predict(split):
    users = np.asarray(split.user_id, dtype=np.int64)
    result = np.zeros(len(users), dtype=np.float32)

    # More specific fields get slightly greater influence.
    field_weights = {
        "video_id": 1.00,
        "author_id": 0.90,
        "tag": 0.65,
        "tab": 0.55,
        "duration_bucket": 0.55,
        "onehot_feat3": 0.55,
        "onehot_feat7": 0.45,
        "onehot_feat8": 0.45,
        "music_type": 0.35,
        "upload_type": 0.35,
    }

    for field in AFFINITY_FIELDS:
        category = np.asarray(split.X[field], dtype=np.int64)
        cardinality, unique_keys, pair_value = affinity_tables[field]
        composite = users * np.int64(cardinality) + category
        personal = sorted_lookup(composite, unique_keys, pair_value)

        rate_table = category_rate_tables[field]
        safe_category = np.minimum(category, len(rate_table) - 1)
        population = rate_table[safe_category] - global_rate

        result += field_weights[field] * (
            personal + 0.18 * population.astype(np.float32)
        )
    return result


affinity_valid = affinity_predict(valid)
affinity_test = affinity_predict(test)

# ---------------------------------------------------------------------
# Family 3: demographic-cohort conditional popularity.
#
# Users with sparse evaluation histories cannot support reliable identity
# effects. Stable train-observed user attributes define cohorts; cohort-item
# target statistics transfer preferences to sparse and nearly cold users.
# ---------------------------------------------------------------------

COHORT_USER_FIELDS = [
    "user_active_degree",
    "fans_user_num_range",
    "register_days_bucket",
]
COHORT_ITEM_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "onehot_feat3",
]

# Mixed-radix cohort id.
cohort_multiplier = 1
train_cohort = np.zeros(len(y), dtype=np.int64)
valid_cohort = np.zeros(len(valid.user_id), dtype=np.int64)
test_cohort = np.zeros(len(test.user_id), dtype=np.int64)

for field in COHORT_USER_FIELDS:
    train_cohort += (
        np.asarray(train.X[field], dtype=np.int64) * cohort_multiplier
    )
    valid_cohort += (
        np.asarray(valid.X[field], dtype=np.int64) * cohort_multiplier
    )
    test_cohort += (
        np.asarray(test.X[field], dtype=np.int64) * cohort_multiplier
    )
    cohort_multiplier *= int(FEATURE_CARDINALITIES[field])

cohort_tables = {}

for field in COHORT_ITEM_FIELDS:
    cardinality = int(FEATURE_CARDINALITIES[field])
    item = np.asarray(train.X[field], dtype=np.int64)
    composite = train_cohort * np.int64(cardinality) + item

    unique_keys, inverse = np.unique(composite, return_inverse=True)
    counts = np.bincount(inverse, weights=recency).astype(np.float64)
    positives = np.bincount(
        inverse, weights=recency * y
    ).astype(np.float64)

    # Global item rate is the backoff target.
    item_count = np.bincount(
        item, weights=recency, minlength=cardinality
    ).astype(np.float64)
    item_positive = np.bincount(
        item, weights=recency * y, minlength=cardinality
    ).astype(np.float64)
    item_rate = (
        item_positive + 30.0 * global_rate
    ) / np.maximum(item_count + 30.0, 1e-8)

    prior = item_rate[item]
    prior_sum = np.bincount(
        inverse, weights=recency * prior
    ).astype(np.float64)
    cohort_residual = (
        (positives - prior_sum) / np.maximum(counts + 28.0, 1e-8)
    ).astype(np.float32)

    cohort_tables[field] = (
        cardinality,
        unique_keys.astype(np.int64),
        cohort_residual,
        item_rate.astype(np.float32),
    )


def cohort_predict(split, cohorts):
    result = np.zeros(len(cohorts), dtype=np.float32)
    for field in COHORT_ITEM_FIELDS:
        item = np.asarray(split.X[field], dtype=np.int64)
        cardinality, keys, residual, item_rate = cohort_tables[field]
        composite = cohorts * np.int64(cardinality) + item
        conditional = sorted_lookup(composite, keys, residual)
        safe_item = np.minimum(item, len(item_rate) - 1)
        result += conditional + 0.22 * (
            item_rate[safe_item] - global_rate
        )
    return result


cohort_valid = cohort_predict(valid, valid_cohort)
cohort_test = cohort_predict(test, test_cohort)

# ---------------------------------------------------------------------
# Family 4: organizer-provided train-only entity histories, combined
# non-parametrically rather than through another fitted boosting model.
# ---------------------------------------------------------------------

def get_long_view_rate(hist, entity):
    expected = entity + "_long_view_rate"
    if expected in hist:
        return np.asarray(hist[expected], dtype=np.float32)
    matches = sorted(
        key for key in hist.keys() if "long_view_rate" in key
    )
    if not matches:
        raise RuntimeError("No long_view_rate history for " + entity)
    return np.asarray(hist[matches[0]], dtype=np.float32)


hv = historical_features("valid", key="video_id")
ha = historical_features("valid", key="author_id")
htv = historical_features("test", key="video_id")
hta = historical_features("test", key="author_id")

video_rate_valid = get_long_view_rate(hv, "video_id")
author_rate_valid = get_long_view_rate(ha, "author_id")
video_rate_test = get_long_view_rate(htv, "video_id")
author_rate_test = get_long_view_rate(hta, "author_id")


def safe_logit(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, global_rate)
    x = np.clip(x, 1e-4, 1.0 - 1e-4)
    return np.log(x / (1.0 - x))


history_valid = (
    0.72 * safe_logit(video_rate_valid)
    + 0.28 * safe_logit(author_rate_valid)
).astype(np.float32)
history_test = (
    0.72 * safe_logit(video_rate_test)
    + 0.28 * safe_logit(author_rate_test)
).astype(np.float32)

del hv, ha, htv, hta
gc.collect()

families_valid = {
    "transition_graph": graph_valid,
    "user_content_residual": affinity_valid,
    "cohort_conditional": cohort_valid,
    "history_popularity": history_valid,
}
families_test = {
    "transition_graph": graph_test,
    "user_content_residual": affinity_test,
    "cohort_conditional": cohort_test,
    "history_popularity": history_test,
}

candidate_scores = {}
candidate_arrays = {}
candidate_raw_arrays = {}

# Record every standalone family and its best incumbent blend.
for name in families_valid:
    raw_v = np.asarray(families_valid[name], dtype=np.float64)
    raw_t = np.asarray(families_test[name], dtype=np.float64)

    raw_metric = evaluate(valid.user_id, yv, raw_v)
    candidate_scores[name + "_raw"] = float(raw_metric["primary"])

    rank_v = per_user_rank(valid.user_id, raw_v)
    rank_t = per_user_rank(test.user_id, raw_t)

    best_local_score = -np.inf
    best_local_v = None
    best_local_t = None
    best_alpha = None

    # alpha is the contribution from the new family.
    for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]:
        blend_v = (1.0 - alpha) * inc_valid_rank + alpha * rank_v
        metric = evaluate(valid.user_id, yv, blend_v)
        candidate_scores[
            "%s_blend_%.2f" % (name, alpha)
        ] = float(metric["primary"])

        if metric["primary"] > best_local_score:
            best_local_score = float(metric["primary"])
            best_local_v = blend_v.copy()
            best_local_t = (
                (1.0 - alpha) * inc_test_rank + alpha * rank_t
            )
            best_alpha = alpha

    key = "%s_best_blend" % name
    candidate_arrays[key] = (best_local_v, best_local_t)
    candidate_raw_arrays[key] = (raw_v, raw_t)
    candidate_scores[key] = best_local_score
    print(
        "FINDINGS %s_raw=%.6f best_blend=%.6f alpha=%.2f"
        % (
            name,
            raw_metric["primary"],
            best_local_score,
            best_alpha,
        )
    )

# Cross-family rank ensemble: averaging structurally different errors can
# reduce variance before a conservative incumbent blend.
family_rank_valid = {
    name: per_user_rank(valid.user_id, values)
    for name, values in families_valid.items()
}
family_rank_test = {
    name: per_user_rank(test.user_id, values)
    for name, values in families_test.items()
}

ensemble_specs = {
    "graph_affinity": ["transition_graph", "user_content_residual"],
    "personalized_three": [
        "transition_graph",
        "user_content_residual",
        "cohort_conditional",
    ],
    "all_four": list(families_valid.keys()),
}

for ensemble_name, members in ensemble_specs.items():
    own_v = np.mean(
        np.column_stack([family_rank_valid[m] for m in members]), axis=1
    )
    own_t = np.mean(
        np.column_stack([family_rank_test[m] for m in members]), axis=1
    )

    raw_metric = evaluate(valid.user_id, yv, own_v)
    candidate_scores[ensemble_name + "_raw"] = float(
        raw_metric["primary"]
    )

    best_score = -np.inf
    best_v = None
    best_t = None
    best_alpha = None

    for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]:
        blend_v = (1.0 - alpha) * inc_valid_rank + alpha * own_v
        metric = evaluate(valid.user_id, yv, blend_v)
        candidate_scores[
            "%s_blend_%.2f" % (ensemble_name, alpha)
        ] = float(metric["primary"])
        if metric["primary"] > best_score:
            best_score = float(metric["primary"])
            best_v = blend_v.copy()
            best_t = (
                (1.0 - alpha) * inc_test_rank + alpha * own_t
            )
            best_alpha = alpha

    key = ensemble_name + "_best_blend"
    candidate_arrays[key] = (best_v, best_t)
    candidate_raw_arrays[key] = (own_v, own_t)
    candidate_scores[key] = best_score
    print(
        "FINDINGS %s_raw=%.6f best_blend=%.6f alpha=%.2f"
        % (
            ensemble_name,
            raw_metric["primary"],
            best_score,
            best_alpha,
        )
    )

# Include the trusted incumbent unchanged as a safe candidate.
inc_metric = evaluate(valid.user_id, yv, inc_valid)
candidate_scores["trusted_incumbent"] = float(inc_metric["primary"])
candidate_arrays["trusted_incumbent"] = (inc_valid, inc_test)
candidate_raw_arrays["trusted_incumbent"] = (inc_valid, inc_test)

selectable = {
    name: candidate_scores[name] for name in candidate_arrays.keys()
}
winner = max(selectable, key=selectable.get)
valid_scores, test_scores = candidate_arrays[winner]
raw_valid_scores, _ = candidate_raw_arrays[winner]

metrics = evaluate(valid.user_id, yv, valid_scores)

print(
    "FINDINGS winner=%s incumbent_primary=%.6f"
    % (winner, inc_metric["primary"])
)
print(
    "CANDIDATES "
    + json.dumps(
        {k: round(float(v), 7) for k, v in candidate_scores.items()},
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
    if winner != "trusted_incumbent":
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