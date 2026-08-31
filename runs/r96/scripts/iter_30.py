import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 29173
THREADS = min(16, os.cpu_count() or 1)
np.random.seed(SEED)

IDENTITY_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket", "hour",
    "tag", "upload_type", "music_type", "user_active_degree",
    "fans_user_num_range", "follow_user_num_range", "friend_user_num_range",
    "register_days_range", "is_live_streamer", "is_video_author",
    "onehot_feat1", "onehot_feat2", "onehot_feat3", "onehot_feat7",
    "onehot_feat8", "video_type",
]
STABLE_FIELDS = [
    "tab", "duration_bucket", "hour", "tag", "upload_type", "music_type",
    "user_active_degree", "fans_user_num_range", "follow_user_num_range",
    "friend_user_num_range", "register_days_range", "is_live_streamer",
    "is_video_author", "onehot_feat0", "onehot_feat1", "onehot_feat2",
    "onehot_feat3", "onehot_feat4", "onehot_feat6", "onehot_feat7",
    "onehot_feat8", "onehot_feat9", "video_type",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates, dtype=np.int32)
    age = dates.max() - dates
    weights = np.power(0.5, age.astype(np.float32) / half_life)
    return (weights / np.maximum(weights.mean(), 1e-8)).astype(np.float32)


def make_matrix(split, fields):
    columns = [
        np.asarray(split.X[field], dtype=np.float32)
        for field in fields
    ]
    for field in NUM_FIELDS:
        value = np.asarray(split.num[field], dtype=np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(value, 0.0)).astype(np.float32))
    matrix = np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.float32)
    return matrix, list(range(len(fields)))


def train_binary(x, y, weights, categorical, rounds, seed):
    params = {
        "objective": "binary",
        "metric": "None",
        "learning_rate": 0.045,
        "num_leaves": 63,
        "min_data_in_leaf": 180,
        "feature_fraction": 0.86,
        "bagging_fraction": 0.86,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 1.4,
        "max_bin": 127,
        "num_threads": THREADS,
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "verbose": -1,
    }
    dataset = lgb.Dataset(
        x, label=y, weight=weights,
        categorical_feature=categorical,
        free_raw_data=True,
    )
    model = lgb.train(params, dataset, num_boost_round=rounds)
    return model


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    group_start = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(np.concatenate((np.array([-1]), end_positions)))
    row_sizes = np.repeat(sizes, sizes)

    position = np.arange(n, dtype=np.int64) - group_start
    ranked = (position.astype(np.float64) + 0.5) / row_sizes
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def packed_pair(a, b, b_cardinality):
    return (
        np.asarray(a, dtype=np.int64) * np.int64(b_cardinality)
        + np.asarray(b, dtype=np.int64)
    )


def map_sorted(unique_keys, values, query, default):
    query = np.asarray(query, dtype=np.int64)
    positions = np.searchsorted(unique_keys, query)
    safe = np.minimum(positions, len(unique_keys) - 1)
    found = (
        (positions < len(unique_keys))
        & (unique_keys[safe] == query)
    )
    output = np.full(len(query), default, dtype=np.float32)
    output[found] = values[safe[found]]
    return output


def fit_rate(source_key, source_y, source_w, queries, prior, smooth):
    source_key = np.asarray(source_key, dtype=np.int64)
    unique_keys, inverse = np.unique(source_key, return_inverse=True)
    total_w = np.bincount(
        inverse, weights=source_w, minlength=len(unique_keys)
    ).astype(np.float64)
    positive_w = np.bincount(
        inverse, weights=source_w * source_y, minlength=len(unique_keys)
    ).astype(np.float64)

    rates = ((positive_w + smooth * prior) /
             (total_w + smooth)).astype(np.float32)
    counts = np.log1p(total_w).astype(np.float32)

    result = []
    for query in queries:
        result.append((
            map_sorted(unique_keys, rates, query, prior),
            map_sorted(unique_keys, counts, query, 0.0),
        ))
    return result


def entity_keys(split):
    x = split.X
    return {
        "user": np.asarray(x["user_id"], dtype=np.int64),
        "video": np.asarray(x["video_id"], dtype=np.int64),
        "author": np.asarray(x["author_id"], dtype=np.int64),
        "tag": np.asarray(x["tag"], dtype=np.int64),
        "duration": np.asarray(x["duration_bucket"], dtype=np.int64),
        "author_tag": packed_pair(
            x["author_id"], x["tag"], FEATURE_CARDINALITIES["tag"]
        ),
        "user_tag": packed_pair(
            x["user_id"], x["tag"], FEATURE_CARDINALITIES["tag"]
        ),
        "user_duration": packed_pair(
            x["user_id"], x["duration_bucket"],
            FEATURE_CARDINALITIES["duration_bucket"]
        ),
    }


def build_eb(source_indices, source_keys, target_key_sets, y, weights):
    source_y = y[source_indices]
    source_w = weights[source_indices]
    prior = float(np.sum(source_y * source_w) / np.sum(source_w))

    target_columns = [[] for _ in target_key_sets]
    target_rates = [[] for _ in target_key_sets]

    smooth_by_name = {
        "user": 24.0,
        "video": 14.0,
        "author": 14.0,
        "tag": 30.0,
        "duration": 40.0,
        "author_tag": 18.0,
        "user_tag": 12.0,
        "user_duration": 14.0,
    }

    for name in source_keys:
        mapped = fit_rate(
            source_keys[name][source_indices],
            source_y,
            source_w,
            [keys[name] for keys in target_key_sets],
            prior,
            smooth_by_name[name],
        )
        for j, (rate, count) in enumerate(mapped):
            target_columns[j].extend([rate, count])
            target_rates[j].append(rate)

    matrices = [
        np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.float32)
        for columns in target_columns
    ]

    # A distinct non-parametric expert: robust average of smoothed log odds.
    eb_scores = []
    rate_weights = np.asarray(
        [0.16, 0.10, 0.10, 0.05, 0.04, 0.13, 0.27, 0.15],
        dtype=np.float64,
    )
    rate_weights /= rate_weights.sum()
    for rates in target_rates:
        rate_matrix = np.stack(rates, axis=1).astype(np.float64)
        rate_matrix = np.clip(rate_matrix, 1e-4, 1.0 - 1e-4)
        logits = np.log(rate_matrix / (1.0 - rate_matrix))
        eb_scores.append(logits @ rate_weights)

    return matrices, eb_scores, prior


def train_stacker(features, labels, weights):
    params = {
        "objective": "binary",
        "metric": "None",
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_data_in_leaf": 120,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "num_threads": THREADS,
        "seed": SEED + 100,
        "verbose": -1,
    }
    dataset = lgb.Dataset(
        features, label=labels, weight=weights, free_raw_data=True
    )
    return lgb.train(params, dataset, num_boost_round=180)


train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
dates = np.asarray(train.date, dtype=np.int32)
weights = recency_weights(dates, half_life=4.0)

# The last three train days are a legal pseudo-future used only for stacking.
holdout_start = 20220419
base_mask = dates < holdout_start
holdout_mask = ~base_mask
base_idx = np.flatnonzero(base_mask)
holdout_idx = np.flatnonzero(holdout_mask)

train_identity_x, identity_categorical = make_matrix(train, IDENTITY_FIELDS)
valid_identity_x, _ = make_matrix(valid, IDENTITY_FIELDS)
test_identity_x, _ = make_matrix(test, IDENTITY_FIELDS)

train_stable_x, stable_categorical = make_matrix(train, STABLE_FIELDS)
valid_stable_x, _ = make_matrix(valid, STABLE_FIELDS)
test_stable_x, _ = make_matrix(test, STABLE_FIELDS)

# First-level pseudo-future experts.
early_identity = train_binary(
    train_identity_x[base_idx],
    y_train[base_idx],
    recency_weights(dates[base_idx], 4.0),
    identity_categorical,
    rounds=240,
    seed=SEED,
)
early_stable = train_binary(
    train_stable_x[base_idx],
    y_train[base_idx],
    recency_weights(dates[base_idx], 6.0),
    stable_categorical,
    rounds=240,
    seed=SEED + 20,
)
holdout_identity = early_identity.predict(train_identity_x[holdout_idx])
holdout_stable = early_stable.predict(train_stable_x[holdout_idx])

train_keys = entity_keys(train)
valid_keys = entity_keys(valid)
test_keys = entity_keys(test)

early_eb_matrices, early_eb_scores, early_prior = build_eb(
    base_idx,
    train_keys,
    [{name: values[holdout_idx] for name, values in train_keys.items()}],
    y_train,
    weights,
)
holdout_eb_features = early_eb_matrices[0]
holdout_eb = early_eb_scores[0]

# The stacker can condition expert weights on entity support, not merely average.
holdout_meta_x = np.ascontiguousarray(
    np.column_stack([
        holdout_identity,
        holdout_stable,
        holdout_eb,
        holdout_eb_features,
    ]),
    dtype=np.float32,
)
stacker = train_stacker(
    holdout_meta_x,
    y_train[holdout_idx],
    recency_weights(dates[holdout_idx], 3.0),
)

del early_identity, early_stable, holdout_meta_x
gc.collect()

# Refit first-level experts using every permitted training row.
full_identity = train_binary(
    train_identity_x,
    y_train,
    weights,
    identity_categorical,
    rounds=300,
    seed=SEED + 40,
)
full_stable = train_binary(
    train_stable_x,
    y_train,
    recency_weights(dates, 6.0),
    stable_categorical,
    rounds=300,
    seed=SEED + 60,
)

identity_valid = full_identity.predict(valid_identity_x)
identity_test = full_identity.predict(test_identity_x)
stable_valid = full_stable.predict(valid_stable_x)
stable_test = full_stable.predict(test_stable_x)

all_idx = np.arange(len(y_train), dtype=np.int64)
full_eb_matrices, full_eb_scores, full_prior = build_eb(
    all_idx,
    train_keys,
    [valid_keys, test_keys],
    y_train,
    weights,
)
valid_eb_features, test_eb_features = full_eb_matrices
eb_valid, eb_test = full_eb_scores

valid_meta_x = np.ascontiguousarray(
    np.column_stack([
        identity_valid,
        stable_valid,
        eb_valid,
        valid_eb_features,
    ]),
    dtype=np.float32,
)
test_meta_x = np.ascontiguousarray(
    np.column_stack([
        identity_test,
        stable_test,
        eb_test,
        test_eb_features,
    ]),
    dtype=np.float32,
)
gated_valid = stacker.predict(valid_meta_x)
gated_test = stacker.predict(test_meta_x)

# Rank aggregation is a separate robust fusion rule that cannot be dominated
# merely because the expert probabilities have different calibration.
valid_experts = {
    "identity_boosting": np.asarray(identity_valid, dtype=np.float64),
    "stationary_content_boosting": np.asarray(stable_valid, dtype=np.float64),
    "empirical_bayes": np.asarray(eb_valid, dtype=np.float64),
    "temporal_stacked_gate": np.asarray(gated_valid, dtype=np.float64),
}
test_experts = {
    "identity_boosting": np.asarray(identity_test, dtype=np.float64),
    "stationary_content_boosting": np.asarray(stable_test, dtype=np.float64),
    "empirical_bayes": np.asarray(eb_test, dtype=np.float64),
    "temporal_stacked_gate": np.asarray(gated_test, dtype=np.float64),
}

rank_names = [
    "identity_boosting",
    "stationary_content_boosting",
    "empirical_bayes",
]
valid_experts["rank_aggregation"] = np.mean(
    np.stack([
        rank_percentile(valid.user_id, valid_experts[name])
        for name in rank_names
    ], axis=1),
    axis=1,
)
test_experts["rank_aggregation"] = np.mean(
    np.stack([
        rank_percentile(test.user_id, test_experts[name])
        for name in rank_names
    ], axis=1),
    axis=1,
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

candidate_metrics = {}
candidate_valid = {}
candidate_spec = {}

# Preserve the trusted incumbent while testing standalone and blended experts.
candidate_valid["trusted_incumbent"] = inc_valid
candidate_metrics["trusted_incumbent"] = evaluate(
    valid.user_id, valid.y, inc_valid
)
candidate_spec["trusted_incumbent"] = ("temporal_stacked_gate", 0.0)

for family, valid_score in valid_experts.items():
    standalone_name = family + "_standalone"
    candidate_valid[standalone_name] = valid_score
    candidate_metrics[standalone_name] = evaluate(
        valid.user_id, valid.y, valid_score
    )
    candidate_spec[standalone_name] = (family, None)

    family_rank = rank_percentile(valid.user_id, valid_score)
    for alpha in (0.10, 0.20, 0.30, 0.40, 0.50):
        name = f"{family}_incumbent_blend_{alpha:.2f}"
        blended = (
            alpha * family_rank
            + (1.0 - alpha) * inc_valid_rank
        )
        candidate_valid[name] = blended
        candidate_metrics[name] = evaluate(
            valid.user_id, valid.y, blended
        )
        candidate_spec[name] = (family, alpha)

best_name = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"]),
)
best_metrics = candidate_metrics[best_name]
best_valid = candidate_valid[best_name]
best_family, best_alpha = candidate_spec[best_name]

if best_name == "trusted_incumbent":
    best_test = inc_test
    raw_valid = valid_experts["temporal_stacked_gate"]
elif best_alpha is None:
    best_test = test_experts[best_family]
    raw_valid = valid_experts[best_family]
else:
    test_family_rank = rank_percentile(
        test.user_id, test_experts[best_family]
    )
    best_test = (
        best_alpha * test_family_rank
        + (1.0 - best_alpha) * inc_test_rank
    )
    raw_valid = valid_experts[best_family]

print("CANDIDATES " + json.dumps(
    {
        name: float(metrics["primary"])
        for name, metrics in candidate_metrics.items()
    },
    sort_keys=True,
))
print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "stacker_holdout_start": int(holdout_start),
    "stacker_rows": int(len(holdout_idx)),
    "base_rows": int(len(base_idx)),
    "early_empirical_bayes_prior": float(early_prior),
    "full_empirical_bayes_prior": float(full_prior),
    "mechanism": "support-conditioned pseudo-future stacking",
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
    if best_name == "trusted_incumbent" or best_alpha is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))