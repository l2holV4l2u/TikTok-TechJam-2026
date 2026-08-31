import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 73191
np.random.seed(SEED)

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tag", "tab", "hour",
    "duration_bucket", "upload_type", "music_type", "video_type",
    "user_active_degree", "is_live_streamer", "is_video_author",
    "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
    "onehot_feat0", "onehot_feat1", "onehot_feat2", "onehot_feat3",
    "onehot_feat4", "onehot_feat6", "onehot_feat7", "onehot_feat8",
    "onehot_feat9", "onehot_feat10", "onehot_feat11", "onehot_feat12",
    "onehot_feat13", "onehot_feat14", "onehot_feat15",
    "onehot_feat16", "onehot_feat17",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
TRANSITION_FIELDS = ["author_id", "tag", "duration_bucket"]
TRANSITION_SMOOTHING = {
    "author_id": 18.0,
    "tag": 35.0,
    "duration_bucket": 70.0,
}
ENTITY_SMOOTHING = {
    "author_id": 28.0,
    "tag": 55.0,
    "duration_bucket": 100.0,
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

    positions = np.arange(n, dtype=np.int64) - first
    ranked = (positions.astype(np.float64) + 0.5) / row_sizes
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def logit(probability):
    probability = np.clip(
        np.asarray(probability, dtype=np.float64), 1e-5, 1.0 - 1e-5
    )
    return np.log(probability / (1.0 - probability))


def chronological_previous(split, values):
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64)
    n = len(values)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, times, users))

    ordered_users = users[order]
    ordered_values = values[order]
    previous_ordered = np.zeros(n, dtype=np.int64)
    if n > 1:
        same_user = ordered_users[1:] == ordered_users[:-1]
        previous_ordered[1:] = np.where(
            same_user, ordered_values[:-1], 0
        )

    previous = np.empty(n, dtype=np.int64)
    previous[order] = previous_ordered
    return previous


def fit_rate(keys, labels, weights, prior, smoothing):
    keys = np.asarray(keys, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    denominator = np.bincount(inverse, weights=weights)
    numerator = np.bincount(inverse, weights=weights * labels)
    rates = (numerator + smoothing * prior) / (
        denominator + smoothing
    )
    return unique_keys, rates.astype(np.float64)


def lookup_rate(keys, fitted_keys, rates, default):
    keys = np.asarray(keys, dtype=np.int64)
    positions = np.searchsorted(fitted_keys, keys)
    output = np.full(len(keys), default, dtype=np.float64)
    in_range = positions < len(fitted_keys)
    indices = np.flatnonzero(in_range)
    if len(indices):
        matched = fitted_keys[positions[indices]] == keys[indices]
        indices = indices[matched]
        output[indices] = rates[positions[indices]]
    return output


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates, dtype=np.int32)
    age = np.maximum(int(dates.max()) - dates, 0).astype(np.float64)
    return np.power(0.5, age / half_life).astype(np.float32)


def fit_transition_model(train):
    y = np.asarray(train.y, dtype=np.float64)
    weights = recency_weights(train.date, half_life=4.0).astype(np.float64)
    prior = float(np.sum(weights * y) / np.sum(weights))
    model = {"prior": prior, "fields": {}}

    for field in TRANSITION_FIELDS:
        current = np.asarray(train.X[field], dtype=np.int64)
        previous = chronological_previous(train, current)
        cardinality = int(max(current.max(), previous.max()) + 1)
        transition_key = previous * np.int64(cardinality) + current

        entity_keys, entity_rates = fit_rate(
            current, y, weights, prior, ENTITY_SMOOTHING[field]
        )
        transition_keys, transition_rates = fit_rate(
            transition_key, y, weights, prior,
            TRANSITION_SMOOTHING[field]
        )
        model["fields"][field] = {
            "cardinality": cardinality,
            "entity_keys": entity_keys,
            "entity_rates": entity_rates,
            "transition_keys": transition_keys,
            "transition_rates": transition_rates,
        }
    return model


def predict_transition(model, split):
    total = np.zeros(len(split), dtype=np.float64)
    for field in TRANSITION_FIELDS:
        state = model["fields"][field]
        current = np.asarray(split.X[field], dtype=np.int64)
        previous = chronological_previous(split, current)
        key = previous * np.int64(state["cardinality"]) + current

        entity_rate = lookup_rate(
            current, state["entity_keys"], state["entity_rates"],
            model["prior"]
        )
        transition_rate = lookup_rate(
            key, state["transition_keys"], state["transition_rates"],
            model["prior"]
        )

        # The transition estimate is deliberately shrunk toward the current
        # entity estimate because many evaluation sequences are very short.
        total += 0.62 * logit(transition_rate) + 0.38 * logit(entity_rate)

    return total / float(len(TRANSITION_FIELDS))


def get_history(split_name):
    video = historical_features(split_name, key="video_id")
    author = historical_features(split_name, key="author_id")
    result = {}
    for key, value in video.items():
        result["video_" + key] = np.asarray(value, dtype=np.float32)
    for key, value in author.items():
        result["author_" + key] = np.asarray(value, dtype=np.float32)
    return result


def make_matrix(split, split_name, history_names=None):
    columns = []

    for field in CAT_FIELDS:
        columns.append(np.asarray(split.X[field], dtype=np.float32))

    for field in NUM_FIELDS:
        value = np.asarray(split.num[field], dtype=np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        value = np.sign(value) * np.log1p(np.abs(value))
        columns.append(value.astype(np.float32))

    # Smooth cyclic time coordinates let trees reuse nearby hours instead of
    # treating every hour exclusively as an unrelated category.
    hour = np.asarray(split.X["hour"], dtype=np.float32)
    columns.append(np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32))
    columns.append(np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32))

    history = get_history(split_name)
    if history_names is None:
        history_names = sorted(history)
    for name in history_names:
        value = np.asarray(history[name], dtype=np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(value)

    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    return matrix, history_names


def sorted_ranking_data(split, matrix):
    users = np.asarray(split.user_id, dtype=np.int64)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    _, group = np.unique(sorted_users, return_counts=True)
    labels = np.asarray(split.y, dtype=np.float32)[order]
    weights = recency_weights(split.date, half_life=4.0)[order]
    return matrix[order], labels, weights, group.astype(np.int32)


def train_ranker(objective, x, labels, weights, group):
    dataset = lgb.Dataset(
        x,
        label=labels,
        weight=weights,
        group=group,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )
    params = {
        "objective": objective,
        "metric": "None",
        "learning_rate": 0.045,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 120,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.15,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "max_cat_threshold": 32,
        "max_cat_to_onehot": 8,
        "verbosity": -1,
        "num_threads": max(1, min(12, os.cpu_count() or 1)),
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "data_random_seed": SEED + 3,
    }
    if objective == "lambdarank":
        params["lambdarank_truncation_level"] = 5
        params["label_gain"] = [0, 1]
    return lgb.train(params, dataset, num_boost_round=420)


train = load("train")
valid = load("valid")
test = load("test")

transition_model = fit_transition_model(train)
transition_valid = predict_transition(transition_model, valid)
transition_test = predict_transition(transition_model, test)

x_train, history_names = make_matrix(train, "train")
x_valid, _ = make_matrix(valid, "valid", history_names)
x_test, _ = make_matrix(test, "test", history_names)

x_rank, y_rank, w_rank, groups = sorted_ranking_data(train, x_train)
del x_train
gc.collect()

family_valid = {
    "chronological_transition": transition_valid,
}
family_test = {
    "chronological_transition": transition_test,
}

model_failures = {}
for objective, family_name in [
    ("lambdarank", "recency_lambdamart"),
    ("rank_xendcg", "recency_rank_xendcg"),
]:
    try:
        model = train_ranker(
            objective, x_rank, y_rank, w_rank, groups
        )
        family_valid[family_name] = model.predict(
            x_valid, num_iteration=model.current_iteration()
        ).astype(np.float64)
        family_test[family_name] = model.predict(
            x_test, num_iteration=model.current_iteration()
        ).astype(np.float64)
        del model
        gc.collect()
    except Exception as exc:
        model_failures[family_name] = repr(exc)

del x_rank, y_rank, w_rank, groups, x_valid, x_test
gc.collect()

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
    name: rank_percentile(valid.user_id, scores)
    for name, scores in family_valid.items()
}
family_test_rank = {
    name: rank_percentile(test.user_id, scores)
    for name, scores in family_test.items()
}

candidate_valid = {"incumbent": inc_valid}
candidate_test = {"incumbent": inc_test}
candidate_raw = {"incumbent": inc_valid}

for name in family_valid:
    candidate_valid[name + "_standalone"] = family_valid[name]
    candidate_test[name + "_standalone"] = family_test[name]
    candidate_raw[name + "_standalone"] = family_valid[name]

    for alpha in (0.10, 0.20, 0.35, 0.50, 0.65):
        key = f"{name}_incblend_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * family_valid_rank[name]
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank
            + alpha * family_test_rank[name]
        )
        candidate_raw[key] = family_valid[name]

# Cross-family compositions test whether session context repairs different
# errors from the globally learned ranking objectives.
ranker_names = [
    name for name in ("recency_lambdamart", "recency_rank_xendcg")
    if name in family_valid
]
for ranker_name in ranker_names:
    combined_valid = (
        0.78 * family_valid_rank[ranker_name]
        + 0.22 * family_valid_rank["chronological_transition"]
    )
    combined_test = (
        0.78 * family_test_rank[ranker_name]
        + 0.22 * family_test_rank["chronological_transition"]
    )
    family_key = ranker_name + "_with_transition"

    candidate_valid[family_key + "_standalone"] = combined_valid
    candidate_test[family_key + "_standalone"] = combined_test
    candidate_raw[family_key + "_standalone"] = combined_valid

    for alpha in (0.20, 0.35, 0.50, 0.65):
        key = f"{family_key}_incblend_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank + alpha * combined_valid
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank + alpha * combined_test
        )
        candidate_raw[key] = combined_valid

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
    "model_failures": model_failures,
    "rank_correlations_with_incumbent": {
        name: float(np.corrcoef(inc_valid_rank, scores)[0, 1])
        for name, scores in family_valid_rank.items()
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