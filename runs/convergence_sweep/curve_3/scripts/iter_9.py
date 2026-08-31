import os
import time
import json
import gc

import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 48117
np.random.seed(SEED)

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag", "upload_type",
    "onehot_feat3", "onehot_feat8", "duration_bucket", "music_type",
    "user_active_degree", "register_days_bucket", "hour",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
HISTORY_KEYS = [
    "train_count_log1p", "long_view_rate", "is_click_rate",
    "play_time_ms_logmean",
]


def recency_weights(dates, half_life=5.0):
    d = np.asarray(dates, dtype=np.int64)
    age = d.max() - d
    w = np.power(0.5, age.astype(np.float32) / half_life)
    w /= max(float(w.mean()), 1e-8)
    return np.ascontiguousarray(w, dtype=np.float32)


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


def aggregate_rate(ids, y, weights, cardinality, strength, prior):
    ids = np.asarray(ids, dtype=np.int64)
    count = np.bincount(ids, weights=weights, minlength=cardinality)
    positive = np.bincount(ids, weights=weights * y, minlength=cardinality)
    return (positive + strength * prior) / (count + strength), count


def sparse_pair_table(left, right, right_card, y, weights):
    keys = (
        np.asarray(left, dtype=np.int64) * np.int64(right_card)
        + np.asarray(right, dtype=np.int64)
    )
    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    unique_keys = sorted_keys[starts]
    counts = np.add.reduceat(np.asarray(weights, dtype=np.float64)[order], starts)
    positives = np.add.reduceat(
        (np.asarray(weights, dtype=np.float64) *
         np.asarray(y, dtype=np.float64))[order],
        starts,
    )
    return unique_keys, counts, positives


def sparse_pair_predict(table, left, right, right_card, right_prior, strength):
    table_keys, table_count, table_positive = table
    query = (
        np.asarray(left, dtype=np.int64) * np.int64(right_card)
        + np.asarray(right, dtype=np.int64)
    )
    positions = np.searchsorted(table_keys, query)
    positions_clip = np.minimum(positions, len(table_keys) - 1)
    found = (positions < len(table_keys)) & (
        table_keys[positions_clip] == query
    )

    count = np.zeros(len(query), dtype=np.float64)
    positive = np.zeros(len(query), dtype=np.float64)
    count[found] = table_count[positions_clip[found]]
    positive[found] = table_positive[positions_clip[found]]

    prior = np.asarray(right_prior, dtype=np.float64)
    posterior = (positive + strength * prior) / (count + strength)
    return safe_logit(posterior) - safe_logit(prior)


def entity_logit(rate, ids):
    return safe_logit(rate[np.asarray(ids, dtype=np.int64)])


def build_memory_scores(train, valid, test, weights):
    y = np.asarray(train.y, dtype=np.float64)
    prior = float(np.sum(weights * y) / np.sum(weights))

    entity_rates = {}
    for name, strength in [
        ("video_id", 24.0),
        ("author_id", 24.0),
        ("tag", 12.0),
        ("tab", 12.0),
        ("upload_type", 12.0),
    ]:
        rate, _ = aggregate_rate(
            train.X[name], y, weights,
            int(FEATURE_CARDINALITIES[name]), strength, prior,
        )
        entity_rates[name] = rate

    pair_specs = [
        ("user_id", "video_id", 7.0, 1.00),
        ("user_id", "author_id", 8.0, 0.85),
        ("user_id", "tag", 7.0, 0.45),
        ("user_id", "tab", 6.0, 0.40),
        ("user_id", "upload_type", 7.0, 0.25),
    ]

    tables = {}
    for left, right, strength, coefficient in pair_specs:
        tables[(left, right)] = sparse_pair_table(
            train.X[left], train.X[right],
            int(FEATURE_CARDINALITIES[right]), y, weights,
        )

    def score(split):
        s = (
            0.62 * entity_logit(entity_rates["video_id"], split.X["video_id"])
            + 0.38 * entity_logit(
                entity_rates["author_id"], split.X["author_id"]
            )
            + 0.12 * entity_logit(entity_rates["tag"], split.X["tag"])
            + 0.10 * entity_logit(entity_rates["tab"], split.X["tab"])
        )
        for left, right, strength, coefficient in pair_specs:
            right_ids = np.asarray(split.X[right], dtype=np.int64)
            right_prior = entity_rates[right][right_ids]
            deviation = sparse_pair_predict(
                tables[(left, right)],
                split.X[left],
                right_ids,
                int(FEATURE_CARDINALITIES[right]),
                right_prior,
                strength,
            )
            s += coefficient * deviation
        return np.asarray(s, dtype=np.float32)

    return score(valid), score(test), entity_rates


def previous_field_values(train, split, field):
    train_users = np.asarray(train.user_id, dtype=np.int64)
    train_values = np.asarray(train.X[field], dtype=np.int64)
    split_users = np.asarray(split.user_id, dtype=np.int64)
    split_values = np.asarray(split.X[field], dtype=np.int64)

    n_users = int(FEATURE_CARDINALITIES["user_id"])
    last_train = np.zeros(n_users, dtype=np.int64)

    train_rows = np.arange(len(train_users), dtype=np.int64)
    train_order = np.lexsort(
        (train_rows, np.asarray(train.time_ms), train_users)
    )
    sorted_train_users = train_users[train_order]
    train_last_mask = np.r_[
        sorted_train_users[1:] != sorted_train_users[:-1], True
    ]
    last_rows = train_order[train_last_mask]
    last_train[train_users[last_rows]] = train_values[last_rows]

    rows = np.arange(len(split_users), dtype=np.int64)
    order = np.lexsort((rows, np.asarray(split.time_ms), split_users))
    sorted_users = split_users[order]

    prev = np.empty(len(split_users), dtype=np.int64)
    is_first = np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    prev_sorted = np.empty(len(split_users), dtype=np.int64)
    prev_sorted[is_first] = last_train[sorted_users[is_first]]
    nonfirst = ~is_first
    prev_sorted[nonfirst] = split_values[order[np.flatnonzero(nonfirst) - 1]]
    prev[order] = prev_sorted
    return prev


def training_previous_values(train, field):
    users = np.asarray(train.user_id, dtype=np.int64)
    values = np.asarray(train.X[field], dtype=np.int64)
    rows = np.arange(len(users), dtype=np.int64)
    order = np.lexsort((rows, np.asarray(train.time_ms), users))
    sorted_users = users[order]

    prev_sorted = np.zeros(len(users), dtype=np.int64)
    nonfirst = np.r_[False, sorted_users[1:] == sorted_users[:-1]]
    prev_sorted[nonfirst] = values[order[np.flatnonzero(nonfirst) - 1]]

    prev = np.zeros(len(users), dtype=np.int64)
    prev[order] = prev_sorted
    return prev


def build_markov_scores(train, valid, test, weights, entity_rates):
    y = np.asarray(train.y, dtype=np.float64)
    relations = [
        ("author_id", "author_id", 15.0, 0.75),
        ("video_id", "author_id", 20.0, 0.45),
        ("tag", "tag", 10.0, 0.50),
        ("tab", "tag", 10.0, 0.25),
    ]

    train_prev = {}
    valid_prev = {}
    test_prev = {}
    for previous_field in sorted(set(x[0] for x in relations)):
        train_prev[previous_field] = training_previous_values(
            train, previous_field
        )
        valid_prev[previous_field] = previous_field_values(
            train, valid, previous_field
        )
        test_prev[previous_field] = previous_field_values(
            train, test, previous_field
        )

    tables = {}
    for previous_field, current_field, strength, coefficient in relations:
        tables[(previous_field, current_field)] = sparse_pair_table(
            train_prev[previous_field],
            train.X[current_field],
            int(FEATURE_CARDINALITIES[current_field]),
            y,
            weights,
        )

    def score(split, previous):
        result = (
            0.65 * entity_logit(
                entity_rates["video_id"], split.X["video_id"]
            )
            + 0.35 * entity_logit(
                entity_rates["author_id"], split.X["author_id"]
            )
        )
        for previous_field, current_field, strength, coefficient in relations:
            current_ids = np.asarray(split.X[current_field], dtype=np.int64)
            current_prior = entity_rates[current_field][current_ids]
            result += coefficient * sparse_pair_predict(
                tables[(previous_field, current_field)],
                previous[previous_field],
                current_ids,
                int(FEATURE_CARDINALITIES[current_field]),
                current_prior,
                strength,
            )
        return np.asarray(result, dtype=np.float32)

    return score(valid, valid_prev), score(test, test_prev)


def make_forest_matrix(split, split_name):
    columns = []
    for name in CAT_FIELDS:
        columns.append(np.asarray(split.X[name], dtype=np.float32))

    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))

    for entity in ("video_id", "author_id"):
        history = historical_features(split_name, key=entity)
        for suffix in HISTORY_KEYS:
            key = entity + "_" + suffix
            x = np.asarray(history[key], dtype=np.float32)
            columns.append(
                np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                .astype(np.float32)
            )
        del history

    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def build_random_forest_scores(train, valid, test, weights):
    x_train = make_forest_matrix(train, "train")
    x_valid = make_forest_matrix(valid, "valid")
    x_test = make_forest_matrix(test, "test")

    categorical_indices = list(range(len(CAT_FIELDS)))
    dataset = lgb.Dataset(
        x_train,
        label=np.asarray(train.y, dtype=np.float32),
        weight=weights,
        categorical_feature=categorical_indices,
        free_raw_data=True,
    )
    params = {
        "objective": "binary",
        "boosting_type": "rf",
        "learning_rate": 0.08,
        "num_leaves": 63,
        "max_depth": 10,
        "min_data_in_leaf": 180,
        "max_bin": 127,
        "feature_fraction": 0.72,
        "feature_fraction_bynode": 0.72,
        "bagging_fraction": 0.66,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 2.0,
        "cat_smooth": 20.0,
        "cat_l2": 12.0,
        "max_cat_to_onehot": 16,
        "seed": SEED,
        "bagging_seed": SEED + 1,
        "feature_fraction_seed": SEED + 2,
        "num_threads": min(8, max(1, os.cpu_count() or 1)),
        "verbose": -1,
    }
    model = lgb.train(params, dataset, num_boost_round=140)
    valid_scores = model.predict(x_valid).astype(np.float32)
    test_scores = model.predict(x_test).astype(np.float32)

    del x_train, x_valid, x_test, dataset, model
    gc.collect()
    return valid_scores, test_scores


def user_center_scale(scores, users):
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users, dtype=np.int64)
    _, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse)
    means = np.bincount(inverse, weights=scores) / np.maximum(counts, 1)
    centered = scores - means[inverse]
    scale = float(np.std(centered))
    if not np.isfinite(scale) or scale < 1e-10:
        scale = 1.0
    return centered / scale


train = load("train")
valid = load("valid")
test = load("test")
weights = recency_weights(train.date, half_life=5.0)

memory_valid, memory_test, entity_rates = build_memory_scores(
    train, valid, test, weights
)
markov_valid, markov_test = build_markov_scores(
    train, valid, test, weights, entity_rates
)
forest_valid, forest_test = build_random_forest_scores(
    train, valid, test, weights
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not (os.path.exists(inc_valid_path) and os.path.exists(inc_test_path)):
    raise FileNotFoundError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

families = {
    "hierarchical_memory": (memory_valid, memory_test),
    "causal_markov": (markov_valid, markov_test),
    "bagged_random_forest": (forest_valid, forest_test),
}

valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)

candidate_scores = {}
candidate_payloads = {}

inc_valid_norm = user_center_scale(inc_valid, valid_users)
inc_test_norm = user_center_scale(inc_test, test_users)

inc_metric = evaluate(valid_users, valid_y, inc_valid)
candidate_scores["trusted_incumbent"] = float(inc_metric["primary"])
candidate_payloads["trusted_incumbent"] = (
    inc_valid, inc_test, memory_valid, "hierarchical_memory", 0.0
)

for family_name, (own_valid, own_test) in families.items():
    metric = evaluate(valid_users, valid_y, own_valid)
    candidate_scores[family_name] = float(metric["primary"])
    candidate_payloads[family_name] = (
        own_valid, own_test, own_valid, family_name, 1.0
    )

    own_valid_norm = user_center_scale(own_valid, valid_users)
    own_test_norm = user_center_scale(own_test, test_users)

    for alpha in (0.25, 0.50, 0.75):
        blended_valid = (
            alpha * own_valid_norm + (1.0 - alpha) * inc_valid_norm
        )
        blended_test = (
            alpha * own_test_norm + (1.0 - alpha) * inc_test_norm
        )
        name = family_name + "_blend_" + str(alpha)
        metric = evaluate(valid_users, valid_y, blended_valid)
        candidate_scores[name] = float(metric["primary"])
        candidate_payloads[name] = (
            blended_valid, blended_test, own_valid, family_name, alpha
        )

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores, test_scores, raw_valid_scores, raw_family, alpha = (
    candidate_payloads[winner]
)
metrics = evaluate(valid_users, valid_y, valid_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS winner=%s raw_family=%s own_weight=%.2f incumbent_primary=%.6f"
    % (winner, raw_family, alpha, inc_metric["primary"])
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
    if alpha < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_valid_scores, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.4f}'
    % (
        metrics["primary"],
        metrics["gauc"],
        metrics["ndcg@5"],
        elapsed,
    )
)