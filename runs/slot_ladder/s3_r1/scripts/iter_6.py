import os
import time
import json
import random
import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 271828
random.seed(SEED)
np.random.seed(SEED)

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
    "user_active_degree",
    "onehot_feat1",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "video_type",
]
NUMERIC = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]
HISTORY_SUFFIXES = (
    "train_count_log1p",
    "long_view_rate",
    "is_click_rate",
    "is_like_rate",
    "is_follow_rate",
    "is_hate_rate",
)

NUM_BOOST_ROUND = 150
THREADS = min(8, max(1, os.cpu_count() or 1))


def clean_float(values):
    values = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(values, nan=-1.0, posinf=20.0, neginf=-1.0)


def get_histories(split_name):
    result = {}
    for key in ("video_id", "author_id"):
        history = historical_features(split_name, key=key)
        for name, values in history.items():
            if any(name.endswith(suffix) for suffix in HISTORY_SUFFIXES):
                result[name] = clean_float(values)
    return result


def build_matrix(split, histories, selected_history_names=None):
    columns = [
        np.asarray(split.X[name], dtype=np.float32)
        for name in FIELDS
    ]

    for name in NUMERIC:
        values = np.asarray(split.num[name], dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=1e8, neginf=0.0)
        values = np.sign(values) * np.log1p(np.abs(values))
        columns.append(values.astype(np.float32))

    if selected_history_names is None:
        selected_history_names = sorted(histories.keys())

    for name in selected_history_names:
        columns.append(clean_float(histories[name]))

    matrix = np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)
    return matrix, selected_history_names


def group_sort_indices(user_ids):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[order]
    if sorted_users.size == 0:
        return order, np.empty(0, dtype=np.int32)
    boundaries = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1], True]
    )
    groups = np.diff(boundaries).astype(np.int32)
    return order, groups


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    if n == 0:
        return scores.copy()

    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.int64) - repeated_starts

    ranked_sorted = np.where(
        repeated_lengths > 1,
        positions / np.maximum(repeated_lengths - 1, 1),
        0.5,
    ).astype(np.float64)

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def sigmoid_np(values):
    values = np.clip(np.asarray(values, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def smoothed_rate(values, labels, cardinality, prior, strength):
    values = np.asarray(values, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    counts = np.bincount(values, minlength=cardinality).astype(np.float64)
    positives = np.bincount(
        values, weights=labels, minlength=cardinality
    ).astype(np.float64)
    return (positives + strength * prior) / (counts + strength)


def map_rate(values, table, prior):
    values = np.asarray(values, dtype=np.int64)
    output = np.full(values.shape[0], prior, dtype=np.float64)
    good = (values >= 0) & (values < table.shape[0])
    output[good] = table[values[good]]
    return output


def history_rate(histories, entity, n_rows, fallback):
    name = entity + "_long_view_rate"
    if name not in histories:
        return np.full(n_rows, fallback, dtype=np.float64)
    values = np.asarray(histories[name], dtype=np.float64)
    values = np.nan_to_num(values, nan=fallback, posinf=fallback, neginf=fallback)
    return np.clip(values, 1e-4, 1.0 - 1e-4)


train = load("train")
y_train = np.asarray(train.y, dtype=np.float32)
train_histories = get_histories("train")
x_train, HISTORY_NAMES = build_matrix(train, train_histories)

train_order, train_groups = group_sort_indices(train.user_id)
x_train_sorted = np.ascontiguousarray(x_train[train_order])
y_train_sorted = np.ascontiguousarray(y_train[train_order])
dates_sorted = np.asarray(train.date, dtype=np.int32)[train_order]

max_train_date = int(np.max(train.date))
unique_dates = np.sort(np.unique(np.asarray(train.date, dtype=np.int32)))
date_to_age = {
    int(date): int(len(unique_dates) - 1 - index)
    for index, date in enumerate(unique_dates)
}
age_sorted = np.fromiter(
    (date_to_age[int(date)] for date in dates_sorted),
    dtype=np.float32,
    count=dates_sorted.size,
)
recency_weights = np.exp2(-age_sorted / 4.0).astype(np.float32)
recency_weights /= np.mean(recency_weights)

categorical_indices = list(range(len(FIELDS)))

common_params = {
    "learning_rate": 0.055,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 2.0,
    "max_bin": 63,
    "min_gain_to_split": 1e-4,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "num_threads": THREADS,
    "verbose": -1,
}

models = {}

binary_dataset = lgb.Dataset(
    x_train_sorted,
    label=y_train_sorted,
    weight=recency_weights,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
binary_params = dict(common_params)
binary_params.update({
    "objective": "binary",
    "metric": "None",
})
models["gbdt_binary_recency"] = lgb.train(
    binary_params,
    binary_dataset,
    num_boost_round=NUM_BOOST_ROUND,
)

rank_uniform_dataset = lgb.Dataset(
    x_train_sorted,
    label=y_train_sorted,
    group=train_groups,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
rank_params = dict(common_params)
rank_params.update({
    "objective": "lambdarank",
    "metric": "None",
    "lambdarank_truncation_level": 8,
    "label_gain": [0, 1],
})
models["lambdarank_uniform"] = lgb.train(
    rank_params,
    rank_uniform_dataset,
    num_boost_round=NUM_BOOST_ROUND,
)

rank_recency_dataset = lgb.Dataset(
    x_train_sorted,
    label=y_train_sorted,
    weight=recency_weights,
    group=train_groups,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
models["lambdarank_recency4"] = lgb.train(
    rank_params,
    rank_recency_dataset,
    num_boost_round=NUM_BOOST_ROUND,
)

prior = float(np.mean(y_train))
eb_tables = {}
for field, strength in (
    ("video_id", 30.0),
    ("author_id", 60.0),
    ("tag", 150.0),
    ("duration_bucket", 200.0),
    ("tab", 300.0),
):
    cardinality = int(np.max(np.asarray(train.X[field], dtype=np.int64))) + 1
    eb_tables[field] = smoothed_rate(
        train.X[field], y_train, cardinality, prior, strength
    )


def empirical_bayes_scores(split, histories):
    n = len(split.user_id)
    video_rate = history_rate(histories, "video_id", n, prior)
    author_rate = history_rate(histories, "author_id", n, prior)

    tag_rate = map_rate(split.X["tag"], eb_tables["tag"], prior)
    duration_rate = map_rate(
        split.X["duration_bucket"], eb_tables["duration_bucket"], prior
    )
    tab_rate = map_rate(split.X["tab"], eb_tables["tab"], prior)

    def logit(rate):
        rate = np.clip(rate, 1e-4, 1.0 - 1e-4)
        return np.log(rate) - np.log1p(-rate)

    return (
        0.48 * logit(video_rate)
        + 0.28 * logit(author_rate)
        + 0.10 * logit(tag_rate)
        + 0.08 * logit(duration_rate)
        + 0.06 * logit(tab_rate)
    )


del x_train, x_train_sorted, binary_dataset
del rank_uniform_dataset, rank_recency_dataset
del train_histories, y_train_sorted, dates_sorted, age_sorted
del recency_weights, train_order, train_groups

valid = load("valid")
valid_histories = get_histories("valid")
x_valid, _ = build_matrix(valid, valid_histories, HISTORY_NAMES)

raw_valid = {}
for name, model in models.items():
    raw_valid[name] = np.asarray(model.predict(x_valid), dtype=np.float64)
raw_valid["empirical_bayes"] = empirical_bayes_scores(valid, valid_histories)

shared_dir = os.environ.get("SHARED_ARTIFACTS")
inc_valid = None
inc_test_path = None
if shared_dir:
    valid_path = os.path.join(shared_dir, "incumbent_valid_scores.npy")
    test_path = os.path.join(shared_dir, "incumbent_test_scores.npy")
    if os.path.exists(valid_path) and os.path.exists(test_path):
        inc_valid = np.asarray(np.load(valid_path), dtype=np.float64)
        inc_test_path = test_path

candidate_scores = {}
candidate_specs = {}
candidate_arrays = {}

for name, prediction in raw_valid.items():
    candidate_arrays[name] = prediction
    candidate_specs[name] = {
        "family": name,
        "blend": False,
        "alpha": 1.0,
    }
    candidate_scores[name] = float(
        evaluate(valid.user_id, valid.y, prediction)["primary"]
    )

if inc_valid is not None:
    incumbent_rank = within_user_rank(valid.user_id, inc_valid)
    for name, prediction in raw_valid.items():
        model_rank = within_user_rank(valid.user_id, prediction)
        for alpha in (0.25, 0.50, 0.75):
            candidate_name = f"{name}_rankblend_{alpha:.2f}"
            blended = alpha * model_rank + (1.0 - alpha) * incumbent_rank
            candidate_arrays[candidate_name] = blended
            candidate_specs[candidate_name] = {
                "family": name,
                "blend": True,
                "alpha": alpha,
            }
            candidate_scores[candidate_name] = float(
                evaluate(valid.user_id, valid.y, blended)["primary"]
            )

winner_name = max(candidate_scores, key=candidate_scores.get)
winner_spec = candidate_specs[winner_name]
valid_scores = candidate_arrays[winner_name]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print(
    "FINDINGS "
    + json.dumps({
        "history_features": len(HISTORY_NAMES),
        "train_last_date": max_train_date,
        "winner": winner_name,
        "rank_blending": inc_valid is not None,
    }, sort_keys=True)
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if winner_spec["blend"]:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(raw_valid[winner_spec["family"]], dtype=np.float64),
        )

del x_valid, valid_histories, candidate_arrays, valid_scores

test = load("test")
test_histories = get_histories("test")
x_test, _ = build_matrix(test, test_histories, HISTORY_NAMES)

family = winner_spec["family"]
if family == "empirical_bayes":
    test_raw = empirical_bayes_scores(test, test_histories)
else:
    test_raw = np.asarray(models[family].predict(x_test), dtype=np.float64)

if winner_spec["blend"]:
    incumbent_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
    model_rank = within_user_rank(test.user_id, test_raw)
    incumbent_rank = within_user_rank(test.user_id, incumbent_test)
    alpha = float(winner_spec["alpha"])
    test_scores = alpha * model_rank + (1.0 - alpha) * incumbent_rank
else:
    test_scores = test_raw

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
result = {
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result))