import os
import time
import json
import gc
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
HALF_LIFE = 10.0

SINGLE_FIELDS = [
    "video_id", "author_id", "tag", "duration_bucket", "tab",
    "upload_type", "music_type", "hour", "video_type",
    "onehot_feat3", "onehot_feat7", "onehot_feat8",
]

PAIR_FIELDS = [
    ("user_id", "author_id"),
    ("user_id", "tag"),
    ("user_id", "duration_bucket"),
    ("user_id", "upload_type"),
    ("user_id", "music_type"),
    ("user_id", "onehot_feat8"),
    ("tab", "tag"),
    ("tab", "author_id"),
    ("tag", "duration_bucket"),
    ("author_id", "duration_bucket"),
    ("onehot_feat3", "tag"),
    ("onehot_feat8", "duration_bucket"),
]

TREE_LEAVES = [
    ("user_id", "author_id"),
    ("user_id", "tag"),
    ("user_id", "duration_bucket"),
    ("user_id", "onehot_feat8"),
    ("tab", "author_id", "duration_bucket"),
    ("tab", "tag", "duration_bucket"),
    ("author_id", "tag"),
    ("author_id", "upload_type"),
    ("tag", "music_type"),
    ("onehot_feat3", "tag", "duration_bucket"),
    ("onehot_feat7", "tab", "tag"),
    ("onehot_feat8", "tab", "duration_bucket"),
    ("video_id", "tab"),
    ("video_id", "hour"),
]

BOOST_STAGES = [
    ("video_id",),
    ("author_id",),
    ("tag",),
    ("duration_bucket",),
    ("tab",),
    ("user_id", "author_id"),
    ("user_id", "tag"),
    ("user_id", "duration_bucket"),
    ("tab", "author_id"),
    ("tab", "tag"),
    ("author_id", "duration_bucket"),
    ("onehot_feat8", "duration_bucket"),
]


def split_arrays(split, include_y):
    result = {
        "user_id": np.asarray(split.user_id, dtype=np.int64),
        "date": np.asarray(split.date, dtype=np.int64),
    }
    needed = set(SINGLE_FIELDS)
    for fields in PAIR_FIELDS + TREE_LEAVES + BOOST_STAGES:
        needed.update(fields)
    needed.discard("user_id")

    for field in needed:
        result[field] = np.asarray(split.X[field], dtype=np.int64)

    if include_y:
        result["y"] = np.asarray(split.y, dtype=np.float64)
    return result


def combine_arrays(a, b):
    result = {
        "user_id": np.concatenate([
            np.asarray(a.user_id, dtype=np.int64),
            np.asarray(b.user_id, dtype=np.int64),
        ]),
        "date": np.concatenate([
            np.asarray(a.date, dtype=np.int64),
            np.asarray(b.date, dtype=np.int64),
        ]),
        "y": np.concatenate([
            np.asarray(a.y, dtype=np.float64),
            np.asarray(b.y, dtype=np.float64),
        ]),
    }

    needed = set(SINGLE_FIELDS)
    for fields in PAIR_FIELDS + TREE_LEAVES + BOOST_STAGES:
        needed.update(fields)
    needed.discard("user_id")

    for field in needed:
        result[field] = np.concatenate([
            np.asarray(a.X[field], dtype=np.int64),
            np.asarray(b.X[field], dtype=np.int64),
        ])
    return result


def temporal_weights(dates):
    dates = np.asarray(dates, dtype=np.int64)
    day = dates % 100
    age = int(day.max()) - day
    weights = np.exp2(-age.astype(np.float64) / HALF_LIFE)
    return weights / max(float(weights.mean()), 1e-12)


def safe_logit(probability):
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def make_key(data, fields):
    first = fields[0]
    key = np.asarray(data[first], dtype=np.uint64).copy()
    for field in fields[1:]:
        cardinality = np.uint64(int(FEATURE_CARDINALITIES[field]))
        key = key * cardinality + np.asarray(data[field], dtype=np.uint64)
    return key


def grouped_lookup(source_key, query_key, target, weights, smoothing, prior):
    source_key = np.asarray(source_key, dtype=np.uint64)
    query_key = np.asarray(query_key, dtype=np.uint64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    order = np.argsort(source_key, kind="mergesort")
    sorted_key = source_key[order]
    sorted_weight = weights[order]
    sorted_target = (weights * target)[order]

    starts = np.r_[0, 1 + np.flatnonzero(sorted_key[1:] != sorted_key[:-1])]
    unique_key = sorted_key[starts]
    denominator = np.add.reduceat(sorted_weight, starts)
    numerator = np.add.reduceat(sorted_target, starts)
    values = (numerator + smoothing * prior) / (denominator + smoothing)

    positions = np.searchsorted(unique_key, query_key)
    found = positions < len(unique_key)
    clipped = np.minimum(positions, len(unique_key) - 1)
    found &= unique_key[clipped] == query_key

    prediction = np.full(len(query_key), prior, dtype=np.float64)
    prediction[found] = values[clipped[found]]

    del order, sorted_key, sorted_weight, sorted_target
    del starts, unique_key, denominator, numerator, values
    return prediction


def grouped_effect(source_key, query_key, residual, weights, smoothing):
    source_key = np.asarray(source_key, dtype=np.uint64)
    query_key = np.asarray(query_key, dtype=np.uint64)
    residual = np.asarray(residual, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    order = np.argsort(source_key, kind="mergesort")
    sorted_key = source_key[order]
    sorted_weight = weights[order]
    sorted_residual = (weights * residual)[order]

    starts = np.r_[0, 1 + np.flatnonzero(sorted_key[1:] != sorted_key[:-1])]
    unique_key = sorted_key[starts]
    denominator = np.add.reduceat(sorted_weight, starts)
    numerator = np.add.reduceat(sorted_residual, starts)
    effects = numerator / (denominator + smoothing)

    source_effect = np.empty(len(source_key), dtype=np.float64)
    inverse_order = np.empty(len(source_key), dtype=np.int64)
    inverse_order[order] = np.arange(len(order), dtype=np.int64)
    group_index_sorted = np.repeat(
        np.arange(len(starts), dtype=np.int64),
        np.diff(np.r_[starts, len(source_key)])
    )
    source_effect[:] = effects[group_index_sorted[inverse_order]]

    positions = np.searchsorted(unique_key, query_key)
    found = positions < len(unique_key)
    clipped = np.minimum(positions, len(unique_key) - 1)
    found &= unique_key[clipped] == query_key
    query_effect = np.zeros(len(query_key), dtype=np.float64)
    query_effect[found] = effects[clipped[found]]

    del order, sorted_key, sorted_weight, sorted_residual
    del starts, unique_key, denominator, numerator, effects
    del inverse_order, group_index_sorted
    return source_effect, query_effect


def empirical_bayes_model(source, query):
    y = source["y"]
    weights = temporal_weights(source["date"])
    prior = float(np.sum(weights * y) / np.sum(weights))

    coefficients = {
        "video_id": (1.8, 35.0),
        "author_id": (1.1, 50.0),
        "tag": (0.75, 90.0),
        "duration_bucket": (0.45, 100.0),
        "tab": (0.35, 160.0),
        "upload_type": (0.25, 120.0),
        "music_type": (0.20, 120.0),
        "hour": (0.15, 180.0),
        "video_type": (0.10, 180.0),
        "onehot_feat3": (0.35, 90.0),
        "onehot_feat7": (0.20, 100.0),
        "onehot_feat8": (0.45, 75.0),
    }

    score = np.zeros(len(query["user_id"]), dtype=np.float64)
    for field, (coefficient, smoothing) in coefficients.items():
        rate = grouped_lookup(
            make_key(source, (field,)),
            make_key(query, (field,)),
            y, weights, smoothing, prior
        )
        score += coefficient * safe_logit(rate)
    return score.astype(np.float32)


def pair_additive_model(source, query):
    y = source["y"]
    weights = temporal_weights(source["date"])
    prior = float(np.sum(weights * y) / np.sum(weights))

    score = empirical_bayes_model(source, query).astype(np.float64)
    pair_weights = [
        0.80, 0.65, 0.45, 0.30, 0.25, 0.25,
        0.25, 0.20, 0.20, 0.20, 0.15, 0.15,
    ]
    pair_smoothing = [
        10.0, 12.0, 16.0, 16.0, 18.0, 18.0,
        30.0, 30.0, 35.0, 35.0, 40.0, 40.0,
    ]

    baseline = safe_logit(prior)
    for fields, coefficient, smoothing in zip(
        PAIR_FIELDS, pair_weights, pair_smoothing
    ):
        rate = grouped_lookup(
            make_key(source, fields),
            make_key(query, fields),
            y, weights, smoothing, prior
        )
        score += coefficient * (safe_logit(rate) - baseline)
    return score.astype(np.float32)


def randomized_tree_bagging(source, query):
    y = source["y"]
    weights = temporal_weights(source["date"])
    prior = float(np.sum(weights * y) / np.sum(weights))

    leaf_scores = []
    for fields in TREE_LEAVES:
        depth = len(fields)
        smoothing = 12.0 if depth == 2 and fields[0] == "user_id" else (
            25.0 if depth == 2 else 45.0
        )
        rate = grouped_lookup(
            make_key(source, fields),
            make_key(query, fields),
            y, weights, smoothing, prior
        )
        leaf_scores.append(safe_logit(rate))

    stacked = np.vstack(leaf_scores)
    # Median and trimmed mean make this a bagged partition estimator rather
    # than allowing one sparse conjunction to dominate.
    median = np.median(stacked, axis=0)
    mean = np.mean(stacked, axis=0)
    del stacked, leaf_scores
    return (0.55 * median + 0.45 * mean).astype(np.float32)


def residual_boosting(source, query):
    y = source["y"]
    weights = temporal_weights(source["date"])
    prior = float(np.sum(weights * y) / np.sum(weights))
    base = float(safe_logit(prior))

    source_score = np.full(len(y), base, dtype=np.float64)
    query_score = np.full(len(query["user_id"]), base, dtype=np.float64)

    learning_rate = 0.65
    for stage, fields in enumerate(BOOST_STAGES):
        residual = y - sigmoid(source_score)
        depth = len(fields)
        smoothing = 35.0 if depth == 1 else (
            14.0 if fields[0] == "user_id" else 30.0
        )
        source_effect, query_effect = grouped_effect(
            make_key(source, fields),
            make_key(query, fields),
            residual, weights, smoothing
        )
        source_score += learning_rate * source_effect
        query_score += learning_rate * query_effect
        del residual, source_effect, query_effect

    return query_score.astype(np.float32)


MODEL_FUNCTIONS = {
    "empirical_bayes": empirical_bayes_model,
    "pair_additive": pair_additive_model,
    "random_tree_bagging": randomized_tree_bagging,
    "residual_boosting": residual_boosting,
}


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]

    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.maximum.accumulate(
        np.where(new_group, np.arange(n, dtype=np.int64), 0)
    )
    ranks = np.arange(n, dtype=np.int64) - starts

    _, counts = np.unique(sorted_users, return_counts=True)
    sizes = np.repeat(counts, counts)
    normalized = ranks.astype(np.float64) / np.maximum(sizes - 1, 1)
    normalized[sizes == 1] = 0.5

    result = np.empty(n, dtype=np.float64)
    result[order] = normalized
    return result


train = load("train")
valid = load("valid")
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64, copy=False)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

train_data = split_arrays(train, include_y=True)
valid_data = split_arrays(valid, include_y=False)

inc_rank = within_user_rank(valid.user_id, inc_valid)
candidate_scores = {}
candidate_metrics = {}
candidate_recipe = {}

inc_metric = evaluate(valid.user_id, y_valid, inc_valid)
candidate_scores["trusted_incumbent"] = inc_valid
candidate_metrics["trusted_incumbent"] = float(inc_metric["primary"])
candidate_recipe["trusted_incumbent"] = ("incumbent", 0.0)

blend_alphas = [0.15, 0.30, 0.50, 0.70]

raw_predictions = {}
for model_name, model_function in MODEL_FUNCTIONS.items():
    prediction = model_function(train_data, valid_data).astype(np.float64)
    raw_predictions[model_name] = prediction

    raw_metric = evaluate(valid.user_id, y_valid, prediction)
    raw_name = model_name + "_raw"
    candidate_scores[raw_name] = prediction
    candidate_metrics[raw_name] = float(raw_metric["primary"])
    candidate_recipe[raw_name] = (model_name, 1.0)

    model_rank = within_user_rank(valid.user_id, prediction)
    for alpha in blend_alphas:
        blended = (1.0 - alpha) * inc_rank + alpha * model_rank
        name = "%s_blend_%02d" % (model_name, int(round(alpha * 100)))
        metric = evaluate(valid.user_id, y_valid, blended)
        candidate_scores[name] = blended
        candidate_metrics[name] = float(metric["primary"])
        candidate_recipe[name] = (model_name, alpha)

    changed = float(np.mean(np.abs(model_rank - inc_rank) > 1e-12))
    print(
        "FINDINGS %s changes_within_user_rank_fraction=%.6f raw_primary=%.6f"
        % (model_name, changed, float(raw_metric["primary"]))
    )
    gc.collect()

best_name = max(candidate_metrics, key=candidate_metrics.get)
valid_scores = np.asarray(candidate_scores[best_name], dtype=np.float64)
best_model_name, best_alpha = candidate_recipe[best_name]
metrics = evaluate(valid.user_id, y_valid, valid_scores)

print("CANDIDATES " + json.dumps(
    {k: round(v, 7) for k, v in sorted(candidate_metrics.items())},
    sort_keys=True
))
print(
    "FINDINGS selected=%s model=%s candidate_weight=%.2f improvement_vs_incumbent=%.6f"
    % (
        best_name, best_model_name, best_alpha,
        float(metrics["primary"]) - float(inc_metric["primary"])
    )
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64)
    )

test = load("test")
inc_test = np.load(inc_test_path).astype(np.float64, copy=False)

if best_model_name == "incumbent" or best_alpha == 0.0:
    test_scores = inc_test
else:
    combined_data = combine_arrays(train, valid)
    test_data = split_arrays(test, include_y=False)
    selected_function = MODEL_FUNCTIONS[best_model_name]
    model_test = selected_function(combined_data, test_data).astype(np.float64)

    if best_alpha >= 1.0:
        test_scores = model_test
    else:
        test_scores = (
            (1.0 - best_alpha) * within_user_rank(test.user_id, inc_test)
            + best_alpha * within_user_rank(test.user_id, model_test)
        )

    del combined_data, test_data, model_test
    gc.collect()

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))