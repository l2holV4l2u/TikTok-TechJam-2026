import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from sklearn.ensemble import ExtraTreesClassifier

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 918273
THREADS = max(1, min(12, os.cpu_count() or 1))
np.random.seed(SEED)

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tag", "tab", "hour",
    "duration_bucket", "upload_type", "music_type", "video_type",
    "user_active_degree", "is_live_streamer", "is_video_author",
    "is_lowactive_period",
    "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
    "onehot_feat0", "onehot_feat1", "onehot_feat2", "onehot_feat3",
    "onehot_feat4", "onehot_feat5", "onehot_feat6", "onehot_feat7",
    "onehot_feat8", "onehot_feat9", "onehot_feat10", "onehot_feat11",
    "onehot_feat12", "onehot_feat13", "onehot_feat14", "onehot_feat15",
    "onehot_feat16", "onehot_feat17",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]
RATE_FIELDS = [
    "video_id", "author_id", "tag", "duration_bucket", "upload_type",
    "music_type", "tab", "hour", "user_active_degree",
    "is_live_streamer", "is_video_author",
    "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
    "onehot_feat1", "onehot_feat2", "onehot_feat3",
    "onehot_feat7", "onehot_feat8",
]
RATE_SMOOTHING = {
    "video_id": 35.0,
    "author_id": 45.0,
    "tag": 100.0,
    "duration_bucket": 140.0,
    "upload_type": 120.0,
    "music_type": 140.0,
    "tab": 180.0,
    "hour": 200.0,
    "user_active_degree": 180.0,
    "is_live_streamer": 250.0,
    "is_video_author": 250.0,
    "follow_user_num_range": 180.0,
    "fans_user_num_range": 180.0,
    "friend_user_num_range": 180.0,
    "register_days_range": 180.0,
    "onehot_feat1": 180.0,
    "onehot_feat2": 130.0,
    "onehot_feat3": 55.0,
    "onehot_feat7": 90.0,
    "onehot_feat8": 70.0,
}


def recency_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.int32)
    if half_life is None:
        return np.ones(len(dates), dtype=np.float32)
    age = (int(dates.max()) - dates).astype(np.float64)
    age = np.maximum(age, 0.0)
    return np.power(0.5, age / float(half_life)).astype(np.float32)


def safe_logit(probability):
    probability = np.clip(
        np.asarray(probability, dtype=np.float64), 1e-5, 1.0 - 1e-5
    )
    return np.log(probability / (1.0 - probability))


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, user_ids))
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
    group_ends = np.flatnonzero(ends)
    sizes = np.diff(np.concatenate((np.array([-1]), group_ends)))
    row_sizes = np.repeat(sizes, sizes)

    positions = np.arange(n, dtype=np.int64) - first
    ranked = (positions.astype(np.float64) + 0.5) / row_sizes

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def get_history(split_name):
    result = {}
    for entity in ("video_id", "author_id"):
        values = historical_features(split_name, key=entity)
        for name, value in values.items():
            result[entity + "__" + name] = np.asarray(
                value, dtype=np.float32
            )
    return result


def make_gbdt_matrix(split, split_name, history_names=None):
    columns = []

    for field in CAT_FIELDS:
        columns.append(np.asarray(split.X[field], dtype=np.float32))

    for field in NUM_FIELDS:
        value = np.asarray(split.num[field], dtype=np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(value, 0.0)).astype(np.float32))

    hour = np.asarray(split.X["hour"], dtype=np.float32)
    columns.append(np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32))
    columns.append(np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32))

    history = get_history(split_name)
    if history_names is None:
        history_names = sorted(history)
    for name in history_names:
        value = np.asarray(history[name], dtype=np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(value.astype(np.float32))

    return np.column_stack(columns).astype(np.float32, copy=False), history_names


def fit_rate_tables(train, half_life=4.0):
    labels = np.asarray(train.y, dtype=np.float64)
    weights = recency_weights(train.date, half_life).astype(np.float64)
    prior = float(np.sum(weights * labels) / np.sum(weights))
    tables = {}

    for field in RATE_FIELDS:
        keys = np.asarray(train.X[field], dtype=np.int64)
        cardinality = int(keys.max()) + 1
        denominator = np.bincount(
            keys, weights=weights, minlength=cardinality
        )
        numerator = np.bincount(
            keys, weights=weights * labels, minlength=cardinality
        )
        smoothing = RATE_SMOOTHING[field]
        rates = (
            numerator + smoothing * prior
        ) / (
            denominator + smoothing
        )
        tables[field] = rates.astype(np.float32)

    return prior, tables


def rate_lookup(values, table, default):
    values = np.asarray(values, dtype=np.int64)
    output = np.full(len(values), default, dtype=np.float32)
    valid = (values >= 0) & (values < len(table))
    output[valid] = table[values[valid]]
    return output


def make_rate_matrix(split, prior, tables):
    columns = []
    logits = []

    for field in RATE_FIELDS:
        rate = rate_lookup(split.X[field], tables[field], prior)
        columns.append(rate)
        logits.append(safe_logit(rate).astype(np.float32))

    for field in NUM_FIELDS:
        value = np.asarray(split.num[field], dtype=np.float32)
        missing = ~np.isfinite(value)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(value, 0.0)).astype(np.float32))
        columns.append(missing.astype(np.float32))

    hour = np.asarray(split.X["hour"], dtype=np.float32)
    columns.append(np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32))
    columns.append(np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32))

    matrix = np.column_stack(columns).astype(np.float32, copy=False)

    # Stable additive empirical-Bayes family. High-cardinality entity
    # evidence gets greater weight, but every component is heavily shrunk.
    field_weights = np.asarray([
        1.45, 1.30, 0.95, 0.85, 0.65,
        0.45, 0.45, 0.35, 0.35, 0.25,
        0.25, 0.35, 0.35, 0.35, 0.35,
        0.35, 0.45, 0.90, 0.60, 0.70,
    ], dtype=np.float64)
    logit_matrix = np.column_stack(logits).astype(np.float64, copy=False)
    additive_score = (
        logit_matrix @ field_weights
    ) / float(np.sum(field_weights))

    return matrix, additive_score


def train_binary_gbdt(x_train, labels, weights):
    dataset = lgb.Dataset(
        x_train,
        label=labels,
        weight=weights,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )
    params = {
        "objective": "binary",
        "metric": "None",
        "learning_rate": 0.04,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 160,
        "feature_fraction": 0.84,
        "bagging_fraction": 0.86,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 3.0,
        "max_bin": 127,
        "max_cat_threshold": 32,
        "max_cat_to_onehot": 8,
        "verbose": -1,
        "num_threads": THREADS,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "data_random_seed": SEED + 3,
    }
    return lgb.train(params, dataset, num_boost_round=360)


train = load("train")
valid = load("valid")
test = load("test")
train_y = np.asarray(train.y, dtype=np.int8)

# Family 1: binary GBDT. The sample-weight sweep is on the main model itself,
# rather than on a side statistic that a later blend can suppress.
x_train, history_names = make_gbdt_matrix(train, "train")
x_valid, _ = make_gbdt_matrix(valid, "valid", history_names)
x_test, _ = make_gbdt_matrix(test, "test", history_names)

family_valid = {}
family_test = {}

for half_life, name in [
    (2.0, "binary_gbdt_hl2"),
    (4.0, "binary_gbdt_hl4"),
    (8.0, "binary_gbdt_hl8"),
    (None, "binary_gbdt_uniform"),
]:
    weights = recency_weights(train.date, half_life)
    model = train_binary_gbdt(x_train, train_y, weights)
    family_valid[name] = model.predict(
        x_valid, num_iteration=model.current_iteration()
    ).astype(np.float64)
    family_test[name] = model.predict(
        x_test, num_iteration=model.current_iteration()
    ).astype(np.float64)
    del model, weights
    gc.collect()

del x_train, x_valid, x_test
gc.collect()

# Families 2 and 3 share leakage-safe train-only marginal estimates but form
# predictions differently: randomized tree partitions versus additive odds.
prior, rate_tables = fit_rate_tables(train, half_life=4.0)
rate_train, additive_train = make_rate_matrix(train, prior, rate_tables)
rate_valid, additive_valid = make_rate_matrix(valid, prior, rate_tables)
rate_test, additive_test = make_rate_matrix(test, prior, rate_tables)

extra = ExtraTreesClassifier(
    n_estimators=180,
    criterion="entropy",
    max_depth=18,
    min_samples_leaf=90,
    max_features=0.72,
    bootstrap=False,
    class_weight=None,
    n_jobs=THREADS,
    random_state=SEED + 50,
)
extra.fit(
    rate_train,
    train_y,
    sample_weight=recency_weights(train.date, 4.0),
)
family_valid["extra_trees_rate_space"] = extra.predict_proba(
    rate_valid
)[:, 1].astype(np.float64)
family_test["extra_trees_rate_space"] = extra.predict_proba(
    rate_test
)[:, 1].astype(np.float64)

family_valid["additive_empirical_bayes"] = additive_valid.astype(np.float64)
family_test["additive_empirical_bayes"] = additive_test.astype(np.float64)

del extra, rate_train, rate_valid, rate_test
del additive_train, additive_valid, additive_test
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
    name: rank_percentile(valid.user_id, score)
    for name, score in family_valid.items()
}
family_test_rank = {
    name: rank_percentile(test.user_id, score)
    for name, score in family_test.items()
}

candidate_valid = {"incumbent": inc_valid}
candidate_test = {"incumbent": inc_test}
candidate_raw = {"incumbent": inc_valid}

for name in family_valid:
    candidate_valid[name + "_standalone"] = family_valid[name]
    candidate_test[name + "_standalone"] = family_test[name]
    candidate_raw[name + "_standalone"] = family_valid[name]

    for alpha in (0.08, 0.15, 0.25, 0.35, 0.50, 0.65):
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

# Cross-family rank aggregation is insensitive to incompatible probability
# calibration and rewards agreement while preserving complementary orderings.
gbdt_names = [
    "binary_gbdt_hl2",
    "binary_gbdt_hl4",
    "binary_gbdt_hl8",
    "binary_gbdt_uniform",
]
for gbdt_name in gbdt_names:
    fusion_valid = (
        0.60 * family_valid_rank[gbdt_name]
        + 0.25 * family_valid_rank["extra_trees_rate_space"]
        + 0.15 * family_valid_rank["additive_empirical_bayes"]
    )
    fusion_test = (
        0.60 * family_test_rank[gbdt_name]
        + 0.25 * family_test_rank["extra_trees_rate_space"]
        + 0.15 * family_test_rank["additive_empirical_bayes"]
    )
    fusion_name = gbdt_name + "_three_family_fusion"

    candidate_valid[fusion_name + "_standalone"] = fusion_valid
    candidate_test[fusion_name + "_standalone"] = fusion_test
    candidate_raw[fusion_name + "_standalone"] = fusion_valid

    for alpha in (0.15, 0.25, 0.35, 0.50, 0.65):
        key = f"{fusion_name}_incblend_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank + alpha * fusion_valid
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank + alpha * fusion_test
        )
        candidate_raw[key] = fusion_valid

candidate_metrics = {
    name: evaluate(valid.user_id, valid.y, score)
    for name, score in candidate_valid.items()
}
best_name = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"])
)
best_metrics = candidate_metrics[best_name]
best_valid = candidate_valid[best_name]
best_test = candidate_test[best_name]

standalone_summary = {
    name: float(
        candidate_metrics[name + "_standalone"]["primary"]
    )
    for name in family_valid
}
correlations = {
    name: float(np.corrcoef(inc_valid_rank, ranked)[0, 1])
    for name, ranked in family_valid_rank.items()
}

print("CANDIDATES " + json.dumps({
    name: float(metrics["primary"])
    for name, metrics in candidate_metrics.items()
}, sort_keys=True))
print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "standalone_primary": standalone_summary,
    "within_user_rank_correlation_with_incumbent": correlations,
    "recency_main_model_best": max(
        gbdt_names,
        key=lambda n: standalone_summary[n]
    ),
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