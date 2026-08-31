import os
import time
import json
import gc

import numpy as np
import lightgbm as lgb
from sklearn.linear_model import SGDClassifier

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
np.random.seed(SEED)

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag", "upload_type",
    "onehot_feat3", "onehot_feat8", "duration_bucket", "onehot_feat1",
    "music_type", "onehot_feat7", "user_active_degree",
    "register_days_bucket", "register_days_range", "onehot_feat0",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
HISTORY_SUFFIXES = [
    "train_count_log1p", "long_view_rate", "is_click_rate",
    "is_like_rate", "is_follow_rate", "is_comment_rate",
    "is_forward_rate", "is_hate_rate", "is_profile_enter_rate",
    "play_time_ms_logmean",
]


def safe_logit(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(x) - np.log1p(-x)


def recency_weights(dates, half_life=5.0):
    dates = np.asarray(dates, dtype=np.int64)
    age = dates.max() - dates
    weights = np.power(0.5, age.astype(np.float64) / half_life)
    weights /= max(float(weights.mean()), 1e-12)
    return weights.astype(np.float32)


def build_category_tables(train, weights):
    y = np.asarray(train.y, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    prior = float(np.sum(w * y) / np.sum(w))
    tables = {}

    for field in CAT_FIELDS:
        ids = np.asarray(train.X[field], dtype=np.int64)
        cardinality = int(FEATURE_CARDINALITIES[field])
        count = np.bincount(
            ids, weights=w, minlength=cardinality
        ).astype(np.float64)
        positive = np.bincount(
            ids, weights=w * y, minlength=cardinality
        ).astype(np.float64)
        tables[field] = (count, positive)

    return prior, tables


def build_matrix_and_nb(split, split_name, train, weights, prior, tables):
    is_train = split_name == "train"
    columns = []
    nb_effects = []

    if is_train:
        train_y = np.asarray(train.y, dtype=np.float64)
        row_weights = np.asarray(weights, dtype=np.float64)
    else:
        train_y = None
        row_weights = None

    prior_logit = float(safe_logit(prior))

    for field in CAT_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        counts, positives = tables[field]

        row_count = counts[ids].copy()
        row_positive = positives[ids].copy()

        if is_train:
            row_count -= row_weights
            row_positive -= row_weights * train_y
            np.maximum(row_count, 0.0, out=row_count)
            np.maximum(row_positive, 0.0, out=row_positive)

        # Stronger smoothing for identities and weaker smoothing for
        # low-cardinality context fields.
        if field == "user_id":
            smoothing = 35.0
        elif field in ("video_id", "author_id"):
            smoothing = 25.0
        else:
            smoothing = 18.0

        posterior = (
            row_positive + smoothing * prior
        ) / (row_count + smoothing)
        effect = safe_logit(posterior) - prior_logit
        confidence = np.sqrt(
            row_count / np.maximum(row_count + smoothing, 1e-12)
        )
        nb_effects.append((effect * confidence).astype(np.float32))

        columns.append(effect.astype(np.float32))
        columns.append(np.log1p(row_count).astype(np.float32))

    for field in NUM_FIELDS:
        values = np.asarray(split.num[field], dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(values, 0.0)).astype(np.float32))

    for entity in ("video_id", "author_id"):
        histories = historical_features(split_name, key=entity)
        for suffix in HISTORY_SUFFIXES:
            key = entity + "_" + suffix
            values = np.asarray(histories[key], dtype=np.float32)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            if suffix.endswith("_rate"):
                values = np.asarray(
                    safe_logit(values) - prior_logit, dtype=np.float32
                )
            columns.append(values)
        del histories

    hour = np.asarray(split.X["hour"], dtype=np.float32)
    angle = 2.0 * np.pi * hour / 24.0
    columns.append(np.sin(angle).astype(np.float32))
    columns.append(np.cos(angle).astype(np.float32))

    base_date = int(np.min(np.asarray(train.date, dtype=np.int64)))
    date_offset = (
        np.asarray(split.date, dtype=np.int64) - base_date
    ).astype(np.float32)
    columns.append((date_offset / 10.0).astype(np.float32))

    matrix = np.ascontiguousarray(
        np.column_stack(columns), dtype=np.float32
    )

    # A weighted categorical Naive-Bayes-style evidence sum. Identity
    # effects are useful but deliberately prevented from dominating.
    field_weights = np.ones(len(CAT_FIELDS), dtype=np.float32)
    field_weights[CAT_FIELDS.index("user_id")] = 0.55
    field_weights[CAT_FIELDS.index("video_id")] = 1.25
    field_weights[CAT_FIELDS.index("author_id")] = 1.10
    nb_score = np.average(
        np.column_stack(nb_effects),
        axis=1,
        weights=field_weights,
    ).astype(np.float32)

    return matrix, nb_score


def standardize_from_train(x_train, x_valid, x_test):
    mean = np.mean(x_train, axis=0, dtype=np.float64)
    std = np.std(x_train, axis=0, dtype=np.float64)
    std[~np.isfinite(std) | (std < 1e-5)] = 1.0
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    result = []
    for x in (x_train, x_valid, x_test):
        z = np.asarray((x - mean) / std, dtype=np.float32)
        np.clip(z, -8.0, 8.0, out=z)
        result.append(np.ascontiguousarray(z))
    return tuple(result)


def within_user_rank(scores, users):
    """Return split-local ordinal percentiles; only within-user order matters."""
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users, dtype=np.int64)
    n = len(scores)

    order = np.lexsort((np.arange(n, dtype=np.int64), scores, users))
    sorted_users = users[order]

    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = sorted_users[1:] != sorted_users[:-1]

    positions = np.arange(n, dtype=np.int64)
    starts = np.maximum.accumulate(np.where(new_group, positions, 0))

    end_flag = np.empty(n, dtype=bool)
    end_flag[-1] = True
    end_flag[:-1] = sorted_users[:-1] != sorted_users[1:]
    ends = np.minimum.accumulate(
        np.where(end_flag, positions, n - 1)[::-1]
    )[::-1]

    denominator = ends - starts
    ranked_sorted = np.where(
        denominator > 0,
        (positions - starts) / np.maximum(denominator, 1),
        0.5,
    ).astype(np.float64)

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def borda(scores_list, users, weights=None):
    if weights is None:
        weights = np.ones(len(scores_list), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()

    result = np.zeros(len(users), dtype=np.float64)
    for weight, scores in zip(weights, scores_list):
        result += weight * within_user_rank(scores, users)
    return result


train = load("train")
valid = load("valid")
test = load("test")

train_y = np.asarray(train.y, dtype=np.int8)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)

weights = recency_weights(train.date, half_life=5.0)
prior, category_tables = build_category_tables(train, weights)

x_train, nb_train = build_matrix_and_nb(
    train, "train", train, weights, prior, category_tables
)
x_valid, nb_valid = build_matrix_and_nb(
    valid, "valid", train, weights, prior, category_tables
)
x_test, nb_test = build_matrix_and_nb(
    test, "test", train, weights, prior, category_tables
)

del category_tables, nb_train
gc.collect()

x_train_std, x_valid_std, x_test_std = standardize_from_train(
    x_train, x_valid, x_test
)

# Family 1: recency-weighted linear discriminative classifier.
linear = SGDClassifier(
    loss="log_loss",
    penalty="elasticnet",
    alpha=2.5e-5,
    l1_ratio=0.06,
    max_iter=14,
    tol=1e-4,
    shuffle=True,
    random_state=SEED,
    average=True,
)
linear.fit(x_train_std, train_y, sample_weight=weights)
linear_valid = linear.decision_function(x_valid_std).astype(np.float32)
linear_test = linear.decision_function(x_test_std).astype(np.float32)
del linear
gc.collect()

# The forest does not need standardized values.
del x_train_std, x_valid_std, x_test_std
gc.collect()

# Family 2: random-subspace bagged trees. Unlike boosted trees, trees are
# independently bagged and averaged, targeting variance and unstable splits.
forest_data = lgb.Dataset(
    x_train,
    label=train_y,
    weight=weights,
    free_raw_data=False,
)
forest_params = {
    "objective": "binary",
    "metric": "None",
    "boosting_type": "rf",
    "num_leaves": 63,
    "max_depth": 10,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.62,
    "bagging_fraction": 0.68,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 2.0,
    "learning_rate": 1.0,
    "max_bin": 127,
    "num_threads": min(8, max(1, os.cpu_count() or 1)),
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "verbose": -1,
}
forest = lgb.train(
    forest_params,
    forest_data,
    num_boost_round=150,
)
forest_valid = forest.predict(x_valid).astype(np.float32)
forest_test = forest.predict(x_test).astype(np.float32)

del forest, forest_data, x_train, x_valid, x_test
gc.collect()

own_valid = {
    "linear_discriminative": linear_valid,
    "categorical_naive_bayes": nb_valid,
    "random_subspace_forest": forest_valid,
}
own_test = {
    "linear_discriminative": linear_test,
    "categorical_naive_bayes": nb_test,
    "random_subspace_forest": forest_test,
}

# Family 3: rank aggregation of structurally diverse own models.
own_aggregate_valid = borda(
    [linear_valid, nb_valid, forest_valid],
    valid_users,
    weights=[0.40, 0.20, 0.40],
)
own_aggregate_test = borda(
    [linear_test, nb_test, forest_test],
    test_users,
    weights=[0.40, 0.20, 0.40],
)
own_valid["own_borda_aggregate"] = own_aggregate_valid
own_test["own_borda_aggregate"] = own_aggregate_test

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

inc_valid = np.load(inc_valid_path).astype(np.float64)
inc_test = np.load(inc_test_path).astype(np.float64)

candidate_scores = {}
candidate_payload = {}

# Score all standalone new families.
for name in own_valid:
    metric = evaluate(valid_users, valid_y, own_valid[name])
    candidate_scores[name] = float(metric["primary"])
    candidate_payload[name] = (
        np.asarray(own_valid[name], dtype=np.float64),
        np.asarray(own_test[name], dtype=np.float64),
        np.asarray(own_valid[name], dtype=np.float64),
        False,
    )

# Score Borda blends with the incumbent. The same fixed weight is applied
# to test, and no test labels or test-based selection are used.
blend_weights = [0.20, 0.35, 0.50, 0.65, 0.80]
for name in own_valid:
    for alpha in blend_weights:
        blended_valid = borda(
            [inc_valid, own_valid[name]],
            valid_users,
            weights=[1.0 - alpha, alpha],
        )
        blended_test = borda(
            [inc_test, own_test[name]],
            test_users,
            weights=[1.0 - alpha, alpha],
        )
        candidate_name = "%s_inc_borda_%.2f" % (name, alpha)
        metric = evaluate(valid_users, valid_y, blended_valid)
        candidate_scores[candidate_name] = float(metric["primary"])
        candidate_payload[candidate_name] = (
            blended_valid,
            blended_test,
            np.asarray(own_valid[name], dtype=np.float64),
            True,
        )

# Include the unchanged incumbent so an unhelpful breadth sweep cannot
# overwrite the trusted result.
inc_metric = evaluate(valid_users, valid_y, inc_valid)
candidate_scores["trusted_incumbent"] = float(inc_metric["primary"])
candidate_payload["trusted_incumbent"] = (
    inc_valid,
    inc_test,
    own_aggregate_valid,
    True,
)

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores, test_scores, raw_valid_scores, uses_incumbent = (
    candidate_payload[winner]
)
final_metric = evaluate(valid_users, valid_y, valid_scores)

print("FINDINGS winner=%s best_new=%s" % (
    winner,
    max(own_valid, key=lambda n: candidate_scores[n]),
))
print("CANDIDATES " + json.dumps(
    {k: round(v, 6) for k, v in candidate_scores.items()},
    sort_keys=True,
))

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
    if uses_incumbent:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_valid_scores, dtype=np.float64),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(final_metric["primary"]),
    "gauc": float(final_metric["gauc"]),
    "ndcg@5": float(final_metric["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))