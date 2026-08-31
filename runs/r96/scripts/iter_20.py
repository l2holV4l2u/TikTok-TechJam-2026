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
SEED = 27183
THREADS = min(16, os.cpu_count() or 1)

CAT_FIELDS = list(FEATURE_CARDINALITIES.keys())
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]
NB_FIELDS = [
    "video_id", "author_id", "tag", "duration_bucket", "tab",
    "upload_type", "music_type", "video_type", "onehot_feat3",
    "onehot_feat8", "hour", "is_video_author",
]
HALF_LIFE = 4.0


def recency_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.int32)
    age = dates.max() - dates
    weights = np.power(
        0.5, age.astype(np.float32) / float(half_life)
    )
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores.copy()

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    ordered_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = ordered_users[1:] != ordered_users[:-1]
    start_index = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = ordered_users[:-1] != ordered_users[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(np.concatenate((
        np.asarray([-1], dtype=np.int64),
        end_positions,
    )))
    row_sizes = np.repeat(sizes, sizes)

    positions = np.arange(n, dtype=np.int64) - start_index
    ordered_ranks = (
        positions.astype(np.float64) + 0.5
    ) / row_sizes.astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[order] = ordered_ranks
    return result


def train_frequency_features(train, query_splits):
    train_features = []
    query_features = [[] for _ in query_splits]

    for field in CAT_FIELDS:
        card = int(FEATURE_CARDINALITIES[field])
        train_ids = np.asarray(train.X[field], dtype=np.int64)
        counts = np.bincount(train_ids, minlength=card).astype(np.float32)
        log_counts = np.log1p(counts)

        train_features.append(log_counts[train_ids])
        for j, split in enumerate(query_splits):
            ids = np.asarray(split.X[field], dtype=np.int64)
            safe = np.minimum(ids, card - 1)
            values = log_counts[safe]
            values = np.where(
                (ids >= 0) & (ids < card), values, 0.0
            )
            query_features[j].append(values.astype(np.float32))

    return train_features, query_features


def load_history_columns(split_name):
    columns = []
    names = []
    for entity in ("video_id", "author_id"):
        history = historical_features(split_name, key=entity)
        for name in sorted(history.keys()):
            value = np.asarray(history[name], dtype=np.float32)
            value = np.nan_to_num(
                value, nan=0.0, posinf=0.0, neginf=0.0
            )
            columns.append(value)
            names.append(name)
    return columns, names


def make_matrix(split, frequency_columns, history_columns):
    columns = []
    categorical_indices = []

    for field in CAT_FIELDS:
        categorical_indices.append(len(columns))
        columns.append(
            np.asarray(split.X[field], dtype=np.float32)
        )

    for field in NUM_FIELDS:
        value = np.asarray(split.num[field], dtype=np.float32)
        value = np.nan_to_num(
            value, nan=0.0, posinf=0.0, neginf=0.0
        )
        columns.append(
            np.log1p(np.maximum(value, 0.0)).astype(np.float32)
        )

    columns.extend(frequency_columns)
    columns.extend(history_columns)

    matrix = np.stack(columns, axis=1)
    return (
        np.ascontiguousarray(matrix, dtype=np.float32),
        categorical_indices,
    )


def map_rate_model(train, valid, test, y, weights, fields, smooth=25.0):
    prior = float(np.sum(weights * y) / np.sum(weights))
    prior = np.clip(prior, 1e-5, 1.0 - 1e-5)
    prior_logit = np.log(prior / (1.0 - prior))

    valid_terms = []
    test_terms = []
    valid_reliabilities = []
    test_reliabilities = []

    for field in fields:
        card = int(FEATURE_CARDINALITIES[field])
        train_ids = np.asarray(train.X[field], dtype=np.int64)

        count = np.bincount(
            train_ids, weights=weights, minlength=card
        ).astype(np.float64)
        positive = np.bincount(
            train_ids, weights=weights * y, minlength=card
        ).astype(np.float64)

        rate = (
            positive + smooth * prior
        ) / (count + smooth)
        rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
        evidence = np.sqrt(count / (count + smooth))

        field_logit = np.log(rate / (1.0 - rate)) - prior_logit

        def query(split):
            ids = np.asarray(split.X[field], dtype=np.int64)
            safe = np.minimum(np.maximum(ids, 0), card - 1)
            known = (ids >= 0) & (ids < card)
            term = np.where(known, field_logit[safe], 0.0)
            reliability = np.where(known, evidence[safe], 0.0)
            return term, reliability

        va_term, va_rel = query(valid)
        te_term, te_rel = query(test)
        valid_terms.append(va_term)
        test_terms.append(te_term)
        valid_reliabilities.append(va_rel)
        test_reliabilities.append(te_rel)

    valid_terms = np.stack(valid_terms, axis=1)
    test_terms = np.stack(test_terms, axis=1)
    valid_rel = np.stack(valid_reliabilities, axis=1)
    test_rel = np.stack(test_reliabilities, axis=1)

    valid_score = prior_logit + (
        np.sum(valid_terms * valid_rel, axis=1)
        / np.maximum(np.sum(valid_rel, axis=1), 1e-6)
    )
    test_score = prior_logit + (
        np.sum(test_terms * test_rel, axis=1)
        / np.maximum(np.sum(test_rel, axis=1), 1e-6)
    )
    return valid_score, test_score


train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
sample_weights = recency_weights(train.date, HALF_LIFE)

train_frequency, query_frequency = train_frequency_features(
    train, [valid, test]
)
valid_frequency, test_frequency = query_frequency

train_history, history_names = load_history_columns("train")
valid_history, valid_history_names = load_history_columns("valid")
test_history, test_history_names = load_history_columns("test")

if history_names != valid_history_names or history_names != test_history_names:
    raise RuntimeError("Historical feature names differ across splits")

x_train, categorical_indices = make_matrix(
    train, train_frequency, train_history
)
x_valid, _ = make_matrix(
    valid, valid_frequency, valid_history
)
x_test, _ = make_matrix(
    test, test_frequency, test_history
)

dataset = lgb.Dataset(
    x_train,
    label=y_train,
    weight=sample_weights,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)

base_params = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.045,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 180,
    "feature_fraction": 0.82,
    "lambda_l1": 0.05,
    "lambda_l2": 1.3,
    "max_bin": 127,
    "num_threads": THREADS,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "drop_seed": SEED + 3,
    "verbose": -1,
}

own_valid = {}
own_test = {}

# GOSS forms an additive predictor while concentrating later trees on
# observations with large gradients.
goss_params = dict(base_params)
goss_params.update({
    "boosting_type": "gbdt",
    "data_sample_strategy": "goss",
    "top_rate": 0.25,
    "other_rate": 0.15,
})
goss_model = lgb.train(
    goss_params, dataset, num_boost_round=420
)
own_valid["historical_goss"] = goss_model.predict(x_valid)
own_test["historical_goss"] = goss_model.predict(x_test)
del goss_model
gc.collect()

# DART forms its prediction from trees trained under stochastic removal
# of existing trees, reducing co-adaptation to train-period identities.
dart_params = dict(base_params)
dart_params.update({
    "boosting_type": "dart",
    "learning_rate": 0.035,
    "drop_rate": 0.08,
    "skip_drop": 0.55,
    "max_drop": 30,
    "uniform_drop": False,
})
dart_model = lgb.train(
    dart_params, dataset, num_boost_round=320
)
own_valid["historical_dart"] = dart_model.predict(x_valid)
own_test["historical_dart"] = dart_model.predict(x_test)
del dart_model
gc.collect()

# LightGBM RF averages independently bagged trees rather than sequentially
# correcting residuals, giving a materially different prediction family.
rf_params = dict(base_params)
rf_params.update({
    "boosting_type": "rf",
    "learning_rate": 1.0,
    "num_leaves": 127,
    "min_data_in_leaf": 120,
    "feature_fraction": 0.70,
    "bagging_fraction": 0.68,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
})
rf_model = lgb.train(
    rf_params, dataset, num_boost_round=280
)
own_valid["historical_random_forest"] = rf_model.predict(x_valid)
own_test["historical_random_forest"] = rf_model.predict(x_test)
del rf_model
gc.collect()

# A generative empirical likelihood family, with no tree partitioning.
nb_valid, nb_test = map_rate_model(
    train, valid, test, y_train, sample_weights, NB_FIELDS
)
own_valid["categorical_likelihood"] = nb_valid
own_test["categorical_likelihood"] = nb_test

del dataset, x_train, x_valid, x_test
gc.collect()

# Rank-space ensembles preserve exactly the ordering scale relevant to both
# metrics and prevent calibration differences from dominating fusion.
individual_names = list(own_valid.keys())
valid_ranks = {
    name: rank_percentile(valid.user_id, own_valid[name])
    for name in individual_names
}
test_ranks = {
    name: rank_percentile(test.user_id, own_test[name])
    for name in individual_names
}

own_valid["dropout_boosting_ensemble"] = (
    0.62 * valid_ranks["historical_goss"]
    + 0.38 * valid_ranks["historical_dart"]
)
own_test["dropout_boosting_ensemble"] = (
    0.62 * test_ranks["historical_goss"]
    + 0.38 * test_ranks["historical_dart"]
)

own_valid["all_family_rank_ensemble"] = np.mean(
    np.stack([valid_ranks[name] for name in individual_names], axis=1),
    axis=1,
)
own_test["all_family_rank_ensemble"] = np.mean(
    np.stack([test_ranks[name] for name in individual_names], axis=1),
    axis=1,
)

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

candidate_scores = {}
candidate_metrics = {}
candidate_specs = {}

candidate_scores["trusted_incumbent"] = inc_valid
candidate_metrics["trusted_incumbent"] = evaluate(
    valid.user_id, valid.y, inc_valid
)
candidate_specs["trusted_incumbent"] = ("historical_goss", 0.0)

for family in own_valid:
    standalone_name = family + "_standalone"
    standalone_score = np.asarray(own_valid[family], dtype=np.float64)
    candidate_scores[standalone_name] = standalone_score
    candidate_metrics[standalone_name] = evaluate(
        valid.user_id, valid.y, standalone_score
    )
    candidate_specs[standalone_name] = (family, None)

    family_valid_rank = rank_percentile(
        valid.user_id, own_valid[family]
    )
    for alpha in (0.10, 0.20, 0.30, 0.40, 0.50):
        candidate_name = f"{family}_incumbent_blend_{alpha:.2f}"
        score = (
            alpha * family_valid_rank
            + (1.0 - alpha) * inc_valid_rank
        )
        candidate_scores[candidate_name] = score
        candidate_metrics[candidate_name] = evaluate(
            valid.user_id, valid.y, score
        )
        candidate_specs[candidate_name] = (family, alpha)

best_name = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"]),
)
best_metrics = candidate_metrics[best_name]
best_valid = candidate_scores[best_name]
best_family, best_alpha = candidate_specs[best_name]

if best_name == "trusted_incumbent":
    best_test = inc_test
    raw_valid = np.asarray(
        own_valid["historical_goss"], dtype=np.float64
    )
elif best_alpha is None:
    best_test = np.asarray(own_test[best_family], dtype=np.float64)
    raw_valid = np.asarray(own_valid[best_family], dtype=np.float64)
else:
    family_test_rank = rank_percentile(
        test.user_id, own_test[best_family]
    )
    best_test = (
        best_alpha * family_test_rank
        + (1.0 - best_alpha) * inc_test_rank
    )
    raw_valid = np.asarray(own_valid[best_family], dtype=np.float64)

print("CANDIDATES " + json.dumps(
    {
        name: float(metrics["primary"])
        for name, metrics in candidate_metrics.items()
    },
    sort_keys=True,
))
print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "best_family": best_family,
    "best_blend_alpha": best_alpha,
    "half_life_days": HALF_LIFE,
    "historical_feature_count": len(history_names),
    "categorical_feature_count": len(CAT_FIELDS),
    "frequency_feature_count": len(CAT_FIELDS),
    "families": list(own_valid.keys()),
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
            raw_valid,
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))