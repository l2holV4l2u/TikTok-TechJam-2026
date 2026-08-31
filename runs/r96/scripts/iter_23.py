import os
import time
import json
import gc
import datetime
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 93217
THREADS = min(16, os.cpu_count() or 1)

CAT_FIELDS = list(FEATURE_CARDINALITIES.keys())
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    start_idx = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_idx = np.flatnonzero(ends)
    sizes = np.diff(np.concatenate((
        np.asarray([-1], dtype=np.int64), end_idx
    )))
    row_sizes = np.repeat(sizes, sizes)

    positions = np.arange(n, dtype=np.int64) - start_idx
    ranks_sorted = (
        positions.astype(np.float64) + 0.5
    ) / row_sizes.astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranks_sorted
    return result


def date_ordinal(values):
    values = np.asarray(values, dtype=np.int32)
    unique = np.unique(values)
    mapping = {}
    for value in unique:
        text = str(int(value))
        d = datetime.date(
            int(text[:4]), int(text[4:6]), int(text[6:8])
        )
        mapping[int(value)] = d.toordinal()
    return np.asarray(
        [mapping[int(v)] for v in values], dtype=np.float32
    )


def train_frequency_columns(train, queries):
    train_columns = []
    query_columns = [[] for _ in queries]

    for field in CAT_FIELDS:
        card = int(FEATURE_CARDINALITIES[field])
        tr_ids = np.asarray(train.X[field], dtype=np.int64)
        counts = np.bincount(
            tr_ids, minlength=card
        ).astype(np.float32)
        values = np.log1p(counts).astype(np.float32)

        train_columns.append(values[tr_ids])
        for j, split in enumerate(queries):
            ids = np.asarray(split.X[field], dtype=np.int64)
            safe = np.clip(ids, 0, card - 1)
            known = (ids >= 0) & (ids < card)
            col = np.where(known, values[safe], 0.0)
            query_columns[j].append(col.astype(np.float32))

    return train_columns, query_columns


def history_columns(split_name):
    columns = []
    names = []
    for entity in ("video_id", "author_id"):
        hist = historical_features(split_name, key=entity)
        for name in sorted(hist.keys()):
            value = np.asarray(hist[name], dtype=np.float32)
            value = np.nan_to_num(
                value, nan=0.0, posinf=0.0, neginf=0.0
            )
            columns.append(value)
            names.append(name)
    return columns, names


def make_matrix(split, frequency, history):
    columns = []
    categorical = []

    for field in CAT_FIELDS:
        categorical.append(len(columns))
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

    columns.extend(frequency)
    columns.extend(history)

    matrix = np.stack(columns, axis=1)
    return np.ascontiguousarray(matrix, dtype=np.float32), categorical


def clipped_logit(prob):
    prob = np.clip(
        np.asarray(prob, dtype=np.float64), 1e-5, 1.0 - 1e-5
    )
    return np.log(prob / (1.0 - prob))


def temporal_forecast(predictions, centers, target_days, slope_limit=0.08):
    logits = np.stack(
        [clipped_logit(p) for p in predictions], axis=1
    )
    x = np.asarray(centers, dtype=np.float64)
    x_center = float(np.mean(x))
    dx = x - x_center
    denominator = float(np.sum(dx * dx))

    intercept = np.mean(logits, axis=1)
    slope = np.sum(logits * dx[None, :], axis=1) / denominator
    slope = np.clip(slope, -slope_limit, slope_limit)

    target_days = np.asarray(target_days, dtype=np.float64)
    forecast = intercept + slope * (target_days - x_center)
    return np.clip(forecast, -12.0, 12.0)


train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
train_days = date_ordinal(train.date)
valid_days = date_ordinal(valid.date)
test_days = date_ordinal(test.date)

train_frequency, query_frequency = train_frequency_columns(
    train, [valid, test]
)
valid_frequency, test_frequency = query_frequency

train_history, history_names = history_columns("train")
valid_history, valid_history_names = history_columns("valid")
test_history, test_history_names = history_columns("test")

if history_names != valid_history_names or history_names != test_history_names:
    raise RuntimeError("Historical feature schemas differ")

x_train, categorical_indices = make_matrix(
    train, train_frequency, train_history
)
x_valid, _ = make_matrix(
    valid, valid_frequency, valid_history
)
x_test, _ = make_matrix(
    test, test_frequency, test_history
)

common = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.04,
    "max_bin": 127,
    "lambda_l1": 0.05,
    "lambda_l2": 1.5,
    "num_threads": THREADS,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "verbose": -1,
}

own_valid = {}
own_test = {}

# Family 1: a generalized additive boosted-stump model. With one split per
# tree, prediction is a sum of univariate response curves rather than an
# interaction partition.
gam_dataset = lgb.Dataset(
    x_train,
    label=y_train,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
gam_params = dict(common)
gam_params.update({
    "boosting_type": "gbdt",
    "learning_rate": 0.035,
    "num_leaves": 2,
    "max_depth": 1,
    "min_data_in_leaf": 300,
    "feature_fraction": 1.0,
})
gam = lgb.train(
    gam_params,
    gam_dataset,
    num_boost_round=1100,
)
own_valid["additive_gam"] = gam.predict(x_valid)
own_test["additive_gam"] = gam.predict(x_test)
del gam, gam_dataset
gc.collect()

# Family 2: leaves contain local linear regressors over numeric columns.
# This forms piecewise-linear response surfaces rather than constants.
linear_dataset = lgb.Dataset(
    x_train,
    label=y_train,
    categorical_feature=categorical_indices,
    params={"linear_tree": True},
    free_raw_data=False,
)
linear_params = dict(common)
linear_params.update({
    "boosting_type": "gbdt",
    "linear_tree": True,
    "learning_rate": 0.035,
    "num_leaves": 31,
    "min_data_in_leaf": 350,
    "feature_fraction": 0.78,
    "linear_lambda": 2.0,
})
linear_model = lgb.train(
    linear_params,
    linear_dataset,
    num_boost_round=260,
)
own_valid["piecewise_linear_tree"] = linear_model.predict(x_valid)
own_test["piecewise_linear_tree"] = linear_model.predict(x_test)
del linear_model, linear_dataset
gc.collect()

# Family 3: independently fitted temporal snapshot experts. Their mean is a
# bagged temporal predictor, while a per-row local linear trend forecasts
# relevance to each query impression's actual future day.
unique_dates = np.sort(np.unique(np.asarray(train.date, dtype=np.int32)))
if len(unique_dates) < 10:
    raise RuntimeError("Insufficient training dates for snapshot experts")

windows = [
    unique_dates[:6],
    unique_dates[4:10],
    unique_dates[-6:],
]

snapshot_valid = []
snapshot_test = []
snapshot_centers = []

snapshot_params = dict(common)
snapshot_params.update({
    "boosting_type": "gbdt",
    "learning_rate": 0.045,
    "num_leaves": 63,
    "min_data_in_leaf": 180,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.86,
    "bagging_freq": 1,
})

for expert_index, window in enumerate(windows):
    mask = np.isin(np.asarray(train.date), window)
    row_idx = np.flatnonzero(mask)

    # Mild within-window recency emphasis keeps each expert representative
    # of its right boundary without collapsing it to one day.
    local_days = train_days[row_idx]
    local_age = float(np.max(local_days)) - local_days
    local_weight = np.power(
        0.5, local_age / 5.0
    ).astype(np.float32)
    local_weight /= max(float(local_weight.mean()), 1e-8)

    dset = lgb.Dataset(
        x_train[row_idx],
        label=y_train[row_idx],
        weight=local_weight,
        categorical_feature=categorical_indices,
        free_raw_data=True,
    )
    params = dict(snapshot_params)
    params["seed"] = SEED + 100 + expert_index
    params["feature_fraction_seed"] = SEED + 200 + expert_index
    params["bagging_seed"] = SEED + 300 + expert_index

    model = lgb.train(
        params,
        dset,
        num_boost_round=260,
    )
    snapshot_valid.append(model.predict(x_valid))
    snapshot_test.append(model.predict(x_test))
    snapshot_centers.append(float(np.mean(local_days)))

    del model, dset, row_idx, local_weight
    gc.collect()

snapshot_valid_array = np.stack(snapshot_valid, axis=1)
snapshot_test_array = np.stack(snapshot_test, axis=1)

own_valid["snapshot_expert_mean"] = np.mean(
    snapshot_valid_array, axis=1
)
own_test["snapshot_expert_mean"] = np.mean(
    snapshot_test_array, axis=1
)

own_valid["snapshot_trend_forecast"] = temporal_forecast(
    snapshot_valid,
    snapshot_centers,
    valid_days,
    slope_limit=0.08,
)
own_test["snapshot_trend_forecast"] = temporal_forecast(
    snapshot_test,
    snapshot_centers,
    test_days,
    slope_limit=0.08,
)

# A robust temporal predictor uses the median expert and is insensitive to
# one anomalous period-specific model.
own_valid["snapshot_expert_median"] = np.median(
    snapshot_valid_array, axis=1
)
own_test["snapshot_expert_median"] = np.median(
    snapshot_test_array, axis=1
)

del x_train, x_valid, x_test
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

if len(inc_valid) != len(valid) or len(inc_test) != len(test):
    raise RuntimeError("Incumbent prediction length mismatch")

inc_valid_rank = rank_percentile(valid.user_id, inc_valid)
inc_test_rank = rank_percentile(test.user_id, inc_test)

candidate_scores = {}
candidate_metrics = {}
candidate_specs = {}

candidate_scores["trusted_incumbent"] = inc_valid
candidate_metrics["trusted_incumbent"] = evaluate(
    valid.user_id, valid.y, inc_valid
)
candidate_specs["trusted_incumbent"] = ("trusted_incumbent", 0.0)

blend_alphas = (0.10, 0.20, 0.30, 0.40, 0.50, 0.65)

for family, valid_prediction in own_valid.items():
    valid_prediction = np.asarray(valid_prediction, dtype=np.float64)
    standalone = family + "_standalone"
    candidate_scores[standalone] = valid_prediction
    candidate_metrics[standalone] = evaluate(
        valid.user_id, valid.y, valid_prediction
    )
    candidate_specs[standalone] = (family, None)

    family_rank = rank_percentile(
        valid.user_id, valid_prediction
    )
    for alpha in blend_alphas:
        name = f"{family}_incumbent_blend_{alpha:.2f}"
        score = (
            alpha * family_rank
            + (1.0 - alpha) * inc_valid_rank
        )
        candidate_scores[name] = score
        candidate_metrics[name] = evaluate(
            valid.user_id, valid.y, score
        )
        candidate_specs[name] = (family, alpha)

best_name = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"]),
)
best_metrics = candidate_metrics[best_name]
best_valid = np.asarray(
    candidate_scores[best_name], dtype=np.float64
)
best_family, best_alpha = candidate_specs[best_name]

if best_name == "trusted_incumbent":
    best_test = inc_test
    raw_valid = np.asarray(
        own_valid["snapshot_trend_forecast"], dtype=np.float64
    )
elif best_alpha is None:
    best_test = np.asarray(
        own_test[best_family], dtype=np.float64
    )
    raw_valid = np.asarray(
        own_valid[best_family], dtype=np.float64
    )
else:
    family_test_rank = rank_percentile(
        test.user_id, own_test[best_family]
    )
    best_test = (
        best_alpha * family_test_rank
        + (1.0 - best_alpha) * inc_test_rank
    )
    raw_valid = np.asarray(
        own_valid[best_family], dtype=np.float64
    )

raw_primary = {
    family: float(evaluate(
        valid.user_id, valid.y, prediction
    )["primary"])
    for family, prediction in own_valid.items()
}

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
    "raw_family_primary": raw_primary,
    "snapshot_windows": [
        [int(v) for v in window] for window in windows
    ],
    "snapshot_centers_ordinal": snapshot_centers,
    "historical_feature_count": len(history_names),
    "families": list(own_valid.keys()),
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        best_valid,
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