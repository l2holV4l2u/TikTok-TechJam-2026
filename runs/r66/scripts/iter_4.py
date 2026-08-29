import os
import time
import json
import gc
import random

import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.evaluate import evaluate


START = time.time()
SEED = 20260828

random.seed(SEED)
np.random.seed(SEED)

try:
    lgb.register_logger(None)
except Exception:
    pass


def group_position_and_size(boundary):
    boundary = np.asarray(boundary, dtype=bool)
    n = len(boundary)
    if n == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty

    boundary = boundary.copy()
    boundary[0] = True
    starts = np.flatnonzero(boundary)
    group_id = np.cumsum(boundary, dtype=np.int64) - 1
    ends = np.r_[starts[1:], n]
    sizes_by_group = ends - starts

    positions = np.arange(n, dtype=np.int64) - starts[group_id]
    sizes = sizes_by_group[group_id]
    return positions.astype(np.float32), sizes.astype(np.float32)


def make_temporal_features(split):
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    dates = np.asarray(split.date, dtype=np.int64)
    n = len(users)
    rows = np.arange(n, dtype=np.int64)

    # The row index is deliberately the final key: tied feed-batch timestamps
    # are specified to be ordered by original row position.
    order = np.lexsort((rows, times, users))
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n, dtype=np.int64)

    u = users[order]
    t = times[order]
    d = dates[order]

    user_boundary = np.empty(n, dtype=bool)
    user_boundary[0] = True
    user_boundary[1:] = u[1:] != u[:-1]

    user_pos, user_size = group_position_and_size(user_boundary)

    day_boundary = user_boundary.copy()
    day_boundary[1:] |= d[1:] != d[:-1]
    day_pos, day_size = group_position_and_size(day_boundary)

    batch_boundary = user_boundary.copy()
    batch_boundary[1:] |= t[1:] != t[:-1]
    batch_pos, batch_size = group_position_and_size(batch_boundary)

    gap_prev_ms = np.zeros(n, dtype=np.int64)
    gap_prev_ms[1:] = np.maximum(t[1:] - t[:-1], 0)
    gap_prev_ms[user_boundary] = 0

    user_end = np.empty(n, dtype=bool)
    user_end[-1] = True
    user_end[:-1] = u[:-1] != u[1:]

    gap_next_ms = np.zeros(n, dtype=np.int64)
    gap_next_ms[:-1] = np.maximum(t[1:] - t[:-1], 0)
    gap_next_ms[user_end] = 0

    # A 30-minute inactivity threshold is a conventional feed-session split.
    session_boundary = user_boundary | (gap_prev_ms > 30 * 60 * 1000)
    session_pos, session_size = group_position_and_size(session_boundary)

    # Compute elapsed time from the first and to the final logged impression
    # of the user without Python loops.
    user_starts = np.flatnonzero(user_boundary)
    user_group_id = np.cumsum(user_boundary, dtype=np.int64) - 1
    user_ends = np.r_[user_starts[1:], n] - 1
    first_time = t[user_starts][user_group_id]
    last_time = t[user_ends][user_group_id]

    elapsed_from_first = np.maximum(t - first_time, 0).astype(np.float64)
    elapsed_to_last = np.maximum(last_time - t, 0).astype(np.float64)

    hour_id = np.asarray(split.X["hour"], dtype=np.float32)[order]
    tab = np.asarray(split.X["tab"], dtype=np.float32)[order]
    activity = np.asarray(
        split.X["user_active_degree"], dtype=np.float32
    )[order]
    lowactive = np.asarray(
        split.X["is_lowactive_period"], dtype=np.float32
    )[order]
    video_type = np.asarray(split.X["video_type"], dtype=np.float32)[order]
    duration_bucket = np.asarray(
        split.X["duration_bucket"], dtype=np.float32
    )[order]

    duration = np.asarray(split.num["duration_ms"], dtype=np.float32)[order]
    finite_duration = np.isfinite(duration)
    duration_missing = (~finite_duration).astype(np.float32)
    duration_clean = np.zeros(n, dtype=np.float32)
    duration_clean[finite_duration] = np.log1p(
        np.maximum(duration[finite_duration], 0.0)
    )

    # Epoch day gives weekday consistently across the April/May boundary.
    epoch_day = np.floor_divide(t, 86400000)
    weekday = ((epoch_day + 3) % 7).astype(np.float32)

    # Ratios expose normalized feed depth while raw/log positions let trees
    # represent sharp first-item and early-session effects.
    user_denom = np.maximum(user_size - 1.0, 1.0)
    day_denom = np.maximum(day_size - 1.0, 1.0)
    batch_denom = np.maximum(batch_size - 1.0, 1.0)
    session_denom = np.maximum(session_size - 1.0, 1.0)

    features_sorted = np.column_stack(
        [
            np.log1p(user_pos),
            np.log1p(np.maximum(user_size - user_pos - 1.0, 0.0)),
            user_pos / user_denom,
            np.log1p(user_size),
            np.log1p(day_pos),
            np.log1p(np.maximum(day_size - day_pos - 1.0, 0.0)),
            day_pos / day_denom,
            np.log1p(day_size),
            np.log1p(batch_pos),
            np.log1p(np.maximum(batch_size - batch_pos - 1.0, 0.0)),
            batch_pos / batch_denom,
            np.log1p(batch_size),
            np.log1p(session_pos),
            np.log1p(np.maximum(session_size - session_pos - 1.0, 0.0)),
            session_pos / session_denom,
            np.log1p(session_size),
            np.log1p(gap_prev_ms.astype(np.float64) / 1000.0),
            np.log1p(gap_next_ms.astype(np.float64) / 1000.0),
            np.log1p(elapsed_from_first / 1000.0),
            np.log1p(elapsed_to_last / 1000.0),
            (user_pos == 0).astype(np.float32),
            (day_pos == 0).astype(np.float32),
            (batch_pos == 0).astype(np.float32),
            (session_pos == 0).astype(np.float32),
            (batch_size > 1).astype(np.float32),
            hour_id,
            np.sin(2.0 * np.pi * hour_id / 24.0),
            np.cos(2.0 * np.pi * hour_id / 24.0),
            weekday,
            np.sin(2.0 * np.pi * weekday / 7.0),
            np.cos(2.0 * np.pi * weekday / 7.0),
            tab,
            activity,
            lowactive,
            video_type,
            duration_bucket,
            duration_clean,
            duration_missing,
        ]
    ).astype(np.float32, copy=False)

    features = np.empty_like(features_sorted)
    features[order] = features_sorted

    diagnostics = {
        "mean_user_impressions": float(np.mean(user_size)),
        "mean_day_impressions": float(np.mean(day_size)),
        "mean_batch_size": float(np.mean(batch_size)),
        "multi_item_batch_fraction": float(np.mean(batch_size > 1)),
        "mean_session_size": float(np.mean(session_size)),
        "session_start_fraction": float(np.mean(session_pos == 0)),
    }

    handcrafted_sorted = {
        "early_batch": -np.log1p(batch_pos).astype(np.float64),
        "early_session": -np.log1p(session_pos).astype(np.float64),
        "early_day": -np.log1p(day_pos).astype(np.float64),
        "early_user": -np.log1p(user_pos).astype(np.float64),
    }
    handcrafted = {}
    for name, sorted_score in handcrafted_sorted.items():
        score = np.empty(n, dtype=np.float64)
        score[order] = sorted_score
        handcrafted[name] = score

    return features, handcrafted, diagnostics


FEATURE_NAMES = [
    "log_user_position",
    "log_user_reverse_position",
    "user_position_fraction",
    "log_user_size",
    "log_day_position",
    "log_day_reverse_position",
    "day_position_fraction",
    "log_day_size",
    "log_batch_position",
    "log_batch_reverse_position",
    "batch_position_fraction",
    "log_batch_size",
    "log_session_position",
    "log_session_reverse_position",
    "session_position_fraction",
    "log_session_size",
    "log_gap_previous_seconds",
    "log_gap_next_seconds",
    "log_elapsed_from_user_first",
    "log_elapsed_to_user_last",
    "is_user_first",
    "is_day_first",
    "is_batch_first",
    "is_session_first",
    "is_multi_item_batch",
    "hour_id",
    "hour_sin",
    "hour_cos",
    "weekday",
    "weekday_sin",
    "weekday_cos",
    "tab",
    "user_active_degree",
    "is_lowactive_period",
    "video_type",
    "duration_bucket",
    "log_duration_ms",
    "duration_missing",
]


def standardize_from_valid(scores):
    scores = np.asarray(scores, dtype=np.float64)
    mean = float(np.mean(scores))
    std = max(float(np.std(scores)), 1e-8)
    return (scores - mean) / std, mean, std


def apply_standardization(scores, mean, std):
    return (np.asarray(scores, dtype=np.float64) - mean) / std


train = load("train")
valid = load("valid")

x_train, train_handcrafted, train_diag = make_temporal_features(train)
x_valid, valid_handcrafted, valid_diag = make_temporal_features(valid)

y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

dtrain = lgb.Dataset(
    x_train,
    label=y_train,
    feature_name=FEATURE_NAMES,
    free_raw_data=True,
)
dvalid = lgb.Dataset(
    x_valid,
    label=y_valid,
    feature_name=FEATURE_NAMES,
    reference=dtrain,
    free_raw_data=True,
)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.035,
    "num_leaves": 15,
    "max_depth": 5,
    "min_data_in_leaf": 800,
    "min_sum_hessian_in_leaf": 2.0,
    "feature_fraction": 0.90,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 8.0,
    "max_bin": 127,
    "verbosity": -1,
    "verbose": -1,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "data_random_seed": SEED + 3,
    "num_threads": min(16, os.cpu_count() or 1),
    "force_col_wise": True,
}

model = lgb.train(
    params,
    dtrain,
    num_boost_round=500,
    valid_sets=[dvalid],
    valid_names=["valid"],
    callbacks=[
        lgb.early_stopping(45, first_metric_only=True, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

temporal_valid = np.asarray(
    model.predict(x_valid, num_iteration=model.best_iteration),
    dtype=np.float64,
)

artifacts = os.environ.get("RUN_ARTIFACTS")
if not artifacts:
    raise RuntimeError("RUN_ARTIFACTS is required")

inc_valid_path = os.path.join(artifacts, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(artifacts, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are missing")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if len(inc_valid) != len(valid_users):
    raise ValueError("Incumbent validation score length mismatch")

inc_valid_z, inc_mean, inc_std = standardize_from_valid(inc_valid)
temporal_valid_z, temporal_mean, temporal_std = standardize_from_valid(
    temporal_valid
)

base_metrics = evaluate(valid_users, y_valid, inc_valid_z)
temporal_metrics = evaluate(valid_users, y_valid, temporal_valid_z)

source_valid = {"temporal_model": temporal_valid_z}
source_scaling = {
    "temporal_model": (temporal_mean, temporal_std),
}

for name, scores in valid_handcrafted.items():
    z, mean, std = standardize_from_valid(scores)
    source_valid[name] = z
    source_scaling[name] = (mean, std)

# Signed coefficients test both possible exposure directions. Zero guarantees
# that an unsupported temporal hypothesis cannot make the submitted result
# worse than the trusted incumbent.
betas = np.array(
    [
        -0.50, -0.35, -0.25, -0.15, -0.10, -0.05,
        0.00,
        0.05, 0.10, 0.15, 0.25, 0.35, 0.50,
    ],
    dtype=np.float64,
)

candidate_primary = {}
best_primary = -np.inf
best_metrics = None
best_valid_scores = None
best_source = None
best_beta = None

for source_name, source_score in source_valid.items():
    for beta in betas:
        score = inc_valid_z + float(beta) * source_score
        metrics = evaluate(valid_users, y_valid, score)
        candidate_name = "%s_beta_%+.2f" % (source_name, float(beta))
        candidate_primary[candidate_name] = float(metrics["primary"])

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_metrics = metrics
            best_valid_scores = np.asarray(score, dtype=np.float64).copy()
            best_source = source_name
            best_beta = float(beta)

if best_valid_scores is None:
    raise RuntimeError("No validation candidate was selected")

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        best_valid_scores,
    )

# Construct test features without ever accessing test labels.
del x_train, x_valid, y_train, dtrain, dvalid
gc.collect()

test = load("test")
x_test, test_handcrafted, test_diag = make_temporal_features(test)

temporal_test = np.asarray(
    model.predict(x_test, num_iteration=model.best_iteration),
    dtype=np.float64,
)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test) != len(test.user_id):
    raise ValueError("Incumbent test score length mismatch")

inc_test_z = apply_standardization(inc_test, inc_mean, inc_std)

if best_source == "temporal_model":
    source_test_raw = temporal_test
else:
    source_test_raw = test_handcrafted[best_source]

source_mean, source_std = source_scaling[best_source]
source_test_z = apply_standardization(
    source_test_raw, source_mean, source_std
)
test_scores = inc_test_z + best_beta * source_test_z

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

try:
    model.save_model(
        os.path.join(artifacts, "temporal_feed_context_lgbm.txt"),
        num_iteration=model.best_iteration,
    )
    with open(
        os.path.join(artifacts, "temporal_feed_context_lgbm.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "seed": SEED,
                "best_iteration": int(model.best_iteration),
                "selected_source": best_source,
                "selected_beta": float(best_beta),
                "incumbent_mean_valid": inc_mean,
                "incumbent_std_valid": inc_std,
                "source_mean_valid": source_mean,
                "source_std_valid": source_std,
                "feature_names": FEATURE_NAMES,
                "train_diagnostics": train_diag,
                "valid_diagnostics": valid_diag,
                "test_diagnostics": test_diag,
            },
            f,
        )
except Exception as exc:
    print(
        "FINDINGS "
        + json.dumps({"artifact_save_warning": str(exc)})
    )

elapsed = time.time() - START

print(
    "FINDINGS "
    + json.dumps(
        {
            "direction": "feed_order_and_session_context",
            "best_iteration": int(model.best_iteration),
            "incumbent_primary": float(base_metrics["primary"]),
            "temporal_model_only_primary": float(
                temporal_metrics["primary"]
            ),
            "selected_source": best_source,
            "selected_beta": float(best_beta),
            "valid_multi_item_batch_fraction": float(
                valid_diag["multi_item_batch_fraction"]
            ),
            "valid_mean_batch_size": float(
                valid_diag["mean_batch_size"]
            ),
            "valid_mean_session_size": float(
                valid_diag["mean_session_size"]
            ),
        }
    )
)
print("CANDIDATES " + json.dumps(candidate_primary))
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)