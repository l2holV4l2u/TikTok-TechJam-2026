import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

_start_time = time.time()

from pipeline.data import load
from pipeline.evaluate import evaluate


SEED = 20260828
SESSION_GAP_MS = 30 * 60 * 1000


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row_id = np.arange(n, dtype=np.int64)

    order = np.lexsort((row_id, scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    positions = np.arange(n, dtype=np.float64) - np.repeat(starts, lengths)
    denominators = np.maximum(np.repeat(lengths, lengths) - 1, 1)

    sorted_ranks = positions / denominators
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def grouped_position(boundary):
    boundary = np.asarray(boundary, dtype=bool)
    n = len(boundary)
    starts = np.flatnonzero(boundary)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    pos = np.arange(n, dtype=np.int64) - np.repeat(starts, lengths)
    total = np.repeat(lengths, lengths)
    return pos, total, starts, lengths


def chronological_features(split):
    user = np.asarray(split.user_id, dtype=np.int64)
    timestamp = np.asarray(split.time_ms, dtype=np.int64)
    date = np.asarray(split.date, dtype=np.int64)
    n = len(user)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, timestamp, user))
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n, dtype=np.int64)

    u = user[order]
    t = timestamp[order]
    d = date[order]

    new_user = np.empty(n, dtype=bool)
    new_user[0] = True
    new_user[1:] = u[1:] != u[:-1]

    new_day = new_user.copy()
    new_day[1:] |= d[1:] != d[:-1]

    gap_prev = np.zeros(n, dtype=np.int64)
    gap_prev[1:] = np.maximum(t[1:] - t[:-1], 0)
    gap_prev[new_user] = 0

    new_session = new_day | (gap_prev > SESSION_GAP_MS)

    new_batch = new_user.copy()
    new_batch[1:] |= t[1:] != t[:-1]

    user_pos, user_total, _, _ = grouped_position(new_user)
    day_pos, day_total, day_starts, day_lengths = grouped_position(new_day)
    session_pos, session_total, session_starts, session_lengths = (
        grouped_position(new_session)
    )
    batch_pos, batch_total, _, _ = grouped_position(new_batch)

    day_start_time = np.repeat(t[day_starts], day_lengths)
    session_start_time = np.repeat(t[session_starts], session_lengths)

    elapsed_day = np.maximum(t - day_start_time, 0)
    elapsed_session = np.maximum(t - session_start_time, 0)

    gap_next = np.zeros(n, dtype=np.int64)
    gap_next[:-1] = np.maximum(t[1:] - t[:-1], 0)
    user_end = np.r_[new_user[1:], True]
    gap_next[user_end] = 0

    def restore(values, dtype=np.float32):
        return np.asarray(values, dtype=dtype)[inverse]

    day_denom = np.maximum(day_total - 1, 1)
    session_denom = np.maximum(session_total - 1, 1)
    user_denom = np.maximum(user_total - 1, 1)
    batch_denom = np.maximum(batch_total - 1, 1)

    feature_columns = [
        restore(np.log1p(user_pos)),
        restore(np.log1p(user_total)),
        restore(user_pos / user_denom),
        restore(np.log1p(day_pos)),
        restore(np.log1p(day_total)),
        restore(day_pos / day_denom),
        restore(np.log1p(session_pos)),
        restore(np.log1p(session_total)),
        restore(session_pos / session_denom),
        restore(np.log1p(batch_pos)),
        restore(np.log1p(batch_total)),
        restore(batch_pos / batch_denom),
        restore(np.log1p(gap_prev / 1000.0)),
        restore(np.log1p(gap_next / 1000.0)),
        restore(np.log1p(elapsed_day / 1000.0)),
        restore(np.log1p(elapsed_session / 1000.0)),
        np.asarray(split.X["hour"], dtype=np.float32),
        np.asarray(split.X["tab"], dtype=np.float32),
        np.asarray(split.X["duration_bucket"], dtype=np.float32),
        np.asarray(split.X["user_active_degree"], dtype=np.float32),
        np.asarray(split.X["is_video_author"], dtype=np.float32),
        np.log1p(
            np.maximum(
                np.asarray(split.num["duration_ms"], dtype=np.float32), 0.0
            )
        ),
    ]

    matrix = np.column_stack(feature_columns).astype(np.float32)

    heuristics = {
        "early_day": -restore(np.log1p(day_pos)),
        "early_session": -restore(np.log1p(session_pos)),
        "early_composite": -(
            0.55 * restore(np.log1p(day_pos))
            + 0.35 * restore(np.log1p(session_pos))
            + 0.10 * restore(np.log1p(batch_pos))
        ),
        "relative_composite": -(
            0.60 * restore(day_pos / day_denom)
            + 0.30 * restore(session_pos / session_denom)
            + 0.10 * restore(batch_pos / batch_denom)
        ),
    }

    summary = {
        "mean_day_total": float(np.mean(day_total)),
        "mean_session_total": float(np.mean(session_total)),
        "mean_batch_total": float(np.mean(batch_total)),
        "session_starts": int(np.sum(new_session)),
    }
    return matrix, heuristics, summary


def z_parameters(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    std = max(float(np.std(values)), 1e-8)
    return mean, std


def apply_z(values, mean, std):
    return (np.asarray(values, dtype=np.float64) - mean) / std


artifacts = os.environ.get("RUN_ARTIFACTS", "")
incumbent_valid_path = os.path.join(
    artifacts, "incumbent_valid_scores.npy"
)
incumbent_test_path = os.path.join(
    artifacts, "incumbent_test_scores.npy"
)

if not (
    os.path.isfile(incumbent_valid_path)
    and os.path.isfile(incumbent_test_path)
):
    raise FileNotFoundError(
        "Trusted incumbent validation/test predictions are required"
    )

train = load("train")
valid = load("valid")

incumbent_valid = np.asarray(
    np.load(incumbent_valid_path), dtype=np.float64
)
if len(incumbent_valid) != len(valid.y):
    raise ValueError("Incumbent validation prediction length mismatch")

incumbent_metrics = evaluate(
    valid.user_id, valid.y, incumbent_valid
)

X_train, train_heuristics, train_summary = chronological_features(train)
X_valid, valid_heuristics, valid_summary = chronological_features(valid)

print(
    "FINDINGS train_order mean_day_total=%.3f "
    "mean_session_total=%.3f mean_batch_total=%.3f sessions=%d"
    % (
        train_summary["mean_day_total"],
        train_summary["mean_session_total"],
        train_summary["mean_batch_total"],
        train_summary["session_starts"],
    )
)
print(
    "FINDINGS valid_order mean_day_total=%.3f "
    "mean_session_total=%.3f mean_batch_total=%.3f sessions=%d"
    % (
        valid_summary["mean_day_total"],
        valid_summary["mean_session_total"],
        valid_summary["mean_batch_total"],
        valid_summary["session_starts"],
    )
)

train_dates = np.asarray(train.date, dtype=np.int64)
days_old = np.maximum(int(train_dates.max()) - train_dates, 0)
train_weights = np.exp(-0.025 * days_old).astype(np.float32)

categorical_indices = [16, 17, 18, 19, 20]

dtrain = lgb.Dataset(
    X_train,
    label=np.asarray(train.y, dtype=np.float32),
    weight=train_weights,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
dvalid = lgb.Dataset(
    X_valid,
    label=np.asarray(valid.y, dtype=np.float32),
    categorical_feature=categorical_indices,
    reference=dtrain,
    free_raw_data=False,
)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.035,
    "num_leaves": 31,
    "max_depth": 8,
    "min_data_in_leaf": 1200,
    "min_sum_hessian_in_leaf": 50.0,
    "feature_fraction": 0.90,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 4.0,
    "max_bin": 127,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "data_random_seed": SEED + 3,
    "num_threads": min(16, max(1, os.cpu_count() or 1)),
    "verbose": -1,
}

model = lgb.train(
    params,
    dtrain,
    num_boost_round=500,
    valid_sets=[dvalid],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

lgb_valid = np.asarray(
    model.predict(X_valid, num_iteration=model.best_iteration),
    dtype=np.float64,
)

source_valid = {
    "position_lgbm": lgb_valid,
    "early_day": np.asarray(valid_heuristics["early_day"], dtype=np.float64),
    "early_session": np.asarray(
        valid_heuristics["early_session"], dtype=np.float64
    ),
    "early_composite": np.asarray(
        valid_heuristics["early_composite"], dtype=np.float64
    ),
    "relative_composite": np.asarray(
        valid_heuristics["relative_composite"], dtype=np.float64
    ),
}

candidate_log = {
    "incumbent": float(incumbent_metrics["primary"])
}

best = {
    "primary": float(incumbent_metrics["primary"]),
    "name": "incumbent",
    "source": "incumbent",
    "mode": "incumbent",
    "alpha": 0.0,
    "scores": incumbent_valid.copy(),
    "metrics": incumbent_metrics,
}

inc_mean, inc_std = z_parameters(incumbent_valid)
inc_z = apply_z(incumbent_valid, inc_mean, inc_std)
inc_rank = within_user_rank(valid.user_id, incumbent_valid)

source_stats = {}

for source_name, source_scores in source_valid.items():
    source_metrics = evaluate(valid.user_id, valid.y, source_scores)
    source_primary = float(source_metrics["primary"])
    candidate_log[source_name] = source_primary

    if source_primary > best["primary"]:
        best = {
            "primary": source_primary,
            "name": source_name,
            "source": source_name,
            "mode": "raw",
            "alpha": 1.0,
            "scores": source_scores.copy(),
            "metrics": source_metrics,
        }

    source_mean, source_std = z_parameters(source_scores)
    source_stats[source_name] = (source_mean, source_std)
    source_z = apply_z(source_scores, source_mean, source_std)
    source_rank = within_user_rank(valid.user_id, source_scores)

    for alpha in np.arange(0.025, 0.401, 0.025):
        alpha = float(alpha)

        z_scores = (1.0 - alpha) * inc_z + alpha * source_z
        z_metrics = evaluate(valid.user_id, valid.y, z_scores)
        z_name = "%s_z_%.3f" % (source_name, alpha)
        z_primary = float(z_metrics["primary"])
        candidate_log[z_name] = z_primary

        if z_primary > best["primary"]:
            best = {
                "primary": z_primary,
                "name": z_name,
                "source": source_name,
                "mode": "zblend",
                "alpha": alpha,
                "scores": z_scores.copy(),
                "metrics": z_metrics,
            }

        rank_scores = (
            (1.0 - alpha) * inc_rank + alpha * source_rank
        )
        rank_metrics = evaluate(valid.user_id, valid.y, rank_scores)
        rank_name = "%s_rank_%.3f" % (source_name, alpha)
        rank_primary = float(rank_metrics["primary"])
        candidate_log[rank_name] = rank_primary

        if rank_primary > best["primary"]:
            best = {
                "primary": rank_primary,
                "name": rank_name,
                "source": source_name,
                "mode": "rankblend",
                "alpha": alpha,
                "scores": rank_scores.copy(),
                "metrics": rank_metrics,
            }

top_candidates = sorted(
    candidate_log.items(), key=lambda item: item[1], reverse=True
)[:20]
print(
    "CANDIDATES "
    + json.dumps(
        {name: score for name, score in top_candidates},
        separators=(", ", ": "),
    )
)
print(
    "FINDINGS lgb_best_iteration=%d lgb_primary=%.6f "
    "selected=%s mode=%s alpha=%.3f incumbent=%.6f selected_primary=%.6f"
    % (
        int(model.best_iteration),
        float(candidate_log["position_lgbm"]),
        best["name"],
        best["mode"],
        float(best["alpha"]),
        float(incumbent_metrics["primary"]),
        float(best["primary"]),
    )
)

valid_scores = np.asarray(best["scores"], dtype=np.float64)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        valid_scores,
    )

del X_train, X_valid, dtrain, dvalid
gc.collect()

# Validation selection is complete. Test labels are never accessed.
test = load("test")
incumbent_test = np.asarray(
    np.load(incumbent_test_path), dtype=np.float64
)
if len(incumbent_test) != len(test.user_id):
    raise ValueError("Incumbent test prediction length mismatch")

if best["mode"] == "incumbent":
    test_scores = incumbent_test.copy()
else:
    X_test, test_heuristics, test_summary = chronological_features(test)

    print(
        "FINDINGS test_order mean_day_total=%.3f "
        "mean_session_total=%.3f mean_batch_total=%.3f sessions=%d"
        % (
            test_summary["mean_day_total"],
            test_summary["mean_session_total"],
            test_summary["mean_batch_total"],
            test_summary["session_starts"],
        )
    )

    selected_source = best["source"]
    if selected_source == "position_lgbm":
        source_test = np.asarray(
            model.predict(X_test, num_iteration=model.best_iteration),
            dtype=np.float64,
        )
    else:
        source_test = np.asarray(
            test_heuristics[selected_source], dtype=np.float64
        )

    if best["mode"] == "raw":
        test_scores = source_test
    elif best["mode"] == "zblend":
        source_mean, source_std = source_stats[selected_source]
        incumbent_test_z = apply_z(incumbent_test, inc_mean, inc_std)
        source_test_z = apply_z(source_test, source_mean, source_std)
        alpha = float(best["alpha"])
        test_scores = (
            (1.0 - alpha) * incumbent_test_z + alpha * source_test_z
        )
    elif best["mode"] == "rankblend":
        incumbent_test_rank = within_user_rank(
            test.user_id, incumbent_test
        )
        source_test_rank = within_user_rank(test.user_id, source_test)
        alpha = float(best["alpha"])
        test_scores = (
            (1.0 - alpha) * incumbent_test_rank
            + alpha * source_test_rank
        )
    else:
        raise ValueError("Unknown selected mode: %s" % best["mode"])

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - _start_time)
metrics = best["metrics"]

print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": elapsed,
        },
        separators=(", ", ": "),
    )
)