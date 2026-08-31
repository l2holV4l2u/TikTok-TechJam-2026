import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 92417
THREADS = min(16, os.cpu_count() or 1)
HALF_LIFE = 4.0
SMOOTH = 80.0

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "hour",
    "duration_bucket", "tag", "upload_type", "music_type",
    "user_active_degree", "fans_user_num_range",
    "follow_user_num_range", "friend_user_num_range",
    "register_days_range", "onehot_feat3", "onehot_feat8",
    "is_video_author", "is_live_streamer", "video_type",
]

NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]


def within_user_rank(user_ids, scores):
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
    ranked_sorted = (
        positions.astype(np.float64) + 0.5
    ) / np.maximum(row_sizes, 1)
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


def group_position(boundary):
    n = len(boundary)
    starts = np.flatnonzero(boundary)
    start_for_row = np.maximum.accumulate(
        np.where(boundary, np.arange(n, dtype=np.int64), 0)
    )
    pos = np.arange(n, dtype=np.int64) - start_for_row
    ends = np.concatenate((starts[1:] - 1, [n - 1]))
    sizes = ends - starts + 1
    row_sizes = np.repeat(sizes, sizes)
    return pos, row_sizes


def chronological_features(split):
    n = len(split)
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    dates = np.asarray(split.date, dtype=np.int64)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, times, users))
    u = users[order]
    t = times[order]
    d = dates[order]

    user_boundary = np.empty(n, dtype=bool)
    user_boundary[0] = True
    user_boundary[1:] = u[1:] != u[:-1]
    user_pos, user_size = group_position(user_boundary)
    user_reverse = user_size - 1 - user_pos

    day_boundary = np.empty(n, dtype=bool)
    day_boundary[0] = True
    day_boundary[1:] = (
        (u[1:] != u[:-1]) | (d[1:] != d[:-1])
    )
    day_pos, day_size = group_position(day_boundary)
    day_reverse = day_size - 1 - day_pos

    previous_gap = np.zeros(n, dtype=np.float64)
    previous_gap[1:] = np.maximum(
        (t[1:] - t[:-1]).astype(np.float64) / 1000.0, 0.0
    )
    previous_gap[user_boundary] = 0.0

    next_gap = np.zeros(n, dtype=np.float64)
    next_gap[:-1] = previous_gap[1:]
    user_end = np.empty(n, dtype=bool)
    user_end[-1] = True
    user_end[:-1] = u[:-1] != u[1:]
    next_gap[user_end] = 0.0

    session_boundary = (
        user_boundary
        | day_boundary
        | (previous_gap > 30.0 * 60.0)
    )
    session_pos, session_size = group_position(session_boundary)
    session_reverse = session_size - 1 - session_pos

    batch_boundary = np.empty(n, dtype=bool)
    batch_boundary[0] = True
    batch_boundary[1:] = (
        (u[1:] != u[:-1]) | (t[1:] != t[:-1])
    )
    batch_pos, batch_size = group_position(batch_boundary)
    batch_reverse = batch_size - 1 - batch_pos

    first_time = np.minimum.accumulate(
        np.where(user_boundary, t, np.iinfo(np.int64).max)
    )
    # The previous expression is not group-local, so obtain first/last
    # timestamps through the already computed positions and sizes.
    first_indices = np.arange(n, dtype=np.int64) - user_pos
    last_indices = first_indices + user_size - 1
    elapsed = np.maximum(
        (t - t[first_indices]).astype(np.float64) / 1000.0, 0.0
    )
    remaining = np.maximum(
        (t[last_indices] - t).astype(np.float64) / 1000.0, 0.0
    )

    hour = np.asarray(split.X["hour"], dtype=np.int64)[order]
    tab = np.asarray(split.X["tab"], dtype=np.int64)[order]

    sorted_features = {
        "user_pos": user_pos.astype(np.float32),
        "user_reverse": user_reverse.astype(np.float32),
        "user_size": user_size.astype(np.float32),
        "user_fraction": (
            (user_pos + 0.5) / np.maximum(user_size, 1)
        ).astype(np.float32),
        "day_pos": day_pos.astype(np.float32),
        "day_reverse": day_reverse.astype(np.float32),
        "day_size": day_size.astype(np.float32),
        "session_pos": session_pos.astype(np.float32),
        "session_reverse": session_reverse.astype(np.float32),
        "session_size": session_size.astype(np.float32),
        "batch_pos": batch_pos.astype(np.float32),
        "batch_reverse": batch_reverse.astype(np.float32),
        "batch_size": batch_size.astype(np.float32),
        "log_previous_gap": np.log1p(previous_gap).astype(np.float32),
        "log_next_gap": np.log1p(next_gap).astype(np.float32),
        "log_elapsed": np.log1p(elapsed).astype(np.float32),
        "log_remaining": np.log1p(remaining).astype(np.float32),
        "hour": hour.astype(np.float32),
        "tab": tab.astype(np.float32),
    }

    result = {}
    for name, sorted_values in sorted_features.items():
        values = np.empty(n, dtype=np.float32)
        values[order] = sorted_values
        result[name] = values
    return result


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int32)
    age = dates.max() - dates
    weights = np.power(
        0.5, age.astype(np.float32) / HALF_LIFE
    )
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def clipped_int(values, maximum):
    return np.minimum(
        np.asarray(values, dtype=np.int64), maximum
    )


def context_keys(context, split):
    position = clipped_int(context["user_pos"], 31)
    reverse = clipped_int(context["user_reverse"], 31)
    day_pos = clipped_int(context["day_pos"], 15)
    session_pos = clipped_int(context["session_pos"], 15)
    batch_pos = clipped_int(context["batch_pos"], 7)
    batch_size = clipped_int(context["batch_size"] - 1, 7)
    size_bin = clipped_int(
        np.floor(np.log2(np.maximum(context["user_size"], 1))), 6
    )
    gap_bin = clipped_int(
        np.floor(context["log_previous_gap"] / 1.5), 9
    )
    hour = np.asarray(split.X["hour"], dtype=np.int64)
    tab = np.asarray(split.X["tab"], dtype=np.int64)

    return [
        (position, 32, "position"),
        (reverse, 32, "reverse"),
        (day_pos, 16, "day_position"),
        (session_pos, 16, "session_position"),
        (batch_pos, 8, "batch_position"),
        (batch_size, 8, "batch_size"),
        (size_bin, 7, "slate_size"),
        (gap_bin, 10, "previous_gap"),
        (
            position * 7 + size_bin,
            32 * 7,
            "position_x_size",
        ),
        (
            session_pos * 8 + batch_size,
            16 * 8,
            "session_x_batch",
        ),
        (
            position * int(FEATURE_CARDINALITIES["tab"]) + tab,
            32 * int(FEATURE_CARDINALITIES["tab"]),
            "position_x_tab",
        ),
        (
            session_pos * int(FEATURE_CARDINALITIES["hour"]) + hour,
            16 * int(FEATURE_CARDINALITIES["hour"]),
            "session_x_hour",
        ),
    ]


def empirical_bayes_features(
    train, valid, test, train_context, valid_context, test_context,
    labels, weights
):
    train_specs = context_keys(train_context, train)
    valid_specs = context_keys(valid_context, valid)
    test_specs = context_keys(test_context, test)

    global_rate = float(
        np.sum(weights * labels) / np.sum(weights)
    )
    global_logit = np.log(
        global_rate / max(1.0 - global_rate, 1e-8)
    )

    tr_columns = []
    va_columns = []
    te_columns = []
    tr_deviations = []
    va_deviations = []
    te_deviations = []

    for tr_spec, va_spec, te_spec in zip(
        train_specs, valid_specs, test_specs
    ):
        tr_key, cardinality, name = tr_spec
        va_key = va_spec[0]
        te_key = te_spec[0]

        counts = np.bincount(
            tr_key, weights=weights, minlength=cardinality
        ).astype(np.float64)
        positives = np.bincount(
            tr_key, weights=weights * labels,
            minlength=cardinality
        ).astype(np.float64)

        own_w = weights.astype(np.float64)
        loo_count = np.maximum(counts[tr_key] - own_w, 0.0)
        loo_positive = np.maximum(
            positives[tr_key] - own_w * labels, 0.0
        )
        train_rate = (
            loo_positive + SMOOTH * global_rate
        ) / (loo_count + SMOOTH)

        valid_rate = (
            positives[va_key] + SMOOTH * global_rate
        ) / (counts[va_key] + SMOOTH)
        test_rate = (
            positives[te_key] + SMOOTH * global_rate
        ) / (counts[te_key] + SMOOTH)

        train_count = np.log1p(loo_count)
        valid_count = np.log1p(counts[va_key])
        test_count = np.log1p(counts[te_key])

        tr_columns.extend([
            train_rate.astype(np.float32),
            train_count.astype(np.float32),
        ])
        va_columns.extend([
            valid_rate.astype(np.float32),
            valid_count.astype(np.float32),
        ])
        te_columns.extend([
            test_rate.astype(np.float32),
            test_count.astype(np.float32),
        ])

        def deviation(rate):
            rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
            return np.log(rate / (1.0 - rate)) - global_logit

        reliability_tr = loo_count / (loo_count + SMOOTH)
        reliability_va = counts[va_key] / (counts[va_key] + SMOOTH)
        reliability_te = counts[te_key] / (counts[te_key] + SMOOTH)

        tr_deviations.append(
            deviation(train_rate) * reliability_tr
        )
        va_deviations.append(
            deviation(valid_rate) * reliability_va
        )
        te_deviations.append(
            deviation(test_rate) * reliability_te
        )

    eb_train = np.mean(tr_deviations, axis=0).astype(np.float32)
    eb_valid = np.mean(va_deviations, axis=0).astype(np.float32)
    eb_test = np.mean(te_deviations, axis=0).astype(np.float32)

    return (
        tr_columns, va_columns, te_columns,
        eb_train, eb_valid, eb_test,
    )


def build_matrices(train, valid, test):
    labels = np.asarray(train.y, dtype=np.float32)
    weights = recency_weights(train.date)

    train_context = chronological_features(train)
    valid_context = chronological_features(valid)
    test_context = chronological_features(test)

    (
        tr_cols, va_cols, te_cols,
        eb_train, eb_valid, eb_test,
    ) = empirical_bayes_features(
        train, valid, test,
        train_context, valid_context, test_context,
        labels, weights,
    )

    context_names = [
        "user_pos", "user_reverse", "user_size", "user_fraction",
        "day_pos", "day_reverse", "day_size",
        "session_pos", "session_reverse", "session_size",
        "batch_pos", "batch_reverse", "batch_size",
        "log_previous_gap", "log_next_gap",
        "log_elapsed", "log_remaining",
    ]
    for name in context_names:
        tr_cols.append(train_context[name])
        va_cols.append(valid_context[name])
        te_cols.append(test_context[name])

    for entity in ("video_id", "author_id"):
        histories = [
            historical_features("train", key=entity),
            historical_features("valid", key=entity),
            historical_features("test", key=entity),
        ]
        common_names = sorted(histories[0])
        for name in common_names:
            for history, columns in zip(
                histories, (tr_cols, va_cols, te_cols)
            ):
                values = np.asarray(history[name], dtype=np.float32)
                columns.append(np.nan_to_num(
                    values, nan=0.0, posinf=20.0, neginf=-20.0
                ))

    for field in NUM_FIELDS:
        for split, columns in (
            (train, tr_cols), (valid, va_cols), (test, te_cols)
        ):
            values = np.asarray(split.num[field], dtype=np.float32)
            values = np.nan_to_num(
                values, nan=0.0, posinf=1e8, neginf=0.0
            )
            columns.append(np.log1p(np.maximum(values, 0.0)))

    categorical_indices = []
    for field in CAT_FIELDS:
        categorical_indices.append(len(tr_cols))
        tr_cols.append(np.asarray(train.X[field], dtype=np.float32))
        va_cols.append(np.asarray(valid.X[field], dtype=np.float32))
        te_cols.append(np.asarray(test.X[field], dtype=np.float32))

    x_train = np.ascontiguousarray(
        np.column_stack(tr_cols), dtype=np.float32
    )
    x_valid = np.ascontiguousarray(
        np.column_stack(va_cols), dtype=np.float32
    )
    x_test = np.ascontiguousarray(
        np.column_stack(te_cols), dtype=np.float32
    )

    print("FINDINGS " + json.dumps({
        "matrix_dimension": int(x_train.shape[1]),
        "mean_train_user_position": float(
            train_context["user_pos"].mean()
        ),
        "mean_valid_user_position": float(
            valid_context["user_pos"].mean()
        ),
        "mean_train_slate_size": float(
            train_context["user_size"].mean()
        ),
        "mean_valid_slate_size": float(
            valid_context["user_size"].mean()
        ),
        "train_multirow_batch_rate": float(
            np.mean(train_context["batch_size"] > 1)
        ),
        "valid_multirow_batch_rate": float(
            np.mean(valid_context["batch_size"] > 1)
        ),
    }, sort_keys=True))

    return (
        x_train, x_valid, x_test, labels, weights,
        categorical_indices, eb_train, eb_valid, eb_test,
    )


train = load("train")
valid = load("valid")
test = load("test")

(
    x_train, x_valid, x_test, labels, weights,
    categorical_indices, eb_train, eb_valid, eb_test,
) = build_matrices(train, valid, test)

family_valid = {
    "chronological_empirical_bayes": eb_valid.astype(np.float64)
}
family_test = {
    "chronological_empirical_bayes": eb_test.astype(np.float64)
}

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.035,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 180,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.82,
    "bagging_freq": 1,
    "lambda_l1": 0.15,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "num_threads": THREADS,
    "verbose": -1,
}

lgb_train = lgb.Dataset(
    x_train,
    label=labels,
    weight=weights,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
booster = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=420,
)
family_valid["chronological_lightgbm"] = booster.predict(
    x_valid
).astype(np.float64)
family_test["chronological_lightgbm"] = booster.predict(
    x_test
).astype(np.float64)
del booster, lgb_train
gc.collect()

sample_rng = np.random.default_rng(SEED + 50)
sample_size = min(600000, len(labels))
probability = np.asarray(weights, dtype=np.float64)
probability /= probability.sum()
sample_indices = sample_rng.choice(
    len(labels),
    size=sample_size,
    replace=False,
    p=probability,
)

extra_trees = ExtraTreesRegressor(
    n_estimators=128,
    max_depth=20,
    min_samples_leaf=60,
    max_features=0.70,
    bootstrap=False,
    criterion="squared_error",
    n_jobs=THREADS,
    random_state=SEED + 100,
)
extra_trees.fit(
    x_train[sample_indices],
    labels[sample_indices],
    sample_weight=weights[sample_indices],
)
family_valid["chronological_extra_trees"] = extra_trees.predict(
    x_valid
).astype(np.float64)
family_test["chronological_extra_trees"] = extra_trees.predict(
    x_test
).astype(np.float64)
del extra_trees, sample_indices
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

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

candidate_valid = {"trusted_incumbent": inc_valid}
candidate_test = {"trusted_incumbent": inc_test}
candidate_raw_name = {"trusted_incumbent": None}
candidate_metrics = {
    "trusted_incumbent": evaluate(
        valid.user_id, valid.y, inc_valid
    )
}

for name in family_valid:
    raw_valid = family_valid[name]
    raw_test = family_test[name]
    raw_valid_rank = within_user_rank(valid.user_id, raw_valid)
    raw_test_rank = within_user_rank(test.user_id, raw_test)

    candidate_valid[name] = raw_valid
    candidate_test[name] = raw_test
    candidate_raw_name[name] = name
    candidate_metrics[name] = evaluate(
        valid.user_id, valid.y, raw_valid
    )

    for alpha in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        blend_name = f"{name}_incumbent_{alpha:.2f}"
        blend_valid = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * raw_valid_rank
        )
        blend_test = (
            (1.0 - alpha) * inc_test_rank
            + alpha * raw_test_rank
        )
        candidate_valid[blend_name] = blend_valid
        candidate_test[blend_name] = blend_test
        candidate_raw_name[blend_name] = name
        candidate_metrics[blend_name] = evaluate(
            valid.user_id, valid.y, blend_valid
        )

best_name = max(
    candidate_metrics,
    key=lambda key: float(candidate_metrics[key]["primary"]),
)
best_metrics = candidate_metrics[best_name]
best_valid = np.asarray(candidate_valid[best_name], dtype=np.float64)
best_test = np.asarray(candidate_test[best_name], dtype=np.float64)

standalone_best_name = max(
    family_valid,
    key=lambda key: float(candidate_metrics[key]["primary"]),
)
audit_name = candidate_raw_name[best_name]
if audit_name is None:
    audit_name = standalone_best_name
audit_valid = np.asarray(family_valid[audit_name], dtype=np.float64)

rank_correlations = {}
family_names = list(family_valid)
for i in range(len(family_names)):
    for j in range(i + 1, len(family_names)):
        left = family_names[i]
        right = family_names[j]
        rank_correlations[f"{left}__{right}"] = float(np.corrcoef(
            within_user_rank(valid.user_id, family_valid[left]),
            within_user_rank(valid.user_id, family_valid[right]),
        )[0, 1])

print("CANDIDATES " + json.dumps({
    name: float(metrics["primary"])
    for name, metrics in candidate_metrics.items()
}, sort_keys=True))

print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "best_standalone": standalone_best_name,
    "rank_correlations": rank_correlations,
    "half_life_days": HALF_LIFE,
    "empirical_bayes_smoothing": SMOOTH,
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
        best_test,
    )
    if best_name == "trusted_incumbent" or "_incumbent_" in best_name:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            audit_valid,
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))