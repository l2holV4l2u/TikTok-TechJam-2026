import os
import gc
import json
import time
import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START_TIME = time.time()
OUT_DIR = os.environ.get("ITER_OUT")
ARTIFACTS = os.environ.get("RUN_ARTIFACTS", "")

if OUT_DIR:
    os.makedirs(OUT_DIR, exist_ok=True)

INC_VALID_PATH = os.path.join(
    ARTIFACTS, "incumbent_valid_scores.npy"
)
INC_TEST_PATH = os.path.join(
    ARTIFACTS, "incumbent_test_scores.npy"
)

if not (
    ARTIFACTS
    and os.path.exists(INC_VALID_PATH)
    and os.path.exists(INC_TEST_PATH)
):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "hour",
    "video_type",
    "user_active_degree",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "register_days_bucket",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

BLEND_WEIGHTS = [
    0.0, 0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90, 1.0,
]


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    return (x - mu) / sd


def temporal_features(split):
    """
    Construct label-free feed-context features from the logged ordering.

    Rows are sorted by user, date, timestamp, and original row index. Features
    describe position and spacing inside a user-day and inside tied feed
    batches. Future outcomes are never used.
    """
    users = np.asarray(split.user_id, dtype=np.int64)
    dates = np.asarray(split.date, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    n = len(users)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, times, dates, users))
    u = users[order]
    d = dates[order]
    t = times[order]

    new_day_group = np.empty(n, dtype=bool)
    new_day_group[0] = True
    new_day_group[1:] = (
        (u[1:] != u[:-1]) | (d[1:] != d[:-1])
    )

    group_starts = np.flatnonzero(new_day_group)
    group_ends = np.r_[group_starts[1:], n]
    group_sizes = group_ends - group_starts

    day_position = (
        np.arange(n, dtype=np.int64)
        - np.repeat(group_starts, group_sizes)
    )
    day_size = np.repeat(group_sizes, group_sizes)

    gap_seconds = np.zeros(n, dtype=np.float64)
    same_group = ~new_day_group
    raw_gap = (t[1:] - t[:-1]).astype(np.float64) / 1000.0
    gap_seconds[1:] = np.where(
        same_group[1:], np.maximum(raw_gap, 0.0), 0.0
    )

    new_batch = np.empty(n, dtype=bool)
    new_batch[0] = True
    new_batch[1:] = (
        new_day_group[1:] | (t[1:] != t[:-1])
    )
    batch_starts = np.flatnonzero(new_batch)
    batch_ends = np.r_[batch_starts[1:], n]
    batch_sizes = batch_ends - batch_starts
    batch_position = (
        np.arange(n, dtype=np.int64)
        - np.repeat(batch_starts, batch_sizes)
    )
    batch_size = np.repeat(batch_sizes, batch_sizes)

    min_date = int(np.min(dates))
    date_ordinal = (dates - min_date).astype(np.float32)

    sorted_features = np.column_stack([
        np.log1p(day_position).astype(np.float32),
        np.log1p(day_size).astype(np.float32),
        (
            day_position / np.maximum(day_size - 1, 1)
        ).astype(np.float32),
        np.log1p(np.minimum(gap_seconds, 86400.0)).astype(
            np.float32
        ),
        np.log1p(batch_position).astype(np.float32),
        np.log1p(batch_size).astype(np.float32),
    ])

    result = np.empty((n, sorted_features.shape[1] + 1),
                      dtype=np.float32)
    result[order, :-1] = sorted_features
    result[:, -1] = date_ordinal
    return result


def safe_numeric(x):
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if np.any(finite):
        med = float(np.median(x[finite]))
    else:
        med = 0.0
    x = np.where(finite, x, med)
    return np.log1p(np.maximum(x, 0.0)).astype(np.float32)


def load_history_matrix(split_name):
    columns = []
    names = []

    for entity in ("video_id", "author_id"):
        hist = historical_features(split_name, key=entity)
        for key in sorted(hist.keys()):
            x = np.asarray(hist[key], dtype=np.float32)
            x = np.nan_to_num(
                x, nan=0.0, posinf=0.0, neginf=0.0
            )
            columns.append(x)
            names.append(entity + ":" + key)

    if not columns:
        return np.empty((0, 0), dtype=np.float32), names
    return np.column_stack(columns).astype(np.float32), names


def make_matrix(split, split_name):
    cat = np.column_stack([
        np.asarray(split.X[field], dtype=np.float32)
        for field in CAT_FIELDS
    ])

    num = np.column_stack([
        safe_numeric(split.num[field])
        for field in NUM_FIELDS
    ])

    temp = temporal_features(split)
    hist, hist_names = load_history_matrix(split_name)

    if len(hist) != len(cat):
        raise RuntimeError(
            "Historical feature row count does not match split"
        )

    matrix = np.column_stack([cat, num, temp, hist]).astype(
        np.float32
    )
    return matrix, hist_names


def sorted_groups(users):
    users = np.asarray(users, dtype=np.int64)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    starts = np.r_[
        0,
        1 + np.flatnonzero(sorted_users[1:] != sorted_users[:-1]),
    ]
    groups = np.diff(np.r_[starts, len(users)]).astype(np.int32)
    return order, groups


def train_binary(X_train, y_train, dates_train,
                 X_valid=None, y_valid=None):
    max_date = int(np.max(dates_train))
    age = (max_date - np.asarray(dates_train)).astype(np.float64)
    weights = np.power(0.5, age / 7.0).astype(np.float32)

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        weight=weights,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.045,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 120,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "min_data_per_group": 80,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "max_cat_threshold": 32,
        "num_threads": max(1, os.cpu_count() or 1),
        "seed": 2026,
        "feature_fraction_seed": 2027,
        "bagging_seed": 2028,
        "verbose": -1,
    }

    if X_valid is None:
        model = lgb.train(
            params, dtrain, num_boost_round=train_binary.rounds
        )
    else:
        dvalid = lgb.Dataset(
            X_valid,
            label=y_valid,
            categorical_feature=list(range(len(CAT_FIELDS))),
            reference=dtrain,
            free_raw_data=False,
        )
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=420,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(45, verbose=False)],
        )
        train_binary.rounds = max(30, int(model.best_iteration))

    return model


train_binary.rounds = 250


def train_ranker(X_train, y_train, users_train,
                 X_valid=None, y_valid=None, users_valid=None):
    train_order, train_groups = sorted_groups(users_train)

    dtrain = lgb.Dataset(
        X_train[train_order],
        label=np.asarray(y_train)[train_order],
        group=train_groups,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5],
        "label_gain": [0, 1],
        "learning_rate": 0.04,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "min_data_per_group": 80,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "max_cat_threshold": 32,
        "lambdarank_truncation_level": 10,
        "num_threads": max(1, os.cpu_count() or 1),
        "seed": 3031,
        "feature_fraction_seed": 3032,
        "bagging_seed": 3033,
        "verbose": -1,
    }

    if X_valid is None:
        model = lgb.train(
            params, dtrain, num_boost_round=train_ranker.rounds
        )
    else:
        valid_order, valid_groups = sorted_groups(users_valid)
        dvalid = lgb.Dataset(
            X_valid[valid_order],
            label=np.asarray(y_valid)[valid_order],
            group=valid_groups,
            categorical_feature=list(range(len(CAT_FIELDS))),
            reference=dtrain,
            free_raw_data=False,
        )
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=420,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(45, verbose=False)],
        )
        train_ranker.rounds = max(30, int(model.best_iteration))

    return model


train_ranker.rounds = 250


def find_best_fusion(user_ids, labels, incumbent, candidates):
    incumbent_z = standardize(incumbent)
    base_metrics = evaluate(user_ids, labels, incumbent_z)

    best_scores = incumbent_z.copy()
    best_metrics = base_metrics
    best_model = "incumbent"
    best_weight = 0.0
    summary = {
        "incumbent": round(float(base_metrics["primary"]), 6)
    }

    for name, raw_scores in candidates.items():
        candidate_z = standardize(raw_scores)
        standalone = evaluate(user_ids, labels, candidate_z)
        summary[name + "_standalone"] = round(
            float(standalone["primary"]), 6
        )

        model_best = float(standalone["primary"])
        model_best_weight = 1.0

        for weight in BLEND_WEIGHTS:
            scores = (
                (1.0 - float(weight)) * incumbent_z
                + float(weight) * candidate_z
            )
            metrics = evaluate(user_ids, labels, scores)
            primary = float(metrics["primary"])

            if primary > model_best:
                model_best = primary
                model_best_weight = float(weight)

            if primary > float(best_metrics["primary"]):
                best_scores = np.asarray(
                    scores, dtype=np.float64
                ).copy()
                best_metrics = metrics
                best_model = name
                best_weight = float(weight)

        summary[name + "_best_fusion"] = round(model_best, 6)
        summary[name + "_best_weight"] = model_best_weight

    return (
        best_scores,
        best_metrics,
        best_model,
        best_weight,
        summary,
    )


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)

inc_valid = np.asarray(
    np.load(INC_VALID_PATH), dtype=np.float64
)
if len(inc_valid) != len(y_valid):
    raise RuntimeError("Invalid incumbent validation row count")

X_train, hist_names_train = make_matrix(train, "train")
X_valid, hist_names_valid = make_matrix(valid, "valid")

if hist_names_train != hist_names_valid:
    raise RuntimeError("Historical feature schemas differ")

binary_model = train_binary(
    X_train,
    y_train,
    np.asarray(train.date),
    X_valid,
    y_valid,
)
binary_valid = binary_model.predict(
    X_valid, num_iteration=binary_model.best_iteration
)

rank_model = train_ranker(
    X_train,
    y_train,
    np.asarray(train.user_id),
    X_valid,
    y_valid,
    np.asarray(valid.user_id),
)
rank_valid = rank_model.predict(
    X_valid, num_iteration=rank_model.best_iteration
)

(
    best_valid_scores,
    best_metrics,
    selected_model,
    selected_weight,
    candidate_summary,
) = find_best_fusion(
    np.asarray(valid.user_id),
    y_valid,
    inc_valid,
    {
        "temporal_binary_lgbm": binary_valid,
        "temporal_lambdarank": rank_valid,
    },
)

print(
    "FINDINGS "
    + json.dumps(
        {
            "selected_model": selected_model,
            "selected_candidate_weight": selected_weight,
            "binary_rounds": int(train_binary.rounds),
            "ranker_rounds": int(train_ranker.rounds),
            "n_features": int(X_train.shape[1]),
            "n_categorical": len(CAT_FIELDS),
            "n_historical": len(hist_names_train),
        },
        sort_keys=True,
    )
)
print(
    "CANDIDATES "
    + json.dumps(candidate_summary, sort_keys=True)
)

if OUT_DIR:
    np.save(
        os.path.join(OUT_DIR, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )

# Release validation models before constructing the larger refit table.
del binary_valid, rank_valid, binary_model, rank_model
gc.collect()

test = load("test")
inc_test = np.asarray(
    np.load(INC_TEST_PATH), dtype=np.float64
)
if len(inc_test) != len(test.user_id):
    raise RuntimeError("Invalid incumbent test row count")

inc_test_z = standardize(inc_test)

if selected_model == "incumbent" or selected_weight == 0.0:
    test_scores = inc_test_z
else:
    X_test, hist_names_test = make_matrix(test, "test")
    if hist_names_test != hist_names_train:
        raise RuntimeError("Test historical feature schema differs")

    X_combined = np.concatenate([X_train, X_valid], axis=0)
    y_combined = np.concatenate([y_train, y_valid], axis=0)
    users_combined = np.concatenate([
        np.asarray(train.user_id, dtype=np.int64),
        np.asarray(valid.user_id, dtype=np.int64),
    ])
    dates_combined = np.concatenate([
        np.asarray(train.date, dtype=np.int32),
        np.asarray(valid.date, dtype=np.int32),
    ])

    del X_train, X_valid
    gc.collect()

    if selected_model == "temporal_binary_lgbm":
        final_model = train_binary(
            X_combined,
            y_combined,
            dates_combined,
        )
    elif selected_model == "temporal_lambdarank":
        final_model = train_ranker(
            X_combined,
            y_combined,
            users_combined,
        )
    else:
        raise RuntimeError("Unknown selected model")

    candidate_test = final_model.predict(X_test)
    candidate_test_z = standardize(candidate_test)

    test_scores = (
        (1.0 - float(selected_weight)) * inc_test_z
        + float(selected_weight) * candidate_test_z
    )

if OUT_DIR:
    np.save(
        os.path.join(OUT_DIR, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START_TIME)
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": elapsed,
        }
    )
)