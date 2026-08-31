import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from sklearn.ensemble import ExtraTreesClassifier
from pipeline.data import load
from pipeline.evaluate import evaluate


START = time.time()
SEED = 20260830

CAT_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "music_type",
    "video_type",
    "hour",
    "is_video_author",
    "user_active_degree",
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


def day_number(date):
    date = np.asarray(date, dtype=np.int32)
    month = (date // 100) % 100
    day = date % 100
    return day + np.where(month >= 5, 30, 0)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    su = user_ids[order]

    boundary = np.empty(n, dtype=bool)
    boundary[0] = True
    boundary[1:] = su[1:] != su[:-1]

    starts = np.maximum.accumulate(
        np.where(boundary, np.arange(n, dtype=np.int64), 0)
    )
    pos = np.arange(n, dtype=np.int64) - starts

    _, counts = np.unique(su, return_counts=True)
    group_id = np.cumsum(boundary, dtype=np.int64) - 1
    denominator = np.maximum(counts[group_id] - 1, 1)

    result = np.empty(n, dtype=np.float32)
    result[order] = (pos / denominator).astype(np.float32)
    return result


def chronology_features(split):
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    n = len(users)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, times, users))
    su = users[order]
    st = times[order]

    user_boundary = np.empty(n, dtype=bool)
    user_boundary[0] = True
    user_boundary[1:] = su[1:] != su[:-1]

    user_starts = np.maximum.accumulate(
        np.where(user_boundary, np.arange(n, dtype=np.int64), 0)
    )
    user_pos = np.arange(n, dtype=np.int64) - user_starts
    user_gid = np.cumsum(user_boundary, dtype=np.int64) - 1
    user_sizes = np.bincount(user_gid)
    user_reverse = user_sizes[user_gid] - 1 - user_pos
    user_den = np.maximum(user_sizes[user_gid] - 1, 1)

    gap_prev_ms = np.zeros(n, dtype=np.float64)
    gap_prev_ms[1:] = np.maximum(st[1:] - st[:-1], 0)
    gap_prev_ms[user_boundary] = 0.0

    gap_next_ms = np.zeros(n, dtype=np.float64)
    gap_next_ms[:-1] = np.maximum(st[1:] - st[:-1], 0)
    end_user = np.empty(n, dtype=bool)
    end_user[-1] = True
    end_user[:-1] = su[:-1] != su[1:]
    gap_next_ms[end_user] = 0.0

    session_boundary = user_boundary | (gap_prev_ms > 30.0 * 60.0 * 1000.0)
    session_starts = np.maximum.accumulate(
        np.where(session_boundary, np.arange(n, dtype=np.int64), 0)
    )
    session_pos = np.arange(n, dtype=np.int64) - session_starts
    session_gid = np.cumsum(session_boundary, dtype=np.int64) - 1
    session_sizes = np.bincount(session_gid)
    session_reverse = session_sizes[session_gid] - 1 - session_pos
    session_den = np.maximum(session_sizes[session_gid] - 1, 1)

    batch_boundary = np.empty(n, dtype=bool)
    batch_boundary[0] = True
    batch_boundary[1:] = (
        (su[1:] != su[:-1]) | (st[1:] != st[:-1])
    )
    batch_starts = np.maximum.accumulate(
        np.where(batch_boundary, np.arange(n, dtype=np.int64), 0)
    )
    batch_pos = np.arange(n, dtype=np.int64) - batch_starts
    batch_gid = np.cumsum(batch_boundary, dtype=np.int64) - 1
    batch_sizes = np.bincount(batch_gid)
    batch_reverse = batch_sizes[batch_gid] - 1 - batch_pos
    batch_den = np.maximum(batch_sizes[batch_gid] - 1, 1)

    sorted_features = np.column_stack([
        user_pos.astype(np.float32),
        user_reverse.astype(np.float32),
        user_pos.astype(np.float32) / user_den.astype(np.float32),
        user_reverse.astype(np.float32) / user_den.astype(np.float32),
        np.log1p(user_sizes[user_gid]).astype(np.float32),

        session_pos.astype(np.float32),
        session_reverse.astype(np.float32),
        session_pos.astype(np.float32) / session_den.astype(np.float32),
        session_reverse.astype(np.float32) / session_den.astype(np.float32),
        np.log1p(session_sizes[session_gid]).astype(np.float32),

        batch_pos.astype(np.float32),
        batch_reverse.astype(np.float32),
        batch_pos.astype(np.float32) / batch_den.astype(np.float32),
        batch_reverse.astype(np.float32) / batch_den.astype(np.float32),
        np.log1p(batch_sizes[batch_gid]).astype(np.float32),

        np.log1p(gap_prev_ms / 1000.0).astype(np.float32),
        np.log1p(gap_next_ms / 1000.0).astype(np.float32),
        (gap_prev_ms > 5.0 * 60.0 * 1000.0).astype(np.float32),
        (gap_next_ms > 5.0 * 60.0 * 1000.0).astype(np.float32),
    ]).astype(np.float32, copy=False)

    result = np.empty_like(sorted_features)
    result[order] = sorted_features

    heuristics = {
        "early_user": -result[:, 2],
        "late_user": result[:, 2],
        "early_session": -result[:, 7],
        "late_session": result[:, 7],
        "early_batch": -result[:, 12],
        "gap_before": result[:, 15],
        "gap_after": result[:, 16],
    }
    return result, heuristics


def build_matrix(split):
    chronology, heuristics = chronology_features(split)

    cat_columns = []
    for name in CAT_FIELDS:
        if name == "video_id":
            x = np.asarray(split.video_id, dtype=np.float32)
        else:
            x = np.asarray(split.X[name], dtype=np.float32)
        cat_columns.append(x)

    numeric_columns = []
    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.where(np.isfinite(x), x, np.nan)
        x = np.log1p(np.maximum(x, 0.0)).astype(np.float32)
        numeric_columns.append(x)

    date = np.asarray(split.date, dtype=np.int32)
    hour = np.asarray(split.X["hour"], dtype=np.float32)
    calendar = np.column_stack([
        ((date % 100) % 7).astype(np.float32),
        np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32),
        np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32),
    ])

    matrix = np.column_stack(
        cat_columns + numeric_columns + [chronology, calendar]
    ).astype(np.float32, copy=False)
    return matrix, heuristics


def temporal_weights(date, half_life=7.0):
    d = day_number(date).astype(np.float64)
    endpoint = float(np.max(d))
    return np.exp2(-(endpoint - d) / half_life).astype(np.float32)


def train_lgb(X, y, date):
    weights = temporal_weights(date, half_life=7.0)
    dtrain = lgb.Dataset(
        X,
        label=np.asarray(y, dtype=np.float32),
        weight=weights,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.055,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 180,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 0.3,
        "lambda_l2": 3.0,
        "max_bin": 127,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "num_threads": 8,
        "verbose": -1,
    }
    return lgb.train(params, dtrain, num_boost_round=230)


def extra_sample_indices(date, max_rows=420000):
    n = len(date)
    if n <= max_rows:
        return np.arange(n, dtype=np.int64)

    d = day_number(date)
    endpoint = np.max(d)
    recent = np.flatnonzero(d >= endpoint - 5)
    if len(recent) >= max_rows:
        return recent[-max_rows:]

    older = np.flatnonzero(d < endpoint - 5)
    need = max_rows - len(recent)
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(older, size=need, replace=False)
    return np.concatenate([recent, chosen])


def train_extra(X, y, date):
    indices = extra_sample_indices(date)
    weights = temporal_weights(date, half_life=7.0)[indices]
    X_fit = np.nan_to_num(
        X[indices], nan=-1.0, posinf=20.0, neginf=-1.0
    )
    model = ExtraTreesClassifier(
        n_estimators=110,
        criterion="entropy",
        max_depth=20,
        min_samples_leaf=24,
        max_features=0.72,
        bootstrap=False,
        class_weight=None,
        n_jobs=8,
        random_state=SEED,
    )
    model.fit(
        X_fit,
        np.asarray(y, dtype=np.int8)[indices],
        sample_weight=weights,
    )
    return model


def predict_extra(model, X):
    clean = np.nan_to_num(X, nan=-1.0, posinf=20.0, neginf=-1.0)
    return model.predict_proba(clean)[:, 1].astype(np.float32)


def concatenate_training(train, valid, X_train, X_valid):
    X = np.concatenate([X_train, X_valid], axis=0)
    y = np.concatenate([
        np.asarray(train.y, dtype=np.int8),
        np.asarray(valid.y, dtype=np.int8),
    ])
    date = np.concatenate([
        np.asarray(train.date, dtype=np.int32),
        np.asarray(valid.date, dtype=np.int32),
    ])
    return X, y, date


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float32, copy=False)
inc_rank_valid = within_user_rank(valid.user_id, inc_valid)

X_train, train_heuristics = build_matrix(train)
X_valid, valid_heuristics = build_matrix(valid)

candidate_raw = {}

lgb_model = train_lgb(X_train, y_train, train.date)
lgb_valid = lgb_model.predict(X_valid).astype(np.float32)
candidate_raw["session_lgb_binary"] = lgb_valid

extra_model = train_extra(X_train, y_train, train.date)
extra_valid = predict_extra(extra_model, X_valid)
candidate_raw["session_extra_trees"] = extra_valid

for name, values in valid_heuristics.items():
    candidate_raw["heuristic_" + name] = np.asarray(values, dtype=np.float32)

# Two structurally different model outputs can also complement each other.
candidate_raw["lgb_extra_rank_average"] = (
    0.5 * within_user_rank(valid.user_id, lgb_valid)
    + 0.5 * within_user_rank(valid.user_id, extra_valid)
).astype(np.float32)

candidate_log = {
    "trusted_incumbent": float(
        evaluate(valid.user_id, y_valid, inc_valid)["primary"]
    )
}

best = {
    "primary": candidate_log["trusted_incumbent"],
    "name": "trusted_incumbent",
    "alpha": 0.0,
    "scores": inc_valid.copy(),
}

alpha_grid = np.linspace(0.0, 0.80, 17)

for name, raw in candidate_raw.items():
    rank = within_user_rank(valid.user_id, raw)
    standalone = float(
        evaluate(valid.user_id, y_valid, rank)["primary"]
    )
    candidate_log[name + "_standalone"] = standalone

    local_best = -np.inf
    local_alpha = 0.0
    local_scores = None
    for alpha in alpha_grid:
        scores = (
            (1.0 - float(alpha)) * inc_rank_valid
            + float(alpha) * rank
        ).astype(np.float32)
        primary = float(
            evaluate(valid.user_id, y_valid, scores)["primary"]
        )
        if primary > local_best:
            local_best = primary
            local_alpha = float(alpha)
            local_scores = scores.copy()

    candidate_log[name + "_blend"] = local_best

    if local_best > best["primary"]:
        best = {
            "primary": local_best,
            "name": name,
            "alpha": local_alpha,
            "scores": local_scores,
        }

valid_scores = np.asarray(best["scores"], dtype=np.float32)
metrics = evaluate(valid.user_id, y_valid, valid_scores)

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print("FINDINGS " + json.dumps({
    "selected_family": best["name"],
    "selected_alpha": best["alpha"],
    "lgb_extra_rank_correlation": float(np.corrcoef(
        within_user_rank(valid.user_id, lgb_valid),
        within_user_rank(valid.user_id, extra_valid),
    )[0, 1]),
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

test = load("test")
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float32, copy=False)

if best["name"] == "trusted_incumbent" or best["alpha"] <= 0.0:
    test_scores = inc_test
else:
    X_test, test_heuristics = build_matrix(test)
    inc_rank_test = within_user_rank(test.user_id, inc_test)

    if best["name"] == "session_lgb_binary":
        X_combined, y_combined, date_combined = concatenate_training(
            train, valid, X_train, X_valid
        )
        del lgb_model
        gc.collect()
        final_model = train_lgb(X_combined, y_combined, date_combined)
        raw_test = final_model.predict(X_test).astype(np.float32)

    elif best["name"] == "session_extra_trees":
        X_combined, y_combined, date_combined = concatenate_training(
            train, valid, X_train, X_valid
        )
        del extra_model
        gc.collect()
        final_model = train_extra(X_combined, y_combined, date_combined)
        raw_test = predict_extra(final_model, X_test)

    elif best["name"] == "lgb_extra_rank_average":
        X_combined, y_combined, date_combined = concatenate_training(
            train, valid, X_train, X_valid
        )
        del lgb_model
        del extra_model
        gc.collect()

        final_lgb = train_lgb(X_combined, y_combined, date_combined)
        pred_lgb = final_lgb.predict(X_test).astype(np.float32)

        final_extra = train_extra(X_combined, y_combined, date_combined)
        pred_extra = predict_extra(final_extra, X_test)

        raw_test = (
            0.5 * within_user_rank(test.user_id, pred_lgb)
            + 0.5 * within_user_rank(test.user_id, pred_extra)
        ).astype(np.float32)

    elif best["name"].startswith("heuristic_"):
        heuristic_name = best["name"][len("heuristic_"):]
        raw_test = np.asarray(
            test_heuristics[heuristic_name], dtype=np.float32
        )

    else:
        raise RuntimeError("Unknown selected family: " + best["name"])

    new_rank_test = within_user_rank(test.user_id, raw_test)
    test_scores = (
        (1.0 - best["alpha"]) * inc_rank_test
        + best["alpha"] * new_rank_test
    ).astype(np.float32)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))