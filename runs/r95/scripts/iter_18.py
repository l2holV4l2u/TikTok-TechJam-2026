import os
import time
import json
import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

START = time.time()
np.random.seed(2026)

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.float32)
yva = np.asarray(valid.y, dtype=np.int8)
uva = np.asarray(valid.user_id, dtype=np.int64)
ute = np.asarray(test.user_id, dtype=np.int64)

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "user_active_degree",
    "register_days_bucket",
    "music_type",
    "video_type",
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


def safe_log_numeric(a):
    a = np.asarray(a, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = np.maximum(a, 0.0)
    return np.log1p(a).astype(np.float32)


# Historical features are computed by the provided train-only implementation:
# train values are leave-one-out and validation/test values use full train.
history_cache = {}
for split_name in ["train", "valid", "test"]:
    history_cache[split_name] = {}
    for entity in ["video_id", "author_id"]:
        h = historical_features(split_name, key=entity)
        history_cache[split_name][entity] = h

common_history_names = {}
for entity in ["video_id", "author_id"]:
    names = set(history_cache["train"][entity].keys())
    names &= set(history_cache["valid"][entity].keys())
    names &= set(history_cache["test"][entity].keys())
    common_history_names[entity] = sorted(names)


def make_matrix(sample, split_name):
    columns = []

    for field in CAT_FIELDS:
        columns.append(np.asarray(sample.X[field], dtype=np.float32))

    for field in NUM_FIELDS:
        raw = np.asarray(sample.num[field], dtype=np.float32)
        missing = (~np.isfinite(raw)).astype(np.float32)
        columns.append(safe_log_numeric(raw))
        columns.append(missing)

    for entity in ["video_id", "author_id"]:
        for name in common_history_names[entity]:
            a = np.asarray(
                history_cache[split_name][entity][name],
                dtype=np.float32,
            )
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            # Counts and rates can differ greatly in scale. Signed log scaling
            # preserves ordering while limiting extreme entity histories.
            a = np.sign(a) * np.log1p(np.abs(a))
            columns.append(a.astype(np.float32))

    return np.column_stack(columns).astype(np.float32, copy=False)


Xtr = make_matrix(train, "train")
Xva = make_matrix(valid, "valid")
Xte = make_matrix(test, "test")

categorical_indices = list(range(len(CAT_FIELDS)))

# Main-model temporal weighting. Recent observations better represent the
# immediately following validation and test periods.
dates = np.asarray(train.date, dtype=np.int64)
unique_dates = np.unique(dates)
date_to_index = {int(d): i for i, d in enumerate(unique_dates)}
day_index = np.fromiter(
    (date_to_index[int(d)] for d in dates),
    dtype=np.int32,
    count=len(dates),
)
age = (len(unique_dates) - 1 - day_index).astype(np.float32)
sample_weight = np.exp2(-age / 4.0).astype(np.float32)
sample_weight /= np.mean(sample_weight)

play_name = None
for candidate in [
    "play_time_ms",
    "play_time",
    "playing_time_ms",
    "playing_time",
]:
    if candidate in train.aux:
        arr = np.asarray(train.aux[candidate])
        if len(arr) == len(ytr):
            play_name = candidate
            break

if play_name is None:
    # Keeps the experiment runnable on builds with renamed auxiliary fields.
    play_ms = ytr * np.maximum(
        np.asarray(train.num["duration_ms"], dtype=np.float32),
        1000.0,
    )
else:
    play_ms = np.asarray(train.aux[play_name], dtype=np.float32)

play_ms = np.nan_to_num(
    play_ms,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)
play_ms = np.maximum(play_ms, 0.0)

duration_tr = np.asarray(train.num["duration_ms"], dtype=np.float32)
duration_tr = np.nan_to_num(
    duration_tr,
    nan=1000.0,
    posinf=1000.0,
    neginf=1000.0,
)
duration_tr = np.maximum(duration_tr, 1000.0)

# Robust clipping is fitted on train only.
play_cap = float(np.quantile(play_ms, 0.999))
play_ms_clipped = np.minimum(play_ms, max(play_cap, 1.0))

# Accelerated-failure-time target: dense log watch duration.
watch_target = np.log1p(play_ms_clipped / 1000.0).astype(np.float32)

# Completion target separates the propensity to watch from video length.
# Clipping limits looping videos and logging outliers.
completion = play_ms_clipped / duration_tr
completion_cap = float(np.quantile(completion, 0.999))
completion = np.minimum(completion, max(completion_cap, 1.0))
completion_target = np.log1p(completion).astype(np.float32)

print(
    "FINDINGS "
    + json.dumps({
        "play_target": play_name,
        "play_ms_mean": float(np.mean(play_ms)),
        "play_ms_cap_999": play_cap,
        "completion_mean": float(np.mean(completion)),
        "history_features": {
            k: common_history_names[k] for k in common_history_names
        },
        "matrix_shape": list(Xtr.shape),
    }, sort_keys=True)
)

common_params = {
    "boosting_type": "gbdt",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 1.0,
    "max_bin": 127,
    "num_threads": min(12, os.cpu_count() or 12),
    "seed": 2026,
    "feature_fraction_seed": 2027,
    "bagging_seed": 2028,
    "data_random_seed": 2029,
    "verbose": -1,
}


def fit_predict(target, objective, metric, rounds):
    params = dict(common_params)
    params["objective"] = objective
    params["metric"] = metric

    dtrain = lgb.Dataset(
        Xtr,
        label=np.asarray(target, dtype=np.float32),
        weight=sample_weight,
        categorical_feature=categorical_indices,
        free_raw_data=False,
    )
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=rounds,
    )
    va = np.asarray(model.predict(Xva), dtype=np.float64)
    te = np.asarray(model.predict(Xte), dtype=np.float64)
    return va, te


direct_valid, direct_test = fit_predict(
    ytr,
    objective="binary",
    metric="binary_logloss",
    rounds=240,
)

watch_valid, watch_test = fit_predict(
    watch_target,
    objective="regression_l1",
    metric="l1",
    rounds=210,
)

completion_valid, completion_test = fit_predict(
    completion_target,
    objective="regression_l1",
    metric="l1",
    rounds=210,
)


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    row = np.arange(len(scores), dtype=np.int64)

    order = np.lexsort((row, scores, users))
    ordered_users = users[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_users[1:] != ordered_users[:-1]]
    )
    ends = np.r_[starts[1:], len(order)]
    lengths = ends - starts

    positions = (
        np.arange(len(order), dtype=np.float64)
        - np.repeat(starts, lengths)
    )
    denominators = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    ordered_ranks = positions / denominators

    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = ordered_ranks
    return ranks


direct_rank_valid = within_user_rank(uva, direct_valid)
direct_rank_test = within_user_rank(ute, direct_test)
watch_rank_valid = within_user_rank(uva, watch_valid)
watch_rank_test = within_user_rank(ute, watch_test)
completion_rank_valid = within_user_rank(uva, completion_valid)
completion_rank_test = within_user_rank(ute, completion_test)

# These hybrids aggregate structurally different notions of relevance:
# direct probability, absolute watch duration, and duration-normalized watch.
families_valid = {
    "direct_long_view_gbdt": direct_rank_valid,
    "watch_time_aft": watch_rank_valid,
    "completion_ratio": completion_rank_valid,
    "direct_watch_hybrid":
        0.60 * direct_rank_valid + 0.40 * watch_rank_valid,
    "direct_completion_hybrid":
        0.60 * direct_rank_valid + 0.40 * completion_rank_valid,
    "three_signal_hybrid":
        0.50 * direct_rank_valid
        + 0.25 * watch_rank_valid
        + 0.25 * completion_rank_valid,
}

families_test = {
    "direct_long_view_gbdt": direct_rank_test,
    "watch_time_aft": watch_rank_test,
    "completion_ratio": completion_rank_test,
    "direct_watch_hybrid":
        0.60 * direct_rank_test + 0.40 * watch_rank_test,
    "direct_completion_hybrid":
        0.60 * direct_rank_test + 0.40 * completion_rank_test,
    "three_signal_hybrid":
        0.50 * direct_rank_test
        + 0.25 * watch_rank_test
        + 0.25 * completion_rank_test,
}

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.asarray(
    np.load(os.path.join(shared_dir, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(shared_dir, "incumbent_test_scores.npy")),
    dtype=np.float64,
)
inc_rank_valid = within_user_rank(uva, inc_valid)
inc_rank_test = within_user_rank(ute, inc_test)

candidate_scores = {}
best_primary = -np.inf
best_metrics = None
best_valid = None
best_test = None
best_raw = None
best_name = None

# The trusted-incumbent contract permits selecting these blend weights on
# validation and applying the identical selected weight to test.
alphas = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]

for name, own_valid in families_valid.items():
    own_test = families_test[name]
    standalone = evaluate(uva, yva, own_valid)
    candidate_scores[name + "_standalone"] = float(
        standalone["primary"]
    )

    for alpha in alphas:
        blend_valid = (
            (1.0 - alpha) * inc_rank_valid + alpha * own_valid
        )
        blend_test = (
            (1.0 - alpha) * inc_rank_test + alpha * own_test
        )
        metrics = evaluate(uva, yva, blend_valid)
        primary = float(metrics["primary"])
        candidate_name = "%s_blend_%.2f" % (name, alpha)
        candidate_scores[candidate_name] = primary

        if primary > best_primary:
            best_primary = primary
            best_metrics = metrics
            best_valid = blend_valid.copy()
            best_test = blend_test.copy()
            best_raw = own_valid.copy()
            best_name = candidate_name

correlations = {
    "direct_watch": float(np.corrcoef(
        direct_rank_valid, watch_rank_valid
    )[0, 1]),
    "direct_completion": float(np.corrcoef(
        direct_rank_valid, completion_rank_valid
    )[0, 1]),
    "watch_completion": float(np.corrcoef(
        watch_rank_valid, completion_rank_valid
    )[0, 1]),
    "incumbent_direct": float(np.corrcoef(
        inc_rank_valid, direct_rank_valid
    )[0, 1]),
    "incumbent_watch": float(np.corrcoef(
        inc_rank_valid, watch_rank_valid
    )[0, 1]),
    "incumbent_completion": float(np.corrcoef(
        inc_rank_valid, completion_rank_valid
    )[0, 1]),
}

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps({
        "selected": best_name,
        "selected_primary": best_primary,
        "rank_correlations": correlations,
    }, sort_keys=True)
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)