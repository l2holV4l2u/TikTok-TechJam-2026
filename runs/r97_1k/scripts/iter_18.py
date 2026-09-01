import os
import gc
import json
import time
import warnings
import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")

START = time.time()
OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")

if OUT:
    os.makedirs(OUT, exist_ok=True)

if not SHARED:
    raise RuntimeError("SHARED_ARTIFACTS is required")

INC_VALID_PATH = os.path.join(SHARED, "incumbent_valid_scores.npy")
INC_TEST_PATH = os.path.join(SHARED, "incumbent_test_scores.npy")

if not os.path.exists(INC_VALID_PATH) or not os.path.exists(INC_TEST_PATH):
    raise RuntimeError("Trusted incumbent predictions are missing")


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    starts_at = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local = np.arange(n, dtype=np.float32) - starts_at.astype(np.float32)

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_pos = np.flatnonzero(ends)
    sizes = np.diff(np.r_[-1, end_pos]).astype(np.float32)

    groups = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[groups] - 1.0, 1.0)

    result = np.empty(n, dtype=np.float32)
    result[order] = local / denom
    return result


CAT_FIELDS = [
    "user_id",
    "author_id",
    "video_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat1",
    "onehot_feat7",
    "music_type",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_bucket",
    "hour",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

HISTORY_SUFFIXES = (
    "train_count_log1p",
    "long_view_rate",
    "is_click_rate",
    "play_time_ms_logmean",
    "comment_stay_time_logmean",
)


def history_arrays(split_name):
    arrays = []
    names = []
    for key in ("video_id", "author_id"):
        h = historical_features(split_name, key=key)
        for name in sorted(h):
            if name.endswith(HISTORY_SUFFIXES):
                arrays.append(np.asarray(h[name], dtype=np.float32))
                names.append(name)
    return arrays, names


def build_matrix(split, split_name):
    columns = []

    for field in CAT_FIELDS:
        columns.append(np.asarray(split.X[field], dtype=np.float32))

    for field in NUM_FIELDS:
        x = np.asarray(split.num[field], dtype=np.float32)
        x = np.log1p(np.maximum(x, 0.0)).astype(np.float32)
        columns.append(x)

    hist, hist_names = history_arrays(split_name)
    columns.extend(hist)

    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    return matrix, hist_names


def temporal_weights(dates, half_life=6.0):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.sort(np.unique(dates))
    date_to_age = {
        int(d): float(len(unique_dates) - 1 - i)
        for i, d in enumerate(unique_dates)
    }
    age = np.fromiter(
        (date_to_age[int(d)] for d in dates),
        dtype=np.float32,
        count=len(dates),
    )
    return np.exp2(-age / float(half_life)).astype(np.float32)


COMMON_PARAMS = {
    "boosting_type": "gbdt",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "min_data_in_leaf": 600,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.86,
    "bagging_freq": 1,
    "lambda_l1": 0.15,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "min_gain_to_split": 1e-4,
    "num_threads": max(1, min(16, os.cpu_count() or 8)),
    "verbose": -1,
    "seed": 20260831,
    "feature_fraction_seed": 20260831,
    "bagging_seed": 20260832,
}


def train_binary(X, target, weights, rounds=220):
    params = dict(COMMON_PARAMS)
    params.update({
        "objective": "binary",
        "metric": "binary_logloss",
    })
    dataset = lgb.Dataset(
        X,
        label=np.asarray(target, dtype=np.float32),
        weight=np.asarray(weights, dtype=np.float32),
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )
    model = lgb.train(params, dataset, num_boost_round=rounds)
    del dataset
    return model


def train_regression(X, target, weights, rounds=260):
    params = dict(COMMON_PARAMS)
    params.update({
        "objective": "regression_l1",
        "metric": "l1",
        "learning_rate": 0.045,
        "num_leaves": 79,
    })
    dataset = lgb.Dataset(
        X,
        label=np.asarray(target, dtype=np.float32),
        weight=np.asarray(weights, dtype=np.float32),
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )
    model = lgb.train(params, dataset, num_boost_round=rounds)
    del dataset
    return model


train = load("train")
valid = load("valid")

X_train, history_names = build_matrix(train, "train")
X_valid, _ = build_matrix(valid, "valid")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_uid = np.asarray(valid.user_id, dtype=np.int64)
sample_weight = temporal_weights(train.date, half_life=6.0)

duration = np.asarray(train.num["duration_ms"], dtype=np.float32)
duration = np.maximum(np.nan_to_num(duration, nan=1.0), 1.0)

play = np.asarray(train.aux["play_time_ms"], dtype=np.float32)
play = np.maximum(np.nan_to_num(play, nan=0.0), 0.0)

click = np.asarray(train.aux["is_click"], dtype=np.float32)
click = (np.nan_to_num(click, nan=0.0) > 0).astype(np.float32)

completion = np.clip(play / duration, 0.0, 3.0).astype(np.float32)
completion_reg_target = np.log1p(completion).astype(np.float32)

ordinal_half_target = (completion >= 0.50).astype(np.float32)
ordinal_full_target = (completion >= 0.90).astype(np.float32)

print(
    "FINDINGS train_long_rate=%.6f click_rate=%.6f "
    "completion_half_rate=%.6f completion_full_rate=%.6f "
    "corr_long_completion=%.6f"
    % (
        float(np.mean(y_train)),
        float(np.mean(click)),
        float(np.mean(ordinal_half_target)),
        float(np.mean(ordinal_full_target)),
        float(np.corrcoef(y_train, completion)[0, 1]),
    ),
    flush=True,
)

# Family 1: distributional regression. It predicts continuous normalized
# watch completion rather than the organizer's binary relevance label.
watch_reg_model = train_regression(
    X_train,
    completion_reg_target,
    sample_weight,
    rounds=260,
)

# Family 2: ordinal cumulative survival. Separate models estimate survival
# beyond moderate and near-complete watch thresholds.
ordinal_half_model = train_binary(
    X_train,
    ordinal_half_target,
    sample_weight,
    rounds=210,
)
ordinal_full_model = train_binary(
    X_train,
    ordinal_full_target,
    sample_weight,
    rounds=210,
)

# Family 3: click-hurdle decomposition. Exposure first has to generate a click;
# conditional relevance is estimated only in the clicked population.
click_model = train_binary(
    X_train,
    click,
    sample_weight,
    rounds=200,
)

clicked_mask = click > 0
conditional_weights = sample_weight[clicked_mask]
conditional_long_model = train_binary(
    X_train[clicked_mask],
    y_train[clicked_mask],
    conditional_weights,
    rounds=220,
)

del duration
del play
del completion
del completion_reg_target
del ordinal_half_target
del ordinal_full_target
del conditional_weights
gc.collect()

valid_watch = watch_reg_model.predict(X_valid).astype(np.float32)
valid_half = ordinal_half_model.predict(X_valid).astype(np.float32)
valid_full = ordinal_full_model.predict(X_valid).astype(np.float32)
valid_click = click_model.predict(X_valid).astype(np.float32)
valid_conditional = conditional_long_model.predict(X_valid).astype(np.float32)

rank_watch = within_user_rank(valid_uid, valid_watch)
rank_half = within_user_rank(valid_uid, valid_half)
rank_full = within_user_rank(valid_uid, valid_full)
rank_click = within_user_rank(valid_uid, valid_click)
rank_conditional = within_user_rank(valid_uid, valid_conditional)

valid_ordinal = (
    0.38 * rank_half + 0.62 * rank_full
).astype(np.float32)

# A small non-click branch prevents the hurdle score from collapsing all
# non-click-like impressions into nearly identical scores.
valid_hurdle_probability = (
    valid_click * valid_conditional
    + 0.04 * (1.0 - valid_click) * valid_conditional
).astype(np.float32)
valid_hurdle = within_user_rank(valid_uid, valid_hurdle_probability)

valid_survival_consensus = (
    0.24 * rank_watch
    + 0.43 * valid_ordinal
    + 0.33 * valid_hurdle
).astype(np.float32)

inc_valid = np.load(INC_VALID_PATH, mmap_mode="r")
valid_incumbent = within_user_rank(valid_uid, inc_valid)

own_valid_scores = {
    "watch_distributional_regression": rank_watch,
    "ordinal_completion_survival": valid_ordinal,
    "click_hurdle_esmm": valid_hurdle,
    "auxiliary_survival_consensus": valid_survival_consensus,
}

candidate_arrays = {}
candidate_metrics = {}

for name, scores in own_valid_scores.items():
    candidate_arrays[name] = scores
    m = evaluate(valid_uid, y_valid, scores)
    candidate_metrics[name] = float(m["primary"])
    print(
        "FINDINGS family=%s primary=%.6f gauc=%.6f ndcg5=%.6f"
        % (
            name,
            float(m["primary"]),
            float(m["gauc"]),
            float(m["ndcg@5"]),
        ),
        flush=True,
    )

inc_metrics = evaluate(valid_uid, y_valid, valid_incumbent)
candidate_arrays["trusted_incumbent_control"] = valid_incumbent
candidate_metrics["trusted_incumbent_control"] = float(inc_metrics["primary"])

# Every auxiliary family is also tested as a rank-level blend. Rank fusion
# avoids incompatible probability scales across the three target mechanisms.
blend_alphas = (0.70, 0.82, 0.90, 0.95)
for family_name, own_scores in own_valid_scores.items():
    for alpha in blend_alphas:
        name = "%s_incumbent_%.2f" % (family_name, alpha)
        blended = (
            alpha * valid_incumbent + (1.0 - alpha) * own_scores
        ).astype(np.float32)
        candidate_arrays[name] = blended
        m = evaluate(valid_uid, y_valid, blended)
        candidate_metrics[name] = float(m["primary"])

# A nonlinear agreement score promotes impressions simultaneously favored by
# the incumbent and by the auxiliary survival consensus.
agreement = np.sqrt(
    np.maximum(valid_incumbent, 0.0)
    * np.maximum(valid_survival_consensus, 0.0)
).astype(np.float32)
agreement = within_user_rank(valid_uid, agreement)

for alpha in (0.78, 0.88, 0.94):
    name = "geometric_agreement_incumbent_%.2f" % alpha
    scores = (
        alpha * valid_incumbent + (1.0 - alpha) * agreement
    ).astype(np.float32)
    candidate_arrays[name] = scores
    m = evaluate(valid_uid, y_valid, scores)
    candidate_metrics[name] = float(m["primary"])

best_name = max(candidate_metrics, key=candidate_metrics.get)
best_valid = candidate_arrays[best_name]
final_metrics = evaluate(valid_uid, y_valid, best_valid)

if best_name.startswith("watch_distributional_regression"):
    selected_own_name = "watch_distributional_regression"
elif best_name.startswith("ordinal_completion_survival"):
    selected_own_name = "ordinal_completion_survival"
elif best_name.startswith("click_hurdle_esmm"):
    selected_own_name = "click_hurdle_esmm"
else:
    selected_own_name = "auxiliary_survival_consensus"

selected_own_valid = own_valid_scores[selected_own_name]

print(
    "FINDINGS winner=%s incumbent_primary=%.6f winner_primary=%.6f "
    "delta=%+.6f"
    % (
        best_name,
        float(inc_metrics["primary"]),
        float(final_metrics["primary"]),
        float(final_metrics["primary"] - inc_metrics["primary"]),
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_metrics, sort_keys=True),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(OUT, "scores_valid_raw.npy"),
        np.asarray(selected_own_valid, dtype=np.float64),
    )

del X_train
del X_valid
del train
del valid
del inc_valid
del candidate_arrays
gc.collect()

test = load("test")
test_uid = np.asarray(test.user_id, dtype=np.int64)
X_test, _ = build_matrix(test, "test")

test_watch = watch_reg_model.predict(X_test).astype(np.float32)
test_half = ordinal_half_model.predict(X_test).astype(np.float32)
test_full = ordinal_full_model.predict(X_test).astype(np.float32)
test_click = click_model.predict(X_test).astype(np.float32)
test_conditional = conditional_long_model.predict(X_test).astype(np.float32)

test_rank_watch = within_user_rank(test_uid, test_watch)
test_rank_half = within_user_rank(test_uid, test_half)
test_rank_full = within_user_rank(test_uid, test_full)

test_ordinal = (
    0.38 * test_rank_half + 0.62 * test_rank_full
).astype(np.float32)

test_hurdle_probability = (
    test_click * test_conditional
    + 0.04 * (1.0 - test_click) * test_conditional
).astype(np.float32)
test_hurdle = within_user_rank(test_uid, test_hurdle_probability)

test_consensus = (
    0.24 * test_rank_watch
    + 0.43 * test_ordinal
    + 0.33 * test_hurdle
).astype(np.float32)

inc_test = np.load(INC_TEST_PATH, mmap_mode="r")
test_incumbent = within_user_rank(test_uid, inc_test)

if best_name == "trusted_incumbent_control":
    test_scores = test_incumbent
elif best_name == "watch_distributional_regression":
    test_scores = test_rank_watch
elif best_name == "ordinal_completion_survival":
    test_scores = test_ordinal
elif best_name == "click_hurdle_esmm":
    test_scores = test_hurdle
elif best_name == "auxiliary_survival_consensus":
    test_scores = test_consensus
elif best_name.startswith("geometric_agreement_incumbent_"):
    alpha = float(best_name.rsplit("_", 1)[1])
    test_agreement = np.sqrt(
        np.maximum(test_incumbent, 0.0)
        * np.maximum(test_consensus, 0.0)
    ).astype(np.float32)
    test_agreement = within_user_rank(test_uid, test_agreement)
    test_scores = (
        alpha * test_incumbent + (1.0 - alpha) * test_agreement
    ).astype(np.float32)
else:
    alpha = float(best_name.rsplit("_", 1)[1])
    if best_name.startswith("watch_distributional_regression"):
        own_test = test_rank_watch
    elif best_name.startswith("ordinal_completion_survival"):
        own_test = test_ordinal
    elif best_name.startswith("click_hurdle_esmm"):
        own_test = test_hurdle
    else:
        own_test = test_consensus
    test_scores = (
        alpha * test_incumbent + (1.0 - alpha) * own_test
    ).astype(np.float32)

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print(
    "METRICS "
    + json.dumps({
        "primary": float(final_metrics["primary"]),
        "gauc": float(final_metrics["gauc"]),
        "ndcg@5": float(final_metrics["ndcg@5"]),
        "gpu_seconds": elapsed,
    })
)