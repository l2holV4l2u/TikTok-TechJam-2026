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
SEED = 20260831


def within_user_rank(scores, users):
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users, dtype=np.int64)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e20, neginf=-1e20)
    n = len(scores)
    if n == 0:
        return scores.copy()

    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, scores, users))
    su = users[order]
    ss = scores[order]
    pos = np.arange(n, dtype=np.int64)

    user_start = np.empty(n, dtype=bool)
    user_start[0] = True
    user_start[1:] = su[1:] != su[:-1]
    user_starts = np.maximum.accumulate(np.where(user_start, pos, 0))

    user_end = np.empty(n, dtype=bool)
    user_end[-1] = True
    user_end[:-1] = su[:-1] != su[1:]
    user_ends = np.minimum.accumulate(
        np.where(user_end, pos, n - 1)[::-1]
    )[::-1]

    tie_start = np.empty(n, dtype=bool)
    tie_start[0] = True
    tie_start[1:] = (su[1:] != su[:-1]) | (ss[1:] != ss[:-1])
    tie_starts = np.maximum.accumulate(np.where(tie_start, pos, 0))

    tie_end = np.empty(n, dtype=bool)
    tie_end[-1] = True
    tie_end[:-1] = (su[:-1] != su[1:]) | (ss[:-1] != ss[1:])
    tie_ends = np.minimum.accumulate(
        np.where(tie_end, pos, n - 1)[::-1]
    )[::-1]

    local_rank = 0.5 * (tie_starts + tie_ends) - user_starts
    denom = np.maximum(user_ends - user_starts, 1)
    ranked = local_rank / denom

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def sequence_context(split):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    authors = np.asarray(split.X["author_id"], dtype=np.int64)
    dates = np.asarray(split.date, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    n = len(users)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, times, users))
    ou = users[order]
    od = dates[order]
    ot = times[order]
    p = np.arange(n, dtype=np.int64)

    user_boundary = np.empty(n, dtype=bool)
    user_boundary[0] = True
    user_boundary[1:] = ou[1:] != ou[:-1]
    user_start = np.maximum.accumulate(np.where(user_boundary, p, 0))
    user_position_sorted = p - user_start

    day_boundary = np.empty(n, dtype=bool)
    day_boundary[0] = True
    day_boundary[1:] = (
        (ou[1:] != ou[:-1]) | (od[1:] != od[:-1])
    )
    day_start = np.maximum.accumulate(np.where(day_boundary, p, 0))
    day_position_sorted = p - day_start

    gap_seconds_sorted = np.zeros(n, dtype=np.float64)
    same_user = ~user_boundary
    gap_seconds_sorted[same_user] = np.maximum(
        (ot[same_user] - ot[np.nonzero(same_user)[0] - 1]) / 1000.0,
        0.0,
    )

    session_boundary = user_boundary | (gap_seconds_sorted > 1800.0)
    session_start = np.maximum.accumulate(
        np.where(session_boundary, p, 0)
    )
    session_position_sorted = p - session_start

    user_position = np.empty(n, dtype=np.int32)
    day_position = np.empty(n, dtype=np.int32)
    session_position = np.empty(n, dtype=np.int32)
    gap_seconds = np.empty(n, dtype=np.float32)

    user_position[order] = np.minimum(user_position_sorted, 100000)
    day_position[order] = np.minimum(day_position_sorted, 100000)
    session_position[order] = np.minimum(session_position_sorted, 100000)
    gap_seconds[order] = np.minimum(gap_seconds_sorted, 86400.0)

    def prior_occurrences(entity):
        entity = np.asarray(entity, dtype=np.int64)
        eorder = np.lexsort((rows, times, entity, users))
        eu = users[eorder]
        ee = entity[eorder]
        ep = np.arange(n, dtype=np.int64)
        boundary = np.empty(n, dtype=bool)
        boundary[0] = True
        boundary[1:] = (eu[1:] != eu[:-1]) | (ee[1:] != ee[:-1])
        starts = np.maximum.accumulate(np.where(boundary, ep, 0))
        occurrence_sorted = ep - starts
        occurrence = np.empty(n, dtype=np.int32)
        occurrence[eorder] = np.minimum(occurrence_sorted, 100000)
        return occurrence

    video_repeat = prior_occurrences(videos)
    author_repeat = prior_occurrences(authors)

    session_bin = np.minimum(
        np.floor(np.log2(session_position.astype(np.float64) + 1.0)),
        7,
    ).astype(np.int32)
    day_bin = np.minimum(
        np.floor(np.log2(day_position.astype(np.float64) + 1.0)),
        7,
    ).astype(np.int32)
    user_pos_bin = np.minimum(
        np.floor(np.log2(user_position.astype(np.float64) + 1.0)),
        9,
    ).astype(np.int32)
    gap_bin = np.searchsorted(
        np.array([1, 5, 15, 30, 60, 180, 600, 1800, 7200],
                 dtype=np.float64),
        gap_seconds,
        side="right",
    ).astype(np.int32)
    video_repeat_bin = np.minimum(video_repeat, 5).astype(np.int32)
    author_repeat_bin = np.minimum(author_repeat, 5).astype(np.int32)

    return {
        "log_user_position": np.log1p(user_position).astype(np.float32),
        "log_day_position": np.log1p(day_position).astype(np.float32),
        "log_session_position": np.log1p(session_position).astype(np.float32),
        "log_gap_seconds": np.log1p(gap_seconds).astype(np.float32),
        "log_video_repeat": np.log1p(video_repeat).astype(np.float32),
        "log_author_repeat": np.log1p(author_repeat).astype(np.float32),
        "session_bin": session_bin,
        "day_bin": day_bin,
        "user_pos_bin": user_pos_bin,
        "gap_bin": gap_bin,
        "video_repeat_bin": video_repeat_bin,
        "author_repeat_bin": author_repeat_bin,
    }


CAT_FIELDS = [
    "video_id",
    "author_id",
    "user_id",
    "tab",
    "tag",
    "onehot_feat3",
    "upload_type",
    "onehot_feat8",
    "duration_bucket",
    "onehot_feat1",
    "music_type",
    "onehot_feat7",
    "user_active_degree",
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

HISTORY_KEYS = [
    "video_id_train_count_log1p",
    "video_id_long_view_rate",
    "video_id_is_click_rate",
    "video_id_play_time_ms_logmean",
    "video_id_is_like_rate",
    "video_id_is_profile_enter_rate",
    "author_id_train_count_log1p",
    "author_id_long_view_rate",
    "author_id_is_click_rate",
    "author_id_play_time_ms_logmean",
    "author_id_is_like_rate",
]


def get_histories(split_name):
    vh = historical_features(split_name, key="video_id")
    ah = historical_features(split_name, key="author_id")
    result = {}
    result.update(vh)
    result.update(ah)
    return result


def make_matrix(split, context, histories):
    columns = []
    names = []
    categorical_indices = []

    for field in CAT_FIELDS:
        values = np.asarray(split.X[field], dtype=np.float32)
        categorical_indices.append(len(columns))
        columns.append(values)
        names.append(field)

    for field in NUM_FIELDS:
        x = np.asarray(split.num[field], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))
        names.append("log_" + field)

    continuous_context = [
        "log_user_position",
        "log_day_position",
        "log_session_position",
        "log_gap_seconds",
        "log_video_repeat",
        "log_author_repeat",
    ]
    categorical_context = [
        "session_bin",
        "day_bin",
        "user_pos_bin",
        "gap_bin",
        "video_repeat_bin",
        "author_repeat_bin",
    ]

    for name in continuous_context:
        columns.append(np.asarray(context[name], dtype=np.float32))
        names.append(name)

    for name in categorical_context:
        categorical_indices.append(len(columns))
        columns.append(np.asarray(context[name], dtype=np.float32))
        names.append(name)

    for key in HISTORY_KEYS:
        if key in histories:
            x = np.asarray(histories[key], dtype=np.float32)
            finite = np.isfinite(x)
            fill = float(np.median(x[finite])) if np.any(finite) else 0.0
            columns.append(
                np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)
                .astype(np.float32)
            )
            names.append(key)

    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    return matrix, names, categorical_indices


def recency_weights(dates, half_life=5.0):
    dates = np.asarray(dates, dtype=np.int32)
    age = np.max(dates) - dates
    return np.power(0.5, age.astype(np.float64) / half_life).astype(
        np.float32
    )


def fit_rate(values, y, weights, cardinality, strength, prior=None):
    values = np.asarray(values, dtype=np.int64)
    total = np.bincount(
        values, weights=weights, minlength=cardinality
    ).astype(np.float64)
    positive = np.bincount(
        values, weights=weights * y, minlength=cardinality
    ).astype(np.float64)

    if prior is None:
        global_rate = float(
            np.sum(weights * y) / max(float(np.sum(weights)), 1e-12)
        )
        prior_array = np.full(cardinality, global_rate, dtype=np.float64)
    else:
        prior_array = np.asarray(prior, dtype=np.float64)

    rate = (
        positive + strength * prior_array
    ) / np.maximum(total + strength, 1e-12)
    return rate.astype(np.float32), total.astype(np.float32)


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def fit_additive_hazard(train, context, weights):
    y = np.asarray(train.y, dtype=np.float32)
    global_rate = float(np.sum(weights * y) / np.sum(weights))

    specifications = [
        ("video_id", np.asarray(train.video_id), 8000, 30.0, 1.00),
        ("author_id", np.asarray(train.X["author_id"]), 7000, 35.0, 0.75),
        ("tab", np.asarray(train.X["tab"]), 20, 80.0, 0.45),
        ("tag", np.asarray(train.X["tag"]), 64, 80.0, 0.38),
        ("duration_bucket", np.asarray(train.X["duration_bucket"]),
         12, 100.0, 0.28),
        ("session_bin", context["session_bin"], 8, 150.0, 0.42),
        ("day_bin", context["day_bin"], 8, 150.0, 0.30),
        ("gap_bin", context["gap_bin"], 10, 150.0, 0.28),
        ("video_repeat_bin", context["video_repeat_bin"],
         6, 100.0, 0.38),
        ("author_repeat_bin", context["author_repeat_bin"],
         6, 100.0, 0.28),
    ]

    tables = {}
    for name, values, cardinality, strength, coefficient in specifications:
        rate, count = fit_rate(
            values, y, weights, cardinality, strength
        )
        tables[name] = (rate, count, coefficient)

    user = np.asarray(train.user_id, dtype=np.int64)
    session_bin = np.asarray(context["session_bin"], dtype=np.int64)
    n_users = int(FEATURE_CARDINALITIES["user_id"])
    interaction = user * 8 + session_bin
    user_session_rate, user_session_count = fit_rate(
        interaction, y, weights, n_users * 8, 18.0
    )

    return {
        "global_rate": global_rate,
        "tables": tables,
        "user_session_rate": user_session_rate,
        "user_session_count": user_session_count,
    }


def additive_hazard_scores(split, context, model):
    global_logit = float(logit(model["global_rate"]))
    score = np.full(len(split.user_id), global_logit, dtype=np.float64)

    values_by_name = {
        "video_id": np.asarray(split.video_id, dtype=np.int64),
        "author_id": np.asarray(split.X["author_id"], dtype=np.int64),
        "tab": np.asarray(split.X["tab"], dtype=np.int64),
        "tag": np.asarray(split.X["tag"], dtype=np.int64),
        "duration_bucket": np.asarray(
            split.X["duration_bucket"], dtype=np.int64
        ),
        "session_bin": np.asarray(context["session_bin"], dtype=np.int64),
        "day_bin": np.asarray(context["day_bin"], dtype=np.int64),
        "gap_bin": np.asarray(context["gap_bin"], dtype=np.int64),
        "video_repeat_bin": np.asarray(
            context["video_repeat_bin"], dtype=np.int64
        ),
        "author_repeat_bin": np.asarray(
            context["author_repeat_bin"], dtype=np.int64
        ),
    }

    for name, (rate, count, coefficient) in model["tables"].items():
        values = np.minimum(values_by_name[name], len(rate) - 1)
        reliability = count[values] / (count[values] + 30.0)
        score += (
            coefficient
            * reliability
            * (logit(rate[values]) - global_logit)
        )

    users = np.asarray(split.user_id, dtype=np.int64)
    session_bin = np.asarray(context["session_bin"], dtype=np.int64)
    users = np.minimum(
        users, int(FEATURE_CARDINALITIES["user_id"]) - 1
    )
    index = users * 8 + np.minimum(session_bin, 7)
    rate = model["user_session_rate"][index]
    count = model["user_session_count"][index]
    reliability = count / (count + 20.0)
    score += 0.45 * reliability * (logit(rate) - global_logit)

    return score.astype(np.float32)


train = load("train")
valid = load("valid")
test = load("test")

train_context = sequence_context(train)
valid_context = sequence_context(valid)
test_context = sequence_context(test)

train_hist = get_histories("train")
valid_hist = get_histories("valid")
test_hist = get_histories("test")

X_train, feature_names, categorical_indices = make_matrix(
    train, train_context, train_hist
)
X_valid, _, _ = make_matrix(valid, valid_context, valid_hist)
X_test, _, _ = make_matrix(test, test_context, test_hist)

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)
weights = recency_weights(train.date, half_life=5.0)

dataset = lgb.Dataset(
    X_train,
    label=y_train,
    weight=weights,
    feature_name=feature_names,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)

gbdt_params = {
    "objective": "binary",
    "metric": "None",
    "boosting_type": "gbdt",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.82,
    "bagging_freq": 1,
    "lambda_l1": 0.3,
    "lambda_l2": 4.0,
    "max_bin": 127,
    "cat_smooth": 20.0,
    "cat_l2": 8.0,
    "seed": SEED,
    "num_threads": -1,
    "verbose": -1,
}

gbdt = lgb.train(
    gbdt_params,
    dataset,
    num_boost_round=210,
)
gbdt_valid = gbdt.predict(X_valid).astype(np.float64)
gbdt_test = gbdt.predict(X_test).astype(np.float64)

rf_params = {
    "objective": "binary",
    "metric": "None",
    "boosting_type": "rf",
    "learning_rate": 0.10,
    "num_leaves": 127,
    "max_depth": 12,
    "min_data_in_leaf": 250,
    "feature_fraction": 0.62,
    "bagging_fraction": 0.62,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 6.0,
    "max_bin": 127,
    "cat_smooth": 25.0,
    "cat_l2": 10.0,
    "seed": SEED + 17,
    "num_threads": -1,
    "verbose": -1,
}

rf = lgb.train(
    rf_params,
    dataset,
    num_boost_round=150,
)
rf_valid = rf.predict(X_valid).astype(np.float64)
rf_test = rf.predict(X_test).astype(np.float64)

hazard_model = fit_additive_hazard(
    train, train_context, weights
)
hazard_valid = additive_hazard_scores(
    valid, valid_context, hazard_model
).astype(np.float64)
hazard_test = additive_hazard_scores(
    test, test_context, hazard_model
).astype(np.float64)

del dataset, gbdt, rf, X_train, X_valid, X_test
gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)
if not (
    os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise FileNotFoundError("Trusted incumbent predictions unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

raw_valid = {
    "context_gbdt": gbdt_valid,
    "context_bagged_forest": rf_valid,
    "additive_session_hazard": hazard_valid,
}
raw_test = {
    "context_gbdt": gbdt_test,
    "context_bagged_forest": rf_test,
    "additive_session_hazard": hazard_test,
}

valid_rank = {
    name: within_user_rank(score, valid_users)
    for name, score in raw_valid.items()
}
test_rank = {
    name: within_user_rank(score, test_users)
    for name, score in raw_test.items()
}
inc_valid_rank = within_user_rank(inc_valid, valid_users)
inc_test_rank = within_user_rank(inc_test, test_users)

candidate_scores = {}
candidate_predictions = {}
candidate_test_predictions = {}
candidate_raw_name = {}

inc_metrics = evaluate(valid_users, y_valid, inc_valid_rank)
candidate_scores["trusted_incumbent"] = float(inc_metrics["primary"])
candidate_predictions["trusted_incumbent"] = inc_valid_rank
candidate_test_predictions["trusted_incumbent"] = inc_test_rank
candidate_raw_name["trusted_incumbent"] = "context_gbdt"

alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

for name in raw_valid:
    standalone_metrics = evaluate(
        valid_users, y_valid, valid_rank[name]
    )
    candidate_scores[name] = float(standalone_metrics["primary"])
    candidate_predictions[name] = valid_rank[name]
    candidate_test_predictions[name] = test_rank[name]
    candidate_raw_name[name] = name

    for alpha in alphas:
        blend_name = "{}_blend_{:.2f}".format(name, alpha)
        blend_valid = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * valid_rank[name]
        )
        blend_test = (
            (1.0 - alpha) * inc_test_rank
            + alpha * test_rank[name]
        )
        blend_metrics = evaluate(valid_users, y_valid, blend_valid)
        candidate_scores[blend_name] = float(
            blend_metrics["primary"]
        )
        candidate_predictions[blend_name] = blend_valid
        candidate_test_predictions[blend_name] = blend_test
        candidate_raw_name[blend_name] = name

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = np.asarray(
    candidate_predictions[winner], dtype=np.float64
)
test_scores = np.asarray(
    candidate_test_predictions[winner], dtype=np.float64
)
own_raw_name = candidate_raw_name[winner]
own_raw_valid = np.asarray(
    raw_valid[own_raw_name], dtype=np.float64
)

metrics = evaluate(valid_users, y_valid, valid_scores)

best_blends = {}
for name in raw_valid:
    relevant = {
        k: v for k, v in candidate_scores.items()
        if k == name or k.startswith(name + "_blend_")
    }
    best_name = max(relevant, key=relevant.get)
    best_blends[best_name] = relevant[best_name]
best_blends["trusted_incumbent"] = candidate_scores["trusted_incumbent"]

print("FINDINGS winner={} raw_family={}".format(winner, own_raw_name))
print("CANDIDATES " + json.dumps(
    best_blends, sort_keys=True
))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        valid_scores.astype(np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        test_scores.astype(np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        own_raw_valid.astype(np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)