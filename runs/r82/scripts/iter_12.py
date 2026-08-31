import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 20260830

CAT_FIELDS = [
    "tab", "tag", "duration_bucket", "upload_type", "music_type",
    "video_type", "hour", "user_active_degree", "is_live_streamer",
    "is_video_author", "fans_user_num_range", "follow_user_num_range",
    "friend_user_num_range", "register_days_bucket",
    "onehot_feat3", "onehot_feat7", "onehot_feat8",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]


class CombinedSplit:
    pass


def combine_splits(a, b):
    c = CombinedSplit()
    c.user_id = np.concatenate([
        np.asarray(a.user_id), np.asarray(b.user_id)
    ])
    c.video_id = np.concatenate([
        np.asarray(a.video_id), np.asarray(b.video_id)
    ])
    c.date = np.concatenate([
        np.asarray(a.date), np.asarray(b.date)
    ])
    c.time_ms = np.concatenate([
        np.asarray(a.time_ms), np.asarray(b.time_ms)
    ])
    c.y = np.concatenate([
        np.asarray(a.y), np.asarray(b.y)
    ])
    c.X = {
        name: np.concatenate([
            np.asarray(a.X[name]), np.asarray(b.X[name])
        ])
        for name in set(CAT_FIELDS + [
            "author_id", "tag", "duration_bucket"
        ])
    }
    c.num = {
        name: np.concatenate([
            np.asarray(a.num[name]), np.asarray(b.num[name])
        ])
        for name in NUM_FIELDS
    }
    return c


def compact_day(date):
    date = np.asarray(date, dtype=np.int64)
    month = (date // 100) % 100
    day = date % 100
    return day + np.where(month >= 5, 30, 0)


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def within_group_rank(group, values):
    group = np.asarray(group, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, values, group))
    sg = group[order]

    boundary = np.empty(n, dtype=bool)
    boundary[0] = True
    boundary[1:] = sg[1:] != sg[:-1]
    starts = np.maximum.accumulate(
        np.where(boundary, np.arange(n, dtype=np.int64), 0)
    )
    position = np.arange(n, dtype=np.int64) - starts
    ends = np.r_[np.flatnonzero(boundary)[1:], n]
    counts_for_groups = ends - np.flatnonzero(boundary)
    group_number = np.cumsum(boundary) - 1
    denominator = np.maximum(counts_for_groups[group_number] - 1, 1)

    result = np.empty(n, dtype=np.float32)
    result[order] = (position / denominator).astype(np.float32)
    return result


def chronological_features(user, date, time_ms):
    user = np.asarray(user, dtype=np.int64)
    day = compact_day(date)
    time_ms = np.asarray(time_ms, dtype=np.int64)
    n = len(user)
    row = np.arange(n, dtype=np.int64)

    result = []
    for group in (
        user,
        user * np.int64(64) + day,
    ):
        order = np.lexsort((row, time_ms, group))
        sg = group[order]
        st = time_ms[order]

        boundary = np.empty(n, dtype=bool)
        boundary[0] = True
        boundary[1:] = sg[1:] != sg[:-1]
        start_idx = np.flatnonzero(boundary)
        ends = np.r_[start_idx[1:], n]
        counts_by_group = ends - start_idx
        group_number = np.cumsum(boundary) - 1
        starts = start_idx[group_number]
        position = np.arange(n, dtype=np.int64) - starts
        counts = counts_by_group[group_number]
        denominator = np.maximum(counts - 1, 1)

        pos = position / denominator
        rev = (counts - 1 - position) / denominator

        gap = np.zeros(n, dtype=np.float64)
        valid_prev = ~boundary
        gap[valid_prev] = np.maximum(
            st[valid_prev] - st[np.flatnonzero(valid_prev) - 1], 0
        ) / 1000.0

        out_pos = np.empty(n, dtype=np.float32)
        out_rev = np.empty(n, dtype=np.float32)
        out_count = np.empty(n, dtype=np.float32)
        out_gap = np.empty(n, dtype=np.float32)
        out_pos[order] = pos.astype(np.float32)
        out_rev[order] = rev.astype(np.float32)
        out_count[order] = np.log1p(counts).astype(np.float32)
        out_gap[order] = np.log1p(np.minimum(gap, 86400.0)).astype(np.float32)
        result.extend([out_pos, out_rev, out_count, out_gap])

    return result


def entity_statistics(source_ids, y, query_ids, cardinality,
                      smooth, training):
    source_ids = np.asarray(source_ids, dtype=np.int64)
    query_ids = np.asarray(query_ids, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    count = np.bincount(
        source_ids, minlength=cardinality
    ).astype(np.float64)
    positive = np.bincount(
        source_ids, weights=y, minlength=cardinality
    ).astype(np.float64)
    prior = float(y.mean())

    if training:
        c = count[source_ids] - 1.0
        p = positive[source_ids] - y
    else:
        c = count[query_ids]
        p = positive[query_ids]

    rate = (p + smooth * prior) / (c + smooth)
    count_feature = np.log1p(np.maximum(c, 0.0))
    return rate.astype(np.float32), count_feature.astype(np.float32)


def pair_statistics(source_user, source_value, y,
                    query_user, query_value, value_cardinality,
                    smooth, training):
    source_user = np.asarray(source_user, dtype=np.int64)
    source_value = np.asarray(source_value, dtype=np.int64)
    query_user = np.asarray(query_user, dtype=np.int64)
    query_value = np.asarray(query_value, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)

    source_key = (
        source_user * np.int64(value_cardinality) + source_value
    )
    query_key = (
        query_user * np.int64(value_cardinality) + query_value
    )
    size = int(max(
        source_key.max(initial=0),
        query_key.max(initial=0),
    )) + 1

    count = np.bincount(source_key, minlength=size).astype(np.float64)
    positive = np.bincount(
        source_key, weights=y, minlength=size
    ).astype(np.float64)
    prior = float(y.mean())

    if training:
        c = count[source_key] - 1.0
        p = positive[source_key] - y
    else:
        c = count[query_key]
        p = positive[query_key]

    rate = (p + smooth * prior) / (c + smooth)
    count_feature = np.log1p(np.maximum(c, 0.0))
    return rate.astype(np.float32), count_feature.astype(np.float32)


def build_features(source, query, training=False):
    y = np.asarray(source.y, dtype=np.float64)
    if training and len(source.user_id) != len(query.user_id):
        raise ValueError("training features require source == query")

    features = []
    names = []

    entity_specs = [
        ("video", source.video_id, query.video_id,
         int(FEATURE_CARDINALITIES["video_id"]), 18.0),
        ("author", source.X["author_id"], query.X["author_id"],
         int(FEATURE_CARDINALITIES["author_id"]), 18.0),
        ("tag", source.X["tag"], query.X["tag"],
         int(FEATURE_CARDINALITIES["tag"]), 25.0),
        ("duration", source.X["duration_bucket"],
         query.X["duration_bucket"],
         int(FEATURE_CARDINALITIES["duration_bucket"]), 25.0),
    ]

    entity_rates = {}
    for name, sid, qid, cardinality, smooth in entity_specs:
        rate, count = entity_statistics(
            sid, y, qid, cardinality, smooth, training
        )
        entity_rates[name] = rate
        features.extend([
            logit(rate).astype(np.float32),
            count,
        ])
        names.extend([name + "_logit", name + "_log_count"])

    pair_specs = [
        ("user_tag", source.X["tag"], query.X["tag"],
         int(FEATURE_CARDINALITIES["tag"]), 6.0),
        ("user_duration", source.X["duration_bucket"],
         query.X["duration_bucket"],
         int(FEATURE_CARDINALITIES["duration_bucket"]), 7.0),
    ]
    pair_rates = {}
    for name, sval, qval, cardinality, smooth in pair_specs:
        rate, count = pair_statistics(
            source.user_id, sval, y,
            query.user_id, qval, cardinality,
            smooth, training,
        )
        pair_rates[name] = rate
        features.extend([
            logit(rate).astype(np.float32),
            count,
        ])
        names.extend([name + "_logit", name + "_log_count"])

    # Personalized residuals remove broad popularity and emphasize hard
    # negatives whose category is globally attractive but poor for this user.
    tag_residual = (
        logit(pair_rates["user_tag"]) - logit(entity_rates["tag"])
    ).astype(np.float32)
    duration_residual = (
        logit(pair_rates["user_duration"])
        - logit(entity_rates["duration"])
    ).astype(np.float32)
    features.extend([tag_residual, duration_residual])
    names.extend(["user_tag_residual", "user_duration_residual"])

    for name in NUM_FIELDS:
        x = np.asarray(query.num[name], dtype=np.float64)
        finite = np.isfinite(x)
        x = np.where(finite, np.maximum(x, 0.0), 0.0)
        features.append(np.log1p(x).astype(np.float32))
        names.append("log_" + name)

    for name in CAT_FIELDS:
        features.append(np.asarray(query.X[name], dtype=np.float32))
        names.append("cat_" + name)

    # Relative candidate-set ranks are invariant to drifting probability
    # calibration and expose which impressions are hardest within each user.
    quser = np.asarray(query.user_id, dtype=np.int64)
    day_group = quser * np.int64(64) + compact_day(query.date)

    rank_sources = [
        ("video_rate", entity_rates["video"]),
        ("author_rate", entity_rates["author"]),
        ("tag_rate", entity_rates["tag"]),
        ("duration_rate", entity_rates["duration"]),
        ("user_tag_rate", pair_rates["user_tag"]),
        ("user_duration_rate", pair_rates["user_duration"]),
        ("raw_duration", np.asarray(query.num["duration_ms"])),
    ]
    for name, value in rank_sources:
        features.append(within_group_rank(quser, value))
        names.append(name + "_user_rank")
        features.append(within_group_rank(day_group, value))
        names.append(name + "_day_rank")

    chrono = chronological_features(
        query.user_id, query.date, query.time_ms
    )
    chrono_names = [
        "user_forward_position", "user_reverse_position",
        "user_log_count", "user_previous_gap",
        "day_forward_position", "day_reverse_position",
        "day_log_count", "day_previous_gap",
    ]
    features.extend(chrono)
    names.extend(chrono_names)

    X = np.column_stack(features).astype(np.float32, copy=False)

    # A deliberately non-parametric family using only leakage-free rates.
    nonparam = (
        0.30 * within_group_rank(quser, entity_rates["video"])
        + 0.23 * within_group_rank(quser, entity_rates["author"])
        + 0.12 * within_group_rank(quser, entity_rates["tag"])
        + 0.20 * within_group_rank(quser, pair_rates["user_tag"])
        + 0.15 * within_group_rank(quser, pair_rates["user_duration"])
    ).astype(np.float32)

    return X, nonparam, names


def rank_score(user_ids, scores):
    return within_group_rank(user_ids, scores)


def train_model(X, y, family, rounds):
    dtrain = lgb.Dataset(
        X,
        label=np.asarray(y, dtype=np.float32),
        free_raw_data=False,
    )

    common = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.055,
        "num_leaves": 63,
        "min_data_in_leaf": 350,
        "feature_fraction": 0.78,
        "lambda_l1": 0.15,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "verbose": -1,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "num_threads": max(1, min(12, os.cpu_count() or 4)),
    }

    if family == "gbdt":
        params = dict(common)
        params.update({
            "boosting_type": "gbdt",
            "bagging_fraction": 0.88,
            "bagging_freq": 1,
        })
    elif family == "random_forest":
        params = dict(common)
        params.update({
            "boosting_type": "rf",
            "learning_rate": 1.0,
            "bagging_fraction": 0.66,
            "bagging_freq": 1,
            "feature_fraction": 0.62,
            "num_leaves": 95,
            "min_data_in_leaf": 500,
        })
    else:
        raise ValueError(family)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=rounds,
    )
    return model


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
inc_valid_rank = rank_score(valid.user_id, inc_valid)

X_train, train_nonparam, feature_names = build_features(
    train, train, training=True
)
X_valid, valid_nonparam, _ = build_features(
    train, valid, training=False
)

models = {}
raw_predictions = {
    "nonparametric_set_rank": valid_nonparam,
}

models["gbdt"] = train_model(X_train, y_train, "gbdt", 210)
raw_predictions["boosted_candidate_tree"] = models["gbdt"].predict(
    X_valid
).astype(np.float32)

models["random_forest"] = train_model(
    X_train, y_train, "random_forest", 120
)
raw_predictions["bagged_candidate_forest"] = models[
    "random_forest"
].predict(X_valid).astype(np.float32)

candidate_log = {}
inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)
candidate_log["trusted_incumbent"] = float(inc_metrics["primary"])

best = {
    "name": "trusted_incumbent",
    "family": "incumbent",
    "alpha": 0.0,
    "primary": float(inc_metrics["primary"]),
    "scores": inc_valid.copy(),
}

alpha_grid = np.linspace(0.0, 0.80, 17)

for name, raw in raw_predictions.items():
    new_rank = rank_score(valid.user_id, raw)
    standalone = evaluate(valid.user_id, y_valid, new_rank)
    candidate_log[name + "_standalone"] = float(standalone["primary"])

    local_best = -np.inf
    local_alpha = 0.0
    local_scores = None
    for alpha in alpha_grid:
        blended = (
            (1.0 - float(alpha)) * inc_valid_rank
            + float(alpha) * new_rank
        ).astype(np.float32)
        result = evaluate(valid.user_id, y_valid, blended)
        if float(result["primary"]) > local_best:
            local_best = float(result["primary"])
            local_alpha = float(alpha)
            local_scores = blended.copy()

    candidate_log[name + "_blend"] = local_best
    if local_best > best["primary"]:
        if name == "boosted_candidate_tree":
            family = "gbdt"
        elif name == "bagged_candidate_forest":
            family = "random_forest"
        else:
            family = "nonparametric"
        best = {
            "name": name,
            "family": family,
            "alpha": local_alpha,
            "primary": local_best,
            "scores": local_scores,
        }

valid_scores = np.asarray(best["scores"], dtype=np.float32)
metrics = evaluate(valid.user_id, y_valid, valid_scores)

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print("FINDINGS " + json.dumps({
    "selected_family": best["family"],
    "selected_name": best["name"],
    "selected_alpha": best["alpha"],
    "n_features": int(X_train.shape[1]),
    "feature_names": feature_names,
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Release train-only models and matrices before constructing the refit table.
models.clear()
del X_train, X_valid
gc.collect()

test = load("test")
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float32, copy=False)

if best["family"] == "incumbent" or best["alpha"] <= 0.0:
    test_scores = inc_test
else:
    combined = combine_splits(train, valid)
    combined_y = np.asarray(combined.y, dtype=np.int8)

    X_combined, _, _ = build_features(
        combined, combined, training=True
    )
    X_test, test_nonparam, _ = build_features(
        combined, test, training=False
    )

    if best["family"] == "nonparametric":
        raw_test = test_nonparam
    elif best["family"] == "gbdt":
        final_model = train_model(
            X_combined, combined_y, "gbdt", 210
        )
        raw_test = final_model.predict(X_test).astype(np.float32)
    elif best["family"] == "random_forest":
        final_model = train_model(
            X_combined, combined_y, "random_forest", 120
        )
        raw_test = final_model.predict(X_test).astype(np.float32)
    else:
        raise ValueError(best["family"])

    new_test_rank = rank_score(test.user_id, raw_test)
    inc_test_rank = rank_score(test.user_id, inc_test)
    test_scores = (
        (1.0 - best["alpha"]) * inc_test_rank
        + best["alpha"] * new_test_rank
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