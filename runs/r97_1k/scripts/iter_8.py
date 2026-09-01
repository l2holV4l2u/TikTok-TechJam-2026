import os
import gc
import json
import time
import warnings
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")

START = time.time()
SEED = 20260831
np.random.seed(SEED)

OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")
if OUT:
    os.makedirs(OUT, exist_ok=True)

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "user_active_degree",
    "fans_user_num_range",
    "register_days_bucket",
    "music_type",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

HALF_LIVES = [2.0, 4.0, 8.0]
RANK_HALF_LIFE = 4.0
MAX_QUERY_ROWS = 8000


def finite32(x, fill=0.0):
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float32), 1e-5, 1.0 - 1e-5)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def choose_history_keys(d):
    terms = [
        "long_view_rate",
        "count_log1p",
        "is_click_rate",
        "play_time_ms_logmean",
    ]
    selected = []
    for term in terms:
        matches = [k for k in sorted(d) if term in k.lower()]
        if matches:
            selected.append(matches[0])
    for key in sorted(d):
        if key not in selected:
            selected.append(key)
        if len(selected) >= 4:
            break
    return selected[:4]


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)
    if n == 0:
        return np.empty(0, dtype=np.float32)

    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=np.bool_)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    start_positions = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local = np.arange(n, dtype=np.float32) - start_positions.astype(np.float32)

    ends = np.empty(n, dtype=np.bool_)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(np.r_[-1, end_positions]).astype(np.float32)

    group_index = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[group_index] - 1.0, 1.0)
    ranked_sorted = local / denom

    result = np.empty(n, dtype=np.float32)
    result[order] = ranked_sorted
    return result


def temporal_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.int32)
    max_date = int(np.max(dates))
    age = (max_date - dates).astype(np.float32)
    w = np.power(0.5, age / float(half_life)).astype(np.float32)
    w /= max(float(np.mean(w)), 1e-8)
    return w


train = load("train")
train_y = np.asarray(train.y, dtype=np.float32)
train_dates = np.asarray(train.date, dtype=np.int32)
n_train = len(train.user_id)

te_weight = temporal_weights(train_dates, 4.0)
weighted_prior = float(
    np.sum(te_weight * train_y) / np.sum(te_weight)
)

user_card = int(FEATURE_CARDINALITIES["user_id"])
tag_card = int(FEATURE_CARDINALITIES["tag"])
duration_card = int(FEATURE_CARDINALITIES["duration_bucket"])
tab_card = int(FEATURE_CARDINALITIES["tab"])

TE_SPECS = [
    ("video", "video_id", int(FEATURE_CARDINALITIES["video_id"]), 30.0),
    ("author", "author_id", int(FEATURE_CARDINALITIES["author_id"]), 60.0),
    ("user_tag", None, user_card * tag_card, 80.0),
    ("user_duration", None, user_card * duration_card, 100.0),
    ("user_tab", None, user_card * tab_card, 120.0),
]


def composite_keys(split, name):
    if name == "video":
        return np.asarray(split.X["video_id"], dtype=np.int64)
    if name == "author":
        return np.asarray(split.X["author_id"], dtype=np.int64)

    users = np.asarray(split.X["user_id"], dtype=np.int64)
    if name == "user_tag":
        values = np.asarray(split.X["tag"], dtype=np.int64)
        return users * tag_card + values
    if name == "user_duration":
        values = np.asarray(split.X["duration_bucket"], dtype=np.int64)
        return users * duration_card + values
    if name == "user_tab":
        values = np.asarray(split.X["tab"], dtype=np.int64)
        return users * tab_card + values
    raise KeyError(name)


te_maps = {}
train_te = {}
train_eb_logits = []

for name, _, card, alpha in TE_SPECS:
    keys = composite_keys(train, name)
    sw = np.bincount(
        keys, weights=te_weight, minlength=card
    ).astype(np.float32)
    sy = np.bincount(
        keys, weights=te_weight * train_y, minlength=card
    ).astype(np.float32)

    loo_count = np.maximum(sw[keys] - te_weight, 0.0)
    loo_pos = sy[keys] - te_weight * train_y
    loo_rate = (
        loo_pos + float(alpha) * weighted_prior
    ) / np.maximum(loo_count + float(alpha), 1e-6)

    logit = safe_logit(loo_rate)
    count = np.log1p(loo_count).astype(np.float32)

    train_te[name] = (logit, count)
    train_eb_logits.append(logit)
    te_maps[name] = (sw, sy, float(alpha), card)

    del keys, loo_count, loo_pos, loo_rate
    gc.collect()

EB_COEFFICIENTS = np.asarray(
    [0.24, 0.28, 0.27, 0.12, 0.09], dtype=np.float32
)
train_eb_score = np.zeros(n_train, dtype=np.float32)
for coefficient, values in zip(EB_COEFFICIENTS, train_eb_logits):
    train_eb_score += coefficient * values
del train_eb_logits


def external_te(split):
    columns = {}
    eb_score = np.zeros(len(split.user_id), dtype=np.float32)

    for coefficient, (name, _, _, _) in zip(EB_COEFFICIENTS, TE_SPECS):
        keys = composite_keys(split, name)
        sw, sy, alpha, card = te_maps[name]

        rate = np.full(len(keys), weighted_prior, dtype=np.float32)
        count = np.zeros(len(keys), dtype=np.float32)
        ok = (keys >= 0) & (keys < card)
        selected = keys[ok]

        rate[ok] = (
            sy[selected] + alpha * weighted_prior
        ) / np.maximum(sw[selected] + alpha, 1e-6)
        count[ok] = np.log1p(sw[selected]).astype(np.float32)

        logit = safe_logit(rate)
        columns[name] = (logit, count)
        eb_score += coefficient * logit

        del keys, selected, rate
    return columns, eb_score


h_video_train = historical_features("train", key="video_id")
h_author_train = historical_features("train", key="author_id")
VIDEO_HKEYS = choose_history_keys(h_video_train)
AUTHOR_HKEYS = choose_history_keys(h_author_train)

feature_names = (
    ["cat_" + x for x in CAT_FIELDS]
    + ["num_" + x for x in NUM_FIELDS]
    + ["hist_video_" + x for x in VIDEO_HKEYS]
    + ["hist_author_" + x for x in AUTHOR_HKEYS]
    + [
        item
        for name, _, _, _ in TE_SPECS
        for item in ("te_" + name, "count_" + name)
    ]
)
categorical_indices = list(range(len(CAT_FIELDS)))


def build_matrix(split, hv, ha, te_columns):
    columns = []

    for field in CAT_FIELDS:
        columns.append(np.asarray(split.X[field], dtype=np.float32))

    for field in NUM_FIELDS:
        z = np.maximum(finite32(split.num[field]), 0.0)
        columns.append(np.log1p(z).astype(np.float32))

    for key in VIDEO_HKEYS:
        columns.append(finite32(hv[key]))
    for key in AUTHOR_HKEYS:
        columns.append(finite32(ha[key]))

    for name, _, _, _ in TE_SPECS:
        rate, count = te_columns[name]
        columns.append(finite32(rate))
        columns.append(finite32(count))

    return np.column_stack(columns).astype(np.float32, copy=False)


X_train = build_matrix(
    train, h_video_train, h_author_train, train_te
)
del h_video_train, h_author_train, train_te
gc.collect()

common_params = {
    "learning_rate": 0.075,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.82,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "max_cat_threshold": 32,
    "cat_smooth": 40.0,
    "cat_l2": 15.0,
    "verbosity": -1,
    "verbose": -1,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "data_random_seed": SEED + 3,
    "num_threads": min(16, os.cpu_count() or 1),
    "force_col_wise": True,
}

boosters = {}

for half_life in HALF_LIVES:
    name = "binary_hl%d" % int(half_life)
    weights = temporal_weights(train_dates, half_life)

    dataset = lgb.Dataset(
        X_train,
        label=train_y,
        weight=weights,
        categorical_feature=categorical_indices,
        feature_name=feature_names,
        free_raw_data=False,
    )
    params = dict(common_params)
    params.update({
        "objective": "binary",
        "metric": "binary_logloss",
    })
    boosters[name] = lgb.train(
        params,
        dataset,
        num_boost_round=150,
    )
    boosters[name].free_dataset()
    del dataset, weights
    gc.collect()


def make_rank_order_and_groups(user_ids, time_ms):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    time_ms = np.asarray(time_ms, dtype=np.int64)
    n = len(user_ids)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, time_ms, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=np.bool_)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    start_positions = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    within_user_position = np.arange(n, dtype=np.int64) - start_positions
    chunks = within_user_position // MAX_QUERY_ROWS

    boundaries = np.empty(n, dtype=np.bool_)
    boundaries[0] = True
    boundaries[1:] = (
        (sorted_users[1:] != sorted_users[:-1])
        | (chunks[1:] != chunks[:-1])
    )
    group_starts = np.flatnonzero(boundaries)
    groups = np.diff(np.r_[group_starts, n]).astype(np.int32)
    return order, groups


rank_order, rank_groups = make_rank_order_and_groups(
    train.user_id, train.time_ms
)
assert int(np.max(rank_groups)) <= MAX_QUERY_ROWS
assert int(np.sum(rank_groups)) == n_train

rank_weights = temporal_weights(train_dates, RANK_HALF_LIFE)
rank_dataset = lgb.Dataset(
    X_train[rank_order],
    label=train_y[rank_order],
    weight=rank_weights[rank_order],
    group=rank_groups,
    categorical_feature=categorical_indices,
    feature_name=feature_names,
    free_raw_data=False,
)

rank_params = dict(common_params)
rank_params.update({
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "lambdarank_truncation_level": 20,
    "label_gain": [0, 1],
    "learning_rate": 0.055,
    "num_leaves": 63,
})

boosters["lambdarank_chunked"] = lgb.train(
    rank_params,
    rank_dataset,
    num_boost_round=190,
)
boosters["lambdarank_chunked"].free_dataset()

print(
    "FINDINGS rank_queries=%d max_query_rows=%d median_query_rows=%.1f"
    % (
        len(rank_groups),
        int(np.max(rank_groups)),
        float(np.median(rank_groups)),
    ),
    flush=True,
)

del rank_dataset, rank_order, rank_groups, rank_weights
del X_train, train_y, train_dates, te_weight
gc.collect()

valid = load("valid")
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_uid = np.asarray(valid.user_id)

valid_te, valid_eb_score = external_te(valid)
h_video_valid = historical_features("valid", key="video_id")
h_author_valid = historical_features("valid", key="author_id")
X_valid = build_matrix(
    valid, h_video_valid, h_author_valid, valid_te
)
del h_video_valid, h_author_valid, valid_te
gc.collect()

valid_predictions = {
    name: booster.predict(X_valid).astype(np.float32)
    for name, booster in boosters.items()
}
valid_predictions["empirical_bayes"] = valid_eb_score.astype(np.float32)

inc_valid_path = (
    os.path.join(SHARED, "incumbent_valid_scores.npy")
    if SHARED else ""
)
inc_test_path = (
    os.path.join(SHARED, "incumbent_test_scores.npy")
    if SHARED else ""
)
has_incumbent = bool(
    inc_valid_path
    and inc_test_path
    and os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
)

candidate_scores = {}
best_primary = -np.inf
best_name = None
best_family = None
best_weight = 1.0
best_valid_scores = None
best_raw_valid = None

for name, scores in valid_predictions.items():
    metrics = evaluate(valid_uid, valid_y, scores)
    primary = float(metrics["primary"])
    candidate_scores[name] = primary

    if primary > best_primary:
        best_primary = primary
        best_name = name
        best_family = name
        best_weight = 1.0
        best_valid_scores = scores.copy()
        best_raw_valid = scores.copy()

if has_incumbent:
    incumbent_valid = np.load(inc_valid_path, mmap_mode="r")
    incumbent_metrics = evaluate(valid_uid, valid_y, incumbent_valid)
    incumbent_primary = float(incumbent_metrics["primary"])
    candidate_scores["trusted_incumbent"] = incumbent_primary
    incumbent_rank = within_user_rank(valid_uid, incumbent_valid)

    for name, scores in valid_predictions.items():
        own_rank = within_user_rank(valid_uid, scores)
        local_best = -np.inf
        local_weight = 0.0

        for weight in (
            0.05, 0.10, 0.15, 0.20, 0.25,
            0.30, 0.40, 0.50, 0.60, 0.70,
        ):
            blended = (
                weight * own_rank
                + (1.0 - weight) * incumbent_rank
            ).astype(np.float32)
            metrics = evaluate(valid_uid, valid_y, blended)
            primary = float(metrics["primary"])

            if primary > local_best:
                local_best = primary
                local_weight = float(weight)

            if primary > best_primary:
                best_primary = primary
                best_name = "%s_incblend_w%.2f" % (name, weight)
                best_family = name
                best_weight = float(weight)
                best_valid_scores = blended.copy()
                best_raw_valid = scores.copy()

        candidate_scores[name + "_best_blend"] = local_best
        print(
            "FINDINGS family=%s best_blend_weight=%.2f "
            "blend_primary=%.6f"
            % (name, local_weight, local_best),
            flush=True,
        )

    if incumbent_primary >= best_primary:
        best_standalone = max(
            valid_predictions,
            key=lambda x: candidate_scores[x]
        )
        best_primary = incumbent_primary
        best_name = "trusted_incumbent"
        best_family = "trusted_incumbent"
        best_weight = 0.0
        best_valid_scores = np.asarray(
            incumbent_valid, dtype=np.float32
        ).copy()
        best_raw_valid = valid_predictions[best_standalone].copy()

final_metrics = evaluate(valid_uid, valid_y, best_valid_scores)

print(
    "FINDINGS winner=%s video_history=%s author_history=%s features=%d"
    % (
        best_name,
        ",".join(VIDEO_HKEYS),
        ",".join(AUTHOR_HKEYS),
        X_valid.shape[1],
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_scores, sort_keys=True),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    if best_weight < 1.0:
        np.save(
            os.path.join(OUT, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

del X_valid, valid_eb_score, best_valid_scores, best_raw_valid, valid_y
gc.collect()

test = load("test")

if best_family == "trusted_incumbent":
    test_scores = np.asarray(
        np.load(inc_test_path, mmap_mode="r"),
        dtype=np.float32,
    ).copy()
else:
    test_te, test_eb_score = external_te(test)

    if best_family == "empirical_bayes":
        own_test = test_eb_score.astype(np.float32)
    else:
        h_video_test = historical_features("test", key="video_id")
        h_author_test = historical_features("test", key="author_id")
        X_test = build_matrix(
            test, h_video_test, h_author_test, test_te
        )
        del h_video_test, h_author_test
        gc.collect()

        own_test = boosters[best_family].predict(
            X_test
        ).astype(np.float32)
        del X_test

    if best_weight < 1.0 and has_incumbent:
        incumbent_test = np.load(inc_test_path, mmap_mode="r")
        own_test_rank = within_user_rank(test.user_id, own_test)
        incumbent_test_rank = within_user_rank(
            test.user_id, incumbent_test
        )
        test_scores = (
            best_weight * own_test_rank
            + (1.0 - best_weight) * incumbent_test_rank
        ).astype(np.float32)
    else:
        test_scores = own_test

    del test_te, test_eb_score, own_test
    gc.collect()

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