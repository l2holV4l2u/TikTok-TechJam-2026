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
OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")

if OUT:
    os.makedirs(OUT, exist_ok=True)

CAT_FIELDS = [
    "user_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat1",
    "onehot_feat7",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

TE_FIELDS = [
    "author_id",
    "video_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
]

TE_ALPHA = {
    "author_id": 55.0,
    "video_id": 30.0,
    "tag": 180.0,
    "tab": 240.0,
    "duration_bucket": 240.0,
    "upload_type": 200.0,
    "onehot_feat3": 110.0,
    "onehot_feat8": 110.0,
}

TE_WEIGHT = {
    "author_id": 1.00,
    "video_id": 0.65,
    "tag": 0.60,
    "tab": 0.85,
    "duration_bucket": 0.35,
    "upload_type": 0.45,
    "onehot_feat3": 0.70,
    "onehot_feat8": 0.60,
}

HISTORY_SUFFIXES = (
    "train_count_log1p",
    "long_view_rate",
    "is_click_rate",
    "is_like_rate",
    "is_comment_rate",
    "is_hate_rate",
    "play_time_ms_logmean",
    "comment_stay_time_logmean",
)


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float32), 1e-4, 1.0 - 1e-4)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    group_start = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local = (
        np.arange(n, dtype=np.float32)
        - group_start.astype(np.float32)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(np.r_[-1, end_positions]).astype(np.float32)

    group_index = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[group_index] - 1.0, 1.0)

    result = np.empty(n, dtype=np.float32)
    result[order] = local / denom
    return result


def transformed_numeric(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.log1p(np.maximum(x, 0.0)).astype(np.float32)


def selected_history_names(history_dict):
    names = []
    for name in sorted(history_dict.keys()):
        if any(name.endswith(suffix) for suffix in HISTORY_SUFFIXES):
            names.append(name)
    return names


def build_dense(split, hist_video, hist_author, history_names):
    n = len(split.user_id)
    columns = []

    for field in CAT_FIELDS:
        columns.append(np.asarray(split.X[field], dtype=np.float32))

    for field in NUM_FIELDS:
        columns.append(transformed_numeric(split.num[field]))

    for source, prefix in (
        (hist_video, "video_id_"),
        (hist_author, "author_id_"),
    ):
        for name in history_names:
            if name.startswith(prefix) and name in source:
                arr = np.asarray(source[name], dtype=np.float32)
                arr = np.nan_to_num(
                    arr, nan=0.0, posinf=0.0, neginf=0.0
                )
                columns.append(arr)

    matrix = np.empty((n, len(columns)), dtype=np.float32)
    for j, col in enumerate(columns):
        matrix[:, j] = col

    del columns
    return matrix


def build_linear_dense(split, hist_video, hist_author, linear_names):
    columns = []

    for field in NUM_FIELDS:
        columns.append(transformed_numeric(split.num[field]))

    for source in (hist_video, hist_author):
        for name in linear_names:
            if name in source:
                arr = np.asarray(source[name], dtype=np.float32)
                arr = np.nan_to_num(
                    arr, nan=0.0, posinf=0.0, neginf=0.0
                )
                columns.append(arr)

    matrix = np.empty((len(split.user_id), len(columns)), dtype=np.float32)
    for j, col in enumerate(columns):
        matrix[:, j] = col

    del columns
    return matrix


train = load("train")
train_y = np.asarray(train.y, dtype=np.float32)
train_date = np.asarray(train.date, dtype=np.int32)
max_train_date = int(np.max(train_date))
age = (max_train_date - train_date).astype(np.float32)

# Main models emphasize observations nearest the deployment boundary.
sample_weight = np.power(0.5, age / 4.0).astype(np.float32)
sample_weight /= np.mean(sample_weight)
prior = float(np.sum(sample_weight * train_y) / np.sum(sample_weight))

# Recency-weighted additive target-statistics family.
te_maps = {}
for field in TE_FIELDS:
    ids = np.asarray(train.X[field], dtype=np.int64)
    card = int(FEATURE_CARDINALITIES[field])
    sw = np.bincount(
        ids, weights=sample_weight, minlength=card
    ).astype(np.float32)
    sy = np.bincount(
        ids, weights=sample_weight * train_y, minlength=card
    ).astype(np.float32)
    te_maps[field] = (sw, sy)

hist_video_train = historical_features("train", key="video_id")
hist_author_train = historical_features("train", key="author_id")

all_history_names = sorted(
    set(selected_history_names(hist_video_train))
    | set(selected_history_names(hist_author_train))
)

linear_names = sorted(
    [
        name for name in all_history_names
        if (
            name.endswith("train_count_log1p")
            or name.endswith("long_view_rate")
            or name.endswith("is_click_rate")
            or name.endswith("play_time_ms_logmean")
            or name.endswith("comment_stay_time_logmean")
            or name.endswith("is_hate_rate")
        )
    ]
)

X_train = build_dense(
    train,
    hist_video_train,
    hist_author_train,
    all_history_names,
)

categorical_indices = list(range(len(CAT_FIELDS)))

dtrain = lgb.Dataset(
    X_train,
    label=train_y,
    weight=sample_weight,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "max_depth": 10,
    "min_data_in_leaf": 900,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.86,
    "bagging_freq": 1,
    "lambda_l1": 0.15,
    "lambda_l2": 2.5,
    "max_bin": 127,
    "cat_smooth": 25.0,
    "cat_l2": 12.0,
    "max_cat_threshold": 64,
    "verbosity": -1,
    "num_threads": max(1, min(16, os.cpu_count() or 8)),
    "seed": 2026,
}

gbdt = lgb.train(
    params,
    dtrain,
    num_boost_round=210,
)

# A structurally different weighted ridge model over only continuous and
# train-derived history features. It cannot form tree interactions and thus
# supplies a smoother ordering under temporal drift.
X_linear_train = build_linear_dense(
    train,
    hist_video_train,
    hist_author_train,
    linear_names,
)

w_sum = float(np.sum(sample_weight))
linear_mean = (
    np.sum(
        X_linear_train * sample_weight[:, None],
        axis=0,
        dtype=np.float64,
    ) / w_sum
).astype(np.float32)

centered = X_linear_train - linear_mean
linear_var = (
    np.sum(
        centered * centered * sample_weight[:, None],
        axis=0,
        dtype=np.float64,
    ) / w_sum
)
linear_scale = np.sqrt(np.maximum(linear_var, 1e-6)).astype(np.float32)
centered /= linear_scale

p = centered.shape[1]
gram = (
    centered.T.astype(np.float64)
    @ (centered.astype(np.float64) * sample_weight[:, None])
)
rhs = (
    centered.T.astype(np.float64)
    @ (
        sample_weight.astype(np.float64)
        * (train_y.astype(np.float64) - prior)
    )
)

ridge_penalty = 8.0 * w_sum / max(len(train_y), 1)
gram.flat[::p + 1] += ridge_penalty
linear_coef = np.linalg.solve(gram, rhs).astype(np.float32)

print(
    "FINDINGS train_prior=%.6f dense_features=%d linear_features=%d "
    "history_features=%d"
    % (
        prior,
        X_train.shape[1],
        X_linear_train.shape[1],
        len(all_history_names),
    ),
    flush=True,
)

del dtrain, X_train, X_linear_train, centered, gram, rhs
del hist_video_train, hist_author_train
del train_date, age
del train
gc.collect()


def target_stat_score(split):
    total = np.zeros(len(split.user_id), dtype=np.float32)
    scale = 0.0

    for field in TE_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        sw, sy = te_maps[field]
        alpha = float(TE_ALPHA[field])

        rate = np.full(len(ids), prior, dtype=np.float32)
        ok = (ids >= 0) & (ids < len(sw))
        selected = ids[ok]
        rate[ok] = (
            sy[selected] + alpha * prior
        ) / np.maximum(sw[selected] + alpha, 1e-6)

        fw = float(TE_WEIGHT[field])
        total += fw * safe_logit(rate)
        scale += fw

    return total / scale


def predict_new_families(split_name, split):
    hist_video = historical_features(split_name, key="video_id")
    hist_author = historical_features(split_name, key="author_id")

    X = build_dense(
        split,
        hist_video,
        hist_author,
        all_history_names,
    )
    pred_gbdt = gbdt.predict(
        X,
        num_iteration=gbdt.current_iteration(),
    ).astype(np.float32)
    del X
    gc.collect()

    Xlin = build_linear_dense(
        split,
        hist_video,
        hist_author,
        linear_names,
    )
    Xlin -= linear_mean
    Xlin /= linear_scale
    pred_linear = (
        Xlin @ linear_coef + prior
    ).astype(np.float32)

    pred_te = target_stat_score(split)

    del Xlin, hist_video, hist_author
    gc.collect()

    return {
        "gbdt_binary": pred_gbdt,
        "linear_history": pred_linear,
        "marginal_te": pred_te,
    }


valid = load("valid")
valid_uid = np.asarray(valid.user_id)
valid_y = np.asarray(valid.y)

valid_pred = predict_new_families("valid", valid)

inc_valid_path = (
    os.path.join(SHARED, "incumbent_valid_scores.npy")
    if SHARED else ""
)
inc_test_path = (
    os.path.join(SHARED, "incumbent_test_scores.npy")
    if SHARED else ""
)

if not (
    inc_valid_path
    and inc_test_path
    and os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise RuntimeError("trusted incumbent predictions are required")

inc_valid = np.asarray(
    np.load(inc_valid_path, mmap_mode="r"),
    dtype=np.float32,
)

rank_names = [
    "incumbent",
    "gbdt_binary",
    "linear_history",
    "marginal_te",
]

valid_ranks = {
    "incumbent": within_user_rank(valid_uid, inc_valid)
}
for name, pred in valid_pred.items():
    valid_ranks[name] = within_user_rank(valid_uid, pred)

candidate_scores = {}
candidate_specs = {}

for name in rank_names:
    met = evaluate(valid_uid, valid_y, valid_ranks[name])
    candidate_scores[name] = float(met["primary"])
    candidate_specs[name] = {name: 1.0}

# One-new-family Borda blends establish which model contributes genuinely
# complementary ordering to the incumbent.
blend_grid = (
    0.025, 0.05, 0.075, 0.10, 0.15,
    0.20, 0.25, 0.30, 0.40, 0.50,
)

for name in ("gbdt_binary", "linear_history", "marginal_te"):
    local_best = -np.inf
    local_weight = 0.0

    for weight in blend_grid:
        score = (
            (1.0 - weight) * valid_ranks["incumbent"]
            + weight * valid_ranks[name]
        ).astype(np.float32)
        met = evaluate(valid_uid, valid_y, score)
        value = float(met["primary"])

        if value > local_best:
            local_best = value
            local_weight = float(weight)

    key = name + "_best_inc_blend"
    candidate_scores[key] = local_best
    candidate_specs[key] = {
        "incumbent": 1.0 - local_weight,
        name: local_weight,
    }

    print(
        "FINDINGS family=%s best_single_blend_weight=%.3f "
        "primary=%.6f"
        % (name, local_weight, local_best),
        flush=True,
    )

# Joint rank aggregation. The grid is deliberately concentrated around small
# challenger weights because the incumbent is already substantially stronger.
joint_weights = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25)

joint_counter = 0
for wg in joint_weights:
    for wl in joint_weights:
        for wt in joint_weights:
            own_sum = wg + wl + wt
            if own_sum <= 0.0 or own_sum > 0.45:
                continue
            if sum(v > 0 for v in (wg, wl, wt)) < 2:
                continue

            wi = 1.0 - own_sum
            score = (
                wi * valid_ranks["incumbent"]
                + wg * valid_ranks["gbdt_binary"]
                + wl * valid_ranks["linear_history"]
                + wt * valid_ranks["marginal_te"]
            ).astype(np.float32)

            met = evaluate(valid_uid, valid_y, score)
            value = float(met["primary"])
            key = "joint_%03d" % joint_counter
            joint_counter += 1

            candidate_scores[key] = value
            candidate_specs[key] = {
                "incumbent": wi,
                "gbdt_binary": wg,
                "linear_history": wl,
                "marginal_te": wt,
            }

best_name = max(candidate_scores, key=candidate_scores.get)
best_spec = candidate_specs[best_name]

best_valid = np.zeros(len(valid_uid), dtype=np.float32)
for name, weight in best_spec.items():
    best_valid += float(weight) * valid_ranks[name]

final_metrics = evaluate(valid_uid, valid_y, best_valid)

# Compact candidate log: retain standalone scores, each single-family blend,
# and the best joint result rather than printing hundreds of grid entries.
logged_candidates = {
    key: value
    for key, value in candidate_scores.items()
    if not key.startswith("joint_")
}
logged_candidates["best_joint_or_overall"] = float(
    candidate_scores[best_name]
)

corr = np.corrcoef(
    np.stack(
        [
            valid_ranks["incumbent"],
            valid_ranks["gbdt_binary"],
            valid_ranks["linear_history"],
            valid_ranks["marginal_te"],
        ],
        axis=0,
    )
)

print(
    "FINDINGS winner=%s weights=%s rank_corr_inc_gbdt=%.5f "
    "rank_corr_inc_linear=%.5f rank_corr_inc_te=%.5f"
    % (
        best_name,
        json.dumps(best_spec, sort_keys=True),
        float(corr[0, 1]),
        float(corr[0, 2]),
        float(corr[0, 3]),
    ),
    flush=True,
)

print(
    "CANDIDATES " + json.dumps(logged_candidates, sort_keys=True),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )

    own_weight = sum(
        weight for name, weight in best_spec.items()
        if name != "incumbent"
    )
    if own_weight > 0:
        own_valid = np.zeros(len(valid_uid), dtype=np.float32)
        for name, weight in best_spec.items():
            if name != "incumbent":
                own_valid += (
                    float(weight) / own_weight
                ) * valid_ranks[name]
    else:
        strongest_own = max(
            valid_pred,
            key=lambda name: candidate_scores[name],
        )
        own_valid = valid_ranks[strongest_own]

    np.save(
        os.path.join(OUT, "scores_valid_raw.npy"),
        np.asarray(own_valid, dtype=np.float64),
    )

del inc_valid, best_valid, own_valid
del valid_pred, valid_ranks, valid
gc.collect()

test = load("test")
test_pred = predict_new_families("test", test)

inc_test = np.asarray(
    np.load(inc_test_path, mmap_mode="r"),
    dtype=np.float32,
)

test_ranks = {
    "incumbent": within_user_rank(test.user_id, inc_test)
}
for name, pred in test_pred.items():
    test_ranks[name] = within_user_rank(test.user_id, pred)

test_scores = np.zeros(len(test.user_id), dtype=np.float32)
for name, weight in best_spec.items():
    test_scores += float(weight) * test_ranks[name]

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