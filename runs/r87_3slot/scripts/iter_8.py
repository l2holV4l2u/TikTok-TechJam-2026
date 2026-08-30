import os
import time
import json
import gc
import random
import numpy as np
import lightgbm as lgb
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7351
THREADS = max(1, min(8, os.cpu_count() or 1))
np.random.seed(SEED)
random.seed(SEED)

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "duration_bucket", "upload_type", "music_type",
    "onehot_feat3", "onehot_feat8",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
HALF_LIFE_DAYS = 6.0
SMOOTH_ENTITY = 24.0
SMOOTH_PAIR = 8.0


def concatenate_parts(parts, source, name, dtype=None):
    if source == "X":
        arrays = [np.asarray(p.X[name]) for p in parts]
    elif source == "num":
        arrays = [np.asarray(p.num[name]) for p in parts]
    elif source == "date":
        arrays = [np.asarray(p.date) for p in parts]
    elif source == "y":
        arrays = [np.asarray(p.y) for p in parts]
    else:
        raise ValueError(source)
    x = arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
    if dtype is not None:
        x = x.astype(dtype, copy=False)
    return x


def date_age(date):
    # All training/refit dates are in April 2022, so subtraction is exact.
    date = np.asarray(date, dtype=np.int64)
    return np.max(date) - date


def recency_weights(date):
    age = date_age(date).astype(np.float64)
    return np.exp2(-age / HALF_LIFE_DAYS).astype(np.float32)


def make_key(parts, fields):
    if len(fields) == 1:
        return concatenate_parts(parts, "X", fields[0], np.int64)

    a = concatenate_parts(parts, "X", fields[0], np.int64)
    b = concatenate_parts(parts, "X", fields[1], np.int64)
    card_b = int(FEATURE_CARDINALITIES[fields[1]])
    return a * np.int64(card_b) + b


class SmoothedHistory:
    def __init__(self, smoothing):
        self.smoothing = float(smoothing)
        self.keys = None
        self.count = None
        self.positive = None
        self.prior = None

    def fit_transform(self, key, y, row_weight):
        key = np.asarray(key, dtype=np.int64)
        y = np.asarray(y, dtype=np.float64)
        w = np.asarray(row_weight, dtype=np.float64)

        keys, inv = np.unique(key, return_inverse=True)
        count = np.bincount(inv, weights=w).astype(np.float64)
        positive = np.bincount(inv, weights=w * y).astype(np.float64)
        prior = float(np.sum(w * y) / max(np.sum(w), 1.0))

        loo_count = np.maximum(count[inv] - w, 0.0)
        loo_positive = positive[inv] - w * y
        rate = (
            loo_positive + self.smoothing * prior
        ) / (loo_count + self.smoothing)

        self.keys = keys
        self.count = count
        self.positive = positive
        self.prior = prior

        return (
            rate.astype(np.float32),
            np.log1p(loo_count).astype(np.float32),
        )

    def transform(self, key):
        key = np.asarray(key, dtype=np.int64)
        pos = np.searchsorted(self.keys, key)
        good = pos < len(self.keys)
        matched = np.zeros(len(key), dtype=bool)
        matched[good] = self.keys[pos[good]] == key[good]

        count = np.zeros(len(key), dtype=np.float64)
        positive = np.zeros(len(key), dtype=np.float64)
        count[matched] = self.count[pos[matched]]
        positive[matched] = self.positive[pos[matched]]

        rate = (
            positive + self.smoothing * self.prior
        ) / (count + self.smoothing)
        return (
            rate.astype(np.float32),
            np.log1p(count).astype(np.float32),
        )


STAT_SPECS = [
    ("video", ("video_id",), SMOOTH_ENTITY),
    ("author", ("author_id",), SMOOTH_ENTITY),
    ("tag", ("tag",), SMOOTH_ENTITY),
    ("duration", ("duration_bucket",), SMOOTH_ENTITY),
    ("upload", ("upload_type",), SMOOTH_ENTITY),
    ("user_video", ("user_id", "video_id"), SMOOTH_PAIR),
    ("user_author", ("user_id", "author_id"), SMOOTH_PAIR),
    ("user_tag", ("user_id", "tag"), SMOOTH_PAIR),
    ("user_duration", ("user_id", "duration_bucket"), SMOOTH_PAIR),
    ("user_tab", ("user_id", "tab"), SMOOTH_PAIR),
]


def fit_history_features(fit_parts, eval_parts):
    y = concatenate_parts(fit_parts, "y", None, np.float32)
    dates = concatenate_parts(fit_parts, "date", None, np.int64)
    weights = recency_weights(dates)

    train_columns = []
    eval_columns = []
    models = {}

    for name, fields, smoothing in STAT_SPECS:
        estimator = SmoothedHistory(smoothing)
        train_key = make_key(fit_parts, fields)
        eval_key = make_key(eval_parts, fields)

        train_rate, train_count = estimator.fit_transform(
            train_key, y, weights
        )
        eval_rate, eval_count = estimator.transform(eval_key)

        train_columns.extend([train_rate, train_count])
        eval_columns.extend([eval_rate, eval_count])
        models[name] = estimator

    return (
        np.ascontiguousarray(np.column_stack(train_columns), dtype=np.float32),
        np.ascontiguousarray(np.column_stack(eval_columns), dtype=np.float32),
        models,
        weights,
    )


def numeric_statistics(parts):
    centers = []
    scales = []
    for field in NUM_FIELDS:
        x = concatenate_parts(parts, "num", field, np.float64)
        x = np.log1p(np.maximum(
            np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), 0.0
        ))
        centers.append(float(np.median(x)))
        q25, q75 = np.percentile(x, [25.0, 75.0])
        scales.append(max(float(q75 - q25), 0.25))
    return np.asarray(centers), np.asarray(scales)


def make_numeric(parts, centers, scales):
    columns = []
    for j, field in enumerate(NUM_FIELDS):
        x = concatenate_parts(parts, "num", field, np.float32)
        x = np.log1p(np.maximum(
            np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), 0.0
        ))
        x = np.clip((x - centers[j]) / scales[j], -6.0, 6.0)
        columns.append(x.astype(np.float32))
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def make_categories(parts):
    return np.ascontiguousarray(np.column_stack([
        concatenate_parts(parts, "X", field, np.int32)
        for field in CAT_FIELDS
    ]), dtype=np.int32)


def make_gbdt_matrix(parts, history, centers, scales):
    cats = make_categories(parts).astype(np.float32)
    nums = make_numeric(parts, centers, scales)
    return np.ascontiguousarray(
        np.column_stack([cats, history, nums]), dtype=np.float32
    )


def logit(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 1.0e-4, 1.0 - 1.0e-4)
    return np.log(x) - np.log1p(-x)


def empirical_bayes_scores(history, variant):
    # Each statistic contributes rate,count in adjacent columns.
    rate = {name: history[:, 2 * i] for i, (name, _, _) in enumerate(STAT_SPECS)}
    cnt = {name: history[:, 2 * i + 1] for i, (name, _, _) in enumerate(STAT_SPECS)}

    if variant == "recent_personal":
        terms = [
            (0.35, "user_author"),
            (0.20, "user_tag"),
            (0.10, "user_duration"),
            (0.05, "user_tab"),
            (0.12, "user_video"),
            (0.09, "video"),
            (0.07, "author"),
            (0.02, "duration"),
        ]
    elif variant == "stable_personal":
        terms = [
            (0.25, "user_author"),
            (0.16, "user_tag"),
            (0.08, "user_duration"),
            (0.04, "user_tab"),
            (0.08, "user_video"),
            (0.18, "video"),
            (0.16, "author"),
            (0.05, "duration"),
        ]
    else:
        raise ValueError(variant)

    score = np.zeros(history.shape[0], dtype=np.float64)
    for coefficient, name in terms:
        confidence = 1.0 - np.exp(-np.asarray(cnt[name], np.float64) / 2.0)
        score += coefficient * confidence * logit(rate[name])
    return score


def fit_svd(parts, rank=20):
    users = concatenate_parts(parts, "X", "user_id", np.int64)
    videos = concatenate_parts(parts, "X", "video_id", np.int64)
    y = concatenate_parts(parts, "y", None, np.float64)
    dates = concatenate_parts(parts, "date", None, np.int64)
    w = recency_weights(dates).astype(np.float64)

    n_users = int(FEATURE_CARDINALITIES["user_id"])
    n_videos = int(FEATURE_CARDINALITIES["video_id"])
    code = users * np.int64(n_videos) + videos
    unique_code, inverse = np.unique(code, return_inverse=True)

    count = np.bincount(inverse, weights=w)
    positive = np.bincount(inverse, weights=w * y)
    prior = float(np.sum(w * y) / np.sum(w))
    mean = (positive + 3.0 * prior) / (count + 3.0)
    values = (mean - prior) * np.sqrt(np.minimum(count, 8.0))

    rows = (unique_code // n_videos).astype(np.int32)
    cols = (unique_code % n_videos).astype(np.int32)
    matrix = coo_matrix(
        (values.astype(np.float32), (rows, cols)),
        shape=(n_users, n_videos),
    ).tocsr()

    u, s, vt = svds(
        matrix,
        k=rank,
        which="LM",
        random_state=SEED,
        return_singular_vectors=True,
    )
    order = np.argsort(s)[::-1]
    s = s[order]
    u = u[:, order]
    vt = vt[order]
    user_factor = u * np.sqrt(s)[None, :]
    video_factor = vt.T * np.sqrt(s)[None, :]
    return user_factor.astype(np.float32), video_factor.astype(np.float32)


def predict_svd(model, parts):
    user_factor, video_factor = model
    users = concatenate_parts(parts, "X", "user_id", np.int64)
    videos = concatenate_parts(parts, "X", "video_id", np.int64)
    return np.sum(
        user_factor[users] * video_factor[videos], axis=1
    ).astype(np.float64)


def fit_gbdt(x_train, y_train, weights, x_valid, y_valid):
    categorical = list(range(len(CAT_FIELDS)))
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        weight=weights,
        categorical_feature=categorical,
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        x_valid,
        label=y_valid,
        categorical_feature=categorical,
        reference=train_set,
        free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.055,
        "num_leaves": 63,
        "min_data_in_leaf": 500,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 1.0,
        "max_cat_to_onehot": 16,
        "cat_smooth": 20.0,
        "cat_l2": 8.0,
        "force_col_wise": True,
        "num_threads": THREADS,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "verbose": -1,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=360,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(35, verbose=False)],
    )
    return booster, params


def refit_gbdt(x, y, weights, params, rounds):
    dset = lgb.Dataset(
        x,
        label=y,
        weight=weights,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )
    return lgb.train(
        params,
        dset,
        num_boost_round=max(1, int(rounds)),
    )


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

hist_train, hist_valid, _, train_weights = fit_history_features(
    [train], [valid]
)

eb_recent = empirical_bayes_scores(hist_valid, "recent_personal")
eb_stable = empirical_bayes_scores(hist_valid, "stable_personal")

svd_model = fit_svd([train], rank=20)
svd_valid = predict_svd(svd_model, [valid])
# Add a modest stable item prior to the low-rank interaction score.
video_rate_valid = hist_valid[:, 0]
svd_valid = svd_valid + 0.12 * logit(video_rate_valid)

centers, scales = numeric_statistics([train])
x_train = make_gbdt_matrix([train], hist_train, centers, scales)
x_valid = make_gbdt_matrix([valid], hist_valid, centers, scales)

gbdt, gbdt_params = fit_gbdt(
    x_train, y_train, train_weights, x_valid, y_valid
)
gbdt_valid = gbdt.predict(
    x_valid, num_iteration=gbdt.best_iteration
).astype(np.float64)
gbdt_rounds = int(gbdt.best_iteration)

raw_predictions = {
    "empirical_bayes_recent": eb_recent,
    "empirical_bayes_stable": eb_stable,
    "low_rank_svd": svd_valid,
    "history_gbdt": gbdt_valid,
}

inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)
candidate_values = {"incumbent": float(inc_metrics["primary"])}

best_name = "incumbent"
best_scores = inc_valid.copy()
best_metrics = inc_metrics
best_raw = None
best_family = "incumbent"
best_alpha = 0.0
best_scale = 1.0

inc_std = max(float(np.std(inc_valid)), 1.0e-8)
blend_alphas = [0.15, 0.25, 0.35, 0.50, 0.65]

for family, raw in raw_predictions.items():
    raw_metrics = evaluate(valid.user_id, y_valid, raw)
    candidate_values[family] = float(raw_metrics["primary"])

    if raw_metrics["primary"] > best_metrics["primary"]:
        best_name = family
        best_scores = raw.copy()
        best_metrics = raw_metrics
        best_raw = raw.copy()
        best_family = family
        best_alpha = 1.0
        best_scale = 1.0

    scale_factor = inc_std / max(float(np.std(raw)), 1.0e-8)
    scaled = raw * scale_factor

    for alpha in blend_alphas:
        blended = (1.0 - alpha) * inc_valid + alpha * scaled
        metrics = evaluate(valid.user_id, y_valid, blended)
        name = "%s_blend_%.2f" % (family, alpha)
        candidate_values[name] = float(metrics["primary"])

        if metrics["primary"] > best_metrics["primary"]:
            best_name = name
            best_scores = blended.copy()
            best_metrics = metrics
            best_raw = raw.copy()
            best_family = family
            best_alpha = float(alpha)
            best_scale = float(scale_factor)

print("CANDIDATES " + json.dumps(candidate_values, sort_keys=True), flush=True)
print(
    "FINDINGS selected=%s gbdt_rounds=%d "
    "eb_recent=%.6f eb_stable=%.6f svd=%.6f gbdt=%.6f"
    % (
        best_name,
        gbdt_rounds,
        candidate_values["empirical_bayes_recent"],
        candidate_values["empirical_bayes_stable"],
        candidate_values["low_rank_svd"],
        candidate_values["history_gbdt"],
    ),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_raw is not None and best_alpha < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

test = load("test")
inc_test = np.load(inc_test_path).astype(np.float64)

if best_family == "incumbent":
    test_scores = inc_test.copy()
else:
    y_tv = np.concatenate([
        y_train, y_valid.astype(np.float32)
    ]).astype(np.float32)

    hist_tv, hist_test, _, tv_weights = fit_history_features(
        [train, valid], [test]
    )

    if best_family == "empirical_bayes_recent":
        test_raw = empirical_bayes_scores(hist_test, "recent_personal")

    elif best_family == "empirical_bayes_stable":
        test_raw = empirical_bayes_scores(hist_test, "stable_personal")

    elif best_family == "low_rank_svd":
        del svd_model
        gc.collect()
        svd_tv_model = fit_svd([train, valid], rank=20)
        test_raw = predict_svd(svd_tv_model, [test])
        test_raw = test_raw + 0.12 * logit(hist_test[:, 0])

    elif best_family == "history_gbdt":
        centers_tv, scales_tv = numeric_statistics([train, valid])
        x_tv = make_gbdt_matrix(
            [train, valid], hist_tv, centers_tv, scales_tv
        )
        x_test = make_gbdt_matrix(
            [test], hist_test, centers_tv, scales_tv
        )
        refit = refit_gbdt(
            x_tv, y_tv, tv_weights, gbdt_params, gbdt_rounds
        )
        test_raw = refit.predict(
            x_test, num_iteration=gbdt_rounds
        ).astype(np.float64)

    else:
        raise ValueError(best_family)

    if best_alpha >= 1.0:
        test_scores = np.asarray(test_raw, dtype=np.float64)
    else:
        test_scores = (
            (1.0 - best_alpha) * inc_test
            + best_alpha * best_scale * np.asarray(test_raw, dtype=np.float64)
        )

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.4f}'
    % (
        float(best_metrics["primary"]),
        float(best_metrics["gauc"]),
        float(best_metrics["ndcg@5"]),
        elapsed,
    ),
    flush=True,
)