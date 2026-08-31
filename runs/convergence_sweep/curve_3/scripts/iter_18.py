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
rng = np.random.default_rng(SEED)


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def within_user_rank(scores, users):
    scores = np.nan_to_num(
        np.asarray(scores, dtype=np.float64),
        nan=0.0,
        posinf=1e20,
        neginf=-1e20,
    )
    users = np.asarray(users, dtype=np.int64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]
    sorted_scores = scores[order]
    positions = np.arange(n, dtype=np.int64)

    user_start_flag = np.empty(n, dtype=bool)
    user_start_flag[0] = True
    user_start_flag[1:] = sorted_users[1:] != sorted_users[:-1]
    user_starts = np.maximum.accumulate(
        np.where(user_start_flag, positions, 0)
    )

    user_end_flag = np.empty(n, dtype=bool)
    user_end_flag[-1] = True
    user_end_flag[:-1] = sorted_users[:-1] != sorted_users[1:]
    user_ends = np.minimum.accumulate(
        np.where(user_end_flag, positions, n - 1)[::-1]
    )[::-1]

    tie_start_flag = np.empty(n, dtype=bool)
    tie_start_flag[0] = True
    tie_start_flag[1:] = (
        (sorted_users[1:] != sorted_users[:-1])
        | (sorted_scores[1:] != sorted_scores[:-1])
    )
    tie_starts = np.maximum.accumulate(
        np.where(tie_start_flag, positions, 0)
    )

    tie_end_flag = np.empty(n, dtype=bool)
    tie_end_flag[-1] = True
    tie_end_flag[:-1] = (
        (sorted_users[:-1] != sorted_users[1:])
        | (sorted_scores[:-1] != sorted_scores[1:])
    )
    tie_ends = np.minimum.accumulate(
        np.where(tie_end_flag, positions, n - 1)[::-1]
    )[::-1]

    local = 0.5 * (tie_starts + tie_ends) - user_starts
    denom = np.maximum(user_ends - user_starts, 1)
    ranked_sorted = local / denom

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


def recency_weights(dates, half_life=5.0):
    dates = np.asarray(dates, dtype=np.int32)
    age = np.max(dates) - dates
    return np.power(0.5, age.astype(np.float64) / half_life).astype(
        np.float32
    )


train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)
weights = recency_weights(train.date, half_life=5.0)
global_rate = float(np.sum(weights * y_train) / np.sum(weights))


TE_FIELDS = [
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

TE_STRENGTHS = {
    "video_id": 35.0,
    "author_id": 40.0,
    "user_id": 30.0,
    "tab": 150.0,
    "tag": 120.0,
    "onehot_feat3": 70.0,
    "upload_type": 150.0,
    "onehot_feat8": 80.0,
    "duration_bucket": 180.0,
    "onehot_feat1": 180.0,
    "music_type": 180.0,
    "onehot_feat7": 120.0,
    "user_active_degree": 180.0,
    "register_days_bucket": 180.0,
    "hour": 200.0,
}

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


def field_values(split, field):
    if field == "video_id":
        return np.asarray(split.video_id, dtype=np.int64)
    if field == "user_id":
        return np.asarray(split.user_id, dtype=np.int64)
    return np.asarray(split.X[field], dtype=np.int64)


def fit_target_table(field):
    values = field_values(train, field)
    cardinality = int(FEATURE_CARDINALITIES[field])
    count = np.bincount(
        values, weights=weights, minlength=cardinality
    ).astype(np.float64)
    positive = np.bincount(
        values, weights=weights * y_train, minlength=cardinality
    ).astype(np.float64)
    return count, positive


target_tables = {field: fit_target_table(field) for field in TE_FIELDS}


def target_encoding(split, field, train_loo=False):
    values = field_values(split, field)
    count, positive = target_tables[field]
    strength = TE_STRENGTHS[field]

    safe_values = np.minimum(values, len(count) - 1)
    c = count[safe_values].copy()
    p = positive[safe_values].copy()

    if train_loo:
        c -= weights
        p -= weights * y_train

    rate = (p + strength * global_rate) / np.maximum(
        c + strength, 1e-12
    )
    reliability = c / (c + strength)
    centered_logit = safe_logit(rate) - safe_logit(global_rate)
    return (
        centered_logit.astype(np.float32),
        np.log1p(np.maximum(c, 0.0)).astype(np.float32),
        reliability.astype(np.float32),
    )


def get_histories(split_name):
    result = {}
    result.update(historical_features(split_name, key="video_id"))
    result.update(historical_features(split_name, key="author_id"))
    return result


train_hist = get_histories("train")
valid_hist = get_histories("valid")
test_hist = get_histories("test")


def build_dense(split, histories, train_loo=False):
    columns = []
    names = []

    for field in TE_FIELDS:
        encoded, log_count, reliability = target_encoding(
            split, field, train_loo=train_loo
        )
        columns.append(encoded)
        names.append(field + "_te")
        if field in ("video_id", "author_id", "user_id"):
            columns.append(log_count)
            names.append(field + "_log_count")
            columns.append(reliability)
            names.append(field + "_reliability")

    for field in NUM_FIELDS:
        x = np.asarray(split.num[field], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))
        names.append("log_" + field)

    for key in HISTORY_KEYS:
        x = np.asarray(histories[key], dtype=np.float32)
        finite = np.isfinite(x)
        fill = float(np.median(x[finite])) if np.any(finite) else 0.0
        x = np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)
        columns.append(x.astype(np.float32))
        names.append(key)

    hour = np.asarray(split.X["hour"], dtype=np.float32)
    columns.append(np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32))
    names.append("hour_sin")
    columns.append(np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32))
    names.append("hour_cos")

    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    return matrix, names


X_train, feature_names = build_dense(train, train_hist, train_loo=True)
X_valid, _ = build_dense(valid, valid_hist, train_loo=False)
X_test, _ = build_dense(test, test_hist, train_loo=False)

finite_train = np.where(np.isfinite(X_train), X_train, np.nan)
feature_medians = np.nanmedian(finite_train, axis=0)
feature_medians = np.nan_to_num(feature_medians, nan=0.0)

for X in (X_train, X_valid, X_test):
    bad = ~np.isfinite(X)
    if np.any(bad):
        X[bad] = np.take(feature_medians, np.nonzero(bad)[1])


class AdditiveGAM:
    def __init__(self, n_bins=32, passes=8, learning_rate=0.16):
        self.n_bins = n_bins
        self.passes = passes
        self.learning_rate = learning_rate
        self.edges = []
        self.tables = []
        self.base = 0.0

    def fit(self, X, y, sample_weight):
        n, d = X.shape
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        self.edges = []
        bins = []

        for j in range(d):
            edge = np.unique(np.quantile(X[:, j], quantiles))
            self.edges.append(edge.astype(np.float32))
            bins.append(
                np.searchsorted(edge, X[:, j], side="right").astype(
                    np.int16
                )
            )

        rate = float(
            np.sum(sample_weight * y) / np.sum(sample_weight)
        )
        self.base = float(safe_logit(rate))
        prediction = np.full(n, self.base, dtype=np.float64)
        self.tables = [
            np.zeros(len(edge) + 1, dtype=np.float64)
            for edge in self.edges
        ]

        for _ in range(self.passes):
            for j in range(d):
                probability = sigmoid(prediction)
                residual = y - probability
                hessian = np.maximum(
                    probability * (1.0 - probability), 0.02
                )
                bj = bins[j]
                numerator = np.bincount(
                    bj,
                    weights=sample_weight * residual,
                    minlength=len(self.tables[j]),
                )
                denominator = np.bincount(
                    bj,
                    weights=sample_weight * hessian,
                    minlength=len(self.tables[j]),
                )
                update = numerator / (denominator + 30.0)
                update = np.clip(update, -0.6, 0.6)
                update *= self.learning_rate
                self.tables[j] += update
                prediction += update[bj]

        return self

    def predict(self, X):
        prediction = np.full(len(X), self.base, dtype=np.float64)
        for j, edge in enumerate(self.edges):
            bj = np.searchsorted(edge, X[:, j], side="right")
            prediction += self.tables[j][bj]
        return prediction.astype(np.float32)


gam = AdditiveGAM(n_bins=32, passes=8, learning_rate=0.16)
gam.fit(X_train, y_train, weights)
gam_valid = gam.predict(X_valid)
gam_test = gam.predict(X_test)


mean = np.average(X_train, axis=0, weights=weights).astype(np.float64)
variance = np.average(
    (X_train - mean) ** 2, axis=0, weights=weights
).astype(np.float64)
scale = np.sqrt(np.maximum(variance, 1e-4))

Z_train = np.clip((X_train - mean) / scale, -6.0, 6.0).astype(np.float32)
Z_valid = np.clip((X_valid - mean) / scale, -6.0, 6.0).astype(np.float32)
Z_test = np.clip((X_test - mean) / scale, -6.0, 6.0).astype(np.float32)

recent_pool = np.nonzero(np.asarray(train.date) >= 20220417)[0]
if len(recent_pool) < 96:
    recent_pool = np.arange(len(train.user_id))
center_indices = rng.choice(recent_pool, size=96, replace=False)
centers = Z_train[center_indices].copy()
gammas = [0.45, 1.20, 3.00]
ridge = 300.0

normal_matrices = [
    np.zeros((len(centers), len(centers)), dtype=np.float64)
    for _ in gammas
]
normal_vectors = [
    np.zeros(len(centers), dtype=np.float64) for _ in gammas
]
target_centered = y_train.astype(np.float64) - global_rate
center_norm = np.sum(centers * centers, axis=1).astype(np.float32)


def squared_distances(block):
    block_norm = np.sum(block * block, axis=1, keepdims=True)
    dist = block_norm + center_norm[None, :] - 2.0 * block @ centers.T
    np.maximum(dist, 0.0, out=dist)
    dist /= float(block.shape[1])
    return dist


chunk_size = 40000
for start in range(0, len(Z_train), chunk_size):
    end = min(start + chunk_size, len(Z_train))
    distance = squared_distances(Z_train[start:end])
    sw = weights[start:end].astype(np.float64)
    target = target_centered[start:end]
    root_weight = np.sqrt(sw)[:, None]

    for k, gamma in enumerate(gammas):
        kernel = np.exp(-gamma * distance).astype(np.float64)
        weighted_kernel = kernel * root_weight
        normal_matrices[k] += weighted_kernel.T @ weighted_kernel
        normal_vectors[k] += kernel.T @ (sw * target)

kernel_betas = []
for matrix, vector in zip(normal_matrices, normal_vectors):
    matrix.flat[:: len(centers) + 1] += ridge
    kernel_betas.append(np.linalg.solve(matrix, vector))


def kernel_predict(Z, gamma, beta):
    output = np.empty(len(Z), dtype=np.float32)
    for start in range(0, len(Z), chunk_size):
        end = min(start + chunk_size, len(Z))
        distance = squared_distances(Z[start:end])
        kernel = np.exp(-gamma * distance)
        output[start:end] = (
            safe_logit(global_rate) + kernel @ beta
        ).astype(np.float32)
    return output


kernel_valid_candidates = [
    kernel_predict(Z_valid, gamma, beta)
    for gamma, beta in zip(gammas, kernel_betas)
]
kernel_metrics = [
    evaluate(valid_users, y_valid, score)
    for score in kernel_valid_candidates
]
best_kernel_index = int(
    np.argmax([metric["primary"] for metric in kernel_metrics])
)
kernel_valid = kernel_valid_candidates[best_kernel_index]
kernel_test = kernel_predict(
    Z_test,
    gammas[best_kernel_index],
    kernel_betas[best_kernel_index],
)

del Z_train, normal_matrices, normal_vectors
gc.collect()

lgb_dataset = lgb.Dataset(
    X_train,
    label=y_train,
    weight=weights,
    feature_name=feature_names,
    free_raw_data=False,
)

linear_tree_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.045,
    "num_leaves": 15,
    "max_depth": 6,
    "min_data_in_leaf": 1000,
    "feature_fraction": 0.80,
    "bagging_fraction": 0.82,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 5.0,
    "linear_tree": True,
    "linear_lambda": 20.0,
    "max_bin": 127,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "num_threads": 8,
    "verbose": -1,
}

linear_tree = lgb.train(
    linear_tree_params,
    lgb_dataset,
    num_boost_round=90,
)
linear_valid = linear_tree.predict(X_valid).astype(np.float32)
linear_test = linear_tree.predict(X_test).astype(np.float32)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.load(inc_valid_path).astype(np.float64)
inc_test = np.load(inc_test_path).astype(np.float64)

models = {
    "nonlinear_additive_gam": (gam_valid, gam_test),
    "nystrom_rbf_kernel": (kernel_valid, kernel_test),
    "piecewise_linear_gbdt": (linear_valid, linear_test),
}

candidate_scores = {}
raw_metrics = {}
for name, (valid_score, _) in models.items():
    metric = evaluate(valid_users, y_valid, valid_score)
    raw_metrics[name] = metric
    candidate_scores[name + "_raw"] = float(metric["primary"])

for i, metric in enumerate(kernel_metrics):
    candidate_scores[
        "kernel_gamma_" + str(gammas[i])
    ] = float(metric["primary"])

inc_metric = evaluate(valid_users, y_valid, inc_valid)
candidate_scores["trusted_incumbent"] = float(inc_metric["primary"])

inc_valid_rank = within_user_rank(inc_valid, valid_users)
inc_test_rank = within_user_rank(inc_test, test_users)
blend_alphas = [0.04, 0.08, 0.12, 0.16, 0.24, 0.32, 0.50, 0.75]

best_primary = float(inc_metric["primary"])
best_valid_scores = inc_valid.copy()
best_test_scores = inc_test.copy()
best_raw_valid = gam_valid
best_name = "trusted_incumbent"
best_alpha = 0.0

for name, (valid_score, test_score) in models.items():
    valid_rank = within_user_rank(valid_score, valid_users)
    test_rank = within_user_rank(test_score, test_users)

    raw_primary = float(raw_metrics[name]["primary"])
    if raw_primary > best_primary:
        best_primary = raw_primary
        best_valid_scores = valid_score.astype(np.float64)
        best_test_scores = test_score.astype(np.float64)
        best_raw_valid = valid_score
        best_name = name
        best_alpha = 1.0

    family_best_blend = -1.0
    family_best_alpha = 0.0
    for alpha in blend_alphas:
        blended_valid = (
            (1.0 - alpha) * inc_valid_rank + alpha * valid_rank
        )
        metric = evaluate(valid_users, y_valid, blended_valid)
        primary = float(metric["primary"])

        if primary > family_best_blend:
            family_best_blend = primary
            family_best_alpha = alpha

        if primary > best_primary:
            best_primary = primary
            best_valid_scores = blended_valid
            best_test_scores = (
                (1.0 - alpha) * inc_test_rank + alpha * test_rank
            )
            best_raw_valid = valid_score
            best_name = name + "_incumbent_blend"
            best_alpha = alpha

    candidate_scores[name + "_best_blend"] = family_best_blend
    candidate_scores[name + "_blend_alpha"] = family_best_alpha

final_metrics = evaluate(valid_users, y_valid, best_valid_scores)

print(
    "FINDINGS "
    + json.dumps(
        {
            "selected": best_name,
            "selected_alpha": best_alpha,
            "selected_kernel_gamma": gammas[best_kernel_index],
            "dense_features": len(feature_names),
        },
        sort_keys=True,
    )
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test_scores, dtype=np.float64),
    )
    if "blend" in best_name or best_name == "trusted_incumbent":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(final_metrics["primary"]),
            "gauc": float(final_metrics["gauc"]),
            "ndcg@5": float(final_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)