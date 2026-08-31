import os
import time
import json
import gc

import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 20260831
rng = np.random.default_rng(SEED)


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def logsumexp(a, axis=1):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


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
    su = users[order]
    ss = scores[order]
    pos = np.arange(n, dtype=np.int64)

    user_start = np.empty(n, dtype=bool)
    user_start[0] = True
    user_start[1:] = su[1:] != su[:-1]
    starts = np.maximum.accumulate(np.where(user_start, pos, 0))

    user_end = np.empty(n, dtype=bool)
    user_end[-1] = True
    user_end[:-1] = su[:-1] != su[1:]
    ends = np.minimum.accumulate(
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

    local = 0.5 * (tie_starts + tie_ends) - starts
    denom = np.maximum(ends - starts, 1)
    ranked = local / denom

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)

age = np.max(np.asarray(train.date, dtype=np.int32)) - np.asarray(
    train.date, dtype=np.int32
)
weights = np.power(0.5, age.astype(np.float64) / 5.0).astype(np.float32)
weights /= np.mean(weights)
global_rate = float(np.sum(weights * y_train) / np.sum(weights))
global_logit = float(safe_logit(global_rate))

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
    "register_days_range",
    "onehot_feat0",
    "onehot_feat12",
    "fans_user_num_range",
    "onehot_feat11",
    "onehot_feat6",
    "hour",
]

STRENGTH = {
    "video_id": 35.0,
    "author_id": 40.0,
    "user_id": 35.0,
    "tab": 140.0,
    "tag": 110.0,
    "onehot_feat3": 70.0,
    "upload_type": 140.0,
    "onehot_feat8": 80.0,
}
DEFAULT_STRENGTH = 170.0

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
    "author_id_is_profile_enter_rate",
]


def values_for(split, field):
    if field == "video_id":
        return np.asarray(split.video_id, dtype=np.int64)
    if field == "user_id":
        return np.asarray(split.user_id, dtype=np.int64)
    return np.asarray(split.X[field], dtype=np.int64)


target_tables = {}
for field in TE_FIELDS:
    v = values_for(train, field)
    cardinality = int(FEATURE_CARDINALITIES[field])
    count = np.bincount(
        v, weights=weights, minlength=cardinality
    ).astype(np.float64)
    positive = np.bincount(
        v, weights=weights * y_train, minlength=cardinality
    ).astype(np.float64)
    target_tables[field] = (count, positive)


def target_features(split, field, train_loo):
    v = values_for(split, field)
    count, positive = target_tables[field]
    vv = np.minimum(v, len(count) - 1)
    c = count[vv].copy()
    p = positive[vv].copy()
    if train_loo:
        c -= weights
        p -= weights * y_train
    strength = STRENGTH.get(field, DEFAULT_STRENGTH)
    rate = (p + strength * global_rate) / np.maximum(c + strength, 1e-9)
    encoded = safe_logit(rate) - global_logit
    reliability = c / np.maximum(c + strength, 1e-9)
    return encoded.astype(np.float32), reliability.astype(np.float32)


def all_histories(split_name):
    result = {}
    result.update(historical_features(split_name, key="video_id"))
    result.update(historical_features(split_name, key="author_id"))
    return result


hist_train = all_histories("train")
hist_valid = all_histories("valid")
hist_test = all_histories("test")


def build_matrix(split, histories, train_loo=False):
    cols = []
    for field in TE_FIELDS:
        te, reliability = target_features(split, field, train_loo)
        cols.append(te)
        if field in ("video_id", "author_id", "user_id"):
            cols.append(reliability)

    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))

    for key in HISTORY_KEYS:
        x = np.asarray(histories[key], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(x)

    hour = np.asarray(split.X["hour"], dtype=np.float32)
    cols.append(np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32))
    cols.append(np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32))

    return np.column_stack(cols).astype(np.float32, copy=False)


X_train = build_matrix(train, hist_train, train_loo=True)
X_valid = build_matrix(valid, hist_valid, train_loo=False)
X_test = build_matrix(test, hist_test, train_loo=False)

del hist_train, hist_valid, hist_test
gc.collect()

mean = np.average(X_train, axis=0, weights=weights).astype(np.float64)
variance = np.average(
    (X_train.astype(np.float64) - mean) ** 2,
    axis=0,
    weights=weights,
)
scale = np.sqrt(np.maximum(variance, 1e-5))

Z_train = np.clip((X_train - mean) / scale, -7.0, 7.0).astype(np.float32)
Z_valid = np.clip((X_valid - mean) / scale, -7.0, 7.0).astype(np.float32)
Z_test = np.clip((X_test - mean) / scale, -7.0, 7.0).astype(np.float32)

del X_train, X_valid, X_test
gc.collect()

n, d = Z_train.shape
candidate_valid = {}
candidate_test = {}

# Family 1: shrinkage full-covariance linear discriminant analysis.
class_means = []
class_covs = []
class_priors = []
for cls in (0, 1):
    mask = y_train == cls
    w = weights[mask].astype(np.float64)
    z = Z_train[mask].astype(np.float64)
    sw = np.sum(w)
    mu = np.sum(z * w[:, None], axis=0) / sw
    centered = z - mu
    cov = (centered.T @ (centered * w[:, None])) / sw
    class_means.append(mu)
    class_covs.append(cov)
    class_priors.append(sw)

pooled_cov = (
    class_covs[0] * class_priors[0] + class_covs[1] * class_priors[1]
) / (class_priors[0] + class_priors[1])
diag_cov = np.diag(np.diag(pooled_cov))
shrink_cov = 0.72 * pooled_cov + 0.28 * diag_cov
shrink_cov.flat[:: d + 1] += 0.18
delta = class_means[1] - class_means[0]
lda_beta = np.linalg.solve(shrink_cov, delta)
lda_intercept = (
    -0.5 * np.dot(class_means[1] + class_means[0], lda_beta)
    + global_logit
)
candidate_valid["shrinkage_lda"] = (
    Z_valid.astype(np.float64) @ lda_beta + lda_intercept
).astype(np.float32)
candidate_test["shrinkage_lda"] = (
    Z_test.astype(np.float64) @ lda_beta + lda_intercept
).astype(np.float32)

# Family 2: diagonal quadratic discriminant analysis.
qda_params = []
for cls in (0, 1):
    mask = y_train == cls
    w = weights[mask].astype(np.float64)
    z = Z_train[mask].astype(np.float64)
    sw = np.sum(w)
    mu = np.sum(z * w[:, None], axis=0) / sw
    var = np.sum(((z - mu) ** 2) * w[:, None], axis=0) / sw
    var = 0.72 * var + 0.28
    var = np.maximum(var, 0.12)
    prior = sw / np.sum(weights)
    qda_params.append((mu, var, prior))


def qda_predict(z):
    z = z.astype(np.float64)
    scores = []
    for mu, var, prior in qda_params:
        s = (
            -0.5 * np.sum(((z - mu) ** 2) / var, axis=1)
            -0.5 * np.sum(np.log(var))
            + np.log(prior + 1e-12)
        )
        scores.append(s)
    return (scores[1] - scores[0]).astype(np.float32)


candidate_valid["diagonal_qda"] = qda_predict(Z_valid)
candidate_test["diagonal_qda"] = qda_predict(Z_test)

# Family 3: class-conditional multiple-prototype mixture.
recent_indices = np.nonzero(np.asarray(train.date) >= 20220415)[0]
prototype_sets = []
prototype_vars = []
for cls in (0, 1):
    pool = recent_indices[y_train[recent_indices] == cls]
    sample_n = min(70000, len(pool))
    sample_idx = rng.choice(pool, size=sample_n, replace=False)
    sample = Z_train[sample_idx].astype(np.float32)
    k = 7
    centers = sample[rng.choice(len(sample), size=k, replace=False)].copy()

    for _ in range(7):
        labels = np.empty(len(sample), dtype=np.int16)
        for start in range(0, len(sample), 10000):
            block = sample[start:start + 10000]
            dist = (
                np.sum(block * block, axis=1, keepdims=True)
                + np.sum(centers * centers, axis=1)[None, :]
                - 2.0 * block @ centers.T
            )
            labels[start:start + len(block)] = np.argmin(dist, axis=1)

        for j in range(k):
            member = sample[labels == j]
            if len(member) >= 20:
                centers[j] = np.mean(member, axis=0)

    assigned_var = np.empty(k, dtype=np.float32)
    for j in range(k):
        member = sample[labels == j]
        if len(member) < 20:
            assigned_var[j] = 1.0
        else:
            assigned_var[j] = np.maximum(
                np.mean((member - centers[j]) ** 2), 0.20
            )
    prototype_sets.append(centers)
    prototype_vars.append(assigned_var)


def prototype_predict(z):
    outputs = []
    for centers, variances in zip(prototype_sets, prototype_vars):
        pieces = []
        center_norm = np.sum(centers * centers, axis=1)
        for start in range(0, len(z), 25000):
            block = z[start:start + 25000]
            dist = (
                np.sum(block * block, axis=1, keepdims=True)
                + center_norm[None, :]
                - 2.0 * block @ centers.T
            )
            score = -dist / (2.0 * d * variances[None, :])
            score -= 0.5 * d * np.log(variances[None, :])
            pieces.append(logsumexp(score, axis=1))
        outputs.append(np.concatenate(pieces))
    return (outputs[1] - outputs[0] + global_logit).astype(np.float32)


candidate_valid["prototype_mixture"] = prototype_predict(Z_valid)
candidate_test["prototype_mixture"] = prototype_predict(Z_test)

# Family 4: random-oblique projection histograms.
n_oblique = 72
n_bins = 18
projection = rng.normal(
    0.0, 1.0 / np.sqrt(d), size=(d, n_oblique)
).astype(np.float32)
projection_sample_idx = rng.choice(
    n, size=min(180000, n), replace=False
)
sample_projection = Z_train[projection_sample_idx] @ projection
quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
edges = [
    np.unique(np.quantile(sample_projection[:, j], quantiles)).astype(np.float32)
    for j in range(n_oblique)
]
counts = [np.zeros(len(e) + 1, dtype=np.float64) for e in edges]
positives = [np.zeros(len(e) + 1, dtype=np.float64) for e in edges]

for start in range(0, n, 30000):
    end = min(start + 30000, n)
    projected = Z_train[start:end] @ projection
    wb = weights[start:end]
    yb = y_train[start:end]
    for j, edge in enumerate(edges):
        bins = np.searchsorted(edge, projected[:, j], side="right")
        counts[j] += np.bincount(
            bins, weights=wb, minlength=len(edge) + 1
        )
        positives[j] += np.bincount(
            bins, weights=wb * yb, minlength=len(edge) + 1
        )

oblique_tables = []
for c, p in zip(counts, positives):
    rate = (p + 180.0 * global_rate) / (c + 180.0)
    oblique_tables.append((safe_logit(rate) - global_logit).astype(np.float32))


def oblique_predict(z):
    out = np.full(len(z), global_logit, dtype=np.float32)
    for start in range(0, len(z), 30000):
        end = min(start + 30000, len(z))
        projected = z[start:end] @ projection
        block_score = np.zeros(end - start, dtype=np.float32)
        for j, edge in enumerate(edges):
            bins = np.searchsorted(edge, projected[:, j], side="right")
            block_score += oblique_tables[j][bins]
        out[start:end] += block_score / np.sqrt(float(n_oblique))
    return out


candidate_valid["oblique_histogram"] = oblique_predict(Z_valid)
candidate_test["oblique_histogram"] = oblique_predict(Z_test)

del sample_projection, counts, positives
gc.collect()

# Family 5: ridge regression over random nonlinear conjunction rules.
rule_count = 128
rule_feature_a = rng.integers(0, d, size=rule_count)
rule_feature_b = rng.integers(0, d, size=rule_count)
rule_feature_c = rng.integers(0, d, size=rule_count)
rule_sign_a = rng.choice(np.array([-1.0, 1.0]), size=rule_count)
rule_sign_b = rng.choice(np.array([-1.0, 1.0]), size=rule_count)
rule_sign_c = rng.choice(np.array([-1.0, 1.0]), size=rule_count)

threshold_pool = Z_train[
    rng.choice(n, size=min(150000, n), replace=False)
]
rule_threshold_a = np.array([
    np.quantile(
        threshold_pool[:, rule_feature_a[j]],
        rng.uniform(0.20, 0.80),
    )
    for j in range(rule_count)
], dtype=np.float32)
rule_threshold_b = np.array([
    np.quantile(
        threshold_pool[:, rule_feature_b[j]],
        rng.uniform(0.20, 0.80),
    )
    for j in range(rule_count)
], dtype=np.float32)
rule_threshold_c = np.array([
    np.quantile(
        threshold_pool[:, rule_feature_c[j]],
        rng.uniform(0.25, 0.75),
    )
    for j in range(rule_count)
], dtype=np.float32)


def make_rules(z):
    a = (
        rule_sign_a[None, :]
        * (z[:, rule_feature_a] - rule_threshold_a[None, :])
        > 0.0
    )
    b = (
        rule_sign_b[None, :]
        * (z[:, rule_feature_b] - rule_threshold_b[None, :])
        > 0.0
    )
    c = (
        rule_sign_c[None, :]
        * (z[:, rule_feature_c] - rule_threshold_c[None, :])
        > 0.0
    )
    return (a & b & c).astype(np.float32)


fit_idx = rng.choice(n, size=min(210000, n), replace=False)
R_fit = make_rules(Z_train[fit_idx])
R_design = np.column_stack([
    np.ones(len(R_fit), dtype=np.float32),
    Z_train[fit_idx],
    R_fit,
]).astype(np.float64)
w_fit = weights[fit_idx].astype(np.float64)
target_fit = y_train[fit_idx].astype(np.float64) - global_rate
root_w = np.sqrt(w_fit)[:, None]
weighted_design = R_design * root_w
normal = weighted_design.T @ weighted_design
vector = R_design.T @ (w_fit * target_fit)
ridge_diag = np.full(normal.shape[0], 90.0, dtype=np.float64)
ridge_diag[0] = 1.0
ridge_diag[1:d + 1] = 180.0
normal.flat[:: normal.shape[0] + 1] += ridge_diag
rule_beta = np.linalg.solve(normal, vector)


def rule_predict(z):
    out = np.empty(len(z), dtype=np.float32)
    for start in range(0, len(z), 25000):
        end = min(start + 25000, len(z))
        rules = make_rules(z[start:end])
        design = np.column_stack([
            np.ones(end - start, dtype=np.float32),
            z[start:end],
            rules,
        ])
        out[start:end] = (
            global_logit + design @ rule_beta
        ).astype(np.float32)
    return out


candidate_valid["conjunction_rules"] = rule_predict(Z_valid)
candidate_test["conjunction_rules"] = rule_predict(Z_test)

del R_fit, R_design, weighted_design, normal, threshold_pool
gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.load(inc_valid_path).astype(np.float64)
inc_test = np.load(inc_test_path).astype(np.float64)
inc_valid_rank = within_user_rank(inc_valid, valid_users)
inc_test_rank = within_user_rank(inc_test, test_users)

alphas = np.array(
    [0.0, 0.03, 0.06, 0.10, 0.14, 0.20, 0.28, 0.40, 0.55, 0.75, 1.0],
    dtype=np.float64,
)

candidate_log = {}
best_primary = -np.inf
best_name = None
best_alpha = None
best_valid_scores = None
best_test_scores = None
best_raw_valid = None

inc_metric = evaluate(valid_users, y_valid, inc_valid_rank)
candidate_log["trusted_incumbent"] = float(inc_metric["primary"])

for name in candidate_valid:
    own_valid_rank = within_user_rank(candidate_valid[name], valid_users)
    own_test_rank = within_user_rank(candidate_test[name], test_users)
    own_metric = evaluate(valid_users, y_valid, own_valid_rank)
    candidate_log[name] = float(own_metric["primary"])

    local_best = -np.inf
    local_alpha = 0.0
    local_metric = None
    local_scores = None

    for alpha in alphas:
        blended = (1.0 - alpha) * inc_valid_rank + alpha * own_valid_rank
        metric = evaluate(valid_users, y_valid, blended)
        if metric["primary"] > local_best:
            local_best = float(metric["primary"])
            local_alpha = float(alpha)
            local_metric = metric
            local_scores = blended.copy()

    candidate_log[name + "_blend"] = local_best

    if local_best > best_primary:
        best_primary = local_best
        best_name = name
        best_alpha = local_alpha
        best_valid_scores = local_scores
        best_test_scores = (
            (1.0 - local_alpha) * inc_test_rank
            + local_alpha * own_test_rank
        )
        best_raw_valid = own_valid_rank.copy()

final_metrics = evaluate(valid_users, y_valid, best_valid_scores)

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": best_name,
            "own_weight": best_alpha,
            "feature_dimension": int(d),
            "incumbent_primary": float(inc_metric["primary"]),
        },
        sort_keys=True,
    )
)

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
    if best_alpha < 1.0:
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