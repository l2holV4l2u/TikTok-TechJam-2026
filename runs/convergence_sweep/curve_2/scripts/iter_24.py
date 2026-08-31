import os
import time
import json
import gc
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

START = time.time()
rng = np.random.default_rng(20260831)

train = load("train")
valid = load("valid")
test = load("test")

y = np.asarray(train.y, dtype=np.float64)
yv = np.asarray(valid.y, dtype=np.int8)

ntr = len(y)
nva = len(valid.user_id)
nte = len(test.user_id)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)


def per_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]

    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = sorted_users[1:] != sorted_users[:-1]

    starts = np.where(starts_flag, np.arange(n), 0)
    starts = np.maximum.accumulate(starts)
    within = np.arange(n, dtype=np.float64) - starts

    group_starts = np.flatnonzero(starts_flag)
    group_ends = np.r_[group_starts[1:], n]
    sizes = np.repeat(group_ends - group_starts,
                      group_ends - group_starts).astype(np.float64)

    ranked = np.where(sizes > 1, within / (sizes - 1.0), 0.5)
    out = np.empty(n, dtype=np.float64)
    out[order] = ranked
    return out


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


# Recency is determined solely from training dates. A moderate half-life avoids
# throwing away the high-support early training days while tracking the
# declining late-window positive rate.
dates = np.asarray(train.date, dtype=np.int32)
age = dates.max() - dates
recency_weight = np.exp(-np.log(2.0) * age.astype(np.float64) / 8.0)
recency_weight /= recency_weight.mean()
global_rate = float(np.sum(recency_weight * y) / recency_weight.sum())

categorical_fields = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "video_type",
    "hour",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat11",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_bucket",
]

numeric_fields = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

# Build leakage-controlled target evidence. Train rows use leave-one-out
# evidence; validation and test use mappings formed from train only.
cat_tr = []
cat_va = []
cat_te = []
cat_counts = {}

for field in categorical_fields:
    ids_tr = np.asarray(train.X[field], dtype=np.int64)
    ids_va = np.asarray(valid.X[field], dtype=np.int64)
    ids_te = np.asarray(test.X[field], dtype=np.int64)
    card = int(FEATURE_CARDINALITIES[field])

    count = np.bincount(
        ids_tr, weights=recency_weight, minlength=card
    ).astype(np.float64)
    positive = np.bincount(
        ids_tr, weights=recency_weight * y, minlength=card
    ).astype(np.float64)

    prior = 16.0 if card < 100 else 38.0
    loo_count = np.maximum(count[ids_tr] - recency_weight, 0.0)
    loo_pos = np.maximum(
        positive[ids_tr] - recency_weight * y, 0.0
    )

    rate_tr = (
        loo_pos + prior * global_rate
    ) / np.maximum(loo_count + prior, 1e-12)

    full_rate = (
        positive + prior * global_rate
    ) / np.maximum(count + prior, 1e-12)

    cat_tr.append(rate_tr.astype(np.float32))
    cat_va.append(full_rate[ids_va].astype(np.float32))
    cat_te.append(full_rate[ids_te].astype(np.float32))
    cat_counts[field] = count


# The supplied histories are train-only and leave-one-out on training rows.
hist_tr = []
hist_va = []
hist_te = []

for entity in ["video_id", "author_id"]:
    htr = historical_features("train", key=entity)
    hva = historical_features("valid", key=entity)
    hte = historical_features("test", key=entity)

    common = sorted(set(htr) & set(hva) & set(hte))
    for name in common:
        hist_tr.append(
            np.nan_to_num(np.asarray(htr[name], dtype=np.float32))
        )
        hist_va.append(
            np.nan_to_num(np.asarray(hva[name], dtype=np.float32))
        )
        hist_te.append(
            np.nan_to_num(np.asarray(hte[name], dtype=np.float32))
        )


dim = len(cat_tr) + len(numeric_fields) + len(hist_tr)
Xtr = np.empty((ntr, dim), dtype=np.float32)
Xva = np.empty((nva, dim), dtype=np.float32)
Xte = np.empty((nte, dim), dtype=np.float32)

col = 0
for a, b, c in zip(cat_tr, cat_va, cat_te):
    Xtr[:, col] = a
    Xva[:, col] = b
    Xte[:, col] = c
    col += 1

for field in numeric_fields:
    tr_raw = np.asarray(train.num[field], dtype=np.float64)
    va_raw = np.asarray(valid.num[field], dtype=np.float64)
    te_raw = np.asarray(test.num[field], dtype=np.float64)

    Xtr[:, col] = np.log1p(
        np.maximum(np.nan_to_num(tr_raw), 0.0)
    ).astype(np.float32)
    Xva[:, col] = np.log1p(
        np.maximum(np.nan_to_num(va_raw), 0.0)
    ).astype(np.float32)
    Xte[:, col] = np.log1p(
        np.maximum(np.nan_to_num(te_raw), 0.0)
    ).astype(np.float32)
    col += 1

for a, b, c in zip(hist_tr, hist_va, hist_te):
    Xtr[:, col] = a
    Xva[:, col] = b
    Xte[:, col] = c
    col += 1

del cat_tr, cat_va, cat_te, hist_tr, hist_va, hist_te
gc.collect()

# Training-only weighted scaling.
weight_sum = float(recency_weight.sum())
mean = np.sum(
    Xtr.astype(np.float64) * recency_weight[:, None], axis=0
) / weight_sum
second = np.sum(
    Xtr.astype(np.float64) ** 2 * recency_weight[:, None], axis=0
) / weight_sum
scale = np.sqrt(np.maximum(second - mean * mean, 1e-5))

Xtr = np.clip((Xtr - mean) / scale, -7.0, 7.0).astype(np.float32)
Xva = np.clip((Xva - mean) / scale, -7.0, 7.0).astype(np.float32)
Xte = np.clip((Xte - mean) / scale, -7.0, 7.0).astype(np.float32)


# ---------------------------------------------------------------------
# Family 1: supervised Krylov latent regression.
#
# Rather than selecting unsupervised PCA directions, this constructs a compact
# latent subspace from repeated applications of the feature covariance to the
# feature-label covariance. It therefore retains correlated evidence
# directions specifically predictive of long_view.
# ---------------------------------------------------------------------

fit_size = min(360000, ntr)
sampling_probability = recency_weight / recency_weight.sum()
fit_idx = rng.choice(
    ntr, size=fit_size, replace=False, p=sampling_probability
)

Xs = Xtr[fit_idx].astype(np.float64)
ys = y[fit_idx]
ws = recency_weight[fit_idx].astype(np.float64)
ws /= ws.mean()

sw = np.sqrt(ws)
Xsw = Xs * sw[:, None]
ys_centered = ys - np.sum(ws * ys) / np.sum(ws)
ysw = ys_centered * sw

cov = (Xsw.T @ Xsw) / np.sum(ws)
cross = (Xsw.T @ ysw) / np.sum(ws)

basis = []
v = cross.copy()
for _ in range(min(14, dim)):
    for q in basis:
        v -= q * np.dot(q, v)
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        break
    q = v / norm
    basis.append(q)
    v = cov @ q

Q = np.column_stack(basis)
Zs = Xs @ Q

A = (Zs * ws[:, None]).T @ Zs
b = (Zs * ws[:, None]).T @ ys_centered
latent_coef = np.linalg.solve(
    A + 4.0 * np.eye(A.shape[0]), b
)
coef_krylov = Q @ latent_coef
intercept_krylov = float(np.sum(ws * ys) / np.sum(ws))

krylov_valid = Xva.astype(np.float64) @ coef_krylov + intercept_krylov
krylov_test = Xte.astype(np.float64) @ coef_krylov + intercept_krylov


# ---------------------------------------------------------------------
# Family 2: landmark radial-basis kernel ridge.
#
# Landmark responses create a nonlinear local evidence representation. Unlike
# nearest-neighbor voting, all landmark activations are jointly regularized,
# allowing overlapping relevance regimes to cooperate and suppress noisy
# local neighborhoods.
# ---------------------------------------------------------------------

kernel_fit_size = min(110000, fit_size)
kernel_idx = rng.choice(fit_size, size=kernel_fit_size, replace=False)
Xk = Xs[kernel_idx]
yk = ys[kernel_idx]
wk = ws[kernel_idx]

n_landmarks = 144
positive_local = np.flatnonzero(yk > 0.5)
negative_local = np.flatnonzero(yk < 0.5)

half = n_landmarks // 2
lm_pos = rng.choice(positive_local, size=half, replace=False)
lm_neg = rng.choice(negative_local, size=n_landmarks - half, replace=False)
landmarks = Xk[np.r_[lm_pos, lm_neg]].copy()

# Estimate a robust kernel scale from training-only landmark distances.
probe = Xk[rng.choice(len(Xk), size=min(3000, len(Xk)), replace=False)]
probe_d2 = (
    np.sum(probe * probe, axis=1)[:, None]
    + np.sum(landmarks * landmarks, axis=1)[None, :]
    - 2.0 * probe @ landmarks.T
)
median_d2 = float(np.median(np.maximum(probe_d2, 0.0)))
gamma = 1.0 / max(median_d2, 1e-3)


def rbf_features(X, centers, gamma_value):
    result = np.empty((len(X), len(centers)), dtype=np.float32)
    center_norm = np.sum(centers * centers, axis=1)
    chunk = 16000
    for start in range(0, len(X), chunk):
        end = min(start + chunk, len(X))
        z = X[start:end].astype(np.float64)
        d2 = (
            np.sum(z * z, axis=1)[:, None]
            + center_norm[None, :]
            - 2.0 * z @ centers.T
        )
        result[start:end] = np.exp(
            -gamma_value * np.maximum(d2, 0.0)
        ).astype(np.float32)
    return result


Phi = rbf_features(Xk, landmarks, gamma).astype(np.float64)
kernel_mean = np.average(Phi, axis=0, weights=wk)
Phi -= kernel_mean

target_mean = float(np.average(yk, weights=wk))
target = yk - target_mean

Aw = (Phi * wk[:, None]).T @ Phi
bw = (Phi * wk[:, None]).T @ target
kernel_coef = np.linalg.solve(
    Aw + 18.0 * np.eye(n_landmarks), bw
)


def kernel_predict(X):
    result = np.empty(len(X), dtype=np.float64)
    chunk = 18000
    center_norm = np.sum(landmarks * landmarks, axis=1)
    for start in range(0, len(X), chunk):
        end = min(start + chunk, len(X))
        z = X[start:end].astype(np.float64)
        d2 = (
            np.sum(z * z, axis=1)[:, None]
            + center_norm[None, :]
            - 2.0 * z @ landmarks.T
        )
        phi = np.exp(-gamma * np.maximum(d2, 0.0))
        result[start:end] = (
            (phi - kernel_mean) @ kernel_coef + target_mean
        )
    return result


kernel_valid = kernel_predict(Xva)
kernel_test = kernel_predict(Xte)

del Phi, Xs, Xk, Zs, cov, Q
gc.collect()


# ---------------------------------------------------------------------
# Family 3: relational graph diffusion.
#
# An item estimate is diffused toward its author, tag, upload type, and
# duration neighborhoods according to item support. This differs from a flat
# additive target encoder: high-support videos retain item evidence, while
# sparse videos inherit content-neighborhood evidence.
# ---------------------------------------------------------------------

field_index = {f: i for i, f in enumerate(categorical_fields)}

video_col = field_index["video_id"]
author_col = field_index["author_id"]
tag_col = field_index["tag"]
upload_col = field_index["upload_type"]
duration_col = field_index["duration_bucket"]

# Recover the unstandardized smoothed rates from standardized columns.
def unscale_rate(X, index):
    return (
        X[:, index].astype(np.float64) * scale[index] + mean[index]
    )


def graph_score(split, X):
    video_rate = unscale_rate(X, video_col)
    author_rate = unscale_rate(X, author_col)
    tag_rate = unscale_rate(X, tag_col)
    upload_rate = unscale_rate(X, upload_col)
    duration_rate = unscale_rate(X, duration_col)

    video_ids = np.asarray(split.X["video_id"], dtype=np.int64)
    support = cat_counts["video_id"][video_ids]
    retention = support / (support + 55.0)

    neighborhood_logit = (
        0.52 * safe_logit(author_rate)
        + 0.25 * safe_logit(tag_rate)
        + 0.13 * safe_logit(upload_rate)
        + 0.10 * safe_logit(duration_rate)
    )
    return (
        retention * safe_logit(video_rate)
        + (1.0 - retention) * neighborhood_logit
    )


graph_valid = graph_score(valid, Xva)
graph_test = graph_score(test, Xte)

del Xtr, Xva, Xte
gc.collect()


# Rank normalization puts heterogeneous families on a common within-user scale.
inc_vr = per_user_rank(valid.user_id, inc_valid)
inc_tr = per_user_rank(test.user_id, inc_test)

raw_valid = {
    "krylov": per_user_rank(valid.user_id, krylov_valid),
    "kernel": per_user_rank(valid.user_id, kernel_valid),
    "graph_diffusion": per_user_rank(valid.user_id, graph_valid),
}
raw_test = {
    "krylov": per_user_rank(test.user_id, krylov_test),
    "kernel": per_user_rank(test.user_id, kernel_test),
    "graph_diffusion": per_user_rank(test.user_id, graph_test),
}

# Add a cross-family rank aggregate. Averaging ranks makes this a genuine
# consensus predictor rather than choosing one family's score scale.
raw_valid["family_consensus"] = (
    raw_valid["krylov"]
    + raw_valid["kernel"]
    + raw_valid["graph_diffusion"]
) / 3.0
raw_test["family_consensus"] = (
    raw_test["krylov"]
    + raw_test["kernel"]
    + raw_test["graph_diffusion"]
) / 3.0

candidate_scores = {}
best_primary = -np.inf
best_valid = None
best_test = None
best_raw_valid = None
best_name = None
best_metrics = None

# Include the unchanged incumbent as a guard against validation-destructive
# additions, while explicitly testing every new family blended with it.
inc_metrics = evaluate(valid.user_id, yv, inc_vr)
candidate_scores["incumbent"] = float(inc_metrics["primary"])

if inc_metrics["primary"] > best_primary:
    best_primary = float(inc_metrics["primary"])
    best_valid = inc_vr.copy()
    best_test = inc_tr.copy()
    best_raw_valid = raw_valid["family_consensus"].copy()
    best_name = "incumbent"
    best_metrics = inc_metrics

alphas = [0.04, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 1.0]

for family in raw_valid:
    own_v = raw_valid[family]
    own_t = raw_test[family]

    standalone_metrics = evaluate(valid.user_id, yv, own_v)
    candidate_scores[family + "_standalone"] = float(
        standalone_metrics["primary"]
    )

    for alpha in alphas:
        blend_v = (1.0 - alpha) * inc_vr + alpha * own_v
        metrics = evaluate(valid.user_id, yv, blend_v)
        name = family + "_blend_" + format(alpha, ".2f")
        candidate_scores[name] = float(metrics["primary"])

        if metrics["primary"] > best_primary:
            best_primary = float(metrics["primary"])
            best_valid = blend_v.copy()
            best_test = (1.0 - alpha) * inc_tr + alpha * own_t
            best_raw_valid = own_v.copy()
            best_name = name
            best_metrics = metrics

print("FINDINGS selected=" + str(best_name))
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    # The reported result may contain the trusted incumbent.
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))