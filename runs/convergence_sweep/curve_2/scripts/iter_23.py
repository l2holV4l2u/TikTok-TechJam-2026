import os
import time
import json
import gc
import numpy as np

from scipy.spatial import cKDTree
from scipy.cluster.vq import kmeans2
from scipy.special import logsumexp

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

START = time.time()

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
    if n == 0:
        return scores.copy()

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
    group_sizes = group_ends - group_starts
    sizes = np.repeat(group_sizes, group_sizes).astype(np.float64)

    ranked = np.where(sizes > 1, within / (sizes - 1.0), 0.5)
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


# Recency weights are computed exclusively from the training split.
dates = np.asarray(train.date, dtype=np.int32)
age = dates.max() - dates
recency_weight = np.exp(-np.log(2.0) * age.astype(np.float64) / 9.0)
recency_weight /= recency_weight.mean()

global_rate = float(np.sum(recency_weight * y) / np.sum(recency_weight))

categorical_fields = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "video_type",
    "music_type",
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

# Historical features are supplied as leakage-controlled train-only statistics:
# leave-one-out on train and full-train mappings on validation/test.
hist_train_parts = []
hist_valid_parts = []
hist_test_parts = []

for entity in ["video_id", "author_id"]:
    htr = historical_features("train", key=entity)
    hva = historical_features("valid", key=entity)
    hte = historical_features("test", key=entity)

    common = sorted(set(htr.keys()) & set(hva.keys()) & set(hte.keys()))
    for name in common:
        hist_train_parts.append(np.asarray(htr[name], dtype=np.float32))
        hist_valid_parts.append(np.asarray(hva[name], dtype=np.float32))
        hist_test_parts.append(np.asarray(hte[name], dtype=np.float32))

nhist = len(hist_train_parts)
dim = len(categorical_fields) + len(numeric_fields) + nhist

Xtr = np.empty((ntr, dim), dtype=np.float32)
Xva = np.empty((nva, dim), dtype=np.float32)
Xte = np.empty((nte, dim), dtype=np.float32)

col = 0

# Leave-one-out empirical-Bayes evidence for every selected categorical field.
for field in categorical_fields:
    card = int(FEATURE_CARDINALITIES[field])
    tr_ids = np.asarray(train.X[field], dtype=np.int64)
    va_ids = np.asarray(valid.X[field], dtype=np.int64)
    te_ids = np.asarray(test.X[field], dtype=np.int64)

    count = np.bincount(
        tr_ids, weights=recency_weight, minlength=card
    ).astype(np.float64)
    positive = np.bincount(
        tr_ids, weights=recency_weight * y, minlength=card
    ).astype(np.float64)

    prior = 18.0 if card < 100 else 42.0

    loo_count = np.maximum(count[tr_ids] - recency_weight, 0.0)
    loo_positive = np.maximum(
        positive[tr_ids] - recency_weight * y, 0.0
    )

    Xtr[:, col] = (
        (loo_positive + prior * global_rate)
        / np.maximum(loo_count + prior, 1e-12)
    ).astype(np.float32)

    full_rate = (
        positive + prior * global_rate
    ) / np.maximum(count + prior, 1e-12)

    Xva[:, col] = full_rate[va_ids].astype(np.float32)
    Xte[:, col] = full_rate[te_ids].astype(np.float32)
    col += 1

# Heavy-tailed numerical attributes use training-only log scaling.
for field in numeric_fields:
    tr_raw = np.asarray(train.num[field], dtype=np.float64)
    va_raw = np.asarray(valid.num[field], dtype=np.float64)
    te_raw = np.asarray(test.num[field], dtype=np.float64)

    tr_log = np.log1p(np.maximum(np.nan_to_num(tr_raw), 0.0))
    va_log = np.log1p(np.maximum(np.nan_to_num(va_raw), 0.0))
    te_log = np.log1p(np.maximum(np.nan_to_num(te_raw), 0.0))

    Xtr[:, col] = tr_log.astype(np.float32)
    Xva[:, col] = va_log.astype(np.float32)
    Xte[:, col] = te_log.astype(np.float32)
    col += 1

for j in range(nhist):
    Xtr[:, col] = np.nan_to_num(hist_train_parts[j], nan=0.0)
    Xva[:, col] = np.nan_to_num(hist_valid_parts[j], nan=0.0)
    Xte[:, col] = np.nan_to_num(hist_test_parts[j], nan=0.0)
    col += 1

del hist_train_parts, hist_valid_parts, hist_test_parts
gc.collect()

# Training-only standardization. Constant or nearly constant evidence columns
# are harmless after the minimum scale is imposed.
mean = np.average(Xtr.astype(np.float64), axis=0, weights=recency_weight)
centered = Xtr.astype(np.float64) - mean
variance = np.average(centered * centered, axis=0, weights=recency_weight)
scale = np.sqrt(np.maximum(variance, 1e-6))

Xtr = np.clip((Xtr - mean) / scale, -7.0, 7.0).astype(np.float32)
Xva = np.clip((Xva - mean) / scale, -7.0, 7.0).astype(np.float32)
Xte = np.clip((Xte - mean) / scale, -7.0, 7.0).astype(np.float32)

del centered
gc.collect()


# ----------------------------------------------------------------------
# Family 1: regularized full-covariance quadratic discriminant.
#
# Unlike additive target statistics or a linear likelihood ratio, QDA assigns
# relevance from class-specific correlations between evidence dimensions.
# For example, a high item rate may have different significance depending on
# its author rate, duration, and support/history pattern.
# ----------------------------------------------------------------------

def weighted_class_moments(X, class_mask, weights):
    w = weights[class_mask].astype(np.float64)
    z = X[class_mask].astype(np.float64)
    wsum = float(w.sum())
    mu = np.sum(z * w[:, None], axis=0) / max(wsum, 1e-12)
    centered_local = z - mu
    cov = (centered_local * w[:, None]).T @ centered_local
    cov /= max(wsum, 1e-12)
    return mu, cov


mu0, cov0 = weighted_class_moments(Xtr, y < 0.5, recency_weight)
mu1, cov1 = weighted_class_moments(Xtr, y > 0.5, recency_weight)

# Shrink class covariance matrices toward a common diagonal-stabilized matrix.
pooled = (1.0 - global_rate) * cov0 + global_rate * cov1
diag_target = np.diag(np.diag(pooled))
shrink = 0.28
ridge = 0.12

cov0_reg = (
    (1.0 - shrink) * cov0
    + shrink * diag_target
    + ridge * np.eye(dim)
)
cov1_reg = (
    (1.0 - shrink) * cov1
    + shrink * diag_target
    + ridge * np.eye(dim)
)

sign0, logdet0 = np.linalg.slogdet(cov0_reg)
sign1, logdet1 = np.linalg.slogdet(cov1_reg)
inv0 = np.linalg.inv(cov0_reg)
inv1 = np.linalg.inv(cov1_reg)


def qda_predict(X):
    out = np.empty(len(X), dtype=np.float64)
    chunk = 40000
    prior_log_odds = np.log(
        np.clip(global_rate, 1e-6, 1 - 1e-6)
        / np.clip(1.0 - global_rate, 1e-6, 1 - 1e-6)
    )
    for start in range(0, len(X), chunk):
        end = min(start + chunk, len(X))
        z = X[start:end].astype(np.float64)
        d0 = z - mu0
        d1 = z - mu1
        m0 = np.einsum("ij,jk,ik->i", d0, inv0, d0)
        m1 = np.einsum("ij,jk,ik->i", d1, inv1, d1)
        out[start:end] = (
            prior_log_odds
            - 0.5 * (m1 + logdet1)
            + 0.5 * (m0 + logdet0)
        )
    return out


qda_valid = qda_predict(Xva)
qda_test = qda_predict(Xte)


# Shared training-only PCA projection for the two local/prototype families.
# It removes redundant target-rate dimensions while retaining correlated
# evidence geometry.
weighted_x = Xtr.astype(np.float64) * np.sqrt(recency_weight[:, None])
cov_all = (weighted_x.T @ weighted_x) / recency_weight.sum()
eigval, eigvec = np.linalg.eigh(cov_all)
pca_dim = min(12, dim)
projection = eigvec[:, np.argsort(eigval)[-pca_dim:]].astype(np.float32)

Ztr = (Xtr @ projection).astype(np.float32)
Zva = (Xva @ projection).astype(np.float32)
Zte = (Xte @ projection).astype(np.float32)

del weighted_x, cov_all, eigval, eigvec
gc.collect()


# ----------------------------------------------------------------------
# Family 2: local-neighbor regression in evidence space.
#
# This forms each score from labels of nearby training impressions rather
# than fitting a single global response surface. It can represent irregular
# local interactions and multiple relevance regimes without imposing tree,
# neural, or low-rank interaction structure.
# ----------------------------------------------------------------------

rng = np.random.default_rng(20260831)
recent_factor = np.exp(-np.log(2.0) * age.astype(np.float64) / 6.0)
sampling_prob = recent_factor / recent_factor.sum()
reference_size = min(180000, ntr)
reference_idx = rng.choice(
    ntr, size=reference_size, replace=False, p=sampling_prob
)

Zref = Ztr[reference_idx]
yref = y[reference_idx].astype(np.float32)
tree = cKDTree(Zref, leafsize=48, compact_nodes=True)


def knn_predict(Z):
    distance, neighbor = tree.query(
        Z, k=32, workers=-1, eps=0.08
    )
    local_y = yref[neighbor]
    # Adaptive scale makes weights meaningful in both dense and sparse areas.
    local_scale = np.maximum(
        np.median(distance, axis=1, keepdims=True), 1e-3
    )
    weight = np.exp(-distance / local_scale)
    prediction = (
        np.sum(weight * local_y, axis=1)
        + 6.0 * global_rate
    ) / (np.sum(weight, axis=1) + 6.0)
    return prediction.astype(np.float64)


knn_valid = knn_predict(Zva)
knn_test = knn_predict(Zte)

del tree, Zref, yref, reference_idx
gc.collect()


# ----------------------------------------------------------------------
# Family 3: class-conditional prototype mixture.
#
# Each class is represented by several modes rather than one Gaussian or a
# pointwise discriminative boundary. The log likelihood ratio rewards rows
# near frequently observed positive evidence regimes and penalizes proximity
# to negative regimes.
# ----------------------------------------------------------------------

n_proto = 12
sample_per_class = 60000
centers = []
mix_log_weights = []
mix_variances = []

for class_value in [0, 1]:
    idx_all = np.flatnonzero((y > 0.5) == bool(class_value))
    if len(idx_all) > sample_per_class:
        fit_idx = rng.choice(
            idx_all, size=sample_per_class, replace=False
        )
    else:
        fit_idx = idx_all

    sample = Ztr[fit_idx].astype(np.float64)
    init_idx = rng.choice(len(sample), size=n_proto, replace=False)
    init = sample[init_idx]

    center, labels = kmeans2(
        sample,
        init,
        iter=18,
        minit="matrix",
        check_finite=False,
    )

    # Reassign a larger class sample to obtain stable mixture masses and
    # diagonal component variances.
    estimate_size = min(180000, len(idx_all))
    estimate_idx = rng.choice(
        idx_all, size=estimate_size, replace=False
    )
    estimate = Ztr[estimate_idx].astype(np.float64)

    d2 = np.sum(
        (estimate[:, None, :] - center[None, :, :]) ** 2,
        axis=2,
    )
    assignment = np.argmin(d2, axis=1)

    counts = np.bincount(
        assignment, minlength=n_proto
    ).astype(np.float64)
    variances = np.empty((n_proto, pca_dim), dtype=np.float64)

    global_var = np.var(estimate, axis=0) + 0.20
    for k in range(n_proto):
        mask = assignment == k
        if np.sum(mask) >= 30:
            variances[k] = np.var(estimate[mask], axis=0) + 0.16
        else:
            variances[k] = global_var

    centers.append(center.astype(np.float64))
    mix_variances.append(variances)
    mix_log_weights.append(
        np.log((counts + 3.0) / (counts.sum() + 3.0 * n_proto))
    )


def mixture_log_density(Z, class_value):
    center = centers[class_value]
    variance_local = mix_variances[class_value]
    log_weight = mix_log_weights[class_value]

    result = np.empty(len(Z), dtype=np.float64)
    chunk = 20000
    for start in range(0, len(Z), chunk):
        end = min(start + chunk, len(Z))
        z = Z[start:end].astype(np.float64)
        diff = z[:, None, :] - center[None, :, :]
        component = (
            -0.5 * np.sum(
                diff * diff / variance_local[None, :, :], axis=2
            )
            -0.5 * np.sum(np.log(variance_local), axis=1)[None, :]
            + log_weight[None, :]
        )
        result[start:end] = logsumexp(component, axis=1)
    return result


mixture_valid = (
    mixture_log_density(Zva, 1)
    - mixture_log_density(Zva, 0)
    + np.log(global_rate / (1.0 - global_rate))
)
mixture_test = (
    mixture_log_density(Zte, 1)
    - mixture_log_density(Zte, 0)
    + np.log(global_rate / (1.0 - global_rate))
)

del Ztr, Zva, Zte, Xtr, Xva, Xte
gc.collect()


# Compare standalone rankings and blends with the trusted incumbent. Rank
# normalization is monotone within each user, so it retains each component's
# ranking while putting structurally different score scales on common units.
inc_valid_rank = per_user_rank(valid.user_id, inc_valid)
inc_test_rank = per_user_rank(test.user_id, inc_test)

families = {
    "full_covariance_qda": (qda_valid, qda_test),
    "local_neighbor_regression": (knn_valid, knn_test),
    "prototype_mixture": (mixture_valid, mixture_test),
}

blend_weights = [0.0, 0.04, 0.08, 0.14, 0.22, 0.35, 1.0]
candidate_scores = {}
best_primary = -np.inf
best_valid = None
best_test = None
best_raw_valid = None
best_name = None

for family_name, (raw_valid, raw_test) in families.items():
    own_valid_rank = per_user_rank(valid.user_id, raw_valid)
    own_test_rank = per_user_rank(test.user_id, raw_test)

    standalone_metrics = evaluate(
        valid.user_id, yv, own_valid_rank
    )
    candidate_scores[family_name + "_standalone"] = float(
        standalone_metrics["primary"]
    )

    family_best = -np.inf
    family_best_alpha = None

    for alpha in blend_weights:
        valid_score = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * own_valid_rank
        )
        metrics = evaluate(valid.user_id, yv, valid_score)
        primary = float(metrics["primary"])

        if primary > family_best:
            family_best = primary
            family_best_alpha = alpha

        if primary > best_primary:
            best_primary = primary
            best_name = family_name + "_blend_" + format(alpha, ".2f")
            best_valid = valid_score.copy()
            best_test = (
                (1.0 - alpha) * inc_test_rank
                + alpha * own_test_rank
            )
            best_raw_valid = own_valid_rank.copy()

    candidate_scores[
        family_name + "_best_blend"
    ] = float(family_best)
    candidate_scores[
        family_name + "_best_alpha"
    ] = float(family_best_alpha)

final_metrics = evaluate(valid.user_id, yv, best_valid)

print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": best_name,
            "evidence_dimensions": int(dim),
            "historical_dimensions": int(nhist),
            "knn_reference_rows": int(reference_size),
        },
        sort_keys=True,
    )
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
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