import os
import time
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
np.random.seed(2026)

train = load("train")
valid = load("valid")

y = np.asarray(train.y, dtype=np.float64)
train_uid = np.asarray(train.user_id, dtype=np.int64)
train_vid = np.asarray(train.video_id, dtype=np.int64)

MAX_DATE = int(np.max(train.date))
# Four-day half-life directly targets the documented temporal drift.
age = MAX_DATE - np.asarray(train.date, dtype=np.int64)
recency_w = np.exp2(-age.astype(np.float64) / 4.0)
global_rate = float(np.sum(recency_w * y) / np.sum(recency_w))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


GLOBAL_LOGIT = float(logit(global_rate))


def weighted_rate(field, cardinality, prior_strength):
    ids = np.asarray(train.X[field], dtype=np.int64)
    cnt = np.bincount(ids, weights=recency_w, minlength=cardinality)
    pos = np.bincount(ids, weights=recency_w * y, minlength=cardinality)
    return (pos + prior_strength * global_rate) / (cnt + prior_strength)


# Empirical-Bayes model: high-cardinality entities plus stationary side fields.
EB_SPECS = [
    ("video_id", 28.0, 0.42),
    ("author_id", 42.0, 0.28),
    ("tab", 100.0, 0.11),
    ("tag", 80.0, 0.07),
    ("duration_bucket", 100.0, 0.05),
    ("upload_type", 100.0, 0.035),
    ("music_type", 100.0, 0.025),
    ("hour", 120.0, 0.02),
]
eb_tables = {}
for name, strength, coef in EB_SPECS:
    eb_tables[name] = weighted_rate(
        name, int(FEATURE_CARDINALITIES[name]), strength
    )


def eb_predict(split):
    out = np.zeros(len(split.user_id), dtype=np.float64)
    for name, _, coef in EB_SPECS:
        ids = np.asarray(split.X[name], dtype=np.int64)
        table = eb_tables[name]
        safe = np.minimum(ids, len(table) - 1)
        vals = table[safe]
        vals = np.where(ids < len(table), vals, global_rate)
        out += coef * (logit(vals) - GLOBAL_LOGIT)
    return out


eb_valid = eb_predict(valid)

# A simpler item-popularity family is retained as a useful diagnostic.
video_rate = eb_tables["video_id"]
author_rate = eb_tables["author_id"]


def popularity_predict(split):
    vi = np.asarray(split.X["video_id"], dtype=np.int64)
    ai = np.asarray(split.X["author_id"], dtype=np.int64)
    vr = video_rate[np.minimum(vi, len(video_rate) - 1)]
    ar = author_rate[np.minimum(ai, len(author_rate) - 1)]
    vr = np.where(vi < len(video_rate), vr, global_rate)
    ar = np.where(ai < len(author_rate), ar, global_rate)
    return 0.62 * logit(vr) + 0.38 * logit(ar)


pop_valid = popularity_predict(valid)

# Residual implicit matrix factorization. Observed interactions contribute
# recency-weighted residuals, preventing prolific generally-positive users
# and globally popular videos from dominating every latent component.
n_users = int(FEATURE_CARDINALITIES["user_id"])
n_videos = int(FEATURE_CARDINALITIES["video_id"])
residual_values = recency_w * (y - global_rate)

interaction = sparse.coo_matrix(
    (residual_values, (train_uid, train_vid)),
    shape=(n_users, n_videos),
    dtype=np.float64,
).tocsr()
interaction.sum_duplicates()

try:
    u_svd, singular, vt_svd = svds(
        interaction,
        k=24,
        which="LM",
        tol=2e-3,
        maxiter=500,
        random_state=2026,
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order]
    u_svd = u_svd[:, order]
    vt_svd = vt_svd[order, :]
    user_latent = u_svd * np.sqrt(singular)[None, :]
    video_latent = vt_svd.T * np.sqrt(singular)[None, :]
except Exception:
    # Keep the iteration valid if ARPACK has an unusual convergence failure.
    user_latent = np.zeros((n_users, 1), dtype=np.float64)
    video_latent = np.zeros((n_videos, 1), dtype=np.float64)


def svd_predict(split):
    ui = np.asarray(split.user_id, dtype=np.int64)
    vi = np.asarray(split.video_id, dtype=np.int64)
    good = (ui >= 0) & (ui < n_users) & (vi >= 0) & (vi < n_videos)
    out = np.zeros(len(ui), dtype=np.float64)
    out[good] = np.einsum(
        "ij,ij->i", user_latent[ui[good]], video_latent[vi[good]]
    )
    return out


svd_valid = svd_predict(valid)

# First-order positive-history transition model. It uses only ordered train
# events and only a user's final positive train item at evaluation time.
order = np.lexsort(
    (
        np.arange(len(y), dtype=np.int64),
        np.asarray(train.time_ms, dtype=np.int64),
        train_uid,
    )
)
ou = train_uid[order]
ov = train_vid[order]
oy = y[order]
ow = recency_w[order]

adjacent = (ou[1:] == ou[:-1]) & (oy[:-1] > 0.5)
src = ov[:-1][adjacent]
dst = ov[1:][adjacent]
transition_values = ow[1:][adjacent] * (oy[1:][adjacent] - global_rate)

transition = sparse.coo_matrix(
    (transition_values, (src, dst)),
    shape=(n_videos, n_videos),
    dtype=np.float64,
).tocsr()
transition.sum_duplicates()

row_mass = np.asarray(
    np.abs(transition).sum(axis=1)
).ravel()
row_scale = 1.0 / np.sqrt(np.maximum(row_mass, 1.0))
transition = sparse.diags(row_scale).dot(transition).tocsr()

last_positive_item = np.zeros(n_users, dtype=np.int64)
pos_ordered = np.flatnonzero(oy > 0.5)
# Ordered assignment intentionally leaves the chronologically final item.
for start in range(0, len(pos_ordered), 200000):
    idx = pos_ordered[start:start + 200000]
    # Chunking bounds temporary memory. Repeated indices within a chunk are
    # resolved by taking the final occurrence via reversed unique indices.
    ru = ou[idx][::-1]
    rv = ov[idx][::-1]
    uniq, first = np.unique(ru, return_index=True)
    last_positive_item[uniq] = rv[first]


def markov_predict(split):
    ui = np.asarray(split.user_id, dtype=np.int64)
    vi = np.asarray(split.video_id, dtype=np.int64)
    good = (ui >= 0) & (ui < n_users) & (vi >= 0) & (vi < n_videos)
    out = np.zeros(len(ui), dtype=np.float64)
    if np.any(good):
        src_items = last_positive_item[ui[good]]
        vals = transition[src_items, vi[good]]
        out[good] = np.asarray(vals).ravel()
    # A small popularity prior resolves the many unseen transition ties.
    vi_safe = np.minimum(vi, len(video_rate) - 1)
    prior = video_rate[vi_safe]
    prior = np.where(vi < len(video_rate), prior, global_rate)
    out += 0.04 * (logit(prior) - GLOBAL_LOGIT)
    return out


markov_valid = markov_predict(valid)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)


def quantile_score(x):
    """Monotone empirical-CDF calibration, with equal values retaining ties."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    clean = np.where(finite, x, 0.0)
    sorted_x = np.sort(clean)
    left = np.searchsorted(sorted_x, clean, side="left")
    right = np.searchsorted(sorted_x, clean, side="right")
    return (left + right) / (2.0 * max(1, len(clean)))


candidate_valid = {
    "empirical_bayes": eb_valid,
    "item_author_popularity": pop_valid,
    "residual_svd": svd_valid,
    "markov_transition": markov_valid,
}

candidate_scores = {}
best_name = None
best_scores = None
best_primary = -np.inf
best_family = None
best_alpha = None

for name, scores in candidate_valid.items():
    met = evaluate(valid.user_id, valid.y, scores)
    candidate_scores[name] = float(met["primary"])
    if met["primary"] > best_primary:
        best_primary = float(met["primary"])
        best_name = name
        best_scores = scores
        best_family = name
        best_alpha = 1.0

inc_q = quantile_score(inc_valid)
alphas = (0.10, 0.20, 0.30, 0.40, 0.50, 0.65)

for name, scores in candidate_valid.items():
    own_q = quantile_score(scores)
    for alpha in alphas:
        blended = (1.0 - alpha) * inc_q + alpha * own_q
        met = evaluate(valid.user_id, valid.y, blended)
        key = "%s_blend_%.2f" % (name, alpha)
        candidate_scores[key] = float(met["primary"])
        if met["primary"] > best_primary:
            best_primary = float(met["primary"])
            best_name = key
            best_scores = blended
            best_family = name
            best_alpha = float(alpha)

final_metrics = evaluate(valid.user_id, valid.y, best_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": best_name,
            "family": best_family,
            "incumbent_weight": float(1.0 - best_alpha),
            "own_weight": float(best_alpha),
            "weighted_train_rate": global_rate,
            "svd_nnz": int(interaction.nnz),
            "transition_nnz": int(transition.nnz),
        },
        sort_keys=True,
    )
)

# Produce the corresponding test component and exact same family/blend rule.
test = load("test")
if best_family == "empirical_bayes":
    own_test = eb_predict(test)
elif best_family == "item_author_popularity":
    own_test = popularity_predict(test)
elif best_family == "residual_svd":
    own_test = svd_predict(test)
else:
    own_test = markov_predict(test)

if best_alpha < 1.0:
    inc_test = np.load(
        os.path.join(shared, "incumbent_test_scores.npy")
    ).astype(np.float64)
    test_scores = (
        (1.0 - best_alpha) * quantile_score(inc_test)
        + best_alpha * quantile_score(own_test)
    )
    own_valid = candidate_valid[best_family]
else:
    test_scores = own_test
    own_valid = candidate_valid[best_family]

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if best_alpha < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(own_valid, dtype=np.float64),
        )

elapsed = time.time() - START
result = {
    "primary": float(final_metrics["primary"]),
    "gauc": float(final_metrics["gauc"]),
    "ndcg@5": float(final_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result, separators=(", ", ": ")))