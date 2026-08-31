import os
import time
import json
import numpy as np
import scipy.sparse as sp

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.int8)
yva = np.asarray(valid.y, dtype=np.int8)

utr = np.asarray(train.user_id, dtype=np.int64)
uva = np.asarray(valid.user_id, dtype=np.int64)
ute = np.asarray(test.user_id, dtype=np.int64)

vtr = np.asarray(train.video_id, dtype=np.int64)
vva = np.asarray(valid.video_id, dtype=np.int64)
vte = np.asarray(test.video_id, dtype=np.int64)

n_users = int(FEATURE_CARDINALITIES["user_id"])
n_videos = int(FEATURE_CARDINALITIES["video_id"])

# Recent positive interactions receive more weight when constructing both the
# item graph and user profiles. All weights and graph statistics use train only.
train_dates = np.asarray(train.date, dtype=np.int64)
max_date = int(train_dates.max())
age = (max_date - train_dates).astype(np.float32)
recency = np.exp2(-age / 6.0).astype(np.float32)

positive = ytr == 1
pos_users = utr[positive]
pos_videos = vtr[positive]
pos_weights = recency[positive]

# Binary positive-history matrix.
R_binary = sp.coo_matrix(
    (
        np.ones(len(pos_users), dtype=np.float32),
        (pos_users, pos_videos),
    ),
    shape=(n_users, n_videos),
).tocsr()
R_binary.sum_duplicates()
R_binary.data[:] = 1.0
R_binary.eliminate_zeros()
R_binary.sort_indices()

# Recency-weighted positive-history matrix. Repeated user-video interactions
# legitimately add evidence but are sublinearly compressed.
R_recent = sp.coo_matrix(
    (pos_weights, (pos_users, pos_videos)),
    shape=(n_users, n_videos),
).tocsr()
R_recent.sum_duplicates()
R_recent.data[:] = np.log1p(R_recent.data)
R_recent.eliminate_zeros()
R_recent.sort_indices()

history_count = np.diff(R_binary.indptr).astype(np.int32)


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    row = np.arange(len(scores), dtype=np.int64)
    order = np.lexsort((row, scores, users))
    sorted_users = users[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], len(scores)]
    lengths = ends - starts
    position = (
        np.arange(len(scores), dtype=np.float64)
        - np.repeat(starts, lengths)
    )
    denominator = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    ranked = np.empty(len(scores), dtype=np.float64)
    ranked[order] = position / denominator
    return ranked


def sparse_history_score(users, candidates, history, similarity):
    """Sum similarity(candidate, each positive-history item), vectorized."""
    users = np.asarray(users, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)

    counts = history.indptr[users + 1] - history.indptr[users]
    total = int(counts.sum())
    result = np.zeros(len(users), dtype=np.float32)
    if total == 0:
        return result

    eval_row = np.repeat(np.arange(len(users), dtype=np.int64), counts)
    group_starts = np.repeat(
        np.cumsum(counts, dtype=np.int64) - counts, counts
    )
    within = np.arange(total, dtype=np.int64) - group_starts
    history_locations = history.indptr[users[eval_row]] + within

    history_items = history.indices[history_locations]
    history_weights = history.data[history_locations]

    pair_values = np.asarray(
        similarity[candidates[eval_row], history_items]
    ).reshape(-1)
    pair_values = pair_values.astype(np.float32, copy=False)
    pair_values *= history_weights

    result = np.bincount(
        eval_row, weights=pair_values, minlength=len(users)
    ).astype(np.float32)

    # Avoid rewarding users merely for having longer logged histories.
    mass = np.bincount(
        eval_row, weights=history_weights, minlength=len(users)
    ).astype(np.float32)
    result /= np.sqrt(np.maximum(mass, 1.0))
    return result


# -------------------------------------------------------------------------
# Family 1: memory-based item-item collaborative filtering.
#
# Co-positive item association captures nonlinear taste neighborhoods without
# forcing them through a low-rank factorization. Cosine normalization prevents
# globally popular videos from dominating every user's score.
# -------------------------------------------------------------------------
item_graph = (R_recent.T @ R_recent).tocsr()
item_graph.setdiag(0.0)
item_graph.eliminate_zeros()
item_graph.sort_indices()

item_norm = np.sqrt(
    np.asarray(R_recent.power(2).sum(axis=0)).reshape(-1)
).astype(np.float32)
graph_rows = np.repeat(
    np.arange(n_videos, dtype=np.int64), np.diff(item_graph.indptr)
)
graph_cols = item_graph.indices
denom = item_norm[graph_rows] * item_norm[graph_cols] + 1.0e-6
item_graph.data = (item_graph.data / denom).astype(np.float32)

# Suppress extremely weak chance co-occurrences while retaining rare but
# coherent neighborhoods.
item_graph.data[item_graph.data < 0.006] = 0.0
item_graph.eliminate_zeros()
item_graph.sort_indices()

cf_valid = sparse_history_score(uva, vva, R_recent, item_graph)
cf_test = sparse_history_score(ute, vte, R_recent, item_graph)


# -------------------------------------------------------------------------
# Family 2: TF-IDF content-profile retrieval.
#
# Each video is represented by its stable categorical content tokens. A user
# profile is the recency-weighted sum of positively watched video tokens.
# Cosine similarity ranks candidates matching rare profile attributes and can
# generalize when exact item-item co-occurrence is sparse.
# -------------------------------------------------------------------------
PROFILE_FIELDS = [
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "video_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

offsets = {}
token_count = 0
for field in PROFILE_FIELDS:
    offsets[field] = token_count
    token_count += int(FEATURE_CARDINALITIES[field])

video_parts = []
token_parts = []
for field in PROFILE_FIELDS:
    video_parts.append(vtr)
    token_parts.append(
        offsets[field] + np.asarray(train.X[field], dtype=np.int64)
    )

B = sp.coo_matrix(
    (
        np.ones(sum(len(x) for x in video_parts), dtype=np.float32),
        (np.concatenate(video_parts), np.concatenate(token_parts)),
    ),
    shape=(n_videos, token_count),
).tocsr()
B.sum_duplicates()
B.data[:] = 1.0
B.eliminate_zeros()
B.sort_indices()

token_df = np.asarray((B != 0).sum(axis=0)).reshape(-1).astype(np.float32)
idf = np.log((1.0 + n_videos) / (1.0 + token_df)) + 1.0
B.data *= idf[B.indices]

user_profile = (R_recent @ B).tocsr()
user_profile.sort_indices()

profile_norm = np.sqrt(
    np.asarray(user_profile.power(2).sum(axis=1)).reshape(-1)
).astype(np.float32)
video_profile_norm = np.sqrt(
    np.asarray(B.power(2).sum(axis=1)).reshape(-1)
).astype(np.float32)


def content_profile_score(users, candidates):
    users = np.asarray(users, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)

    counts = B.indptr[candidates + 1] - B.indptr[candidates]
    total = int(counts.sum())
    result = np.zeros(len(users), dtype=np.float32)
    if total == 0:
        return result

    eval_row = np.repeat(np.arange(len(users), dtype=np.int64), counts)
    group_starts = np.repeat(
        np.cumsum(counts, dtype=np.int64) - counts, counts
    )
    within = np.arange(total, dtype=np.int64) - group_starts
    locations = B.indptr[candidates[eval_row]] + within

    tokens = B.indices[locations]
    candidate_token_weights = B.data[locations]

    profile_values = np.asarray(
        user_profile[users[eval_row], tokens]
    ).reshape(-1).astype(np.float32, copy=False)
    contributions = profile_values * candidate_token_weights

    result = np.bincount(
        eval_row, weights=contributions, minlength=len(users)
    ).astype(np.float32)
    result /= (
        profile_norm[users] * video_profile_norm[candidates] + 1.0e-6
    )
    return result


profile_valid = content_profile_score(uva, vva)
profile_test = content_profile_score(ute, vte)


# -------------------------------------------------------------------------
# Rank aggregation and history-size gating.
# -------------------------------------------------------------------------
shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(shared, "incumbent_test_scores.npy")),
    dtype=np.float64,
)

inc_rank_valid = within_user_rank(uva, inc_valid)
inc_rank_test = within_user_rank(ute, inc_test)

cf_rank_valid = within_user_rank(uva, cf_valid)
cf_rank_test = within_user_rank(ute, cf_test)
profile_rank_valid = within_user_rank(uva, profile_valid)
profile_rank_test = within_user_rank(ute, profile_test)

# The hybrid is a third prediction rule: graph evidence supplies exact
# collaborative neighborhoods while content profiles supply attribute-level
# generalization.
hybrid_rank_valid = 0.60 * cf_rank_valid + 0.40 * profile_rank_valid
hybrid_rank_test = 0.60 * cf_rank_test + 0.40 * profile_rank_test

families_valid = {
    "item_knn": cf_rank_valid,
    "content_profile": profile_rank_valid,
    "graph_profile_hybrid": hybrid_rank_valid,
}
families_test = {
    "item_knn": cf_rank_test,
    "content_profile": profile_rank_test,
    "graph_profile_hybrid": hybrid_rank_test,
}

valid_hist = history_count[uva]
test_hist = history_count[ute]

candidate_scores = {}
best_primary = -np.inf
best_metrics = None
best_valid = None
best_test = None
best_raw = None
best_name = None

alphas = [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0]

for family_name, own_valid in families_valid.items():
    own_test = families_test[family_name]

    standalone = evaluate(uva, yva, own_valid)
    candidate_scores[family_name + "_standalone"] = float(
        standalone["primary"]
    )

    # Uniform blends provide a direct complementarity check.
    for alpha in alphas:
        va_score = (1.0 - alpha) * inc_rank_valid + alpha * own_valid
        te_score = (1.0 - alpha) * inc_rank_test + alpha * own_test
        metrics = evaluate(uva, yva, va_score)
        primary = float(metrics["primary"])
        name = "%s_uniform_%.2f" % (family_name, alpha)
        candidate_scores[name] = primary

        if primary > best_primary:
            best_primary = primary
            best_metrics = metrics
            best_valid = va_score.copy()
            best_test = te_score.copy()
            best_raw = own_valid.copy()
            best_name = name

    # Sparse-history users have too little signal for neighborhood models.
    # Search a two-bin gate while keeping the same learned rule for test.
    cold_valid = valid_hist <= 3
    cold_test = test_hist <= 3

    for alpha_cold in [0.0, 0.15, 0.30]:
        for alpha_warm in [0.30, 0.45, 0.60, 0.80, 1.0]:
            va_alpha = np.where(
                cold_valid, alpha_cold, alpha_warm
            ).astype(np.float64)
            te_alpha = np.where(
                cold_test, alpha_cold, alpha_warm
            ).astype(np.float64)

            va_score = (
                (1.0 - va_alpha) * inc_rank_valid
                + va_alpha * own_valid
            )
            te_score = (
                (1.0 - te_alpha) * inc_rank_test
                + te_alpha * own_test
            )

            metrics = evaluate(uva, yva, va_score)
            primary = float(metrics["primary"])
            name = "%s_gate_%.2f_%.2f" % (
                family_name, alpha_cold, alpha_warm
            )
            candidate_scores[name] = primary

            if primary > best_primary:
                best_primary = primary
                best_metrics = metrics
                best_valid = va_score.copy()
                best_test = te_score.copy()
                best_raw = own_valid.copy()
                best_name = name

finding = {
    "winner": best_name,
    "item_graph_edges": int(item_graph.nnz),
    "valid_no_positive_history_fraction": float(
        np.mean(valid_hist == 0)
    ),
    "valid_history_median": float(np.median(valid_hist)),
    "cf_nonzero_fraction": float(np.mean(cf_valid != 0)),
    "profile_nonzero_fraction": float(np.mean(profile_valid != 0)),
}
print("FINDINGS " + json.dumps(finding, sort_keys=True))
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw, dtype=np.float64),
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
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)