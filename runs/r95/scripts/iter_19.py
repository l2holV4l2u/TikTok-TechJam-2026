import os
import time
import json
import numpy as np
from scipy.special import expit
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
np.random.seed(2026)

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.float64)
yva = np.asarray(valid.y, dtype=np.int8)
uva = np.asarray(valid.user_id, dtype=np.int64)
ute = np.asarray(test.user_id, dtype=np.int64)

# Recency weighting is applied to the main graph estimators, not merely to a
# side feature. The evaluation interval immediately follows the train split.
dates = np.asarray(train.date, dtype=np.int64)
unique_dates = np.unique(dates)
date_position = np.searchsorted(unique_dates, dates)
age = (len(unique_dates) - 1 - date_position).astype(np.float64)
row_weight = np.exp2(-age / 4.0)
row_weight /= row_weight.mean()

GAM_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
]

xtr_fields = {
    f: np.asarray(train.X[f], dtype=np.int64) for f in GAM_FIELDS
}


def weighted_logit_mean(labels, weights):
    rate = float(np.sum(weights * labels) / np.sum(weights))
    rate = np.clip(rate, 1.0e-5, 1.0 - 1.0e-5)
    return float(np.log(rate / (1.0 - rate)))


# Family 1: a recency-weighted discriminative graph-potential model. Each
# categorical value is a graph node and coordinate Newton updates estimate its
# additive relevance potential. User_id is deliberately omitted because a
# user-only intercept cannot alter within-user ranking.
global_logit = weighted_logit_mean(ytr, row_weight)
train_eta = np.full(len(ytr), global_logit, dtype=np.float64)
gam_effects = {
    f: np.zeros(int(FEATURE_CARDINALITIES[f]), dtype=np.float64)
    for f in GAM_FIELDS
}

for sweep in range(6):
    total_abs_update = 0.0
    for field in GAM_FIELDS:
        ids = xtr_fields[field]
        p = expit(np.clip(train_eta, -18.0, 18.0))
        gradient = np.bincount(
            ids,
            weights=row_weight * (ytr - p),
            minlength=len(gam_effects[field]),
        )
        hessian = np.bincount(
            ids,
            weights=row_weight * p * (1.0 - p),
            minlength=len(gam_effects[field]),
        )

        # Stronger shrinkage for sparse identity nodes and lighter shrinkage
        # for stable content/context nodes.
        if field in ("video_id", "author_id"):
            ridge = 24.0
        elif field in ("onehot_feat3", "onehot_feat8"):
            ridge = 16.0
        else:
            ridge = 8.0

        delta = 0.70 * gradient / (hessian + ridge)
        delta = np.clip(delta, -0.60, 0.60)
        delta[0] = 0.0
        gam_effects[field] += delta
        train_eta += delta[ids]
        total_abs_update += float(np.mean(np.abs(delta)))

    loss = np.average(
        np.logaddexp(0.0, train_eta) - ytr * train_eta,
        weights=row_weight,
    )
    print(
        "FINDINGS graph_gam_sweep=%d loss=%.6f mean_abs_node_update=%.6f"
        % (sweep + 1, loss, total_abs_update / len(GAM_FIELDS))
    )


def gam_predict(sample):
    n = len(np.asarray(sample.user_id))
    score = np.full(n, global_logit, dtype=np.float64)
    for field in GAM_FIELDS:
        ids = np.asarray(sample.X[field], dtype=np.int64)
        score += gam_effects[field][ids]
    return score


gam_valid = gam_predict(valid)
gam_test = gam_predict(test)

# Compute residuals against item/content graph potentials. Personalized graph
# families below therefore model preference deviations rather than duplicating
# global item popularity.
train_probability = expit(np.clip(train_eta, -18.0, 18.0))
residual = ytr - train_probability

tr_user = np.asarray(train.user_id, dtype=np.int64)
tr_video = np.asarray(train.video_id, dtype=np.int64)
tr_author = np.asarray(train.X["author_id"], dtype=np.int64)
tr_tag = np.asarray(train.X["tag"], dtype=np.int64)

va_user = np.asarray(valid.user_id, dtype=np.int64)
va_video = np.asarray(valid.video_id, dtype=np.int64)
va_author = np.asarray(valid.X["author_id"], dtype=np.int64)
va_tag = np.asarray(valid.X["tag"], dtype=np.int64)

te_user = np.asarray(test.user_id, dtype=np.int64)
te_video = np.asarray(test.video_id, dtype=np.int64)
te_author = np.asarray(test.X["author_id"], dtype=np.int64)
te_tag = np.asarray(test.X["tag"], dtype=np.int64)


def fit_edge_memory(left, right, right_cardinality, shrinkage):
    key = (
        np.asarray(left, dtype=np.int64) * np.int64(right_cardinality)
        + np.asarray(right, dtype=np.int64)
    )
    unique_key, inverse = np.unique(key, return_inverse=True)
    weighted_count = np.bincount(
        inverse, weights=row_weight, minlength=len(unique_key)
    )
    weighted_residual = np.bincount(
        inverse,
        weights=row_weight * residual,
        minlength=len(unique_key),
    )
    value = weighted_residual / (weighted_count + float(shrinkage))
    return unique_key, value, weighted_count


def lookup_edge_memory(
    left, right, right_cardinality, unique_key, values
):
    query = (
        np.asarray(left, dtype=np.int64) * np.int64(right_cardinality)
        + np.asarray(right, dtype=np.int64)
    )
    position = np.searchsorted(unique_key, query)
    output = np.zeros(len(query), dtype=np.float64)
    inside = position < len(unique_key)
    matching = np.zeros(len(query), dtype=bool)
    matching[inside] = unique_key[position[inside]] == query[inside]
    output[matching] = values[position[matching]]
    return output, matching


# Family 2: exact graph-edge memory. This is structurally distinct from node
# potentials: predictions depend on whether a user previously interacted with
# this precise video, author, or tag. Bayesian shrinkage prevents one-event
# paths from dominating.
video_card = int(FEATURE_CARDINALITIES["video_id"])
author_card = int(FEATURE_CARDINALITIES["author_id"])
tag_card = int(FEATURE_CARDINALITIES["tag"])

uv_key, uv_value, uv_count = fit_edge_memory(
    tr_user, tr_video, video_card, shrinkage=10.0
)
ua_key, ua_value, ua_count = fit_edge_memory(
    tr_user, tr_author, author_card, shrinkage=14.0
)
ut_key, ut_value, ut_count = fit_edge_memory(
    tr_user, tr_tag, tag_card, shrinkage=22.0
)

uv_va, uv_seen_va = lookup_edge_memory(
    va_user, va_video, video_card, uv_key, uv_value
)
ua_va, ua_seen_va = lookup_edge_memory(
    va_user, va_author, author_card, ua_key, ua_value
)
ut_va, ut_seen_va = lookup_edge_memory(
    va_user, va_tag, tag_card, ut_key, ut_value
)

uv_te, _ = lookup_edge_memory(
    te_user, te_video, video_card, uv_key, uv_value
)
ua_te, _ = lookup_edge_memory(
    te_user, te_author, author_card, ua_key, ua_value
)
ut_te, _ = lookup_edge_memory(
    te_user, te_tag, tag_card, ut_key, ut_value
)

edge_valid = gam_valid + 1.20 * uv_va + 1.35 * ua_va + 0.80 * ut_va
edge_test = gam_test + 1.20 * uv_te + 1.35 * ua_te + 0.80 * ut_te

print(
    "FINDINGS valid_edge_coverage user_video=%.4f user_author=%.4f user_tag=%.4f"
    % (
        float(np.mean(uv_seen_va)),
        float(np.mean(ua_seen_va)),
        float(np.mean(ut_seen_va)),
    )
)

# Family 3: low-rank propagation over the weighted user-author residual graph.
# Unlike exact edge memory, SVD connects unseen user-author pairs through
# shared latent neighborhoods of authors and users.
user_card = int(FEATURE_CARDINALITIES["user_id"])

weighted_residual = row_weight * residual
weighted_count = row_weight

residual_graph = sparse.coo_matrix(
    (weighted_residual, (tr_user, tr_author)),
    shape=(user_card, author_card),
).tocsr()
count_graph = sparse.coo_matrix(
    (weighted_count, (tr_user, tr_author)),
    shape=(user_card, author_card),
).tocsr()

# Normalize by graph degree to keep prolific users and authors from controlling
# the singular vectors.
user_degree = np.asarray(count_graph.sum(axis=1)).ravel()
author_degree = np.asarray(count_graph.sum(axis=0)).ravel()
user_scale = 1.0 / np.sqrt(np.maximum(user_degree, 1.0))
author_scale = 1.0 / np.sqrt(np.maximum(author_degree, 1.0))

normalized_graph = sparse.diags(user_scale).dot(
    residual_graph
).dot(sparse.diags(author_scale))

rank = 20
try:
    left_vec, singular_value, right_vec_t = svds(
        normalized_graph,
        k=rank,
        which="LM",
        random_state=2026,
        tol=2.0e-3,
        maxiter=500,
    )
    order = np.argsort(singular_value)[::-1]
    singular_value = singular_value[order]
    left_vec = left_vec[:, order]
    right_vec_t = right_vec_t[order, :]

    # Split singular values symmetrically into graph embeddings.
    root_s = np.sqrt(np.maximum(singular_value, 0.0))
    user_embedding = left_vec * root_s[None, :]
    author_embedding = right_vec_t.T * root_s[None, :]

    def spectral_affinity(users, authors):
        return np.sum(
            user_embedding[np.asarray(users, dtype=np.int64)]
            * author_embedding[np.asarray(authors, dtype=np.int64)],
            axis=1,
        )

    spectral_va = spectral_affinity(va_user, va_author)
    spectral_te = spectral_affinity(te_user, te_author)

    # Robust scale matching lets the latent path contribute comparably to an
    # ordinary logit without relying on validation labels.
    train_probe = spectral_affinity(tr_user[:200000], tr_author[:200000])
    probe_scale = float(np.std(train_probe))
    spectral_multiplier = 0.20 / max(probe_scale, 1.0e-4)

    spectral_valid = gam_valid + spectral_multiplier * spectral_va
    spectral_test = gam_test + spectral_multiplier * spectral_te
    print(
        "FINDINGS spectral_top_values=%s multiplier=%.6f"
        % (
            json.dumps([float(x) for x in singular_value[:5]]),
            spectral_multiplier,
        )
    )
except Exception as exc:
    # Preserve a valid complete experiment if ARPACK fails to converge.
    spectral_valid = gam_valid.copy()
    spectral_test = gam_test.copy()
    print("FINDINGS spectral_failure=%s" % repr(exc))


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    row = np.arange(len(scores), dtype=np.int64)

    order = np.lexsort((row, scores, users))
    ordered_users = users[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_users[1:] != ordered_users[:-1]]
    )
    ends = np.r_[starts[1:], len(order)]
    lengths = ends - starts

    position = (
        np.arange(len(order), dtype=np.float64)
        - np.repeat(starts, lengths)
    )
    denominator = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    ranked = position / denominator

    result = np.empty(len(scores), dtype=np.float64)
    result[order] = ranked
    return result


gam_rank_valid = within_user_rank(uva, gam_valid)
gam_rank_test = within_user_rank(ute, gam_test)
edge_rank_valid = within_user_rank(uva, edge_valid)
edge_rank_test = within_user_rank(ute, edge_test)
spectral_rank_valid = within_user_rank(uva, spectral_valid)
spectral_rank_test = within_user_rank(ute, spectral_test)

# A heterogeneous graph ensemble combines exact high-confidence paths with
# low-rank paths capable of generalizing to unseen user-author edges.
graph_ensemble_valid = (
    0.45 * edge_rank_valid + 0.55 * spectral_rank_valid
)
graph_ensemble_test = (
    0.45 * edge_rank_test + 0.55 * spectral_rank_test
)

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.asarray(
    np.load(os.path.join(shared_dir, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(shared_dir, "incumbent_test_scores.npy")),
    dtype=np.float64,
)
inc_rank_valid = within_user_rank(uva, inc_valid)
inc_rank_test = within_user_rank(ute, inc_test)

families_valid = {
    "graph_node_potentials": gam_rank_valid,
    "exact_user_entity_edges": edge_rank_valid,
    "spectral_user_author_graph": spectral_rank_valid,
    "heterogeneous_graph_ensemble": graph_ensemble_valid,
}
families_test = {
    "graph_node_potentials": gam_rank_test,
    "exact_user_entity_edges": edge_rank_test,
    "spectral_user_author_graph": spectral_rank_test,
    "heterogeneous_graph_ensemble": graph_ensemble_test,
}

candidate_scores = {}
best_primary = -np.inf
best_metrics = None
best_valid = None
best_test = None
best_raw = None
best_name = None

# The trusted-incumbent interface permits selecting a blend weight on public
# validation and applying that identical weight to test.
alphas = [0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]

for family_name, own_valid in families_valid.items():
    own_test = families_test[family_name]
    standalone = evaluate(uva, yva, own_valid)
    candidate_scores[family_name + "_standalone"] = float(
        standalone["primary"]
    )

    for alpha in alphas:
        blended_valid = (
            (1.0 - alpha) * inc_rank_valid + alpha * own_valid
        )
        blended_test = (
            (1.0 - alpha) * inc_rank_test + alpha * own_test
        )
        metrics = evaluate(uva, yva, blended_valid)
        primary = float(metrics["primary"])
        name = "%s_blend_%.2f" % (family_name, alpha)
        candidate_scores[name] = primary

        if primary > best_primary:
            best_primary = primary
            best_metrics = metrics
            best_valid = blended_valid.copy()
            best_test = blended_test.copy()
            best_raw = own_valid.copy()
            best_name = name

# Measure whether exact graph paths particularly help users with substantial
# train history, without using this observation to fit another model.
train_user_rows = np.bincount(tr_user, minlength=user_card)
valid_history = train_user_rows[va_user]
for lower, upper, label in [
    (0, 5, "0_4"),
    (5, 20, "5_19"),
    (20, 60, "20_59"),
    (60, 10**9, "60_plus"),
]:
    mask = (valid_history >= lower) & (valid_history < upper)
    if np.sum(mask) > 100:
        base_seg = evaluate(
            uva[mask], yva[mask], gam_rank_valid[mask]
        )
        edge_seg = evaluate(
            uva[mask], yva[mask], edge_rank_valid[mask]
        )
        print(
            "FINDINGS history_bucket=%s rows=%d gam_primary=%.6f edge_primary=%.6f"
            % (
                label,
                int(np.sum(mask)),
                float(base_seg["primary"]),
                float(edge_seg["primary"]),
            )
        )

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS selected=%s primary=%.6f"
    % (best_name, best_primary)
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
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
    + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)