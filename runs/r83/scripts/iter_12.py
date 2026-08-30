import os
import time
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 27183
RANK = 40
GRAPH_DIM = 112
HALF_LIFE = 7.0

rng_global = np.random.default_rng(SEED)

CONTENT_FIELDS = ["author_id", "tag", "tab", "duration_bucket", "upload_type"]


def temporal_weights(dates):
    dates = np.asarray(dates, dtype=np.int64)
    unique = np.unique(dates)
    day = np.searchsorted(unique, dates)
    age = day.max() - day
    w = np.exp2(-age / HALF_LIFE).astype(np.float32)
    return w / max(float(w.mean()), 1e-8)


def make_combined(a, b):
    class Combined:
        pass

    c = Combined()
    c.X = {
        name: np.concatenate([
            np.asarray(a.X[name]), np.asarray(b.X[name])
        ])
        for name in a.X
    }
    c.y = np.concatenate([
        np.asarray(a.y, dtype=np.int8),
        np.asarray(b.y, dtype=np.int8)
    ])
    c.date = np.concatenate([
        np.asarray(a.date),
        np.asarray(b.date)
    ])
    return c


def build_positive_matrix(reference, field="video_id"):
    users = np.asarray(reference.X["user_id"], dtype=np.int64)
    entities = np.asarray(reference.X[field], dtype=np.int64)
    y = np.asarray(reference.y, dtype=np.int8)
    w = temporal_weights(reference.date)

    keep = y == 1
    n_users = int(FEATURE_CARDINALITIES["user_id"])
    n_entities = int(FEATURE_CARDINALITIES[field])

    mat = sparse.coo_matrix(
        (
            w[keep],
            (users[keep], entities[keep])
        ),
        shape=(n_users, n_entities),
        dtype=np.float32
    ).tocsr()
    mat.sum_duplicates()
    mat.data = np.log1p(mat.data).astype(np.float32)
    return mat


def build_content_matrix(reference):
    users = np.asarray(reference.X["user_id"], dtype=np.int64)
    y = np.asarray(reference.y, dtype=np.int8)
    w = temporal_weights(reference.date)
    keep = y == 1

    offsets = {}
    offset = 0
    all_rows = []
    all_cols = []
    all_data = []

    for field in CONTENT_FIELDS:
        card = int(FEATURE_CARDINALITIES[field])
        offsets[field] = offset
        values = np.asarray(reference.X[field], dtype=np.int64)

        all_rows.append(users[keep])
        all_cols.append(values[keep] + offset)
        all_data.append(w[keep])
        offset += card

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    data = np.concatenate(all_data).astype(np.float32)

    mat = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(int(FEATURE_CARDINALITIES["user_id"]), offset),
        dtype=np.float32
    ).tocsr()
    mat.sum_duplicates()
    mat.data = np.log1p(mat.data).astype(np.float32)
    return mat, offsets


def build_signed_video_matrix(reference):
    users = np.asarray(reference.X["user_id"], dtype=np.int64)
    videos = np.asarray(reference.X["video_id"], dtype=np.int64)
    y = np.asarray(reference.y, dtype=np.float32)
    w = temporal_weights(reference.date)

    prior = float(np.sum(w * y) / np.sum(w))
    values = w * (y - prior)

    mat = sparse.coo_matrix(
        (values, (users, videos)),
        shape=(
            int(FEATURE_CARDINALITIES["user_id"]),
            int(FEATURE_CARDINALITIES["video_id"])
        ),
        dtype=np.float32
    ).tocsr()
    mat.sum_duplicates()
    return mat


def normalize_sparse(mat, item_power=0.25):
    mat = mat.astype(np.float32, copy=True)

    row_energy = np.asarray(mat.power(2).sum(axis=1)).ravel()
    row_scale = np.power(np.maximum(row_energy, 1e-8), -0.5).astype(np.float32)

    col_energy = np.asarray(mat.power(2).sum(axis=0)).ravel()
    col_scale = np.power(
        np.maximum(col_energy, 1e-8), -0.5 * item_power
    ).astype(np.float32)

    return sparse.diags(row_scale) @ mat @ sparse.diags(col_scale)


def fit_spectral(mat, rank=RANK):
    norm = normalize_sparse(mat)
    k = min(rank, min(norm.shape) - 2)
    v0 = np.sin(
        np.arange(min(norm.shape), dtype=np.float64) * 0.017 + 0.3
    )

    try:
        u, s, vt = svds(
            norm,
            k=k,
            which="LM",
            v0=v0,
            tol=0.006,
            maxiter=180,
            return_singular_vectors=True
        )
    except Exception:
        u, s, vt = svds(
            norm,
            k=max(12, k // 2),
            which="LM",
            tol=0.015,
            maxiter=260,
            return_singular_vectors=True
        )

    order = np.argsort(s)[::-1]
    s = s[order].astype(np.float32)
    u = u[:, order].astype(np.float32)
    vt = vt[order].astype(np.float32)

    user_factors = u * np.sqrt(s)[None, :]
    entity_factors = vt.T * np.sqrt(s)[None, :]
    return user_factors, entity_factors


def spectral_video_predict(reference, evaluation):
    mat = build_positive_matrix(reference, "video_id")
    uf, vf = fit_spectral(mat)
    users = np.asarray(evaluation.X["user_id"], dtype=np.int64)
    videos = np.asarray(evaluation.X["video_id"], dtype=np.int64)
    return np.einsum(
        "ij,ij->i", uf[users], vf[videos], optimize=True
    ).astype(np.float64)


def signed_spectral_predict(reference, evaluation):
    mat = build_signed_video_matrix(reference)
    uf, vf = fit_spectral(mat)
    users = np.asarray(evaluation.X["user_id"], dtype=np.int64)
    videos = np.asarray(evaluation.X["video_id"], dtype=np.int64)
    return np.einsum(
        "ij,ij->i", uf[users], vf[videos], optimize=True
    ).astype(np.float64)


def content_spectral_predict(reference, evaluation):
    mat, offsets = build_content_matrix(reference)
    uf, ef = fit_spectral(mat)

    users = np.asarray(evaluation.X["user_id"], dtype=np.int64)
    user_vectors = uf[users]
    score = np.zeros(len(users), dtype=np.float64)

    field_weights = {
        "author_id": 1.00,
        "tag": 0.75,
        "tab": 0.55,
        "duration_bucket": 0.45,
        "upload_type": 0.35
    }

    for field in CONTENT_FIELDS:
        nodes = (
            np.asarray(evaluation.X[field], dtype=np.int64)
            + offsets[field]
        )
        contribution = np.einsum(
            "ij,ij->i", user_vectors, ef[nodes], optimize=True
        )
        score += field_weights[field] * contribution

    return score


def graph_diffusion_predict(reference, evaluation):
    mat = build_positive_matrix(reference, "video_id").astype(np.float32)
    n_users, n_items = mat.shape

    user_degree = np.asarray(mat.sum(axis=1)).ravel().astype(np.float32)
    item_degree = np.asarray(mat.sum(axis=0)).ravel().astype(np.float32)

    rs = np.random.default_rng(SEED + 503)
    projection = rs.choice(
        np.array([-1.0, 1.0], dtype=np.float32),
        size=(n_items, GRAPH_DIM)
    )
    projection /= np.sqrt(float(GRAPH_DIM))

    # A is a randomized representation of each user's consumed-item set.
    user_repr = np.asarray(mat @ projection, dtype=np.float32)
    user_repr /= np.sqrt(np.maximum(user_degree, 1.0))[:, None]

    # Propagating user representations back to items approximates the
    # degree-corrected three-hop kernel R R^T R without materializing it.
    weighted_users = user_repr / np.sqrt(
        np.maximum(user_degree, 1.0)
    )[:, None]
    item_repr = np.asarray(mat.T @ weighted_users, dtype=np.float32)
    item_repr /= np.power(
        np.maximum(item_degree, 1.0), 0.65
    )[:, None]

    norms_u = np.linalg.norm(user_repr, axis=1)
    norms_i = np.linalg.norm(item_repr, axis=1)
    user_repr /= np.maximum(norms_u, 1e-6)[:, None]
    item_repr /= np.maximum(norms_i, 1e-6)[:, None]

    users = np.asarray(evaluation.X["user_id"], dtype=np.int64)
    videos = np.asarray(evaluation.X["video_id"], dtype=np.int64)
    return np.einsum(
        "ij,ij->i",
        user_repr[users],
        item_repr[videos],
        optimize=True
    ).astype(np.float64)


def predict_family(name, reference, evaluation):
    if name == "spectral_video":
        return spectral_video_predict(reference, evaluation)
    if name == "spectral_content":
        return content_spectral_predict(reference, evaluation)
    if name == "signed_residual_svd":
        return signed_spectral_predict(reference, evaluation)
    if name == "graph_3hop":
        return graph_diffusion_predict(reference, evaluation)
    raise ValueError(name)


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    std = float(np.std(x))
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / std


def within_user_rank(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((np.arange(n), scores, users))
    sorted_users = users[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    positions = np.arange(n) - np.repeat(starts, lengths)
    denom = np.repeat(np.maximum(lengths - 1, 1), lengths)
    ranked = positions.astype(np.float64) / denom

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


train = load("train")
valid = load("valid")
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

if not os.path.exists(inc_valid_path):
    raise FileNotFoundError("Trusted incumbent validation scores unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if len(inc_valid) != len(valid_y):
    raise ValueError("Incumbent validation score length mismatch")

families = [
    "spectral_video",
    "spectral_content",
    "signed_residual_svd",
    "graph_3hop"
]

raw_predictions = {}
candidate_predictions = {}
candidate_recipe = {}
candidate_scores = {}

candidate_predictions["incumbent"] = inc_valid
candidate_recipe["incumbent"] = ("incumbent", "raw", 1.0)
candidate_scores["incumbent"] = float(
    evaluate(valid_users, valid_y, inc_valid)["primary"]
)

inc_z = zscore(inc_valid)
inc_rank = within_user_rank(valid_users, inc_valid)

for family in families:
    scores = predict_family(family, train, valid)
    raw_predictions[family] = scores

    raw_name = family + "_raw"
    candidate_predictions[raw_name] = scores
    candidate_recipe[raw_name] = (family, "raw", 0.0)
    candidate_scores[raw_name] = float(
        evaluate(valid_users, valid_y, scores)["primary"]
    )

    score_z = zscore(scores)
    score_rank = within_user_rank(valid_users, scores)

    for incumbent_weight in (0.25, 0.50, 0.75):
        z_name = family + "_z_inc%.2f" % incumbent_weight
        z_blend = (
            incumbent_weight * inc_z
            + (1.0 - incumbent_weight) * score_z
        )
        candidate_predictions[z_name] = z_blend
        candidate_recipe[z_name] = (
            family, "z", incumbent_weight
        )
        candidate_scores[z_name] = float(
            evaluate(valid_users, valid_y, z_blend)["primary"]
        )

        rank_name = family + "_rank_inc%.2f" % incumbent_weight
        rank_blend = (
            incumbent_weight * inc_rank
            + (1.0 - incumbent_weight) * score_rank
        )
        candidate_predictions[rank_name] = rank_blend
        candidate_recipe[rank_name] = (
            family, "rank", incumbent_weight
        )
        candidate_scores[rank_name] = float(
            evaluate(valid_users, valid_y, rank_blend)["primary"]
        )

best_name = max(candidate_scores, key=candidate_scores.get)
valid_scores = candidate_predictions[best_name]
best_family, best_mode, best_inc_weight = candidate_recipe[best_name]
metrics = evaluate(valid_users, valid_y, valid_scores)

correlations = {}
for family, pred in raw_predictions.items():
    if np.std(pred) > 1e-12:
        correlations[family] = float(np.corrcoef(pred, inc_valid)[0, 1])
    else:
        correlations[family] = 0.0

print("FINDINGS " + json.dumps({
    "best": best_name,
    "raw_incumbent_correlations": correlations
}, sort_keys=True))
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64)
    )

test = load("test")

if best_family == "incumbent":
    if not os.path.exists(inc_test_path):
        raise FileNotFoundError("Trusted incumbent test scores unavailable")
    test_scores = np.asarray(np.load(inc_test_path), dtype=np.float64)
else:
    combined = make_combined(train, valid)
    new_test = predict_family(best_family, combined, test)

    if best_mode == "raw":
        test_scores = new_test
    else:
        if not os.path.exists(inc_test_path):
            raise FileNotFoundError("Trusted incumbent test scores unavailable")
        inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

        if best_mode == "z":
            test_scores = (
                best_inc_weight * zscore(inc_test)
                + (1.0 - best_inc_weight) * zscore(new_test)
            )
        elif best_mode == "rank":
            test_users = np.asarray(test.user_id)
            test_scores = (
                best_inc_weight
                * within_user_rank(test_users, inc_test)
                + (1.0 - best_inc_weight)
                * within_user_rank(test_users, new_test)
            )
        else:
            raise ValueError(best_mode)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed)
}))