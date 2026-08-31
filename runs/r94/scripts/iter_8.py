import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 18427
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

N_USER = int(FEATURE_CARDINALITIES["user_id"])
N_VIDEO = int(FEATURE_CARDINALITIES["video_id"])
N_AUTHOR = int(FEATURE_CARDINALITIES["author_id"])
N_TAG = int(FEATURE_CARDINALITIES["tag"])

DIM = 24
EPOCHS = 4
LR = 0.012
WEIGHT_DECAY = 2.0e-6
HALF_LIFE = 4.0


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int32)
    latest = int(dates.max())
    w = np.power(
        2.0,
        (dates.astype(np.float64) - latest) / HALF_LIFE,
    )
    w /= w.mean()
    return w.astype(np.float32)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    groups = np.cumsum(starts_mask) - 1
    positions = np.arange(n, dtype=np.int64) - starts[groups]
    sizes = np.diff(np.append(starts, n))
    denom = np.maximum(sizes[groups] - 1, 1)

    ranked_sorted = positions.astype(np.float64) / denom
    ranked_sorted[sizes[groups] == 1] = 0.5

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def scipy_to_torch(matrix):
    matrix = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack([matrix.row, matrix.col]).astype(np.int64)
    )
    values = torch.from_numpy(matrix.data.astype(np.float32))
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=matrix.shape,
        dtype=torch.float32,
    ).coalesce()


def normalized_symmetric_adjacency(n_nodes, rows, cols, values):
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)

    rr = np.concatenate([rows, cols])
    cc = np.concatenate([cols, rows])
    vv = np.concatenate([values, values])

    adjacency = sp.coo_matrix(
        (vv, (rr, cc)),
        shape=(n_nodes, n_nodes),
        dtype=np.float32,
    ).tocsr()
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()

    degree = np.asarray(abs(adjacency).sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])

    normalized = sp.diags(inv_sqrt).dot(adjacency).dot(
        sp.diags(inv_sqrt)
    )
    return scipy_to_torch(normalized)


class GraphRanker(nn.Module):
    def __init__(self, n_nodes, adjacency, user_offset, video_offset):
        super().__init__()
        self.n_nodes = n_nodes
        self.adjacency = adjacency
        self.user_offset = user_offset
        self.video_offset = video_offset

        self.embedding = nn.Parameter(torch.empty(n_nodes, DIM))
        self.user_bias = nn.Embedding(N_USER, 1)
        self.video_bias = nn.Embedding(N_VIDEO, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.embedding, mean=0.0, std=0.08)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.video_bias.weight)

    def propagated(self):
        e0 = self.embedding
        e1 = torch.sparse.mm(self.adjacency, e0)
        e2 = torch.sparse.mm(self.adjacency, e1)
        return (e0 + e1 + e2) / 3.0

    def logits_from_embedding(self, propagated, users, videos):
        ue = propagated[users + self.user_offset]
        ve = propagated[videos + self.video_offset]
        interaction = (ue * ve).sum(dim=1) / np.sqrt(float(DIM))
        return (
            interaction
            + self.user_bias(users).squeeze(1)
            + self.video_bias(videos).squeeze(1)
            + self.global_bias
        )


def train_graph_model(adjacency, n_nodes, user_offset, video_offset,
                      users, videos, labels, sample_weights, seed):
    torch.manual_seed(seed)
    model = GraphRanker(
        n_nodes, adjacency, user_offset, video_offset
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    users_t = torch.from_numpy(users.astype(np.int64, copy=False))
    videos_t = torch.from_numpy(videos.astype(np.int64, copy=False))
    labels_t = torch.from_numpy(labels.astype(np.float32, copy=False))
    weights_t = torch.from_numpy(
        sample_weights.astype(np.float32, copy=False)
    )

    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        propagated = model.propagated()
        logits = model.logits_from_embedding(
            propagated, users_t, videos_t
        )
        losses = nn.functional.binary_cross_entropy_with_logits(
            logits, labels_t, reduction="none"
        )
        loss = (losses * weights_t).sum() / weights_t.sum()
        loss.backward()
        optimizer.step()

    return model


def predict_graph(model, users, videos):
    users_t = torch.from_numpy(
        np.asarray(users, dtype=np.int64)
    )
    videos_t = torch.from_numpy(
        np.asarray(videos, dtype=np.int64)
    )
    model.eval()
    result = np.empty(len(users), dtype=np.float64)
    with torch.inference_mode():
        propagated = model.propagated()
        for start in range(0, len(users), 65536):
            end = min(start + 65536, len(users))
            result[start:end] = model.logits_from_embedding(
                propagated,
                users_t[start:end],
                videos_t[start:end],
            ).numpy().astype(np.float64)
    return result


def fit_spectral_residual(users, videos, labels, weights):
    global_rate = float(
        np.sum(weights * labels) / np.sum(weights)
    )
    video_pos = np.bincount(
        videos,
        weights=weights * labels,
        minlength=N_VIDEO,
    )
    video_weight = np.bincount(
        videos,
        weights=weights,
        minlength=N_VIDEO,
    )
    smooth = 30.0
    video_rate = (
        video_pos + smooth * global_rate
    ) / (video_weight + smooth)

    residual = weights.astype(np.float64) * (
        labels.astype(np.float64) - video_rate[videos]
    )
    matrix = sp.coo_matrix(
        (residual, (users, videos)),
        shape=(N_USER, N_VIDEO),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()

    row_degree = np.asarray(abs(matrix).sum(axis=1)).ravel()
    col_degree = np.asarray(abs(matrix).sum(axis=0)).ravel()
    row_scale = np.zeros_like(row_degree)
    col_scale = np.zeros_like(col_degree)
    row_scale[row_degree > 0] = 1.0 / np.sqrt(
        row_degree[row_degree > 0]
    )
    col_scale[col_degree > 0] = 1.0 / np.sqrt(
        col_degree[col_degree > 0]
    )

    normalized = sp.diags(row_scale).dot(matrix).dot(
        sp.diags(col_scale)
    )
    u, singular, vt = svds(
        normalized,
        k=32,
        which="LM",
        random_state=SEED,
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order]
    u = u[:, order]
    vt = vt[order]

    user_factors = u * singular[None, :]
    video_factors = vt.T
    base_logit = np.log(
        np.clip(video_rate, 1e-5, 1.0 - 1e-5)
        / np.clip(1.0 - video_rate, 1e-5, 1.0)
    )
    return user_factors, video_factors, base_logit


def predict_spectral(fitted, users, videos):
    user_factors, video_factors, base_logit = fitted
    interaction = np.sum(
        user_factors[users] * video_factors[videos],
        axis=1,
    )
    return base_logit[videos] + 3.0 * interaction


train = load("train")
valid = load("valid")

tr_user = np.asarray(train.X["user_id"], dtype=np.int64)
tr_video = np.asarray(train.X["video_id"], dtype=np.int64)
tr_author = np.asarray(train.X["author_id"], dtype=np.int64)
tr_tag = np.asarray(train.X["tag"], dtype=np.int64)
tr_y = np.asarray(train.y, dtype=np.float32)
tr_w = recency_weights(train.date)

va_user = np.asarray(valid.X["user_id"], dtype=np.int64)
va_video = np.asarray(valid.X["video_id"], dtype=np.int64)

# Positive collaborative graph.
positive = tr_y > 0.5
collab_user_offset = 0
collab_video_offset = N_USER
collab_nodes = N_USER + N_VIDEO
collab_adj = normalized_symmetric_adjacency(
    collab_nodes,
    tr_user[positive] + collab_user_offset,
    tr_video[positive] + collab_video_offset,
    tr_w[positive],
)

# Signed graph: dislikes and short views send an opposing graph message.
signed_values = tr_w * np.where(positive, 1.0, -0.35).astype(
    np.float32
)
signed_adj = normalized_symmetric_adjacency(
    collab_nodes,
    tr_user + collab_user_offset,
    tr_video + collab_video_offset,
    signed_values,
)

# Heterogeneous graph connects each video to its stable author and tag.
hetero_user_offset = 0
hetero_video_offset = N_USER
hetero_author_offset = N_USER + N_VIDEO
hetero_tag_offset = N_USER + N_VIDEO + N_AUTHOR
hetero_nodes = N_USER + N_VIDEO + N_AUTHOR + N_TAG

unique_video, first_index = np.unique(
    tr_video, return_index=True
)
video_author = tr_author[first_index]
video_tag = tr_tag[first_index]

hetero_rows = np.concatenate([
    tr_user[positive] + hetero_user_offset,
    unique_video + hetero_video_offset,
    unique_video + hetero_video_offset,
])
hetero_cols = np.concatenate([
    tr_video[positive] + hetero_video_offset,
    video_author + hetero_author_offset,
    video_tag + hetero_tag_offset,
])
hetero_values = np.concatenate([
    tr_w[positive],
    np.full(unique_video.size, 1.5, dtype=np.float32),
    np.full(unique_video.size, 0.7, dtype=np.float32),
])
hetero_adj = normalized_symmetric_adjacency(
    hetero_nodes,
    hetero_rows,
    hetero_cols,
    hetero_values,
)

models = {}
valid_raw = {}

models["lightgcn_positive"] = train_graph_model(
    collab_adj,
    collab_nodes,
    collab_user_offset,
    collab_video_offset,
    tr_user,
    tr_video,
    tr_y,
    tr_w,
    SEED + 10,
)
valid_raw["lightgcn_positive"] = predict_graph(
    models["lightgcn_positive"], va_user, va_video
)

models["signed_feedback_gcn"] = train_graph_model(
    signed_adj,
    collab_nodes,
    collab_user_offset,
    collab_video_offset,
    tr_user,
    tr_video,
    tr_y,
    tr_w,
    SEED + 20,
)
valid_raw["signed_feedback_gcn"] = predict_graph(
    models["signed_feedback_gcn"], va_user, va_video
)

models["heterogeneous_lightgcn"] = train_graph_model(
    hetero_adj,
    hetero_nodes,
    hetero_user_offset,
    hetero_video_offset,
    tr_user,
    tr_video,
    tr_y,
    tr_w,
    SEED + 30,
)
valid_raw["heterogeneous_lightgcn"] = predict_graph(
    models["heterogeneous_lightgcn"], va_user, va_video
)

spectral_model = fit_spectral_residual(
    tr_user, tr_video, tr_y, tr_w
)
valid_raw["spectral_residual_svd"] = predict_spectral(
    spectral_model, va_user, va_video
)

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared_dir, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared_dir, "incumbent_test_scores.npy"
)
if not os.path.exists(inc_valid_path):
    raise FileNotFoundError(
        "Trusted incumbent validation scores unavailable"
    )
if not os.path.exists(inc_test_path):
    raise FileNotFoundError(
        "Trusted incumbent test scores unavailable"
    )

inc_valid = np.asarray(
    np.load(inc_valid_path), dtype=np.float64
)
inc_valid_rank = within_user_rank(
    valid.user_id, inc_valid
)

candidate_scores = {}
candidate_arrays = {}
candidate_family = {}
candidate_alpha = {}

inc_metric = evaluate(valid.user_id, valid.y, inc_valid)
candidate_scores["trusted_incumbent"] = float(
    inc_metric["primary"]
)
candidate_arrays["trusted_incumbent"] = inc_valid
candidate_family["trusted_incumbent"] = "lightgcn_positive"
candidate_alpha["trusted_incumbent"] = 0.0

for family, raw in valid_raw.items():
    standalone = evaluate(valid.user_id, valid.y, raw)
    candidate_scores[family] = float(standalone["primary"])
    candidate_arrays[family] = raw
    candidate_family[family] = family
    candidate_alpha[family] = 1.0

    graph_rank = within_user_rank(valid.user_id, raw)
    for alpha in (0.10, 0.20, 0.35, 0.50, 0.65, 0.80):
        blended = (
            alpha * graph_rank
            + (1.0 - alpha) * inc_valid_rank
        )
        name = f"{family}_blend_{alpha:.2f}"
        result = evaluate(valid.user_id, valid.y, blended)
        candidate_scores[name] = float(result["primary"])
        candidate_arrays[name] = blended
        candidate_family[name] = family
        candidate_alpha[name] = alpha

winner = max(candidate_scores, key=candidate_scores.get)
winner_family = candidate_family[winner]
winner_alpha = candidate_alpha[winner]
valid_scores = candidate_arrays[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(
    candidate_scores, sort_keys=True
))
print("FINDINGS " + json.dumps({
    "winner": winner,
    "family": winner_family,
    "graph_weight": float(winner_alpha),
    "positive_graph_edges": int(positive.sum()),
    "signed_graph_rows": int(len(tr_y)),
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if winner_alpha < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(
                valid_raw[winner_family], dtype=np.float64
            ),
        )

test = load("test")
te_user = np.asarray(test.X["user_id"], dtype=np.int64)
te_video = np.asarray(test.X["video_id"], dtype=np.int64)

if winner_family == "spectral_residual_svd":
    test_raw = predict_spectral(
        spectral_model, te_user, te_video
    )
else:
    test_raw = predict_graph(
        models[winner_family], te_user, te_video
    )

if winner_alpha <= 0.0:
    test_scores = np.asarray(
        np.load(inc_test_path), dtype=np.float64
    )
elif winner_alpha >= 1.0:
    test_scores = test_raw
else:
    inc_test = np.asarray(
        np.load(inc_test_path), dtype=np.float64
    )
    test_scores = (
        winner_alpha
        * within_user_rank(test.user_id, test_raw)
        + (1.0 - winner_alpha)
        * within_user_rank(test.user_id, inc_test)
    )

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))