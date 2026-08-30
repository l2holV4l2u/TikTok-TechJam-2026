import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 73129
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

USER_CARD = int(FEATURE_CARDINALITIES["user_id"])
VIDEO_CARD = int(FEATURE_CARDINALITIES["video_id"])
AUTHOR_CARD = int(FEATURE_CARDINALITIES["author_id"])
ENTITY_CARD = VIDEO_CARD + AUTHOR_CARD
LATENT_DIM = 24
BPR_DIM = 24
BPR_EPOCHS = 3
BPR_BATCH = 8192
DEVICE = torch.device("cpu")


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates, dtype=np.float64)
    age = float(np.max(dates)) - dates
    w = np.exp2(-age / float(half_life))
    w /= max(float(w.mean()), 1e-12)
    return w.astype(np.float32)


def within_user_percentile(user_ids, scores):
    uid = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, uid))
    su = uid[order]

    starts_flag = np.r_[True, su[1:] != su[:-1]]
    starts = np.maximum.accumulate(
        np.where(starts_flag, np.arange(n, dtype=np.int64), 0)
    )
    ends_flag = np.r_[su[:-1] != su[1:], True]
    ends = np.minimum.accumulate(
        np.where(ends_flag, np.arange(n, dtype=np.int64), n - 1)[::-1]
    )[::-1]

    lengths = ends - starts + 1
    positions = np.arange(n, dtype=np.int64) - starts
    ranks = np.where(
        lengths > 1,
        positions.astype(np.float64) / np.maximum(lengths - 1, 1),
        0.5,
    )
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def build_joint_matrix(user_ids, video_ids, author_ids, labels, dates, mode):
    u = np.asarray(user_ids, dtype=np.int64)
    v = np.asarray(video_ids, dtype=np.int64)
    a = np.asarray(author_ids, dtype=np.int64) + VIDEO_CARD
    y = np.asarray(labels, dtype=np.float32)
    w = recency_weights(dates, 4.0)

    if mode in ("positive", "normalized"):
        keep = y > 0
        rows = np.concatenate([u[keep], u[keep]])
        cols = np.concatenate([v[keep], a[keep]])
        vals = np.concatenate([w[keep], w[keep]]).astype(np.float32)
    elif mode == "signed":
        # Center observed labels so both positive and negative exposure carries
        # collaborative information rather than treating all missing pairs alike.
        base = float(np.average(y, weights=w))
        residual = (y - base) * w
        rows = np.concatenate([u, u])
        cols = np.concatenate([v, a])
        vals = np.concatenate([residual, residual]).astype(np.float32)
    else:
        raise ValueError(mode)

    mat = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(USER_CARD, ENTITY_CARD),
        dtype=np.float32,
    ).tocsr()
    mat.sum_duplicates()
    mat.eliminate_zeros()

    if mode == "normalized":
        row_degree = np.asarray(np.abs(mat).sum(axis=1)).ravel()
        col_degree = np.asarray(np.abs(mat).sum(axis=0)).ravel()
        row_scale = np.zeros_like(row_degree, dtype=np.float32)
        col_scale = np.zeros_like(col_degree, dtype=np.float32)
        row_scale[row_degree > 0] = 1.0 / np.sqrt(row_degree[row_degree > 0])
        col_scale[col_degree > 0] = 1.0 / np.sqrt(col_degree[col_degree > 0])
        mat = sp.diags(row_scale).dot(mat).dot(sp.diags(col_scale)).tocsr()

    return mat


def fit_spectral(user_ids, video_ids, author_ids, labels, dates, mode):
    mat = build_joint_matrix(
        user_ids, video_ids, author_ids, labels, dates, mode
    )
    u, singular, vt = svds(
        mat,
        k=LATENT_DIM,
        which="LM",
        tol=2e-3,
        maxiter=250,
        random_state=SEED + {"positive": 1, "signed": 2, "normalized": 3}[mode],
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order].astype(np.float32)
    u = u[:, order].astype(np.float32)
    vt = vt[order].astype(np.float32)
    user_factors = u * singular[None, :]
    entity_factors = vt.T
    return user_factors, entity_factors


def predict_spectral(model, user_ids, video_ids, author_ids):
    user_factors, entity_factors = model
    u = np.asarray(user_ids, dtype=np.int64)
    v = np.asarray(video_ids, dtype=np.int64)
    a = np.asarray(author_ids, dtype=np.int64) + VIDEO_CARD
    uv = np.einsum(
        "ij,ij->i", user_factors[u], entity_factors[v], optimize=True
    )
    ua = np.einsum(
        "ij,ij->i", user_factors[u], entity_factors[a], optimize=True
    )
    return (uv + ua).astype(np.float64)


class GraphBPR(nn.Module):
    def __init__(self):
        super().__init__()
        self.user = nn.Embedding(USER_CARD, BPR_DIM)
        self.video = nn.Embedding(VIDEO_CARD, BPR_DIM)
        self.author = nn.Embedding(AUTHOR_CARD, BPR_DIM)
        self.video_bias = nn.Embedding(VIDEO_CARD, 1)
        self.author_bias = nn.Embedding(AUTHOR_CARD, 1)
        nn.init.normal_(self.user.weight, std=0.035)
        nn.init.normal_(self.video.weight, std=0.035)
        nn.init.normal_(self.author.weight, std=0.035)
        nn.init.zeros_(self.video_bias.weight)
        nn.init.zeros_(self.author_bias.weight)

    def forward(self, u, v, a):
        ue = self.user(u)
        score = (ue * self.video(v)).sum(dim=1)
        score = score + (ue * self.author(a)).sum(dim=1)
        score = score + self.video_bias(v).squeeze(1)
        score = score + self.author_bias(a).squeeze(1)
        return score


def pair_sampling_state(user_ids, labels):
    uid = np.asarray(user_ids, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int8)
    _, inverse = np.unique(uid, return_inverse=True)
    group_count = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=group_count).astype(np.int64)
    positives = np.bincount(
        inverse, weights=y, minlength=group_count
    ).astype(np.int64)
    negatives = counts - positives
    row_order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts[:-1])].astype(np.int64)
    anchors = np.flatnonzero(
        (y == 1) & (negatives[inverse] > 0)
    ).astype(np.int64)
    return inverse, counts, starts, row_order, anchors


def sample_observed_negatives(state, labels, rng):
    inverse, counts, starts, row_order, anchors = state
    groups = inverse[anchors]
    offsets = (rng.random(len(anchors)) * counts[groups]).astype(np.int64)
    negatives = row_order[starts[groups] + offsets]
    y = np.asarray(labels, dtype=np.int8)
    bad = y[negatives] != 0
    while bad.any():
        bad_groups = groups[bad]
        offsets = (
            rng.random(int(bad.sum())) * counts[bad_groups]
        ).astype(np.int64)
        negatives[bad] = row_order[starts[bad_groups] + offsets]
        bad = y[negatives] != 0
    return anchors, negatives


def fit_bpr(user_ids, video_ids, author_ids, labels, dates, seed):
    torch.manual_seed(seed)
    model = GraphBPR().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=2e-6
    )

    users_t = torch.from_numpy(np.asarray(user_ids, dtype=np.int64))
    videos_t = torch.from_numpy(np.asarray(video_ids, dtype=np.int64))
    authors_t = torch.from_numpy(np.asarray(author_ids, dtype=np.int64))
    weights_t = torch.from_numpy(recency_weights(dates, 4.0))
    labels_np = np.asarray(labels, dtype=np.int8)
    state = pair_sampling_state(user_ids, labels_np)
    rng = np.random.default_rng(seed + 41)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 73)

    model.train()
    for _ in range(BPR_EPOCHS):
        positives, negatives = sample_observed_negatives(state, labels_np, rng)
        pos_t = torch.from_numpy(positives)
        neg_t = torch.from_numpy(negatives)
        order = torch.randperm(len(positives), generator=generator)

        for start in range(0, len(order), BPR_BATCH):
            idx = order[start:start + BPR_BATCH]
            p = pos_t[idx]
            n = neg_t[idx]

            optimizer.zero_grad(set_to_none=True)
            pos_score = model(users_t[p], videos_t[p], authors_t[p])
            neg_score = model(users_t[n], videos_t[n], authors_t[n])
            loss = (
                nn.functional.softplus(-(pos_score - neg_score)) * weights_t[p]
            ).mean()
            loss.backward()
            optimizer.step()

    return model


@torch.inference_mode()
def predict_bpr(model, user_ids, video_ids, author_ids, batch=32768):
    model.eval()
    u = torch.from_numpy(np.asarray(user_ids, dtype=np.int64))
    v = torch.from_numpy(np.asarray(video_ids, dtype=np.int64))
    a = torch.from_numpy(np.asarray(author_ids, dtype=np.int64))
    out = np.empty(len(u), dtype=np.float64)
    for start in range(0, len(u), batch):
        end = min(start + batch, len(u))
        out[start:end] = model(
            u[start:end], v[start:end], a[start:end]
        ).cpu().numpy().astype(np.float64)
    return out


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_rank = within_user_percentile(valid.user_id, inc_valid)

raw_predictions = {}
fitted_models = {}

for mode in ("positive", "signed", "normalized"):
    model = fit_spectral(
        train.user_id,
        train.video_id,
        train.X["author_id"],
        y_train,
        train.date,
        mode,
    )
    fitted_models[mode] = model
    raw_predictions[mode] = predict_spectral(
        model,
        valid.user_id,
        valid.video_id,
        valid.X["author_id"],
    )

bpr_model = fit_bpr(
    train.user_id,
    train.video_id,
    train.X["author_id"],
    y_train,
    train.date,
    SEED + 100,
)
fitted_models["bpr"] = bpr_model
raw_predictions["bpr"] = predict_bpr(
    bpr_model,
    valid.user_id,
    valid.video_id,
    valid.X["author_id"],
)

candidate_scores = {}
candidate_predictions = {}
candidate_recipes = {}

inc_metric = evaluate(valid.user_id, valid.y, inc_valid)
candidate_scores["incumbent"] = float(inc_metric["primary"])
candidate_predictions["incumbent"] = inc_valid
candidate_recipes["incumbent"] = ("incumbent", 0.0)

for family, raw_pred in raw_predictions.items():
    raw_metric = evaluate(valid.user_id, valid.y, raw_pred)
    candidate_scores[family + "_raw"] = float(raw_metric["primary"])
    candidate_predictions[family + "_raw"] = raw_pred
    candidate_recipes[family + "_raw"] = (family, 1.0)

    graph_rank = within_user_percentile(valid.user_id, raw_pred)
    rank_metric = evaluate(valid.user_id, valid.y, graph_rank)
    candidate_scores[family + "_rank"] = float(rank_metric["primary"])
    candidate_predictions[family + "_rank"] = graph_rank
    candidate_recipes[family + "_rank"] = (family, 1.0)

    for alpha in (0.10, 0.20, 0.30, 0.40, 0.50):
        name = "%s_rankblend_%.2f" % (family, alpha)
        blended = (1.0 - alpha) * inc_rank + alpha * graph_rank
        metric = evaluate(valid.user_id, valid.y, blended)
        candidate_scores[name] = float(metric["primary"])
        candidate_predictions[name] = blended
        candidate_recipes[name] = (family, alpha)

winner = max(candidate_scores, key=candidate_scores.get)
family, alpha = candidate_recipes[winner]
valid_scores = np.asarray(candidate_predictions[winner], dtype=np.float64)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS graph_standalone positive=%.6f signed=%.6f normalized=%.6f bpr=%.6f winner=%s"
    % (
        candidate_scores["positive_rank"],
        candidate_scores["signed_rank"],
        candidate_scores["normalized_rank"],
        candidate_scores["bpr_rank"],
        winner,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if family == "incumbent":
    test_scores = inc_test.copy()
else:
    y_valid = np.asarray(valid.y, dtype=np.int8)
    combined_users = np.concatenate([
        np.asarray(train.user_id, dtype=np.int64),
        np.asarray(valid.user_id, dtype=np.int64),
    ])
    combined_videos = np.concatenate([
        np.asarray(train.video_id, dtype=np.int64),
        np.asarray(valid.video_id, dtype=np.int64),
    ])
    combined_authors = np.concatenate([
        np.asarray(train.X["author_id"], dtype=np.int64),
        np.asarray(valid.X["author_id"], dtype=np.int64),
    ])
    combined_labels = np.concatenate([y_train, y_valid])
    combined_dates = np.concatenate([
        np.asarray(train.date),
        np.asarray(valid.date),
    ])

    if family in ("positive", "signed", "normalized"):
        final_model = fit_spectral(
            combined_users,
            combined_videos,
            combined_authors,
            combined_labels,
            combined_dates,
            family,
        )
        graph_test_raw = predict_spectral(
            final_model,
            test.user_id,
            test.video_id,
            test.X["author_id"],
        )
    elif family == "bpr":
        final_model = fit_bpr(
            combined_users,
            combined_videos,
            combined_authors,
            combined_labels,
            combined_dates,
            SEED + 100,
        )
        graph_test_raw = predict_bpr(
            final_model,
            test.user_id,
            test.video_id,
            test.X["author_id"],
        )
    else:
        raise RuntimeError("Unknown winning family: " + family)

    graph_test_rank = within_user_percentile(test.user_id, graph_test_raw)
    if alpha >= 0.999:
        test_scores = graph_test_rank
    else:
        inc_test_rank = within_user_percentile(test.user_id, inc_test)
        test_scores = (1.0 - alpha) * inc_test_rank + alpha * graph_test_rank

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.4f}'
    % (
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        elapsed,
    )
)