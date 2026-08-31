import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 73129
DIM = 32
HISTORY_LEN = 8
EPOCHS = 4
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 65536
HALF_LIFE = 4.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def day_indices(dates, reference_dates=None):
    dates = np.asarray(dates, dtype=np.int64)
    if reference_dates is None:
        unique = np.sort(np.unique(dates))
    else:
        unique = np.sort(np.unique(np.asarray(reference_dates, dtype=np.int64)))
    mapping = {int(d): i for i, d in enumerate(unique)}
    return np.asarray([mapping[int(d)] for d in dates], dtype=np.int64)


def build_negative_pool(user_ids, labels, n_users):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    negative_rows = np.flatnonzero(labels == 0)
    order = np.argsort(user_ids[negative_rows], kind="stable")
    negative_rows = negative_rows[order]
    counts = np.bincount(user_ids[negative_rows], minlength=n_users).astype(np.int64)
    bases = np.zeros(n_users, dtype=np.int64)
    if n_users > 1:
        bases[1:] = np.cumsum(counts[:-1])
    return negative_rows, counts, bases


def build_positive_video_history(train, target=None):
    train_users = np.asarray(train.X["user_id"], dtype=np.int64)
    train_videos = np.asarray(train.X["video_id"], dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.int8)
    times = np.asarray(train.time_ms, dtype=np.int64)
    rows = np.arange(train_users.size, dtype=np.int64)

    order = np.lexsort((rows, times, train_users))
    sorted_users = train_users[order]
    sorted_labels = labels[order]
    sorted_positive = sorted_labels > 0

    n_users = int(FEATURE_CARDINALITIES["user_id"])
    positive_counts = np.bincount(
        sorted_users[sorted_positive], minlength=n_users
    ).astype(np.int64)
    positive_bases = np.zeros(n_users, dtype=np.int64)
    if n_users > 1:
        positive_bases[1:] = np.cumsum(positive_counts[:-1])

    positive_tokens = train_videos[order[sorted_positive]]

    if target is not None:
        target_users = np.asarray(target.X["user_id"], dtype=np.int64)
        available = positive_counts[target_users]
        sequence = np.zeros((target_users.size, HISTORY_LEN), dtype=np.int64)
        mask = np.zeros((target_users.size, HISTORY_LEN), dtype=np.float32)

        for col, distance in enumerate(range(HISTORY_LEN, 0, -1)):
            valid = available >= distance
            indices = (
                positive_bases[target_users[valid]]
                + available[valid]
                - distance
            )
            sequence[valid, col] = positive_tokens[indices]
            mask[valid, col] = 1.0
        return sequence, mask

    cumulative = np.cumsum(sorted_positive.astype(np.int64))
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], sorted_users.size]
    lengths = ends - starts

    cumulative_before = np.zeros(starts.size, dtype=np.int64)
    nonzero = starts > 0
    cumulative_before[nonzero] = cumulative[starts[nonzero] - 1]
    base_per_row = np.repeat(cumulative_before, lengths)
    prior_count = cumulative - base_per_row - sorted_positive.astype(np.int64)

    seq_sorted = np.zeros((sorted_users.size, HISTORY_LEN), dtype=np.int64)
    mask_sorted = np.zeros((sorted_users.size, HISTORY_LEN), dtype=np.float32)

    for col, distance in enumerate(range(HISTORY_LEN, 0, -1)):
        valid = prior_count >= distance
        indices = (
            positive_bases[sorted_users[valid]]
            + prior_count[valid]
            - distance
        )
        seq_sorted[valid, col] = positive_tokens[indices]
        mask_sorted[valid, col] = 1.0

    sequence = np.zeros_like(seq_sorted)
    mask = np.zeros_like(mask_sorted)
    sequence[order] = seq_sorted
    mask[order] = mask_sorted
    return sequence, mask


class PairwiseMF(nn.Module):
    def __init__(self, sequential=False):
        super().__init__()
        self.sequential = sequential
        nu = int(FEATURE_CARDINALITIES["user_id"])
        nv = int(FEATURE_CARDINALITIES["video_id"])
        na = int(FEATURE_CARDINALITIES["author_id"])
        nt = int(FEATURE_CARDINALITIES["tab"])
        nd = int(FEATURE_CARDINALITIES["duration_bucket"])

        self.user_video = nn.Embedding(nu, DIM)
        self.video = nn.Embedding(nv, DIM)
        self.user_author = nn.Embedding(nu, DIM // 2)
        self.author = nn.Embedding(na, DIM // 2)

        self.video_bias = nn.Embedding(nv, 1)
        self.author_bias = nn.Embedding(na, 1)
        self.tab_bias = nn.Embedding(nt, 1)
        self.duration_bias = nn.Embedding(nd, 1)

        nn.init.normal_(self.user_video.weight, std=0.035)
        nn.init.normal_(self.video.weight, std=0.035)
        nn.init.normal_(self.user_author.weight, std=0.035)
        nn.init.normal_(self.author.weight, std=0.035)
        nn.init.zeros_(self.video_bias.weight)
        nn.init.zeros_(self.author_bias.weight)
        nn.init.zeros_(self.tab_bias.weight)
        nn.init.zeros_(self.duration_bias.weight)

        if sequential:
            self.history_projection = nn.Linear(DIM, DIM, bias=False)
            nn.init.eye_(self.history_projection.weight)
            self.history_scale = nn.Parameter(torch.tensor(0.5))

    def score(self, users, videos, authors, tabs, durations,
              history=None, history_mask=None):
        uv = self.user_video(users)
        vv = self.video(videos)
        score = (uv * vv).sum(dim=1) / np.sqrt(DIM)

        ua = self.user_author(users)
        aa = self.author(authors)
        score = score + (ua * aa).sum(dim=1) / np.sqrt(DIM // 2)

        score = (
            score
            + self.video_bias(videos).squeeze(1)
            + self.author_bias(authors).squeeze(1)
            + self.tab_bias(tabs).squeeze(1)
            + self.duration_bias(durations).squeeze(1)
        )

        if self.sequential:
            hist_emb = self.video(history)
            mask = history_mask.unsqueeze(2)
            pooled = (hist_emb * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = pooled / denom
            pooled = self.history_projection(pooled)
            seq_score = (pooled * vv).sum(dim=1) / np.sqrt(DIM)
            has_history = (history_mask.sum(dim=1) > 0).float()
            score = score + self.history_scale * seq_score * has_history

        return score


def fit_pairwise(train, sequential=False, history=None, history_mask=None):
    users_np = np.asarray(train.X["user_id"], dtype=np.int64)
    videos_np = np.asarray(train.X["video_id"], dtype=np.int64)
    authors_np = np.asarray(train.X["author_id"], dtype=np.int64)
    tabs_np = np.asarray(train.X["tab"], dtype=np.int64)
    durations_np = np.asarray(train.X["duration_bucket"], dtype=np.int64)
    labels_np = np.asarray(train.y, dtype=np.int8)

    n_users = int(FEATURE_CARDINALITIES["user_id"])
    negative_rows, negative_counts, negative_bases = build_negative_pool(
        users_np, labels_np, n_users
    )

    positive_rows = np.flatnonzero(labels_np > 0)
    positive_rows = positive_rows[negative_counts[users_np[positive_rows]] > 0]

    unique_days = np.sort(np.unique(np.asarray(train.date, dtype=np.int64)))
    day_map = {int(d): i for i, d in enumerate(unique_days)}
    day_idx = np.asarray(
        [day_map[int(d)] for d in np.asarray(train.date, dtype=np.int64)],
        dtype=np.float32
    )
    age = float(len(unique_days) - 1) - day_idx[positive_rows]
    pair_weights = np.exp(-np.log(2.0) * age / HALF_LIFE).astype(np.float32)
    pair_weights /= pair_weights.mean()

    users = torch.from_numpy(users_np)
    videos = torch.from_numpy(videos_np)
    authors = torch.from_numpy(authors_np)
    tabs = torch.from_numpy(tabs_np)
    durations = torch.from_numpy(durations_np)

    hist_t = torch.from_numpy(history) if sequential else None
    hist_mask_t = torch.from_numpy(history_mask) if sequential else None

    model = PairwiseMF(sequential=sequential)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-6)
    rng = np.random.default_rng(SEED + (101 if sequential else 0))

    for epoch in range(EPOCHS):
        pos_order = rng.permutation(positive_rows.size)
        epoch_pos = positive_rows[pos_order]
        epoch_weights = pair_weights[pos_order]

        pos_users = users_np[epoch_pos]
        counts = negative_counts[pos_users]
        draws = (rng.random(epoch_pos.size) * counts).astype(np.int64)
        epoch_neg = negative_rows[negative_bases[pos_users] + draws]

        model.train()
        for start in range(0, epoch_pos.size, BATCH_SIZE):
            end = min(start + BATCH_SIZE, epoch_pos.size)
            pr_np = epoch_pos[start:end]
            nr_np = epoch_neg[start:end]
            pr = torch.from_numpy(pr_np)
            nr = torch.from_numpy(nr_np)
            w = torch.from_numpy(epoch_weights[start:end])

            optimizer.zero_grad(set_to_none=True)

            if sequential:
                pos_score = model.score(
                    users[pr], videos[pr], authors[pr], tabs[pr], durations[pr],
                    hist_t[pr], hist_mask_t[pr]
                )
                neg_score = model.score(
                    users[pr], videos[nr], authors[nr], tabs[nr], durations[nr],
                    hist_t[pr], hist_mask_t[pr]
                )
            else:
                pos_score = model.score(
                    users[pr], videos[pr], authors[pr], tabs[pr], durations[pr]
                )
                neg_score = model.score(
                    users[pr], videos[nr], authors[nr], tabs[nr], durations[nr]
                )

            loss = (F.softplus(-(pos_score - neg_score)) * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def predict_pairwise(model, split, history=None, history_mask=None):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    videos = np.asarray(split.X["video_id"], dtype=np.int64)
    authors = np.asarray(split.X["author_id"], dtype=np.int64)
    tabs = np.asarray(split.X["tab"], dtype=np.int64)
    durations = np.asarray(split.X["duration_bucket"], dtype=np.int64)

    out = np.empty(users.size, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, users.size, PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, users.size)
            args = [
                torch.from_numpy(users[start:end]),
                torch.from_numpy(videos[start:end]),
                torch.from_numpy(authors[start:end]),
                torch.from_numpy(tabs[start:end]),
                torch.from_numpy(durations[start:end]),
            ]
            if model.sequential:
                score = model.score(
                    *args,
                    torch.from_numpy(history[start:end]),
                    torch.from_numpy(history_mask[start:end])
                )
            else:
                score = model.score(*args)
            out[start:end] = score.cpu().numpy().astype(np.float64)
    return out


def fit_svd(train, rank=32):
    users = np.asarray(train.X["user_id"], dtype=np.int64)
    videos = np.asarray(train.X["video_id"], dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.int8)
    positive = labels > 0

    unique_days = np.sort(np.unique(np.asarray(train.date, dtype=np.int64)))
    day_map = {int(d): i for i, d in enumerate(unique_days)}
    day_idx = np.asarray(
        [day_map[int(d)] for d in np.asarray(train.date, dtype=np.int64)],
        dtype=np.float32
    )
    age = float(len(unique_days) - 1) - day_idx
    weights = np.exp(-np.log(2.0) * age / HALF_LIFE).astype(np.float32)

    matrix = sparse.coo_matrix(
        (
            weights[positive],
            (users[positive], videos[positive])
        ),
        shape=(
            int(FEATURE_CARDINALITIES["user_id"]),
            int(FEATURE_CARDINALITIES["video_id"])
        ),
        dtype=np.float32
    ).tocsr()
    matrix.sum_duplicates()
    matrix.data = np.log1p(matrix.data)

    u, singular, vt = svds(
        matrix, k=rank, which="LM",
        return_singular_vectors=True,
        random_state=SEED
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order]
    u = u[:, order]
    vt = vt[order]
    user_factors = u * singular[None, :]
    item_factors = vt.T

    item_count = np.asarray(matrix.sum(axis=0)).ravel()
    item_bias = np.log1p(item_count).astype(np.float64)
    return user_factors.astype(np.float32), item_factors.astype(np.float32), item_bias


def predict_svd(split, factors):
    user_factors, item_factors, item_bias = factors
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    videos = np.asarray(split.X["video_id"], dtype=np.int64)
    latent = np.sum(user_factors[users] * item_factors[videos], axis=1)
    return latent.astype(np.float64) + 0.03 * item_bias[videos]


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    lengths = ends - starts
    local_rank = np.arange(n, dtype=np.int64) - np.repeat(starts, lengths)

    ranked_sorted = (local_rank.astype(np.float64) + 0.5) / np.repeat(
        lengths, lengths
    )
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


train = load("train")
valid = load("valid")
test = load("test")

train_history, train_history_mask = build_positive_video_history(train)
valid_history, valid_history_mask = build_positive_video_history(train, valid)
test_history, test_history_mask = build_positive_video_history(train, test)

plain_model = fit_pairwise(train, sequential=False)
plain_valid = predict_pairwise(plain_model, valid)
plain_test = predict_pairwise(plain_model, test)

sequence_model = fit_pairwise(
    train, sequential=True,
    history=train_history, history_mask=train_history_mask
)
sequence_valid = predict_pairwise(
    sequence_model, valid, valid_history, valid_history_mask
)
sequence_test = predict_pairwise(
    sequence_model, test, test_history, test_history_mask
)

svd_factors = fit_svd(train, rank=32)
svd_valid = predict_svd(valid, svd_factors)
svd_test = predict_svd(test, svd_factors)

valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)
valid_labels = np.asarray(valid.y, dtype=np.int8)

raw_models = {
    "bpr_mf": (plain_valid, plain_test),
    "sequence_bpr": (sequence_valid, sequence_test),
    "truncated_svd": (svd_valid, svd_test),
}

candidate_scores = {}
for name, (va_score, _) in raw_models.items():
    candidate_scores[name] = float(
        evaluate(valid_users, valid_labels, va_score)["primary"]
    )

plain_rank_v = within_user_rank(valid_users, plain_valid)
plain_rank_t = within_user_rank(test_users, plain_test)
seq_rank_v = within_user_rank(valid_users, sequence_valid)
seq_rank_t = within_user_rank(test_users, sequence_test)
svd_rank_v = within_user_rank(valid_users, svd_valid)
svd_rank_t = within_user_rank(test_users, svd_test)

ensemble_valid = (plain_rank_v + seq_rank_v + svd_rank_v) / 3.0
ensemble_test = (plain_rank_t + seq_rank_t + svd_rank_t) / 3.0
raw_models["latent_rank_ensemble"] = (ensemble_valid, ensemble_test)
candidate_scores["latent_rank_ensemble"] = float(
    evaluate(valid_users, valid_labels, ensemble_valid)["primary"]
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

best_primary = -np.inf
best_valid = None
best_test = None
best_raw_valid = None
best_name = None

if os.path.exists(inc_valid_path) and os.path.exists(inc_test_path):
    incumbent_valid = np.load(inc_valid_path).astype(np.float64)
    incumbent_test = np.load(inc_test_path).astype(np.float64)
    inc_rank_v = within_user_rank(valid_users, incumbent_valid)
    inc_rank_t = within_user_rank(test_users, incumbent_test)

    inc_metric = evaluate(valid_users, valid_labels, incumbent_valid)
    candidate_scores["incumbent"] = float(inc_metric["primary"])

    blend_alphas = [0.15, 0.25, 0.35, 0.50, 0.65, 0.80]
    for name, (raw_valid, raw_test) in raw_models.items():
        own_rank_v = within_user_rank(valid_users, raw_valid)
        own_rank_t = within_user_rank(test_users, raw_test)
        for alpha in blend_alphas:
            blend_valid = (1.0 - alpha) * inc_rank_v + alpha * own_rank_v
            blend_test = (1.0 - alpha) * inc_rank_t + alpha * own_rank_t
            metric = evaluate(valid_users, valid_labels, blend_valid)
            key = "%s_blend_%.2f" % (name, alpha)
            candidate_scores[key] = float(metric["primary"])
            if metric["primary"] > best_primary:
                best_primary = float(metric["primary"])
                best_valid = blend_valid.copy()
                best_test = blend_test.copy()
                best_raw_valid = raw_valid.copy()
                best_name = key

    if inc_metric["primary"] > best_primary:
        best_primary = float(inc_metric["primary"])
        best_valid = incumbent_valid.copy()
        best_test = incumbent_test.copy()
        best_raw_valid = ensemble_valid.copy()
        best_name = "incumbent"
else:
    for name, (va_score, te_score) in raw_models.items():
        metric = evaluate(valid_users, valid_labels, va_score)
        if metric["primary"] > best_primary:
            best_primary = float(metric["primary"])
            best_valid = va_score.copy()
            best_test = te_score.copy()
            best_raw_valid = va_score.copy()
            best_name = name

final_metrics = evaluate(valid_users, valid_labels, best_valid)

print("FINDINGS winner=%s raw_bpr=%.6f raw_sequence=%.6f raw_svd=%.6f" % (
    best_name,
    candidate_scores["bpr_mf"],
    candidate_scores["sequence_bpr"],
    candidate_scores["truncated_svd"],
))
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64)
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64)
    )
    if best_raw_valid is not None and best_name not in raw_models:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64)
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(final_metrics["primary"]),
    "gauc": float(final_metrics["gauc"]),
    "ndcg@5": float(final_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))