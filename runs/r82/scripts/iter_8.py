import os
import gc
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 271828
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "hour",
]
DIM = 12
HISTORY_LEN = 8
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768
EPOCHS = 3
LR = 0.002
WEIGHT_DECAY = 1e-6
SVD_DIM = 24


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_all(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))
video_field_index = FIELDS.index("video_id")
video_offset = int(offsets[video_field_index])
num_users = int(FEATURE_CARDINALITIES["user_id"])
num_videos = int(FEATURE_CARDINALITIES["video_id"])


def make_current_matrix(split):
    n = len(split.user_id)
    x = np.empty((n, len(FIELDS)), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:, j] = np.asarray(split.X[field], dtype=np.int64) + offsets[j]
    return x


def make_combined_matrix(a, b):
    na = len(a.user_id)
    nb = len(b.user_id)
    x = np.empty((na + nb, len(FIELDS)), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:na, j] = np.asarray(a.X[field], dtype=np.int64) + offsets[j]
        x[na:, j] = np.asarray(b.X[field], dtype=np.int64) + offsets[j]
    return x


def make_history(reference_splits, target):
    """
    For each target impression, return the previous HISTORY_LEN impressed
    videos for the same user. Only logged item identities and timestamps are
    used; no target or auxiliary outcomes enter the feature.
    """
    ref_lengths = [len(s.user_id) for s in reference_splits]
    target_n = len(target.user_id)

    users_parts = [
        np.asarray(s.user_id, dtype=np.int64) for s in reference_splits
    ] + [np.asarray(target.user_id, dtype=np.int64)]
    times_parts = [
        np.asarray(s.time_ms, dtype=np.int64) for s in reference_splits
    ] + [np.asarray(target.time_ms, dtype=np.int64)]
    videos_parts = [
        np.asarray(s.video_id, dtype=np.int64) for s in reference_splits
    ] + [np.asarray(target.video_id, dtype=np.int64)]

    users = np.concatenate(users_parts)
    times = np.concatenate(times_parts)
    videos = np.concatenate(videos_parts)
    n = len(users)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, times, users))
    sorted_users = users[order]
    sorted_videos = videos[order]

    history_all = np.zeros((n, HISTORY_LEN), dtype=np.int64)
    positions = np.arange(n, dtype=np.int64)

    for lag in range(1, HISTORY_LEN + 1):
        valid = positions >= lag
        same = np.zeros(n, dtype=bool)
        same[valid] = (
            sorted_users[valid] == sorted_users[positions[valid] - lag]
        )
        dest_sorted = positions[same]
        src_sorted = dest_sorted - lag
        dest_rows = order[dest_sorted]
        history_all[dest_rows, HISTORY_LEN - lag] = (
            sorted_videos[src_sorted] + video_offset
        )

    ref_n = int(sum(ref_lengths))
    return history_all[ref_n:ref_n + target_n]


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    boundary = np.empty(n, dtype=bool)
    boundary[0] = True
    boundary[1:] = sorted_users[1:] != sorted_users[:-1]

    starts = np.maximum.accumulate(
        np.where(boundary, np.arange(n, dtype=np.int64), 0)
    )
    pos = np.arange(n, dtype=np.int64) - starts

    _, counts = np.unique(sorted_users, return_counts=True)
    denom = np.repeat(np.maximum(counts - 1, 1), counts).astype(np.float64)

    result = np.empty(n, dtype=np.float32)
    result[order] = (pos / denom).astype(np.float32)
    return result


class BaseSequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, DIM)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def wide(self, current):
        return self.bias + self.linear(current).sum(dim=1).squeeze(-1)

    def encode(self, current, history):
        cur_emb = self.embedding(current)
        hist_emb = self.embedding(history)
        mask = history != video_offset
        hist_emb = hist_emb * mask.unsqueeze(-1)
        candidate = cur_emb[:, video_field_index, :]
        return cur_emb, hist_emb, candidate, mask


class DINModel(BaseSequentialModel):
    def __init__(self):
        super().__init__()
        current_dim = len(FIELDS) * DIM
        joined_dim = current_dim + 4 * DIM
        self.attention = nn.Sequential(
            nn.Linear(4 * DIM, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.output = nn.Sequential(
            nn.Linear(joined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, current, history):
        cur_emb, hist_emb, candidate, mask = self.encode(current, history)
        cand = candidate.unsqueeze(1).expand_as(hist_emb)
        attention_input = torch.cat(
            [hist_emb, cand, hist_emb - cand, hist_emb * cand], dim=2
        )
        logits = self.attention(attention_input).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1)
        weights = weights * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        context = (weights.unsqueeze(-1) * hist_emb).sum(dim=1)

        interaction = torch.cat(
            [
                context,
                candidate,
                context - candidate,
                context * candidate,
            ],
            dim=1,
        )
        features = torch.cat([cur_emb.flatten(1), interaction], dim=1)
        return self.wide(current) + self.output(features).squeeze(-1)


class GRUModel(BaseSequentialModel):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(DIM, DIM, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(len(FIELDS) * DIM + 4 * DIM, 112),
            nn.ReLU(),
            nn.Linear(112, 40),
            nn.ReLU(),
            nn.Linear(40, 1),
        )

    def forward(self, current, history):
        cur_emb, hist_emb, candidate, mask = self.encode(current, history)
        lengths = mask.sum(dim=1)
        sequence, _ = self.gru(hist_emb)

        last_index = (lengths - 1).clamp_min(0)
        batch_index = torch.arange(
            len(current), device=current.device, dtype=torch.long
        )
        context = sequence[batch_index, last_index]
        context = context * (lengths > 0).unsqueeze(1)

        interaction = torch.cat(
            [
                context,
                candidate,
                context - candidate,
                context * candidate,
            ],
            dim=1,
        )
        features = torch.cat([cur_emb.flatten(1), interaction], dim=1)
        return self.wide(current) + self.output(features).squeeze(-1)


def make_model(name, seed):
    seed_all(seed)
    if name == "din_sequence":
        return DINModel()
    if name == "gru_sequence":
        return GRUModel()
    raise ValueError(name)


def train_epoch(model, optimizer, x_tensor, h_tensor, labels, rng):
    model.train()
    order = rng.permutation(len(labels))
    total = 0.0

    for start in range(0, len(order), BATCH_SIZE):
        idx_np = order[start:start + BATCH_SIZE]
        idx = torch.from_numpy(idx_np)
        yb = torch.from_numpy(labels[idx_np].astype(np.float32, copy=False))

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tensor[idx], h_tensor[idx])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach()) * len(idx_np)

    return total / len(labels)


@torch.no_grad()
def predict_sequence(model, x, history):
    model.eval()
    xt = torch.from_numpy(x)
    ht = torch.from_numpy(history)
    result = np.empty(len(x), dtype=np.float32)

    for start in range(0, len(x), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(x))
        result[start:end] = (
            model(xt[start:end], ht[start:end])
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
    return result


def fit_sequence_candidate(
    name, x_train, h_train, y_train, x_valid, h_valid, valid, y_valid
):
    model = make_model(name, SEED + 100)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    rng = np.random.default_rng(SEED + 500)

    xt = torch.from_numpy(x_train)
    ht = torch.from_numpy(h_train)

    best_primary = -np.inf
    best_epoch = 1
    best_scores = None
    epoch_scores = []

    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, optimizer, xt, ht, y_train, rng)
        scores = predict_sequence(model, x_valid, h_valid)
        metrics = evaluate(valid.user_id, y_valid, scores)
        primary = float(metrics["primary"])
        epoch_scores.append(primary)

        if primary > best_primary:
            best_primary = primary
            best_epoch = epoch
            best_scores = scores.copy()

    del model, optimizer, xt, ht
    gc.collect()
    return best_scores, best_epoch, epoch_scores


def fit_svd(train_split):
    users = np.asarray(train_split.user_id, dtype=np.int64)
    videos = np.asarray(train_split.video_id, dtype=np.int64)
    labels = np.asarray(train_split.y, dtype=np.int8)

    positive = labels == 1
    matrix = sp.coo_matrix(
        (
            np.ones(int(positive.sum()), dtype=np.float32),
            (users[positive], videos[positive]),
        ),
        shape=(num_users, num_videos),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.data = np.log1p(matrix.data).astype(np.float32)

    item_count = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float32)
    item_prior = np.log1p(item_count)

    u, singular, vt = svds(
        matrix,
        k=SVD_DIM,
        which="LM",
        random_state=SEED,
        return_singular_vectors=True,
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order].astype(np.float32)
    u = u[:, order].astype(np.float32)
    vt = vt[order].astype(np.float32)

    root = np.sqrt(np.maximum(singular, 1e-8))
    user_factor = u * root[None, :]
    item_factor = vt.T * root[None, :]
    return user_factor, item_factor, item_prior


def fit_svd_combined(train_split, valid_split):
    users = np.concatenate([
        np.asarray(train_split.user_id, dtype=np.int64),
        np.asarray(valid_split.user_id, dtype=np.int64),
    ])
    videos = np.concatenate([
        np.asarray(train_split.video_id, dtype=np.int64),
        np.asarray(valid_split.video_id, dtype=np.int64),
    ])
    labels = np.concatenate([
        np.asarray(train_split.y, dtype=np.int8),
        np.asarray(valid_split.y, dtype=np.int8),
    ])

    positive = labels == 1
    matrix = sp.coo_matrix(
        (
            np.ones(int(positive.sum()), dtype=np.float32),
            (users[positive], videos[positive]),
        ),
        shape=(num_users, num_videos),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.data = np.log1p(matrix.data).astype(np.float32)

    item_count = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float32)
    item_prior = np.log1p(item_count)

    u, singular, vt = svds(
        matrix,
        k=SVD_DIM,
        which="LM",
        random_state=SEED,
        return_singular_vectors=True,
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order].astype(np.float32)
    u = u[:, order].astype(np.float32)
    vt = vt[order].astype(np.float32)

    root = np.sqrt(np.maximum(singular, 1e-8))
    return (
        u * root[None, :],
        vt.T * root[None, :],
        item_prior,
    )


def svd_predict(factors, split, popularity_weight=0.06):
    user_factor, item_factor, item_prior = factors
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    dot = np.sum(user_factor[users] * item_factor[videos], axis=1)
    return (
        dot + float(popularity_weight) * item_prior[videos]
    ).astype(np.float32)


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float32, copy=False)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

inc_rank = within_user_rank(valid.user_id, inc_valid)
inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)

x_train = make_current_matrix(train)
x_valid = make_current_matrix(valid)
h_train = make_history([], train)
h_valid = make_history([train], valid)

candidate_log = {
    "trusted_incumbent": float(inc_metrics["primary"])
}
findings = {}
blend_grid = np.linspace(0.0, 1.0, 11)

winner = {
    "name": "trusted_incumbent",
    "epoch": 0,
    "alpha": 0.0,
    "valid_scores": inc_valid.copy(),
}
winner_primary = float(inc_metrics["primary"])


def consider_candidate(name, raw_scores, metadata):
    global winner, winner_primary
    raw_metrics = evaluate(valid.user_id, y_valid, raw_scores)
    candidate_log[name] = float(raw_metrics["primary"])

    rank = within_user_rank(valid.user_id, raw_scores)
    best_primary = -np.inf
    best_alpha = 0.0
    best_scores = None

    for alpha in blend_grid:
        blended = (
            (1.0 - float(alpha)) * inc_rank
            + float(alpha) * rank
        )
        metrics = evaluate(valid.user_id, y_valid, blended)
        primary = float(metrics["primary"])
        if primary > best_primary:
            best_primary = primary
            best_alpha = float(alpha)
            best_scores = blended.copy()

    candidate_log[name + "_rank_blend"] = float(best_primary)
    findings[name] = dict(metadata)
    findings[name]["blend_alpha"] = float(best_alpha)
    findings[name]["standalone_primary"] = float(raw_metrics["primary"])

    if best_primary > winner_primary:
        winner_primary = best_primary
        winner = {
            "name": name,
            "epoch": int(metadata.get("best_epoch", 0)),
            "alpha": best_alpha,
            "valid_scores": best_scores,
        }


# Family 1: latent positive-interaction truncated SVD.
svd_factors = fit_svd(train)
svd_valid = svd_predict(svd_factors, valid)
consider_candidate(
    "positive_svd",
    svd_valid,
    {"dimension": SVD_DIM},
)
del svd_valid
if winner["name"] != "positive_svd":
    del svd_factors
gc.collect()

# Families 2 and 3: attention and recurrent sequence predictors.
for family_index, name in enumerate(["din_sequence", "gru_sequence"]):
    scores, best_epoch, epoch_scores = fit_sequence_candidate(
        name,
        x_train,
        h_train,
        y_train,
        x_valid,
        h_valid,
        valid,
        y_valid,
    )
    consider_candidate(
        name,
        scores,
        {
            "best_epoch": int(best_epoch),
            "epoch_primary": [float(x) for x in epoch_scores],
            "history_length": HISTORY_LEN,
        },
    )
    del scores
    gc.collect()

valid_scores = np.asarray(winner["valid_scores"], dtype=np.float32)
metrics = evaluate(valid.user_id, y_valid, valid_scores)

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print("FINDINGS " + json.dumps({
    "selected": winner["name"],
    "selected_epoch": int(winner["epoch"]),
    "selected_blend_alpha": float(winner["alpha"]),
    "details": findings,
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

selected_name = winner["name"]
selected_epoch = int(winner["epoch"])
selected_alpha = float(winner["alpha"])

del x_train, x_valid, h_train, h_valid
gc.collect()

test = load("test")
inc_test = np.load(inc_test_path).astype(np.float32, copy=False)

if selected_name == "trusted_incumbent" or selected_alpha <= 0.0:
    test_scores = inc_test

elif selected_name == "positive_svd":
    if "svd_factors" in globals():
        del svd_factors
    combined_factors = fit_svd_combined(train, valid)
    raw_test = svd_predict(combined_factors, test)
    raw_test_rank = within_user_rank(test.user_id, raw_test)
    inc_test_rank = within_user_rank(test.user_id, inc_test)
    test_scores = (
        (1.0 - selected_alpha) * inc_test_rank
        + selected_alpha * raw_test_rank
    )
    del combined_factors, raw_test, raw_test_rank, inc_test_rank

else:
    x_combined = make_combined_matrix(train, valid)
    y_combined = np.concatenate([
        np.asarray(train.y, dtype=np.int8),
        np.asarray(valid.y, dtype=np.int8),
    ])
    h_combined = make_history([], _CombinedView(train, valid)) if False else None

    class CombinedSplit:
        pass

    combined = CombinedSplit()
    combined.user_id = np.concatenate([
        np.asarray(train.user_id, dtype=np.int64),
        np.asarray(valid.user_id, dtype=np.int64),
    ])
    combined.video_id = np.concatenate([
        np.asarray(train.video_id, dtype=np.int64),
        np.asarray(valid.video_id, dtype=np.int64),
    ])
    combined.time_ms = np.concatenate([
        np.asarray(train.time_ms, dtype=np.int64),
        np.asarray(valid.time_ms, dtype=np.int64),
    ])

    h_combined = make_history([], combined)
    x_test = make_current_matrix(test)
    h_test = make_history([train, valid], test)

    model = make_model(selected_name, SEED + 100)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    xt = torch.from_numpy(x_combined)
    ht = torch.from_numpy(h_combined)
    rng = np.random.default_rng(SEED + 500)

    for _ in range(selected_epoch):
        train_epoch(model, optimizer, xt, ht, y_combined, rng)

    raw_test = predict_sequence(model, x_test, h_test)
    raw_test_rank = within_user_rank(test.user_id, raw_test)
    inc_test_rank = within_user_rank(test.user_id, inc_test)
    test_scores = (
        (1.0 - selected_alpha) * inc_test_rank
        + selected_alpha * raw_test_rank
    )

    del (
        model, optimizer, xt, ht, x_combined, h_combined,
        x_test, h_test, raw_test, raw_test_rank, inc_test_rank
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