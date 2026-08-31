import os
import gc
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 314159
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
]
N_FIELDS = len(FIELDS)
EMBED_DIM = 12
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768
MAX_EPOCHS = 4
LR = 0.002
WEIGHT_DECAY = 1e-6


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_everything(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_matrix(split):
    n = len(split.user_id)
    x = np.empty((n, N_FIELDS), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:, j] = (
            np.asarray(split.X[field], dtype=np.int64) + offsets[j]
        )
    return x


def make_combined_matrix(a, b):
    na = len(a.user_id)
    nb = len(b.user_id)
    x = np.empty((na + nb, N_FIELDS), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:na, j] = np.asarray(a.X[field], dtype=np.int64) + offsets[j]
        x[na:, j] = np.asarray(b.X[field], dtype=np.int64) + offsets[j]
    return x


class CommonEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def wide_score(self, x):
        return self.bias + self.linear(x).sum(dim=1).squeeze(-1)


class PNN(CommonEmbedding):
    """Product-based network with explicit pairwise inner products."""

    def __init__(self):
        super().__init__()
        pair_count = N_FIELDS * (N_FIELDS - 1) // 2
        input_dim = N_FIELDS * EMBED_DIM + pair_count
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        pairs_i = []
        pairs_j = []
        for i in range(N_FIELDS):
            for j in range(i + 1, N_FIELDS):
                pairs_i.append(i)
                pairs_j.append(j)
        self.register_buffer(
            "pairs_i", torch.tensor(pairs_i, dtype=torch.long)
        )
        self.register_buffer(
            "pairs_j", torch.tensor(pairs_j, dtype=torch.long)
        )

    def forward(self, x):
        v = self.embedding(x)
        products = (
            v[:, self.pairs_i, :] * v[:, self.pairs_j, :]
        ).sum(dim=2)
        features = torch.cat([v.flatten(1), products], dim=1)
        return self.wide_score(x) + self.mlp(features).squeeze(-1)


class DCN(CommonEmbedding):
    """Deep & Cross Network with explicit bounded-degree feature crosses."""

    def __init__(self):
        super().__init__()
        dim = N_FIELDS * EMBED_DIM
        self.cross_w = nn.ParameterList([
            nn.Parameter(torch.empty(dim)) for _ in range(3)
        ])
        self.cross_b = nn.ParameterList([
            nn.Parameter(torch.zeros(dim)) for _ in range(3)
        ])
        for w in self.cross_w:
            nn.init.normal_(w, mean=0.0, std=0.02)

        self.deep = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(dim + 64, 1)

    def forward(self, x):
        x0 = self.embedding(x).flatten(1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scale = (cross * w).sum(dim=1, keepdim=True)
            cross = x0 * scale + b + cross
        deep = self.deep(x0)
        joined = torch.cat([cross, deep], dim=1)
        return self.wide_score(x) + self.output(joined).squeeze(-1)


class AutoInt(CommonEmbedding):
    """Field-level self-attention interaction model."""

    def __init__(self):
        super().__init__()
        self.attention1 = nn.MultiheadAttention(
            EMBED_DIM, num_heads=3, batch_first=True
        )
        self.attention2 = nn.MultiheadAttention(
            EMBED_DIM, num_heads=3, batch_first=True
        )
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        self.output = nn.Sequential(
            nn.Linear(N_FIELDS * EMBED_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        v = self.embedding(x)
        h, _ = self.attention1(v, v, v, need_weights=False)
        h = self.norm1(v + h)
        h2, _ = self.attention2(h, h, h, need_weights=False)
        h = self.norm2(h + h2)
        return self.wide_score(x) + self.output(h.flatten(1)).squeeze(-1)


def make_model(name, seed):
    seed_everything(seed)
    if name == "pnn_bpr":
        return PNN()
    if name == "dcn_bpr":
        return DCN()
    if name == "autoint_bpr":
        return AutoInt()
    raise ValueError(name)


class PairSampler:
    """
    Stores positive rows belonging to users with at least one negative and
    samples a logged negative from the same user without Python user loops.
    """

    def __init__(self, user_ids, labels, seed):
        user_ids = np.asarray(user_ids, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.int8)

        max_user = int(user_ids.max()) + 1
        negative_rows = np.flatnonzero(labels == 0)
        neg_sort = np.argsort(
            user_ids[negative_rows], kind="stable"
        )
        self.negative_rows = negative_rows[neg_sort].astype(
            np.int64, copy=False
        )

        neg_counts = np.bincount(
            user_ids[self.negative_rows], minlength=max_user
        ).astype(np.int64)
        neg_starts = np.zeros(max_user, dtype=np.int64)
        if max_user > 1:
            neg_starts[1:] = np.cumsum(neg_counts[:-1])

        positive_rows = np.flatnonzero(labels == 1)
        positive_users = user_ids[positive_rows]
        usable = neg_counts[positive_users] > 0

        self.positive_rows = positive_rows[usable].astype(
            np.int64, copy=False
        )
        self.positive_users = positive_users[usable].astype(
            np.int64, copy=False
        )
        self.neg_counts = neg_counts
        self.neg_starts = neg_starts
        self.rng = np.random.default_rng(seed)

    def sample_epoch(self):
        users = self.positive_users
        counts = self.neg_counts[users]
        offsets = (
            self.rng.random(len(users)) * counts
        ).astype(np.int64)
        negative_rows = self.negative_rows[
            self.neg_starts[users] + offsets
        ]
        order = self.rng.permutation(len(self.positive_rows))
        return self.positive_rows[order], negative_rows[order]


def train_bpr_epoch(model, optimizer, x_tensor, sampler):
    model.train()
    pos_rows, neg_rows = sampler.sample_epoch()
    total_loss = 0.0
    n = len(pos_rows)

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        pi = torch.from_numpy(pos_rows[start:end])
        ni = torch.from_numpy(neg_rows[start:end])

        optimizer.zero_grad(set_to_none=True)
        pos_score = model(x_tensor[pi])
        neg_score = model(x_tensor[ni])
        loss = nn.functional.softplus(
            -(pos_score - neg_score)
        ).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += float(loss.detach()) * (end - start)

    return total_loss / max(n, 1)


@torch.no_grad()
def predict(model, x):
    model.eval()
    x_tensor = torch.from_numpy(x)
    scores = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(x))
        scores[start:end] = (
            model(x_tensor[start:end])
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
    return scores


def within_user_rank(user_ids, scores):
    """
    Ascending rank in [0,1] within each user's logged impressions.
    The row index provides deterministic tie breaking.
    """
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
    positions = np.arange(n, dtype=np.int64) - starts

    _, counts = np.unique(sorted_users, return_counts=True)
    denominators = np.repeat(
        np.maximum(counts - 1, 1), counts
    ).astype(np.float64)

    ranked = np.empty(n, dtype=np.float32)
    ranked[order] = (positions / denominators).astype(np.float32)
    return ranked


def fit_family(name, x_train_t, sampler, x_valid, valid, y_valid):
    model = make_model(name, SEED + 1000)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    best_primary = -np.inf
    best_epoch = 1
    best_scores = None
    epoch_scores = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_bpr_epoch(model, optimizer, x_train_t, sampler)
        scores = predict(model, x_valid)
        metrics = evaluate(valid.user_id, y_valid, scores)
        primary = float(metrics["primary"])
        epoch_scores.append(primary)

        if primary > best_primary:
            best_primary = primary
            best_epoch = epoch
            best_scores = scores.copy()

    del model, optimizer
    gc.collect()
    return best_scores, best_epoch, best_primary, epoch_scores


train = load("train")
valid = load("valid")

x_train = make_matrix(train)
x_valid = make_matrix(valid)
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)
x_train_t = torch.from_numpy(x_train)

shared_dir = os.environ.get("SHARED_ARTIFACTS")
if not shared_dir:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared_dir, "incumbent_valid_scores.npy")
).astype(np.float32, copy=False)
inc_test_path = os.path.join(
    shared_dir, "incumbent_test_scores.npy"
)

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)

family_names = ["pnn_bpr", "dcn_bpr", "autoint_bpr"]
blend_grid = np.linspace(0.0, 1.0, 11)

candidate_log = {
    "trusted_incumbent": float(inc_metrics["primary"])
}
findings = {}
winner = {
    "family": None,
    "epoch": 0,
    "alpha": 0.0,
    "valid_scores": inc_valid.copy(),
}
winner_primary = float(inc_metrics["primary"])

for family_index, name in enumerate(family_names):
    sampler = PairSampler(
        train.user_id,
        y_train,
        SEED + 2000 + family_index,
    )
    scores, best_epoch, standalone_primary, epoch_primaries = (
        fit_family(
            name,
            x_train_t,
            sampler,
            x_valid,
            valid,
            y_valid,
        )
    )

    candidate_log[name] = float(standalone_primary)
    model_rank = within_user_rank(valid.user_id, scores)

    best_blend_primary = -np.inf
    best_alpha = 0.0
    best_blend_scores = inc_valid_rank

    for alpha in blend_grid:
        blended = (
            float(alpha) * model_rank
            + (1.0 - float(alpha)) * inc_valid_rank
        )
        blend_metrics = evaluate(
            valid.user_id, y_valid, blended
        )
        primary = float(blend_metrics["primary"])
        if primary > best_blend_primary:
            best_blend_primary = primary
            best_alpha = float(alpha)
            best_blend_scores = blended.copy()

    candidate_log[name + "_rank_blend"] = float(
        best_blend_primary
    )
    findings[name] = {
        "best_epoch": int(best_epoch),
        "epoch_primary": [
            float(v) for v in epoch_primaries
        ],
        "blend_weight": float(best_alpha),
    }

    if best_blend_primary > winner_primary:
        winner_primary = best_blend_primary
        winner = {
            "family": name,
            "epoch": int(best_epoch),
            "alpha": float(best_alpha),
            "valid_scores": best_blend_scores,
        }

    del sampler, scores, model_rank
    gc.collect()

valid_scores = np.asarray(
    winner["valid_scores"], dtype=np.float32
)
metrics = evaluate(valid.user_id, y_valid, valid_scores)

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print("FINDINGS " + json.dumps({
    "selected_family": winner["family"],
    "selected_epoch": int(winner["epoch"]),
    "selected_rank_blend_weight": float(winner["alpha"]),
    "families": findings,
}, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

selected_family = winner["family"]
selected_epoch = int(winner["epoch"])
selected_alpha = float(winner["alpha"])

del x_train_t, x_train, x_valid
gc.collect()

test = load("test")
inc_test = np.load(inc_test_path).astype(
    np.float32, copy=False
)

if selected_family is None or selected_alpha <= 0.0:
    test_scores = inc_test
else:
    x_combined = make_combined_matrix(train, valid)
    y_combined = np.concatenate([
        np.asarray(train.y, dtype=np.int8),
        np.asarray(valid.y, dtype=np.int8),
    ])
    combined_users = np.concatenate([
        np.asarray(train.user_id, dtype=np.int64),
        np.asarray(valid.user_id, dtype=np.int64),
    ])
    x_test = make_matrix(test)

    combined_x_t = torch.from_numpy(x_combined)
    combined_sampler = PairSampler(
        combined_users,
        y_combined,
        SEED + 2000 + family_names.index(selected_family),
    )

    combined_model = make_model(
        selected_family, SEED + 1000
    )
    combined_optimizer = torch.optim.Adam(
        combined_model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    for _ in range(selected_epoch):
        train_bpr_epoch(
            combined_model,
            combined_optimizer,
            combined_x_t,
            combined_sampler,
        )

    new_test_scores = predict(combined_model, x_test)
    new_test_rank = within_user_rank(
        test.user_id, new_test_scores
    )
    inc_test_rank = within_user_rank(
        test.user_id, inc_test
    )
    test_scores = (
        selected_alpha * new_test_rank
        + (1.0 - selected_alpha) * inc_test_rank
    )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START_TIME
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))