import os
import time
import json
import random
import gc
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 18473
FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour",
]
EMBED_DIM = 8
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768
MAX_EPOCHS = 3
HALF_LIFE_DAYS = 10.0
AUX_WEIGHTS = (1.0, 0.30, 0.15)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_all(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))
n_fields = len(FIELDS)
flat_dim = n_fields * EMBED_DIM


def make_matrix(split):
    n = len(split.user_id)
    x = np.empty((n, n_fields), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:, j] = np.asarray(split.X[field], dtype=np.int64) + offsets[j]
    return x


def make_combined_matrix(a, b):
    n1 = len(a.user_id)
    n2 = len(b.user_id)
    x = np.empty((n1 + n2, n_fields), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:n1, j] = np.asarray(a.X[field], dtype=np.int64) + offsets[j]
        x[n1:, j] = np.asarray(b.X[field], dtype=np.int64) + offsets[j]
    return x


def clean_binary(values):
    x = np.asarray(values, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(x, 0.0, 1.0)


def make_targets(split):
    long_view = clean_binary(split.y)
    click = clean_binary(split.aux["is_click"])
    like = clean_binary(split.aux["is_like"])
    return np.column_stack([long_view, click, like]).astype(np.float32)


def make_combined_targets(a, b):
    return np.concatenate([make_targets(a), make_targets(b)], axis=0)


def sample_weights(users, dates):
    users = np.asarray(users, dtype=np.int64)
    dates = np.asarray(dates, dtype=np.int64)
    size = max(int(users.max(initial=0)) + 1, cards[0])
    counts = np.bincount(users, minlength=size).astype(np.float32)
    user_weight = 1.0 / np.sqrt(np.maximum(counts[users], 1.0))

    day = (dates % 100).astype(np.int32)
    age = int(day.max()) - day
    recency = np.exp2(-age.astype(np.float32) / HALF_LIFE_DAYS)

    w = user_weight * recency
    w /= max(float(w.mean()), 1e-8)
    return w.astype(np.float32)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores.copy()

    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, users))
    sorted_users = users[order]

    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.maximum.accumulate(
        np.where(starts_flag, np.arange(n, dtype=np.int64), 0)
    )
    rank_sorted = np.arange(n, dtype=np.int64) - starts

    _, counts = np.unique(sorted_users, return_counts=True)
    sizes = np.repeat(counts, counts)
    rank_sorted = rank_sorted.astype(np.float64) / np.maximum(sizes - 1, 1)
    rank_sorted[sizes == 1] = 0.5

    result = np.empty(n, dtype=np.float64)
    result[order] = rank_sorted
    return result


class InputEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x):
        return self.embedding(x).reshape(x.shape[0], -1)


class SharedBottom(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = InputEmbedding()
        self.bottom = nn.Sequential(
            nn.Linear(flat_dim, 72),
            nn.ReLU(),
            nn.Linear(72, 40),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(40, 1) for _ in range(3)])

    def forward(self, x):
        h = self.bottom(self.input(x))
        return torch.cat([head(h) for head in self.heads], dim=1)


class MMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = InputEmbedding()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(flat_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 40),
                nn.ReLU(),
            )
            for _ in range(4)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(flat_dim, 4) for _ in range(3)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(40, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(3)
        ])

    def forward(self, x):
        z = self.input(x)
        experts = torch.stack([expert(z) for expert in self.experts], dim=1)
        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            weights = torch.softmax(gate(z), dim=1).unsqueeze(-1)
            mixed = torch.sum(experts * weights, dim=1)
            outputs.append(tower(mixed))
        return torch.cat(outputs, dim=1)


class PLE(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = InputEmbedding()
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(flat_dim, 56),
                nn.ReLU(),
                nn.Linear(56, 36),
                nn.ReLU(),
            )
            for _ in range(2)
        ])
        self.private_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(flat_dim, 56),
                nn.ReLU(),
                nn.Linear(56, 36),
                nn.ReLU(),
            )
            for _ in range(3)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(flat_dim, 3) for _ in range(3)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(36, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(3)
        ])

    def forward(self, x):
        z = self.input(x)
        shared = [expert(z) for expert in self.shared_experts]
        outputs = []
        for task in range(3):
            available = torch.stack(
                [shared[0], shared[1], self.private_experts[task](z)], dim=1
            )
            weights = torch.softmax(self.gates[task](z), dim=1).unsqueeze(-1)
            mixed = torch.sum(available * weights, dim=1)
            outputs.append(self.towers[task](mixed))
        return torch.cat(outputs, dim=1)


def make_model(name):
    if name == "shared_bottom":
        return SharedBottom()
    if name == "mmoe":
        return MMoE()
    if name == "ple":
        return PLE()
    raise ValueError(name)


@torch.no_grad()
def predict(model, x):
    model.eval()
    xt = torch.from_numpy(x)
    result = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(x))
        logits = model(xt[start:end])
        result[start:end] = logits[:, 0].cpu().numpy()
    return result


def train_epoch(model, optimizer, xt, yt, wt, rng):
    model.train()
    n = xt.shape[0]
    order = rng.permutation(n)
    task_weights = torch.tensor(AUX_WEIGHTS, dtype=torch.float32)

    total_loss = 0.0
    seen = 0
    for start in range(0, n, BATCH_SIZE):
        idx_np = order[start:start + BATCH_SIZE]
        idx = torch.from_numpy(idx_np)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xt[idx])
        targets = yt[idx]
        rows = wt[idx]

        element_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        per_row = (element_loss * task_weights).sum(dim=1)
        loss = (per_row * rows).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += float(loss.detach()) * len(idx_np)
        seen += len(idx_np)

    return total_loss / max(seen, 1)


def fit_family(name, x_train, targets, weights, x_valid, valid_users, valid_y):
    family_seed = {
        "shared_bottom": SEED + 101,
        "mmoe": SEED + 307,
        "ple": SEED + 509,
    }[name]
    seed_all(family_seed)

    model = make_model(name)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0018, weight_decay=2e-6
    )

    xt = torch.from_numpy(x_train)
    yt = torch.from_numpy(targets)
    wt = torch.from_numpy(weights)
    rng = np.random.default_rng(family_seed + 9001)

    best_primary = -np.inf
    best_epoch = 1
    best_scores = None
    epoch_scores = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_epoch(model, optimizer, xt, yt, wt, rng)
        scores = predict(model, x_valid)
        metric = evaluate(valid_users, valid_y, scores)
        primary = float(metric["primary"])
        epoch_scores.append(primary)
        if primary > best_primary:
            best_primary = primary
            best_epoch = epoch
            best_scores = scores.copy()

    del model, optimizer, xt, yt, wt
    gc.collect()
    return best_scores, best_epoch, best_primary, epoch_scores


def refit_family(name, epochs, x, targets, weights, x_test):
    family_seed = {
        "shared_bottom": SEED + 101,
        "mmoe": SEED + 307,
        "ple": SEED + 509,
    }[name]
    seed_all(family_seed)

    model = make_model(name)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0018, weight_decay=2e-6
    )
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(targets)
    wt = torch.from_numpy(weights)
    rng = np.random.default_rng(family_seed + 9001)

    for _ in range(epochs):
        train_epoch(model, optimizer, xt, yt, wt, rng)

    scores = predict(model, x_test)
    del model, optimizer, xt, yt, wt
    gc.collect()
    return scores


train = load("train")
valid = load("valid")
valid_y = np.asarray(valid.y, dtype=np.int8)

shared_dir = os.environ.get("SHARED_ARTIFACTS")
if not shared_dir:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared_dir, "incumbent_valid_scores.npy")
).astype(np.float64, copy=False)
inc_test_path = os.path.join(shared_dir, "incumbent_test_scores.npy")

x_train = make_matrix(train)
x_valid = make_matrix(valid)
train_targets = make_targets(train)
train_weights = sample_weights(train.user_id, train.date)

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
candidate_log = {}
family_results = {}

inc_metric = evaluate(valid.user_id, valid_y, inc_valid)
candidate_log["incumbent"] = float(inc_metric["primary"])

for family in ["shared_bottom", "mmoe", "ple"]:
    scores, best_epoch, standalone_primary, epoch_scores = fit_family(
        family,
        x_train,
        train_targets,
        train_weights,
        x_valid,
        valid.user_id,
        valid_y,
    )
    family_rank = within_user_rank(valid.user_id, scores)
    family_results[family] = {
        "scores": scores,
        "rank": family_rank,
        "epoch": best_epoch,
        "standalone": standalone_primary,
    }
    candidate_log[family] = standalone_primary
    print(
        "FINDINGS " + json.dumps({
            "family": family,
            "epoch_primary": [round(v, 6) for v in epoch_scores],
            "selected_epoch": int(best_epoch),
        }, sort_keys=True)
    )

alphas = [0.20, 0.40, 0.60, 0.80]
best_name = "incumbent"
best_family = None
best_alpha = 0.0
best_scores = inc_valid.copy()
best_metric = inc_metric

for family, result in family_results.items():
    raw_metric = evaluate(valid.user_id, valid_y, result["scores"])
    if float(raw_metric["primary"]) > float(best_metric["primary"]):
        best_name = family
        best_family = family
        best_alpha = 1.0
        best_scores = result["scores"].astype(np.float64)
        best_metric = raw_metric

    for alpha in alphas:
        fused = (
            alpha * result["rank"]
            + (1.0 - alpha) * inc_valid_rank
        )
        metric = evaluate(valid.user_id, valid_y, fused)
        name = "{}_rankblend_{:.1f}".format(family, alpha)
        candidate_log[name] = float(metric["primary"])
        if float(metric["primary"]) > float(best_metric["primary"]):
            best_name = name
            best_family = family
            best_alpha = alpha
            best_scores = fused
            best_metric = metric

print("CANDIDATES " + json.dumps(
    {k: round(float(v), 7) for k, v in candidate_log.items()},
    sort_keys=True
))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )

if out:
    test = load("test")
    inc_test = np.load(inc_test_path).astype(np.float64, copy=False)

    if best_family is None:
        test_scores = inc_test
    else:
        selected_epoch = int(family_results[best_family]["epoch"])

        x_combined = make_combined_matrix(train, valid)
        combined_targets = make_combined_targets(train, valid)
        combined_users = np.concatenate([
            np.asarray(train.user_id, dtype=np.int64),
            np.asarray(valid.user_id, dtype=np.int64),
        ])
        combined_dates = np.concatenate([
            np.asarray(train.date, dtype=np.int64),
            np.asarray(valid.date, dtype=np.int64),
        ])
        combined_weights = sample_weights(combined_users, combined_dates)
        x_test = make_matrix(test)

        family_test = refit_family(
            best_family,
            selected_epoch,
            x_combined,
            combined_targets,
            combined_weights,
            x_test,
        )

        if best_alpha >= 1.0:
            test_scores = family_test
        else:
            family_test_rank = within_user_rank(test.user_id, family_test)
            incumbent_test_rank = within_user_rank(test.user_id, inc_test)
            test_scores = (
                best_alpha * family_test_rank
                + (1.0 - best_alpha) * incumbent_test_rank
            )

        del x_combined, combined_targets, combined_weights, x_test
        gc.collect()

    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print(
    "FINDINGS " + json.dumps({
        "selected": best_name,
        "family": best_family,
        "alpha": float(best_alpha),
        "epoch": (
            None if best_family is None
            else int(family_results[best_family]["epoch"])
        ),
    }, sort_keys=True)
)

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.4f}'
    % (
        float(best_metric["primary"]),
        float(best_metric["gauc"]),
        float(best_metric["ndcg@5"]),
        elapsed,
    )
)