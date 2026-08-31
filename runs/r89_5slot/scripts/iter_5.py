import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 7319
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
BATCH = 8192
PRED_BATCH = 32768
EPOCHS = 3
DIN_EPOCHS = 5
HIST_K = 20

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

offsets = []
total_cardinality = 0
for f in FIELDS:
    offsets.append(total_cardinality)
    total_cardinality += int(FEATURE_CARDINALITIES[f])
OFFSETS = np.asarray(offsets, dtype=np.int64)


def make_x(s):
    x = np.empty((len(s.user_id), len(FIELDS)), dtype=np.int64)
    for j, f in enumerate(FIELDS):
        x[:, j] = np.asarray(s.X[f], dtype=np.int64) + OFFSETS[j]
    return x


def make_tasks(s):
    outputs = [np.asarray(s.y, dtype=np.float32)]
    for name in ("is_click", "is_like"):
        if name in s.aux:
            a = np.asarray(s.aux[name])
            a = np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
            outputs.append((a > 0).astype(np.float32))
    return np.column_stack(outputs).astype(np.float32, copy=False)


def last_positive_history(s, eligible_mask=None, k=HIST_K):
    n_users = int(FEATURE_CARDINALITIES["user_id"])
    hist = np.zeros((n_users, k), dtype=np.int64)
    y = np.asarray(s.y, dtype=np.int8)
    if eligible_mask is None:
        take = y > 0
    else:
        take = (y > 0) & np.asarray(eligible_mask, dtype=bool)

    idx = np.flatnonzero(take)
    if idx.size == 0:
        return hist

    users = np.asarray(s.user_id, dtype=np.int64)[idx]
    times = np.asarray(s.time_ms, dtype=np.int64)[idx]
    order = np.lexsort((idx, times, users))
    idx = idx[order]
    users = users[order]

    starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
    lengths = np.diff(np.r_[starts, users.size])
    group_starts = np.repeat(starts, lengths)
    group_lengths = np.repeat(lengths, lengths)
    positions = np.arange(users.size, dtype=np.int64) - group_starts
    from_end = group_lengths - 1 - positions
    keep = from_end < k

    videos = np.asarray(s.video_id, dtype=np.int64)[idx]
    hist[users[keep], from_end[keep]] = videos[keep]
    return hist


def concatenate_splits(a, b):
    class Combined:
        pass

    c = Combined()
    c.user_id = np.concatenate([np.asarray(a.user_id), np.asarray(b.user_id)])
    c.video_id = np.concatenate([np.asarray(a.video_id), np.asarray(b.video_id)])
    c.time_ms = np.concatenate([np.asarray(a.time_ms), np.asarray(b.time_ms)])
    c.date = np.concatenate([np.asarray(a.date), np.asarray(b.date)])
    c.y = np.concatenate([np.asarray(a.y), np.asarray(b.y)])
    c.X = {
        f: np.concatenate([np.asarray(a.X[f]), np.asarray(b.X[f])])
        for f in FIELDS
    }
    common_aux = set(a.aux.keys()).intersection(b.aux.keys())
    c.aux = {
        name: np.concatenate([np.asarray(a.aux[name]), np.asarray(b.aux[name])])
        for name in common_aux
    }
    return c


class ContextLatentMF(nn.Module):
    def __init__(self, rank=32):
        super().__init__()
        self.latent = nn.Embedding(total_cardinality, rank)
        self.linear = nn.Embedding(total_cardinality, 1)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.latent.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        v = self.latent(x)
        user = v[:, 0, :]
        context = v[:, 1:, :].mean(dim=1)
        interaction = (user * context).sum(dim=1)
        wide = self.linear(x).sum(dim=1).squeeze(-1)
        return self.bias + wide + interaction


class MMoE(nn.Module):
    def __init__(self, n_tasks=3, emb_dim=10, n_experts=4):
        super().__init__()
        self.n_tasks = n_tasks
        self.embedding = nn.Embedding(total_cardinality, emb_dim)
        dim = len(FIELDS) * emb_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 48),
                nn.ReLU(),
                nn.Linear(48, 24),
                nn.ReLU(),
            )
            for _ in range(n_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(dim, n_experts) for _ in range(n_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(nn.Linear(24, 16), nn.ReLU(), nn.Linear(16, 1))
            for _ in range(n_tasks)
        ])
        self.wide = nn.Embedding(total_cardinality, n_tasks)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.wide.weight)

    def forward(self, x):
        z = self.embedding(x).flatten(1)
        experts = torch.stack([e(z) for e in self.experts], dim=1)
        wide = self.wide(x).sum(dim=1)
        outputs = []
        for t in range(self.n_tasks):
            gate = torch.softmax(self.gates[t](z), dim=1)
            mixed = (experts * gate.unsqueeze(-1)).sum(dim=1)
            outputs.append(self.towers[t](mixed).squeeze(-1) + wide[:, t])
        return torch.stack(outputs, dim=1)


class DIN(nn.Module):
    def __init__(self, history, emb_dim=12):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, emb_dim)
        self.video_embedding = nn.Embedding(
            int(FEATURE_CARDINALITIES["video_id"]), emb_dim, padding_idx=0
        )
        self.register_buffer(
            "history",
            torch.from_numpy(np.asarray(history, dtype=np.int64)),
            persistent=False,
        )
        dim = len(FIELDS) * emb_dim + emb_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )
        self.wide = nn.Embedding(total_cardinality, 1)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.video_embedding.weight, std=0.02)
        with torch.no_grad():
            self.video_embedding.weight[0].zero_()
        nn.init.zeros_(self.wide.weight)

    def set_history(self, history):
        self.history = torch.from_numpy(
            np.asarray(history, dtype=np.int64)
        ).to(self.embedding.weight.device)

    def forward(self, x):
        user_local = x[:, 0] - int(OFFSETS[0])
        video_local = x[:, 1] - int(OFFSETS[1])
        history_ids = self.history[user_local]

        target = self.video_embedding(video_local)
        history_vectors = self.video_embedding(history_ids)
        mask = history_ids.ne(0)

        logits = (history_vectors * target.unsqueeze(1)).sum(dim=2)
        logits = logits / np.sqrt(float(target.shape[1]))
        logits = logits.masked_fill(~mask, -1e4)
        attention = torch.softmax(logits, dim=1)
        attention = attention * mask.float()
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-6)
        interest = (history_vectors * attention.unsqueeze(-1)).sum(dim=1)

        categorical = self.embedding(x).flatten(1)
        features = torch.cat(
            [categorical, target, interest * target], dim=1
        )
        wide = self.wide(x).sum(dim=1).squeeze(-1)
        return self.mlp(features).squeeze(-1) + wide


def train_model(model, x_np, y_np, epochs, seed):
    torch.manual_seed(seed)
    model.train()
    x = torch.from_numpy(np.asarray(x_np, dtype=np.int64))
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    generator = torch.Generator().manual_seed(seed)
    n = x.shape[0]

    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for st in range(0, n, BATCH):
            idx = order[st:min(st + BATCH, n)]
            pred = model(x[idx])
            target = y[idx]
            if pred.ndim == 1:
                if target.ndim == 2:
                    target = target[:, 0]
            else:
                if target.ndim == 1:
                    target = target[:, None]
                target = target[:, :pred.shape[1]]
            loss = nn.functional.binary_cross_entropy_with_logits(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict(model, x_np):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float32)
    for st in range(0, len(x_np), PRED_BATCH):
        en = min(st + PRED_BATCH, len(x_np))
        pred = model(torch.from_numpy(x_np[st:en]))
        if pred.ndim == 2:
            pred = pred[:, 0]
        result[st:en] = pred.cpu().numpy()
    return result


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((scores, users))
    sorted_users = users[order]
    starts = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1]])
    lengths = np.diff(np.r_[starts, len(order)])
    group_starts = np.repeat(starts, lengths)
    group_lengths = np.repeat(lengths, lengths)
    positions = np.arange(len(order), dtype=np.float64) - group_starts
    denom = np.maximum(group_lengths - 1, 1)
    ranked_sorted = positions / denom
    ranked_sorted[group_lengths == 1] = 0.5
    ranked = np.empty(len(order), dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


train = load("train")
valid = load("valid")
train_x = make_x(train)
valid_x = make_x(valid)
train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
train_tasks = make_tasks(train)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_valid_rank = within_user_rank(valid_users, inc_valid)

candidate_models = {}
candidate_scores = {}

mf = train_model(
    ContextLatentMF(),
    train_x,
    train_y,
    EPOCHS,
    SEED + 1,
)
candidate_models["latent_mf"] = mf
candidate_scores["latent_mf"] = predict(mf, valid_x)

mmoe = train_model(
    MMoE(n_tasks=train_tasks.shape[1]),
    train_x,
    train_tasks,
    EPOCHS,
    SEED + 2,
)
candidate_models["mmoe"] = mmoe
candidate_scores["mmoe"] = predict(mmoe, valid_x)

train_dates = np.asarray(train.date)
din_cutoff = np.sort(np.unique(train_dates))[-7]
din_history_train = last_positive_history(train, train_dates < din_cutoff)
din_target_mask = train_dates >= din_cutoff
din = train_model(
    DIN(din_history_train),
    train_x[din_target_mask],
    train_y[din_target_mask],
    DIN_EPOCHS,
    SEED + 3,
)
din.set_history(last_positive_history(train))
candidate_models["din"] = din
candidate_scores["din"] = predict(din, valid_x)

candidate_log = {}
best_primary = -np.inf
best_name = None
best_alpha = None
best_scores = None
best_raw_scores = None

for name, raw in candidate_scores.items():
    raw_metrics = evaluate(valid_users, valid_y, raw)
    candidate_log[name + "_raw"] = float(raw_metrics["primary"])
    if float(raw_metrics["primary"]) > best_primary:
        best_primary = float(raw_metrics["primary"])
        best_name = name
        best_alpha = 1.0
        best_scores = raw.copy()
        best_raw_scores = raw.copy()

    own_rank = within_user_rank(valid_users, raw)
    family_best = -np.inf
    family_alpha = None
    family_scores = None
    for alpha in np.linspace(0.10, 0.90, 9):
        blended = alpha * own_rank + (1.0 - alpha) * inc_valid_rank
        metric = evaluate(valid_users, valid_y, blended)
        primary = float(metric["primary"])
        if primary > family_best:
            family_best = primary
            family_alpha = float(alpha)
            family_scores = blended.copy()

    candidate_log[name + "_blend"] = family_best
    if family_best > best_primary:
        best_primary = family_best
        best_name = name
        best_alpha = family_alpha
        best_scores = family_scores
        best_raw_scores = raw.copy()

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True), flush=True)

valid_metrics = evaluate(valid_users, valid_y, best_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_alpha < 0.999:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(best_raw_scores, dtype=np.float64),
        )

combined = concatenate_splits(train, valid)
combined_x = make_x(combined)
combined_y = np.asarray(combined.y, dtype=np.float32)

if best_name == "latent_mf":
    final_model = train_model(
        ContextLatentMF(),
        combined_x,
        combined_y,
        EPOCHS,
        SEED + 1,
    )
elif best_name == "mmoe":
    combined_tasks = make_tasks(combined)
    final_model = train_model(
        MMoE(n_tasks=combined_tasks.shape[1]),
        combined_x,
        combined_tasks,
        EPOCHS,
        SEED + 2,
    )
else:
    combined_dates = np.asarray(combined.date)
    combined_cutoff = np.sort(np.unique(combined_dates))[-7]
    base_history = last_positive_history(
        combined, combined_dates < combined_cutoff
    )
    target_mask = combined_dates >= combined_cutoff
    final_model = train_model(
        DIN(base_history),
        combined_x[target_mask],
        combined_y[target_mask],
        DIN_EPOCHS,
        SEED + 3,
    )
    final_model.set_history(last_positive_history(combined))

test = load("test")
test_x = make_x(test)
test_users = np.asarray(test.user_id, dtype=np.int64)
own_test = predict(final_model, test_x)

if best_alpha < 0.999:
    inc_test = np.load(
        os.path.join(shared, "incumbent_test_scores.npy")
    ).astype(np.float64)
    own_test_rank = within_user_rank(test_users, own_test)
    inc_test_rank = within_user_rank(test_users, inc_test)
    test_scores = (
        best_alpha * own_test_rank + (1.0 - best_alpha) * inc_test_rank
    )
else:
    test_scores = own_test

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
result = {
    "primary": float(valid_metrics["primary"]),
    "gauc": float(valid_metrics["gauc"]),
    "ndcg@5": float(valid_metrics["ndcg@5"]),
    "gpu_seconds": elapsed,
}
print("METRICS " + json.dumps(result, separators=(", ", ": ")))