import os
import time
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 2025
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
BATCH_SIZE = 8192

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

train = load("train")
valid = load("valid")
test = load("test")

CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1]).astype(np.int64)
TOTAL_CARD = int(sum(CARDS))


def matrix(split):
    return np.ascontiguousarray(
        np.column_stack([np.asarray(split.X[f], dtype=np.int64) for f in FIELDS]),
        dtype=np.int64,
    )


xtr_np = matrix(train)
xva_np = matrix(valid)
xte_np = matrix(test)
ytr_np = np.asarray(train.y, dtype=np.float32)

xtr = torch.from_numpy(xtr_np)
ytr = torch.from_numpy(ytr_np)

max_train_date = int(np.max(np.asarray(train.date)))
ages = (max_train_date - np.asarray(train.date, dtype=np.int64)).astype(np.float32)


def recency_weights(half_life):
    w = np.exp2(-ages / float(half_life)).astype(np.float32)
    w /= max(float(w.mean()), 1e-8)
    return torch.from_numpy(w)


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-np.minimum(x, 40.0))),
        np.exp(np.maximum(x, -40.0)) / (1.0 + np.exp(np.maximum(x, -40.0))),
    )


def predict(model, x_np, batch_size=32768):
    model.eval()
    out = np.empty(len(x_np), dtype=np.float32)
    with torch.inference_mode():
        for st in range(0, len(x_np), batch_size):
            en = min(st + batch_size, len(x_np))
            z = model(torch.from_numpy(x_np[st:en]))
            if isinstance(z, tuple):
                z = z[0]
            out[st:en] = z.detach().cpu().numpy()
    return out


class FM(nn.Module):
    def __init__(self, rank=16):
        super().__init__()
        self.register_buffer("offsets", torch.tensor(OFFSETS, dtype=torch.long))
        self.linear = nn.Embedding(TOTAL_CARD, 1, sparse=True)
        self.factor = nn.Embedding(TOTAL_CARD, rank, sparse=True)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factor.weight, std=0.01)

    def forward(self, x):
        z = x + self.offsets
        linear = self.linear(z).sum(1).squeeze(-1)
        v = self.factor(z)
        inter = 0.5 * (
            v.sum(1).square() - v.square().sum(1)
        ).sum(1)
        return self.bias + linear + inter

    def sparse_params(self):
        return [self.linear.weight, self.factor.weight]

    def dense_params(self):
        return [self.bias]


class FieldAwareFM(nn.Module):
    def __init__(self, rank=8):
        super().__init__()
        self.n_fields = len(FIELDS)
        self.rank = rank
        self.register_buffer("offsets", torch.tensor(OFFSETS, dtype=torch.long))
        self.linear = nn.Embedding(TOTAL_CARD, 1, sparse=True)
        self.factor = nn.Embedding(
            TOTAL_CARD, self.n_fields * rank, sparse=True
        )
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factor.weight, std=0.015)

    def forward(self, x):
        ids = x + self.offsets
        linear = self.linear(ids).sum(1).squeeze(-1)
        v = self.factor(ids).view(
            len(x), self.n_fields, self.n_fields, self.rank
        )
        inter = torch.zeros(len(x), dtype=v.dtype)
        for i in range(self.n_fields):
            for j in range(i + 1, self.n_fields):
                inter = inter + (v[:, i, j] * v[:, j, i]).sum(-1)
        return self.bias + linear + inter

    def sparse_params(self):
        return [self.linear.weight, self.factor.weight]

    def dense_params(self):
        return [self.bias]


class MatrixFactorization(nn.Module):
    def __init__(self, rank=24):
        super().__init__()
        self.user = nn.Embedding(CARDS[0], rank)
        self.video = nn.Embedding(CARDS[1], rank)
        self.ub = nn.Embedding(CARDS[0], 1)
        self.vb = nn.Embedding(CARDS[1], 1)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.user.weight, std=0.03)
        nn.init.normal_(self.video.weight, std=0.03)
        nn.init.zeros_(self.ub.weight)
        nn.init.zeros_(self.vb.weight)

    def forward(self, x):
        u, v = x[:, 0], x[:, 1]
        return (
            (self.user(u) * self.video(v)).sum(-1)
            + self.ub(u).squeeze(-1)
            + self.vb(v).squeeze(-1)
            + self.bias
        )


class PNN(nn.Module):
    def __init__(self, dim=12):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c, dim) for c in CARDS])
        pairs = len(FIELDS) * (len(FIELDS) - 1) // 2
        self.net = nn.Sequential(
            nn.Linear(len(FIELDS) * dim + pairs, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.wide = nn.ModuleList([nn.Embedding(c, 1) for c in CARDS])
        for e in self.emb:
            nn.init.normal_(e.weight, std=0.02)
        for e in self.wide:
            nn.init.zeros_(e.weight)

    def forward(self, x):
        e = torch.stack([self.emb[j](x[:, j]) for j in range(len(FIELDS))], 1)
        products = []
        for i in range(len(FIELDS)):
            for j in range(i + 1, len(FIELDS)):
                products.append((e[:, i] * e[:, j]).sum(-1, keepdim=True))
        p = torch.cat(products, 1)
        deep = self.net(torch.cat([e.flatten(1), p], 1)).squeeze(-1)
        wide = sum(self.wide[j](x[:, j]).squeeze(-1)
                   for j in range(len(FIELDS)))
        return deep + wide


class AutoInt(nn.Module):
    def __init__(self, dim=16, heads=4):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c, dim) for c in CARDS])
        self.attn1 = nn.MultiheadAttention(
            dim, heads, dropout=0.0, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            dim, heads, dropout=0.0, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.out = nn.Linear(len(FIELDS) * dim, 1)
        self.wide = nn.ModuleList([nn.Embedding(c, 1) for c in CARDS])
        for e in self.emb:
            nn.init.normal_(e.weight, std=0.02)
        for e in self.wide:
            nn.init.zeros_(e.weight)

    def forward(self, x):
        z = torch.stack([self.emb[j](x[:, j]) for j in range(len(FIELDS))], 1)
        a, _ = self.attn1(z, z, z, need_weights=False)
        z = self.norm1(z + F.relu(a))
        a, _ = self.attn2(z, z, z, need_weights=False)
        z = self.norm2(z + F.relu(a))
        deep = self.out(z.flatten(1)).squeeze(-1)
        wide = sum(self.wide[j](x[:, j]).squeeze(-1)
                   for j in range(len(FIELDS)))
        return deep + wide


class MMoE(nn.Module):
    def __init__(self, n_aux, dim=10, n_experts=3):
        super().__init__()
        self.n_aux = n_aux
        self.emb = nn.ModuleList([nn.Embedding(c, dim) for c in CARDS])
        inp = len(FIELDS) * dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inp, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            for _ in range(n_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(inp, n_experts) for _ in range(1 + n_aux)
        ])
        self.heads = nn.ModuleList([
            nn.Linear(32, 1) for _ in range(1 + n_aux)
        ])
        self.wide = nn.ModuleList([nn.Embedding(c, 1) for c in CARDS])
        for e in self.emb:
            nn.init.normal_(e.weight, std=0.02)
        for e in self.wide:
            nn.init.zeros_(e.weight)

    def forward(self, x):
        z = torch.cat([self.emb[j](x[:, j]) for j in range(len(FIELDS))], 1)
        experts = torch.stack([e(z) for e in self.experts], 1)
        outputs = []
        for gate, head in zip(self.gates, self.heads):
            g = torch.softmax(gate(z), dim=1).unsqueeze(-1)
            h = (experts * g).sum(1)
            outputs.append(head(h).squeeze(-1))
        wide = sum(self.wide[j](x[:, j]).squeeze(-1)
                   for j in range(len(FIELDS)))
        outputs[0] = outputs[0] + wide
        return tuple(outputs)


def train_sparse(model, weights, epochs=5, lr=0.001):
    sparse_opt = torch.optim.SparseAdam(model.sparse_params(), lr=lr)
    dense_opt = torch.optim.Adam(model.dense_params(), lr=lr)
    n = len(ytr)
    gen = torch.Generator().manual_seed(SEED + int(weights[0].item() * 1000))
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for st in range(0, n, BATCH_SIZE):
            idx = perm[st:min(st + BATCH_SIZE, n)]
            sparse_opt.zero_grad(set_to_none=True)
            dense_opt.zero_grad(set_to_none=True)
            logits = model(xtr.index_select(0, idx))
            losses = F.binary_cross_entropy_with_logits(
                logits, ytr.index_select(0, idx), reduction="none"
            )
            loss = (losses * weights.index_select(0, idx)).mean()
            loss.backward()
            sparse_opt.step()
            dense_opt.step()
            total += float(loss.detach()) * len(idx)
        print("TRAIN sparse epoch=%d loss=%.6f" %
              (ep + 1, total / n), flush=True)


def train_dense(model, weights, epochs=4, lr=0.001,
                aux_targets=None, aux_weight=0.15):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(ytr)
    gen = torch.Generator().manual_seed(SEED + 37)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for st in range(0, n, BATCH_SIZE):
            idx = perm[st:min(st + BATCH_SIZE, n)]
            opt.zero_grad(set_to_none=True)
            outputs = model(xtr.index_select(0, idx))
            w = weights.index_select(0, idx)
            if isinstance(outputs, tuple):
                main = F.binary_cross_entropy_with_logits(
                    outputs[0], ytr.index_select(0, idx), reduction="none"
                )
                loss = (main * w).mean()
                for k, target in enumerate(aux_targets):
                    al = F.binary_cross_entropy_with_logits(
                        outputs[k + 1],
                        target.index_select(0, idx),
                        reduction="none",
                    )
                    loss = loss + aux_weight * (al * w).mean()
            else:
                losses = F.binary_cross_entropy_with_logits(
                    outputs, ytr.index_select(0, idx), reduction="none"
                )
                loss = (losses * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
        print("TRAIN dense epoch=%d loss=%.6f" %
              (ep + 1, total / n), flush=True)


raw_valid = {}
raw_test = {}

# Main-model recency sweep: identical architecture but materially different
# temporal training distributions.
for hl in (2.0, 4.0, 8.0):
    name = "recency_fm_hl%d" % int(hl)
    model = FM(rank=16)
    train_sparse(model, recency_weights(hl), epochs=5, lr=0.001)
    raw_valid[name] = predict(model, xva_np)
    raw_test[name] = predict(model, xte_np)
    del model

# Field-aware factorization forms each pair with field-specific embeddings.
model = FieldAwareFM(rank=8)
train_sparse(model, recency_weights(4.0), epochs=4, lr=0.001)
raw_valid["field_aware_fm"] = predict(model, xva_np)
raw_test["field_aware_fm"] = predict(model, xte_np)
del model

# Pure user-video latent preference family.
model = MatrixFactorization(rank=24)
train_dense(model, recency_weights(4.0), epochs=5, lr=0.002)
raw_valid["latent_mf"] = predict(model, xva_np)
raw_test["latent_mf"] = predict(model, xte_np)
del model

# Explicit neural product interactions.
model = PNN(dim=12)
train_dense(model, recency_weights(4.0), epochs=4, lr=0.001)
raw_valid["pnn"] = predict(model, xva_np)
raw_test["pnn"] = predict(model, xte_np)
del model

# Attention-based feature interaction.
model = AutoInt(dim=16, heads=4)
train_dense(model, recency_weights(4.0), epochs=4, lr=0.001)
raw_valid["autoint"] = predict(model, xva_np)
raw_test["autoint"] = predict(model, xte_np)
del model

# Multi-task outcomes are used only as training targets, never as row inputs.
aux_names = [n for n in ("is_click", "is_like") if n in train.aux]
if aux_names:
    aux_targets = [
        torch.from_numpy(np.asarray(train.aux[n], dtype=np.float32))
        for n in aux_names
    ]
    model = MMoE(len(aux_names), dim=10, n_experts=3)
    train_dense(
        model,
        recency_weights(4.0),
        epochs=4,
        lr=0.001,
        aux_targets=aux_targets,
        aux_weight=0.12,
    )
    raw_valid["mmoe_" + "_".join(aux_names)] = predict(model, xva_np)
    raw_test["mmoe_" + "_".join(aux_names)] = predict(model, xte_np)
    del model

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_va_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_te_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_va = np.load(inc_va_path)
inc_te = np.load(inc_te_path)

candidate_scores = {}
candidate_test = {}
candidate_raw_name = {}

inc_va_prob = sigmoid_np(inc_va)
inc_te_prob = sigmoid_np(inc_te)

for name in raw_valid:
    va = raw_valid[name]
    te = raw_test[name]
    candidate_scores[name] = va
    candidate_test[name] = te
    candidate_raw_name[name] = name

    va_prob = sigmoid_np(va)
    te_prob = sigmoid_np(te)
    for own_weight in (0.25, 0.50, 0.75):
        cname = "%s_blend_%.2f" % (name, own_weight)
        candidate_scores[cname] = (
            own_weight * va_prob + (1.0 - own_weight) * inc_va_prob
        )
        candidate_test[cname] = (
            own_weight * te_prob + (1.0 - own_weight) * inc_te_prob
        )
        candidate_raw_name[cname] = name

candidate_primary = {}
candidate_metrics = {}
best_name = None
best_primary = -1.0

for name, scores in candidate_scores.items():
    m = evaluate(valid.user_id, valid.y, scores)
    candidate_metrics[name] = m
    candidate_primary[name] = float(m["primary"])
    if float(m["primary"]) > best_primary:
        best_primary = float(m["primary"])
        best_name = name

print("CANDIDATES " + json.dumps(candidate_primary, sort_keys=True), flush=True)

valid_scores = candidate_scores[best_name]
test_scores = candidate_test[best_name]
metrics = candidate_metrics[best_name]

print(
    "FINDINGS selected=%s raw_family=%s aux_targets=%s" %
    (best_name, candidate_raw_name[best_name], ",".join(aux_names)),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if "blend_" in best_name:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(
                raw_valid[candidate_raw_name[best_name]], dtype=np.float64
            ),
        )

elapsed = time.time() - START
print(
    "METRICS " + json.dumps({
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)