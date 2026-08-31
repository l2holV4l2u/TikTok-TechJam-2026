import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 7319
BATCH = 8192
PRED_BATCH = 32768

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

# Every family receives exactly the same fields. This mixes the three important
# identities with relatively compact context/content fields, rather than all 37
# potentially non-stationary fields.
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "user_active_degree",
    "onehot_feat3",
    "onehot_feat8",
]
CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
N_FIELDS = len(FIELDS)

train = load("train")
valid = load("valid")
test = load("test")


def make_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[f], dtype=np.int64) for f in FIELDS
        ]),
        dtype=np.int64,
    )


xtr_np = make_matrix(train)
xva_np = make_matrix(valid)
xte_np = make_matrix(test)
ytr_np = np.asarray(train.y, dtype=np.float32)

xtr = torch.from_numpy(xtr_np)
ytr = torch.from_numpy(ytr_np)

# The established temporal recipe is held fixed across families so the
# comparison isolates how predictions are formed.
last_date = int(np.max(np.asarray(train.date, dtype=np.int64)))
age = (last_date - np.asarray(train.date, dtype=np.int64)).astype(np.float32)
w_np = np.exp2(-age / 4.0).astype(np.float32)
w_np /= float(np.mean(w_np))
wtr = torch.from_numpy(w_np)


class FieldEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c, dim) for c in CARDS])
        self.wide = nn.ModuleList([nn.Embedding(c, 1) for c in CARDS])
        for e in self.emb:
            nn.init.normal_(e.weight, mean=0.0, std=0.02)
        for e in self.wide:
            nn.init.zeros_(e.weight)

    def dense(self, x):
        return torch.stack(
            [self.emb[j](x[:, j]) for j in range(N_FIELDS)], dim=1
        )

    def linear(self, x):
        terms = [
            self.wide[j](x[:, j]).squeeze(-1) for j in range(N_FIELDS)
        ]
        return torch.stack(terms, dim=1).sum(dim=1)


class WideLinear(nn.Module):
    """Purely additive categorical logistic regression."""

    def __init__(self):
        super().__init__()
        self.wide = nn.ModuleList([nn.Embedding(c, 1) for c in CARDS])
        self.bias = nn.Parameter(torch.zeros(()))
        for e in self.wide:
            nn.init.zeros_(e.weight)

    def forward(self, x):
        z = torch.stack(
            [self.wide[j](x[:, j]).squeeze(-1) for j in range(N_FIELDS)],
            dim=1,
        ).sum(dim=1)
        return z + self.bias


class XDeepFM(nn.Module):
    """
    CIN explicitly creates bounded-degree vector-wise interactions. Each layer
    forms all outer field/channel combinations independently at each embedding
    coordinate and projects them to new interaction channels.
    """

    def __init__(self, dim=8, cin_sizes=(20, 16)):
        super().__init__()
        self.base = FieldEmbeddings(dim)
        self.dim = dim
        self.cin = nn.ModuleList()
        previous = N_FIELDS
        for width in cin_sizes:
            layer = nn.Linear(N_FIELDS * previous, width)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.cin.append(layer)
            previous = width

        deep_in = N_FIELDS * dim
        self.deep = nn.Sequential(
            nn.Linear(deep_in, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.cin_out = nn.Linear(sum(cin_sizes), 1)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        x0 = self.base.dense(x)       # B,F,D
        h = x0
        pooled = []
        for layer in self.cin:
            # B,F,H,D, then each embedding coordinate gets the same projection.
            outer = torch.einsum("bfd,bhd->bfhd", x0, h)
            b, f, old_h, d = outer.shape
            q = outer.reshape(b, f * old_h, d).transpose(1, 2)
            q = layer(q)             # B,D,new_H
            h = F.relu(q.transpose(1, 2))
            pooled.append(h.sum(dim=2))

        cin_score = self.cin_out(torch.cat(pooled, dim=1)).squeeze(-1)
        deep_score = self.deep(x0.flatten(1)).squeeze(-1)
        return self.base.linear(x) + cin_score + deep_score + self.bias


class FiBiNET(nn.Module):
    """
    Squeeze-excitation estimates row-specific field importance, after which
    learned field-pair bilinear maps form interactions.
    """

    def __init__(self, dim=8):
        super().__init__()
        self.base = FieldEmbeddings(dim)
        hidden = max(4, N_FIELDS // 2)
        self.se = nn.Sequential(
            nn.Linear(N_FIELDS, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_FIELDS),
            nn.Sigmoid(),
        )
        self.pairs = [
            (i, j) for i in range(N_FIELDS) for j in range(i + 1, N_FIELDS)
        ]
        self.bilinear = nn.Parameter(
            torch.empty(len(self.pairs), dim, dim)
        )
        nn.init.xavier_uniform_(self.bilinear)

        pair_dim = len(self.pairs) * dim
        self.net = nn.Sequential(
            nn.Linear(pair_dim + N_FIELDS * dim, 128),
            nn.ReLU(),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        e = self.base.dense(x)
        importance = self.se(e.mean(dim=2)).unsqueeze(-1)
        e = e * importance

        interactions = []
        for p, (i, j) in enumerate(self.pairs):
            left = torch.matmul(e[:, i, :], self.bilinear[p])
            interactions.append(left * e[:, j, :])
        pair_tensor = torch.cat(interactions, dim=1)
        score = self.net(
            torch.cat([e.flatten(1), pair_tensor], dim=1)
        ).squeeze(-1)
        return self.base.linear(x) + score + self.bias


class LowRankCross(nn.Module):
    """One DCNv2 low-rank matrix cross layer."""

    def __init__(self, width, rank):
        super().__init__()
        self.u = nn.Linear(width, rank, bias=False)
        self.v = nn.Linear(rank, width, bias=False)
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.xavier_uniform_(self.u.weight)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, x0, x):
        # x_{l+1} = x0 * (W_l x_l + b_l) + x_l
        return x0 * (self.v(F.relu(self.u(x))) + self.bias) + x


class DCNV2(nn.Module):
    """
    Low-rank matrix cross layers can learn coordinate-specific interactions,
    unlike the scalar vector-cross operator of the original DCN.
    """

    def __init__(self, dim=8, rank=24):
        super().__init__()
        self.base = FieldEmbeddings(dim)
        width = N_FIELDS * dim
        self.cross = nn.ModuleList([
            LowRankCross(width, rank),
            LowRankCross(width, rank),
            LowRankCross(width, rank),
        ])
        self.deep = nn.Sequential(
            nn.Linear(width, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
        )
        self.out = nn.Linear(width + 48, 1)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        e = self.base.dense(x)
        x0 = e.flatten(1)
        cross = x0
        for layer in self.cross:
            cross = layer(x0, cross)
        deep = self.deep(x0)
        score = self.out(torch.cat([cross, deep], dim=1)).squeeze(-1)
        return self.base.linear(x) + score + self.bias


def fit_model(model, name, epochs=3, lr=1.0e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(ytr)
    generator = torch.Generator().manual_seed(
        SEED + sum(ord(c) for c in name)
    )

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=generator)
        loss_sum = 0.0

        for st in range(0, n, BATCH):
            idx = perm[st:min(st + BATCH, n)]
            xb = xtr.index_select(0, idx)
            yb = ytr.index_select(0, idx)
            wb = wtr.index_select(0, idx)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            per_row = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = (per_row * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss.detach()) * len(idx)

        print(
            "TRAIN %s epoch=%d loss=%.6f" %
            (name, epoch + 1, loss_sum / n),
            flush=True,
        )


def predict(model, x_np):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float32)
    with torch.inference_mode():
        for st in range(0, len(x_np), PRED_BATCH):
            en = min(st + PRED_BATCH, len(x_np))
            result[st:en] = (
                model(torch.from_numpy(x_np[st:en]))
                .detach().cpu().numpy()
            )
    return result


families = [
    ("wide_linear", WideLinear(), 3, 0.0020),
    ("xdeepfm", XDeepFM(dim=8, cin_sizes=(20, 16)), 3, 0.0010),
    ("fibinet", FiBiNET(dim=8), 3, 0.0010),
    ("dcnv2", DCNV2(dim=8, rank=24), 3, 0.0010),
]

raw_valid = {}
raw_test = {}

for name, model, epochs, lr in families:
    fit_model(model, name, epochs=epochs, lr=lr)
    raw_valid[name] = predict(model, xva_np)
    raw_test[name] = predict(model, xte_np)
    del model

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_va = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_te = np.asarray(
    np.load(os.path.join(shared, "incumbent_test_scores.npy")),
    dtype=np.float64,
)

valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)

# Compare every standalone family and several direct logit blends. Direct
# blending avoids deriving any normalizer or feature statistic from validation.
own_weights = (0.0, 0.15, 0.30, 0.50, 0.70, 1.0)
candidate_primary = {}
best_primary = -1.0
best_scores = None
best_test_scores = None
best_raw_scores = None
best_name = None
best_metrics = None

for name in raw_valid:
    own_va = np.asarray(raw_valid[name], dtype=np.float64)
    own_te = np.asarray(raw_test[name], dtype=np.float64)

    for alpha in own_weights:
        if alpha == 0.0:
            cname = name + "_incumbent"
        elif alpha == 1.0:
            cname = name + "_raw"
        else:
            cname = name + "_blend_%.2f" % alpha

        va_score = alpha * own_va + (1.0 - alpha) * inc_va
        metrics = evaluate(valid_users, valid_y, va_score)
        primary = float(metrics["primary"])
        candidate_primary[cname] = primary

        if primary > best_primary:
            best_primary = primary
            best_scores = va_score.copy()
            best_test_scores = (
                alpha * own_te + (1.0 - alpha) * inc_te
            )
            best_raw_scores = own_va.copy()
            best_name = cname
            best_metrics = metrics

print("CANDIDATES " + json.dumps(candidate_primary, sort_keys=True))
print(
    "FINDINGS selected=%s raw_rank_std=%.6f selected_primary=%.6f" %
    (
        best_name,
        float(np.std(best_raw_scores)),
        float(best_metrics["primary"]),
    ),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.6f}' %
    (
        float(best_metrics["primary"]),
        float(best_metrics["gauc"]),
        float(best_metrics["ndcg@5"]),
        elapsed,
    )
)