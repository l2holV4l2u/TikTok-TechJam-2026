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
SEED = 9143
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768
EPOCHS = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

# Identical, relatively stationary input fields for every family. Auxiliary
# outcomes are used only as training targets by MMoE, never as input features.
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
EMBED_DIM = 12

train = load("train")
valid = load("valid")
test = load("test")


def make_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[name], dtype=np.int64) for name in FIELDS
        ]),
        dtype=np.int64,
    )


xtr_np = make_matrix(train)
xva_np = make_matrix(valid)
xte_np = make_matrix(test)

y_long_np = np.asarray(train.y, dtype=np.float32)
y_click_np = np.asarray(train.aux["is_click"], dtype=np.float32)
y_like_np = np.asarray(train.aux["is_like"], dtype=np.float32)

# Defensive conversion in case an auxiliary signal contains non-finite values.
y_click_np = np.nan_to_num(y_click_np, nan=0.0, posinf=1.0, neginf=0.0)
y_like_np = np.nan_to_num(y_like_np, nan=0.0, posinf=1.0, neginf=0.0)
y_click_np = np.clip(y_click_np, 0.0, 1.0)
y_like_np = np.clip(y_like_np, 0.0, 1.0)

xtr = torch.from_numpy(xtr_np)
y_long = torch.from_numpy(y_long_np)
y_click = torch.from_numpy(y_click_np)
y_like = torch.from_numpy(y_like_np)

# Four-day half-life emphasizes the end of train, which is closest to both
# evaluation windows. The weights are derived exclusively from train dates.
last_train_date = int(np.max(np.asarray(train.date, dtype=np.int64)))
train_age = (
    last_train_date - np.asarray(train.date, dtype=np.int64)
).astype(np.float32)
sample_weight_np = np.exp2(-train_age / 4.0).astype(np.float32)
sample_weight_np /= float(sample_weight_np.mean())
sample_weight = torch.from_numpy(sample_weight_np)


def capped_pos_weight(target, cap=5.0):
    p = float(np.mean(target))
    if p <= 0.0 or p >= 1.0:
        return 1.0
    return float(min(cap, (1.0 - p) / p))


CLICK_POS_WEIGHT = capped_pos_weight(y_click_np)
LIKE_POS_WEIGHT = capped_pos_weight(y_like_np)


class CategoricalBase(nn.Module):
    def __init__(self, dim=EMBED_DIM):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, dim) for cardinality in CARDS
        ])
        self.wide = nn.ModuleList([
            nn.Embedding(cardinality, 1) for cardinality in CARDS
        ])
        self.bias = nn.Parameter(torch.zeros(()))

        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        for wide in self.wide:
            nn.init.zeros_(wide.weight)

    def embed(self, x):
        return torch.stack(
            [self.embeddings[j](x[:, j]) for j in range(N_FIELDS)],
            dim=1,
        )

    def linear(self, x):
        return torch.stack(
            [
                self.wide[j](x[:, j]).squeeze(-1)
                for j in range(N_FIELDS)
            ],
            dim=1,
        ).sum(dim=1) + self.bias


class PNN(nn.Module):
    """
    Product Neural Network. Explicit pairwise inner products expose the
    compatibility of every field pair before nonlinear prediction.
    """

    def __init__(self, dim=EMBED_DIM):
        super().__init__()
        self.base = CategoricalBase(dim)
        n_pairs = N_FIELDS * (N_FIELDS - 1) // 2
        width = N_FIELDS * dim + n_pairs
        self.net = nn.Sequential(
            nn.Linear(width, 160),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.pairs = [
            (i, j)
            for i in range(N_FIELDS)
            for j in range(i + 1, N_FIELDS)
        ]

    def forward(self, x):
        emb = self.base.embed(x)
        products = torch.stack(
            [
                (emb[:, i, :] * emb[:, j, :]).sum(dim=1)
                for i, j in self.pairs
            ],
            dim=1,
        )
        dense_input = torch.cat([emb.flatten(1), products], dim=1)
        return self.base.linear(x) + self.net(dense_input).squeeze(-1)


class AutoInt(nn.Module):
    """
    Multi-head field self-attention forms row-dependent interactions: each
    field can select different contextual fields for different impressions.
    """

    def __init__(self, dim=EMBED_DIM, heads=3):
        super().__init__()
        self.base = CategoricalBase(dim)
        self.attn1 = nn.MultiheadAttention(
            dim, heads, dropout=0.05, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            dim, heads, dropout=0.05, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.out = nn.Sequential(
            nn.Linear(N_FIELDS * dim, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

    def forward(self, x):
        emb = self.base.embed(x)
        a1, _ = self.attn1(emb, emb, emb, need_weights=False)
        h = self.norm1(emb + F.relu(a1))
        a2, _ = self.attn2(h, h, h, need_weights=False)
        h = self.norm2(h + F.relu(a2))
        return self.base.linear(x) + self.out(h.flatten(1)).squeeze(-1)


class MMoE(nn.Module):
    """
    Multi-gate mixture of experts. Click and like are train-only auxiliary
    targets. Task-specific gates decide which shared nonlinear experts should
    inform long_view versus each auxiliary response.
    """

    def __init__(self, dim=EMBED_DIM, n_experts=4, expert_dim=64):
        super().__init__()
        self.base = CategoricalBase(dim)
        in_dim = N_FIELDS * dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, expert_dim),
                nn.ReLU(),
            )
            for _ in range(n_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(in_dim, n_experts) for _ in range(3)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            for _ in range(3)
        ])

    def forward(self, x):
        emb = self.base.embed(x)
        flat = emb.flatten(1)
        expert_values = torch.stack(
            [expert(flat) for expert in self.experts],
            dim=1,
        )

        outputs = []
        for task in range(3):
            gate = torch.softmax(self.gates[task](flat), dim=1)
            mixture = (
                expert_values * gate.unsqueeze(-1)
            ).sum(dim=1)
            outputs.append(self.towers[task](mixture).squeeze(-1))

        # A direct categorical wide path is retained only for the primary task.
        outputs[0] = outputs[0] + self.base.linear(x)
        return outputs


def weighted_bce(logits, target, weights, pos_weight=None):
    if pos_weight is None:
        loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
    else:
        pw = torch.as_tensor(
            pos_weight, dtype=logits.dtype, device=logits.device
        )
        loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
            pos_weight=pw,
        )
    return (loss * weights).mean()


def fit_single_task(model, name):
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    n = len(y_long)
    generator = torch.Generator().manual_seed(
        SEED + sum(ord(c) for c in name)
    )

    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        running_loss = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:min(start + BATCH_SIZE, n)]
            xb = xtr.index_select(0, idx)
            yb = y_long.index_select(0, idx)
            wb = sample_weight.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = weighted_bce(logits, yb, wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.detach()) * len(idx)

        print(
            "TRAIN %s epoch=%d loss=%.6f" %
            (name, epoch + 1, running_loss / n),
            flush=True,
        )


def fit_mmoe(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    n = len(y_long)
    generator = torch.Generator().manual_seed(SEED + 1701)

    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        running_loss = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:min(start + BATCH_SIZE, n)]
            xb = xtr.index_select(0, idx)
            yl = y_long.index_select(0, idx)
            yc = y_click.index_select(0, idx)
            yi = y_like.index_select(0, idx)
            wb = sample_weight.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            out_long, out_click, out_like = model(xb)

            primary_loss = weighted_bce(out_long, yl, wb)
            click_loss = weighted_bce(
                out_click, yc, wb, CLICK_POS_WEIGHT
            )
            like_loss = weighted_bce(
                out_like, yi, wb, LIKE_POS_WEIGHT
            )
            loss = primary_loss + 0.15 * click_loss + 0.15 * like_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.detach()) * len(idx)

        print(
            "TRAIN mmoe epoch=%d loss=%.6f" %
            (epoch + 1, running_loss / n),
            flush=True,
        )


def predict(model, x_np, multitask=False):
    result = np.empty(len(x_np), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x_np), PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, len(x_np))
            xb = torch.from_numpy(x_np[start:end])
            output = model(xb)
            if multitask:
                output = output[0]
            result[start:end] = output.detach().cpu().numpy()
    return result


models = [
    ("pnn", PNN(), False),
    ("autoint", AutoInt(), False),
    ("mmoe", MMoE(), True),
]

raw_valid = {}
raw_test = {}

for name, model, multitask in models:
    if multitask:
        fit_mmoe(model)
    else:
        fit_single_task(model, name)

    raw_valid[name] = predict(model, xva_np, multitask=multitask)
    raw_test[name] = predict(model, xte_np, multitask=multitask)
    del model

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.asarray(
    np.load(os.path.join(shared_dir, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(shared_dir, "incumbent_test_scores.npy")),
    dtype=np.float64,
)

valid_users = np.asarray(valid.user_id)
valid_labels = np.asarray(valid.y, dtype=np.int8)

# Direct logit blends use no validation-derived normalization. Alpha is applied
# unchanged to hidden-test scores.
alphas = (0.0, 0.15, 0.30, 0.50, 0.70, 1.0)
candidate_scores = {}

best_primary = -np.inf
best_name = None
best_metrics = None
best_valid_scores = None
best_test_scores = None
best_raw_valid = None

for family_name in raw_valid:
    own_valid = np.asarray(raw_valid[family_name], dtype=np.float64)
    own_test = np.asarray(raw_test[family_name], dtype=np.float64)

    for alpha in alphas:
        if alpha == 0.0:
            candidate_name = family_name + "_incumbent"
        elif alpha == 1.0:
            candidate_name = family_name + "_raw"
        else:
            candidate_name = "%s_blend_%.2f" % (family_name, alpha)

        blended_valid = (
            alpha * own_valid + (1.0 - alpha) * inc_valid
        )
        metrics = evaluate(
            valid_users, valid_labels, blended_valid
        )
        primary = float(metrics["primary"])
        candidate_scores[candidate_name] = primary

        if primary > best_primary:
            best_primary = primary
            best_name = candidate_name
            best_metrics = metrics
            best_valid_scores = blended_valid.copy()
            best_test_scores = (
                alpha * own_test + (1.0 - alpha) * inc_test
            )
            best_raw_valid = own_valid.copy()

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS selected=%s click_rate=%.6f like_rate=%.6f "
    "click_pos_weight=%.4f like_pos_weight=%.4f" %
    (
        best_name,
        float(y_click_np.mean()),
        float(y_like_np.mean()),
        CLICK_POS_WEIGHT,
        LIKE_POS_WEIGHT,
    ),
    flush=True,
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
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