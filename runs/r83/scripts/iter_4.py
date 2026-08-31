import os
import time
import json
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7319
FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour",
]
EMBED_DIM = 8
BATCH_SIZE = 8192
MAX_EPOCHS = 6
LR = 0.002

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
TOTAL_CATEGORIES = int(sum(cards))
N_FIELDS = len(FIELDS)


def make_X(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[f], dtype=np.int64) + offsets[j]
            for j, f in enumerate(FIELDS)
        ]),
        dtype=np.int64,
    )


def frequency_weights(user_ids):
    u = np.asarray(user_ids, dtype=np.int64)
    counts = np.bincount(u, minlength=int(u.max()) + 1)
    w = 1.0 / np.sqrt(np.maximum(counts[u], 1).astype(np.float32))
    w /= max(float(w.mean()), 1e-8)
    return np.clip(w, 0.25, 4.0).astype(np.float32)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)
    order = np.lexsort((scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    group_start_for_row = np.maximum.accumulate(
        np.where(starts_mask, np.arange(n), 0)
    )
    positions = np.arange(n) - group_start_for_row
    repeated_sizes = np.repeat(sizes, sizes)

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = (positions + 0.5) / repeated_sizes
    return ranked


class BaseCTR(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CATEGORIES, EMBED_DIM)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def embedded(self, x):
        return self.embedding(x)


class PNN(BaseCTR):
    """Explicit pair-product representation followed by a nonlinear tower."""
    def __init__(self, initial_rate):
        super().__init__()
        pair_i, pair_j = np.triu_indices(N_FIELDS, k=1)
        self.register_buffer("pair_i", torch.tensor(pair_i, dtype=torch.long))
        self.register_buffer("pair_j", torch.tensor(pair_j, dtype=torch.long))
        in_dim = N_FIELDS * EMBED_DIM + len(pair_i)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.bias = nn.Parameter(torch.tensor(
            np.log(np.clip(initial_rate, 1e-5, 1 - 1e-5) /
                   (1 - np.clip(initial_rate, 1e-5, 1 - 1e-5))),
            dtype=torch.float32,
        ))

    def forward(self, x):
        e = self.embedded(x)
        products = (e[:, self.pair_i, :] * e[:, self.pair_j, :]).sum(dim=2)
        z = torch.cat([e.flatten(1), products], dim=1)
        return self.mlp(z).squeeze(1) + self.bias


class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.empty(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.w, std=0.02)

    def forward(self, x0, x):
        return x0 * (x @ self.w).unsqueeze(1) + self.b + x


class DCN(BaseCTR):
    """Explicit bounded-degree cross network plus a deep generalization tower."""
    def __init__(self, initial_rate):
        super().__init__()
        dim = N_FIELDS * EMBED_DIM
        self.cross1 = CrossLayer(dim)
        self.cross2 = CrossLayer(dim)
        self.deep = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.out = nn.Linear(dim + 64, 1)
        self.bias = nn.Parameter(torch.tensor(
            np.log(np.clip(initial_rate, 1e-5, 1 - 1e-5) /
                   (1 - np.clip(initial_rate, 1e-5, 1 - 1e-5))),
            dtype=torch.float32,
        ))

    def forward(self, x):
        x0 = self.embedded(x).flatten(1)
        xc = self.cross1(x0, x0)
        xc = self.cross2(x0, xc)
        xd = self.deep(x0)
        return self.out(torch.cat([xc, xd], dim=1)).squeeze(1) + self.bias


class MMoE(BaseCTR):
    """Task-specific gates combine shared experts for long-view/click/like."""
    def __init__(self, initial_rate):
        super().__init__()
        dim = N_FIELDS * EMBED_DIM
        hidden = 64
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 96),
                nn.ReLU(),
                nn.Linear(96, hidden),
                nn.ReLU(),
            )
            for _ in range(4)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(dim, 4) for _ in range(3)
        ])
        self.heads = nn.ModuleList([
            nn.Linear(hidden, 1) for _ in range(3)
        ])
        self.main_bias = nn.Parameter(torch.tensor(
            np.log(np.clip(initial_rate, 1e-5, 1 - 1e-5) /
                   (1 - np.clip(initial_rate, 1e-5, 1 - 1e-5))),
            dtype=torch.float32,
        ))

    def forward(self, x):
        z = self.embedded(x).flatten(1)
        expert_values = torch.stack([expert(z) for expert in self.experts], dim=1)
        outputs = []
        for task in range(3):
            gate = torch.softmax(self.gates[task](z), dim=1).unsqueeze(2)
            mixed = (expert_values * gate).sum(dim=1)
            outputs.append(self.heads[task](mixed).squeeze(1))
        outputs[0] = outputs[0] + self.main_bias
        return torch.stack(outputs, dim=1)


def build_model(name, initial_rate):
    torch.manual_seed(SEED)
    if name == "pnn":
        return PNN(initial_rate)
    if name == "dcn":
        return DCN(initial_rate)
    if name == "mmoe":
        return MMoE(initial_rate)
    raise ValueError(name)


@torch.inference_mode()
def predict(model, name, X):
    model.eval()
    xt = torch.from_numpy(X)
    out = np.empty(len(X), dtype=np.float32)
    for start in range(0, len(X), 32768):
        end = min(start + 32768, len(X))
        logits = model(xt[start:end])
        if name == "mmoe":
            logits = logits[:, 0]
        out[start:end] = logits.cpu().numpy()
    return out


def fit_model(name, X, y, user_ids, epochs, initial_rate,
              aux_targets=None, valid_pack=None):
    model = build_model(name, initial_rate)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=2e-6
    )

    xt = torch.from_numpy(X)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    wt = torch.from_numpy(frequency_weights(user_ids))
    at = None
    if aux_targets is not None:
        at = torch.from_numpy(np.asarray(aux_targets, dtype=np.float32))

    generator = torch.Generator()
    generator.manual_seed(SEED)

    best_primary = -np.inf
    best_epoch = epochs
    best_state = None
    best_scores = None

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(X), generator=generator)

        for start in range(0, len(X), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(xt[idx])
            w = wt[idx]

            if name == "mmoe":
                targets = torch.column_stack((yt[idx], at[idx, 0], at[idx, 1]))
                element_loss = F.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none"
                )
                task_loss = (
                    0.70 * element_loss[:, 0]
                    + 0.18 * element_loss[:, 1]
                    + 0.12 * element_loss[:, 2]
                )
                loss = (task_loss * w).sum() / w.sum()
            else:
                element_loss = F.binary_cross_entropy_with_logits(
                    logits, yt[idx], reduction="none"
                )
                loss = (element_loss * w).sum() / w.sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if valid_pack is not None:
            Xv, yv, uv = valid_pack
            scores = predict(model, name, Xv)
            metrics = evaluate(uv, yv, scores)
            if float(metrics["primary"]) > best_primary:
                best_primary = float(metrics["primary"])
                best_epoch = epoch
                best_scores = scores.copy()
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_epoch, best_scores


train = load("train")
valid = load("valid")

X_train = make_X(train)
X_valid = make_X(valid)
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
u_train = np.asarray(train.user_id, dtype=np.int64)
u_valid = np.asarray(valid.user_id, dtype=np.int64)

aux_train = np.column_stack([
    np.asarray(train.aux["is_click"], dtype=np.float32),
    np.asarray(train.aux["is_like"], dtype=np.float32),
])
aux_valid = np.column_stack([
    np.asarray(valid.aux["is_click"], dtype=np.float32),
    np.asarray(valid.aux["is_like"], dtype=np.float32),
])

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.load(inc_valid_path).astype(np.float64)
inc_valid_rank = within_user_rank(u_valid, inc_valid)

family_names = ["pnn", "dcn", "mmoe"]
family_epochs = {}
family_valid_scores = {}
candidate_log = {}

best_primary = -np.inf
best_name = None
best_alpha = 0.0
best_mode = "standalone"
best_valid_scores = None

for name in family_names:
    model, selected_epoch, scores = fit_model(
        name=name,
        X=X_train,
        y=y_train,
        user_ids=u_train,
        epochs=MAX_EPOCHS,
        initial_rate=float(y_train.mean()),
        aux_targets=aux_train if name == "mmoe" else None,
        valid_pack=(X_valid, y_valid, u_valid),
    )
    family_epochs[name] = selected_epoch
    family_valid_scores[name] = scores.astype(np.float64)

    standalone_metrics = evaluate(u_valid, y_valid, scores)
    standalone_primary = float(standalone_metrics["primary"])
    candidate_log[name] = standalone_primary

    if standalone_primary > best_primary:
        best_primary = standalone_primary
        best_name = name
        best_alpha = 0.0
        best_mode = "standalone"
        best_valid_scores = scores.astype(np.float64).copy()

    new_rank = within_user_rank(u_valid, scores)
    for alpha in (0.2, 0.4, 0.6, 0.8):
        blend = alpha * inc_valid_rank + (1.0 - alpha) * new_rank
        blend_metrics = evaluate(u_valid, y_valid, blend)
        primary = float(blend_metrics["primary"])
        key = f"{name}_rankblend_inc{alpha:.1f}"
        candidate_log[key] = primary

        if primary > best_primary:
            best_primary = primary
            best_name = name
            best_alpha = alpha
            best_mode = "rankblend"
            best_valid_scores = blend.astype(np.float64).copy()

    del model

metrics = evaluate(u_valid, y_valid, best_valid_scores)

# Refit the selected family on train + validation with its validation-selected epoch.
X_combined = np.concatenate([X_train, X_valid], axis=0)
y_combined = np.concatenate([y_train, y_valid.astype(np.float32)], axis=0)
u_combined = np.concatenate([u_train, u_valid], axis=0)
aux_combined = np.concatenate([aux_train, aux_valid], axis=0)

final_model, _, _ = fit_model(
    name=best_name,
    X=X_combined,
    y=y_combined,
    user_ids=u_combined,
    epochs=family_epochs[best_name],
    initial_rate=float(y_combined.mean()),
    aux_targets=aux_combined if best_name == "mmoe" else None,
    valid_pack=None,
)

test = load("test")
X_test = make_X(test)
u_test = np.asarray(test.user_id, dtype=np.int64)
new_test_scores = predict(final_model, best_name, X_test).astype(np.float64)

if best_mode == "rankblend":
    inc_test = np.load(inc_test_path).astype(np.float64)
    inc_test_rank = within_user_rank(u_test, inc_test)
    new_test_rank = within_user_rank(u_test, new_test_scores)
    test_scores = (
        best_alpha * inc_test_rank + (1.0 - best_alpha) * new_test_rank
    )
else:
    test_scores = new_test_scores

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print(
    "FINDINGS " + json.dumps({
        "winner_family": best_name,
        "winner_mode": best_mode,
        "incumbent_weight": float(best_alpha),
        "selected_epoch": int(family_epochs[best_name]),
    }, sort_keys=True)
)
print(
    "METRICS " + json.dumps({
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(time.time() - START),
    })
)