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
SEED = 19427
BATCH = 8192
PRED_BATCH = 32768
HIST_LEN = 8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

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
VIDEO_FIELD = FIELDS.index("video_id")
USER_CARD = int(FEATURE_CARDINALITIES["user_id"])

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

# Construct strictly causal positive histories for training. Validation and
# test receive only positives from the training split.
def make_histories():
    n = len(ytr_np)
    users = np.asarray(train.X["user_id"], dtype=np.int64)
    videos = np.asarray(train.X["video_id"], dtype=np.int64)
    times = np.asarray(train.time_ms, dtype=np.int64)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, times, users))
    su = users[order]
    sv = videos[order]
    sy = ytr_np[order].astype(np.int64)

    starts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    global_pos = np.cumsum(sy, dtype=np.int64)
    before_group = global_pos[starts] - sy[starts]
    prior_local = global_pos - sy - np.repeat(before_group, sizes)

    pos_mask = sy == 1
    pos_videos = sv[pos_mask]
    pos_users = su[pos_mask]
    pos_counts = np.bincount(pos_users, minlength=USER_CARD).astype(np.int64)
    pos_offsets = np.zeros(USER_CARD, dtype=np.int64)
    if USER_CARD > 1:
        pos_offsets[1:] = np.cumsum(pos_counts[:-1], dtype=np.int64)

    hs = np.zeros((n, HIST_LEN), dtype=np.int64)
    row_offsets = pos_offsets[np.clip(su, 0, USER_CARD - 1)]
    for lag in range(HIST_LEN):
        local = prior_local - 1 - lag
        ok = (local >= 0) & (su != 0)
        hs[ok, lag] = pos_videos[row_offsets[ok] + local[ok]]

    htr = np.empty_like(hs)
    htr[order] = hs

    def full_train_history(split):
        q_users = np.asarray(split.X["user_id"], dtype=np.int64)
        h = np.zeros((len(q_users), HIST_LEN), dtype=np.int64)
        clipped = np.clip(q_users, 0, USER_CARD - 1)
        counts = pos_counts[clipped]
        offsets = pos_offsets[clipped]
        for lag in range(HIST_LEN):
            local = counts - 1 - lag
            ok = (local >= 0) & (q_users != 0) & (q_users < USER_CARD)
            h[ok, lag] = pos_videos[offsets[ok] + local[ok]]
        return h

    return (
        np.ascontiguousarray(htr),
        np.ascontiguousarray(full_train_history(valid)),
        np.ascontiguousarray(full_train_history(test)),
        pos_counts,
    )


htr_np, hva_np, hte_np, positive_counts = make_histories()

print(
    "FINDINGS history_nonempty_train=%.6f valid=%.6f test=%.6f" %
    (
        float(np.mean(np.any(htr_np != 0, axis=1))),
        float(np.mean(np.any(hva_np != 0, axis=1))),
        float(np.mean(np.any(hte_np != 0, axis=1))),
    ),
    flush=True,
)

# Auxiliary outcomes are used only as training targets, never as row inputs.
# Prefer common binary engagement signals and verify their observed support.
aux_names = []
aux_arrays = []
for name in [
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_collect",
]:
    if name in train.aux:
        a = np.asarray(train.aux[name])
        finite = np.isfinite(a)
        if finite.all() and np.all((a == 0) | (a == 1)):
            aux_names.append(name)
            aux_arrays.append(a.astype(np.float32))
        if len(aux_names) == 2:
            break

print(
    "FINDINGS auxiliary_targets=" +
    json.dumps({
        n: float(np.mean(a)) for n, a in zip(aux_names, aux_arrays)
    }, sort_keys=True),
    flush=True,
)

xtr = torch.from_numpy(xtr_np)
htr = torch.from_numpy(htr_np)
ytr = torch.from_numpy(ytr_np)

last_date = int(np.max(np.asarray(train.date, dtype=np.int64)))
age = (last_date - np.asarray(train.date, dtype=np.int64)).astype(np.float32)
weight_np = np.exp2(-age / 4.0).astype(np.float32)
weight_np /= float(np.mean(weight_np))
wtr = torch.from_numpy(weight_np)

if aux_arrays:
    multitask_y_np = np.column_stack([ytr_np] + aux_arrays).astype(np.float32)
else:
    # The normal benchmark exposes auxiliary binary outcomes. This fallback
    # keeps the script executable without pretending duplicated labels are
    # separate supervision.
    multitask_y_np = ytr_np[:, None]
multitask_y = torch.from_numpy(multitask_y_np)
N_TASKS = multitask_y_np.shape[1]


class CategoricalBase(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(c, dim, padding_idx=0) for c in CARDS
        ])
        self.linear = nn.ModuleList([
            nn.Embedding(c, 1, padding_idx=0) for c in CARDS
        ])
        self.bias = nn.Parameter(torch.zeros(()))
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.025)
            with torch.no_grad():
                emb.weight[0].zero_()
        for emb in self.linear:
            nn.init.zeros_(emb.weight)

    def dense(self, x):
        return torch.stack(
            [self.embeddings[j](x[:, j]) for j in range(N_FIELDS)],
            dim=1,
        )

    def wide(self, x):
        return torch.stack(
            [self.linear[j](x[:, j]).squeeze(-1)
             for j in range(N_FIELDS)],
            dim=1,
        ).sum(dim=1) + self.bias


class FieldWeightedFM(nn.Module):
    """
    FwFM gives every field pair a separate learned importance while retaining
    low-rank value interactions. It can suppress unstable identity crosses and
    emphasize candidate/content crosses without a deep tower.
    """
    def __init__(self, dim=16):
        super().__init__()
        self.base = CategoricalBase(dim)
        self.pairs = [
            (i, j) for i in range(N_FIELDS) for j in range(i + 1, N_FIELDS)
        ]
        self.pair_weight = nn.Parameter(torch.ones(len(self.pairs)))

    def forward(self, x, history=None):
        e = self.base.dense(x)
        terms = []
        for p, (i, j) in enumerate(self.pairs):
            terms.append(
                (e[:, i] * e[:, j]).sum(dim=1) * self.pair_weight[p]
            )
        return self.base.wide(x) + torch.stack(terms, dim=1).sum(dim=1)


class DINHistory(nn.Module):
    """
    Candidate-conditioned attention selects prior positive videos relevant to
    the current candidate. Unlike a static user embedding, the pooled history
    changes for each candidate in the user's logged impression set.
    """
    def __init__(self, dim=16):
        super().__init__()
        self.base = CategoricalBase(dim)
        self.hist_embedding = self.base.embeddings[VIDEO_FIELD]
        self.attention = nn.Sequential(
            nn.Linear(4 * dim, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        width = N_FIELDS * dim + dim
        self.net = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x, history):
        e = self.base.dense(x)
        candidate = e[:, VIDEO_FIELD]
        hist = self.hist_embedding(history)
        q = candidate.unsqueeze(1).expand_as(hist)
        attention_input = torch.cat(
            [q, hist, q - hist, q * hist], dim=2
        )
        logits = self.attention(attention_input).squeeze(-1)
        mask = history != 0
        logits = logits.masked_fill(~mask, -1.0e9)
        weights = torch.softmax(logits, dim=1) * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        pooled = (hist * weights.unsqueeze(-1)).sum(dim=1)
        deep = self.net(torch.cat([e.flatten(1), pooled], dim=1)).squeeze(-1)
        return self.base.wide(x) + deep


class GRUHistory(nn.Module):
    """
    A recurrent state models order and repeated transitions among positive
    videos. Its candidate interaction is distinct from DIN's independent
    attention over an unordered set.
    """
    def __init__(self, dim=16):
        super().__init__()
        self.base = CategoricalBase(dim)
        self.hist_embedding = self.base.embeddings[VIDEO_FIELD]
        self.gru = nn.GRU(dim, 24, batch_first=True)
        width = N_FIELDS * dim + 24 + dim + 24
        self.net = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.project_candidate = nn.Linear(dim, 24)

    def forward(self, x, history):
        e = self.base.dense(x)
        lengths = (history != 0).sum(dim=1)
        positions = torch.arange(
            HIST_LEN, device=history.device
        ).unsqueeze(0)
        gather_index = lengths.unsqueeze(1) - 1 - positions
        valid = positions < lengths.unsqueeze(1)
        gather_index = gather_index.clamp_min(0)
        chronological = history.gather(1, gather_index)
        chronological = chronological * valid.long()

        seq = self.hist_embedding(chronological)
        out, _ = self.gru(seq)
        final_index = (lengths - 1).clamp_min(0)
        state = out[
            torch.arange(len(x), device=x.device), final_index
        ]
        state = state * (lengths > 0).float().unsqueeze(1)

        candidate = e[:, VIDEO_FIELD]
        candidate_state = self.project_candidate(candidate)
        interaction = state * candidate_state
        z = torch.cat(
            [e.flatten(1), state, candidate, interaction], dim=1
        )
        return self.base.wide(x) + self.net(z).squeeze(-1)


class MMoE(nn.Module):
    """
    Multiple experts are shared across long-view and auxiliary engagement
    tasks, while task-specific gates decide which behavioral factors transfer.
    Only the long-view head is used at evaluation.
    """
    def __init__(self, dim=12, experts=4):
        super().__init__()
        self.base = CategoricalBase(dim)
        width = N_FIELDS * dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width, 80),
                nn.ReLU(),
                nn.Linear(80, 40),
                nn.ReLU(),
            )
            for _ in range(experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(width, experts) for _ in range(N_TASKS)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(40, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(N_TASKS)
        ])

    def forward(self, x, history=None):
        dense = self.base.dense(x).flatten(1)
        expert_values = torch.stack(
            [expert(dense) for expert in self.experts], dim=1
        )
        outputs = []
        wide = self.base.wide(x)
        for task in range(N_TASKS):
            gate = torch.softmax(self.gates[task](dense), dim=1)
            mixed = (expert_values * gate.unsqueeze(-1)).sum(dim=1)
            task_logit = self.towers[task](mixed).squeeze(-1)
            if task == 0:
                task_logit = task_logit + wide
            outputs.append(task_logit)
        return torch.stack(outputs, dim=1)


def fit_model(model, name, epochs, lr, multitask=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(ytr)
    generator = torch.Generator().manual_seed(
        SEED + sum(ord(c) for c in name)
    )

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=generator)
        loss_total = 0.0

        for start in range(0, n, BATCH):
            idx = perm[start:min(start + BATCH, n)]
            xb = xtr.index_select(0, idx)
            hb = htr.index_select(0, idx)
            wb = wtr.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, hb)

            if multitask:
                target = multitask_y.index_select(0, idx)
                losses = F.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                )
                task_weights = torch.full(
                    (N_TASKS,), 0.15, dtype=torch.float32
                )
                task_weights[0] = 1.0
                per_row = (losses * task_weights.unsqueeze(0)).sum(dim=1)
            else:
                target = ytr.index_select(0, idx)
                per_row = F.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                )

            loss = (per_row * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += float(loss.detach()) * len(idx)

        print(
            "TRAIN %s epoch=%d loss=%.6f" %
            (name, epoch + 1, loss_total / n),
            flush=True,
        )


def predict(model, x_np, h_np, multitask=False):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(x_np), PRED_BATCH):
            end = min(start + PRED_BATCH, len(x_np))
            xb = torch.from_numpy(x_np[start:end])
            hb = torch.from_numpy(h_np[start:end])
            logits = model(xb, hb)
            if multitask:
                logits = logits[:, 0]
            result[start:end] = logits.detach().cpu().numpy()
    return result


families = [
    ("fwfm", FieldWeightedFM(dim=16), 3, 1.0e-3, False),
    ("din_history", DINHistory(dim=16), 2, 1.0e-3, False),
    ("gru_history", GRUHistory(dim=16), 2, 1.0e-3, False),
    ("mmoe_aux", MMoE(dim=12, experts=4), 2, 1.0e-3, True),
]

raw_valid = {}
raw_test = {}

for name, model, epochs, lr, is_multitask in families:
    fit_model(
        model, name, epochs=epochs, lr=lr, multitask=is_multitask
    )
    raw_valid[name] = predict(
        model, xva_np, hva_np, multitask=is_multitask
    )
    raw_test[name] = predict(
        model, xte_np, hte_np, multitask=is_multitask
    )
    del model

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(shared, "incumbent_test_scores.npy")),
    dtype=np.float64,
)

valid_users = np.asarray(valid.user_id)
valid_labels = np.asarray(valid.y, dtype=np.int8)

alphas = (0.0, 0.15, 0.30, 0.50, 0.70, 1.0)
candidate_primary = {}
best_primary = -np.inf
best_name = None
best_metrics = None
best_valid = None
best_test = None
best_raw = None

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

        scores = alpha * own_valid + (1.0 - alpha) * inc_valid
        metrics = evaluate(valid_users, valid_labels, scores)
        primary = float(metrics["primary"])
        candidate_primary[candidate_name] = primary

        if primary > best_primary:
            best_primary = primary
            best_name = candidate_name
            best_metrics = metrics
            best_valid = scores.copy()
            best_test = (
                alpha * own_test + (1.0 - alpha) * inc_test
            )
            best_raw = own_valid.copy()

print("CANDIDATES " + json.dumps(candidate_primary, sort_keys=True))
print(
    "FINDINGS selected=%s selected_primary=%.6f raw_std=%.6f" %
    (
        best_name,
        best_primary,
        float(np.std(best_raw)),
    ),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
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