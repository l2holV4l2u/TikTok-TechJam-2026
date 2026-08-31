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
SEED = 18473
BATCH = 6144
EPOCHS = 3
PRED_BATCH = 65536

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

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
    "onehot_feat3",
    "onehot_feat8",
    "user_active_degree",
]

AUX_CANDIDATES = [
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
]


def make_offsets():
    offsets = []
    total = 0
    for name in FIELDS:
        offsets.append(total)
        total += int(FEATURE_CARDINALITIES[name])
    return np.asarray(offsets, dtype=np.int64), total


OFFSETS, TOTAL_CARDINALITY = make_offsets()


def categorical_matrix(s):
    cols = []
    for j, name in enumerate(FIELDS):
        x = np.asarray(s.X[name], dtype=np.int64)
        x = np.clip(x, 0, int(FEATURE_CARDINALITIES[name]) - 1)
        cols.append(x + OFFSETS[j])
    return np.column_stack(cols).astype(np.int64, copy=False)


def available_auxiliary_keys(train, valid):
    keys = []
    for name in AUX_CANDIDATES:
        if name not in train.aux or name not in valid.aux:
            continue
        a = np.asarray(train.aux[name])
        b = np.asarray(valid.aux[name])
        if a.ndim != 1 or b.ndim != 1:
            continue
        finite_a = a[np.isfinite(a)] if np.issubdtype(a.dtype, np.number) else a
        if finite_a.size == 0:
            continue
        vals = np.unique(finite_a[:min(finite_a.size, 200000)])
        if np.all(np.isin(vals, [0, 1, False, True])):
            keys.append(name)
    return keys[:3]


def target_matrix(s, aux_keys):
    columns = [np.asarray(s.y, dtype=np.float32)]
    for name in aux_keys:
        a = np.asarray(s.aux[name], dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
        columns.append((a > 0).astype(np.float32))
    return np.column_stack(columns).astype(np.float32, copy=False)


def recency_weights(dates, half_life_days):
    dates = np.asarray(dates, dtype=np.int64)
    date_strings = dates.astype(str)
    dt = date_strings.astype("datetime64[D]")
    newest = dt.max()
    age = (newest - dt).astype("timedelta64[D]").astype(np.float32)
    w = np.exp2(-age / float(half_life_days)).astype(np.float32)
    w /= max(float(w.mean()), 1e-6)
    return w


class SharedBottom(nn.Module):
    def __init__(self, n_tasks, embedding_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, embedding_dim)
        in_dim = len(FIELDS) * embedding_dim
        self.bottom = nn.Sequential(
            nn.Linear(in_dim, 96),
            nn.SiLU(),
            nn.LayerNorm(96),
            nn.Linear(96, 48),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(48, 1) for _ in range(n_tasks)])
        nn.init.normal_(self.embedding.weight, std=0.025)

    def forward(self, x):
        z = self.embedding(x).flatten(1)
        h = self.bottom(z)
        return torch.cat([head(h) for head in self.heads], dim=1)


class MMoE(nn.Module):
    def __init__(self, n_tasks, embedding_dim=8, n_experts=4):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, embedding_dim)
        in_dim = len(FIELDS) * embedding_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.SiLU(),
                nn.Linear(64, 40),
                nn.SiLU(),
            )
            for _ in range(n_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(in_dim, n_experts) for _ in range(n_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(40, 24),
                nn.SiLU(),
                nn.Linear(24, 1),
            )
            for _ in range(n_tasks)
        ])
        nn.init.normal_(self.embedding.weight, std=0.025)

    def forward(self, x):
        z = self.embedding(x).flatten(1)
        expert_values = torch.stack([expert(z) for expert in self.experts], dim=1)
        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            weights = torch.softmax(gate(z), dim=1).unsqueeze(2)
            mixed = torch.sum(expert_values * weights, dim=1)
            outputs.append(tower(mixed))
        return torch.cat(outputs, dim=1)


def construct_model(family, n_tasks, seed):
    torch.manual_seed(seed)
    if family == "shared_bottom":
        return SharedBottom(n_tasks)
    if family == "mmoe":
        return MMoE(n_tasks)
    raise ValueError(family)


def fit_model(x, targets, weights, family, seed):
    model = construct_model(family, targets.shape[1], seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0022 if family == "shared_bottom" else 0.0018,
        weight_decay=1e-5,
    )

    task_weights = torch.ones(targets.shape[1], dtype=torch.float32)
    if targets.shape[1] > 1:
        task_weights[1:] = 0.22

    n = x.shape[0]
    generator = torch.Generator()
    generator.manual_seed(seed + 91)

    for epoch in range(EPOCHS):
        order = torch.randperm(n, generator=generator)
        model.train()
        for st in range(0, n, BATCH):
            idx_t = order[st:min(st + BATCH, n)]
            idx = idx_t.numpy()

            xb = torch.from_numpy(x[idx])
            yb = torch.from_numpy(targets[idx])
            wb = torch.from_numpy(weights[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            row_loss = (losses * task_weights.unsqueeze(0)).sum(dim=1)
            loss = (row_loss * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.no_grad()
def predict_model(model, x):
    model.eval()
    predictions = np.empty(x.shape[0], dtype=np.float32)
    for st in range(0, x.shape[0], PRED_BATCH):
        en = min(st + PRED_BATCH, x.shape[0])
        xb = torch.from_numpy(x[st:en])
        predictions[st:en] = model(xb)[:, 0].cpu().numpy()
    return predictions


def standardize(scores):
    scores = np.asarray(scores, dtype=np.float64)
    mean = float(scores.mean())
    std = float(scores.std())
    if std < 1e-12:
        std = 1.0
    return (scores - mean) / std


train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.int8)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

aux_keys = available_auxiliary_keys(train, valid)
print("FINDINGS multitask_auxiliary_targets=" + ",".join(aux_keys), flush=True)

train_x = categorical_matrix(train)
valid_x = categorical_matrix(valid)
train_targets = target_matrix(train, aux_keys)

# Both architectures use the same drift-aware weighting so their difference is
# attributable to prediction formation rather than to temporal sampling.
train_weights = recency_weights(train.date, half_life_days=6.0)

families = ["shared_bottom", "mmoe"]
raw_valid = {}
models = {}

for i, family in enumerate(families):
    model = fit_model(
        train_x,
        train_targets,
        train_weights,
        family,
        SEED + 1000 * i,
    )
    models[family] = model
    raw_valid[family] = predict_model(model, valid_x)

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_vz = standardize(inc_valid)

candidate_results = {}
best_primary = -np.inf
best_metrics = None
best_scores = None
best_family = None
best_weight = None

for family in families:
    own = raw_valid[family]
    own_metrics = evaluate(valid_users, valid_y, own)
    candidate_results[family] = float(own_metrics["primary"])

    if float(own_metrics["primary"]) > best_primary:
        best_primary = float(own_metrics["primary"])
        best_metrics = own_metrics
        best_scores = own.copy()
        best_family = family
        best_weight = 1.0

    own_z = standardize(own)
    for own_weight in (0.08, 0.14, 0.20, 0.28, 0.36, 0.45, 0.55):
        blend = (1.0 - own_weight) * inc_vz + own_weight * own_z
        metrics = evaluate(valid_users, valid_y, blend)
        name = "%s_blend_%.2f" % (family, own_weight)
        candidate_results[name] = float(metrics["primary"])

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_metrics = metrics
            best_scores = blend.copy()
            best_family = family
            best_weight = float(own_weight)

print("CANDIDATES " + json.dumps(candidate_results, sort_keys=True), flush=True)
print(
    "FINDINGS selected_family=%s own_weight=%.2f auxiliary_task_count=%d"
    % (best_family, best_weight, len(aux_keys)),
    flush=True,
)

# Refit the identical selected recipe on train + validation.
combined_x = np.concatenate([train_x, valid_x], axis=0)
valid_targets = target_matrix(valid, aux_keys)
combined_targets = np.concatenate([train_targets, valid_targets], axis=0)
combined_dates = np.concatenate([
    np.asarray(train.date),
    np.asarray(valid.date),
])
combined_weights = recency_weights(combined_dates, half_life_days=6.0)

del models
final_model = fit_model(
    combined_x,
    combined_targets,
    combined_weights,
    best_family,
    SEED + 1000 * families.index(best_family),
)

test = load("test")
test_x = categorical_matrix(test)
own_test = predict_model(final_model, test_x)

if best_weight < 1.0:
    incumbent_test = np.load(inc_test_path).astype(np.float64)
    test_scores = (
        (1.0 - best_weight) * standardize(incumbent_test)
        + best_weight * standardize(own_test)
    )
else:
    test_scores = own_test

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if best_weight < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_valid[best_family], dtype=np.float64),
        )

elapsed = float(time.time() - START)
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": elapsed,
}
print("METRICS " + json.dumps(result, separators=(", ", ": ")))