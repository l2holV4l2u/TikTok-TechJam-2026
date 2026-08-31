import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
torch.manual_seed(2026)
np.random.seed(2026)
torch.set_num_threads(min(8, os.cpu_count() or 8))

train = load("train")
valid = load("valid")
test = load("test")

ytr_np = np.asarray(train.y, dtype=np.float32)
yva = np.asarray(valid.y, dtype=np.int8)
uva = np.asarray(valid.user_id, dtype=np.int64)
ute = np.asarray(test.user_id, dtype=np.int64)

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "user_active_degree",
    "onehot_feat3",
]
CARDINALITIES = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]

# Main-model temporal weighting: the evaluation interval immediately follows
# training, so recent rows receive exponentially larger weight.
dates = np.asarray(train.date, dtype=np.int64)
unique_dates = np.unique(dates)
day_map = {int(d): i for i, d in enumerate(unique_dates)}
day_index = np.fromiter(
    (day_map[int(d)] for d in dates),
    dtype=np.int64,
    count=len(dates),
)
age = (len(unique_dates) - 1 - day_index).astype(np.float32)
recency_weight_np = np.exp2(-age / 4.0).astype(np.float32)
recency_weight_np /= recency_weight_np.mean()

# Auxiliary outcomes are used only as training targets. No auxiliary outcome
# from validation or test is read or used as an input.
preferred_aux = [
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
]
aux_names = []
aux_arrays = []

for name in preferred_aux:
    if name not in train.aux:
        continue
    a = np.asarray(train.aux[name])
    if len(a) != len(ytr_np):
        continue
    af = np.asarray(a, dtype=np.float32)
    finite = np.isfinite(af)
    if not np.any(finite):
        continue
    amin = float(np.min(af[finite]))
    amax = float(np.max(af[finite]))
    if amin >= 0.0 and amax <= 1.0:
        af = np.nan_to_num(af, nan=0.0, posinf=1.0, neginf=0.0)
        aux_names.append(name)
        aux_arrays.append(af)
    if len(aux_names) >= 4:
        break

if len(aux_arrays) == 0:
    # The model remains runnable if a dataset build exposes no binary auxiliary
    # signals; this duplicate target simply removes the multi-task advantage.
    aux_names = ["long_view_duplicate"]
    aux_arrays = [ytr_np.copy()]

target_np = np.column_stack([ytr_np] + aux_arrays).astype(np.float32)
n_tasks = target_np.shape[1]

# Positive weighting is computed solely from train. Capping prevents very rare
# engagement outcomes from overwhelming the relevance task.
rates = target_np.mean(axis=0).astype(np.float64)
pos_weights_np = np.clip(
    (1.0 - rates) / np.maximum(rates, 1.0e-5),
    0.5,
    8.0,
).astype(np.float32)

task_weights_np = np.full(n_tasks, 0.20, dtype=np.float32)
task_weights_np[0] = 1.0
task_weights_np /= task_weights_np.sum()

print(
    "FINDINGS auxiliary_targets=%s train_rates=%s"
    % (
        json.dumps(["long_view"] + aux_names),
        json.dumps([float(x) for x in rates]),
    )
)

xtr = torch.from_numpy(
    np.column_stack([
        np.asarray(train.X[f], dtype=np.int64) for f in FIELDS
    ])
)
targets = torch.from_numpy(target_np)
recency_weight = torch.from_numpy(recency_weight_np)
pos_weights = torch.from_numpy(pos_weights_np)
task_weights = torch.from_numpy(task_weights_np)


class CategoricalEncoder(nn.Module):
    def __init__(self, cardinalities, embedding_dim=12):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(k, embedding_dim) for k in cardinalities
        ])
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.025)
            with torch.no_grad():
                emb.weight[0].zero_()

    def forward(self, x):
        return torch.cat(
            [emb(x[:, j]) for j, emb in enumerate(self.embeddings)],
            dim=1,
        )


class SharedBottomMTL(nn.Module):
    """One shared representation followed by independent task heads."""

    def __init__(self, cardinalities, tasks):
        super().__init__()
        self.encoder = CategoricalEncoder(cardinalities, 12)
        input_dim = len(cardinalities) * 12
        self.bottom = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(tasks)])

    def forward(self, x):
        h = self.bottom(self.encoder(x))
        return torch.cat([head(h) for head in self.heads], dim=1)


class MMoE(nn.Module):
    """Task-specific soft mixtures of nonlinear experts."""

    def __init__(self, cardinalities, tasks, n_experts=4):
        super().__init__()
        self.tasks = tasks
        self.n_experts = n_experts
        self.encoder = CategoricalEncoder(cardinalities, 12)
        input_dim = len(cardinalities) * 12

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 96),
                nn.ReLU(),
                nn.Linear(96, 48),
                nn.ReLU(),
            )
            for _ in range(n_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, n_experts) for _ in range(tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(48, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            for _ in range(tasks)
        ])

    def forward(self, x):
        base = self.encoder(x)
        expert_values = torch.stack(
            [expert(base) for expert in self.experts],
            dim=1,
        )
        outputs = []
        for task in range(self.tasks):
            gate = torch.softmax(self.gates[task](base), dim=1)
            mixed = torch.sum(expert_values * gate.unsqueeze(-1), dim=1)
            outputs.append(self.towers[task](mixed))
        return torch.cat(outputs, dim=1)


def multitask_loss(logits, labels, row_weight):
    raw = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
        pos_weight=pos_weights,
    )
    raw = raw * task_weights.unsqueeze(0)
    return torch.mean(torch.sum(raw, dim=1) * row_weight)


def train_model(model, epochs=3, batch_size=32768):
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=2.0e-6,
    )
    n = len(xtr)

    for epoch in range(epochs):
        generator = torch.Generator()
        generator.manual_seed(9137 + epoch)
        permutation = torch.randperm(n, generator=generator)

        epoch_loss = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = xtr[idx]
            yb = targets[idx]
            wb = recency_weight[idx]

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = multitask_loss(logits, yb, wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            count = len(idx)
            epoch_loss += float(loss.detach()) * count
            seen += count

        print(
            "FINDINGS model=%s epoch=%d loss=%.6f"
            % (model.__class__.__name__, epoch + 1, epoch_loss / seen)
        )
    return model


def predict(model, sample, batch_size=65536):
    model.eval()
    arrays = [
        np.asarray(sample.X[f], dtype=np.int64) for f in FIELDS
    ]
    n = len(arrays[0])
    scores = np.empty(n, dtype=np.float64)

    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.from_numpy(
                np.column_stack([a[start:end] for a in arrays])
            )
            scores[start:end] = (
                model(xb)[:, 0].detach().cpu().numpy().astype(np.float64)
            )
    return scores


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    row = np.arange(len(scores), dtype=np.int64)

    # Ascending ordering assigns rank one to the largest score.
    order = np.lexsort((row, scores, users))
    sorted_users = users[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], len(order)]
    lengths = ends - starts

    position = (
        np.arange(len(order), dtype=np.float64)
        - np.repeat(starts, lengths)
    )
    denominator = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    sorted_rank = position / denominator

    result = np.empty(len(scores), dtype=np.float64)
    result[order] = sorted_rank
    return result


shared_model = train_model(
    SharedBottomMTL(CARDINALITIES, n_tasks),
    epochs=3,
)
shared_valid = predict(shared_model, valid)
shared_test = predict(shared_model, test)
del shared_model

mmoe_model = train_model(
    MMoE(CARDINALITIES, n_tasks, n_experts=4),
    epochs=3,
)
mmoe_valid = predict(mmoe_model, valid)
mmoe_test = predict(mmoe_model, test)
del mmoe_model

shared_rank_valid = within_user_rank(uva, shared_valid)
shared_rank_test = within_user_rank(ute, shared_test)
mmoe_rank_valid = within_user_rank(uva, mmoe_valid)
mmoe_rank_test = within_user_rank(ute, mmoe_test)

# This ensemble combines globally shared auxiliary representations with
# relevance-specific expert routing.
ensemble_rank_valid = 0.5 * shared_rank_valid + 0.5 * mmoe_rank_valid
ensemble_rank_test = 0.5 * shared_rank_test + 0.5 * mmoe_rank_test

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.asarray(
    np.load(os.path.join(shared_dir, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(shared_dir, "incumbent_test_scores.npy")),
    dtype=np.float64,
)
inc_rank_valid = within_user_rank(uva, inc_valid)
inc_rank_test = within_user_rank(ute, inc_test)

families_valid = {
    "shared_bottom_mtl": shared_rank_valid,
    "mmoe": mmoe_rank_valid,
    "shared_mmoe_ensemble": ensemble_rank_valid,
}
families_test = {
    "shared_bottom_mtl": shared_rank_test,
    "mmoe": mmoe_rank_test,
    "shared_mmoe_ensemble": ensemble_rank_test,
}

candidate_scores = {}
best_primary = -np.inf
best_metrics = None
best_valid = None
best_test = None
best_raw = None
best_name = None

# The trusted-incumbent contract explicitly permits validation selection of
# blend weights, with the identical selected weight applied to test.
alphas = [0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]

for name, own_valid in families_valid.items():
    own_test = families_test[name]
    standalone = evaluate(uva, yva, own_valid)
    candidate_scores[name + "_standalone"] = float(standalone["primary"])

    for alpha in alphas:
        blend_valid = (
            (1.0 - alpha) * inc_rank_valid + alpha * own_valid
        )
        blend_test = (
            (1.0 - alpha) * inc_rank_test + alpha * own_test
        )
        metrics = evaluate(uva, yva, blend_valid)
        primary = float(metrics["primary"])
        candidate_name = "%s_blend_%.2f" % (name, alpha)
        candidate_scores[candidate_name] = primary

        if primary > best_primary:
            best_primary = primary
            best_metrics = metrics
            best_valid = blend_valid.copy()
            best_test = blend_test.copy()
            best_raw = own_valid.copy()
            best_name = candidate_name

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS selected=%s primary=%.6f"
    % (best_name, best_primary)
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
    "METRICS "
    + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)