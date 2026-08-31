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
SEED = 9473
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
AUX_CANDIDATES = ["is_click", "is_like", "is_follow"]
EMBED_DIM = 8
BATCH_SIZE = 8192
CHECKPOINTS = (2, 4)
MAX_EPOCHS = max(CHECKPOINTS)
LR = 0.002
HALF_LIFE = 7.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

cards = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))
n_fields = len(FIELDS)
input_dim = n_fields * EMBED_DIM


def encode(split):
    return np.stack(
        [
            np.asarray(split.X[name], dtype=np.int64) + offsets[j]
            for j, name in enumerate(FIELDS)
        ],
        axis=1,
    )


def recency_weights(dates, half_life=HALF_LIFE):
    dates = np.asarray(dates)
    unique_dates = np.unique(dates)
    day_index = np.searchsorted(unique_dates, dates).astype(np.float32)
    age = float(len(unique_dates) - 1) - day_index
    weights = np.exp2(-age / float(half_life)).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-6)
    return weights


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)

    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    counts = ends - starts

    group_starts = np.repeat(starts, counts)
    group_counts = np.repeat(counts, counts)
    positions = np.arange(n, dtype=np.float64) - group_starts

    ranked_sorted = np.full(n, 0.5, dtype=np.float64)
    mask = group_counts > 1
    ranked_sorted[mask] = positions[mask] / (group_counts[mask] - 1.0)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


class MultiTaskBase(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        self.n_tasks = n_tasks
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)
        self.wide = nn.Embedding(total_cardinality, n_tasks)
        self.bias = nn.Parameter(torch.zeros(n_tasks))
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.wide.weight)

    def inputs(self, x):
        embedded = self.embedding(x).reshape(x.shape[0], -1)
        wide = self.wide(x).sum(dim=1) + self.bias
        return embedded, wide


class MMoE(MultiTaskBase):
    def __init__(self, n_tasks):
        super().__init__(n_tasks)
        self.n_experts = 4
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                )
                for _ in range(self.n_experts)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(input_dim, self.n_experts) for _ in range(n_tasks)]
        )
        self.towers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                )
                for _ in range(n_tasks)
            ]
        )

    def forward(self, x):
        z, wide = self.inputs(x)
        expert_outputs = torch.stack([expert(z) for expert in self.experts], dim=1)
        outputs = []
        for task in range(self.n_tasks):
            gate = torch.softmax(self.gates[task](z), dim=1).unsqueeze(2)
            representation = (expert_outputs * gate).sum(dim=1)
            outputs.append(self.towers[task](representation).squeeze(1))
        return torch.stack(outputs, dim=1) + wide


class PLE(MultiTaskBase):
    def __init__(self, n_tasks):
        super().__init__(n_tasks)
        self.n_shared = 2
        self.shared_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                )
                for _ in range(self.n_shared)
            ]
        )
        self.private_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                )
                for _ in range(n_tasks)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(input_dim, self.n_shared + 1) for _ in range(n_tasks)]
        )
        self.towers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                )
                for _ in range(n_tasks)
            ]
        )

    def forward(self, x):
        z, wide = self.inputs(x)
        shared = [expert(z) for expert in self.shared_experts]
        outputs = []

        for task in range(self.n_tasks):
            private = self.private_experts[task](z)
            candidates = torch.stack(shared + [private], dim=1)
            gate = torch.softmax(self.gates[task](z), dim=1).unsqueeze(2)
            representation = (candidates * gate).sum(dim=1)
            outputs.append(self.towers[task](representation).squeeze(1))

        return torch.stack(outputs, dim=1) + wide


class ESMM(MultiTaskBase):
    def __init__(self, n_tasks):
        super().__init__(n_tasks)
        self.bottom = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
        )
        self.click_tower = nn.Sequential(
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )
        self.conditional_long_tower = nn.Sequential(
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )
        self.extra_towers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(48, 24),
                    nn.ReLU(),
                    nn.Linear(24, 1),
                )
                for _ in range(max(0, n_tasks - 2))
            ]
        )

    def forward(self, x):
        z, wide = self.inputs(x)
        h = self.bottom(z)

        click_logit = self.click_tower(h).squeeze(1) + wide[:, 1]
        conditional_logit = (
            self.conditional_long_tower(h).squeeze(1) + wide[:, 0]
        )

        log_product = (
            nn.functional.logsigmoid(click_logit)
            + nn.functional.logsigmoid(conditional_logit)
        )
        product = torch.exp(log_product).clamp(max=1.0 - 1e-6)
        long_logit = log_product - torch.log1p(-product)

        outputs = [long_logit, click_logit]
        for j, tower in enumerate(self.extra_towers):
            outputs.append(tower(h).squeeze(1) + wide[:, j + 2])

        return torch.stack(outputs, dim=1)


def make_model(name, n_tasks):
    if name == "mmoe":
        return MMoE(n_tasks)
    if name == "ple":
        return PLE(n_tasks)
    if name == "esmm":
        return ESMM(n_tasks)
    raise ValueError(name)


def predict(model, x_np):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float32)
    with torch.inference_mode():
        for lo in range(0, len(x_np), BATCH_SIZE * 2):
            hi = min(lo + BATCH_SIZE * 2, len(x_np))
            xb = torch.from_numpy(x_np[lo:hi])
            result[lo:hi] = model(xb)[:, 0].cpu().numpy()
    return result


def multitask_loss(logits, targets, masks, row_weights, task_weights):
    element_loss = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    effective = (
        masks
        * row_weights.unsqueeze(1)
        * task_weights.unsqueeze(0)
    )
    return (element_loss * effective).sum() / effective.sum().clamp_min(1.0)


def train_select(
    family,
    x_train,
    targets_train,
    masks_train,
    dates_train,
    x_valid,
    y_valid,
    valid_users,
    task_weights,
):
    torch.manual_seed(SEED)
    model = make_model(family, targets_train.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-6
    )

    x_t = torch.from_numpy(x_train)
    targets_t = torch.from_numpy(targets_train)
    masks_t = torch.from_numpy(masks_train)
    weights_t = torch.from_numpy(recency_weights(dates_train))
    task_weights_t = torch.from_numpy(task_weights)
    n = len(x_train)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    best_primary = -np.inf
    best_epoch = None
    best_scores = None
    best_metrics = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(n, generator=generator)

        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:min(lo + BATCH_SIZE, n)]
            logits = model(x_t[idx])
            loss = multitask_loss(
                logits,
                targets_t[idx],
                masks_t[idx],
                weights_t[idx],
                task_weights_t,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if epoch in CHECKPOINTS:
            scores = predict(model, x_valid)
            metrics = evaluate(valid_users, y_valid, scores)
            if float(metrics["primary"]) > best_primary:
                best_primary = float(metrics["primary"])
                best_epoch = epoch
                best_scores = scores.copy()
                best_metrics = metrics

    return best_epoch, best_scores, best_metrics


def fit_fixed(
    family,
    x_fit,
    targets_fit,
    masks_fit,
    dates_fit,
    task_weights,
    epochs,
):
    torch.manual_seed(SEED)
    model = make_model(family, targets_fit.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-6
    )

    x_t = torch.from_numpy(x_fit)
    targets_t = torch.from_numpy(targets_fit)
    masks_t = torch.from_numpy(masks_fit)
    weights_t = torch.from_numpy(recency_weights(dates_fit))
    task_weights_t = torch.from_numpy(task_weights)
    n = len(x_fit)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    for _ in range(epochs):
        model.train()
        order = torch.randperm(n, generator=generator)
        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:min(lo + BATCH_SIZE, n)]
            logits = model(x_t[idx])
            loss = multitask_loss(
                logits,
                targets_t[idx],
                masks_t[idx],
                weights_t[idx],
                task_weights_t,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


train = load("train")
valid = load("valid")

# Auxiliary outcomes are read from TRAIN only and used exclusively as targets.
aux_names = [name for name in AUX_CANDIDATES if name in train.aux]
if "is_click" not in aux_names:
    raise RuntimeError("is_click is required for the ESMM comparison")

# ESMM expects click to be the first auxiliary task.
aux_names = ["is_click"] + [name for name in aux_names if name != "is_click"]

x_train = encode(train)
x_valid = encode(valid)
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)
dates_train = np.asarray(train.date)

train_aux_targets = [
    np.asarray(train.aux[name], dtype=np.float32) for name in aux_names
]
targets_train = np.column_stack([y_train] + train_aux_targets).astype(
    np.float32, copy=False
)
masks_train = np.ones_like(targets_train, dtype=np.float32)

task_weights = np.asarray(
    [1.0] + [0.25 if name == "is_click" else 0.12 for name in aux_names],
    dtype=np.float32,
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.load(inc_valid_path)
inc_valid_rank = within_user_rank(valid_users, inc_valid)

families = ["mmoe", "ple", "esmm"]
candidate_scores = {}
candidate_metrics = {}
candidate_epochs = {}
recorded = {}

for family in families:
    epoch, scores, metrics = train_select(
        family=family,
        x_train=x_train,
        targets_train=targets_train,
        masks_train=masks_train,
        dates_train=dates_train,
        x_valid=x_valid,
        y_valid=y_valid,
        valid_users=valid_users,
        task_weights=task_weights,
    )
    candidate_scores[family] = scores
    candidate_metrics[family] = metrics
    candidate_epochs[family] = epoch
    recorded[family + "_standalone"] = float(metrics["primary"])
    recorded[family + "_epoch"] = int(epoch)

best_primary = -np.inf
best_family = None
best_alpha = None
best_scores = None
best_raw_scores = None
best_metrics = None

alphas = np.linspace(0.0, 1.0, 11)

for family in families:
    raw_scores = candidate_scores[family]
    model_rank = within_user_rank(valid_users, raw_scores)
    local_best = -np.inf
    local_alpha = None

    for alpha in alphas:
        blended = (1.0 - alpha) * inc_valid_rank + alpha * model_rank
        metrics = evaluate(valid_users, y_valid, blended)
        primary = float(metrics["primary"])

        if primary > local_best:
            local_best = primary
            local_alpha = float(alpha)

        if primary > best_primary:
            best_primary = primary
            best_family = family
            best_alpha = float(alpha)
            best_scores = blended.copy()
            best_raw_scores = raw_scores.copy()
            best_metrics = metrics

    recorded[family + "_best_blend"] = float(local_best)
    recorded[family + "_blend_alpha"] = float(local_alpha)

print("CANDIDATES " + json.dumps(recorded, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "auxiliary_targets": aux_names,
            "validation_aux_read": False,
            "selected_family": best_family,
            "selected_epoch": int(candidate_epochs[best_family]),
            "selected_model_weight": float(best_alpha),
        },
        sort_keys=True,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(best_raw_scores, dtype=np.float64),
    )

# Refit on train + validation long_view labels. Validation auxiliary outcomes
# are deliberately never accessed; their auxiliary losses are masked out.
n_train = len(x_train)
n_valid = len(x_valid)
n_tasks = targets_train.shape[1]

x_fit = np.concatenate([x_train, x_valid], axis=0)
targets_valid_refit = np.zeros((n_valid, n_tasks), dtype=np.float32)
targets_valid_refit[:, 0] = y_valid.astype(np.float32)
masks_valid_refit = np.zeros((n_valid, n_tasks), dtype=np.float32)
masks_valid_refit[:, 0] = 1.0

targets_fit = np.concatenate(
    [targets_train, targets_valid_refit], axis=0
)
masks_fit = np.concatenate(
    [masks_train, masks_valid_refit], axis=0
)
dates_fit = np.concatenate(
    [dates_train, np.asarray(valid.date)], axis=0
)

test_model = fit_fixed(
    family=best_family,
    x_fit=x_fit,
    targets_fit=targets_fit,
    masks_fit=masks_fit,
    dates_fit=dates_fit,
    task_weights=task_weights,
    epochs=candidate_epochs[best_family],
)

test = load("test")
x_test = encode(test)
raw_test_scores = predict(test_model, x_test)
test_users = np.asarray(test.user_id)

inc_test = np.load(inc_test_path)
inc_test_rank = within_user_rank(test_users, inc_test)
model_test_rank = within_user_rank(test_users, raw_test_scores)
test_scores = (
    (1.0 - best_alpha) * inc_test_rank
    + best_alpha * model_test_rank
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(", ", ": "),
    )
)