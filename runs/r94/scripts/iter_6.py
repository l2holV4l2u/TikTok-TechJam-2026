import os
import time
import json
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7319
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
NUM_TASKS = 3
EMBED_DIM = 12
BATCH_SIZE = 4096
EPOCHS = 3
LR = 1.0e-3
HALF_LIFE = 4.0
AUX_WEIGHTS = (1.0, 0.20, 0.15)

offsets = []
total_cardinality = 0
for field in FIELDS:
    offsets.append(total_cardinality)
    total_cardinality += int(FEATURE_CARDINALITIES[field])
offsets = np.asarray(offsets, dtype=np.int64)
input_dim = len(FIELDS) * EMBED_DIM


def make_x(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[field], dtype=np.int64) + offsets[j]
            for j, field in enumerate(FIELDS)
        ]),
        dtype=np.int64,
    )


def recency_weights(dates):
    d = np.asarray(dates, dtype=np.int32)
    latest = int(d.max())
    w = np.power(2.0, (d.astype(np.float64) - latest) / HALF_LIFE)
    w /= w.mean()
    return w.astype(np.float32)


class MultiTaskBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            total_cardinality, EMBED_DIM, sparse=True
        )
        self.task_linear = nn.ModuleList([
            nn.Embedding(total_cardinality, 1, sparse=True)
            for _ in range(NUM_TASKS)
        ])
        self.task_bias = nn.Parameter(torch.zeros(NUM_TASKS))
        nn.init.normal_(self.embedding.weight, std=0.02)
        for layer in self.task_linear:
            nn.init.zeros_(layer.weight)

    def embedded(self, x):
        return self.embedding(x).flatten(1)

    def wide_logits(self, x):
        return torch.stack([
            layer(x).squeeze(-1).sum(dim=1)
            for layer in self.task_linear
        ], dim=1) + self.task_bias

    def sparse_parameters(self):
        return (
            [self.embedding.weight]
            + [layer.weight for layer in self.task_linear]
        )

    def dense_parameters(self):
        sparse_ids = {id(p) for p in self.sparse_parameters()}
        return [p for p in self.parameters() if id(p) not in sparse_ids]

    def score(self, x):
        return self.forward(x)[:, 0]


class SharedBottom(MultiTaskBase):
    def __init__(self):
        super().__init__()
        self.bottom = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(48, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(NUM_TASKS)
        ])

    def forward(self, x):
        h = self.bottom(self.embedded(x))
        task_logits = torch.cat([head(h) for head in self.heads], dim=1)
        return task_logits + self.wide_logits(x)


class MMoE(MultiTaskBase):
    def __init__(self):
        super().__init__()
        self.num_experts = 4
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 40),
                nn.ReLU(),
            )
            for _ in range(self.num_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, self.num_experts)
            for _ in range(NUM_TASKS)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(40, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(NUM_TASKS)
        ])

    def forward(self, x):
        z = self.embedded(x)
        expert_values = torch.stack(
            [expert(z) for expert in self.experts], dim=1
        )
        outputs = []
        for task in range(NUM_TASKS):
            gate = torch.softmax(self.gates[task](z), dim=1)
            mixed = torch.sum(
                expert_values * gate.unsqueeze(-1), dim=1
            )
            outputs.append(self.towers[task](mixed))
        return torch.cat(outputs, dim=1) + self.wide_logits(x)


class PLE(MultiTaskBase):
    def __init__(self):
        super().__init__()
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 40),
                nn.ReLU(),
            )
            for _ in range(2)
        ])
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 40),
                    nn.ReLU(),
                )
                for _ in range(2)
            ])
            for _ in range(NUM_TASKS)
        ])
        self.task_gates = nn.ModuleList([
            nn.Linear(input_dim, 4)
            for _ in range(NUM_TASKS)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(40, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(NUM_TASKS)
        ])

    def forward(self, x):
        z = self.embedded(x)
        shared = [expert(z) for expert in self.shared_experts]
        outputs = []

        for task in range(NUM_TASKS):
            private = [
                expert(z) for expert in self.task_experts[task]
            ]
            candidates = torch.stack(shared + private, dim=1)
            gate = torch.softmax(self.task_gates[task](z), dim=1)
            mixed = torch.sum(candidates * gate.unsqueeze(-1), dim=1)
            outputs.append(self.towers[task](mixed))

        return torch.cat(outputs, dim=1) + self.wide_logits(x)


def train_model(model, x_train, targets, weights, seed):
    sparse_opt = torch.optim.SparseAdam(
        model.sparse_parameters(), lr=LR
    )
    dense_opt = torch.optim.Adam(
        model.dense_parameters(), lr=LR
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    n = x_train.shape[0]
    task_weights = torch.tensor(
        AUX_WEIGHTS, dtype=torch.float32
    ).view(1, -1)

    model.train()
    for epoch in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            xb = x_train[idx]
            yb = targets[idx]
            wb = weights[idx].view(-1, 1)

            sparse_opt.zero_grad(set_to_none=True)
            dense_opt.zero_grad(set_to_none=True)

            logits = model(xb)
            element_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            weighted = element_loss * task_weights * wb
            loss = weighted.sum() / (
                wb.sum() * float(sum(AUX_WEIGHTS))
            )

            loss.backward()
            sparse_opt.step()
            dense_opt.step()

    return model


def predict_model(model, x_np):
    model.eval()
    result = np.empty(x_np.shape[0], dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, x_np.shape[0], 32768):
            end = min(start + 32768, x_np.shape[0])
            xb = torch.from_numpy(x_np[start:end])
            result[start:end] = (
                model.score(xb).cpu().numpy().astype(np.float64)
            )
    return result


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    groups = np.cumsum(starts_mask) - 1
    positions = np.arange(n, dtype=np.int64) - starts[groups]
    sizes = np.diff(np.append(starts, n))
    denominators = np.maximum(sizes[groups] - 1, 1)

    ranked_sorted = positions.astype(np.float64) / denominators
    ranked_sorted[sizes[groups] == 1] = 0.5

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


train = load("train")
valid = load("valid")

x_train_np = make_x(train)
x_valid_np = make_x(valid)
x_train = torch.from_numpy(x_train_np)

long_target = np.asarray(train.y, dtype=np.float32)
click_target = np.clip(
    np.nan_to_num(
        np.asarray(train.aux["is_click"], dtype=np.float32),
        nan=0.0,
    ),
    0.0,
    1.0,
)
like_target = np.clip(
    np.nan_to_num(
        np.asarray(train.aux["is_like"], dtype=np.float32),
        nan=0.0,
    ),
    0.0,
    1.0,
)
targets = torch.from_numpy(np.ascontiguousarray(
    np.column_stack([long_target, click_target, like_target]),
    dtype=np.float32,
))
weights_np = recency_weights(train.date)
weights = torch.from_numpy(weights_np)

print(
    "FINDINGS train_target_rates "
    + json.dumps({
        "long_view": float(long_target.mean()),
        "is_click": float(click_target.mean()),
        "is_like": float(like_target.mean()),
    }, sort_keys=True)
)

constructors = [
    ("multitask_shared_bottom", SharedBottom),
    ("multitask_mmoe", MMoE),
    ("multitask_ple", PLE),
]

models = {}
valid_raw = {}
for model_index, (name, constructor) in enumerate(constructors):
    torch.manual_seed(SEED + 100 * model_index)
    model = constructor()
    model = train_model(
        model,
        x_train,
        targets,
        weights,
        SEED + 1000 * model_index,
    )
    models[name] = model
    valid_raw[name] = predict_model(model, x_valid_np)

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared_dir, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared_dir, "incumbent_test_scores.npy"
)
if not os.path.exists(inc_valid_path):
    raise FileNotFoundError("Trusted incumbent validation scores unavailable")
if not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent test scores unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

candidate_metrics = {}
candidate_arrays = {}
candidate_family = {}
candidate_alpha = {}

for family_name, raw_scores in valid_raw.items():
    standalone = evaluate(valid.user_id, valid.y, raw_scores)
    candidate_metrics[family_name] = float(standalone["primary"])
    candidate_arrays[family_name] = raw_scores
    candidate_family[family_name] = family_name
    candidate_alpha[family_name] = 1.0

    raw_rank = within_user_rank(valid.user_id, raw_scores)
    for alpha in (0.20, 0.35, 0.50, 0.65, 0.80):
        blended = alpha * raw_rank + (1.0 - alpha) * inc_valid_rank
        candidate_name = f"{family_name}_blend_{alpha:.2f}"
        result = evaluate(valid.user_id, valid.y, blended)
        candidate_metrics[candidate_name] = float(result["primary"])
        candidate_arrays[candidate_name] = blended
        candidate_family[candidate_name] = family_name
        candidate_alpha[candidate_name] = alpha

winner_name = max(candidate_metrics, key=candidate_metrics.get)
winner_family = candidate_family[winner_name]
winner_alpha = candidate_alpha[winner_name]
valid_scores = candidate_arrays[winner_name]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(candidate_metrics, sort_keys=True))
print(
    "FINDINGS winner "
    + json.dumps({
        "candidate": winner_name,
        "family": winner_family,
        "own_weight": float(winner_alpha),
    }, sort_keys=True)
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if winner_alpha < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(valid_raw[winner_family], dtype=np.float64),
        )

test = load("test")
x_test_np = make_x(test)
test_raw = predict_model(models[winner_family], x_test_np)

if winner_alpha < 1.0:
    inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
    test_scores = (
        winner_alpha * within_user_rank(test.user_id, test_raw)
        + (1.0 - winner_alpha)
        * within_user_rank(test.user_id, inc_test)
    )
else:
    test_scores = test_raw

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)