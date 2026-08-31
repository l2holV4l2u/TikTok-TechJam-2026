import os
import time
import json
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "hour",
    "tag",
    "upload_type",
    "music_type",
    "user_active_degree",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "video_type",
]

EMBED_DIM = 8
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 16384
EPOCHS = 3
LEARNING_RATE = 0.002
WEIGHT_DECAY = 1e-6
RECENCY_HALF_LIFE_DAYS = 6.0


def make_matrix(split):
    cardinalities = [FEATURE_CARDINALITIES[f] for f in FIELDS]
    offsets = np.cumsum(
        np.asarray([0] + cardinalities[:-1], dtype=np.int64)
    )
    cols = [
        np.asarray(split.X[f], dtype=np.int64) + offsets[i]
        for i, f in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.int64)


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int32)
    last_date = int(dates.max())
    # Training lies entirely in April 2022, so YYYYMMDD subtraction is exact.
    age_days = (last_date - dates).astype(np.float32)
    w = np.power(0.5, age_days / RECENCY_HALF_LIFE_DAYS).astype(np.float32)
    w /= max(float(w.mean()), 1e-8)
    return w


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    group_start = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_positions = np.where(ends)[0]
    group_sizes = np.diff(
        np.concatenate((np.asarray([-1], dtype=np.int64), end_positions))
    )
    size_per_sorted_row = np.repeat(group_sizes, group_sizes)

    position = np.arange(n, dtype=np.int64) - group_start
    percentile_sorted = (position.astype(np.float64) + 0.5) / size_per_sorted_row

    result = np.empty(n, dtype=np.float64)
    result[order] = percentile_sorted
    return result


class NFM(nn.Module):
    def __init__(self, num_features, num_fields, initial_bias):
        super().__init__()
        self.linear = nn.Embedding(num_features, 1)
        self.embedding = nn.Embedding(num_features, EMBED_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(EMBED_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.bias = nn.Parameter(torch.tensor(initial_bias, dtype=torch.float32))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x):
        e = self.embedding(x)
        summed = e.sum(dim=1)
        bi = 0.5 * (summed.square() - e.square().sum(dim=1))
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        return self.bias + linear + self.mlp(bi).squeeze(-1)


class AutoInt(nn.Module):
    def __init__(self, num_features, num_fields, initial_bias):
        super().__init__()
        self.num_fields = num_fields
        self.embedding = nn.Embedding(num_features, EMBED_DIM)
        self.linear = nn.Embedding(num_features, 1)
        self.attn1 = nn.MultiheadAttention(
            EMBED_DIM, num_heads=2, dropout=0.05, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            EMBED_DIM, num_heads=2, dropout=0.05, batch_first=True
        )
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Sequential(
            nn.Linear(num_fields * EMBED_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(64, 1),
        )
        self.bias = nn.Parameter(torch.tensor(initial_bias, dtype=torch.float32))
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        e = self.embedding(x)
        a1, _ = self.attn1(e, e, e, need_weights=False)
        h = self.norm1(e + a1)
        a2, _ = self.attn2(h, h, h, need_weights=False)
        h = self.norm2(h + a2)
        deep = self.head(h.flatten(1)).squeeze(-1)
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        return self.bias + linear + deep


class MMoE(nn.Module):
    def __init__(self, num_features, num_fields, num_tasks, initial_bias):
        super().__init__()
        self.num_tasks = num_tasks
        self.embedding = nn.Embedding(num_features, EMBED_DIM)
        input_dim = num_fields * EMBED_DIM
        expert_dim = 48
        num_experts = 4

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 96),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(96, expert_dim),
                nn.ReLU(),
            )
            for _ in range(num_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_experts) for _ in range(num_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(num_tasks)
        ])
        self.main_bias = nn.Parameter(
            torch.tensor(initial_bias, dtype=torch.float32)
        )
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x):
        base = self.embedding(x).flatten(1)
        expert_values = torch.stack(
            [expert(base) for expert in self.experts], dim=1
        )
        outputs = []
        for task in range(self.num_tasks):
            gate = torch.softmax(self.gates[task](base), dim=1).unsqueeze(-1)
            mixed = (gate * expert_values).sum(dim=1)
            logit = self.towers[task](mixed).squeeze(-1)
            if task == 0:
                logit = logit + self.main_bias
            outputs.append(logit)
        return torch.stack(outputs, dim=1)


def train_single_task(model, x_train, y_train, sample_weights, seed_offset):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    n = len(y_train)
    generator = torch.Generator()
    generator.manual_seed(SEED + seed_offset)

    model.train()
    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            logits = model(x_train[idx])
            losses = loss_fn(logits, y_train[idx])
            loss = (losses * sample_weights[idx]).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def train_multitask(model, x_train, targets, sample_weights):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    n = len(targets)
    generator = torch.Generator()
    generator.manual_seed(SEED + 300)

    # Main task remains dominant; auxiliary outcomes only shape shared experts.
    task_coefficients = torch.ones(targets.shape[1], dtype=torch.float32)
    if targets.shape[1] > 1:
        task_coefficients[1:] = 0.30

    model.train()
    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            logits = model(x_train[idx])
            losses = loss_fn(logits, targets[idx])
            weighted_tasks = losses * task_coefficients.unsqueeze(0)
            loss = (
                weighted_tasks.sum(dim=1)
                * sample_weights[idx]
                / task_coefficients.sum()
            ).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def predict(model, matrix, multitask=False):
    model.eval()
    result = np.empty(len(matrix), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(matrix), PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, len(matrix))
            xb = torch.from_numpy(matrix[start:end])
            logits = model(xb)
            if multitask:
                logits = logits[:, 0]
            result[start:end] = logits.cpu().numpy()
    return result


train = load("train")
valid = load("valid")

x_train_np = make_matrix(train)
x_valid_np = make_matrix(valid)
x_train = torch.from_numpy(x_train_np)
y_train_np = np.asarray(train.y, dtype=np.float32)
y_train = torch.from_numpy(y_train_np)
weight_np = recency_weights(train.date)
sample_weights = torch.from_numpy(weight_np)

num_features = int(sum(FEATURE_CARDINALITIES[f] for f in FIELDS))
num_fields = len(FIELDS)
positive_rate = float(y_train_np.mean())
initial_bias = float(
    np.log(
        np.clip(positive_rate, 1e-6, 1 - 1e-6)
        / np.clip(1 - positive_rate, 1e-6, 1 - 1e-6)
    )
)

models = {}
valid_raw = {}
standalone_metrics = {}

nfm = NFM(num_features, num_fields, initial_bias)
nfm = train_single_task(nfm, x_train, y_train, sample_weights, 10)
models["nfm"] = (nfm, False)
valid_raw["nfm"] = predict(nfm, x_valid_np)
standalone_metrics["nfm"] = evaluate(
    valid.user_id, valid.y, valid_raw["nfm"]
)

autoint = AutoInt(num_features, num_fields, initial_bias)
autoint = train_single_task(autoint, x_train, y_train, sample_weights, 20)
models["autoint"] = (autoint, False)
valid_raw["autoint"] = predict(autoint, x_valid_np)
standalone_metrics["autoint"] = evaluate(
    valid.user_id, valid.y, valid_raw["autoint"]
)

aux_names = [
    name for name in ("is_click", "is_like", "is_follow")
    if name in train.aux
][:2]
target_arrays = [y_train_np]
for name in aux_names:
    target_arrays.append(np.asarray(train.aux[name], dtype=np.float32))
multitask_targets = torch.from_numpy(
    np.ascontiguousarray(np.stack(target_arrays, axis=1), dtype=np.float32)
)

mmoe = MMoE(
    num_features, num_fields, multitask_targets.shape[1], initial_bias
)
mmoe = train_multitask(mmoe, x_train, multitask_targets, sample_weights)
models["mmoe"] = (mmoe, True)
valid_raw["mmoe"] = predict(mmoe, x_valid_np, multitask=True)
standalone_metrics["mmoe"] = evaluate(
    valid.user_id, valid.y, valid_raw["mmoe"]
)

shared_dir = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.load(
    os.path.join(shared_dir, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_valid_rank = rank_percentile(valid.user_id, inc_valid)

candidate_scores = {}
candidate_metrics = {}
candidate_description = {}

for name in ("nfm", "autoint", "mmoe"):
    raw = valid_raw[name]
    metric = standalone_metrics[name]
    key = name + "_standalone"
    candidate_scores[key] = raw
    candidate_metrics[key] = metric
    candidate_description[key] = (name, 1.0, False)

    own_rank = rank_percentile(valid.user_id, raw)
    for alpha in (0.25, 0.50, 0.75):
        blend = alpha * own_rank + (1.0 - alpha) * inc_valid_rank
        blend_key = f"{name}_blend_{alpha:.2f}"
        blend_metric = evaluate(valid.user_id, valid.y, blend)
        candidate_scores[blend_key] = blend
        candidate_metrics[blend_key] = blend_metric
        candidate_description[blend_key] = (name, alpha, True)

best_key = max(
    candidate_metrics,
    key=lambda k: float(candidate_metrics[k]["primary"])
)
best_metrics = candidate_metrics[best_key]
best_valid_scores = candidate_scores[best_key]
best_family, best_alpha, best_is_blend = candidate_description[best_key]

candidate_summary = {
    key: float(value["primary"])
    for key, value in candidate_metrics.items()
}
print("CANDIDATES " + json.dumps(candidate_summary, sort_keys=True))
print(
    "FINDINGS " + json.dumps({
        "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
        "weight_oldest_to_newest": [
            float(weight_np.min()), float(weight_np.max())
        ],
        "multitask_auxiliary_targets": aux_names,
        "best_candidate": best_key,
    }, sort_keys=True)
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    if best_is_blend:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(valid_raw[best_family], dtype=np.float64),
        )

# Score test using exactly the selected family and blend coefficient.
test = load("test")
x_test_np = make_matrix(test)
best_model, best_multitask = models[best_family]
test_raw = predict(best_model, x_test_np, multitask=best_multitask)

if best_is_blend:
    inc_test = np.load(
        os.path.join(shared_dir, "incumbent_test_scores.npy")
    ).astype(np.float64)
    own_test_rank = rank_percentile(test.user_id, test_raw)
    inc_test_rank = rank_percentile(test.user_id, inc_test)
    test_scores = (
        best_alpha * own_test_rank
        + (1.0 - best_alpha) * inc_test_rank
    )
else:
    test_scores = test_raw

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START_TIME
print(
    "METRICS " + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)