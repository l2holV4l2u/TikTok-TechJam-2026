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
SEED = 18437
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
EMBED_DIM = 10
HISTORY_LEN = 8
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768
EPOCHS = 3
HALF_LIFE_DAYS = 4.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def make_offsets():
    offsets = []
    total = 0
    for name in FIELDS:
        offsets.append(total)
        total += int(FEATURE_CARDINALITIES[name])
    return np.asarray(offsets, dtype=np.int64), total


OFFSETS, TOTAL_CARDINALITY = make_offsets()
AUTHOR_FIELD_INDEX = FIELDS.index("author_id")
AUTHOR_OFFSET = int(OFFSETS[AUTHOR_FIELD_INDEX])


def encode(split):
    n = len(split.user_id)
    result = np.empty((n, len(FIELDS)), dtype=np.int64)
    for j, name in enumerate(FIELDS):
        values = np.asarray(split.X[name], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[name])
        if values.size and (values.min() < 0 or values.max() >= card):
            raise ValueError("Out-of-range values for " + name)
        result[:, j] = values + OFFSETS[j]
    return result


def positive_history_index(train, encoded_train, target_split=None):
    """
    For train rows, return causal histories containing only positive authors
    strictly before the current impression. For valid/test, use the final
    train-only positive history for that user.
    """
    users = np.asarray(train.user_id, dtype=np.int64)
    rows = np.arange(users.size, dtype=np.int64)
    times = np.asarray(train.time_ms, dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.int8)

    order = np.lexsort((rows, times, users))
    sorted_users = users[order]
    sorted_positive = labels[order] > 0

    max_user = max(
        int(users.max()) if users.size else 0,
        int(np.asarray(target_split.user_id).max())
        if target_split is not None and len(target_split.user_id) else 0,
    )
    positive_counts = np.bincount(
        sorted_users[sorted_positive], minlength=max_user + 1
    ).astype(np.int64)
    positive_bases = np.zeros(max_user + 1, dtype=np.int64)
    if positive_bases.size > 1:
        positive_bases[1:] = np.cumsum(positive_counts[:-1])

    positive_tokens = encoded_train[order[sorted_positive], AUTHOR_FIELD_INDEX]

    if target_split is None:
        cumulative = np.cumsum(sorted_positive.astype(np.int64))
        new_user = np.empty(sorted_users.size, dtype=bool)
        new_user[0] = True
        new_user[1:] = sorted_users[1:] != sorted_users[:-1]

        cumulative_before_group = np.zeros(sorted_users.size, dtype=np.int64)
        starts = np.flatnonzero(new_user)
        previous_global = np.zeros(starts.size, dtype=np.int64)
        has_previous = starts > 0
        previous_global[has_previous] = cumulative[starts[has_previous] - 1]
        cumulative_before_group[starts] = previous_global
        cumulative_before_group = np.maximum.accumulate(cumulative_before_group)

        inclusive_in_group = cumulative - cumulative_before_group
        prior_count = inclusive_in_group - sorted_positive.astype(np.int64)

        seq_sorted = np.zeros(
            (sorted_users.size, HISTORY_LEN), dtype=np.int64
        )
        mask_sorted = np.zeros(
            (sorted_users.size, HISTORY_LEN), dtype=np.float32
        )

        for column, distance in enumerate(range(HISTORY_LEN, 0, -1)):
            valid = prior_count >= distance
            indices = (
                positive_bases[sorted_users[valid]]
                + prior_count[valid]
                - distance
            )
            seq_sorted[valid, column] = positive_tokens[indices]
            mask_sorted[valid, column] = 1.0

        sequence = np.zeros_like(seq_sorted)
        mask = np.zeros_like(mask_sorted)
        sequence[order] = seq_sorted
        mask[order] = mask_sorted
        return sequence, mask

    target_users = np.asarray(target_split.user_id, dtype=np.int64)
    available = positive_counts[target_users]
    sequence = np.zeros((target_users.size, HISTORY_LEN), dtype=np.int64)
    mask = np.zeros((target_users.size, HISTORY_LEN), dtype=np.float32)

    for column, distance in enumerate(range(HISTORY_LEN, 0, -1)):
        valid = available >= distance
        indices = (
            positive_bases[target_users[valid]]
            + available[valid]
            - distance
        )
        sequence[valid, column] = positive_tokens[indices]
        mask[valid, column] = 1.0

    return sequence, mask


def choose_auxiliary_targets(train):
    preferred = [
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_collect",
    ]
    chosen = []
    arrays = []

    for name in preferred:
        if name not in train.aux:
            continue
        values = np.asarray(train.aux[name])
        if values.ndim != 1 or values.shape[0] != len(train.user_id):
            continue
        finite = np.isfinite(values)
        if not finite.any():
            continue
        target = np.zeros(values.shape[0], dtype=np.float32)
        target[finite] = (values[finite] > 0).astype(np.float32)
        if 0.001 < float(target.mean()) < 0.999:
            chosen.append(name)
            arrays.append(target)
        if len(chosen) == 2:
            break

    if not arrays:
        raise RuntimeError("No suitable binary auxiliary training outcomes found")

    return chosen, np.stack(arrays, axis=1)


class DIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, EMBED_DIM)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.025)
        width = len(FIELDS) * EMBED_DIM + 3 * EMBED_DIM
        self.network = nn.Sequential(
            nn.Linear(width, 96),
            nn.ReLU(),
            nn.Linear(96, 40),
            nn.ReLU(),
            nn.Linear(40, 1),
        )
        self.wide = nn.Embedding(TOTAL_CARDINALITY, 1)
        nn.init.zeros_(self.wide.weight)

    def forward(self, x, sequence, mask):
        fields = self.embedding(x)
        candidate = fields[:, AUTHOR_FIELD_INDEX, :]
        history = self.embedding(sequence)

        attention_logits = (
            history * candidate.unsqueeze(1)
        ).sum(dim=2) / math.sqrt(EMBED_DIM)
        attention_logits = attention_logits.masked_fill(mask <= 0, -1e4)
        attention = torch.softmax(attention_logits, dim=1) * mask
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = (history * attention.unsqueeze(2)).sum(dim=1)

        features = torch.cat([
            fields.reshape(x.shape[0], -1),
            candidate,
            pooled,
            candidate * pooled,
        ], dim=1)
        return (
            self.network(features).squeeze(1)
            + self.wide(x).sum(dim=1).squeeze(1)
        )


class GRUHistory(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, EMBED_DIM)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.025)
        self.gru = nn.GRU(EMBED_DIM, 20, batch_first=True)
        width = len(FIELDS) * EMBED_DIM + 20
        self.network = nn.Sequential(
            nn.Linear(width, 80),
            nn.ReLU(),
            nn.Linear(80, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.wide = nn.Embedding(TOTAL_CARDINALITY, 1)
        nn.init.zeros_(self.wide.weight)

    def forward(self, x, sequence, mask):
        fields = self.embedding(x)
        history = self.embedding(sequence) * mask.unsqueeze(2)
        outputs, _ = self.gru(history)
        lengths = mask.sum(dim=1).long()
        gather_at = (lengths - 1).clamp_min(0)
        encoded = outputs[
            torch.arange(x.shape[0], device=x.device), gather_at
        ]
        encoded = encoded * (lengths > 0).float().unsqueeze(1)
        features = torch.cat(
            [fields.reshape(x.shape[0], -1), encoded], dim=1
        )
        return (
            self.network(features).squeeze(1)
            + self.wide(x).sum(dim=1).squeeze(1)
        )


class MMoE(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        self.n_tasks = n_tasks
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, EMBED_DIM)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.025)
        width = len(FIELDS) * EMBED_DIM
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            for _ in range(4)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(width, len(self.experts)) for _ in range(n_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
            for _ in range(n_tasks)
        ])
        self.wide = nn.Embedding(TOTAL_CARDINALITY, n_tasks)
        nn.init.zeros_(self.wide.weight)

    def forward(self, x):
        z = self.embedding(x).reshape(x.shape[0], -1)
        experts = torch.stack([expert(z) for expert in self.experts], dim=1)
        outputs = []
        for task in range(self.n_tasks):
            gate = torch.softmax(self.gates[task](z), dim=1)
            mixed = (experts * gate.unsqueeze(2)).sum(dim=1)
            outputs.append(self.towers[task](mixed).squeeze(1))
        deep = torch.stack(outputs, dim=1)
        return deep + self.wide(x).sum(dim=1)


class PLE(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        self.n_tasks = n_tasks
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, EMBED_DIM)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.025)
        width = len(FIELDS) * EMBED_DIM

        def expert():
            return nn.Sequential(
                nn.Linear(width, 56),
                nn.ReLU(),
                nn.Linear(56, 28),
                nn.ReLU(),
            )

        self.shared_experts = nn.ModuleList([expert(), expert()])
        self.private_experts = nn.ModuleList([
            nn.ModuleList([expert(), expert()]) for _ in range(n_tasks)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(width, 4) for _ in range(n_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(nn.Linear(28, 14), nn.ReLU(), nn.Linear(14, 1))
            for _ in range(n_tasks)
        ])
        self.wide = nn.Embedding(TOTAL_CARDINALITY, n_tasks)
        nn.init.zeros_(self.wide.weight)

    def forward(self, x):
        z = self.embedding(x).reshape(x.shape[0], -1)
        shared = [expert(z) for expert in self.shared_experts]
        outputs = []
        for task in range(self.n_tasks):
            available = shared + [
                expert(z) for expert in self.private_experts[task]
            ]
            stack = torch.stack(available, dim=1)
            gate = torch.softmax(self.gates[task](z), dim=1)
            mixed = (stack * gate.unsqueeze(2)).sum(dim=1)
            outputs.append(self.towers[task](mixed).squeeze(1))
        deep = torch.stack(outputs, dim=1)
        return deep + self.wide(x).sum(dim=1)


def fit_sequence_model(model, x, sequence, mask, y, weights, seed):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-6)
    n = x.shape[0]
    generator = torch.Generator()
    generator.manual_seed(seed)

    for _ in range(EPOCHS):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[idx], sequence[idx], mask[idx])
            row_loss = F.binary_cross_entropy_with_logits(
                logits, y[idx], reduction="none"
            )
            loss = (row_loss * weights[idx]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def fit_multitask_model(model, x, targets, weights, seed):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-6)
    n = x.shape[0]
    generator = torch.Generator()
    generator.manual_seed(seed)
    task_weights = torch.ones(targets.shape[1], dtype=torch.float32)
    if targets.shape[1] > 1:
        task_weights[1:] = 0.25

    for _ in range(EPOCHS):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[idx])
            losses = F.binary_cross_entropy_with_logits(
                logits, targets[idx], reduction="none"
            )
            weighted_tasks = (losses * task_weights.unsqueeze(0)).sum(dim=1)
            loss = (weighted_tasks * weights[idx]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def predict_sequence(model, x, sequence, mask):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, x.shape[0], PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, x.shape[0])
            result[start:end] = model(
                torch.from_numpy(x[start:end]),
                torch.from_numpy(sequence[start:end]),
                torch.from_numpy(mask[start:end]),
            ).cpu().numpy().astype(np.float64)
    return result


def predict_multitask(model, x):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, x.shape[0], PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, x.shape[0])
            result[start:end] = model(
                torch.from_numpy(x[start:end])
            )[:, 0].cpu().numpy().astype(np.float64)
    return result


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    n = values.size
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, values, users))
    sorted_users = users[order]

    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(new_group)
    ends = np.r_[starts[1:], n]
    counts = ends - starts
    positions = np.arange(n) - np.repeat(starts, counts)
    ranked = (positions.astype(np.float64) + 0.5) / np.repeat(counts, counts)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


train = load("train")
valid = load("valid")

x_train_np = encode(train)
x_valid_np = encode(valid)
seq_train_np, mask_train_np = positive_history_index(train, x_train_np)
seq_valid_np, mask_valid_np = positive_history_index(
    train, x_train_np, target_split=valid
)

aux_names, aux_values = choose_auxiliary_targets(train)
long_target = np.asarray(train.y, dtype=np.float32)[:, None]
multi_targets_np = np.concatenate([long_target, aux_values], axis=1)

dates = np.asarray(train.date, dtype=np.int64)
age = dates.max() - dates
sample_weights_np = np.exp(
    -math.log(2.0) * age.astype(np.float64) / HALF_LIFE_DAYS
).astype(np.float32)
sample_weights_np /= sample_weights_np.mean()

x_train = torch.from_numpy(x_train_np)
seq_train = torch.from_numpy(seq_train_np)
mask_train = torch.from_numpy(mask_train_np)
y_train = torch.from_numpy(long_target[:, 0])
multi_targets = torch.from_numpy(multi_targets_np)
sample_weights = torch.from_numpy(sample_weights_np)

models = {
    "din_positive_author": DIN(),
    "gru_positive_author": GRUHistory(),
    "mmoe_auxiliary": MMoE(multi_targets_np.shape[1]),
    "ple_auxiliary": PLE(multi_targets_np.shape[1]),
}

fit_sequence_model(
    models["din_positive_author"], x_train, seq_train, mask_train,
    y_train, sample_weights, SEED + 11
)
fit_sequence_model(
    models["gru_positive_author"], x_train, seq_train, mask_train,
    y_train, sample_weights, SEED + 23
)
fit_multitask_model(
    models["mmoe_auxiliary"], x_train, multi_targets,
    sample_weights, SEED + 37
)
fit_multitask_model(
    models["ple_auxiliary"], x_train, multi_targets,
    sample_weights, SEED + 53
)

valid_predictions = {
    "din_positive_author": predict_sequence(
        models["din_positive_author"],
        x_valid_np, seq_valid_np, mask_valid_np
    ),
    "gru_positive_author": predict_sequence(
        models["gru_positive_author"],
        x_valid_np, seq_valid_np, mask_valid_np
    ),
    "mmoe_auxiliary": predict_multitask(
        models["mmoe_auxiliary"], x_valid_np
    ),
    "ple_auxiliary": predict_multitask(
        models["ple_auxiliary"], x_valid_np
    ),
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not (os.path.exists(inc_valid_path) and os.path.exists(inc_test_path)):
    raise FileNotFoundError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if inc_valid.shape[0] != len(valid.user_id):
    raise ValueError("Incumbent validation length mismatch")

candidate_scores = {}
candidate_specs = {}

inc_metric = evaluate(valid.user_id, valid.y, inc_valid)
candidate_scores["trusted_incumbent"] = float(inc_metric["primary"])
candidate_specs["trusted_incumbent"] = ("incumbent", None, 0.0)
inc_rank = within_user_rank(valid.user_id, inc_valid)

for name, prediction in valid_predictions.items():
    metric = evaluate(valid.user_id, valid.y, prediction)
    candidate_scores[name] = float(metric["primary"])
    candidate_specs[name] = ("raw", name, 1.0)

    own_rank = within_user_rank(valid.user_id, prediction)
    for alpha in [0.15, 0.25, 0.35, 0.50, 0.65, 0.80]:
        blend_name = "%s_rankblend_%.2f" % (name, alpha)
        blended = alpha * own_rank + (1.0 - alpha) * inc_rank
        blend_metric = evaluate(valid.user_id, valid.y, blended)
        candidate_scores[blend_name] = float(blend_metric["primary"])
        candidate_specs[blend_name] = ("blend", name, alpha)

winner = max(candidate_scores, key=candidate_scores.get)
winner_kind, winner_model, winner_alpha = candidate_specs[winner]

if winner_kind == "raw":
    valid_scores = valid_predictions[winner_model]
elif winner_kind == "blend":
    valid_scores = (
        winner_alpha * within_user_rank(
            valid.user_id, valid_predictions[winner_model]
        )
        + (1.0 - winner_alpha) * inc_rank
    )
else:
    valid_scores = inc_valid.copy()

metrics = evaluate(valid.user_id, valid.y, valid_scores)
best_raw_name = max(
    valid_predictions,
    key=lambda name: candidate_scores[name]
)

print("FINDINGS " + json.dumps({
    "auxiliary_targets": aux_names,
    "history_length": HISTORY_LEN,
    "half_life_days": HALF_LIFE_DAYS,
    "winner": winner,
    "best_raw_family": best_raw_name,
    "best_raw_primary": candidate_scores[best_raw_name],
    "incumbent_primary": candidate_scores["trusted_incumbent"],
}, sort_keys=True))
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64)
    )
    if winner_kind in ("blend", "incumbent"):
        raw_to_save = winner_model if winner_model is not None else best_raw_name
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(valid_predictions[raw_to_save], dtype=np.float64)
        )

test = load("test")
x_test_np = encode(test)
seq_test_np, mask_test_np = positive_history_index(
    train, x_train_np, target_split=test
)

test_predictions = {
    "din_positive_author": predict_sequence(
        models["din_positive_author"],
        x_test_np, seq_test_np, mask_test_np
    ),
    "gru_positive_author": predict_sequence(
        models["gru_positive_author"],
        x_test_np, seq_test_np, mask_test_np
    ),
    "mmoe_auxiliary": predict_multitask(
        models["mmoe_auxiliary"], x_test_np
    ),
    "ple_auxiliary": predict_multitask(
        models["ple_auxiliary"], x_test_np
    ),
}

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if inc_test.shape[0] != len(test.user_id):
    raise ValueError("Incumbent test length mismatch")

if winner_kind == "raw":
    test_scores = test_predictions[winner_model]
elif winner_kind == "blend":
    test_scores = (
        winner_alpha * within_user_rank(
            test.user_id, test_predictions[winner_model]
        )
        + (1.0 - winner_alpha) * within_user_rank(test.user_id, inc_test)
    )
else:
    test_scores = inc_test.copy()

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))