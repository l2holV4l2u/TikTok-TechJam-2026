import os
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 1847
DEVICE = "cpu"
BATCH_SIZE = 8192
EPOCHS = 2
LR = 0.004
EMBED_DIM = 8

torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
np.random.seed(SEED)
torch.manual_seed(SEED)

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "onehot_feat3",
    "onehot_feat8",
    "upload_type",
    "music_type",
    "hour",
    "user_active_degree",
]
NUMERIC_FIELDS = [
    "duration_ms",
    "user_register_days",
]
AUXILIARY_CANDIDATES = [
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
]


def within_user_ranks(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    group_index = np.cumsum(starts_mask) - 1
    local_rank = np.arange(n, dtype=np.int64) - starts[group_index]
    sizes = np.diff(np.r_[starts, n])
    percentile = (
        local_rank.astype(np.float64) + 0.5
    ) / sizes[group_index].astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[order] = percentile
    return result


class FeatureScaler:
    def fit(self, split):
        self.center = {}
        self.scale = {}
        for name in NUMERIC_FIELDS:
            x = np.asarray(split.num[name], dtype=np.float64)
            x = np.log1p(np.maximum(np.nan_to_num(x, nan=0.0), 0.0))
            self.center[name] = float(np.median(x))
            q25, q75 = np.percentile(x, [25.0, 75.0])
            self.scale[name] = float(max(q75 - q25, 0.25))
        return self

    def transform(self, split):
        columns = []
        for name in NUMERIC_FIELDS:
            raw = np.asarray(split.num[name], dtype=np.float64)
            missing = ~np.isfinite(raw)
            x = np.log1p(np.maximum(np.nan_to_num(raw, nan=0.0), 0.0))
            x = (x - self.center[name]) / self.scale[name]
            x = np.clip(x, -6.0, 6.0)
            columns.append(x.astype(np.float32))
            columns.append(missing.astype(np.float32))
        return np.column_stack(columns).astype(np.float32)


def extract_categorical(split):
    return tuple(
        np.asarray(split.X[name], dtype=np.int64)
        for name in FIELDS
    )


def discover_auxiliary_tasks(train, valid):
    tasks = []
    train_keys = set(train.aux.keys())
    valid_keys = set(valid.aux.keys())
    for name in AUXILIARY_CANDIDATES:
        if name not in train_keys or name not in valid_keys:
            continue
        a = np.asarray(train.aux[name])
        b = np.asarray(valid.aux[name])
        if a.ndim != 1 or b.ndim != 1:
            continue
        sample = a[:min(len(a), 100000)]
        finite = sample[np.isfinite(sample)]
        if len(finite) == 0:
            continue
        unique = np.unique(finite)
        if np.all(np.isin(unique, [0, 1])):
            rate = float(np.mean(a))
            if 0.0002 < rate < 0.9998:
                tasks.append(name)
    return tasks[:4]


def make_targets(split, auxiliary_tasks, include_long_view=True):
    columns = []
    if include_long_view:
        columns.append(
            np.asarray(split.y, dtype=np.float32)
        )
    for name in auxiliary_tasks:
        x = np.asarray(split.aux[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0)
        columns.append(np.clip(x, 0.0, 1.0))
    return np.column_stack(columns).astype(np.float32)


class CombinedSplit:
    pass


def combine_splits(a, b):
    out = CombinedSplit()
    out.X = {
        name: np.concatenate([
            np.asarray(a.X[name]),
            np.asarray(b.X[name])
        ])
        for name in FIELDS
    }
    out.num = {
        name: np.concatenate([
            np.asarray(a.num[name]),
            np.asarray(b.num[name])
        ])
        for name in NUMERIC_FIELDS
    }
    out.aux = {}
    common_aux = set(a.aux.keys()).intersection(b.aux.keys())
    for name in common_aux:
        out.aux[name] = np.concatenate([
            np.asarray(a.aux[name]),
            np.asarray(b.aux[name])
        ])
    out.user_id = out.X["user_id"]
    out.video_id = out.X["video_id"]
    return out


class InputEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(int(FEATURE_CARDINALITIES[name]), EMBED_DIM)
            for name in FIELDS
        ])
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.035)

    @property
    def output_dim(self):
        return len(FIELDS) * EMBED_DIM + 2 * len(NUMERIC_FIELDS)

    def forward(self, categorical, numeric):
        pieces = [
            emb(categorical[:, i])
            for i, emb in enumerate(self.embeddings)
        ]
        pieces.append(numeric)
        return torch.cat(pieces, dim=1)


class SharedBottomMTL(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        self.encoder = InputEncoder()
        d = self.encoder.output_dim
        self.shared = nn.Sequential(
            nn.Linear(d, 48),
            nn.ReLU(),
            nn.LayerNorm(48),
            nn.Linear(48, 24),
            nn.ReLU(),
        )
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(24, 12),
                nn.ReLU(),
                nn.Linear(12, 1),
            )
            for _ in range(n_tasks)
        ])

    def forward(self, categorical, numeric):
        x = self.encoder(categorical, numeric)
        h = self.shared(x)
        return torch.cat([tower(h) for tower in self.towers], dim=1)


class MMoE(nn.Module):
    def __init__(self, n_tasks, n_experts=3):
        super().__init__()
        self.encoder = InputEncoder()
        d = self.encoder.output_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, 32),
                nn.ReLU(),
                nn.Linear(32, 24),
                nn.ReLU(),
            )
            for _ in range(n_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(d, n_experts)
            for _ in range(n_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(24, 12),
                nn.ReLU(),
                nn.Linear(12, 1),
            )
            for _ in range(n_tasks)
        ])

    def forward(self, categorical, numeric):
        x = self.encoder(categorical, numeric)
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )
        task_logits = []
        for gate, tower in zip(self.gates, self.towers):
            weights = torch.softmax(gate(x), dim=1).unsqueeze(2)
            representation = torch.sum(
                expert_outputs * weights, dim=1
            )
            task_logits.append(tower(representation))
        return torch.cat(task_logits, dim=1)


def task_positive_weights(targets):
    rates = np.mean(targets, axis=0)
    weights = np.ones(targets.shape[1], dtype=np.float32)
    for i in range(1, targets.shape[1]):
        odds = (1.0 - rates[i]) / max(rates[i], 1e-5)
        weights[i] = float(np.clip(math.sqrt(odds), 1.0, 5.0))
    return weights, rates


def fit_model(model_name, categorical, numeric, targets):
    torch.manual_seed(SEED)
    n_tasks = targets.shape[1]
    if model_name == "shared_bottom_multitask":
        model = SharedBottomMTL(n_tasks)
    elif model_name == "task_gated_mmoe":
        model = MMoE(n_tasks)
    else:
        raise ValueError(model_name)

    model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=2e-6
    )

    cat_tensor = torch.from_numpy(
        np.column_stack(categorical).astype(np.int64, copy=False)
    )
    num_tensor = torch.from_numpy(
        numeric.astype(np.float32, copy=False)
    )
    target_tensor = torch.from_numpy(
        targets.astype(np.float32, copy=False)
    )

    positive_weights, _ = task_positive_weights(targets)
    positive_weights_t = torch.from_numpy(positive_weights)
    task_weights = torch.ones(n_tasks, dtype=torch.float32)
    if n_tasks > 1:
        task_weights[1:] = 0.18

    n = len(targets)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    model.train()
    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:min(start + BATCH_SIZE, n)]
            optimizer.zero_grad(set_to_none=True)
            logits = model(cat_tensor[idx], num_tensor[idx])

            element_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                target_tensor[idx],
                reduction="none",
                pos_weight=positive_weights_t,
            )
            per_task = element_loss.mean(dim=0)
            loss = torch.sum(per_task * task_weights) / torch.sum(task_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def predict_all_tasks(model, categorical, numeric):
    cat_tensor = torch.from_numpy(
        np.column_stack(categorical).astype(np.int64, copy=False)
    )
    num_tensor = torch.from_numpy(
        numeric.astype(np.float32, copy=False)
    )
    n = len(numeric)
    result = None
    model.eval()
    with torch.no_grad():
        for start in range(0, n, 32768):
            end = min(start + 32768, n)
            logits = model(
                cat_tensor[start:end],
                num_tensor[start:end]
            ).cpu().numpy()
            if result is None:
                result = np.empty(
                    (n, logits.shape[1]), dtype=np.float32
                )
            result[start:end] = logits
    return result


train = load("train")
valid = load("valid")
y_valid = np.asarray(valid.y, dtype=np.int8)
auxiliary_tasks = discover_auxiliary_tasks(train, valid)

scaler = FeatureScaler().fit(train)
train_cat = extract_categorical(train)
valid_cat = extract_categorical(valid)
train_num = scaler.transform(train)
valid_num = scaler.transform(valid)
train_targets = make_targets(train, auxiliary_tasks)

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared_dir, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_valid_rank = within_user_ranks(valid.user_id, inc_valid)

candidate_scores = {}
model_predictions = {}
model_objects = {}

for model_name in ["shared_bottom_multitask", "task_gated_mmoe"]:
    model = fit_model(
        model_name, train_cat, train_num, train_targets
    )
    predictions = predict_all_tasks(
        model, valid_cat, valid_num
    )
    long_view_logits = predictions[:, 0].astype(np.float64)
    model_predictions[model_name] = long_view_logits
    model_objects[model_name] = model

    raw_metrics = evaluate(
        valid.user_id, y_valid, long_view_logits
    )
    candidate_scores[model_name + "_raw"] = float(
        raw_metrics["primary"]
    )

    # Auxiliary heads can carry a robust engagement ordering signal.
    if predictions.shape[1] > 1:
        auxiliary_mean = np.mean(predictions[:, 1:], axis=1)
        supervised_consensus = (
            0.85 * predictions[:, 0] + 0.15 * auxiliary_mean
        ).astype(np.float64)
        model_predictions[
            model_name + "_head_consensus"
        ] = supervised_consensus
        consensus_metrics = evaluate(
            valid.user_id, y_valid, supervised_consensus
        )
        candidate_scores[
            model_name + "_head_consensus_raw"
        ] = float(consensus_metrics["primary"])

blend_alphas = [0.15, 0.30, 0.50, 0.70, 1.00]
best_primary = -np.inf
best_metrics = None
best_scores = None
best_raw = None
best_prediction_name = None
best_architecture = None
best_alpha = None
best_use_consensus = False

for prediction_name, raw_scores in model_predictions.items():
    architecture = (
        "task_gated_mmoe"
        if prediction_name.startswith("task_gated_mmoe")
        else "shared_bottom_multitask"
    )
    use_consensus = prediction_name.endswith("_head_consensus")
    raw_rank = within_user_ranks(valid.user_id, raw_scores)

    for alpha in blend_alphas:
        blended = (
            (1.0 - alpha) * inc_valid_rank +
            alpha * raw_rank
        )
        metrics = evaluate(valid.user_id, y_valid, blended)
        name = prediction_name + "_blend_" + str(alpha)
        candidate_scores[name] = float(metrics["primary"])

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_metrics = metrics
            best_scores = blended.copy()
            best_raw = raw_scores.copy()
            best_prediction_name = prediction_name
            best_architecture = architecture
            best_alpha = float(alpha)
            best_use_consensus = bool(use_consensus)

rates = {
    "long_view": float(np.mean(train_targets[:, 0]))
}
for i, name in enumerate(auxiliary_tasks, start=1):
    rates[name] = float(np.mean(train_targets[:, i]))

print("CANDIDATES " + json.dumps(
    candidate_scores, sort_keys=True
))
print("FINDINGS " + json.dumps({
    "auxiliary_tasks": auxiliary_tasks,
    "task_rates": rates,
    "winner": best_prediction_name,
    "winner_architecture": best_architecture,
    "blend_alpha": best_alpha,
    "incumbent_primary": float(
        evaluate(valid.user_id, y_valid, inc_valid)["primary"]
    ),
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64)
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw, dtype=np.float64)
    )

# Refit the validation-selected recipe on train + validation.
combined = combine_splits(train, valid)
combined.y = np.concatenate([
    np.asarray(train.y, dtype=np.float32),
    np.asarray(valid.y, dtype=np.float32),
])
combined_scaler = FeatureScaler().fit(combined)
combined_cat = extract_categorical(combined)
combined_num = combined_scaler.transform(combined)
combined_targets = make_targets(combined, auxiliary_tasks)

for name in list(model_objects.keys()):
    del model_objects[name]

refit_model = fit_model(
    best_architecture,
    combined_cat,
    combined_num,
    combined_targets,
)

test = load("test")
test_cat = extract_categorical(test)
test_num = combined_scaler.transform(test)
test_task_predictions = predict_all_tasks(
    refit_model, test_cat, test_num
)

if best_use_consensus and test_task_predictions.shape[1] > 1:
    test_raw = (
        0.85 * test_task_predictions[:, 0] +
        0.15 * np.mean(test_task_predictions[:, 1:], axis=1)
    ).astype(np.float64)
else:
    test_raw = test_task_predictions[:, 0].astype(np.float64)

inc_test = np.load(
    os.path.join(shared_dir, "incumbent_test_scores.npy")
).astype(np.float64)
inc_test_rank = within_user_ranks(test.user_id, inc_test)
test_raw_rank = within_user_ranks(test.user_id, test_raw)
test_scores = (
    (1.0 - best_alpha) * inc_test_rank +
    best_alpha * test_raw_rank
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))