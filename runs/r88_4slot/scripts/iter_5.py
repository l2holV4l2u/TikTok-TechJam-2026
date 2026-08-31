import os
import time
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7319
EPOCHS = 3
BATCH_SIZE = 16384
HALF_LIFE_DAYS = 9.0

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
    "user_active_degree",
    "onehot_feat3",
    "onehot_feat8",
]
USER_FIELD_POS = [0, 9, 10, 11]
ITEM_FIELD_POS = [1, 2, 3, 4, 5, 6, 7, 8]
AUX_CANDIDATES = ["is_click", "is_like", "is_follow", "is_comment"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
TOTAL_CARD = int(sum(cards))


def encode(split):
    return np.stack(
        [
            np.asarray(split.X[name], dtype=np.int64) + offsets[j]
            for j, name in enumerate(FIELDS)
        ],
        axis=1,
    )


def available_aux_keys(split):
    return [
        k for k in AUX_CANDIDATES
        if k in split.aux and len(np.asarray(split.aux[k])) == len(split.user_id)
    ]


def make_targets(split, aux_keys):
    cols = [np.asarray(split.y, dtype=np.float32)]
    for key in aux_keys:
        value = np.asarray(split.aux[key], dtype=np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
        cols.append(np.clip(value, 0.0, 1.0))
    return np.stack(cols, axis=1).astype(np.float32, copy=False)


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int32)
    ds = dates.astype(str)
    day_ord = np.array(
        [
            np.datetime64(f"{v[:4]}-{v[4:6]}-{v[6:8]}", "D").astype(np.int64)
            for v in np.unique(ds)
        ],
        dtype=np.int64,
    )
    unique_dates = np.unique(dates)
    mapping = dict(zip(unique_dates.tolist(), day_ord.tolist()))
    ordinal = np.fromiter(
        (mapping[int(v)] for v in dates),
        count=len(dates),
        dtype=np.int64,
    )
    age = ordinal.max() - ordinal
    weights = np.exp2(-age.astype(np.float32) / HALF_LIFE_DAYS)
    weights /= max(float(weights.mean()), 1e-6)
    return weights.astype(np.float32)


class WideAdditive(nn.Module):
    def __init__(self, n_categories, prior):
        super().__init__()
        self.linear = nn.Embedding(n_categories, 1)
        nn.init.zeros_(self.linear.weight)
        p = min(max(float(prior), 1e-5), 1.0 - 1e-5)
        self.bias = nn.Parameter(
            torch.tensor(math.log(p / (1.0 - p)), dtype=torch.float32)
        )

    def forward(self, x):
        return self.bias + self.linear(x).squeeze(-1).sum(dim=1)


class TwoTower(nn.Module):
    def __init__(self, n_categories, prior, emb_dim=8, hidden=32):
        super().__init__()
        self.embedding = nn.Embedding(n_categories, emb_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.user_tower = nn.Sequential(
            nn.Linear(len(USER_FIELD_POS) * emb_dim, 48),
            nn.ReLU(),
            nn.Linear(48, hidden),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(len(ITEM_FIELD_POS) * emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden),
        )
        self.user_bias = nn.Linear(len(USER_FIELD_POS) * emb_dim, 1)
        self.item_bias = nn.Linear(len(ITEM_FIELD_POS) * emb_dim, 1)

        p = min(max(float(prior), 1e-5), 1.0 - 1e-5)
        self.global_bias = nn.Parameter(
            torch.tensor(math.log(p / (1.0 - p)), dtype=torch.float32)
        )
        self.scale = math.sqrt(hidden)

    def forward(self, x):
        e = self.embedding(x)
        ue = e[:, USER_FIELD_POS, :].reshape(len(x), -1)
        ie = e[:, ITEM_FIELD_POS, :].reshape(len(x), -1)
        u = self.user_tower(ue)
        v = self.item_tower(ie)
        return (
            (u * v).sum(dim=1) / self.scale
            + self.user_bias(ue).squeeze(1)
            + self.item_bias(ie).squeeze(1)
            + self.global_bias
        )


class MMoE(nn.Module):
    def __init__(self, n_categories, n_tasks, prior, emb_dim=6, n_experts=3):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_experts = n_experts
        self.embedding = nn.Embedding(n_categories, emb_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        input_dim = len(FIELDS) * emb_dim

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                )
                for _ in range(n_experts)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(input_dim, n_experts) for _ in range(n_tasks)]
        )
        self.heads = nn.ModuleList([nn.Linear(32, 1) for _ in range(n_tasks)])

        p = min(max(float(prior), 1e-5), 1.0 - 1e-5)
        with torch.no_grad():
            self.heads[0].bias.fill_(math.log(p / (1.0 - p)))

    def forward(self, x):
        z = self.embedding(x).reshape(len(x), -1)
        expert_values = torch.stack([expert(z) for expert in self.experts], dim=1)
        outputs = []
        for gate, head in zip(self.gates, self.heads):
            weights = torch.softmax(gate(z), dim=1).unsqueeze(2)
            shared = (weights * expert_values).sum(dim=1)
            outputs.append(head(shared).squeeze(1))
        return torch.stack(outputs, dim=1)


def construct_model(family, n_tasks, prior):
    torch.manual_seed(SEED)
    if family == "wide_additive":
        return WideAdditive(TOTAL_CARD, prior)
    if family == "two_tower":
        return TwoTower(TOTAL_CARD, prior)
    if family == "mmoe_multitask":
        return MMoE(TOTAL_CARD, n_tasks, prior)
    raise ValueError(family)


def train_model(family, x, targets, weights):
    model = construct_model(family, targets.shape[1], targets[:, 0].mean())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(targets)
    w_t = torch.from_numpy(weights)
    n = len(x)

    if family == "mmoe_multitask":
        task_weights = torch.ones(targets.shape[1], dtype=torch.float32)
        if targets.shape[1] > 1:
            task_weights[1:] = 0.20
    else:
        task_weights = None

    for epoch in range(EPOCHS):
        model.train()
        generator = torch.Generator()
        generator.manual_seed(SEED + epoch * 1009)
        order = torch.randperm(n, generator=generator)

        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:min(lo + BATCH_SIZE, n)]
            xb = x_t[idx]
            wb = w_t[idx]

            if family == "mmoe_multitask":
                logits = model(xb)
                loss_matrix = nn.functional.binary_cross_entropy_with_logits(
                    logits, y_t[idx], reduction="none"
                )
                per_row = (loss_matrix * task_weights).sum(dim=1)
                loss = (per_row * wb).sum() / (
                    wb.sum() * float(task_weights.sum()) + 1e-8
                )
            else:
                logits = model(xb)
                per_row = nn.functional.binary_cross_entropy_with_logits(
                    logits, y_t[idx, 0], reduction="none"
                )
                loss = (per_row * wb).sum() / (wb.sum() + 1e-8)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


def predict(model, family, x):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.inference_mode():
        for lo in range(0, len(x), BATCH_SIZE * 2):
            hi = min(lo + BATCH_SIZE * 2, len(x))
            logits = model(torch.from_numpy(x[lo:hi]))
            if family == "mmoe_multitask":
                logits = logits[:, 0]
            result[lo:hi] = logits.cpu().numpy()
    return result


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    order = np.lexsort((np.arange(len(scores)), scores, user_ids))
    sorted_users = user_ids[order]
    _, starts, counts = np.unique(
        sorted_users, return_index=True, return_counts=True
    )
    positions = np.arange(len(scores), dtype=np.float64) - np.repeat(starts, counts)
    denominators = np.maximum(np.repeat(counts, counts) - 1, 1)
    ranked = np.empty(len(scores), dtype=np.float64)
    ranked[order] = positions / denominators
    return ranked


train = load("train")
valid = load("valid")

aux_keys = available_aux_keys(train)
aux_keys = [k for k in aux_keys if k in valid.aux]

x_train = encode(train)
x_valid = encode(valid)
targets_train = make_targets(train, aux_keys)
train_weights = recency_weights(train.date)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared_dir, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared_dir, "incumbent_test_scores.npy")
inc_valid = np.load(inc_valid_path).astype(np.float64, copy=False)
inc_valid_rank = within_user_rank(valid_users, inc_valid)

families = ["wide_additive", "two_tower", "mmoe_multitask"]
models = {}
raw_predictions = {}
candidate_scores = {}
candidate_metrics = {}

best_name = None
best_family = None
best_weight = None
best_scores = None
best_raw_scores = None
best_metrics = None

for family in families:
    model = train_model(family, x_train, targets_train, train_weights)
    models[family] = model
    raw = predict(model, family, x_valid).astype(np.float64)
    raw_predictions[family] = raw

    metrics_raw = evaluate(valid_users, valid_y, raw)
    raw_name = family + "_raw"
    candidate_scores[raw_name] = float(metrics_raw["primary"])
    candidate_metrics[raw_name] = metrics_raw

    if best_metrics is None or metrics_raw["primary"] > best_metrics["primary"]:
        best_name = raw_name
        best_family = family
        best_weight = None
        best_scores = raw.copy()
        best_raw_scores = raw.copy()
        best_metrics = metrics_raw

    own_rank = within_user_rank(valid_users, raw)
    for weight in (0.20, 0.35, 0.50, 0.65, 0.80):
        blended = weight * own_rank + (1.0 - weight) * inc_valid_rank
        metrics_blend = evaluate(valid_users, valid_y, blended)
        name = family + "_blend_" + str(weight)
        candidate_scores[name] = float(metrics_blend["primary"])
        candidate_metrics[name] = metrics_blend

        if metrics_blend["primary"] > best_metrics["primary"]:
            best_name = name
            best_family = family
            best_weight = float(weight)
            best_scores = blended.copy()
            best_raw_scores = raw.copy()
            best_metrics = metrics_blend

print(
    "FINDINGS "
    + json.dumps(
        {
            "auxiliary_tasks": aux_keys,
            "selected": best_name,
            "half_life_days": HALF_LIFE_DAYS,
        },
        separators=(", ", ": "),
    )
)
print(
    "CANDIDATES "
    + json.dumps(candidate_scores, sort_keys=True, separators=(", ", ": "))
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_weight is not None:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(best_raw_scores, dtype=np.float64),
        )

# Refit the selected family on train plus validation, preserving the same
# architecture, recency rule, auxiliary tasks, epoch count, and blend weight.
targets_valid = make_targets(valid, aux_keys)
x_fit = np.concatenate([x_train, x_valid], axis=0)
targets_fit = np.concatenate([targets_train, targets_valid], axis=0)
fit_dates = np.concatenate(
    [np.asarray(train.date), np.asarray(valid.date)], axis=0
)
fit_weights = recency_weights(fit_dates)

test_model = train_model(best_family, x_fit, targets_fit, fit_weights)

test = load("test")
x_test = encode(test)
test_raw = predict(test_model, best_family, x_test).astype(np.float64)

if best_weight is None:
    test_scores = test_raw
else:
    inc_test = np.load(inc_test_path).astype(np.float64, copy=False)
    own_test_rank = within_user_rank(np.asarray(test.user_id), test_raw)
    inc_test_rank = within_user_rank(np.asarray(test.user_id), inc_test)
    test_scores = (
        best_weight * own_test_rank
        + (1.0 - best_weight) * inc_test_rank
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