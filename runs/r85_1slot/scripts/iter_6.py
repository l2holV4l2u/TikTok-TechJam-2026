import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 73931
N_THREADS = min(8, os.cpu_count() or 1)
BATCH_SIZE = 8192
PRED_BATCH = 32768
EPOCHS = 3
HALF_LIFE_DAYS = 7.0

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(N_THREADS)

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "hour",
    "video_type",
    "user_active_degree",
    "onehot_feat3",
    "onehot_feat8",
]

CONTEXT_FIELDS = ["video_id", "author_id", "tag", "tab"]
CONTEXT_INDEX = [FIELDS.index(f) for f in CONTEXT_FIELDS]

OFFSETS = {}
running = 0
for field in FIELDS:
    OFFSETS[field] = running
    running += int(FEATURE_CARDINALITIES[field])
TOTAL_IDS = running
PAD_ID = TOTAL_IDS


def encoded_matrix(split):
    columns = []
    for field in FIELDS:
        values = np.asarray(split.X[field], dtype=np.int64)
        columns.append(values + OFFSETS[field])
    return np.column_stack(columns).astype(np.int64, copy=False)


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int32)
    day = dates % 100
    age = np.max(day) - day
    weights = np.exp2(-age.astype(np.float32) / HALF_LIFE_DAYS)
    weights /= np.mean(weights)
    return weights.astype(np.float32)


def predecessor_features(encoded_parts, user_parts, time_parts, target_start):
    x = np.concatenate(encoded_parts, axis=0)
    users = np.concatenate(
        [np.asarray(v, dtype=np.int64) for v in user_parts]
    )
    times = np.concatenate(
        [np.asarray(v, dtype=np.int64) for v in time_parts]
    )

    n = users.size
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, times, users))

    predecessor = np.full(n, -1, dtype=np.int64)
    if n > 1:
        left = order[:-1]
        right = order[1:]
        same_user = users[left] == users[right]
        predecessor[right[same_user]] = left[same_user]

    target_rows = np.arange(target_start, n, dtype=np.int64)
    pred_rows = predecessor[target_rows]

    previous = np.full(
        (target_rows.size, len(CONTEXT_INDEX)),
        PAD_ID,
        dtype=np.int64,
    )
    has_previous = pred_rows >= 0
    if np.any(has_previous):
        previous[has_previous] = x[pred_rows[has_previous]][
            :, CONTEXT_INDEX
        ]

    gap = np.zeros(target_rows.size, dtype=np.float32)
    if np.any(has_previous):
        delta_ms = (
            times[target_rows[has_previous]]
            - times[pred_rows[has_previous]]
        )
        delta_seconds = np.maximum(delta_ms, 0).astype(np.float64) / 1000.0
        gap[has_previous] = np.log1p(delta_seconds).astype(np.float32)
    gap = np.minimum(gap, np.float32(16.0)) / np.float32(16.0)

    return previous, gap


def get_aux_targets(split, names):
    targets = [np.asarray(split.y, dtype=np.float32)]
    for name in names:
        value = np.asarray(split.aux[name])
        value = np.nan_to_num(
            value.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        targets.append((value > 0).astype(np.float32))
    return np.column_stack(targets).astype(np.float32, copy=False)


class MMoE(nn.Module):
    def __init__(self, n_tasks, dim=12, hidden=72, n_experts=4):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_experts = n_experts
        self.embedding = nn.Embedding(TOTAL_IDS + 1, dim)
        input_dim = len(FIELDS) * dim

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden),
                    nn.SiLU(),
                    nn.Dropout(0.08),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                )
                for _ in range(n_experts)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(input_dim, n_experts) for _ in range(n_tasks)]
        )
        self.towers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, 32),
                    nn.SiLU(),
                    nn.Linear(32, 1),
                )
                for _ in range(n_tasks)
            ]
        )

        nn.init.normal_(self.embedding.weight, std=0.025)

    def forward(self, current, previous=None, gap=None):
        z = self.embedding(current).flatten(1)
        expert_values = torch.stack(
            [expert(z) for expert in self.experts], dim=1
        )

        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            weights = torch.softmax(gate(z), dim=1).unsqueeze(2)
            mixed = torch.sum(weights * expert_values, dim=1)
            outputs.append(tower(mixed).squeeze(1))
        return torch.stack(outputs, dim=1)


class CausalContextNet(nn.Module):
    def __init__(self, dim=12):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_IDS + 1, dim)
        current_dim = len(FIELDS) * dim
        previous_dim = len(CONTEXT_FIELDS) * dim

        self.current_projection = nn.Sequential(
            nn.Linear(current_dim, 80),
            nn.LayerNorm(80),
            nn.SiLU(),
        )
        self.context_projection = nn.Sequential(
            nn.Linear(previous_dim + 1, 48),
            nn.LayerNorm(48),
            nn.SiLU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(80 + 48 + 80 * 48 // 16, 96),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(96, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        self.current_interaction = nn.Linear(80, 16)
        self.context_interaction = nn.Linear(48, 16)

        nn.init.normal_(self.embedding.weight, std=0.025)

    def forward(self, current, previous, gap):
        current_vector = self.embedding(current).flatten(1)
        previous_vector = self.embedding(previous).flatten(1)

        current_hidden = self.current_projection(current_vector)
        context_hidden = self.context_projection(
            torch.cat([previous_vector, gap.unsqueeze(1)], dim=1)
        )

        left = self.current_interaction(current_hidden)
        right = self.context_interaction(context_hidden)
        interaction = (
            left.unsqueeze(2) * right.unsqueeze(1)
        ).flatten(1)

        combined = torch.cat(
            [current_hidden, context_hidden, interaction], dim=1
        )
        return self.predictor(combined).squeeze(1)


def fit_mmoe(x, targets, weights, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)

    model = MMoE(n_tasks=targets.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0025, weight_decay=2e-6
    )
    task_weights = torch.tensor(
        [1.0] + [0.30] * (targets.shape[1] - 1),
        dtype=torch.float32,
    )

    n = x.shape[0]
    for _ in range(EPOCHS):
        permutation = rng.permutation(n)
        model.train()

        for begin in range(0, n, BATCH_SIZE):
            rows = permutation[begin:begin + BATCH_SIZE]
            bx = torch.from_numpy(x[rows])
            by = torch.from_numpy(targets[rows])
            bw = torch.from_numpy(weights[rows])

            logits = model(bx)
            losses = F.binary_cross_entropy_with_logits(
                logits, by, reduction="none"
            )
            losses = losses * task_weights.unsqueeze(0)
            loss = (losses.mean(dim=1) * bw).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def fit_context(x, previous, gap, labels, weights, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)

    model = CausalContextNet()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0025, weight_decay=3e-6
    )

    labels = np.asarray(labels, dtype=np.float32)
    n = x.shape[0]

    for _ in range(EPOCHS):
        permutation = rng.permutation(n)
        model.train()

        for begin in range(0, n, BATCH_SIZE):
            rows = permutation[begin:begin + BATCH_SIZE]
            bx = torch.from_numpy(x[rows])
            bp = torch.from_numpy(previous[rows])
            bg = torch.from_numpy(gap[rows])
            by = torch.from_numpy(labels[rows])
            bw = torch.from_numpy(weights[rows])

            logits = model(bx, bp, bg)
            losses = F.binary_cross_entropy_with_logits(
                logits, by, reduction="none"
            )
            loss = (losses * bw).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.no_grad()
def predict_mmoe(model, x):
    model.eval()
    output = np.empty(x.shape[0], dtype=np.float32)
    for begin in range(0, x.shape[0], PRED_BATCH):
        end = min(begin + PRED_BATCH, x.shape[0])
        logits = model(torch.from_numpy(x[begin:end]))[:, 0]
        output[begin:end] = logits.cpu().numpy()
    return output


@torch.no_grad()
def predict_context(model, x, previous, gap):
    model.eval()
    output = np.empty(x.shape[0], dtype=np.float32)
    for begin in range(0, x.shape[0], PRED_BATCH):
        end = min(begin + PRED_BATCH, x.shape[0])
        logits = model(
            torch.from_numpy(x[begin:end]),
            torch.from_numpy(previous[begin:end]),
            torch.from_numpy(gap[begin:end]),
        )
        output[begin:end] = logits.cpu().numpy()
    return output


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = users.size
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, users))
    sorted_users = users[order]
    starts_mask = np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    starts = np.flatnonzero(starts_mask)
    group_number = np.cumsum(starts_mask) - 1
    positions = np.arange(n) - starts[group_number]
    sizes = np.diff(np.r_[starts, n])
    denominator = np.maximum(sizes[group_number] - 1, 1)

    ranked_sorted = positions.astype(np.float64) / denominator
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


train = load("train")
valid = load("valid")

x_train = encoded_matrix(train)
x_valid = encoded_matrix(valid)

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
train_users = np.asarray(train.user_id, dtype=np.int64)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

weights_train = recency_weights(train.date)

available_aux = []
for name in ["is_click", "is_like", "is_follow", "is_comment"]:
    if name in train.aux and name in valid.aux:
        available_aux.append(name)
available_aux = available_aux[:3]
if not available_aux:
    raise RuntimeError("No supported auxiliary training targets found")

targets_train = get_aux_targets(train, available_aux)

previous_train, gap_train = predecessor_features(
    [x_train],
    [train.user_id],
    [train.time_ms],
    0,
)
previous_valid, gap_valid = predecessor_features(
    [x_train, x_valid],
    [train.user_id, valid.user_id],
    [train.time_ms, valid.time_ms],
    x_train.shape[0],
)

mmoe_model = fit_mmoe(
    x_train,
    targets_train,
    weights_train,
    SEED + 100,
)
mmoe_valid = predict_mmoe(mmoe_model, x_valid).astype(np.float64)

context_model = fit_context(
    x_train,
    previous_train,
    gap_train,
    y_train,
    weights_train,
    SEED + 200,
)
context_valid = predict_context(
    context_model, x_valid, previous_valid, gap_valid
).astype(np.float64)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if inc_valid.size != y_valid.size:
    raise ValueError("Incumbent validation prediction length mismatch")
inc_valid_rank = within_user_rank(valid_users, inc_valid)

families = {
    "mmoe_auxiliary": mmoe_valid,
    "causal_context": context_valid,
}

candidate_results = {}
best_family = None
best_alpha = None
best_scores = None
best_raw = None
best_metrics = None

inc_metrics = evaluate(valid_users, y_valid, inc_valid)
candidate_results["trusted_incumbent"] = float(inc_metrics["primary"])

for family, raw_scores in families.items():
    raw_metrics = evaluate(valid_users, y_valid, raw_scores)
    candidate_results[family + "_raw"] = float(raw_metrics["primary"])

    own_rank = within_user_rank(valid_users, raw_scores)
    local_metrics = raw_metrics
    local_scores = raw_scores
    local_alpha = 1.0

    for alpha in np.linspace(0.0, 1.0, 21):
        blended = alpha * own_rank + (1.0 - alpha) * inc_valid_rank
        metrics = evaluate(valid_users, y_valid, blended)
        if float(metrics["primary"]) > float(local_metrics["primary"]):
            local_metrics = metrics
            local_scores = blended.copy()
            local_alpha = float(alpha)

    candidate_results[family + "_best_blend"] = float(
        local_metrics["primary"]
    )

    if best_metrics is None or float(local_metrics["primary"]) > float(
        best_metrics["primary"]
    ):
        best_family = family
        best_alpha = local_alpha
        best_scores = np.asarray(local_scores, dtype=np.float64)
        best_raw = np.asarray(raw_scores, dtype=np.float64)
        best_metrics = local_metrics

print("CANDIDATES " + json.dumps(candidate_results, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "selected_family": best_family,
            "own_rank_weight": best_alpha,
            "auxiliary_targets": available_aux,
            "epochs": EPOCHS,
            "recency_half_life_days": HALF_LIFE_DAYS,
            "causal_context_fields": CONTEXT_FIELDS,
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
    if best_alpha < 1.0 - 1e-12:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

del mmoe_model, context_model
gc.collect()

test = load("test")
x_test = encoded_matrix(test)
test_users = np.asarray(test.user_id, dtype=np.int64)

x_refit = np.concatenate([x_train, x_valid], axis=0)
refit_users = np.concatenate(
    [train_users, valid_users]
)
refit_times = np.concatenate(
    [
        np.asarray(train.time_ms, dtype=np.int64),
        np.asarray(valid.time_ms, dtype=np.int64),
    ]
)
refit_dates = np.concatenate(
    [
        np.asarray(train.date, dtype=np.int32),
        np.asarray(valid.date, dtype=np.int32),
    ]
)
y_refit = np.concatenate(
    [y_train, y_valid.astype(np.float32)]
)
weights_refit = recency_weights(refit_dates)

if best_family == "mmoe_auxiliary":
    targets_valid = get_aux_targets(valid, available_aux)
    targets_refit = np.concatenate(
        [targets_train, targets_valid], axis=0
    )
    refit_model = fit_mmoe(
        x_refit,
        targets_refit,
        weights_refit,
        SEED + 100,
    )
    own_test = predict_mmoe(refit_model, x_test).astype(np.float64)
else:
    previous_refit, gap_refit = predecessor_features(
        [x_refit],
        [refit_users],
        [refit_times],
        0,
    )
    previous_test, gap_test = predecessor_features(
        [x_refit, x_test],
        [refit_users, test.user_id],
        [refit_times, test.time_ms],
        x_refit.shape[0],
    )
    refit_model = fit_context(
        x_refit,
        previous_refit,
        gap_refit,
        y_refit,
        weights_refit,
        SEED + 200,
    )
    own_test = predict_context(
        refit_model, x_test, previous_test, gap_test
    ).astype(np.float64)

if best_alpha < 1.0 - 1e-12:
    inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
    if inc_test.size != x_test.shape[0]:
        raise ValueError("Incumbent test prediction length mismatch")
    own_test_rank = within_user_rank(test_users, own_test)
    inc_test_rank = within_user_rank(test_users, inc_test)
    test_scores = (
        best_alpha * own_test_rank
        + (1.0 - best_alpha) * inc_test_rank
    )
else:
    test_scores = own_test

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
        }
    )
)