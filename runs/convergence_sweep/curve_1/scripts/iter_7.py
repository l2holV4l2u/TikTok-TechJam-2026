import os
import time
import json
import random
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 8))

DEVICE = torch.device("cpu")
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
VIDEO_INDEX = FIELDS.index("video_id")
DIM = 12
HISTORY_LENGTH = 8
BATCH_SIZE = 8192
EPOCHS = 2
LR = 0.002
HALF_LIFE = 2.5
BLEND_WEIGHTS = [0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85]


def recency_weights(dates, half_life=HALF_LIFE):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    positions = np.searchsorted(unique_dates, dates)
    age = unique_dates.size - 1 - positions
    weights = np.exp2(-age.astype(np.float32) / np.float32(half_life))
    weights /= weights.mean()
    return weights.astype(np.float32)


def stack_fields(split):
    return np.column_stack([
        np.asarray(split.X[name], dtype=np.int64)
        for name in FIELDS
    ])


def make_causal_history(train, target=None):
    if target is None:
        users = np.asarray(train.user_id, dtype=np.int64)
        times = np.asarray(train.time_ms, dtype=np.int64)
        videos = np.asarray(train.video_id, dtype=np.int64)
        offset = 0
    else:
        train_n = len(train.user_id)
        users = np.concatenate([
            np.asarray(train.user_id, dtype=np.int64),
            np.asarray(target.user_id, dtype=np.int64),
        ])
        times = np.concatenate([
            np.asarray(train.time_ms, dtype=np.int64),
            np.asarray(target.time_ms, dtype=np.int64),
        ])
        videos = np.concatenate([
            np.asarray(train.video_id, dtype=np.int64),
            np.asarray(target.video_id, dtype=np.int64),
        ])
        offset = train_n

    n = users.size
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, times, users))
    ordered_users = users[order]
    ordered_videos = videos[order]

    ordered_history = np.zeros((n, HISTORY_LENGTH), dtype=np.int64)
    for lag in range(1, HISTORY_LENGTH + 1):
        same = ordered_users[lag:] == ordered_users[:-lag]
        destinations = np.flatnonzero(same) + lag
        ordered_history[destinations, lag - 1] = ordered_videos[:-lag][same]

    history = np.empty_like(ordered_history)
    history[order] = ordered_history
    return history[offset:]


def init_embedding(embedding, padding=False):
    nn.init.normal_(embedding.weight, mean=0.0, std=0.025)
    if padding:
        with torch.no_grad():
            embedding.weight[0].zero_()


class CategoricalBackbone(nn.Module):
    def __init__(self, cardinalities, dim=DIM):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, dim)
            for cardinality in cardinalities
        ])
        self.linear = nn.ModuleList([
            nn.Embedding(cardinality, 1)
            for cardinality in cardinalities
        ])
        self.bias = nn.Parameter(torch.zeros(()))
        for embedding in self.embeddings:
            init_embedding(embedding)
        for embedding in self.linear:
            init_embedding(embedding)

    def embed(self, x):
        return torch.stack([
            embedding(x[:, index])
            for index, embedding in enumerate(self.embeddings)
        ], dim=1)

    def wide(self, x):
        result = self.bias.expand(x.shape[0])
        for index, embedding in enumerate(self.linear):
            result = result + embedding(x[:, index]).squeeze(1)
        return result


class XDeepFM(nn.Module):
    """CIN explicitly builds bounded high-order field interactions."""

    def __init__(self, cardinalities, dim=DIM):
        super().__init__()
        self.backbone = CategoricalBackbone(cardinalities, dim)
        fields = len(cardinalities)
        cin_width = 12

        self.cin1 = nn.Linear(fields * fields, cin_width)
        self.cin2 = nn.Linear(fields * cin_width, cin_width)
        self.deep = nn.Sequential(
            nn.Linear(fields * dim, 80),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(80, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.cin_output = nn.Linear(2 * cin_width, 1)

    def cin_layer(self, x0, hidden, projection):
        products = torch.einsum("bfd,bhd->bdfh", x0, hidden)
        products = products.flatten(start_dim=2)
        return F.relu(projection(products)).transpose(1, 2)

    def forward(self, x, history=None):
        x0 = self.backbone.embed(x)
        h1 = self.cin_layer(x0, x0, self.cin1)
        h2 = self.cin_layer(x0, h1, self.cin2)
        cin_features = torch.cat([h1.sum(dim=2), h2.sum(dim=2)], dim=1)
        return (
            self.backbone.wide(x)
            + self.deep(x0.flatten(start_dim=1)).squeeze(1)
            + self.cin_output(cin_features).squeeze(1)
        )


class FiBiNET(nn.Module):
    """Squeeze-excitation reweights fields before bilinear interactions."""

    def __init__(self, cardinalities, dim=DIM):
        super().__init__()
        self.backbone = CategoricalBackbone(cardinalities, dim)
        fields = len(cardinalities)
        pairs = fields * (fields - 1) // 2

        self.squeeze = nn.Sequential(
            nn.Linear(fields, max(4, fields)),
            nn.ReLU(),
            nn.Linear(max(4, fields), fields),
            nn.Sigmoid(),
        )
        self.bilinear = nn.ModuleList([
            nn.Linear(dim, dim, bias=False)
            for _ in range(fields)
        ])
        self.tower = nn.Sequential(
            nn.Linear(fields * dim + pairs * dim, 112),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(112, 40),
            nn.ReLU(),
            nn.Linear(40, 1),
        )

    def forward(self, x, history=None):
        embeddings = self.backbone.embed(x)
        field_summary = embeddings.mean(dim=2)
        gates = self.squeeze(field_summary).unsqueeze(2)
        reweighted = embeddings * gates

        interactions = []
        fields = reweighted.shape[1]
        for left in range(fields):
            transformed = self.bilinear[left](reweighted[:, left])
            for right in range(left + 1, fields):
                interactions.append(transformed * reweighted[:, right])

        tower_input = torch.cat(
            [reweighted.flatten(start_dim=1)] + interactions,
            dim=1,
        )
        return self.backbone.wide(x) + self.tower(tower_input).squeeze(1)


class GRU4RecExposure(nn.Module):
    """
    A recurrent state summarizes ordered prior video exposures. Padding steps
    do not update the state, so short validation histories remain well defined.
    """

    def __init__(self, cardinalities, video_cardinality, dim=DIM):
        super().__init__()
        self.backbone = CategoricalBackbone(cardinalities, dim)
        self.sequence_embedding = nn.Embedding(
            video_cardinality, dim, padding_idx=0
        )
        init_embedding(self.sequence_embedding, padding=True)
        self.cell = nn.GRUCell(dim, dim)
        self.tower = nn.Sequential(
            nn.Linear(len(cardinalities) * dim + 2 * dim, 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, history):
        fields = self.backbone.embed(x)
        sequence_ids = torch.flip(history, dims=[1])
        sequence = self.sequence_embedding(sequence_ids)

        state = torch.zeros(
            x.shape[0], sequence.shape[2],
            device=x.device, dtype=sequence.dtype
        )
        for step in range(sequence.shape[1]):
            proposed = self.cell(sequence[:, step], state)
            mask = (sequence_ids[:, step] != 0).unsqueeze(1)
            state = torch.where(mask, proposed, state)

        current_video = fields[:, VIDEO_INDEX]
        tower_input = torch.cat([
            fields.flatten(start_dim=1),
            state,
            state * current_video,
        ], dim=1)
        return self.backbone.wide(x) + self.tower(tower_input).squeeze(1)


class PLE(nn.Module):
    """
    One-level progressive layered extraction with shared and task-specific
    experts for long-view, click, and like. Only the long-view head is scored.
    """

    def __init__(self, cardinalities, dim=DIM, tasks=3):
        super().__init__()
        self.backbone = CategoricalBackbone(cardinalities, dim)
        input_dim = len(cardinalities) * dim
        expert_dim = 40
        self.tasks = tasks

        def expert():
            return nn.Sequential(
                nn.Linear(input_dim, 72),
                nn.ReLU(),
                nn.Linear(72, expert_dim),
                nn.ReLU(),
            )

        self.shared_experts = nn.ModuleList([expert(), expert()])
        self.task_experts = nn.ModuleList([
            nn.ModuleList([expert(), expert()])
            for _ in range(tasks)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, 4)
            for _ in range(tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(tasks)
        ])

    def forward(self, x, history=None):
        flat = self.backbone.embed(x).flatten(start_dim=1)
        shared = [expert(flat) for expert in self.shared_experts]
        outputs = []

        for task in range(self.tasks):
            specific = [
                expert(flat) for expert in self.task_experts[task]
            ]
            candidates = torch.stack(specific + shared, dim=1)
            gate = torch.softmax(self.gates[task](flat), dim=1)
            representation = torch.sum(
                candidates * gate.unsqueeze(2), dim=1
            )
            outputs.append(
                self.towers[task](representation).squeeze(1)
            )

        return torch.stack(outputs, dim=1)


def train_model(model, x, history, labels, weights, name, multitask=False):
    model.to(DEVICE)
    model.train()

    x_tensor = torch.from_numpy(x)
    history_tensor = torch.from_numpy(history)
    label_tensor = torch.from_numpy(labels.astype(np.float32, copy=False))
    weight_tensor = torch.from_numpy(weights.astype(np.float32, copy=False))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=1e-6
    )
    rng = np.random.RandomState(SEED + sum(ord(c) for c in name))
    n = x.shape[0]

    for epoch in range(EPOCHS):
        permutation = rng.permutation(n)
        total_loss = 0.0
        total_weight = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = torch.from_numpy(permutation[start:start + BATCH_SIZE])
            xb = x_tensor.index_select(0, idx).to(DEVICE)
            hb = history_tensor.index_select(0, idx).to(DEVICE)
            yb = label_tensor.index_select(0, idx).to(DEVICE)
            wb = weight_tensor.index_select(0, idx).to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, hb)

            if multitask:
                row_loss = F.binary_cross_entropy_with_logits(
                    logits, yb, reduction="none"
                )
                task_weights = torch.tensor(
                    [1.0, 0.30, 0.20],
                    dtype=row_loss.dtype,
                    device=row_loss.device,
                )
                row_loss = (
                    row_loss * task_weights.unsqueeze(0)
                ).sum(dim=1) / task_weights.sum()
            else:
                row_loss = F.binary_cross_entropy_with_logits(
                    logits, yb, reduction="none"
                )

            loss = torch.sum(row_loss * wb) / torch.sum(wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(torch.sum(row_loss * wb).detach())
            total_weight += float(torch.sum(wb))

        print(
            "FINDINGS family={} epoch={} weighted_loss={:.6f}".format(
                name, epoch + 1, total_loss / max(total_weight, 1e-8)
            ),
            flush=True,
        )

    return model


@torch.no_grad()
def predict(model, x, history, multitask=False):
    model.eval()
    x_tensor = torch.from_numpy(x)
    history_tensor = torch.from_numpy(history)
    result = np.empty(x.shape[0], dtype=np.float64)

    for start in range(0, x.shape[0], BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, x.shape[0])
        logits = model(
            x_tensor[start:end].to(DEVICE),
            history_tensor[start:end].to(DEVICE),
        )
        if multitask:
            logits = logits[:, 0]
        result[start:end] = logits.cpu().numpy().astype(np.float64)

    return result


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    n = users.size
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, values, users))
    ordered_users = users[order]
    starts = np.r_[
        0,
        np.flatnonzero(ordered_users[1:] != ordered_users[:-1]) + 1,
    ]
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n) - repeated_starts
    ordered_ranks = positions / np.maximum(repeated_lengths - 1, 1)

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ordered_ranks
    return ranks


train = load("train")
valid = load("valid")

x_train = stack_fields(train)
x_valid = stack_fields(valid)
train_history = make_causal_history(train)
valid_history = make_causal_history(train, valid)

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
weights = recency_weights(train.date)

print(
    "FINDINGS half_life={} effective_rows={:.0f} weight_range={:.4f}:{:.4f}".format(
        HALF_LIFE,
        float(weights.sum() ** 2 / np.square(weights).sum()),
        float(weights.min()),
        float(weights.max()),
    ),
    flush=True,
)

cardinalities = [FEATURE_CARDINALITIES[name] for name in FIELDS]

shared = os.environ.get("SHARED_ARTIFACTS", "")
incumbent_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
incumbent_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)
incumbent_valid = np.load(incumbent_valid_path).astype(np.float64)
incumbent_valid_rank = within_user_rank(valid.user_id, incumbent_valid)

candidate_scores = {}
best_primary = -np.inf
best_scores = None
best_model = None
best_name = None
best_alpha = 1.0
best_multitask = False
best_own_valid = None

aux_click = np.asarray(train.aux["is_click"], dtype=np.float32)
aux_like = np.asarray(train.aux["is_like"], dtype=np.float32)
multi_labels = np.column_stack([y_train, aux_click, aux_like]).astype(
    np.float32
)

families = [
    (
        "xdeepfm",
        lambda: XDeepFM(cardinalities),
        y_train,
        False,
    ),
    (
        "fibinet",
        lambda: FiBiNET(cardinalities),
        y_train,
        False,
    ),
    (
        "gru4rec_exposure",
        lambda: GRU4RecExposure(
            cardinalities,
            FEATURE_CARDINALITIES["video_id"],
        ),
        y_train,
        False,
    ),
    (
        "ple_multitask",
        lambda: PLE(cardinalities),
        multi_labels,
        True,
    ),
]

for family_name, constructor, labels, multitask in families:
    model = constructor()
    model = train_model(
        model,
        x_train,
        train_history,
        labels,
        weights,
        family_name,
        multitask=multitask,
    )

    raw_valid = predict(
        model, x_valid, valid_history, multitask=multitask
    )
    own_rank = within_user_rank(valid.user_id, raw_valid)
    own_metrics = evaluate(valid.user_id, y_valid, own_rank)
    candidate_scores[family_name] = float(own_metrics["primary"])

    family_best_primary = float(own_metrics["primary"])
    family_best_scores = own_rank
    family_best_alpha = 1.0
    family_best_label = family_name

    for alpha in BLEND_WEIGHTS:
        blended = (
            alpha * own_rank
            + (1.0 - alpha) * incumbent_valid_rank
        )
        metrics = evaluate(valid.user_id, y_valid, blended)
        label = "{}_blend_{:.2f}".format(family_name, alpha)
        candidate_scores[label] = float(metrics["primary"])

        if float(metrics["primary"]) > family_best_primary:
            family_best_primary = float(metrics["primary"])
            family_best_scores = blended
            family_best_alpha = alpha
            family_best_label = label

    print(
        "FINDINGS family={} standalone={:.6f} selected={} selected_primary={:.6f}".format(
            family_name,
            float(own_metrics["primary"]),
            family_best_label,
            family_best_primary,
        ),
        flush=True,
    )

    if family_best_primary > best_primary:
        if best_model is not None:
            del best_model
        best_primary = family_best_primary
        best_scores = family_best_scores.copy()
        best_model = model
        best_name = family_name
        best_alpha = family_best_alpha
        best_multitask = multitask
        best_own_valid = own_rank.copy()
    else:
        del model

    gc.collect()

final_metrics = evaluate(valid.user_id, y_valid, best_scores)

print(
    "CANDIDATES " + json.dumps(
        candidate_scores, sort_keys=True
    ),
    flush=True,
)
print(
    "FINDINGS winner={} own_weight={:.2f} primary={:.6f}".format(
        best_name, best_alpha, float(final_metrics["primary"])
    ),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_alpha < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_own_valid, dtype=np.float64),
        )

test = load("test")
x_test = stack_fields(test)
test_history = make_causal_history(train, test)
raw_test = predict(
    best_model, x_test, test_history, multitask=best_multitask
)
own_test_rank = within_user_rank(test.user_id, raw_test)

if best_alpha < 1.0:
    incumbent_test = np.load(incumbent_test_path).astype(np.float64)
    incumbent_test_rank = within_user_rank(
        test.user_id, incumbent_test
    )
    test_scores = (
        best_alpha * own_test_rank
        + (1.0 - best_alpha) * incumbent_test_rank
    )
else:
    test_scores = own_test_rank

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.4f}'
    % (
        float(final_metrics["primary"]),
        float(final_metrics["gauc"]),
        float(final_metrics["ndcg@5"]),
        elapsed,
    )
)