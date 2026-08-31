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
SEED = 271828
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 8))

DEVICE = torch.device("cpu")
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
VIDEO_FIELD_INDEX = FIELDS.index("video_id")
EMBED_DIM = 12
HISTORY_LENGTH = 8
BATCH_SIZE = 8192
EPOCHS = 2
LEARNING_RATE = 0.002
BLEND_OWN_WEIGHTS = [0.15, 0.25, 0.35, 0.50, 0.65, 0.80]


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    position = np.searchsorted(unique_dates, dates)
    age = unique_dates.size - 1 - position
    weights = np.exp2(-age.astype(np.float32) / np.float32(half_life))
    weights /= np.mean(weights)
    return weights.astype(np.float32)


def stack_fields(split):
    return np.column_stack([
        np.asarray(split.X[name], dtype=np.int64)
        for name in FIELDS
    ])


def causal_video_history(split, history_length=HISTORY_LENGTH):
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    rows = np.arange(users.size, dtype=np.int64)

    order = np.lexsort((rows, times, users))
    ordered_users = users[order]
    ordered_videos = videos[order]

    history_ordered = np.zeros(
        (users.size, history_length),
        dtype=np.int64,
    )

    for lag in range(1, history_length + 1):
        if lag >= users.size:
            break
        same_user = ordered_users[lag:] == ordered_users[:-lag]
        destination = np.flatnonzero(same_user) + lag
        history_ordered[destination, lag - 1] = (
            ordered_videos[:-lag][same_user]
        )

    history = np.empty_like(history_ordered)
    history[order] = history_ordered
    return history


def history_with_train_context(train, target, history_length=HISTORY_LENGTH):
    train_users = np.asarray(train.user_id, dtype=np.int64)
    target_users = np.asarray(target.user_id, dtype=np.int64)
    train_times = np.asarray(train.time_ms, dtype=np.int64)
    target_times = np.asarray(target.time_ms, dtype=np.int64)
    train_videos = np.asarray(train.video_id, dtype=np.int64)
    target_videos = np.asarray(target.video_id, dtype=np.int64)

    users = np.concatenate([train_users, target_users])
    times = np.concatenate([train_times, target_times])
    videos = np.concatenate([train_videos, target_videos])
    rows = np.arange(users.size, dtype=np.int64)

    order = np.lexsort((rows, times, users))
    ordered_users = users[order]
    ordered_videos = videos[order]

    history_ordered = np.zeros(
        (users.size, history_length),
        dtype=np.int64,
    )

    for lag in range(1, history_length + 1):
        same_user = ordered_users[lag:] == ordered_users[:-lag]
        destination = np.flatnonzero(same_user) + lag
        history_ordered[destination, lag - 1] = (
            ordered_videos[:-lag][same_user]
        )

    history = np.empty_like(history_ordered)
    history[order] = history_ordered
    return history[train_users.size:]


def embedding_init(embedding):
    nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
    if embedding.padding_idx is not None:
        with torch.no_grad():
            embedding.weight[embedding.padding_idx].zero_()


class FieldAwareFM(nn.Module):
    """Each field has a different embedding when interacting with each peer."""

    def __init__(self, cardinalities, dim=12):
        super().__init__()
        self.num_fields = len(cardinalities)
        self.dim = dim

        self.linear = nn.ModuleList([
            nn.Embedding(card, 1)
            for card in cardinalities
        ])
        self.field_aware = nn.ModuleList([
            nn.Embedding(card, self.num_fields * dim)
            for card in cardinalities
        ])
        self.bias = nn.Parameter(torch.zeros(()))

        for embedding in self.linear:
            embedding_init(embedding)
        for embedding in self.field_aware:
            embedding_init(embedding)

    def forward(self, x, history=None):
        batch_size = x.shape[0]
        score = self.bias.expand(batch_size)

        conditioned = []
        for field_index in range(self.num_fields):
            score = score + self.linear[field_index](
                x[:, field_index]
            ).squeeze(1)
            conditioned.append(
                self.field_aware[field_index](
                    x[:, field_index]
                ).view(batch_size, self.num_fields, self.dim)
            )

        interaction = torch.zeros_like(score)
        for left in range(self.num_fields):
            for right in range(left + 1, self.num_fields):
                interaction = interaction + (
                    conditioned[left][:, right, :]
                    * conditioned[right][:, left, :]
                ).sum(dim=1)

        return score + interaction


class AutoIntModel(nn.Module):
    """Multi-head self-attention forms context-dependent field interactions."""

    def __init__(self, cardinalities, dim=12):
        super().__init__()
        self.num_fields = len(cardinalities)
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim)
            for card in cardinalities
        ])
        self.linear = nn.ModuleList([
            nn.Embedding(card, 1)
            for card in cardinalities
        ])

        self.attention1 = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=3,
            dropout=0.05,
            batch_first=True,
        )
        self.attention2 = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=3,
            dropout=0.05,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.output = nn.Sequential(
            nn.Linear(self.num_fields * dim, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.bias = nn.Parameter(torch.zeros(()))

        for embedding in self.embeddings:
            embedding_init(embedding)
        for embedding in self.linear:
            embedding_init(embedding)

    def forward(self, x, history=None):
        tokens = torch.stack([
            embedding(x[:, index])
            for index, embedding in enumerate(self.embeddings)
        ], dim=1)

        attended, _ = self.attention1(
            tokens, tokens, tokens, need_weights=False
        )
        tokens = self.norm1(tokens + attended)

        attended, _ = self.attention2(
            tokens, tokens, tokens, need_weights=False
        )
        tokens = self.norm2(tokens + attended)

        wide = self.bias.expand(x.shape[0])
        for index, embedding in enumerate(self.linear):
            wide = wide + embedding(x[:, index]).squeeze(1)

        deep = self.output(tokens.flatten(start_dim=1)).squeeze(1)
        return wide + deep


class DINExposureModel(nn.Module):
    """
    Attends over previously exposed videos using the current video as query.
    The sequence contains identities only, never outcomes or auxiliary signals.
    """

    def __init__(self, cardinalities, video_cardinality, dim=12):
        super().__init__()
        self.field_embeddings = nn.ModuleList([
            nn.Embedding(card, dim)
            for card in cardinalities
        ])
        self.video_sequence_embedding = nn.Embedding(
            video_cardinality,
            dim,
            padding_idx=0,
        )
        self.linear = nn.ModuleList([
            nn.Embedding(card, 1)
            for card in cardinalities
        ])

        self.attention = nn.Sequential(
            nn.Linear(4 * dim, 48),
            nn.ReLU(),
            nn.Linear(48, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.tower = nn.Sequential(
            nn.Linear((len(cardinalities) + 1) * dim, 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.bias = nn.Parameter(torch.zeros(()))

        for embedding in self.field_embeddings:
            embedding_init(embedding)
        embedding_init(self.video_sequence_embedding)
        for embedding in self.linear:
            embedding_init(embedding)

    def forward(self, x, history):
        field_vectors = [
            embedding(x[:, index])
            for index, embedding in enumerate(self.field_embeddings)
        ]

        query = self.video_sequence_embedding(x[:, VIDEO_FIELD_INDEX])
        sequence = self.video_sequence_embedding(history)
        expanded_query = query.unsqueeze(1).expand_as(sequence)

        attention_input = torch.cat([
            sequence,
            expanded_query,
            sequence * expanded_query,
            sequence - expanded_query,
        ], dim=2)

        attention_logits = self.attention(attention_input).squeeze(2)
        mask = history != 0
        attention_logits = attention_logits.masked_fill(~mask, -1e4)
        attention_weights = torch.softmax(attention_logits, dim=1)
        attention_weights = attention_weights * mask.float()
        attention_weights = attention_weights / (
            attention_weights.sum(dim=1, keepdim=True) + 1e-8
        )
        interest = torch.sum(
            sequence * attention_weights.unsqueeze(2),
            dim=1,
        )

        wide = self.bias.expand(x.shape[0])
        for index, embedding in enumerate(self.linear):
            wide = wide + embedding(x[:, index]).squeeze(1)

        tower_input = torch.cat(field_vectors + [interest], dim=1)
        return wide + self.tower(tower_input).squeeze(1)


def train_model(model, x, history, labels, weights, name):
    model.to(DEVICE)
    model.train()

    x_tensor = torch.from_numpy(x)
    history_tensor = torch.from_numpy(history)
    y_tensor = torch.from_numpy(labels.astype(np.float32, copy=False))
    weight_tensor = torch.from_numpy(weights.astype(np.float32, copy=False))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-6,
    )

    n = x.shape[0]
    generator = np.random.RandomState(SEED + sum(map(ord, name)))

    for epoch in range(EPOCHS):
        permutation = generator.permutation(n)
        loss_sum = 0.0
        weight_sum = 0.0

        for start in range(0, n, BATCH_SIZE):
            indices_np = permutation[start:start + BATCH_SIZE]
            indices = torch.from_numpy(indices_np)

            xb = x_tensor.index_select(0, indices).to(DEVICE)
            hb = history_tensor.index_select(0, indices).to(DEVICE)
            yb = y_tensor.index_select(0, indices).to(DEVICE)
            wb = weight_tensor.index_select(0, indices).to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, hb)
            row_loss = F.binary_cross_entropy_with_logits(
                logits,
                yb,
                reduction="none",
            )
            loss = torch.sum(row_loss * wb) / torch.sum(wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loss_sum += float(torch.sum(row_loss * wb).detach())
            weight_sum += float(torch.sum(wb))

        print(
            "FINDINGS model={} epoch={} weighted_logloss={:.6f}".format(
                name,
                epoch + 1,
                loss_sum / max(weight_sum, 1e-8),
            ),
            flush=True,
        )

    return model


@torch.no_grad()
def predict_model(model, x, history):
    model.eval()
    x_tensor = torch.from_numpy(x)
    history_tensor = torch.from_numpy(history)
    predictions = np.empty(x.shape[0], dtype=np.float64)

    for start in range(0, x.shape[0], BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, x.shape[0])
        logits = model(
            x_tensor[start:end].to(DEVICE),
            history_tensor[start:end].to(DEVICE),
        )
        predictions[start:end] = logits.cpu().numpy().astype(np.float64)

    return predictions


def within_user_fractional_rank(user_ids, scores):
    """
    Convert a score to [0,1] rank position within each user's impressions.
    This uses no labels and makes heterogeneous model scales comparable.
    """
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    n = users.size
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, values, users))
    ordered_users = users[order]

    group_start = np.r_[
        0,
        np.flatnonzero(ordered_users[1:] != ordered_users[:-1]) + 1,
    ]
    group_end = np.r_[group_start[1:], n]
    group_lengths = group_end - group_start

    starts_per_row = np.repeat(group_start, group_lengths)
    lengths_per_row = np.repeat(group_lengths, group_lengths)
    positions = np.arange(n, dtype=np.int64) - starts_per_row

    ranked_ordered = positions.astype(np.float64) / np.maximum(
        lengths_per_row - 1,
        1,
    )
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_ordered
    return ranked


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
train_weights = recency_weights(train.date, half_life=4.0)

x_train = stack_fields(train)
x_valid = stack_fields(valid)

print(
    "FINDINGS recency_half_life=4 effective_rows={:.0f} min_weight={:.4f} max_weight={:.4f}".format(
        float(train_weights.sum() ** 2 / np.square(train_weights).sum()),
        float(train_weights.min()),
        float(train_weights.max()),
    ),
    flush=True,
)

train_history = causal_video_history(train)
valid_history = history_with_train_context(train, valid)

valid_history_coverage = np.mean(np.any(valid_history != 0, axis=1))
valid_history_mean_length = np.mean(np.sum(valid_history != 0, axis=1))
print(
    "FINDINGS din_history valid_coverage={:.4f} valid_mean_length={:.3f}".format(
        float(valid_history_coverage),
        float(valid_history_mean_length),
    ),
    flush=True,
)

cardinalities = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
video_cardinality = int(FEATURE_CARDINALITIES["video_id"])

model_builders = {
    "field_aware_fm": lambda: FieldAwareFM(
        cardinalities,
        dim=EMBED_DIM,
    ),
    "autoint": lambda: AutoIntModel(
        cardinalities,
        dim=EMBED_DIM,
    ),
    "din_exposure_sequence": lambda: DINExposureModel(
        cardinalities,
        video_cardinality,
        dim=EMBED_DIM,
    ),
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
incumbent_valid_path = os.path.join(
    shared,
    "incumbent_valid_scores.npy",
)
incumbent_test_path = os.path.join(
    shared,
    "incumbent_test_scores.npy",
)

if not os.path.exists(incumbent_valid_path):
    raise FileNotFoundError(
        "Trusted incumbent validation predictions are unavailable"
    )

incumbent_valid = np.asarray(
    np.load(incumbent_valid_path),
    dtype=np.float64,
)
incumbent_valid_rank = within_user_fractional_rank(
    valid.user_id,
    incumbent_valid,
)

candidate_scores = {}
valid_predictions = {}
trained_models = {}

incumbent_metrics = evaluate(
    valid.user_id,
    y_valid,
    incumbent_valid,
)
candidate_scores["trusted_incumbent"] = float(
    incumbent_metrics["primary"]
)

best_name = "trusted_incumbent"
best_primary = float(incumbent_metrics["primary"])
best_valid_scores = incumbent_valid.copy()
best_raw_valid = None
best_model_name = None
best_alpha = None

for model_name, builder in model_builders.items():
    model = builder()
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print(
        "FINDINGS model={} parameters={}".format(
            model_name,
            parameter_count,
        ),
        flush=True,
    )

    model = train_model(
        model,
        x_train,
        train_history,
        y_train,
        train_weights,
        model_name,
    )
    raw_valid = predict_model(model, x_valid, valid_history)
    valid_predictions[model_name] = raw_valid
    trained_models[model_name] = model

    standalone_metrics = evaluate(
        valid.user_id,
        y_valid,
        raw_valid,
    )
    standalone_primary = float(standalone_metrics["primary"])
    candidate_scores[model_name] = standalone_primary

    if standalone_primary > best_primary:
        best_primary = standalone_primary
        best_name = model_name
        best_valid_scores = raw_valid.copy()
        best_raw_valid = raw_valid.copy()
        best_model_name = model_name
        best_alpha = None

    own_rank = within_user_fractional_rank(
        valid.user_id,
        raw_valid,
    )

    family_best_blend = -np.inf
    family_best_alpha = None

    for own_weight in BLEND_OWN_WEIGHTS:
        blended = (
            own_weight * own_rank
            + (1.0 - own_weight) * incumbent_valid_rank
        )
        blend_metrics = evaluate(
            valid.user_id,
            y_valid,
            blended,
        )
        blend_primary = float(blend_metrics["primary"])
        candidate_name = "{}_blend_{:.2f}".format(
            model_name,
            own_weight,
        )
        candidate_scores[candidate_name] = blend_primary

        if blend_primary > family_best_blend:
            family_best_blend = blend_primary
            family_best_alpha = own_weight

        if blend_primary > best_primary:
            best_primary = blend_primary
            best_name = candidate_name
            best_valid_scores = blended.copy()
            best_raw_valid = raw_valid.copy()
            best_model_name = model_name
            best_alpha = own_weight

    print(
        "FINDINGS family={} standalone={:.6f} best_blend={:.6f} best_own_weight={:.2f}".format(
            model_name,
            standalone_primary,
            family_best_blend,
            family_best_alpha,
        ),
        flush=True,
    )

# The incumbent is always available, but an exact tie is resolved in favor of
# a newly trained model so the experiment remains independently reproducible.
if best_model_name is None:
    family_only = {
        name: score
        for name, score in candidate_scores.items()
        if name != "trusted_incumbent"
    }
    fallback_name = max(family_only, key=family_only.get)
    if "_blend_" in fallback_name:
        model_name, alpha_text = fallback_name.rsplit("_blend_", 1)
        best_model_name = model_name
        best_alpha = float(alpha_text)
        raw_valid = valid_predictions[model_name]
        best_raw_valid = raw_valid.copy()
        best_valid_scores = (
            best_alpha
            * within_user_fractional_rank(valid.user_id, raw_valid)
            + (1.0 - best_alpha) * incumbent_valid_rank
        )
        best_name = fallback_name
    else:
        best_model_name = fallback_name
        best_alpha = None
        best_raw_valid = valid_predictions[fallback_name].copy()
        best_valid_scores = best_raw_valid.copy()
        best_name = fallback_name

final_metrics = evaluate(
    valid.user_id,
    y_valid,
    best_valid_scores,
)

print(
    "FINDINGS selected={} selected_model={} blend_own_weight={}".format(
        best_name,
        best_model_name,
        "none" if best_alpha is None else "{:.2f}".format(best_alpha),
    ),
    flush=True,
)

# Build test histories strictly from train plus preceding test exposures.
# Validation rows are not used as test context or as fitting data.
test = load("test")
x_test = stack_fields(test)
test_history = history_with_train_context(train, test)

selected_model = trained_models[best_model_name]
raw_test = predict_model(
    selected_model,
    x_test,
    test_history,
)

if best_alpha is None:
    test_scores = raw_test
else:
    if not os.path.exists(incumbent_test_path):
        raise FileNotFoundError(
            "Trusted incumbent test predictions are unavailable"
        )
    incumbent_test = np.asarray(
        np.load(incumbent_test_path),
        dtype=np.float64,
    )
    own_test_rank = within_user_fractional_rank(
        test.user_id,
        raw_test,
    )
    incumbent_test_rank = within_user_fractional_rank(
        test.user_id,
        incumbent_test,
    )
    test_scores = (
        best_alpha * own_test_rank
        + (1.0 - best_alpha) * incumbent_test_rank
    )

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if best_alpha is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

print(
    "CANDIDATES " + json.dumps(
        {key: float(value) for key, value in candidate_scores.items()},
        sort_keys=True,
    ),
    flush=True,
)

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.4f}'
    % (
        float(final_metrics["primary"]),
        float(final_metrics["gauc"]),
        float(final_metrics["ndcg@5"]),
        float(elapsed),
    )
)