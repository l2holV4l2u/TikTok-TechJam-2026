import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 8675309
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
]
CARDINALITIES = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
OFFSETS = np.cumsum([0] + CARDINALITIES[:-1], dtype=np.int64)
TOTAL_CARDINALITY = int(sum(CARDINALITIES))
VIDEO_CARDINALITY = int(FEATURE_CARDINALITIES["video_id"])
USER_CARDINALITY = int(FEATURE_CARDINALITIES["user_id"])

HISTORY_LENGTH = 12
EMBED_DIM = 32
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 16384
EPOCHS = 2
HALF_LIFE_DAYS = 4.0


def sigmoid_np(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def probability_scale(x):
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
        return np.clip(x, 1e-7, 1.0 - 1e-7)
    return sigmoid_np(x)


def make_current_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[field], dtype=np.int64) + offset
            for field, offset in zip(FIELDS, OFFSETS)
        ]),
        dtype=np.int64,
    )


def chronological_order(split):
    n = len(split.user_id)
    row = np.arange(n, dtype=np.int64)
    return np.lexsort((
        row,
        np.asarray(split.time_ms, dtype=np.int64),
        np.asarray(split.X["user_id"], dtype=np.int64),
    ))


def build_training_histories(train):
    n = len(train.user_id)
    order = chronological_order(train)
    sorted_users = np.asarray(train.X["user_id"], dtype=np.int64)[order]
    videos = np.asarray(train.X["video_id"], dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.int8)

    history_video = np.zeros((n, HISTORY_LENGTH), dtype=np.int32)
    history_outcome = np.zeros((n, HISTORY_LENGTH), dtype=np.int8)

    positions = np.arange(n, dtype=np.int64)
    for distance in range(1, HISTORY_LENGTH + 1):
        source_position = positions - distance
        valid = source_position >= 0
        valid &= np.where(
            source_position >= 0,
            sorted_users[np.maximum(source_position, 0)] == sorted_users,
            False,
        )

        destination_rows = order[valid]
        source_rows = order[source_position[valid]]
        column = HISTORY_LENGTH - distance

        history_video[destination_rows, column] = (
            videos[source_rows].astype(np.int32) + 1
        )
        history_outcome[destination_rows, column] = (
            labels[source_rows].astype(np.int8) + 1
        )

    return history_video, history_outcome, order


def build_terminal_user_histories(train, order):
    sorted_users = np.asarray(train.X["user_id"], dtype=np.int64)[order]
    videos = np.asarray(train.X["video_id"], dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.int8)

    counts = np.bincount(
        sorted_users, minlength=USER_CARDINALITY
    ).astype(np.int64)
    starts = np.zeros(USER_CARDINALITY, dtype=np.int64)
    if USER_CARDINALITY > 1:
        starts[1:] = np.cumsum(counts[:-1])

    table_video = np.zeros(
        (USER_CARDINALITY, HISTORY_LENGTH), dtype=np.int32
    )
    table_outcome = np.zeros(
        (USER_CARDINALITY, HISTORY_LENGTH), dtype=np.int8
    )

    users = np.flatnonzero(counts > 0)
    for distance in range(1, HISTORY_LENGTH + 1):
        eligible = users[counts[users] >= distance]
        sorted_position = starts[eligible] + counts[eligible] - distance
        source_rows = order[sorted_position]
        column = HISTORY_LENGTH - distance
        table_video[eligible, column] = (
            videos[source_rows].astype(np.int32) + 1
        )
        table_outcome[eligible, column] = (
            labels[source_rows].astype(np.int8) + 1
        )

    return table_video, table_outcome


def histories_for_split(split, terminal_video, terminal_outcome):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    return (
        np.ascontiguousarray(terminal_video[users]),
        np.ascontiguousarray(terminal_outcome[users]),
    )


def make_recency_weights(train):
    dates = np.asarray(train.date, dtype=np.int64)
    unique_dates = np.sort(np.unique(dates))
    age_lookup = {
        int(date): len(unique_dates) - 1 - index
        for index, date in enumerate(unique_dates)
    }
    ages = np.fromiter(
        (age_lookup[int(date)] for date in dates),
        dtype=np.float32,
        count=dates.size,
    )
    weights = np.exp2(-ages / HALF_LIFE_DAYS).astype(np.float32)
    weights /= np.mean(weights)
    return weights


def make_user_balanced_weights(train, recency_weights):
    users = np.asarray(train.X["user_id"], dtype=np.int64)
    counts = np.bincount(
        users, minlength=USER_CARDINALITY
    ).astype(np.float32)
    balance = 1.0 / np.sqrt(np.maximum(counts[users], 1.0))
    weights = recency_weights * balance
    weights /= np.mean(weights)
    return weights.astype(np.float32)


class CausalHistoryTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.current_embedding = nn.Embedding(
            TOTAL_CARDINALITY, EMBED_DIM
        )
        self.history_video_embedding = nn.Embedding(
            VIDEO_CARDINALITY + 1, EMBED_DIM, padding_idx=0
        )
        self.outcome_embedding = nn.Embedding(3, EMBED_DIM, padding_idx=0)
        self.position_embedding = nn.Embedding(
            HISTORY_LENGTH + 1, EMBED_DIM
        )

        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=4,
            dim_feedforward=96,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )
        self.wide = nn.Embedding(TOTAL_CARDINALITY, 1)

        causal_mask = torch.triu(
            torch.ones(
                HISTORY_LENGTH + 1,
                HISTORY_LENGTH + 1,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask)

        nn.init.normal_(self.current_embedding.weight, std=0.025)
        nn.init.normal_(self.history_video_embedding.weight, std=0.025)
        nn.init.normal_(self.outcome_embedding.weight, std=0.025)
        nn.init.zeros_(self.wide.weight)

    def forward(self, current, history_video, history_outcome):
        batch_size = current.shape[0]
        history = (
            self.history_video_embedding(history_video)
            + self.outcome_embedding(history_outcome)
        )

        current_token = self.current_embedding(current).mean(dim=1)
        tokens = torch.cat([history, current_token[:, None, :]], dim=1)

        positions = torch.arange(
            HISTORY_LENGTH + 1, device=current.device
        )
        tokens = tokens + self.position_embedding(positions)[None, :, :]

        history_padding = history_video.eq(0)
        current_not_padding = torch.zeros(
            (batch_size, 1), dtype=torch.bool, device=current.device
        )
        padding_mask = torch.cat(
            [history_padding, current_not_padding], dim=1
        )

        encoded = self.encoder(
            tokens,
            mask=self.causal_mask,
            src_key_padding_mask=padding_mask,
        )
        sequential_logit = self.output(encoded[:, -1]).squeeze(1)
        wide_logit = self.wide(current).sum(dim=1).squeeze(1)
        return sequential_logit + wide_logit


class TemporalConvHistoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.current_embedding = nn.Embedding(
            TOTAL_CARDINALITY, EMBED_DIM
        )
        self.history_video_embedding = nn.Embedding(
            VIDEO_CARDINALITY + 1, EMBED_DIM, padding_idx=0
        )
        self.outcome_embedding = nn.Embedding(3, EMBED_DIM, padding_idx=0)

        self.temporal = nn.Sequential(
            nn.Conv1d(EMBED_DIM, 48, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(48, 48, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.network = nn.Sequential(
            nn.Linear(48 + EMBED_DIM, 80),
            nn.GELU(),
            nn.Linear(80, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.wide = nn.Embedding(TOTAL_CARDINALITY, 1)

        nn.init.normal_(self.current_embedding.weight, std=0.025)
        nn.init.normal_(self.history_video_embedding.weight, std=0.025)
        nn.init.normal_(self.outcome_embedding.weight, std=0.025)
        nn.init.zeros_(self.wide.weight)

    def forward(self, current, history_video, history_outcome):
        history = (
            self.history_video_embedding(history_video)
            + self.outcome_embedding(history_outcome)
        )
        convolved = self.temporal(history.transpose(1, 2))

        mask = history_video.ne(0).float()[:, None, :]
        summed = torch.sum(convolved * mask, dim=2)
        denominator = torch.clamp(torch.sum(mask, dim=2), min=1.0)
        history_state = summed / denominator

        current_state = self.current_embedding(current).mean(dim=1)
        deep_logit = self.network(
            torch.cat([history_state, current_state], dim=1)
        ).squeeze(1)
        wide_logit = self.wide(current).sum(dim=1).squeeze(1)
        return deep_logit + wide_logit


def train_model(
    model,
    current_np,
    history_video_np,
    history_outcome_np,
    labels_np,
    weights_np,
    seed,
):
    current_tensor = torch.from_numpy(current_np)
    history_video_tensor = torch.from_numpy(
        history_video_np.astype(np.int64, copy=False)
    )
    history_outcome_tensor = torch.from_numpy(
        history_outcome_np.astype(np.int64, copy=False)
    )
    label_tensor = torch.from_numpy(labels_np)
    weight_tensor = torch.from_numpy(weights_np)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=2e-6
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    n = current_np.shape[0]

    model.train()
    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            current = current_tensor.index_select(0, idx)
            history_video = history_video_tensor.index_select(0, idx)
            history_outcome = history_outcome_tensor.index_select(0, idx)
            labels = label_tensor.index_select(0, idx)
            weights = weight_tensor.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(current, history_video, history_outcome)
            losses = F.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )
            loss = torch.sum(losses * weights) / torch.sum(weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


def predict_model(
    model, current_np, history_video_np, history_outcome_np
):
    predictions = np.empty(current_np.shape[0], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, current_np.shape[0], PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, current_np.shape[0])
            current = torch.from_numpy(current_np[start:end])
            history_video = torch.from_numpy(
                history_video_np[start:end].astype(
                    np.int64, copy=False
                )
            )
            history_outcome = torch.from_numpy(
                history_outcome_np[start:end].astype(
                    np.int64, copy=False
                )
            )
            predictions[start:end] = model(
                current, history_video, history_outcome
            ).cpu().numpy()
    return predictions


train = load("train")
current_train = make_current_matrix(train)
labels_train = np.asarray(train.y, dtype=np.float32)

history_video_train, history_outcome_train, train_order = (
    build_training_histories(train)
)
terminal_video, terminal_outcome = build_terminal_user_histories(
    train, train_order
)

recency_weights = make_recency_weights(train)
balanced_weights = make_user_balanced_weights(train, recency_weights)

models = {
    "causal_transformer": CausalHistoryTransformer(),
    "temporal_convolution": TemporalConvHistoryModel(),
}

train_model(
    models["causal_transformer"],
    current_train,
    history_video_train,
    history_outcome_train,
    labels_train,
    recency_weights,
    SEED + 101,
)
train_model(
    models["temporal_convolution"],
    current_train,
    history_video_train,
    history_outcome_train,
    labels_train,
    balanced_weights,
    SEED + 202,
)

history_coverage = np.mean(history_video_train != 0, axis=1)
print(
    "FINDINGS "
    + json.dumps({
        "history_length": HISTORY_LENGTH,
        "mean_available_history": float(
            history_coverage.mean() * HISTORY_LENGTH
        ),
        "rows_with_no_history": float(
            np.mean(history_coverage == 0.0)
        ),
        "recency_weight_min": float(recency_weights.min()),
        "recency_weight_max": float(recency_weights.max()),
        "balanced_weight_min": float(balanced_weights.min()),
        "balanced_weight_max": float(balanced_weights.max()),
    }, sort_keys=True)
)

del current_train
del history_video_train
del history_outcome_train
del labels_train
del train_order

valid = load("valid")
current_valid = make_current_matrix(valid)
history_video_valid, history_outcome_valid = histories_for_split(
    valid, terminal_video, terminal_outcome
)

raw_valid = {}
for name, model in models.items():
    raw_valid[name] = sigmoid_np(
        predict_model(
            model,
            current_valid,
            history_video_valid,
            history_outcome_valid,
        )
    )

candidate_scores = {}
candidate_specs = {}

for name, scores in raw_valid.items():
    metric = evaluate(valid.user_id, valid.y, scores)
    candidate_scores[name] = float(metric["primary"])
    candidate_specs[name] = {
        "family": name,
        "alpha": 1.0,
        "blended": False,
        "scores": scores,
    }

shared_dir = os.environ.get("SHARED_ARTIFACTS")
incumbent_valid = None
incumbent_test_path = None
if shared_dir:
    valid_path = os.path.join(
        shared_dir, "incumbent_valid_scores.npy"
    )
    test_path = os.path.join(
        shared_dir, "incumbent_test_scores.npy"
    )
    if os.path.exists(valid_path) and os.path.exists(test_path):
        incumbent_valid = probability_scale(np.load(valid_path))
        incumbent_test_path = test_path

blend_alphas = [0.15, 0.30, 0.50, 0.70]
if incumbent_valid is not None:
    for name, scores in raw_valid.items():
        for alpha in blend_alphas:
            blended = (
                alpha * scores
                + (1.0 - alpha) * incumbent_valid
            )
            candidate_name = f"{name}_blend_{alpha:.2f}"
            metric = evaluate(valid.user_id, valid.y, blended)
            candidate_scores[candidate_name] = float(metric["primary"])
            candidate_specs[candidate_name] = {
                "family": name,
                "alpha": alpha,
                "blended": True,
                "scores": blended,
            }

winner_name = max(candidate_scores, key=candidate_scores.get)
winner = candidate_specs[winner_name]
valid_scores = winner["scores"]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps({
        "winner": winner_name,
        "winner_family": winner["family"],
        "winner_alpha": float(winner["alpha"]),
        "valid_history_nonempty": float(
            np.mean(np.any(history_video_valid != 0, axis=1))
        ),
    }, sort_keys=True)
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if winner["blended"]:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(
                raw_valid[winner["family"]], dtype=np.float64
            ),
        )

del current_valid
del history_video_valid
del history_outcome_valid

test = load("test")
current_test = make_current_matrix(test)
history_video_test, history_outcome_test = histories_for_split(
    test, terminal_video, terminal_outcome
)

raw_test = sigmoid_np(
    predict_model(
        models[winner["family"]],
        current_test,
        history_video_test,
        history_outcome_test,
    )
)

if winner["blended"]:
    incumbent_test = probability_scale(
        np.load(incumbent_test_path)
    )
    test_scores = (
        winner["alpha"] * raw_test
        + (1.0 - winner["alpha"]) * incumbent_test
    )
else:
    test_scores = raw_test

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
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