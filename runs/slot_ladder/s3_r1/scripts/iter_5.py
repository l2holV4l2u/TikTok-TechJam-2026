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
SEED = 271828
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
EMBED_DIM = 16
HISTORY_LEN = 12
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 16384
EPOCHS = 3
LR = 0.002
WEIGHT_DECAY = 1e-6

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
NUM_FEATURES = int(sum(cardinalities))
NUM_FIELDS = len(FIELDS)
FLAT_DIM = NUM_FIELDS * EMBED_DIM
VIDEO_CARD = int(FEATURE_CARDINALITIES["video_id"])
USER_CARD = int(FEATURE_CARDINALITIES["user_id"])


def make_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[field], dtype=np.int64) + offset
            for field, offset in zip(FIELDS, offsets)
        ]),
        dtype=np.int64,
    )


def chronological_training_histories(split):
    """For every train row, retrieve only positive videos strictly before it."""
    n = len(split.y)
    row_position = np.arange(n, dtype=np.int64)
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    videos = np.asarray(split.X["video_id"], dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    labels = np.asarray(split.y, dtype=np.int8)

    order = np.lexsort((row_position, times, users))
    sorted_users = users[order]
    sorted_videos = videos[order]
    sorted_labels = labels[order].astype(np.int64)

    global_before = np.cumsum(sorted_labels, dtype=np.int64) - sorted_labels

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    start_positions = np.flatnonzero(starts)
    group_lengths = np.diff(np.append(start_positions, n))
    group_positive_base = global_before[start_positions]
    local_before = global_before - np.repeat(group_positive_base, group_lengths)

    packed_positive_videos = sorted_videos[sorted_labels == 1] + 1
    history_sorted = np.zeros((n, HISTORY_LEN), dtype=np.int32)

    for lag in range(1, HISTORY_LEN + 1):
        valid = local_before >= lag
        source = global_before[valid] - lag
        history_sorted[valid, lag - 1] = packed_positive_videos[source]

    history = np.empty_like(history_sorted)
    history[order] = history_sorted

    positive_counts = np.bincount(
        users[labels == 1], minlength=USER_CARD
    ).astype(np.int64)
    positive_starts = np.cumsum(
        np.r_[0, positive_counts[:-1]], dtype=np.int64
    )
    return history, packed_positive_videos.astype(np.int32), positive_counts, positive_starts


def inference_histories(split, packed_positive_videos, positive_counts, positive_starts):
    """Use the latest train positives for each user; no validation/test outcomes."""
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    history = np.zeros((users.shape[0], HISTORY_LEN), dtype=np.int32)

    known = (users >= 0) & (users < USER_CARD)
    safe_users = np.where(known, users, 0)
    counts = positive_counts[safe_users]
    starts = positive_starts[safe_users]

    for lag in range(1, HISTORY_LEN + 1):
        valid = known & (counts >= lag)
        source = starts[valid] + counts[valid] - lag
        history[valid, lag - 1] = packed_positive_videos[source]
    return history


def init_embedding(embedding):
    nn.init.normal_(embedding.weight, mean=0.0, std=0.015)
    if embedding.padding_idx is not None:
        with torch.no_grad():
            embedding.weight[embedding.padding_idx].zero_()


class DINModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_embedding = nn.Embedding(NUM_FEATURES, EMBED_DIM)
        self.video_embedding = nn.Embedding(
            VIDEO_CARD + 1, EMBED_DIM, padding_idx=0
        )
        init_embedding(self.base_embedding)
        init_embedding(self.video_embedding)

        self.attention = nn.Sequential(
            nn.Linear(4 * EMBED_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.output = nn.Sequential(
            nn.Linear(FLAT_DIM + 3 * EMBED_DIM, 160),
            nn.ReLU(),
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, history):
        base = self.base_embedding(x).reshape(x.shape[0], -1)
        candidate_ids = x[:, 1] - int(offsets[1]) + 1
        candidate = self.video_embedding(candidate_ids)
        historical = self.video_embedding(history)

        candidate_expanded = candidate.unsqueeze(1).expand_as(historical)
        attention_input = torch.cat(
            [
                historical,
                candidate_expanded,
                historical - candidate_expanded,
                historical * candidate_expanded,
            ],
            dim=2,
        )
        attention_logits = self.attention(attention_input).squeeze(2)
        mask = history != 0
        attention_logits = attention_logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(attention_logits, dim=1)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        interest = torch.sum(historical * weights.unsqueeze(2), dim=1)

        interaction = interest * candidate
        features = torch.cat([base, interest, candidate, interaction], dim=1)
        return self.output(features).squeeze(1)


class GRUHistoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_embedding = nn.Embedding(NUM_FEATURES, EMBED_DIM)
        self.video_embedding = nn.Embedding(
            VIDEO_CARD + 1, EMBED_DIM, padding_idx=0
        )
        init_embedding(self.base_embedding)
        init_embedding(self.video_embedding)

        self.gru = nn.GRU(
            input_size=EMBED_DIM,
            hidden_size=EMBED_DIM,
            batch_first=True,
            bias=False,
        )
        self.output = nn.Sequential(
            nn.Linear(FLAT_DIM + 3 * EMBED_DIM, 160),
            nn.ReLU(),
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, history):
        base = self.base_embedding(x).reshape(x.shape[0], -1)
        candidate_ids = x[:, 1] - int(offsets[1]) + 1
        candidate = self.video_embedding(candidate_ids)

        # Histories are newest-first. Reverse so recurrence runs old-to-new.
        sequence = self.video_embedding(torch.flip(history, dims=[1]))
        _, hidden = self.gru(sequence)
        interest = hidden[-1]
        has_history = (history != 0).any(dim=1, keepdim=True).float()
        interest = interest * has_history

        interaction = interest * candidate
        features = torch.cat([base, interest, candidate, interaction], dim=1)
        return self.output(features).squeeze(1)


def train_neural(model, x_np, history_np, labels_np, seed):
    x = torch.from_numpy(x_np)
    history = torch.from_numpy(history_np)
    labels = torch.from_numpy(labels_np.astype(np.float32, copy=False))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    n = x.shape[0]

    model.train()
    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            xb = x.index_select(0, idx)
            hb = history.index_select(0, idx).long()
            yb = labels.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, hb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


def predict_neural(model, x_np, history_np):
    result = np.empty(x_np.shape[0], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, x_np.shape[0], PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, x_np.shape[0])
            xb = torch.from_numpy(x_np[start:end])
            hb = torch.from_numpy(history_np[start:end]).long()
            result[start:end] = model(xb, hb).cpu().numpy()
    return result


def sigmoid_np(values):
    values = np.clip(np.asarray(values, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def score_to_logit(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
        p = np.clip(values, 1e-5, 1.0 - 1e-5)
        return np.log(p) - np.log1p(-p)
    return np.nan_to_num(values, nan=0.0, posinf=20.0, neginf=-20.0)


class TransitionStatistics:
    """Smoothed P(long_view | latest positive video, candidate video)."""

    def fit(self, videos, histories, labels):
        videos = np.asarray(videos, dtype=np.int64)
        previous = np.asarray(histories[:, 0], dtype=np.int64)
        labels = np.asarray(labels, dtype=np.float64)

        global_rate = float(labels.mean())
        video_count = np.bincount(videos, minlength=VIDEO_CARD).astype(np.float64)
        video_sum = np.bincount(
            videos, weights=labels, minlength=VIDEO_CARD
        ).astype(np.float64)
        self.video_rate = (
            video_sum + 20.0 * global_rate
        ) / (video_count + 20.0)

        keys = previous * np.int64(VIDEO_CARD) + videos
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        sorted_labels = labels[order]
        unique_keys, starts, counts = np.unique(
            sorted_keys, return_index=True, return_counts=True
        )
        sums = np.add.reduceat(sorted_labels, starts)
        candidate_video = unique_keys % VIDEO_CARD
        priors = self.video_rate[candidate_video]

        self.keys = unique_keys
        self.rates = (sums + 8.0 * priors) / (counts + 8.0)
        self.support = counts

    def predict(self, videos, histories):
        videos = np.asarray(videos, dtype=np.int64)
        previous = np.asarray(histories[:, 0], dtype=np.int64)
        keys = previous * np.int64(VIDEO_CARD) + videos
        positions = np.searchsorted(self.keys, keys)
        found = positions < self.keys.shape[0]
        safe_positions = np.minimum(positions, self.keys.shape[0] - 1)
        found &= self.keys[safe_positions] == keys

        result = self.video_rate[videos].copy()
        result[found] = self.rates[safe_positions[found]]
        return score_to_logit(result)


train = load("train")
x_train = make_matrix(train)
y_train = np.asarray(train.y, dtype=np.float32)
train_histories, packed_positive_videos, positive_counts, positive_starts = (
    chronological_training_histories(train)
)

history_coverage = float(np.mean(train_histories[:, 0] != 0))
mean_history_length = float(np.mean(np.sum(train_histories != 0, axis=1)))

transition_model = TransitionStatistics()
transition_model.fit(
    np.asarray(train.X["video_id"], dtype=np.int64),
    train_histories,
    y_train,
)

models = {
    "din": DINModel(),
    "gru_history": GRUHistoryModel(),
}
for model_index, model in enumerate(models.values()):
    train_neural(
        model,
        x_train,
        train_histories,
        y_train,
        SEED + 1000 * model_index,
    )

del x_train, train_histories, y_train, train

valid = load("valid")
x_valid = make_matrix(valid)
valid_histories = inference_histories(
    valid, packed_positive_videos, positive_counts, positive_starts
)

raw_valid = {}
raw_valid["din"] = predict_neural(
    models["din"], x_valid, valid_histories
).astype(np.float64)
raw_valid["gru_history"] = predict_neural(
    models["gru_history"], x_valid, valid_histories
).astype(np.float64)
raw_valid["transition_stats"] = transition_model.predict(
    np.asarray(valid.X["video_id"], dtype=np.int64),
    valid_histories,
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
inc_valid = None
inc_test_path = None
if shared_dir:
    valid_path = os.path.join(shared_dir, "incumbent_valid_scores.npy")
    test_path = os.path.join(shared_dir, "incumbent_test_scores.npy")
    if os.path.exists(valid_path) and os.path.exists(test_path):
        inc_valid = score_to_logit(np.load(valid_path))
        inc_test_path = test_path

blend_alphas = [0.15, 0.25, 0.40, 0.60, 0.80]
if inc_valid is not None:
    for family, family_scores in raw_valid.items():
        family_logits = score_to_logit(family_scores)
        for alpha in blend_alphas:
            blended = alpha * family_logits + (1.0 - alpha) * inc_valid
            name = f"{family}_blend_{alpha:.2f}"
            metric = evaluate(valid.user_id, valid.y, blended)
            candidate_scores[name] = float(metric["primary"])
            candidate_specs[name] = {
                "family": family,
                "alpha": alpha,
                "blended": True,
                "scores": blended,
            }

winner_name = max(candidate_scores, key=candidate_scores.get)
winner = candidate_specs[winner_name]
valid_scores = winner["scores"]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print(
    "FINDINGS "
    + json.dumps({
        "train_rows_with_prior_positive": history_coverage,
        "mean_available_positive_history": mean_history_length,
        "valid_rows_with_train_positive_history": float(
            np.mean(valid_histories[:, 0] != 0)
        ),
        "transition_unique_keys": int(transition_model.keys.shape[0]),
        "winner": winner_name,
    }, sort_keys=True)
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

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
            np.asarray(raw_valid[winner["family"]], dtype=np.float64),
        )

del x_valid, valid_histories, valid, raw_valid

test = load("test")
x_test = make_matrix(test)
test_histories = inference_histories(
    test, packed_positive_videos, positive_counts, positive_starts
)

family = winner["family"]
if family == "transition_stats":
    test_raw = transition_model.predict(
        np.asarray(test.X["video_id"], dtype=np.int64),
        test_histories,
    )
else:
    test_raw = predict_neural(
        models[family], x_test, test_histories
    ).astype(np.float64)

if winner["blended"]:
    incumbent_test = score_to_logit(np.load(inc_test_path))
    alpha = float(winner["alpha"])
    test_scores = (
        alpha * score_to_logit(test_raw)
        + (1.0 - alpha) * incumbent_test
    )
else:
    test_scores = test_raw

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
result = {
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result))