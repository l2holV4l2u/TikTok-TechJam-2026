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
SEED = 314159
THREADS = min(8, os.cpu_count() or 8)
HISTORY_LENGTH = 8
BATCH_SIZE = 8192
EPOCHS = 2
HASH_SIZE = 1 << 20

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

CURRENT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
]


def recency_weights(dates, half_life=3.0):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    positions = np.searchsorted(unique_dates, dates)
    ages = len(unique_dates) - 1 - positions
    weights = np.exp2(-ages.astype(np.float32) / np.float32(half_life))
    weights /= weights.mean()
    return weights.astype(np.float32)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    ranked = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ranked[multi] = positions[multi] / (repeated_lengths[multi] - 1.0)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def ordered_training_state(train):
    n = len(train.user_id)
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, train.time_ms, train.user_id))
    sorted_users = np.asarray(train.user_id, dtype=np.int64)[order]
    sorted_videos = np.asarray(train.video_id, dtype=np.int64)[order]
    sorted_labels = np.asarray(train.y, dtype=np.int8)[order]

    train_history_video = np.zeros(
        (n, HISTORY_LENGTH), dtype=np.int32
    )
    train_history_label = np.zeros(
        (n, HISTORY_LENGTH), dtype=np.int8
    )

    sorted_positions = np.arange(n, dtype=np.int64)
    for lag in range(1, HISTORY_LENGTH + 1):
        candidate_positions = sorted_positions - lag
        valid = candidate_positions >= 0
        clipped = np.maximum(candidate_positions, 0)
        valid &= sorted_users[clipped] == sorted_users
        destination_column = HISTORY_LENGTH - lag

        destination_rows = order[valid]
        source_positions = clipped[valid]
        train_history_video[destination_rows, destination_column] = (
            sorted_videos[source_positions] + 1
        ).astype(np.int32)
        train_history_label[destination_rows, destination_column] = (
            sorted_labels[source_positions] + 1
        ).astype(np.int8)

    user_cardinality = FEATURE_CARDINALITIES["user_id"]
    last_position = np.full(user_cardinality, -1, dtype=np.int64)
    if n:
        group_end = np.r_[
            sorted_users[1:] != sorted_users[:-1],
            True,
        ]
        ending_positions = np.flatnonzero(group_end)
        ending_users = sorted_users[ending_positions]
        in_range = (
            (ending_users >= 0) &
            (ending_users < user_cardinality)
        )
        last_position[ending_users[in_range]] = ending_positions[in_range]

    return (
        train_history_video,
        train_history_label,
        order,
        sorted_users,
        sorted_videos,
        sorted_labels,
        last_position,
    )


def future_history(split, sorted_users, sorted_videos,
                   sorted_labels, last_position):
    users = np.asarray(split.user_id, dtype=np.int64)
    n = len(users)
    history_video = np.zeros((n, HISTORY_LENGTH), dtype=np.int32)
    history_label = np.zeros((n, HISTORY_LENGTH), dtype=np.int8)

    safe_users = np.clip(users, 0, len(last_position) - 1)
    base = last_position[safe_users]
    valid_user = (
        (users >= 0) &
        (users < len(last_position)) &
        (base >= 0)
    )

    for column in range(HISTORY_LENGTH):
        offset = HISTORY_LENGTH - 1 - column
        source = base - offset
        valid = valid_user & (source >= 0)
        clipped = np.maximum(source, 0)
        valid &= sorted_users[clipped] == users

        history_video[valid, column] = (
            sorted_videos[clipped[valid]] + 1
        ).astype(np.int32)
        history_label[valid, column] = (
            sorted_labels[clipped[valid]] + 1
        ).astype(np.int8)

    return history_video, history_label


def current_arrays(split):
    return {
        field: np.asarray(split.X[field], dtype=np.int64)
        for field in CURRENT_FIELDS
    }


class CurrentFeatureTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.user = nn.Embedding(
            FEATURE_CARDINALITIES["user_id"], 16
        )
        self.video = nn.Embedding(
            FEATURE_CARDINALITIES["video_id"] + 1, 20,
            padding_idx=0
        )
        self.author = nn.Embedding(
            FEATURE_CARDINALITIES["author_id"], 12
        )
        self.tab = nn.Embedding(
            FEATURE_CARDINALITIES["tab"], 4
        )
        self.duration = nn.Embedding(
            FEATURE_CARDINALITIES["duration_bucket"], 6
        )

    @property
    def output_dim(self):
        return 16 + 20 + 12 + 4 + 6

    def forward(self, current):
        return torch.cat(
            [
                self.user(current["user_id"]),
                self.video(current["video_id"] + 1),
                self.author(current["author_id"]),
                self.tab(current["tab"]),
                self.duration(current["duration_bucket"]),
            ],
            dim=1,
        )


class DIENRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.current_tower = CurrentFeatureTower()
        self.history_video = self.current_tower.video
        self.history_label = nn.Embedding(3, 4, padding_idx=0)
        self.gru = nn.GRU(24, 24, batch_first=True)
        self.query = nn.Linear(20, 24, bias=False)
        self.attention_scale = 24.0 ** -0.5
        self.output = nn.Sequential(
            nn.Linear(self.current_tower.output_dim + 24 + 20, 80),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(80, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, current, history_video, history_label):
        current_features = self.current_tower(current)
        current_video = self.current_tower.video(
            current["video_id"] + 1
        )

        sequence = torch.cat(
            [
                self.history_video(history_video),
                self.history_label(history_label),
            ],
            dim=2,
        )
        evolved, _ = self.gru(sequence)

        query = self.query(current_video).unsqueeze(2)
        attention = torch.bmm(evolved, query).squeeze(2)
        attention = attention * self.attention_scale
        mask = history_video != 0
        attention = attention.masked_fill(~mask, -1e4)
        weights = torch.softmax(attention, dim=1)
        weights = weights * mask.float()
        weights = weights / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        interest = torch.sum(
            evolved * weights.unsqueeze(2), dim=1
        )

        features = torch.cat(
            [current_features, interest, current_video], dim=1
        )
        return self.output(features).squeeze(1)


class BSTSequenceRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.current_tower = CurrentFeatureTower()
        self.video_token = nn.Embedding(
            FEATURE_CARDINALITIES["video_id"] + 1,
            24,
            padding_idx=0,
        )
        self.label_token = nn.Embedding(3, 24, padding_idx=0)
        self.position = nn.Parameter(
            torch.randn(HISTORY_LENGTH + 1, 24) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=24,
            nhead=4,
            dim_feedforward=64,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.output = nn.Sequential(
            nn.Linear(self.current_tower.output_dim + 24, 80),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(80, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, current, history_video, history_label):
        current_features = self.current_tower(current)

        historical_tokens = (
            self.video_token(history_video) +
            self.label_token(history_label)
        )
        current_video = current["video_id"] + 1
        current_token = self.video_token(current_video).unsqueeze(1)

        tokens = torch.cat(
            [historical_tokens, current_token], dim=1
        )
        tokens = tokens + self.position.unsqueeze(0)

        history_padding = history_video == 0
        current_padding = torch.zeros(
            (len(history_video), 1),
            dtype=torch.bool,
            device=history_video.device,
        )
        padding_mask = torch.cat(
            [history_padding, current_padding], dim=1
        )

        encoded = self.transformer(
            tokens,
            src_key_padding_mask=padding_mask,
        )
        representation = encoded[:, -1, :]
        features = torch.cat(
            [current_features, representation], dim=1
        )
        return self.output(features).squeeze(1)


def fit_model(model, name, current, history_video, history_label,
              labels, weights):
    labels = np.asarray(labels, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    rng = np.random.RandomState(SEED + sum(map(ord, name)))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.5e-3,
        weight_decay=2e-5,
    )

    model.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(len(labels))
        epoch_loss = 0.0
        seen = 0

        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            current_batch = {
                field: torch.from_numpy(
                    current[field][idx].astype(np.int64, copy=False)
                )
                for field in CURRENT_FIELDS
            }
            hv = torch.from_numpy(
                history_video[idx].astype(np.int64, copy=False)
            )
            hl = torch.from_numpy(
                history_label[idx].astype(np.int64, copy=False)
            )
            yb = torch.from_numpy(labels[idx])
            wb = torch.from_numpy(weights[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(current_batch, hv, hl)
            losses = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = torch.sum(losses * wb) / torch.sum(wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += float(loss.detach()) * len(idx)
            seen += len(idx)

        print(
            "FINDINGS {} epoch={} weighted_bce={:.6f}".format(
                name, epoch + 1, epoch_loss / max(seen, 1)
            ),
            flush=True,
        )

    return model


@torch.no_grad()
def predict_model(model, current, history_video, history_label):
    model.eval()
    output = np.empty(len(history_video), dtype=np.float64)

    for start in range(0, len(history_video), 16384):
        end = min(start + 16384, len(history_video))
        current_batch = {
            field: torch.from_numpy(
                current[field][start:end].astype(
                    np.int64, copy=False
                )
            )
            for field in CURRENT_FIELDS
        }
        hv = torch.from_numpy(
            history_video[start:end].astype(np.int64, copy=False)
        )
        hl = torch.from_numpy(
            history_label[start:end].astype(np.int64, copy=False)
        )
        output[start:end] = model(
            current_batch, hv, hl
        ).numpy().astype(np.float64)

    return output


class MarkovTransitionEstimator:
    def __init__(self, hash_size=HASH_SIZE, smoothing=35.0):
        self.hash_size = int(hash_size)
        self.smoothing = float(smoothing)
        self.count = None
        self.positive = None
        self.prior = 0.0

    def keys(self, split, history_video):
        previous_video = history_video[:, -1].astype(np.uint64)
        current_video = (
            np.asarray(split.video_id, dtype=np.uint64) + 1
        )
        author = np.asarray(
            split.X["author_id"], dtype=np.uint64
        )
        tab = np.asarray(split.X["tab"], dtype=np.uint64)

        key = (
            previous_video * np.uint64(11995408973635179863) ^
            current_video * np.uint64(10150724397891781847) ^
            author * np.uint64(1442695040888963407) ^
            tab * np.uint64(6364136223846793005)
        )
        return np.asarray(
            key & np.uint64(self.hash_size - 1),
            dtype=np.int64,
        )

    def fit(self, split, history_video, labels, weights):
        labels = np.asarray(labels, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        keys = self.keys(split, history_video)

        self.prior = float(
            np.sum(labels * weights) / np.sum(weights)
        )
        self.count = np.bincount(
            keys,
            weights=weights,
            minlength=self.hash_size,
        ).astype(np.float64)
        self.positive = np.bincount(
            keys,
            weights=weights * labels,
            minlength=self.hash_size,
        ).astype(np.float64)
        return self

    def predict(self, split, history_video,
                labels=None, weights=None):
        keys = self.keys(split, history_video)
        count = self.count[keys].copy()
        positive = self.positive[keys].copy()

        if labels is not None:
            labels = np.asarray(labels, dtype=np.float64)
            weights = np.asarray(weights, dtype=np.float64)
            count -= weights
            positive -= weights * labels

        count = np.maximum(count, 0.0)
        rate = (
            positive + self.smoothing * self.prior
        ) / (count + self.smoothing)
        rate = np.clip(rate, 1e-6, 1.0 - 1e-6)
        return np.log(rate / (1.0 - rate))


def load_incumbent():
    shared = os.environ.get("SHARED_ARTIFACTS", "")
    valid_path = os.path.join(
        shared, "incumbent_valid_scores.npy"
    )
    test_path = os.path.join(
        shared, "incumbent_test_scores.npy"
    )
    if not os.path.exists(valid_path):
        raise FileNotFoundError(valid_path)
    if not os.path.exists(test_path):
        raise FileNotFoundError(test_path)
    return (
        np.load(valid_path).astype(np.float64),
        np.load(test_path).astype(np.float64),
    )


def primary(users, labels, scores):
    return float(evaluate(users, labels, scores)["primary"])


train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
train_weights = recency_weights(train.date, half_life=3.0)

(
    train_history_video,
    train_history_label,
    training_order,
    sorted_train_users,
    sorted_train_videos,
    sorted_train_labels,
    last_train_position,
) = ordered_training_state(train)

valid_history_video, valid_history_label = future_history(
    valid,
    sorted_train_users,
    sorted_train_videos,
    sorted_train_labels,
    last_train_position,
)

train_current = current_arrays(train)
valid_current = current_arrays(valid)

history_coverage = np.mean(train_history_video != 0, axis=0)
print(
    "FINDINGS history_slot_coverage={}".format(
        json.dumps([float(x) for x in history_coverage])
    ),
    flush=True,
)

torch.manual_seed(SEED + 1)
dien_model = fit_model(
    DIENRanker(),
    "dien_interest_evolution",
    train_current,
    train_history_video,
    train_history_label,
    train_y,
    train_weights,
)
dien_valid_raw = predict_model(
    dien_model,
    valid_current,
    valid_history_video,
    valid_history_label,
)

torch.manual_seed(SEED + 2)
bst_model = fit_model(
    BSTSequenceRanker(),
    "behavior_sequence_transformer",
    train_current,
    train_history_video,
    train_history_label,
    train_y,
    train_weights,
)
bst_valid_raw = predict_model(
    bst_model,
    valid_current,
    valid_history_video,
    valid_history_label,
)

markov_model = MarkovTransitionEstimator().fit(
    train,
    train_history_video,
    train_y,
    train_weights,
)
markov_valid_raw = markov_model.predict(
    valid,
    valid_history_video,
)

inc_valid_raw, inc_test_raw = load_incumbent()
if len(inc_valid_raw) != len(valid_y):
    raise RuntimeError("Incumbent validation length mismatch")

inc_valid = within_user_rank(valid.user_id, inc_valid_raw)
family_valid = {
    "dien_interest_evolution": within_user_rank(
        valid.user_id, dien_valid_raw
    ),
    "behavior_sequence_transformer": within_user_rank(
        valid.user_id, bst_valid_raw
    ),
    "markov_transition": within_user_rank(
        valid.user_id, markov_valid_raw
    ),
}

candidate_scores = {
    "trusted_incumbent": primary(
        valid.user_id, valid_y, inc_valid
    )
}

best_score = candidate_scores["trusted_incumbent"]
best_family = "dien_interest_evolution"
best_alpha = 0.0
best_valid_scores = inc_valid.copy()
best_raw_valid = family_valid[best_family].copy()

alphas = [0.10, 0.20, 0.35, 0.50, 0.70, 1.00]

for name, ranked_scores in family_valid.items():
    standalone = primary(
        valid.user_id, valid_y, ranked_scores
    )
    candidate_scores[name] = standalone

    family_best_score = standalone
    family_best_alpha = 1.0

    if standalone > best_score:
        best_score = standalone
        best_family = name
        best_alpha = 1.0
        best_valid_scores = ranked_scores.copy()
        best_raw_valid = ranked_scores.copy()

    for alpha in alphas[:-1]:
        blended = (
            (1.0 - alpha) * inc_valid +
            alpha * ranked_scores
        )
        score = primary(
            valid.user_id, valid_y, blended
        )
        if score > family_best_score:
            family_best_score = score
            family_best_alpha = alpha

        if score > best_score:
            best_score = score
            best_family = name
            best_alpha = alpha
            best_valid_scores = blended.copy()
            best_raw_valid = ranked_scores.copy()

    candidate_scores[
        name + "_best_incumbent_blend"
    ] = family_best_score
    candidate_scores[
        name + "_best_alpha"
    ] = family_best_alpha

print(
    "CANDIDATES " + json.dumps(
        candidate_scores, sort_keys=True
    ),
    flush=True,
)
print(
    "FINDINGS selected_family={} selected_alpha={:.2f}".format(
        best_family, best_alpha
    ),
    flush=True,
)

metrics = evaluate(
    valid.user_id,
    valid_y,
    best_valid_scores,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )

test = load("test")
test_current = current_arrays(test)
test_history_video, test_history_label = future_history(
    test,
    sorted_train_users,
    sorted_train_videos,
    sorted_train_labels,
    last_train_position,
)

if best_family == "dien_interest_evolution":
    test_family_raw = predict_model(
        dien_model,
        test_current,
        test_history_video,
        test_history_label,
    )
elif best_family == "behavior_sequence_transformer":
    test_family_raw = predict_model(
        bst_model,
        test_current,
        test_history_video,
        test_history_label,
    )
elif best_family == "markov_transition":
    test_family_raw = markov_model.predict(
        test,
        test_history_video,
    )
else:
    raise RuntimeError("Unknown selected family")

test_family_rank = within_user_rank(
    test.user_id, test_family_raw
)
inc_test = within_user_rank(
    test.user_id, inc_test_raw
)
test_scores = (
    (1.0 - best_alpha) * inc_test +
    best_alpha * test_family_rank
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS " + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)