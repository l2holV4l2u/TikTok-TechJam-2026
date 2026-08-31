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
THREADS = min(8, os.cpu_count() or 8)
HISTORY_LEN = 12
BATCH_SIZE = 8192
EPOCHS = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

CURRENT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "hour",
]

EMBED_DIMS = {
    "user_id": 12,
    "video_id": 16,
    "author_id": 10,
    "tag": 8,
    "tab": 4,
    "duration_bucket": 5,
    "hour": 4,
}


def recency_weights(dates, half_life=3.0):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    position = np.searchsorted(unique_dates, dates)
    age = len(unique_dates) - 1 - position
    weights = np.exp2(-age.astype(np.float32) / np.float32(half_life))
    weights /= weights.mean()
    return weights.astype(np.float32)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, users))
    ordered_users = users[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = ordered_users[1:] != ordered_users[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    ranked = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ranked[multi] = positions[multi] / (
        repeated_lengths[multi].astype(np.float64) - 1.0
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


class OrderedHistory:
    def __init__(self, train):
        n = len(train.user_id)
        rows = np.arange(n, dtype=np.int64)
        self.order = np.lexsort(
            (rows, np.asarray(train.time_ms), np.asarray(train.user_id))
        )

        self.sorted_users = np.asarray(
            train.user_id, dtype=np.int64
        )[self.order]
        self.sorted_video = np.asarray(
            train.video_id, dtype=np.int64
        )[self.order]
        self.sorted_author = np.asarray(
            train.X["author_id"], dtype=np.int64
        )[self.order]
        self.sorted_tag = np.asarray(
            train.X["tag"], dtype=np.int64
        )[self.order]
        self.sorted_label = np.asarray(
            train.y, dtype=np.int8
        )[self.order]

        user_cardinality = FEATURE_CARDINALITIES["user_id"]
        self.last_position = np.full(
            user_cardinality, -1, dtype=np.int64
        )
        if n:
            group_end = np.r_[
                self.sorted_users[1:] != self.sorted_users[:-1],
                True,
            ]
            ending_positions = np.flatnonzero(group_end)
            ending_users = self.sorted_users[ending_positions]
            in_range = (
                (ending_users >= 0)
                & (ending_users < user_cardinality)
            )
            self.last_position[ending_users[in_range]] = (
                ending_positions[in_range]
            )

    def for_train(self):
        n = len(self.order)
        sorted_positions = np.arange(n, dtype=np.int64)

        video = np.zeros((n, HISTORY_LEN), dtype=np.int32)
        author = np.zeros((n, HISTORY_LEN), dtype=np.int32)
        tag = np.zeros((n, HISTORY_LEN), dtype=np.int32)
        label = np.zeros((n, HISTORY_LEN), dtype=np.int8)

        for lag in range(1, HISTORY_LEN + 1):
            source = sorted_positions - lag
            valid = source >= 0
            clipped = np.maximum(source, 0)
            valid = valid & (
                self.sorted_users[clipped] == self.sorted_users
            )
            column = HISTORY_LEN - lag
            destination = self.order[valid]
            source_valid = clipped[valid]

            video[destination, column] = (
                self.sorted_video[source_valid] + 1
            ).astype(np.int32)
            author[destination, column] = (
                self.sorted_author[source_valid] + 1
            ).astype(np.int32)
            tag[destination, column] = (
                self.sorted_tag[source_valid] + 1
            ).astype(np.int32)
            label[destination, column] = (
                self.sorted_label[source_valid] + 1
            ).astype(np.int8)

        return video, author, tag, label

    def for_split(self, split):
        users = np.asarray(split.user_id, dtype=np.int64)
        n = len(users)

        safe_users = np.clip(
            users, 0, len(self.last_position) - 1
        )
        base = self.last_position[safe_users]
        known_user = (
            (users >= 0)
            & (users < len(self.last_position))
            & (base >= 0)
        )

        offsets = np.arange(
            HISTORY_LEN - 1, -1, -1, dtype=np.int64
        )
        history_positions = base[:, None] - offsets[None, :]

        valid = known_user[:, None] & (history_positions >= 0)
        clipped = np.maximum(history_positions, 0)

        # Deliberately use a non-in-place operation: the left operand has
        # shape (n, 1), while the user comparison broadcasts to (n, H).
        valid = valid & (
            self.sorted_users[clipped] == users[:, None]
        )

        video = np.zeros((n, HISTORY_LEN), dtype=np.int32)
        author = np.zeros((n, HISTORY_LEN), dtype=np.int32)
        tag = np.zeros((n, HISTORY_LEN), dtype=np.int32)
        label = np.zeros((n, HISTORY_LEN), dtype=np.int8)

        video[valid] = (
            self.sorted_video[clipped[valid]] + 1
        ).astype(np.int32)
        author[valid] = (
            self.sorted_author[clipped[valid]] + 1
        ).astype(np.int32)
        tag[valid] = (
            self.sorted_tag[clipped[valid]] + 1
        ).astype(np.int32)
        label[valid] = (
            self.sorted_label[clipped[valid]] + 1
        ).astype(np.int8)

        return video, author, tag, label


def current_batch(split, indices):
    return {
        field: torch.from_numpy(
            np.asarray(split.X[field], dtype=np.int64)[indices]
        )
        for field in CURRENT_FIELDS
    }


class SequenceBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.current_embeddings = nn.ModuleDict()
        for field in CURRENT_FIELDS:
            cardinality = FEATURE_CARDINALITIES[field]
            if field in ("video_id", "author_id", "tag"):
                cardinality += 1
            self.current_embeddings[field] = nn.Embedding(
                cardinality, EMBED_DIMS[field]
            )

        self.history_video = nn.Embedding(
            FEATURE_CARDINALITIES["video_id"] + 1,
            16,
            padding_idx=0,
        )
        self.history_author = nn.Embedding(
            FEATURE_CARDINALITIES["author_id"] + 1,
            10,
            padding_idx=0,
        )
        self.history_tag = nn.Embedding(
            FEATURE_CARDINALITIES["tag"] + 1,
            8,
            padding_idx=0,
        )
        self.history_label = nn.Embedding(
            3, 4, padding_idx=0
        )
        self.token_projection = nn.Linear(38, 32)
        self.position = nn.Parameter(
            torch.randn(HISTORY_LEN + 1, 32) * 0.02
        )

    @property
    def current_dim(self):
        return sum(EMBED_DIMS.values())

    def current_features(self, current):
        pieces = []
        for field in CURRENT_FIELDS:
            values = current[field]
            if field in ("video_id", "author_id", "tag"):
                values = values + 1
            pieces.append(self.current_embeddings[field](values))
        return torch.cat(pieces, dim=1)

    def history_tokens(self, hv, ha, ht, hl):
        raw = torch.cat(
            [
                self.history_video(hv),
                self.history_author(ha),
                self.history_tag(ht),
                self.history_label(hl),
            ],
            dim=2,
        )
        return self.token_projection(raw)

    def candidate_token(self, current):
        n = len(current["video_id"])
        raw = torch.cat(
            [
                self.history_video(current["video_id"] + 1),
                self.history_author(current["author_id"] + 1),
                self.history_tag(current["tag"] + 1),
                torch.zeros(
                    (n, 4),
                    dtype=torch.float32,
                    device=current["video_id"].device,
                ),
            ],
            dim=1,
        )
        return self.token_projection(raw)


class BSTRanker(SequenceBase):
    """Candidate token participates in bidirectional sequence attention."""

    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=80,
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
            nn.Linear(self.current_dim + 32, 80),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(80, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, current, hv, ha, ht, hl):
        history = self.history_tokens(hv, ha, ht, hl)
        candidate = self.candidate_token(current).unsqueeze(1)
        tokens = torch.cat([history, candidate], dim=1)
        tokens = tokens + self.position.unsqueeze(0)

        history_padding = hv == 0
        target_padding = torch.zeros(
            (len(hv), 1),
            dtype=torch.bool,
            device=hv.device,
        )
        padding = torch.cat(
            [history_padding, target_padding], dim=1
        )

        encoded = self.transformer(
            tokens, src_key_padding_mask=padding
        )
        target_state = encoded[:, -1, :]
        features = torch.cat(
            [self.current_features(current), target_state], dim=1
        )
        return self.output(features).squeeze(1)


class SASRecRanker(SequenceBase):
    """Builds a causal state before seeing the candidate, then matches it."""

    def __init__(self):
        super().__init__()
        self.empty_state = nn.Parameter(torch.zeros(32))
        layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=80,
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
        self.current_linear = nn.Sequential(
            nn.Linear(self.current_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.match_scale = 32.0 ** -0.5
        causal = torch.triu(
            torch.ones(HISTORY_LEN, HISTORY_LEN, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal)

    def forward(self, current, hv, ha, ht, hl):
        tokens = self.history_tokens(hv, ha, ht, hl)
        tokens = tokens + self.position[:HISTORY_LEN].unsqueeze(0)
        padding = hv == 0
        no_history = padding.all(dim=1)

        if no_history.any():
            tokens = tokens.clone()
            padding = padding.clone()
            tokens[no_history, -1, :] = self.empty_state
            padding[no_history, -1] = False

        encoded = self.transformer(
            tokens,
            mask=self.causal_mask,
            src_key_padding_mask=padding,
        )
        state = encoded[:, -1, :]
        candidate = self.candidate_token(current)
        match = torch.sum(state * candidate, dim=1) * self.match_scale
        static_score = self.current_linear(
            self.current_features(current)
        ).squeeze(1)
        return static_score + match


def fit_model(model, name, train, histories, weights):
    hv, ha, ht, hl = histories
    labels = np.asarray(train.y, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    rng = np.random.RandomState(SEED + sum(map(ord, name)))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.4e-3,
        weight_decay=2e-5,
    )

    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(len(labels))
        total_loss = 0.0
        total_rows = 0

        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            current = current_batch(train, idx)
            bhv = torch.from_numpy(hv[idx].astype(np.int64, copy=False))
            bha = torch.from_numpy(ha[idx].astype(np.int64, copy=False))
            bht = torch.from_numpy(ht[idx].astype(np.int64, copy=False))
            bhl = torch.from_numpy(hl[idx].astype(np.int64, copy=False))
            yb = torch.from_numpy(labels[idx])
            wb = torch.from_numpy(weights[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(current, bhv, bha, bht, bhl)
            losses = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = torch.sum(losses * wb) / torch.sum(wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss.detach()) * len(idx)
            total_rows += len(idx)

        print(
            "FINDINGS {} epoch={} weighted_bce={:.6f}".format(
                name, epoch + 1, total_loss / max(total_rows, 1)
            ),
            flush=True,
        )

    return model


@torch.no_grad()
def predict_model(model, split, histories):
    hv, ha, ht, hl = histories
    output = np.empty(len(split.user_id), dtype=np.float64)
    model.eval()

    for start in range(0, len(output), 16384):
        end = min(start + 16384, len(output))
        idx = slice(start, end)
        current = current_batch(split, idx)
        bhv = torch.from_numpy(hv[idx].astype(np.int64, copy=False))
        bha = torch.from_numpy(ha[idx].astype(np.int64, copy=False))
        bht = torch.from_numpy(ht[idx].astype(np.int64, copy=False))
        bhl = torch.from_numpy(hl[idx].astype(np.int64, copy=False))
        output[start:end] = model(
            current, bhv, bha, bht, bhl
        ).cpu().numpy().astype(np.float64)

    return output


def hawkes_scores(split, histories):
    hv, ha, ht, hl = histories
    current_video = (
        np.asarray(split.video_id, dtype=np.int64)[:, None] + 1
    )
    current_author = (
        np.asarray(split.X["author_id"], dtype=np.int64)[:, None] + 1
    )
    current_tag = (
        np.asarray(split.X["tag"], dtype=np.int64)[:, None] + 1
    )

    mask = hv != 0
    signed_label = np.where(hl == 2, 1.0, -0.45)
    lag = np.arange(HISTORY_LEN - 1, -1, -1, dtype=np.float64)
    decay = np.exp(-lag / 3.0)[None, :]

    similarity = (
        1.30 * (hv == current_video)
        + 0.65 * (ha == current_author)
        + 0.35 * (ht == current_tag)
    )
    excitation = np.sum(
        mask * decay * signed_label * similarity, axis=1
    )
    normalization = np.sum(mask * decay, axis=1)
    return excitation / np.sqrt(1.0 + normalization)


def select_blend(name, raw_valid, valid, incumbent_valid):
    raw_rank = within_user_rank(valid.user_id, raw_valid)
    incumbent_rank = within_user_rank(
        valid.user_id, incumbent_valid
    )

    best_score = -np.inf
    best_alpha = 0.0
    best_predictions = incumbent_rank

    results = {}
    for alpha in np.linspace(0.0, 1.0, 11):
        predictions = (
            (1.0 - alpha) * incumbent_rank + alpha * raw_rank
        )
        metric = evaluate(
            valid.user_id, valid.y, predictions
        )["primary"]
        results[float(alpha)] = float(metric)
        if metric > best_score:
            best_score = float(metric)
            best_alpha = float(alpha)
            best_predictions = predictions.copy()

    print(
        "FINDINGS {} best_blend_alpha={:.1f} primary={:.6f}".format(
            name, best_alpha, best_score
        ),
        flush=True,
    )
    return best_score, best_alpha, best_predictions, results


train = load("train")
valid = load("valid")

history_builder = OrderedHistory(train)
train_histories = history_builder.for_train()
valid_histories = history_builder.for_split(valid)

train_weights = recency_weights(train.date, half_life=3.0)

bst = fit_model(
    BSTRanker(),
    "bst",
    train,
    train_histories,
    train_weights,
)
valid_bst = predict_model(bst, valid, valid_histories)

sasrec = fit_model(
    SASRecRanker(),
    "sasrec",
    train,
    train_histories,
    train_weights,
)
valid_sasrec = predict_model(sasrec, valid, valid_histories)

valid_hawkes = hawkes_scores(valid, valid_histories)

shared = os.environ.get("SHARED_ARTIFACTS", "")
incumbent_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
incumbent_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)
incumbent_valid = np.load(incumbent_valid_path).astype(np.float64)

raw_candidates = {
    "bst": valid_bst,
    "sasrec": valid_sasrec,
    "hawkes": valid_hawkes,
}

candidate_scores = {}
candidate_details = {}
best_name = None
best_primary = -np.inf
best_alpha = 0.0
best_valid_scores = None

for name, raw in raw_candidates.items():
    raw_metric = evaluate(valid.user_id, valid.y, raw)
    candidate_scores[name + "_raw"] = float(raw_metric["primary"])

    score, alpha, predictions, details = select_blend(
        name, raw, valid, incumbent_valid
    )
    candidate_scores[name + "_blend"] = float(score)
    candidate_details[name] = details

    if score > best_primary:
        best_primary = score
        best_name = name
        best_alpha = alpha
        best_valid_scores = predictions

print(
    "CANDIDATES " + json.dumps(
        candidate_scores, sort_keys=True
    ),
    flush=True,
)

best_raw_valid = raw_candidates[best_name]
metrics = evaluate(
    valid.user_id, valid.y, best_valid_scores
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
test_histories = history_builder.for_split(test)

if best_name == "bst":
    best_raw_test = predict_model(bst, test, test_histories)
elif best_name == "sasrec":
    best_raw_test = predict_model(sasrec, test, test_histories)
else:
    best_raw_test = hawkes_scores(test, test_histories)

incumbent_test = np.load(incumbent_test_path).astype(np.float64)
test_scores = (
    (1.0 - best_alpha)
    * within_user_rank(test.user_id, incumbent_test)
    + best_alpha
    * within_user_rank(test.user_id, best_raw_test)
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print(
    "FINDINGS selected={} alpha={:.1f}".format(
        best_name, best_alpha
    ),
    flush=True,
)

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)