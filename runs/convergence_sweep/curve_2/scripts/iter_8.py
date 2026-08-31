import os
import time
import json
import math
import random
import gc

import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 73129

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

train = load("train")
valid = load("valid")
test = load("test")

y_train_np = np.asarray(train.y, dtype=np.float32)
y_valid_np = np.asarray(valid.y, dtype=np.int8)

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "user_active_degree",
    "onehot_feat3",
    "onehot_feat8",
    "hour",
    "register_days_bucket",
    "music_type",
    "video_type",
]
CARDS = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1], dtype=np.int64)
TOTAL_CARD = int(sum(CARDS))
VIDEO_FIELD_INDEX = FIELDS.index("video_id")
VIDEO_CARD = int(FEATURE_CARDINALITIES["video_id"])
N_FIELDS = len(FIELDS)
HISTORY_LENGTH = 8


def make_cat(split):
    return np.ascontiguousarray(
        np.stack(
            [np.asarray(split.X[name], dtype=np.int64) for name in FIELDS],
            axis=1,
        ),
        dtype=np.int64,
    )


x_train_np = make_cat(train)
x_valid_np = make_cat(valid)
x_test_np = make_cat(test)


def chronological_history(user_ids, time_ms, video_ids, length):
    """
    Construct fixed-length histories using only impressions strictly earlier
    in the stable (user_id, time_ms, row-position) ordering. Column zero is
    the immediately preceding video. Zero is padding; real IDs are shifted
    by one.
    """
    users = np.asarray(user_ids, dtype=np.int64)
    times = np.asarray(time_ms, dtype=np.int64)
    videos = np.asarray(video_ids, dtype=np.int64)
    n = len(users)

    order = np.lexsort(
        (
            np.arange(n, dtype=np.int64),
            times,
            users,
        )
    )
    sorted_users = users[order]
    result = np.zeros((n, length), dtype=np.int64)

    for lag in range(1, length + 1):
        same_user = sorted_users[lag:] == sorted_users[:-lag]
        destinations = order[lag:][same_user]
        sources = order[:-lag][same_user]
        result[destinations, lag - 1] = videos[sources] + 1

    return result


def target_history_with_train_prefix(target):
    """
    Validation and test each receive the train tail plus earlier impressions
    from their own split. No labels or auxiliary outcomes are used.
    Validation is never used when constructing test features.
    """
    train_users = np.asarray(train.user_id, dtype=np.int64)
    target_users = np.asarray(target.user_id, dtype=np.int64)

    users = np.concatenate([train_users, target_users])
    times = np.concatenate([
        np.asarray(train.time_ms, dtype=np.int64),
        np.asarray(target.time_ms, dtype=np.int64),
    ])
    videos = np.concatenate([
        np.asarray(train.video_id, dtype=np.int64),
        np.asarray(target.video_id, dtype=np.int64),
    ])

    combined_history = chronological_history(
        users, times, videos, HISTORY_LENGTH
    )
    return np.ascontiguousarray(combined_history[len(train_users):])


history_train_np = chronological_history(
    train.user_id,
    train.time_ms,
    train.video_id,
    HISTORY_LENGTH,
)
history_valid_np = target_history_with_train_prefix(valid)
history_test_np = target_history_with_train_prefix(test)

history_coverage = {
    "train_any": float(np.mean(history_train_np[:, 0] != 0)),
    "valid_any": float(np.mean(history_valid_np[:, 0] != 0)),
    "test_any": float(np.mean(history_test_np[:, 0] != 0)),
}
print("FINDINGS history_coverage=" + json.dumps(history_coverage))

x_train = torch.from_numpy(x_train_np)
history_train = torch.from_numpy(history_train_np)
y_train = torch.from_numpy(y_train_np)

max_train_date = int(np.max(np.asarray(train.date, dtype=np.int32)))
train_age = (
    max_train_date - np.asarray(train.date, dtype=np.int32)
).astype(np.float32)
sample_weight_np = np.exp(
    -math.log(2.0) * train_age / 6.0
).astype(np.float32)
sample_weight_np /= float(sample_weight_np.mean())
sample_weight = torch.from_numpy(sample_weight_np)

offset_tensor = torch.from_numpy(OFFSETS.copy())


class GRU4RecScorer(nn.Module):
    """
    A chronological recent-impression encoder. The projection on the
    candidate branch explicitly fixes the previous 28-versus-20 dimension
    mismatch before the elementwise matching operation.
    """

    def __init__(self):
        super().__init__()
        field_dim = 10
        sequence_dim = 20
        hidden_dim = 28

        self.register_buffer("offsets", offset_tensor.clone())
        self.field_embedding = nn.Embedding(TOTAL_CARD, field_dim)
        self.sequence_embedding = nn.Embedding(
            VIDEO_CARD + 1,
            sequence_dim,
            padding_idx=0,
        )
        self.gru = nn.GRU(
            input_size=sequence_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.candidate_projection = nn.Linear(sequence_dim, hidden_dim)
        self.base = nn.Sequential(
            nn.Linear(N_FIELDS * field_dim, 80),
            nn.PReLU(),
            nn.Dropout(0.04),
            nn.Linear(80, 32),
            nn.PReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(32 + hidden_dim * 2, 40),
            nn.PReLU(),
            nn.Linear(40, 1),
        )

        nn.init.normal_(self.field_embedding.weight, std=0.025)
        nn.init.normal_(self.sequence_embedding.weight, std=0.025)
        with torch.no_grad():
            self.sequence_embedding.weight[0].zero_()

    def forward(self, x, history):
        field_state = self.base(
            self.field_embedding(x + self.offsets).flatten(1)
        )

        # Histories are stored newest-first. GRU receives oldest-first, with
        # padding before real events so its final state represents recency.
        sequence = torch.flip(history, dims=[1])
        sequence_emb = self.sequence_embedding(sequence)
        _, final_state = self.gru(sequence_emb)
        user_state = final_state[-1]

        candidate_id = x[:, VIDEO_FIELD_INDEX] + 1
        candidate = self.candidate_projection(
            self.sequence_embedding(candidate_id)
        )
        match = user_state * candidate

        features = torch.cat([field_state, user_state, match], dim=1)
        return self.output(features).squeeze(1)


class SASRecScorer(nn.Module):
    """
    A compact self-attentive history encoder. Unlike GRU recurrence, every
    recent impression can interact directly with every other impression.
    """

    def __init__(self):
        super().__init__()
        field_dim = 10
        model_dim = 24

        self.register_buffer("offsets", offset_tensor.clone())
        self.field_embedding = nn.Embedding(TOTAL_CARD, field_dim)
        self.sequence_embedding = nn.Embedding(
            VIDEO_CARD + 1,
            model_dim,
            padding_idx=0,
        )
        self.position_embedding = nn.Embedding(
            HISTORY_LENGTH,
            model_dim,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=64,
            dropout=0.04,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.base = nn.Sequential(
            nn.Linear(N_FIELDS * field_dim, 80),
            nn.PReLU(),
            nn.Dropout(0.04),
            nn.Linear(80, 32),
            nn.PReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(32 + model_dim * 2, 40),
            nn.PReLU(),
            nn.Linear(40, 1),
        )

        nn.init.normal_(self.field_embedding.weight, std=0.025)
        nn.init.normal_(self.sequence_embedding.weight, std=0.025)
        nn.init.normal_(self.position_embedding.weight, std=0.025)
        with torch.no_grad():
            self.sequence_embedding.weight[0].zero_()

    def forward(self, x, history):
        field_state = self.base(
            self.field_embedding(x + self.offsets).flatten(1)
        )

        padding_mask = history.eq(0)
        positions = torch.arange(
            HISTORY_LENGTH, device=history.device
        ).unsqueeze(0)
        sequence = (
            self.sequence_embedding(history)
            + self.position_embedding(positions)
        )
        encoded = self.encoder(
            sequence,
            src_key_padding_mask=padding_mask,
        )

        valid = (~padding_mask).unsqueeze(2).to(encoded.dtype)
        denominator = valid.sum(dim=1).clamp_min(1.0)
        user_state = (encoded * valid).sum(dim=1) / denominator

        candidate_id = x[:, VIDEO_FIELD_INDEX] + 1
        candidate = self.sequence_embedding(candidate_id)
        match = user_state * candidate

        features = torch.cat([field_state, user_state, match], dim=1)
        return self.output(features).squeeze(1)


CONTEXT_FIELDS = [
    "user_id",
    "tab",
    "hour",
    "user_active_degree",
    "register_days_bucket",
]
CONTEXT_INDICES = [FIELDS.index(name) for name in CONTEXT_FIELDS]
CONTEXT_CARDS = [
    int(FEATURE_CARDINALITIES[name]) for name in CONTEXT_FIELDS
]
CONTEXT_OFFSETS = np.cumsum(
    [0] + CONTEXT_CARDS[:-1], dtype=np.int64
)
CONTEXT_TOTAL = int(sum(CONTEXT_CARDS))


class CorrectedTwoTower(nn.Module):
    """
    A user/context tower and an item tower trained with sampled softmax.
    The training logits subtract log sampling probability, preventing popular
    negatives from being over-penalized solely because they are sampled more.
    """

    def __init__(self):
        super().__init__()
        embedding_dim = 24

        self.register_buffer(
            "context_indices",
            torch.tensor(CONTEXT_INDICES, dtype=torch.long),
        )
        self.register_buffer(
            "context_offsets",
            torch.from_numpy(CONTEXT_OFFSETS.copy()),
        )
        self.context_embedding = nn.Embedding(
            CONTEXT_TOTAL, embedding_dim
        )
        self.video_embedding = nn.Embedding(
            VIDEO_CARD, embedding_dim
        )
        self.video_bias = nn.Embedding(VIDEO_CARD, 1)
        self.user_tower = nn.Sequential(
            nn.Linear(len(CONTEXT_FIELDS) * embedding_dim, 72),
            nn.PReLU(),
            nn.Linear(72, embedding_dim),
        )

        nn.init.normal_(self.context_embedding.weight, std=0.025)
        nn.init.normal_(self.video_embedding.weight, std=0.025)
        nn.init.zeros_(self.video_bias.weight)

    def user_representation(self, x):
        context = x[:, self.context_indices] + self.context_offsets
        return self.user_tower(
            self.context_embedding(context).flatten(1)
        )

    def score(self, x):
        user_state = self.user_representation(x)
        video = x[:, VIDEO_FIELD_INDEX]
        item_state = self.video_embedding(video)
        bias = self.video_bias(video).squeeze(1)
        return (
            (user_state * item_state).sum(dim=1)
            / math.sqrt(item_state.shape[1])
            + bias
        )

    def sampled_logits(self, x, negative_ids, log_q):
        user_state = self.user_representation(x)
        positive_ids = x[:, VIDEO_FIELD_INDEX]

        positive_item = self.video_embedding(positive_ids)
        positive = (
            (user_state * positive_item).sum(dim=1)
            / math.sqrt(positive_item.shape[1])
            + self.video_bias(positive_ids).squeeze(1)
            - log_q[positive_ids]
        )

        negative_item = self.video_embedding(negative_ids)
        negative = (
            torch.einsum("bd,bkd->bk", user_state, negative_item)
            / math.sqrt(negative_item.shape[2])
            + self.video_bias(negative_ids).squeeze(2)
            - log_q[negative_ids]
        )

        return torch.cat([positive.unsqueeze(1), negative], dim=1)


@torch.no_grad()
def predict_sequence(model, x_np, history_np, batch_size=16384):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float32)
    for begin in range(0, len(x_np), batch_size):
        end = min(begin + batch_size, len(x_np))
        xb = torch.from_numpy(x_np[begin:end])
        hb = torch.from_numpy(history_np[begin:end])
        result[begin:end] = (
            model(xb, hb).detach().cpu().numpy()
        )
    return result


@torch.no_grad()
def predict_two_tower(model, x_np, batch_size=32768):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float32)
    for begin in range(0, len(x_np), batch_size):
        end = min(begin + batch_size, len(x_np))
        xb = torch.from_numpy(x_np[begin:end])
        result[begin:end] = (
            model.score(xb).detach().cpu().numpy()
        )
    return result


def train_sequence_family(model, name, epochs):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.2e-3,
        weight_decay=3e-6,
    )
    generator = torch.Generator()
    generator.manual_seed(SEED + sum(map(ord, name)))

    epoch_scores = []
    best_primary = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(
            len(y_train_np), generator=generator
        )

        for begin in range(0, len(y_train_np), 4096):
            idx = permutation[begin:begin + 4096]
            logits = model(x_train[idx], history_train[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits,
                y_train[idx],
                reduction="none",
            )
            weights = sample_weight[idx]
            loss = (losses * weights).sum() / weights.sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        validation_scores = predict_sequence(
            model, x_valid_np, history_valid_np
        )
        metrics = evaluate(
            valid.user_id, valid.y, validation_scores
        )
        primary = float(metrics["primary"])
        epoch_scores.append(primary)

        if primary > best_primary:
            best_primary = primary
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    print(
        "FINDINGS %s_epoch_primary=%s"
        % (name, json.dumps(epoch_scores))
    )

    return (
        predict_sequence(model, x_valid_np, history_valid_np),
        predict_sequence(model, x_test_np, history_test_np),
    )


video_counts = np.bincount(
    x_train_np[:, VIDEO_FIELD_INDEX],
    minlength=VIDEO_CARD,
).astype(np.float64)
video_q_np = (video_counts + 1.0) ** 0.75
video_q_np /= video_q_np.sum()
video_q = torch.from_numpy(video_q_np.astype(np.float32))
video_log_q = torch.log(video_q.clamp_min(1e-12))


def train_two_tower(model, epochs=2):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.5e-3,
        weight_decay=5e-6,
    )
    generator = torch.Generator()
    generator.manual_seed(SEED + 9901)

    epoch_scores = []
    best_primary = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(
            len(y_train_np), generator=generator
        )

        for begin in range(0, len(y_train_np), 4096):
            idx = permutation[begin:begin + 4096]
            xb = x_train[idx]
            batch_size = len(idx)

            negative_ids = torch.multinomial(
                video_q,
                batch_size * 8,
                replacement=True,
                generator=generator,
            ).reshape(batch_size, 8)

            sampled_logits = model.sampled_logits(
                xb, negative_ids, video_log_q
            )
            targets = torch.zeros(
                batch_size, dtype=torch.long
            )
            losses = nn.functional.cross_entropy(
                sampled_logits,
                targets,
                reduction="none",
            )

            # Sampled-softmax learns only from positive long-view events.
            # Negative rows receive a lightweight pointwise term so the
            # logged candidate set still informs the decision boundary.
            positive_mask = y_train[idx] > 0.5
            pointwise_logits = model.score(xb)
            pointwise_losses = (
                nn.functional.binary_cross_entropy_with_logits(
                    pointwise_logits,
                    y_train[idx],
                    reduction="none",
                )
            )
            combined = pointwise_losses.clone()
            combined[positive_mask] += losses[positive_mask]

            weights = sample_weight[idx]
            loss = (combined * weights).sum() / weights.sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        validation_scores = predict_two_tower(
            model, x_valid_np
        )
        metrics = evaluate(
            valid.user_id, valid.y, validation_scores
        )
        primary = float(metrics["primary"])
        epoch_scores.append(primary)

        if primary > best_primary:
            best_primary = primary
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    print(
        "FINDINGS corrected_two_tower_epoch_primary="
        + json.dumps(epoch_scores)
    )

    return (
        predict_two_tower(model, x_valid_np),
        predict_two_tower(model, x_test_np),
    )


family_predictions = {}

gru = GRU4RecScorer()
family_predictions["gru4rec"] = train_sequence_family(
    gru, "gru4rec", epochs=2
)
del gru
gc.collect()

sasrec = SASRecScorer()
family_predictions["sasrec"] = train_sequence_family(
    sasrec, "sasrec", epochs=1
)
del sasrec
gc.collect()

two_tower = CorrectedTwoTower()
family_predictions["corrected_two_tower"] = train_two_tower(
    two_tower, epochs=2
)
del two_tower
gc.collect()


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    n = len(values)

    order = np.lexsort(
        (np.arange(n, dtype=np.int64), values, users)
    )
    sorted_users = users[order]

    starts_flag = np.r_[
        True, sorted_users[1:] != sorted_users[:-1]
    ]
    group_starts = np.maximum.accumulate(
        np.where(starts_flag, np.arange(n), 0)
    )
    positions = np.arange(n) - group_starts

    ends_flag = np.r_[
        sorted_users[1:] != sorted_users[:-1], True
    ]
    group_ends = np.minimum.accumulate(
        np.where(ends_flag, np.arange(n), n - 1)[::-1]
    )[::-1]
    counts = group_ends - group_starts + 1

    normalized = np.where(
        counts > 1,
        positions / np.maximum(counts - 1, 1),
        0.5,
    ).astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[order] = normalized
    return result


shared = os.environ.get("SHARED_ARTIFACTS", "")
incumbent_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
incumbent_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = within_user_rank(
    valid.user_id, incumbent_valid
)
inc_test_rank = within_user_rank(
    test.user_id, incumbent_test
)

candidate_scores = {}
best_primary = -1.0
best_valid = None
best_test = None
best_raw_valid = None
best_name = None

for family_name, (raw_valid, raw_test) in family_predictions.items():
    raw_metrics = evaluate(
        valid.user_id, valid.y, raw_valid
    )
    raw_primary = float(raw_metrics["primary"])
    candidate_scores[family_name] = raw_primary

    if raw_primary > best_primary:
        best_primary = raw_primary
        best_valid = raw_valid.astype(np.float64)
        best_test = raw_test.astype(np.float64)
        best_raw_valid = raw_valid.astype(np.float64)
        best_name = family_name

    family_valid_rank = within_user_rank(
        valid.user_id, raw_valid
    )
    family_test_rank = within_user_rank(
        test.user_id, raw_test
    )

    for own_weight in (0.20, 0.40, 0.60, 0.80):
        blended_valid = (
            own_weight * family_valid_rank
            + (1.0 - own_weight) * inc_valid_rank
        )
        blended_test = (
            own_weight * family_test_rank
            + (1.0 - own_weight) * inc_test_rank
        )

        metrics = evaluate(
            valid.user_id, valid.y, blended_valid
        )
        primary = float(metrics["primary"])
        name = "%s_blend_%.2f" % (
            family_name, own_weight
        )
        candidate_scores[name] = primary

        if primary > best_primary:
            best_primary = primary
            best_valid = blended_valid
            best_test = blended_test
            best_raw_valid = raw_valid.astype(np.float64)
            best_name = name

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print("FINDINGS selected=" + str(best_name))

final_metrics = evaluate(
    valid.user_id, valid.y, best_valid
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )

    if "_blend_" in best_name:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(final_metrics["primary"]),
            "gauc": float(final_metrics["gauc"]),
            "ndcg@5": float(final_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)