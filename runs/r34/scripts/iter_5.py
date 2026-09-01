import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


torch.set_num_threads(min(8, os.cpu_count() or 1))
torch.manual_seed(2027)
np.random.seed(2027)

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "onehot_feat3",
    "upload_type",
    "duration_bucket",
    "onehot_feat1",
    "onehot_feat8",
    "onehot_feat7",
    "user_active_degree",
    "music_type",
    "hour",
]
VIDEO_FIELD_INDEX = FIELDS.index("video_id")
USER_CARDINALITY = int(FEATURE_CARDINALITIES["user_id"])
HISTORY_LENGTH = 12

CARDINALITIES = np.asarray(
    [int(FEATURE_CARDINALITIES[name]) for name in FIELDS],
    dtype=np.int64,
)
OFFSETS = np.zeros(len(FIELDS), dtype=np.int64)
OFFSETS[1:] = np.cumsum(CARDINALITIES[:-1], dtype=np.int64)
TOTAL_CARDINALITY = int(CARDINALITIES.sum())
VIDEO_OFFSET = int(OFFSETS[VIDEO_FIELD_INDEX])


def make_features(split):
    x = np.column_stack([split.X[name] for name in FIELDS]).astype(
        np.int64, copy=False
    )
    x = x + OFFSETS[None, :]
    return torch.from_numpy(np.ascontiguousarray(x))


def make_causal_train_histories(train):
    """
    For every training row, construct the K most recent positive videos from
    the same user strictly before that row. Sorting uses date followed by the
    original logged row order, and the current row's label is always excluded.
    """
    users = np.asarray(train.X["user_id"], dtype=np.int64)
    videos = np.asarray(train.X["video_id"], dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.int64)
    dates = np.asarray(train.date, dtype=np.int64)
    row_number = np.arange(users.size, dtype=np.int64)

    order = np.lexsort((row_number, dates, users))
    sorted_users = users[order]
    sorted_videos = videos[order]
    sorted_labels = labels[order]

    starts = np.empty(sorted_users.size, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    start_indices = np.flatnonzero(starts)
    lengths = np.diff(np.append(start_indices, sorted_users.size))

    cumulative_positives = np.cumsum(sorted_labels, dtype=np.int64)
    positives_before_group = (
        cumulative_positives[start_indices] - sorted_labels[start_indices]
    )
    group_bases = np.repeat(positives_before_group, lengths)

    prior_positive_count = (
        cumulative_positives - sorted_labels - group_bases
    )
    positive_videos = sorted_videos[sorted_labels == 1]

    sorted_history = np.zeros(
        (sorted_users.size, HISTORY_LENGTH), dtype=np.int64
    )
    for column in range(HISTORY_LENGTH):
        lag = column + 1
        available = prior_positive_count >= lag
        positive_ordinal = (
            group_bases[available] + prior_positive_count[available] - lag
        )
        sorted_history[available, column] = positive_videos[positive_ordinal]

    history = np.empty_like(sorted_history)
    history[order] = sorted_history
    return history


def make_frozen_user_profiles(train):
    """
    Histories for future splits. These contain only training outcomes and are
    therefore identical in information availability for validation and test.
    """
    users = np.asarray(train.X["user_id"], dtype=np.int64)
    videos = np.asarray(train.X["video_id"], dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.int64)
    dates = np.asarray(train.date, dtype=np.int64)
    rows = np.arange(users.size, dtype=np.int64)

    positive_rows = np.flatnonzero(labels == 1)
    order = np.lexsort(
        (
            rows[positive_rows],
            dates[positive_rows],
            users[positive_rows],
        )
    )
    pos_users = users[positive_rows][order]
    pos_videos = videos[positive_rows][order]

    profile = np.zeros(
        (USER_CARDINALITY, HISTORY_LENGTH), dtype=np.int64
    )
    if pos_users.size == 0:
        return profile

    starts = np.empty(pos_users.size, dtype=bool)
    starts[0] = True
    starts[1:] = pos_users[1:] != pos_users[:-1]
    start_indices = np.flatnonzero(starts)
    end_indices = np.append(start_indices[1:], pos_users.size)
    lengths = end_indices - start_indices
    repeated_ends = np.repeat(end_indices, lengths)

    reverse_rank = repeated_ends - np.arange(pos_users.size) - 1
    keep = reverse_rank < HISTORY_LENGTH
    profile[pos_users[keep], reverse_rank[keep]] = pos_videos[keep]
    return profile


def histories_for_split(split, profiles):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    result = np.zeros((users.size, HISTORY_LENGTH), dtype=np.int64)
    known = (users >= 0) & (users < profiles.shape[0])
    result[known] = profiles[users[known]]
    return result


class HistoryDeepFM(nn.Module):
    def __init__(self, embedding_dim=16):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.linear = nn.Embedding(
            TOTAL_CARDINALITY, 1, sparse=True
        )
        self.embedding = nn.Embedding(
            TOTAL_CARDINALITY, embedding_dim, sparse=True
        )
        self.bias = nn.Parameter(torch.zeros(()))

        deep_input_dim = (
            len(FIELDS) * embedding_dim + 4 * embedding_dim
        )
        self.deep = nn.Sequential(
            nn.Linear(deep_input_dim, 160),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(160, 80),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(80, 1),
        )

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Begin close to the FM solution and let the history tower enter
        # gradually rather than overwhelming the sparse identifier signal.
        nn.init.normal_(self.deep[-1].weight, mean=0.0, std=0.01)

    def forward(self, x, history_video_ids):
        field_vectors = self.embedding(x)

        linear_term = self.linear(x).sum(dim=1).squeeze(-1)
        summed = field_vectors.sum(dim=1)
        fm_term = 0.5 * (
            summed.square().sum(dim=1)
            - field_vectors.square().sum(dim=(1, 2))
        )

        candidate = field_vectors[:, VIDEO_FIELD_INDEX, :]

        history_mask = history_video_ids.ne(0)
        history_global_ids = history_video_ids + VIDEO_OFFSET
        history_vectors = self.embedding(history_global_ids)

        attention_logits = (
            history_vectors * candidate.unsqueeze(1)
        ).sum(dim=2) / np.sqrt(float(self.embedding_dim))

        attention_logits = attention_logits - attention_logits.max(
            dim=1, keepdim=True
        ).values
        attention_weights = (
            torch.exp(attention_logits) * history_mask.float()
        )
        attention_weights = attention_weights / (
            attention_weights.sum(dim=1, keepdim=True) + 1.0e-8
        )
        pooled_history = (
            attention_weights.unsqueeze(2) * history_vectors
        ).sum(dim=1)

        deep_input = torch.cat(
            [
                field_vectors.flatten(start_dim=1),
                pooled_history,
                candidate * pooled_history,
                torch.abs(candidate - pooled_history),
                candidate,
            ],
            dim=1,
        )
        deep_term = self.deep(deep_input).squeeze(1)

        return self.bias + linear_term + fm_term + deep_term


@torch.inference_mode()
def predict(model, x, histories, batch_size=32768):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        logits = model(x[start:end], histories[start:end])
        result[start:end] = logits.double().numpy()
    return result


train = load("train")
valid = load("valid")

x_train = make_features(train)
x_valid = make_features(valid)
y_train = torch.from_numpy(
    np.asarray(train.y, dtype=np.float32)
)
y_valid_np = np.asarray(valid.y)

train_history_np = make_causal_train_histories(train)
user_profiles = make_frozen_user_profiles(train)
valid_history_np = histories_for_split(valid, user_profiles)

train_histories = torch.from_numpy(
    np.ascontiguousarray(train_history_np)
)
valid_histories = torch.from_numpy(
    np.ascontiguousarray(valid_history_np)
)

del train_history_np
del valid_history_np

train_coverage = float(
    train_histories.ne(0).any(dim=1).float().mean().item()
)
valid_coverage = float(
    valid_histories.ne(0).any(dim=1).float().mean().item()
)
print(
    "FINDINGS "
    + json.dumps(
        {
            "train_rows_with_prior_positive_history": train_coverage,
            "valid_rows_with_train_positive_history": valid_coverage,
        },
        separators=(",", ":"),
    )
)

model = HistoryDeepFM(embedding_dim=16)

sparse_optimizer = torch.optim.SparseAdam(
    [model.linear.weight, model.embedding.weight],
    lr=0.0012,
)
dense_parameters = [
    parameter
    for name, parameter in model.named_parameters()
    if name not in ("linear.weight", "embedding.weight")
]
dense_optimizer = torch.optim.Adam(
    dense_parameters,
    lr=0.001,
    weight_decay=1.0e-6,
)

batch_size = 8192
num_epochs = 9
n = x_train.shape[0]
generator = torch.Generator()
generator.manual_seed(2027)

best_primary = -np.inf
best_metrics = None
best_state = None
epoch_candidates = {}

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(n, generator=generator)

    for start in range(0, n, batch_size):
        indices = permutation[start:start + batch_size]

        sparse_optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)

        logits = model(
            x_train[indices],
            train_histories[indices],
        )
        loss = F.binary_cross_entropy_with_logits(
            logits, y_train[indices]
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(dense_parameters, 5.0)
        sparse_optimizer.step()
        dense_optimizer.step()

    valid_scores = predict(
        model, x_valid, valid_histories
    )
    metrics = evaluate(
        valid.user_id, y_valid_np, valid_scores
    )
    primary = float(metrics["primary"])
    epoch_candidates["epoch_%d" % (epoch + 1)] = primary

    if primary > best_primary:
        best_primary = primary
        best_metrics = {
            key: float(value)
            for key, value in metrics.items()
        }
        best_state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

model.load_state_dict(best_state)

valid_scores = predict(model, x_valid, valid_histories)
best_metrics = {
    key: float(value)
    for key, value in evaluate(
        valid.user_id, y_valid_np, valid_scores
    ).items()
}

print(
    "CANDIDATES "
    + json.dumps(epoch_candidates, separators=(",", ":"))
)

test = load("test")
x_test = make_features(test)
test_history_np = histories_for_split(test, user_profiles)
test_histories = torch.from_numpy(
    np.ascontiguousarray(test_history_np)
)
test_scores = predict(model, x_test, test_histories)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": 0.0,
        },
        separators=(",", ":"),
    )
)