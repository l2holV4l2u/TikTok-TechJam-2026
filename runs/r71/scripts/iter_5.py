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
SEED = 2026
K = 16
LR = 0.001
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 16384
EPOCHS = 6
HISTORY_LENGTH = 12

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "video_type",
    "music_type",
    "hour",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_bucket",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

HISTORY_FIELDS = ["video_id", "author_id", "tag"]
FIELD_POSITION = {name: i for i, name in enumerate(FIELDS)}

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))

cardinalities = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))

history_field_positions = [
    FIELD_POSITION[name] for name in HISTORY_FIELDS
]
history_field_offsets = np.asarray(
    [offsets[FIELD_POSITION[name]] for name in HISTORY_FIELDS],
    dtype=np.int64,
)
n_users = int(FEATURE_CARDINALITIES["user_id"])


def make_matrix(split):
    columns = [
        np.asarray(split.X[name], dtype=np.int64) + offsets[j]
        for j, name in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(
        np.column_stack(columns), dtype=np.int64
    )


def chronological_order(split):
    n = len(split.user_id)
    return np.lexsort(
        (
            np.arange(n, dtype=np.int64),
            np.asarray(split.time_ms, dtype=np.int64),
            np.asarray(split.user_id, dtype=np.int64),
        )
    )


def make_positive_store(split, labels):
    labels = np.asarray(labels, dtype=np.int8)
    order = chronological_order(split)
    ordered_users = np.asarray(
        split.user_id, dtype=np.int64
    )[order]
    ordered_labels = labels[order]
    positive_mask = ordered_labels == 1
    positive_users = ordered_users[positive_mask]

    positive_counts = np.bincount(
        positive_users, minlength=n_users
    ).astype(np.int64)

    user_offsets = np.empty(n_users + 1, dtype=np.int64)
    user_offsets[0] = 0
    np.cumsum(positive_counts, out=user_offsets[1:])

    positive_values = {}
    for name in HISTORY_FIELDS:
        positive_values[name] = np.asarray(
            split.X[name], dtype=np.int64
        )[order][positive_mask]

    return {
        "order": order,
        "ordered_users": ordered_users,
        "ordered_labels": ordered_labels,
        "counts": positive_counts,
        "offsets": user_offsets,
        "values": positive_values,
    }


def make_self_histories(split, labels, store=None):
    if store is None:
        store = make_positive_store(split, labels)

    order = store["order"]
    ordered_users = store["ordered_users"]
    ordered_labels = store["ordered_labels"]
    n = len(order)

    cumulative = np.cumsum(
        ordered_labels, dtype=np.int64
    )

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = ordered_users[1:] != ordered_users[:-1]
    start_indices = np.flatnonzero(starts)
    group_lengths = np.diff(
        np.append(start_indices, n)
    )

    group_baselines = (
        cumulative[start_indices]
        - ordered_labels[start_indices]
    )
    baseline_per_row = np.repeat(
        group_baselines, group_lengths
    )

    counts_before = (
        cumulative - ordered_labels - baseline_per_row
    )
    base_indices = (
        store["offsets"][ordered_users] + counts_before
    )

    history_ordered = np.empty(
        (n, HISTORY_LENGTH, len(HISTORY_FIELDS)),
        dtype=np.int32,
    )
    history_mask_ordered = np.empty(
        (n, HISTORY_LENGTH), dtype=np.float32
    )

    for lag_index in range(HISTORY_LENGTH):
        lag = lag_index + 1
        available = counts_before >= lag
        history_mask_ordered[:, lag_index] = available

        source_indices = np.maximum(
            base_indices - lag, 0
        )

        for field_index, name in enumerate(HISTORY_FIELDS):
            values = store["values"][name]
            gathered = np.zeros(n, dtype=np.int64)
            if len(values):
                gathered[available] = values[
                    source_indices[available]
                ]
            gathered += history_field_offsets[field_index]
            history_ordered[
                :, lag_index, field_index
            ] = gathered.astype(np.int32)

    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n, dtype=np.int64)

    history = np.ascontiguousarray(
        history_ordered[inverse], dtype=np.int32
    )
    history_mask = np.ascontiguousarray(
        history_mask_ordered[inverse], dtype=np.float32
    )

    return history, history_mask, store


def make_external_histories(target, source_store):
    users = np.asarray(target.user_id, dtype=np.int64)
    counts = source_store["counts"][users]
    base_indices = (
        source_store["offsets"][users] + counts
    )
    n = len(users)

    history = np.empty(
        (n, HISTORY_LENGTH, len(HISTORY_FIELDS)),
        dtype=np.int32,
    )
    history_mask = np.empty(
        (n, HISTORY_LENGTH), dtype=np.float32
    )

    for lag_index in range(HISTORY_LENGTH):
        lag = lag_index + 1
        available = counts >= lag
        history_mask[:, lag_index] = available
        source_indices = np.maximum(base_indices - lag, 0)

        for field_index, name in enumerate(HISTORY_FIELDS):
            values = source_store["values"][name]
            gathered = np.zeros(n, dtype=np.int64)
            if len(values):
                gathered[available] = values[
                    source_indices[available]
                ]
            gathered += history_field_offsets[field_index]
            history[
                :, lag_index, field_index
            ] = gathered.astype(np.int32)

    return (
        np.ascontiguousarray(history, dtype=np.int32),
        np.ascontiguousarray(history_mask, dtype=np.float32),
    )


class DINDeepFM(nn.Module):
    def __init__(self, initial_bias):
        super().__init__()

        self.embedding = nn.Embedding(
            total_cardinality, K + 1, sparse=True
        )
        self.bias = nn.Parameter(
            torch.tensor(float(initial_bias), dtype=torch.float32)
        )

        interest_dim = len(HISTORY_FIELDS) * K
        base_dim = len(FIELDS) * K

        self.attention = nn.Sequential(
            nn.Linear(interest_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.deep = nn.Sequential(
            nn.Linear(base_dim + interest_dim, 160),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1),
        )

        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(
                mean=0.0, std=0.01
            )

        for module in self.attention:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight, a=np.sqrt(5.0)
                )
                nn.init.zeros_(module.bias)

        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight, a=np.sqrt(5.0)
                )
                nn.init.zeros_(module.bias)

        with torch.no_grad():
            self.attention[-1].weight.mul_(0.1)
            self.deep[-1].weight.mul_(0.05)

    def forward(self, x, history, history_mask):
        base_embedding = self.embedding(x)
        linear = base_embedding[:, :, 0].sum(dim=1)
        vectors = base_embedding[:, :, 1:]

        summed = vectors.sum(dim=1)
        fm_logit = 0.5 * (
            summed.square() - vectors.square().sum(dim=1)
        ).sum(dim=1)

        candidate_vectors = vectors[
            :, history_field_positions, :
        ].reshape(vectors.shape[0], -1)

        history_embedding = self.embedding(history)[
            :, :, :, 1:
        ].reshape(
            history.shape[0],
            history.shape[1],
            -1,
        )

        query = candidate_vectors.unsqueeze(1).expand_as(
            history_embedding
        )
        attention_input = torch.cat(
            [
                query,
                history_embedding,
                query - history_embedding,
                query * history_embedding,
            ],
            dim=2,
        )

        attention_logits = self.attention(
            attention_input
        ).squeeze(2)
        attention_logits = attention_logits - torch.max(
            attention_logits, dim=1, keepdim=True
        ).values

        attention_weights = (
            torch.exp(attention_logits) * history_mask
        )
        attention_weights = attention_weights / (
            attention_weights.sum(dim=1, keepdim=True)
            + 1e-8
        )

        interest = torch.sum(
            history_embedding
            * attention_weights.unsqueeze(2),
            dim=1,
        )

        deep_input = torch.cat(
            [
                vectors.reshape(vectors.shape[0], -1),
                interest,
            ],
            dim=1,
        )
        deep_logit = self.deep(deep_input).squeeze(1)

        return self.bias + linear + fm_logit + deep_logit


def fit_model(X, history, history_mask, labels, seed):
    torch.manual_seed(seed)

    labels = np.asarray(labels, dtype=np.float32)
    rate = float(np.mean(labels))
    initial_bias = np.log(
        np.clip(rate, 1e-6, 1.0 - 1e-6)
        / np.clip(1.0 - rate, 1e-6, 1.0)
    )

    model = DINDeepFM(initial_bias)

    sparse_optimizer = torch.optim.SparseAdam(
        [model.embedding.weight], lr=LR
    )
    dense_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name != "embedding.weight"
    ]
    dense_optimizer = torch.optim.Adam(
        dense_parameters,
        lr=LR,
        weight_decay=1e-6,
    )

    X_tensor = torch.from_numpy(X)
    history_tensor = torch.from_numpy(history)
    mask_tensor = torch.from_numpy(history_mask)
    y_tensor = torch.from_numpy(labels)

    generator = torch.Generator()
    generator.manual_seed(seed + 31)
    n = len(labels)

    model.train()
    for epoch in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        running_loss = 0.0

        for begin in range(0, n, BATCH_SIZE):
            idx = permutation[begin:begin + BATCH_SIZE]

            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)

            logits = model(
                X_tensor[idx],
                history_tensor[idx],
                mask_tensor[idx],
            )
            loss = F.binary_cross_entropy_with_logits(
                logits, y_tensor[idx]
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                dense_parameters, 5.0
            )
            sparse_optimizer.step()
            dense_optimizer.step()

            running_loss += float(loss.detach()) * len(idx)

        print(
            "FINDINGS "
            + json.dumps(
                {
                    "phase": "fit",
                    "epoch": epoch + 1,
                    "loss": running_loss / n,
                    "history_length": HISTORY_LENGTH,
                }
            )
        )

    return model


def predict(model, X, history, history_mask):
    model.eval()
    X_tensor = torch.from_numpy(X)
    history_tensor = torch.from_numpy(history)
    mask_tensor = torch.from_numpy(history_mask)

    scores = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for begin in range(0, len(X), PRED_BATCH_SIZE):
            end = min(begin + PRED_BATCH_SIZE, len(X))
            scores[begin:end] = model(
                X_tensor[begin:end],
                history_tensor[begin:end],
                mask_tensor[begin:end],
            ).numpy()

    return scores.astype(np.float64)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort(
        (
            np.arange(n, dtype=np.int64),
            scores,
            user_ids,
        )
    )
    ordered_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = ordered_users[1:] != ordered_users[:-1]
    start_indices = np.flatnonzero(starts)
    lengths = np.diff(np.append(start_indices, n))
    positions = (
        np.arange(n, dtype=np.int64)
        - np.repeat(start_indices, lengths)
    )
    denominators = np.maximum(
        np.repeat(lengths, lengths) - 1, 1
    )
    ordered_ranks = positions / denominators

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ordered_ranks
    return ranks


def load_incumbent(valid_length):
    artifact_dir = os.environ.get("RUN_ARTIFACTS")
    if not artifact_dir:
        return None, None

    valid_path = os.path.join(
        artifact_dir, "incumbent_valid_scores.npy"
    )
    test_path = os.path.join(
        artifact_dir, "incumbent_test_scores.npy"
    )
    if not (
        os.path.exists(valid_path)
        and os.path.exists(test_path)
    ):
        return None, None

    valid_scores = np.asarray(
        np.load(valid_path), dtype=np.float64
    )
    test_scores = np.asarray(
        np.load(test_path), dtype=np.float64
    )
    if len(valid_scores) != valid_length:
        return None, None

    return valid_scores, test_scores


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
X_train = make_matrix(train)
X_valid = make_matrix(valid)

train_history, train_mask, train_store = make_self_histories(
    train, y_train
)
valid_history, valid_mask = make_external_histories(
    valid, train_store
)

print(
    "FINDINGS "
    + json.dumps(
        {
            "train_mean_available_history": float(
                train_mask.sum(axis=1).mean()
            ),
            "valid_mean_available_history": float(
                valid_mask.sum(axis=1).mean()
            ),
            "valid_no_positive_history_fraction": float(
                np.mean(valid_mask.sum(axis=1) == 0)
            ),
        }
    )
)

model = fit_model(
    X_train,
    train_history,
    train_mask,
    y_train,
    SEED,
)
din_valid = predict(
    model, X_valid, valid_history, valid_mask
)

incumbent_valid, incumbent_test = load_incumbent(
    len(valid.user_id)
)

candidate_scores = {}
candidate_specs = {}

candidate_scores["din_standalone"] = din_valid
candidate_specs["din_standalone"] = {
    "mode": "raw",
    "alpha": 1.0,
}

if incumbent_valid is not None:
    for alpha in np.linspace(0.0, 1.0, 11):
        name = "raw_blend_{:.1f}".format(alpha)
        candidate_scores[name] = (
            (1.0 - alpha) * incumbent_valid
            + alpha * din_valid
        )
        candidate_specs[name] = {
            "mode": "raw",
            "alpha": float(alpha),
        }

    incumbent_rank = within_user_rank(
        valid.user_id, incumbent_valid
    )
    din_rank = within_user_rank(valid.user_id, din_valid)

    for alpha in np.linspace(0.1, 0.9, 9):
        name = "rank_blend_{:.1f}".format(alpha)
        candidate_scores[name] = (
            (1.0 - alpha) * incumbent_rank
            + alpha * din_rank
        )
        candidate_specs[name] = {
            "mode": "rank",
            "alpha": float(alpha),
        }

candidate_metrics = {
    name: evaluate(valid.user_id, valid.y, scores)
    for name, scores in candidate_scores.items()
}
best_name = max(
    candidate_metrics,
    key=lambda name: candidate_metrics[name]["primary"],
)
best_spec = candidate_specs[best_name]
valid_scores = candidate_scores[best_name]
metrics = candidate_metrics[best_name]

print(
    "CANDIDATES "
    + json.dumps(
        {
            name: float(result["primary"])
            for name, result in candidate_metrics.items()
        },
        sort_keys=True,
    )
)
print(
    "FINDINGS "
    + json.dumps(
        {
            "selected": best_name,
            "selected_mode": best_spec["mode"],
            "selected_din_weight": best_spec["alpha"],
            "din_primary": float(
                candidate_metrics["din_standalone"]["primary"]
            ),
            "selected_primary": float(metrics["primary"]),
        }
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

test = load("test")
X_test = make_matrix(test)

X_combined = np.ascontiguousarray(
    np.concatenate([X_train, X_valid], axis=0),
    dtype=np.int64,
)
y_combined = np.concatenate(
    [
        np.asarray(train.y, dtype=np.float32),
        np.asarray(valid.y, dtype=np.float32),
    ]
)

combined_user_id = np.concatenate(
    [
        np.asarray(train.user_id, dtype=np.int64),
        np.asarray(valid.user_id, dtype=np.int64),
    ]
)
combined_video_id = np.concatenate(
    [
        np.asarray(train.video_id, dtype=np.int64),
        np.asarray(valid.video_id, dtype=np.int64),
    ]
)
combined_time_ms = np.concatenate(
    [
        np.asarray(train.time_ms, dtype=np.int64),
        np.asarray(valid.time_ms, dtype=np.int64),
    ]
)

class CombinedSplit:
    pass


combined = CombinedSplit()
combined.user_id = combined_user_id
combined.video_id = combined_video_id
combined.time_ms = combined_time_ms
combined.X = {
    name: np.concatenate(
        [
            np.asarray(train.X[name], dtype=np.int64),
            np.asarray(valid.X[name], dtype=np.int64),
        ]
    )
    for name in HISTORY_FIELDS
}

del model
del train_history
del train_mask
del valid_history
del valid_mask
del train_store
gc.collect()

combined_history, combined_mask, combined_store = (
    make_self_histories(combined, y_combined)
)
test_history, test_mask = make_external_histories(
    test, combined_store
)

combined_model = fit_model(
    X_combined,
    combined_history,
    combined_mask,
    y_combined,
    SEED,
)
din_test = predict(
    combined_model,
    X_test,
    test_history,
    test_mask,
)

if (
    incumbent_test is not None
    and len(incumbent_test) == len(din_test)
):
    alpha = best_spec["alpha"]
    if best_spec["mode"] == "rank":
        test_scores = (
            (1.0 - alpha)
            * within_user_rank(test.user_id, incumbent_test)
            + alpha
            * within_user_rank(test.user_id, din_test)
        )
    else:
        test_scores = (
            (1.0 - alpha) * incumbent_test
            + alpha * din_test
        )
else:
    test_scores = din_test

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
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)