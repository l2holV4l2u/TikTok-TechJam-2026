import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 2024

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat1",
    "onehot_feat7",
    "music_type",
    "user_active_degree",
    "hour",
    "register_days_bucket",
]

CROSS_PAIRS = [
    ("user_id", "author_id"),
    ("user_id", "tag"),
    ("user_id", "duration_bucket"),
    ("user_id", "tab"),
]

HASH_SIZE = 1 << 20
EMBED_DIM = 16
HIDDEN_DIMS = (128, 64)
DROPOUT = 0.10
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 16384
MAX_EPOCHS = 12
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-6

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

num_threads = min(8, os.cpu_count() or 1)
torch.set_num_threads(num_threads)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

device = torch.device("cpu")

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.zeros(len(FIELDS), dtype=np.int64)
if len(FIELDS) > 1:
    offsets[1:] = np.cumsum(cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_features(split):
    columns = [
        np.asarray(split.X[field], dtype=np.int64) + offset
        for field, offset in zip(FIELDS, offsets)
    ]
    categorical = np.ascontiguousarray(
        np.column_stack(columns), dtype=np.int64
    )

    cross_columns = []
    mask = np.int64(HASH_SIZE - 1)
    for cross_number, (left_name, right_name) in enumerate(CROSS_PAIRS):
        left = np.asarray(split.X[left_name], dtype=np.int64)
        right = np.asarray(split.X[right_name], dtype=np.int64)

        # Power-of-two hashing makes signed int64 overflow harmless and fast.
        salt = np.int64((cross_number + 1) * 0x1F123BB5)
        hashed = (
            (left * np.int64(1000003))
            ^ (right * np.int64(9176))
            ^ salt
        ) & mask
        cross_columns.append(hashed)

    crosses = np.ascontiguousarray(
        np.column_stack(cross_columns), dtype=np.int64
    )
    return categorical, crosses


class WideDeepFM(nn.Module):
    def __init__(
        self,
        num_categories,
        num_fields,
        embedding_dim,
        hidden_dims,
        dropout,
        hash_size,
    ):
        super().__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim

        self.linear = nn.Embedding(num_categories, 1)
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        layers = []
        input_dim = num_fields * embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.deep = nn.Sequential(*layers)

        # Sparse scalar lookup for explicit hashed crosses.
        self.wide_cross = nn.Embedding(hash_size, 1, sparse=True)

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.wide_cross.weight)

        linear_layers = [
            module
            for module in self.deep.modules()
            if isinstance(module, nn.Linear)
        ]
        for layer in linear_layers[:-1]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(linear_layers[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(linear_layers[-1].bias)

    def forward(self, x, cross_x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)

        latent = self.embedding(x)
        summed = latent.sum(dim=1)
        fm_term = 0.5 * (
            summed.square() - latent.square().sum(dim=1)
        ).sum(dim=1)

        deep_input = latent.reshape(latent.shape[0], -1)
        deep_term = self.deep(deep_input).squeeze(-1)

        wide_term = self.wide_cross(cross_x).sum(dim=1).squeeze(-1)

        return self.bias + linear_term + fm_term + deep_term + wide_term


def predict(model, x_np, cross_np, batch_size=PRED_BATCH_SIZE):
    model.eval()
    predictions = np.empty(x_np.shape[0], dtype=np.float64)

    with torch.inference_mode():
        for start in range(0, x_np.shape[0], batch_size):
            end = min(start + batch_size, x_np.shape[0])
            xb = torch.from_numpy(x_np[start:end]).to(device)
            cb = torch.from_numpy(cross_np[start:end]).to(device)
            logits = model(xb, cb)
            predictions[start:end] = logits.cpu().numpy().astype(
                np.float64, copy=False
            )

    return predictions


train = load("train")
valid = load("valid")

x_train, cross_train = make_features(train)
y_train = np.asarray(train.y, dtype=np.float32)

x_valid, cross_valid = make_features(valid)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

model = WideDeepFM(
    num_categories=total_cardinality,
    num_fields=len(FIELDS),
    embedding_dim=EMBED_DIM,
    hidden_dims=HIDDEN_DIMS,
    dropout=DROPOUT,
    hash_size=HASH_SIZE,
).to(device)

dense_parameters = [
    parameter
    for name, parameter in model.named_parameters()
    if name != "wide_cross.weight"
]

dense_optimizer = torch.optim.Adam(
    dense_parameters,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
cross_optimizer = torch.optim.SparseAdam(
    [model.wide_cross.weight],
    lr=LEARNING_RATE,
)
criterion = nn.BCEWithLogitsLoss()

rng = np.random.default_rng(SEED)
n_train = x_train.shape[0]

best_primary = -np.inf
best_raw_metrics = None
best_valid_scores = None
best_state = None
epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    permutation = rng.permutation(n_train)
    epoch_loss_sum = 0.0

    for start in range(0, n_train, BATCH_SIZE):
        idx = permutation[start:start + BATCH_SIZE]

        xb = torch.from_numpy(x_train[idx]).to(device)
        cb = torch.from_numpy(cross_train[idx]).to(device)
        yb = torch.from_numpy(y_train[idx]).to(device)

        dense_optimizer.zero_grad(set_to_none=True)
        cross_optimizer.zero_grad(set_to_none=True)

        logits = model(xb, cb)
        loss = criterion(logits, yb)
        loss.backward()

        dense_optimizer.step()
        cross_optimizer.step()

        epoch_loss_sum += float(loss.detach()) * len(idx)

    valid_scores = predict(model, x_valid, cross_valid)
    metrics = evaluate(valid_users, y_valid, valid_scores)
    primary = float(metrics["primary"])

    print(
        "epoch=%d loss=%.6f primary=%.6f gauc=%.6f ndcg5=%.6f"
        % (
            epoch,
            epoch_loss_sum / n_train,
            primary,
            float(metrics["gauc"]),
            float(metrics["ndcg@5"]),
        ),
        flush=True,
    )

    if primary > best_primary:
        best_primary = primary
        best_raw_metrics = {
            "primary": primary,
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
        }
        best_valid_scores = valid_scores.copy()
        best_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch >= 6 and epochs_without_improvement >= 3:
        break

model.load_state_dict(best_state)
model.eval()

artifacts_dir = os.environ.get("RUN_ARTIFACTS")
incumbent_valid = None
incumbent_test = None

if artifacts_dir:
    incumbent_valid_path = os.path.join(
        artifacts_dir, "incumbent_valid_scores.npy"
    )
    incumbent_test_path = os.path.join(
        artifacts_dir, "incumbent_test_scores.npy"
    )

    if (
        os.path.exists(incumbent_valid_path)
        and os.path.exists(incumbent_test_path)
    ):
        loaded_valid = np.asarray(
            np.load(incumbent_valid_path), dtype=np.float64
        )
        if loaded_valid.shape == best_valid_scores.shape:
            incumbent_valid = loaded_valid
            incumbent_test = np.asarray(
                np.load(incumbent_test_path), dtype=np.float64
            )

candidate_results = {
    "wide_deepfm_raw": float(best_raw_metrics["primary"])
}

selected_alpha = 1.0
selected_valid_scores = best_valid_scores
selected_metrics = best_raw_metrics

if incumbent_valid is not None:
    for alpha in (0.25, 0.50, 0.75):
        blended_scores = (
            alpha * best_valid_scores
            + (1.0 - alpha) * incumbent_valid
        )
        blended_metrics = evaluate(
            valid_users, y_valid, blended_scores
        )
        candidate_results[
            "candidate_alpha_%.2f" % alpha
        ] = float(blended_metrics["primary"])

        if float(blended_metrics["primary"]) > float(
            selected_metrics["primary"]
        ):
            selected_alpha = alpha
            selected_valid_scores = blended_scores.copy()
            selected_metrics = {
                "primary": float(blended_metrics["primary"]),
                "gauc": float(blended_metrics["gauc"]),
                "ndcg@5": float(blended_metrics["ndcg@5"]),
            }

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(selected_valid_scores, dtype=np.float64),
    )

test = load("test")
x_test, cross_test = make_features(test)
raw_test_scores = predict(model, x_test, cross_test)

if (
    incumbent_test is not None
    and incumbent_test.shape == raw_test_scores.shape
    and selected_alpha < 1.0
):
    test_scores = (
        selected_alpha * raw_test_scores
        + (1.0 - selected_alpha) * incumbent_test
    )
else:
    test_scores = raw_test_scores

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

candidate_results["selected_alpha"] = float(selected_alpha)
print(
    "CANDIDATES "
    + json.dumps(candidate_results, separators=(",", ":")),
    flush=True,
)

elapsed = float(time.time() - START_TIME)
final_result = {
    "primary": float(selected_metrics["primary"]),
    "gauc": float(selected_metrics["gauc"]),
    "ndcg@5": float(selected_metrics["ndcg@5"]),
    "gpu_seconds": elapsed,
}
print("METRICS " + json.dumps(final_result, separators=(",", ":")))