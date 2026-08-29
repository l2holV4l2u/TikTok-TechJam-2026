import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn

_start_time = time.time()

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat1",
    "onehot_feat7",
    "user_active_degree",
    "register_days_bucket",
    "register_days_range",
    "fans_user_num_range",
    "follow_user_num_range",
    "hour",
]
EMBED_DIM = 16
DEEP_HIDDEN = (128, 64)
DROPOUT = 0.10
LEARNING_RATE = 0.001
BATCH_SIZE = 4096
MAX_EPOCHS = 12

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, max(1, os.cpu_count() or 1)))


def make_offsets():
    offsets = []
    running = 0
    for name in FIELDS:
        offsets.append(running)
        running += int(FEATURE_CARDINALITIES[name])
    return np.asarray(offsets, dtype=np.int64), running


OFFSETS, TOTAL_CARDINALITY = make_offsets()


def encode_split(split):
    x = np.column_stack([split.X[name] for name in FIELDS]).astype(
        np.int64, copy=False
    )
    x = x + OFFSETS[None, :]
    return torch.from_numpy(np.ascontiguousarray(x))


class DeepFM(nn.Module):
    def __init__(self, num_features, num_fields, rank, initial_bias):
        super().__init__()

        self.linear = nn.Embedding(num_features, 1, sparse=True)
        self.factors = nn.Embedding(num_features, rank, sparse=True)
        self.bias = nn.Parameter(
            torch.tensor(float(initial_bias), dtype=torch.float32)
        )

        layers = []
        input_dim = num_fields * rank
        for hidden_dim in DEEP_HIDDEN:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(DROPOUT))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.deep = nn.Sequential(*layers)

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)

        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

        final_layer = self.deep[-1]
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(final_layer.bias)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)

        embeddings = self.factors(x)
        summed = embeddings.sum(dim=1)
        fm_interaction = 0.5 * (
            summed.square() - embeddings.square().sum(dim=1)
        ).sum(dim=1)

        deep_input = embeddings.reshape(embeddings.shape[0], -1)
        deep_term = self.deep(deep_input).squeeze(-1)

        return self.bias + linear_term + fm_interaction + deep_term


@torch.no_grad()
def predict(model, x, batch_size=32768):
    model.eval()
    result = np.empty(len(x), dtype=np.float64)

    for begin in range(0, len(x), batch_size):
        end = min(begin + batch_size, len(x))
        result[begin:end] = (
            model(x[begin:end])
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )

    return result


train = load("train")
valid = load("valid")

x_train = encode_split(train)
y_train = torch.from_numpy(
    np.ascontiguousarray(train.y.astype(np.float32, copy=False))
)
x_valid = encode_split(valid)

positive_rate = float(np.mean(train.y))
initial_bias = np.log(
    np.clip(positive_rate, 1e-6, 1.0 - 1e-6)
    / np.clip(1.0 - positive_rate, 1e-6, 1.0)
)

model = DeepFM(
    num_features=TOTAL_CARDINALITY,
    num_fields=len(FIELDS),
    rank=EMBED_DIM,
    initial_bias=initial_bias,
)

sparse_optimizer = torch.optim.SparseAdam(
    [model.linear.weight, model.factors.weight],
    lr=LEARNING_RATE,
)
dense_parameters = [
    model.bias,
    *list(model.deep.parameters()),
]
dense_optimizer = torch.optim.Adam(
    dense_parameters,
    lr=LEARNING_RATE,
)
criterion = nn.BCEWithLogitsLoss()

best_primary = -np.inf
best_epoch = -1
best_state = None
best_valid_scores = None
best_metrics = None

n_train = len(x_train)
generator = torch.Generator()
generator.manual_seed(SEED)

for epoch in range(MAX_EPOCHS):
    model.train()
    permutation = torch.randperm(n_train, generator=generator)
    loss_sum = 0.0

    for begin in range(0, n_train, BATCH_SIZE):
        idx = permutation[begin:begin + BATCH_SIZE]
        xb = x_train[idx]
        yb = y_train[idx]

        sparse_optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()

        sparse_optimizer.step()
        dense_optimizer.step()

        loss_sum += float(loss.detach()) * len(idx)

    valid_scores_epoch = predict(model, x_valid)
    metrics = evaluate(valid.user_id, valid.y, valid_scores_epoch)

    print(
        "epoch=%d loss=%.6f primary=%.6f gauc=%.6f ndcg5=%.6f"
        % (
            epoch + 1,
            loss_sum / n_train,
            float(metrics["primary"]),
            float(metrics["gauc"]),
            float(metrics["ndcg@5"]),
        ),
        file=sys.stderr,
        flush=True,
    )

    if float(metrics["primary"]) > best_primary:
        best_primary = float(metrics["primary"])
        best_epoch = epoch + 1
        best_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        best_valid_scores = valid_scores_epoch.copy()
        best_metrics = metrics

assert best_state is not None
assert best_valid_scores is not None
assert best_metrics is not None

model.load_state_dict(best_state)
valid_scores = best_valid_scores

print(
    "FINDINGS selected_epoch=%d fields=%d"
    % (best_epoch, len(FIELDS))
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Test labels are never accessed. Only test-side input features are encoded.
test = load("test")
x_test = encode_split(test)
test_scores = predict(model, x_test)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - _start_time
final_result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(final_result, separators=(", ", ": ")))