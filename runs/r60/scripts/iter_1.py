import os
import sys
import time
import copy
import json
import numpy as np
import torch
import torch.nn as nn

_start_time = time.time()

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
EMBED_DIM = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 2048
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
OFFSETS_T = torch.from_numpy(OFFSETS)


def encode_split(split):
    x = np.column_stack([split.X[name] for name in FIELDS]).astype(
        np.int64, copy=False
    )
    x = x + OFFSETS[None, :]
    return torch.from_numpy(np.ascontiguousarray(x))


class FactorizationMachine(nn.Module):
    def __init__(self, num_features, rank, initial_bias):
        super().__init__()
        self.linear = nn.Embedding(num_features, 1, sparse=True)
        self.factors = nn.Embedding(num_features, rank, sparse=True)
        self.bias = nn.Parameter(torch.tensor(float(initial_bias), dtype=torch.float32))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)

        v = self.factors(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)

        return self.bias + linear_term + interaction


@torch.no_grad()
def predict(model, x, batch_size=32768):
    model.eval()
    result = np.empty(len(x), dtype=np.float64)
    for begin in range(0, len(x), batch_size):
        end = min(begin + batch_size, len(x))
        result[begin:end] = model(x[begin:end]).cpu().numpy().astype(
            np.float64, copy=False
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

model = FactorizationMachine(
    num_features=TOTAL_CARDINALITY,
    rank=EMBED_DIM,
    initial_bias=initial_bias,
)

sparse_optimizer = torch.optim.SparseAdam(
    [model.linear.weight, model.factors.weight],
    lr=LEARNING_RATE,
)
bias_optimizer = torch.optim.Adam([model.bias], lr=LEARNING_RATE)
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
        bias_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()

        sparse_optimizer.step()
        bias_optimizer.step()
        loss_sum += float(loss.detach()) * len(idx)

    valid_scores = predict(model, x_valid)
    metrics = evaluate(valid.user_id, valid.y, valid_scores)

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
        best_valid_scores = valid_scores.copy()
        best_metrics = metrics

assert best_state is not None
model.load_state_dict(best_state)
valid_scores = best_valid_scores

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Test labels are never accessed; only features are transformed and scored.
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