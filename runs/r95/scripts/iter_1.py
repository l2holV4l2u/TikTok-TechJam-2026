import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
LR = 0.001
BATCH_SIZE = 4096
EPOCHS = 6

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def make_offsets():
    offsets = []
    total = 0
    for name in FIELDS:
        offsets.append(total)
        total += int(FEATURE_CARDINALITIES[name])
    return np.asarray(offsets, dtype=np.int64), total


OFFSETS, TOTAL_CARDINALITY = make_offsets()


def build_matrix(split):
    cols = []
    for j, name in enumerate(FIELDS):
        x = np.asarray(split.X[name], dtype=np.int64)
        cols.append(x + OFFSETS[j])
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


class FactorizationMachine(nn.Module):
    def __init__(self, n_features, rank):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1, sparse=True)
        self.factors = nn.Embedding(n_features, rank, sparse=True)
        self.bias = nn.Parameter(torch.zeros(1))

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


def predict(model, x_np, batch_size=32768):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(x_np), batch_size):
            end = min(start + batch_size, len(x_np))
            xb = torch.from_numpy(x_np[start:end])
            result[start:end] = model(xb).cpu().numpy()
    return result


train = load("train")
valid = load("valid")

x_train_np = build_matrix(train)
x_valid_np = build_matrix(valid)
y_train_np = np.asarray(train.y, dtype=np.float32)

x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)

model = FactorizationMachine(TOTAL_CARDINALITY, K)

sparse_optimizer = torch.optim.SparseAdam(
    [model.linear.weight, model.factors.weight], lr=LR
)
bias_optimizer = torch.optim.Adam([model.bias], lr=LR)
criterion = nn.BCEWithLogitsLoss()

n = len(y_train)
generator = torch.Generator()
generator.manual_seed(SEED)

model.train()
for epoch in range(EPOCHS):
    permutation = torch.randperm(n, generator=generator)
    epoch_loss = 0.0
    seen = 0

    for start in range(0, n, BATCH_SIZE):
        idx = permutation[start:min(start + BATCH_SIZE, n)]
        xb = x_train.index_select(0, idx)
        yb = y_train.index_select(0, idx)

        sparse_optimizer.zero_grad(set_to_none=True)
        bias_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()

        sparse_optimizer.step()
        bias_optimizer.step()

        batch_n = len(idx)
        epoch_loss += float(loss.detach()) * batch_n
        seen += batch_n

    print("epoch=%d loss=%.6f" % (epoch + 1, epoch_loss / seen), flush=True)

valid_scores = predict(model, x_valid_np)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

test = load("test")
x_test_np = build_matrix(test)
test_scores = predict(model, x_test_np)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
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