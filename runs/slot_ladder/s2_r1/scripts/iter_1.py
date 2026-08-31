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
EPOCHS = 5
BATCH_SIZE = 2048
PRED_BATCH_SIZE = 32768

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def make_offsets():
    offsets = []
    running = 0
    for name in FIELDS:
        offsets.append(running)
        running += int(FEATURE_CARDINALITIES[name])
    return np.asarray(offsets, dtype=np.int64), running


OFFSETS, TOTAL_CARDINALITY = make_offsets()


def encode(split):
    x = np.empty((len(split.user_id), len(FIELDS)), dtype=np.int64)
    for j, name in enumerate(FIELDS):
        values = np.asarray(split.X[name], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[name])
        if values.size and (values.min() < 0 or values.max() >= card):
            raise ValueError("Out-of-range categorical value in %s" % name)
        x[:, j] = values + OFFSETS[j]
    return x


class FactorizationMachine(nn.Module):
    def __init__(self, cardinality, rank):
        super().__init__()
        self.linear = nn.Embedding(cardinality, 1, sparse=True)
        self.factors = nn.Embedding(cardinality, rank, sparse=True)
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


train = load("train")
valid = load("valid")

x_train = torch.from_numpy(encode(train))
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))
x_valid_np = encode(valid)

model = FactorizationMachine(TOTAL_CARDINALITY, K)
embedding_optimizer = torch.optim.SparseAdam(
    [model.linear.weight, model.factors.weight], lr=LR
)
bias_optimizer = torch.optim.Adam([model.bias], lr=LR)
criterion = nn.BCEWithLogitsLoss()

model.train()
n_train = x_train.shape[0]
generator = torch.Generator()
generator.manual_seed(SEED)

for epoch in range(EPOCHS):
    order = torch.randperm(n_train, generator=generator)
    for start in range(0, n_train, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        xb = x_train[idx]
        yb = y_train[idx]

        embedding_optimizer.zero_grad(set_to_none=True)
        bias_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()

        embedding_optimizer.step()
        bias_optimizer.step()


def predict(encoded):
    model.eval()
    result = np.empty(encoded.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, encoded.shape[0], PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, encoded.shape[0])
            xb = torch.from_numpy(encoded[start:end])
            result[start:end] = model(xb).cpu().numpy().astype(np.float64)
    return result


valid_scores = predict(x_valid_np)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

test = load("test")
x_test_np = encode(test)
test_scores = predict(x_test_np)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))