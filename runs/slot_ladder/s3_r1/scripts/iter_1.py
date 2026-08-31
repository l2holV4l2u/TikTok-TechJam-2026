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
SEED = 2024

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
]
LATENT_DIM = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 4096
EPOCHS = 6


def make_matrix(split, offsets):
    cols = []
    for field, offset in zip(FIELDS, offsets):
        col = np.asarray(split.X[field], dtype=np.int64)
        cols.append(col + offset)
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


class FactorizationMachine(nn.Module):
    def __init__(self, num_features, latent_dim):
        super().__init__()
        self.linear = nn.Embedding(num_features, 1)
        self.latent = nn.Embedding(num_features, latent_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.latent(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear_term + interaction


def predict(model, matrix, batch_size=16384):
    model.eval()
    result = np.empty(matrix.shape[0], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, matrix.shape[0], batch_size):
            end = min(start + batch_size, matrix.shape[0])
            xb = torch.from_numpy(matrix[start:end])
            result[start:end] = model(xb).cpu().numpy()
    return result


train = load("train")

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
num_features = int(sum(cardinalities))

x_train = make_matrix(train, offsets)
y_train = np.asarray(train.y, dtype=np.float32)

model = FactorizationMachine(num_features, LATENT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

generator = torch.Generator()
generator.manual_seed(SEED)
n_train = x_train.shape[0]

model.train()
for epoch in range(EPOCHS):
    permutation = torch.randperm(n_train, generator=generator)
    for start in range(0, n_train, BATCH_SIZE):
        idx = permutation[start:start + BATCH_SIZE].numpy()
        xb = torch.from_numpy(x_train[idx])
        yb = torch.from_numpy(y_train[idx])

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        optimizer.step()

del x_train, y_train, train

valid = load("valid")
x_valid = make_matrix(valid, offsets)
valid_scores = predict(model, x_valid)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

del x_valid

test = load("test")
x_test = make_matrix(test, offsets)
test_scores = predict(model, x_test)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
result = {
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result))