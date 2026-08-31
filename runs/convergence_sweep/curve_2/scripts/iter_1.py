import os
import time
import json
import copy
import random

import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
EMBED_DIM = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 2048
EPOCHS = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))


def make_matrix(split):
    cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
    offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
    cols = [
        np.asarray(split.X[f], dtype=np.int64) + offsets[j]
        for j, f in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.int64), cardinalities


class FactorizationMachine(nn.Module):
    def __init__(self, total_cardinality, embedding_dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, embedding_dim)

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)
        latent = self.embedding(x)
        summed = latent.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - latent.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear_term + interaction


@torch.no_grad()
def predict(model, x_np, batch_size=16384):
    model.eval()
    n = x_np.shape[0]
    scores = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = torch.from_numpy(x_np[start:end])
        scores[start:end] = model(xb).cpu().numpy()
    return scores


train = load("train")
valid = load("valid")

x_train_np, cardinalities = make_matrix(train)
x_valid_np, _ = make_matrix(valid)
y_train_np = np.asarray(train.y, dtype=np.float32)

x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)

model = FactorizationMachine(sum(cardinalities), EMBED_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

generator = torch.Generator()
generator.manual_seed(SEED)

best_primary = -np.inf
best_state = None
best_valid_scores = None
best_metrics = None

n_train = x_train.shape[0]

for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(n_train, generator=generator)

    for start in range(0, n_train, BATCH_SIZE):
        idx = permutation[start:start + BATCH_SIZE]
        xb = x_train[idx]
        yb = y_train[idx]

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    valid_scores = predict(model, x_valid_np)
    metrics = evaluate(valid.user_id, valid.y, valid_scores)

    if metrics["primary"] > best_primary:
        best_primary = float(metrics["primary"])
        best_metrics = metrics
        best_valid_scores = valid_scores.copy()
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

model.load_state_dict(best_state)
model.eval()

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )

test = load("test")
x_test_np, _ = make_matrix(test)
test_scores = predict(model, x_test_np)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START_TIME
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result))