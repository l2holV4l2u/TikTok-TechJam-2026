import os
import time
import json
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
EMBED_DIM = 16
LR = 0.001
BATCH_SIZE = 4096
MAX_EPOCHS = 12
PATIENCE = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_matrix(split):
    cols = [
        np.asarray(split.X[field], dtype=np.int64) + offsets[j]
        for j, field in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


class FactorizationMachine(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, initial_bias):
        super().__init__()
        self.linear = nn.Embedding(num_embeddings, 1, sparse=True)
        self.latent = nn.Embedding(num_embeddings, embedding_dim, sparse=True)
        self.bias = nn.Parameter(torch.tensor(float(initial_bias), dtype=torch.float32))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).squeeze(-1).sum(dim=1)
        v = self.latent(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear_term + interaction


@torch.no_grad()
def predict(model, x_np, batch_size=32768):
    model.eval()
    result = np.empty(x_np.shape[0], dtype=np.float64)
    for begin in range(0, x_np.shape[0], batch_size):
        end = min(begin + batch_size, x_np.shape[0])
        xb = torch.from_numpy(x_np[begin:end])
        result[begin:end] = model(xb).cpu().numpy().astype(np.float64)
    return result


train = load("train")
valid = load("valid")

x_train_np = make_matrix(train)
x_valid_np = make_matrix(valid)
y_train_np = np.asarray(train.y, dtype=np.float32)
y_valid_np = np.asarray(valid.y, dtype=np.int8)

x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)

positive_rate = float(np.clip(y_train_np.mean(), 1e-6, 1.0 - 1e-6))
initial_bias = np.log(positive_rate / (1.0 - positive_rate))

model = FactorizationMachine(total_cardinality, EMBED_DIM, initial_bias)
sparse_optimizer = torch.optim.SparseAdam(
    [model.linear.weight, model.latent.weight], lr=LR
)
bias_optimizer = torch.optim.Adam([model.bias], lr=LR)

best_primary = -np.inf
best_state = None
best_valid_scores = None
best_metrics = None
stale_epochs = 0
n_train = x_train.shape[0]

for epoch in range(MAX_EPOCHS):
    model.train()
    permutation = torch.randperm(n_train)

    for begin in range(0, n_train, BATCH_SIZE):
        idx = permutation[begin:min(begin + BATCH_SIZE, n_train)]
        xb = x_train[idx]
        yb = y_train[idx]

        sparse_optimizer.zero_grad(set_to_none=True)
        bias_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()

        sparse_optimizer.step()
        bias_optimizer.step()

    valid_scores = predict(model, x_valid_np)
    metrics = evaluate(valid.user_id, y_valid_np, valid_scores)
    primary = float(metrics["primary"])

    if primary > best_primary + 1e-5:
        best_primary = primary
        best_state = copy.deepcopy(model.state_dict())
        best_valid_scores = valid_scores.copy()
        best_metrics = metrics
        stale_epochs = 0
    else:
        stale_epochs += 1

    if stale_epochs >= PATIENCE:
        break

model.load_state_dict(best_state)
valid_scores = best_valid_scores

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

test = load("test")
x_test_np = make_matrix(test)
test_scores = predict(model, x_test_np)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
report = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(report))