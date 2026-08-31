import os
import time
import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 2024

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1], dtype=np.int64)
TOTAL_CARDINALITY = int(sum(CARDS))


def make_matrix(split):
    x = np.stack(
        [np.asarray(split.X[f], dtype=np.int64) + OFFSETS[j]
         for j, f in enumerate(FIELDS)],
        axis=1,
    )
    return np.ascontiguousarray(x)


class FactorizationMachine(nn.Module):
    def __init__(self, cardinality, embedding_dim, padding_rows):
        super().__init__()
        self.linear = nn.Embedding(cardinality, 1)
        self.embedding = nn.Embedding(cardinality, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer(
            "padding_rows",
            torch.as_tensor(padding_rows, dtype=torch.long),
        )

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        self.zero_padding_rows()

    @torch.no_grad()
    def zero_padding_rows(self):
        self.linear.weight.index_fill_(0, self.padding_rows, 0.0)
        self.embedding.weight.index_fill_(0, self.padding_rows, 0.0)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear_term + interaction


@torch.no_grad()
def predict(model, x_np, batch_size=32768):
    model.eval()
    result = np.empty(x_np.shape[0], dtype=np.float32)
    for start in range(0, x_np.shape[0], batch_size):
        end = min(start + batch_size, x_np.shape[0])
        xb = torch.from_numpy(x_np[start:end])
        result[start:end] = model(xb).cpu().numpy()
    return result


train = load("train")
valid = load("valid")

x_train_np = make_matrix(train)
x_valid_np = make_matrix(valid)
y_train_np = np.asarray(train.y, dtype=np.float32)
y_valid_np = np.asarray(valid.y, dtype=np.int8)

x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)

model = FactorizationMachine(
    cardinality=TOTAL_CARDINALITY,
    embedding_dim=16,
    padding_rows=OFFSETS,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
batch_size = 2048
max_epochs = 12

generator = torch.Generator()
generator.manual_seed(SEED)

best_primary = -np.inf
best_epoch = 0
best_state = None
best_scores = None
best_metrics = None
stale_epochs = 0

n_train = x_train.shape[0]

for epoch in range(1, max_epochs + 1):
    model.train()
    permutation = torch.randperm(n_train, generator=generator)
    epoch_loss = 0.0

    for start in range(0, n_train, batch_size):
        idx = permutation[start:start + batch_size]
        xb = x_train[idx]
        yb = y_train[idx]

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        optimizer.step()
        model.zero_padding_rows()

        epoch_loss += float(loss.detach()) * idx.numel()

    valid_scores = predict(model, x_valid_np)
    metrics = evaluate(valid.user_id, y_valid_np, valid_scores)
    primary = float(metrics["primary"])

    print(
        "epoch=%d loss=%.6f primary=%.6f gauc=%.6f ndcg5=%.6f"
        % (
            epoch,
            epoch_loss / n_train,
            primary,
            float(metrics["gauc"]),
            float(metrics["ndcg@5"]),
        )
    )

    if primary > best_primary:
        improvement = primary - best_primary
        best_primary = primary
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        best_scores = valid_scores.astype(np.float64, copy=True)
        best_metrics = metrics
        stale_epochs = 0 if improvement > 0.0002 else stale_epochs + 1
    else:
        stale_epochs += 1

    if epoch >= 5 and stale_epochs >= 3:
        break

model.load_state_dict(best_state)
model.eval()

# Recompute from the pinned best checkpoint to ensure saved scores and test
# predictions use exactly the same model state.
valid_scores = predict(model, x_valid_np).astype(np.float64)
metrics = evaluate(valid.user_id, y_valid_np, valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "scores_valid.npy"), valid_scores)

test = load("test")
x_test_np = make_matrix(test)
test_scores = predict(model, x_test_np).astype(np.float64)

if out_dir:
    np.save(os.path.join(out_dir, "scores_test.npy"), test_scores)

elapsed = time.time() - START_TIME
print("best_epoch=%d" % best_epoch)
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