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

SEED = 2024
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1]).astype(np.int64)
TOTAL_CARDINALITY = int(sum(CARDS))
K = 16
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 131072
EPOCHS = 8
LR = 0.001


class FactorizationMachine(nn.Module):
    def __init__(self, cardinality, rank, initial_bias):
        super().__init__()
        # Coordinate zero is the first-order weight; the rest are FM factors.
        self.embedding = nn.Embedding(
            cardinality, rank + 1, sparse=True
        )
        self.register_buffer(
            "intercept", torch.tensor(float(initial_bias), dtype=torch.float32)
        )
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)

    def forward(self, indices):
        e = self.embedding(indices)
        linear = e[:, :, 0].sum(dim=1)
        v = e[:, :, 1:]
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.intercept + linear + interaction


def split_to_indices(split, start, end):
    cols = []
    for field, offset in zip(FIELDS, OFFSETS):
        x = np.asarray(split.X[field][start:end], dtype=np.int64)
        cols.append(torch.from_numpy(x) + int(offset))
    return torch.stack(cols, dim=1)


@torch.inference_mode()
def predict_split(model, split):
    model.eval()
    n = int(split.user_id.shape[0])
    result = np.empty(n, dtype=np.float32)
    for start in range(0, n, PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, n)
        xb = split_to_indices(split, start, end)
        result[start:end] = model(xb).cpu().numpy()
    return result


# Copy only the five official baseline fields, allowing the full split object
# (which contains all 37 categorical columns) to be released before training.
train = load("train")
n_train = int(train.user_id.shape[0])
train_x = []
for field, offset in zip(FIELDS, OFFSETS):
    col = np.asarray(train.X[field], dtype=np.int64).copy()
    col += int(offset)
    train_x.append(torch.from_numpy(col))
train_y = torch.from_numpy(np.asarray(train.y, dtype=np.float32).copy())

positive_rate = float(train_y.mean().item())
initial_bias = np.log(positive_rate / (1.0 - positive_rate))
del train
gc.collect()

model = FactorizationMachine(TOTAL_CARDINALITY, K, initial_bias)
optimizer = torch.optim.SparseAdam(model.parameters(), lr=LR)

model.train()
generator = torch.Generator()
generator.manual_seed(SEED)

for epoch in range(EPOCHS):
    permutation = torch.randperm(n_train, generator=generator)
    loss_sum = 0.0
    rows_seen = 0

    for start in range(0, n_train, BATCH_SIZE):
        ids = permutation[start:start + BATCH_SIZE]
        xb = torch.stack([column[ids] for column in train_x], dim=1)
        yb = train_y[ids]

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        optimizer.step()

        batch_n = int(ids.numel())
        loss_sum += float(loss.detach()) * batch_n
        rows_seen += batch_n

    print(
        "epoch=%d train_logloss=%.6f"
        % (epoch + 1, loss_sum / max(rows_seen, 1)),
        flush=True,
    )

del train_x, train_y
gc.collect()

valid = load("valid")
valid_scores = predict_split(model, valid)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

del valid
gc.collect()

test = load("test")
test_scores = predict_split(model, test)
if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": elapsed,
        }
    )
)