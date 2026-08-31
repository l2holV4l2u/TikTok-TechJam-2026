import os
import time
import json
import random
import numpy as np
import torch
from torch import nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
LR = 0.001
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 65536
EPOCHS = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def make_offsets(fields):
    offsets = []
    total = 0
    for name in fields:
        offsets.append(total)
        total += int(FEATURE_CARDINALITIES[name])
    return np.asarray(offsets, dtype=np.int64), total


OFFSETS, TOTAL_CARDINALITY = make_offsets(FIELDS)


def build_matrix(split):
    cols = []
    for j, name in enumerate(FIELDS):
        x = np.asarray(split.X[name], dtype=np.int64)
        cols.append(x + OFFSETS[j])
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


class FactorizationMachine(nn.Module):
    def __init__(self, cardinality, rank):
        super().__init__()
        self.linear = nn.Embedding(cardinality, 1)
        self.factor = nn.Embedding(cardinality, rank)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factor.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(1)
        v = self.factor(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear_term + interaction


def train_model(x_np, y_np):
    model = FactorizationMachine(TOTAL_CARDINALITY, K)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    n = x.shape[0]
    generator = torch.Generator()
    generator.manual_seed(SEED)

    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(x[idx])
            loss = criterion(logits, y[idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


@torch.inference_mode()
def predict(model, x_np):
    model.eval()
    x = torch.from_numpy(x_np)
    result = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, x.shape[0])
        result[start:end] = model(x[start:end]).cpu().numpy().astype(np.float64)
    return result


train = load("train")
x_train = build_matrix(train)
model = train_model(x_train, train.y)

valid = load("valid")
x_valid = build_matrix(valid)
valid_scores = predict(model, x_valid)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

test = load("test")
x_test = build_matrix(test)
test_scores = predict(model, x_test)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS " + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(",", ":"),
    )
)