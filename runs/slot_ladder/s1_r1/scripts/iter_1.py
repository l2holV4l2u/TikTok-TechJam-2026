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
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
LR = 0.001
BATCH_SIZE = 2048
EPOCHS = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))


def make_offsets():
    cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
    offsets = np.zeros(len(cards), dtype=np.int64)
    if len(cards) > 1:
        offsets[1:] = np.cumsum(cards[:-1], dtype=np.int64)
    return cards, offsets, int(sum(cards))


CARDS, OFFSETS, TOTAL_CARDINALITY = make_offsets()


def encode(split):
    x = np.empty((len(split.user_id), len(FIELDS)), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:, j] = split.X[field] + OFFSETS[j]
    return x


class FactorizationMachine(nn.Module):
    def __init__(self, cardinality, rank, base_rate):
        super().__init__()
        # Dimension zero is the first-order coefficient; remaining dimensions
        # are the latent FM factors.
        self.embedding = nn.Embedding(cardinality, rank + 1, sparse=True)
        with torch.no_grad():
            self.embedding.weight.zero_()
            nn.init.normal_(self.embedding.weight[:, 1:], mean=0.0, std=0.01)
        clipped = min(max(float(base_rate), 1e-6), 1.0 - 1e-6)
        self.register_buffer(
            "intercept",
            torch.tensor(np.log(clipped / (1.0 - clipped)), dtype=torch.float32),
        )

    def forward(self, x):
        e = self.embedding(x)
        linear = e[:, :, 0].sum(dim=1)
        v = e[:, :, 1:]
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1) - v.square().sum(dim=(1, 2))
        )
        return self.intercept + linear + interaction


def predict(model, x_np, batch_size=16384):
    model.eval()
    result = np.empty(x_np.shape[0], dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, x_np.shape[0], batch_size):
            end = min(start + batch_size, x_np.shape[0])
            xb = torch.from_numpy(x_np[start:end])
            logits = model(xb)
            result[start:end] = torch.sigmoid(logits).cpu().numpy()
    return result


train = load("train")
valid = load("valid")

x_train = encode(train)
x_valid = encode(valid)
y_train_np = np.asarray(train.y, dtype=np.float32)

model = FactorizationMachine(
    cardinality=TOTAL_CARDINALITY,
    rank=K,
    base_rate=float(y_train_np.mean()),
)
optimizer = torch.optim.SparseAdam(model.parameters(), lr=LR)

n_train = x_train.shape[0]
rng = np.random.default_rng(SEED)

model.train()
for epoch in range(EPOCHS):
    order = rng.permutation(n_train)
    loss_sum = 0.0

    for start in range(0, n_train, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        xb = torch.from_numpy(x_train[idx])
        yb = torch.from_numpy(y_train_np[idx])

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.detach()) * len(idx)

    print(
        "epoch=%d train_loss=%.6f"
        % (epoch + 1, loss_sum / n_train),
        flush=True,
    )

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
x_test = encode(test)
test_scores = predict(model, x_test)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
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