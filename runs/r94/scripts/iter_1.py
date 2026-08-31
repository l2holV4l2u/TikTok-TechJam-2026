import os
import time
import json
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 2024

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
LR = 0.001
BATCH_SIZE = 2048
EPOCHS = 5


def make_offsets():
    offsets = []
    running = 0
    for name in FIELDS:
        offsets.append(running)
        running += int(FEATURE_CARDINALITIES[name])
    return np.asarray(offsets, dtype=np.int64), running


OFFSETS, TOTAL_CARDINALITY = make_offsets()


def build_features(split):
    cols = []
    for j, name in enumerate(FIELDS):
        x = np.asarray(split.X[name], dtype=np.int64)
        cols.append(x + OFFSETS[j])
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


class FactorizationMachine(nn.Module):
    def __init__(self, cardinality, rank):
        super().__init__()
        # Column zero is the first-order coefficient; remaining columns are
        # the latent factors. Sparse gradients make CPU training efficient.
        self.embedding = nn.Embedding(cardinality, rank + 1, sparse=True)
        self.bias = nn.Parameter(torch.zeros(1))

        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)

    def forward(self, x):
        e = self.embedding(x)
        linear = e[:, :, 0].sum(dim=1)
        latent = e[:, :, 1:]

        summed = latent.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1)
            - latent.square().sum(dim=(1, 2))
        )
        return self.bias + linear + interaction


def predict(model, x_np, batch_size=32768):
    model.eval()
    n = x_np.shape[0]
    scores = np.empty(n, dtype=np.float64)

    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.from_numpy(x_np[start:end])
            scores[start:end] = model(xb).cpu().numpy().astype(np.float64)
    return scores


train = load("train")
valid = load("valid")

x_train_np = build_features(train)
x_valid_np = build_features(valid)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))

# Keep the compact categorical matrix in CPU tensor storage.
x_train = torch.from_numpy(x_train_np)
del x_train_np

model = FactorizationMachine(TOTAL_CARDINALITY, K)

# SparseAdam updates only entity rows occurring in each minibatch.
embedding_optimizer = torch.optim.SparseAdam(
    [model.embedding.weight], lr=LR
)
bias_optimizer = torch.optim.Adam([model.bias], lr=LR)
criterion = nn.BCEWithLogitsLoss()

n_train = x_train.shape[0]
generator = torch.Generator()
generator.manual_seed(SEED)

model.train()
for epoch in range(EPOCHS):
    permutation = torch.randperm(n_train, generator=generator)

    for start in range(0, n_train, BATCH_SIZE):
        idx = permutation[start:start + BATCH_SIZE]
        xb = x_train[idx]
        yb = y_train[idx]

        embedding_optimizer.zero_grad(set_to_none=True)
        bias_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()

        embedding_optimizer.step()
        bias_optimizer.step()

valid_scores = predict(model, x_valid_np)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Test labels are never accessed; only feature-based scores are produced.
test = load("test")
x_test_np = build_features(test)
test_scores = predict(model, x_test_np)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START_TIME
result = {
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result, separators=(", ", ": ")))