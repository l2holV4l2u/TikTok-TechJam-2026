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
torch.set_num_threads(min(16, os.cpu_count() or 1))

FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
EMBED_DIM = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 4096
EPOCHS = 5


def make_matrix(split):
    offsets = np.cumsum(
        np.asarray([0] + [FEATURE_CARDINALITIES[f] for f in FIELDS[:-1]],
                   dtype=np.int64)
    )
    columns = [
        np.asarray(split.X[field], dtype=np.int64) + offsets[j]
        for j, field in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.int64)


class FactorizationMachine(nn.Module):
    def __init__(self, num_features, embedding_dim, initial_bias):
        super().__init__()
        self.linear = nn.Embedding(num_features, 1)
        self.embedding = nn.Embedding(num_features, embedding_dim)
        self.bias = nn.Parameter(torch.tensor(float(initial_bias),
                                              dtype=torch.float32))

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


def predict(model, matrix, batch_size=16384):
    model.eval()
    result = np.empty(len(matrix), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(matrix), batch_size):
            end = min(start + batch_size, len(matrix))
            xb = torch.from_numpy(matrix[start:end])
            result[start:end] = model(xb).cpu().numpy()
    return result


train = load("train")
valid = load("valid")

x_train_np = make_matrix(train)
x_valid_np = make_matrix(valid)
y_train_np = np.asarray(train.y, dtype=np.float32)

x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)

num_features = int(sum(FEATURE_CARDINALITIES[f] for f in FIELDS))
positive_rate = float(y_train_np.mean())
initial_bias = np.log(
    np.clip(positive_rate, 1e-6, 1.0 - 1e-6)
    / np.clip(1.0 - positive_rate, 1e-6, 1.0)
)

model = FactorizationMachine(num_features, EMBED_DIM, initial_bias)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

n_train = len(y_train)
generator = torch.Generator()
generator.manual_seed(SEED)

model.train()
for epoch in range(EPOCHS):
    permutation = torch.randperm(n_train, generator=generator)
    for start in range(0, n_train, BATCH_SIZE):
        indices = permutation[start:start + BATCH_SIZE]
        xb = x_train[indices]
        yb = y_train[indices]

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

valid_scores = predict(model, x_valid_np)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

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

elapsed = time.time() - START_TIME
print(
    "METRICS " + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)