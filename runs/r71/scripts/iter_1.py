import os
import time
import json
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
MAX_EPOCHS = 8

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))


cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_matrix(split):
    cols = [
        np.asarray(split.X[name], dtype=np.int64) + offsets[j]
        for j, name in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


class FactorizationMachine(nn.Module):
    def __init__(self, n_categories, rank, initial_bias):
        super().__init__()
        # One sparse table contains both the linear coefficient and latent vector.
        self.embedding = nn.Embedding(n_categories, rank + 1, sparse=True)
        self.bias = nn.Parameter(torch.tensor(float(initial_bias), dtype=torch.float32))
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)

    def forward(self, x):
        e = self.embedding(x)
        linear = e[:, :, 0].sum(dim=1)
        v = e[:, :, 1:]
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


def fit_model(X, y, epochs, seed):
    torch.manual_seed(seed)
    n = len(y)
    positive_rate = float(np.mean(y))
    initial_bias = np.log(
        np.clip(positive_rate, 1e-6, 1.0 - 1e-6)
        / np.clip(1.0 - positive_rate, 1e-6, 1.0)
    )

    model = FactorizationMachine(total_cardinality, K, initial_bias)
    sparse_optimizer = torch.optim.SparseAdam(
        [model.embedding.weight], lr=LR
    )
    bias_optimizer = torch.optim.Adam([model.bias], lr=LR)

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(seed + 17)

    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(n, generator=generator)
        for begin in range(0, n, BATCH_SIZE):
            idx = permutation[begin:begin + BATCH_SIZE]
            xb = X_tensor[idx]
            yb = y_tensor[idx]

            sparse_optimizer.zero_grad(set_to_none=True)
            bias_optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            sparse_optimizer.step()
            bias_optimizer.step()

    return model


def predict(model, X):
    model.eval()
    X_tensor = torch.from_numpy(X)
    result = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for begin in range(0, len(X), 16384):
            end = min(begin + 16384, len(X))
            result[begin:end] = model(X_tensor[begin:end]).cpu().numpy()
    return result


# Train-only fitting and validation evaluation.
train = load("train")
valid = load("valid")
X_train = make_matrix(train)
X_valid = make_matrix(valid)
y_train = np.asarray(train.y, dtype=np.float32)

model = fit_model(X_train, y_train, MAX_EPOCHS, SEED)
valid_scores = predict(model, X_valid)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Refit the identical recipe on train + validation, then score test features only.
test = load("test")
X_test = make_matrix(test)
X_combined = np.ascontiguousarray(
    np.concatenate([X_train, X_valid], axis=0), dtype=np.int64
)
y_combined = np.concatenate(
    [
        np.asarray(train.y, dtype=np.float32),
        np.asarray(valid.y, dtype=np.float32),
    ]
)

del model
combined_model = fit_model(X_combined, y_combined, MAX_EPOCHS, SEED)
test_scores = predict(combined_model, X_test)

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