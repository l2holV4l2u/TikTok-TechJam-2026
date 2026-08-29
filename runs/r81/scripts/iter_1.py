import os
import time
import json
import random
import numpy as np
import torch
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
LR = 0.001
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 32768
MAX_EPOCHS = 8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    torch.set_num_threads(min(12, max(1, os.cpu_count() or 1)))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))


def make_matrix(split):
    cols = []
    for field, offset in zip(FIELDS, offsets):
        x = np.asarray(split.X[field], dtype=np.int64)
        cols.append(x + offset)
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


class FactorizationMachine(torch.nn.Module):
    def __init__(self, n_categories, rank):
        super().__init__()
        # Channel zero stores first-order terms; remaining channels are factors.
        self.embedding = torch.nn.Embedding(
            n_categories, rank + 1, sparse=True
        )
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x):
        w = self.embedding(x)
        linear = w[:, :, 0].sum(dim=1)
        v = w[:, :, 1:]
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1) -
            v.square().sum(dim=(1, 2))
        )
        return self.bias + linear + interaction


def predict(model, x_np):
    model.eval()
    x = torch.from_numpy(x_np)
    out = np.empty(len(x_np), dtype=np.float64)
    with torch.no_grad():
        for begin in range(0, len(x_np), PRED_BATCH_SIZE):
            end = min(begin + PRED_BATCH_SIZE, len(x_np))
            logits = model(x[begin:end])
            out[begin:end] = logits.detach().cpu().numpy().astype(np.float64)
    return out


def initialize_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = FactorizationMachine(total_cardinality, K)
    sparse_params = [model.embedding.weight]
    dense_params = [model.bias]
    sparse_optimizer = torch.optim.SparseAdam(sparse_params, lr=LR)
    dense_optimizer = torch.optim.Adam(dense_params, lr=LR)
    return model, sparse_optimizer, dense_optimizer


def train_one_epoch(model, sparse_optimizer, dense_optimizer, x, y, generator):
    model.train()
    n = len(y)
    permutation = torch.randperm(n, generator=generator)
    total_loss = 0.0

    for begin in range(0, n, BATCH_SIZE):
        idx = permutation[begin:begin + BATCH_SIZE]
        xb = x[idx]
        yb = y[idx]

        sparse_optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()

        sparse_optimizer.step()
        dense_optimizer.step()
        total_loss += float(loss.detach()) * len(idx)

    return total_loss / n


train = load("train")
valid = load("valid")

x_train_np = make_matrix(train)
x_valid_np = make_matrix(valid)
y_train_np = np.asarray(train.y, dtype=np.float32)
y_valid_np = np.asarray(valid.y, dtype=np.int8)

x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)

model, sparse_optimizer, dense_optimizer = initialize_model(SEED)
shuffle_generator = torch.Generator()
shuffle_generator.manual_seed(SEED + 1)

best_epoch = 1
best_metrics = None
best_valid_scores = None

for epoch in range(1, MAX_EPOCHS + 1):
    loss = train_one_epoch(
        model, sparse_optimizer, dense_optimizer,
        x_train, y_train, shuffle_generator
    )
    valid_scores = predict(model, x_valid_np)
    metrics = evaluate(valid.user_id, y_valid_np, valid_scores)

    print(
        "epoch=%d loss=%.6f primary=%.6f gauc=%.6f ndcg@5=%.6f"
        % (
            epoch,
            loss,
            metrics["primary"],
            metrics["gauc"],
            metrics["ndcg@5"],
        ),
        flush=True,
    )

    if best_metrics is None or metrics["primary"] > best_metrics["primary"]:
        best_epoch = epoch
        best_metrics = metrics
        best_valid_scores = valid_scores.copy()

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )

# Refit the identical recipe for the selected number of epochs on train+valid.
x_combined_np = np.ascontiguousarray(
    np.concatenate([x_train_np, x_valid_np], axis=0),
    dtype=np.int64,
)
y_combined_np = np.ascontiguousarray(
    np.concatenate([
        y_train_np,
        y_valid_np.astype(np.float32, copy=False),
    ]),
    dtype=np.float32,
)

x_combined = torch.from_numpy(x_combined_np)
y_combined = torch.from_numpy(y_combined_np)

final_model, final_sparse_optimizer, final_dense_optimizer = initialize_model(SEED)
final_generator = torch.Generator()
final_generator.manual_seed(SEED + 1)

for epoch in range(1, best_epoch + 1):
    refit_loss = train_one_epoch(
        final_model,
        final_sparse_optimizer,
        final_dense_optimizer,
        x_combined,
        y_combined,
        final_generator,
    )
    print(
        "refit_epoch=%d/%d loss=%.6f" % (epoch, best_epoch, refit_loss),
        flush=True,
    )

test = load("test")
x_test_np = make_matrix(test)
test_scores = predict(final_model, x_test_np)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result, separators=(", ", ": ")))