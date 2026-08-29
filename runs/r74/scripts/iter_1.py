import os
import gc
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
LR = 0.001
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 32768
MAX_EPOCHS = 10


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
seed_everything(SEED)

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_matrix(split):
    cols = []
    for field, offset in zip(FIELDS, offsets):
        x = np.asarray(split.X[field], dtype=np.int64)
        cols.append(x + offset)
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


class FactorizationMachine(nn.Module):
    def __init__(self, n_features, embedding_dim):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1, sparse=True)
        self.factors = nn.Embedding(n_features, embedding_dim, sparse=True)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)

    def forward(self, x):
        first_order = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.factors(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + first_order + interaction


@torch.no_grad()
def predict(model, x):
    model.eval()
    x_tensor = torch.from_numpy(x)
    out = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(x))
        logits = model(x_tensor[start:end])
        out[start:end] = logits.cpu().numpy().astype(np.float32, copy=False)
    return out


def train_one_epoch(model, x_tensor, y_tensor, sparse_optimizer,
                    bias_optimizer, generator):
    model.train()
    n = x_tensor.shape[0]
    order = torch.randperm(n, generator=generator)

    for start in range(0, n, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        xb = x_tensor[idx]
        yb = y_tensor[idx]

        sparse_optimizer.zero_grad(set_to_none=True)
        bias_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()

        sparse_optimizer.step()
        bias_optimizer.step()


def build_model_and_optimizers(seed):
    seed_everything(seed)
    model = FactorizationMachine(total_cardinality, K)
    sparse_optimizer = torch.optim.SparseAdam(
        [model.linear.weight, model.factors.weight], lr=LR
    )
    bias_optimizer = torch.optim.Adam([model.bias], lr=LR)
    return model, sparse_optimizer, bias_optimizer


train = load("train")
valid = load("valid")

x_train = make_matrix(train)
x_valid = make_matrix(valid)
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

x_train_tensor = torch.from_numpy(x_train)
y_train_tensor = torch.from_numpy(y_train)

model, sparse_optimizer, bias_optimizer = build_model_and_optimizers(SEED)
generator = torch.Generator()
generator.manual_seed(SEED + 17)

best_epoch = 1
best_primary = -np.inf
best_valid_scores = None
best_metrics = None

for epoch in range(1, MAX_EPOCHS + 1):
    train_one_epoch(
        model,
        x_train_tensor,
        y_train_tensor,
        sparse_optimizer,
        bias_optimizer,
        generator,
    )

    valid_scores_epoch = predict(model, x_valid)
    metrics_epoch = evaluate(valid.user_id, y_valid, valid_scores_epoch)

    if metrics_epoch["primary"] > best_primary:
        best_primary = float(metrics_epoch["primary"])
        best_epoch = epoch
        best_valid_scores = valid_scores_epoch.copy()
        best_metrics = metrics_epoch

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )

del model, sparse_optimizer, bias_optimizer
del x_train_tensor, y_train_tensor
gc.collect()

# Refit the identical recipe on train + validation for the selected number
# of epochs, then score test without accessing test labels.
x_fit = np.ascontiguousarray(
    np.concatenate([x_train, x_valid], axis=0), dtype=np.int64
)
y_fit = np.ascontiguousarray(
    np.concatenate([
        np.asarray(train.y, dtype=np.float32),
        np.asarray(valid.y, dtype=np.float32),
    ]),
    dtype=np.float32,
)

x_fit_tensor = torch.from_numpy(x_fit)
y_fit_tensor = torch.from_numpy(y_fit)

final_model, final_sparse_optimizer, final_bias_optimizer = (
    build_model_and_optimizers(SEED)
)
final_generator = torch.Generator()
final_generator.manual_seed(SEED + 17)

for _ in range(best_epoch):
    train_one_epoch(
        final_model,
        x_fit_tensor,
        y_fit_tensor,
        final_sparse_optimizer,
        final_bias_optimizer,
        final_generator,
    )

test = load("test")
x_test = make_matrix(test)
test_scores = predict(final_model, x_test)

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