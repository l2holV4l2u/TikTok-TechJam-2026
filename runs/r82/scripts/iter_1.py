import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
LR = 0.001
BATCH_SIZE = 4096
MAX_EPOCHS = 8
PRED_BATCH_SIZE = 32768


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_everything(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_matrix(split):
    x = np.empty((len(split.user_id), len(FIELDS)), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:, j] = np.asarray(split.X[field], dtype=np.int64) + offsets[j]
    return x


def make_combined_matrix(a, b):
    n_a = len(a.user_id)
    n_b = len(b.user_id)
    x = np.empty((n_a + n_b, len(FIELDS)), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:n_a, j] = np.asarray(a.X[field], dtype=np.int64) + offsets[j]
        x[n_a:, j] = np.asarray(b.X[field], dtype=np.int64) + offsets[j]
    return x


class FactorizationMachine(nn.Module):
    def __init__(self, n_features, rank):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(n_features, 1)
        self.embedding = nn.Embedding(n_features, rank)

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear_term + interaction


def train_one_epoch(model, optimizer, loss_fn, x_tensor, y_tensor, generator):
    model.train()
    n = x_tensor.shape[0]
    permutation = torch.randperm(n, generator=generator)

    total_loss = 0.0
    for start in range(0, n, BATCH_SIZE):
        idx = permutation[start:start + BATCH_SIZE]
        xb = x_tensor[idx]
        yb = y_tensor[idx]

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(idx)

    return total_loss / n


@torch.no_grad()
def predict(model, x):
    model.eval()
    x_tensor = torch.from_numpy(x)
    result = np.empty(x.shape[0], dtype=np.float32)

    for start in range(0, x.shape[0], PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, x.shape[0])
        logits = model(x_tensor[start:end])
        result[start:end] = logits.cpu().numpy().astype(np.float32, copy=False)

    return result


def fresh_model(seed):
    seed_everything(seed)
    model = FactorizationMachine(total_cardinality, K)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    return model, optimizer


train = load("train")
valid = load("valid")

x_train = make_matrix(train)
x_valid = make_matrix(valid)
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

x_train_t = torch.from_numpy(x_train)
y_train_t = torch.from_numpy(y_train)

model, optimizer = fresh_model(SEED)
loss_fn = nn.BCEWithLogitsLoss()
shuffle_generator = torch.Generator()
shuffle_generator.manual_seed(SEED + 17)

best_primary = -np.inf
best_metrics = None
best_scores = None
best_state = None
best_epoch = 1

for epoch in range(1, MAX_EPOCHS + 1):
    train_one_epoch(
        model, optimizer, loss_fn, x_train_t, y_train_t, shuffle_generator
    )
    valid_scores_epoch = predict(model, x_valid)
    metrics_epoch = evaluate(valid.user_id, y_valid, valid_scores_epoch)

    if float(metrics_epoch["primary"]) > best_primary:
        best_primary = float(metrics_epoch["primary"])
        best_metrics = metrics_epoch
        best_scores = valid_scores_epoch.copy()
        best_epoch = epoch
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

model.load_state_dict(best_state)
valid_scores = predict(model, x_valid)
metrics = evaluate(valid.user_id, y_valid, valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Refit the selected recipe on all labels available before the test period.
test = load("test")
x_combined = make_combined_matrix(train, valid)
y_combined = np.concatenate([
    np.asarray(train.y, dtype=np.float32),
    np.asarray(valid.y, dtype=np.float32),
])
x_test = make_matrix(test)

combined_model, combined_optimizer = fresh_model(SEED)
combined_x_t = torch.from_numpy(x_combined)
combined_y_t = torch.from_numpy(y_combined)
combined_generator = torch.Generator()
combined_generator.manual_seed(SEED + 17)

for _ in range(best_epoch):
    train_one_epoch(
        combined_model,
        combined_optimizer,
        loss_fn,
        combined_x_t,
        combined_y_t,
        combined_generator,
    )

test_scores = predict(combined_model, x_test)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START_TIME
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))