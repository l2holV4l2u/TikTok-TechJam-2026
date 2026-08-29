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
PRED_BATCH_SIZE = 65536
MAX_EPOCHS = 8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))


cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets_np = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_features(s):
    x = np.column_stack([np.asarray(s.X[f], dtype=np.int64) for f in FIELDS])
    x += offsets_np[None, :]
    return np.ascontiguousarray(x, dtype=np.int64)


class FactorizationMachine(nn.Module):
    def __init__(self, n_categories, k):
        super().__init__()
        self.linear = nn.Embedding(n_categories, 1)
        self.embedding = nn.Embedding(n_categories, k)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1) - v.square().sum(dim=(1, 2))
        )
        return self.bias + linear_term + interaction


@torch.no_grad()
def predict(model, x_np):
    model.eval()
    result = np.empty(x_np.shape[0], dtype=np.float64)
    for begin in range(0, x_np.shape[0], PRED_BATCH_SIZE):
        end = min(begin + PRED_BATCH_SIZE, x_np.shape[0])
        xb = torch.from_numpy(x_np[begin:end])
        result[begin:end] = model(xb).cpu().numpy().astype(np.float64)
    return result


def train_epoch(model, optimizer, loss_fn, x, y, generator):
    model.train()
    order = torch.randperm(x.shape[0], generator=generator)
    for begin in range(0, x.shape[0], BATCH_SIZE):
        idx = order[begin:begin + BATCH_SIZE]
        xb = x[idx]
        yb = y[idx]

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()


def fit_train_with_epoch_selection(x_train_np, y_train_np, valid_split, x_valid_np):
    torch.manual_seed(SEED)
    model = FactorizationMachine(total_cardinality, K)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(np.asarray(y_train_np, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(SEED)

    best_primary = -np.inf
    best_epoch = 1
    best_state = None
    best_scores = None
    best_metrics = None

    for epoch in range(1, MAX_EPOCHS + 1):
        train_epoch(model, optimizer, loss_fn, x_train, y_train, generator)
        scores = predict(model, x_valid_np)
        metrics = evaluate(valid_split.user_id, valid_split.y, scores)

        if metrics["primary"] > best_primary:
            best_primary = float(metrics["primary"])
            best_epoch = epoch
            best_scores = scores.copy()
            best_metrics = metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    return model, best_epoch, best_scores, best_metrics


def fit_fixed_epochs(x_np, y_np, epochs):
    torch.manual_seed(SEED)
    model = FactorizationMachine(total_cardinality, K)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(SEED)

    for _ in range(epochs):
        train_epoch(model, optimizer, loss_fn, x, y, generator)

    return model


# Train-only fit and validation evaluation.
train = load("train")
valid = load("valid")

x_train = make_features(train)
x_valid = make_features(valid)

valid_model, selected_epochs, valid_scores, metrics = fit_train_with_epoch_selection(
    x_train, train.y, valid, x_valid
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Refit the identical selected-epoch recipe on train + validation.
x_train_valid = np.concatenate([x_train, x_valid], axis=0)
y_train_valid = np.concatenate([
    np.asarray(train.y, dtype=np.float32),
    np.asarray(valid.y, dtype=np.float32),
])

del valid_model
refit_model = fit_fixed_epochs(x_train_valid, y_train_valid, selected_epochs)

# Produce test scores without accessing test labels.
test = load("test")
x_test = make_features(test)
test_scores = predict(refit_model, x_test)

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