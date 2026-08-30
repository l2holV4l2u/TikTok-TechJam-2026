import os
import time
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
LR = 0.001
BATCH_SIZE = 8192
CHECKPOINT_EPOCHS = {4, 8, 12, 16, 20}
MAX_EPOCHS = max(CHECKPOINT_EPOCHS)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))


cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))
zero_rows = torch.as_tensor(offsets, dtype=torch.long)


def encode(split):
    return np.stack(
        [np.asarray(split.X[f], dtype=np.int64) + offsets[j]
         for j, f in enumerate(FIELDS)],
        axis=1,
    )


class FactorizationMachine(nn.Module):
    def __init__(self, n_categories, rank, prior):
        super().__init__()
        self.linear = nn.Embedding(n_categories, 1, sparse=True)
        self.latent = nn.Embedding(n_categories, rank, sparse=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, mean=0.0, std=0.01)
        prior = min(max(float(prior), 1e-5), 1.0 - 1e-5)
        self.register_buffer(
            "intercept",
            torch.tensor(math.log(prior / (1.0 - prior)), dtype=torch.float32),
        )
        with torch.no_grad():
            self.linear.weight[zero_rows].zero_()
            self.latent.weight[zero_rows].zero_()

    def forward(self, x):
        linear_term = self.linear(x).squeeze(-1).sum(dim=1)
        v = self.latent(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.intercept + linear_term + interaction


def predict(model, x_np):
    model.eval()
    ans = np.empty(len(x_np), dtype=np.float32)
    with torch.inference_mode():
        for lo in range(0, len(x_np), BATCH_SIZE * 2):
            hi = min(lo + BATCH_SIZE * 2, len(x_np))
            xb = torch.from_numpy(x_np[lo:hi])
            ans[lo:hi] = model(xb).cpu().numpy()
    return ans


def train_with_validation(x_train, y_train, x_valid, y_valid, valid_users):
    torch.manual_seed(SEED)
    model = FactorizationMachine(total_cardinality, K, y_train.mean())
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(y_train.astype(np.float32, copy=False))

    best_primary = -np.inf
    best_epoch = None
    best_state = None
    best_scores = None
    best_metrics = None

    n = len(x_train)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(n)
        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:min(lo + BATCH_SIZE, n)]
            logits = model(x_tensor[idx])
            loss = criterion(logits, y_tensor[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if epoch in CHECKPOINT_EPOCHS:
            scores = predict(model, x_valid)
            metrics = evaluate(valid_users, y_valid, scores)
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


def fit_fixed_epochs(x_fit, y_fit, epochs):
    torch.manual_seed(SEED)
    model = FactorizationMachine(total_cardinality, K, y_fit.mean())
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    x_tensor = torch.from_numpy(x_fit)
    y_tensor = torch.from_numpy(y_fit.astype(np.float32, copy=False))
    n = len(x_fit)

    for _ in range(epochs):
        model.train()
        order = torch.randperm(n)
        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:min(lo + BATCH_SIZE, n)]
            logits = model(x_tensor[idx])
            loss = criterion(logits, y_tensor[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


train = load("train")
valid = load("valid")

x_train = encode(train)
x_valid = encode(valid)
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

_, selected_epochs, valid_scores, metrics = train_with_validation(
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
    valid_users=np.asarray(valid.user_id),
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Refit the selected recipe on all labeled data immediately preceding test.
x_fit = np.concatenate([x_train, x_valid], axis=0)
y_fit = np.concatenate(
    [y_train, np.asarray(valid.y, dtype=np.float32)],
    axis=0,
)
test_model = fit_fixed_epochs(x_fit, y_fit, selected_epochs)

test = load("test")
x_test = encode(test)
test_scores = predict(test_model, x_test)

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
        separators=(", ", ": "),
    )
)