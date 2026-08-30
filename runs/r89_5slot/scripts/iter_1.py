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
BATCH_SIZE = 2048
PRED_BATCH_SIZE = 32768
MAX_EPOCHS = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))


def make_offsets():
    offsets = []
    total = 0
    for name in FIELDS:
        offsets.append(total)
        total += int(FEATURE_CARDINALITIES[name])
    return np.asarray(offsets, dtype=np.int64), total


OFFSETS, TOTAL_CARDINALITY = make_offsets()


def make_matrix(split):
    x = np.empty((len(split.user_id), len(FIELDS)), dtype=np.int64)
    for j, name in enumerate(FIELDS):
        x[:, j] = split.X[name] + OFFSETS[j]
    return x


class FactorizationMachine(nn.Module):
    def __init__(self, cardinality, rank, positive_rate):
        super().__init__()
        self.linear = nn.Embedding(cardinality, 1)
        self.embedding = nn.Embedding(cardinality, rank)
        p = float(np.clip(positive_rate, 1e-5, 1.0 - 1e-5))
        self.bias = nn.Parameter(torch.tensor(np.log(p / (1.0 - p)),
                                              dtype=torch.float32))
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


@torch.no_grad()
def predict(model, x_np):
    model.eval()
    n = x_np.shape[0]
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, n)
        xb = torch.from_numpy(x_np[start:end])
        out[start:end] = model(xb).cpu().numpy()
    return out


def train_epochs(x_np, y_np, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FactorizationMachine(
        TOTAL_CARDINALITY, K, float(np.mean(y_np))
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np.astype(np.float32, copy=False))
    generator = torch.Generator()
    generator.manual_seed(seed)

    model.train()
    n = x.shape[0]
    for epoch in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:min(start + BATCH_SIZE, n)]
            xb = x[idx]
            yb = y[idx]

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    return model


def select_epoch(train_x, train_y, valid_x, valid_y, valid_users):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = FactorizationMachine(
        TOTAL_CARDINALITY, K, float(np.mean(train_y))
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    x = torch.from_numpy(train_x)
    y = torch.from_numpy(train_y.astype(np.float32, copy=False))
    generator = torch.Generator()
    generator.manual_seed(SEED)

    best_primary = -np.inf
    best_epoch = 1
    best_scores = None
    epochs_without_gain = 0
    n = x.shape[0]

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(n, generator=generator)

        for start in range(0, n, BATCH_SIZE):
            idx = order[start:min(start + BATCH_SIZE, n)]
            xb = x[idx]
            yb = y[idx]

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        scores = predict(model, valid_x)
        metrics = evaluate(valid_users, valid_y, scores)
        primary = float(metrics["primary"])
        print(
            "epoch=%d primary=%.6f gauc=%.6f ndcg5=%.6f"
            % (epoch, primary, metrics["gauc"], metrics["ndcg@5"]),
            flush=True,
        )

        if primary > best_primary:
            if primary > best_primary + 0.0002:
                epochs_without_gain = 0
            else:
                epochs_without_gain += 1
            best_primary = primary
            best_epoch = epoch
            best_scores = scores.copy()
        else:
            epochs_without_gain += 1

        if epoch >= 5 and epochs_without_gain >= 3:
            break

    return best_epoch, best_scores


train = load("train")
valid = load("valid")

train_x = make_matrix(train)
valid_x = make_matrix(valid)
train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)

best_epoch, valid_scores = select_epoch(
    train_x, train_y, valid_x, valid_y, valid_users
)
valid_metrics = evaluate(valid_users, valid_y, valid_scores)
print("selected_epoch=%d" % best_epoch, flush=True)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Refit the identical selected-epoch recipe on train + validation.
combined_x = np.concatenate([train_x, valid_x], axis=0)
combined_y = np.concatenate(
    [train_y, valid_y.astype(np.float32, copy=False)], axis=0
)

del train_x, valid_x, train_y
combined_model = train_epochs(
    combined_x, combined_y, epochs=best_epoch, seed=SEED
)
del combined_x, combined_y

test = load("test")
test_x = make_matrix(test)
test_scores = predict(combined_model, test_x)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START_TIME)
result = {
    "primary": float(valid_metrics["primary"]),
    "gauc": float(valid_metrics["gauc"]),
    "ndcg@5": float(valid_metrics["ndcg@5"]),
    "gpu_seconds": elapsed,
}
print("METRICS " + json.dumps(result, separators=(", ", ": ")))