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
BATCH_SIZE = 8192
MAX_EPOCHS = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
num_features = int(sum(cardinalities))


def make_matrix(split):
    cols = [
        np.asarray(split.X[name], dtype=np.int64) + offsets[j]
        for j, name in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


class FactorizationMachine(nn.Module):
    def __init__(self, n_features, rank, initial_rate):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1)
        self.embedding = nn.Embedding(n_features, rank)
        self.bias = nn.Parameter(
            torch.tensor(
                np.log(initial_rate / max(1.0 - initial_rate, 1e-7)),
                dtype=torch.float32,
            )
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(1)
        latent = self.embedding(x)
        summed = latent.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - latent.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


def predict(model, x_np):
    model.eval()
    x = torch.from_numpy(x_np)
    result = np.empty(len(x_np), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(x_np), BATCH_SIZE * 2):
            end = min(start + BATCH_SIZE * 2, len(x_np))
            result[start:end] = (
                model(x[start:end]).detach().cpu().numpy().astype(np.float64)
            )
    return result


def train_one_epoch(model, optimizer, loss_fn, x, y, generator):
    model.train()
    order = torch.randperm(len(x), generator=generator)
    total_loss = 0.0

    for start in range(0, len(x), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        logits = model(x[idx])
        loss = loss_fn(logits, y[idx])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach()) * len(idx)

    return total_loss / len(x)


def fit_with_validation(x_train_np, y_train_np, x_valid_np, valid):
    torch.manual_seed(SEED)
    model = FactorizationMachine(
        num_features, K, float(np.mean(y_train_np))
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(
        np.asarray(y_train_np, dtype=np.float32)
    )
    generator = torch.Generator()
    generator.manual_seed(SEED)

    best_primary = -np.inf
    best_epoch = 1
    best_state = None
    best_scores = None
    best_metrics = None

    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_one_epoch(
            model, optimizer, loss_fn, x_train, y_train, generator
        )
        scores = predict(model, x_valid_np)
        metrics = evaluate(valid.user_id, valid.y, scores)
        primary = float(metrics["primary"])

        print(
            "epoch=%d loss=%.6f primary=%.6f gauc=%.6f ndcg5=%.6f"
            % (
                epoch,
                loss,
                primary,
                float(metrics["gauc"]),
                float(metrics["ndcg@5"]),
            ),
            flush=True,
        )

        if primary > best_primary:
            best_primary = primary
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
    model = FactorizationMachine(num_features, K, float(np.mean(y_np)))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(SEED)

    for epoch in range(epochs):
        train_one_epoch(model, optimizer, loss_fn, x, y, generator)

    return model


train = load("train")
valid = load("valid")

x_train = make_matrix(train)
x_valid = make_matrix(valid)
y_train = np.asarray(train.y, dtype=np.float32)

_, selected_epochs, valid_scores, metrics = fit_with_validation(
    x_train, y_train, x_valid, valid
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

test = load("test")
x_test = make_matrix(test)

x_combined = np.ascontiguousarray(
    np.concatenate([x_train, x_valid], axis=0), dtype=np.int64
)
y_combined = np.ascontiguousarray(
    np.concatenate(
        [y_train, np.asarray(valid.y, dtype=np.float32)], axis=0
    ),
    dtype=np.float32,
)

final_model = fit_fixed_epochs(x_combined, y_combined, selected_epochs)
test_scores = predict(final_model, x_test)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START_TIME
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(",", ":"),
    )
)