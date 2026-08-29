import os
import time
import json
import copy
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

torch.set_num_threads(min(8, os.cpu_count() or 1))
torch.manual_seed(SEED)
np.random.seed(SEED)


def make_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([split.X[name] for name in FIELDS]),
        dtype=np.int64,
    )


CARDINALITIES = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
OFFSETS = np.asarray(
    [0] + list(np.cumsum(CARDINALITIES[:-1], dtype=np.int64)),
    dtype=np.int64,
)
TOTAL_CARDINALITY = int(sum(CARDINALITIES))


def offset_matrix(x):
    return np.ascontiguousarray(x + OFFSETS[None, :], dtype=np.int64)


class FactorizationMachine(nn.Module):
    def __init__(self, num_embeddings, rank, initial_bias=0.0):
        super().__init__()
        self.linear = nn.Embedding(num_embeddings, 1)
        self.latent = nn.Embedding(num_embeddings, rank)
        self.bias = nn.Parameter(torch.tensor(float(initial_bias), dtype=torch.float32))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(1)
        v = self.latent(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear_term + interaction


@torch.no_grad()
def predict(model, x_np, batch_size=16384):
    model.eval()
    x = torch.from_numpy(x_np)
    result = np.empty(len(x_np), dtype=np.float64)
    for start in range(0, len(x_np), batch_size):
        end = min(start + batch_size, len(x_np))
        logits = model(x[start:end])
        result[start:end] = logits.cpu().numpy().astype(np.float64, copy=False)
    return result


def train_fixed_epochs(x_np, y_np, epochs, seed):
    torch.manual_seed(seed)
    n = len(y_np)
    prevalence = float(np.clip(np.mean(y_np), 1e-5, 1.0 - 1e-5))
    initial_bias = np.log(prevalence / (1.0 - prevalence))

    model = FactorizationMachine(TOTAL_CARDINALITY, K, initial_bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.ascontiguousarray(y_np, dtype=np.float32))

    generator = torch.Generator()
    generator.manual_seed(seed + 17)

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(n, generator=generator)
        total_loss = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            xb = x.index_select(0, idx)
            yb = y.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(idx)

        print(
            "refit_epoch=%d loss=%.6f"
            % (epoch + 1, total_loss / n),
            flush=True,
        )

    return model


def train_with_validation(x_train, y_train, x_valid, y_valid, valid_users):
    torch.manual_seed(SEED)
    n = len(y_train)
    prevalence = float(np.clip(np.mean(y_train), 1e-5, 1.0 - 1e-5))
    initial_bias = np.log(prevalence / (1.0 - prevalence))

    model = FactorizationMachine(TOTAL_CARDINALITY, K, initial_bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(SEED + 17)

    best_primary = -np.inf
    best_epoch = 1
    best_scores = None
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        total_loss = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            xb = x_tensor.index_select(0, idx)
            yb = y_tensor.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(idx)

        scores = predict(model, x_valid)
        metrics = evaluate(valid_users, y_valid, scores)
        primary = float(metrics["primary"])

        print(
            "epoch=%d loss=%.6f primary=%.6f gauc=%.6f ndcg5=%.6f"
            % (
                epoch + 1,
                total_loss / n,
                primary,
                float(metrics["gauc"]),
                float(metrics["ndcg@5"]),
            ),
            flush=True,
        )

        if primary > best_primary:
            best_primary = primary
            best_epoch = epoch + 1
            best_scores = scores.copy()
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_epoch, best_scores


train = load("train")
valid = load("valid")

x_train = offset_matrix(make_matrix(train))
x_valid = offset_matrix(make_matrix(valid))
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

valid_model, chosen_epochs, valid_scores = train_with_validation(
    x_train,
    y_train,
    x_valid,
    y_valid,
    np.asarray(valid.user_id),
)

valid_metrics = evaluate(valid.user_id, y_valid, valid_scores)
print("chosen_epochs=%d" % chosen_epochs, flush=True)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Refit the identical FM recipe for the selected number of epochs using all
# labels available before the test interval.
x_combined = np.ascontiguousarray(
    np.concatenate([x_train, x_valid], axis=0),
    dtype=np.int64,
)
y_combined = np.ascontiguousarray(
    np.concatenate([
        y_train,
        np.asarray(valid.y, dtype=np.float32),
    ]),
    dtype=np.float32,
)

del valid_model
refit_model = train_fixed_epochs(
    x_combined,
    y_combined,
    epochs=chosen_epochs,
    seed=SEED,
)

test = load("test")
x_test = offset_matrix(make_matrix(test))
test_scores = predict(refit_model, x_test)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(valid_metrics["primary"]),
            "gauc": float(valid_metrics["gauc"]),
            "ndcg@5": float(valid_metrics["ndcg@5"]),
            "gpu_seconds": elapsed,
        },
        separators=(", ", ": "),
    )
)