import os
import time
import json
import math
import copy
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
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 32768
MAX_EPOCHS = 12

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def make_offsets():
    offsets = []
    cur = 0
    for name in FIELDS:
        offsets.append(cur)
        cur += int(FEATURE_CARDINALITIES[name])
    return np.asarray(offsets, dtype=np.int64), cur


OFFSETS, TOTAL_CARDINALITY = make_offsets()


def feature_matrix(split):
    x = np.empty((len(split.user_id), len(FIELDS)), dtype=np.int64)
    for j, name in enumerate(FIELDS):
        x[:, j] = split.X[name] + OFFSETS[j]
    return x


class FactorizationMachine(nn.Module):
    def __init__(self, cardinality, rank, initial_rate):
        super().__init__()
        # Column zero is the first-order coefficient; remaining columns are
        # the latent FM vector. Sparse gradients make CPU fitting efficient.
        self.embedding = nn.Embedding(cardinality, rank + 1, sparse=True)
        self.bias = nn.Parameter(
            torch.tensor(math.log(initial_rate / (1.0 - initial_rate)),
                         dtype=torch.float32)
        )
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


def predict(model, x_np):
    model.eval()
    x = torch.from_numpy(x_np)
    result = np.empty(len(x_np), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x_np), PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, len(x_np))
            result[start:end] = model(x[start:end]).cpu().numpy()
    return result


def fit_with_validation(x_train, y_train, x_valid, y_valid, valid_users):
    torch.manual_seed(SEED)
    model = FactorizationMachine(
        TOTAL_CARDINALITY, K, float(np.mean(y_train))
    )
    sparse_optimizer = torch.optim.SparseAdam(
        [model.embedding.weight], lr=LR
    )
    bias_optimizer = torch.optim.Adam([model.bias], lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(y_train.astype(np.float32, copy=False))

    generator = torch.Generator()
    generator.manual_seed(SEED)

    best_primary = -np.inf
    best_epoch = 1
    best_state = None
    best_scores = None
    best_metrics = None

    n = len(x_train)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(n, generator=generator)

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:min(start + BATCH_SIZE, n)]
            sparse_optimizer.zero_grad(set_to_none=True)
            bias_optimizer.zero_grad(set_to_none=True)

            logits = model(x_tensor[idx])
            loss = criterion(logits, y_tensor[idx])
            loss.backward()

            sparse_optimizer.step()
            bias_optimizer.step()

        valid_scores = predict(model, x_valid)
        metrics = evaluate(valid_users, y_valid, valid_scores)
        if metrics["primary"] > best_primary:
            best_primary = float(metrics["primary"])
            best_epoch = epoch
            best_scores = valid_scores.copy()
            best_metrics = metrics
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    return model, best_epoch, best_scores, best_metrics


def fit_fixed_epochs(x_train, y_train, epochs):
    torch.manual_seed(SEED)
    model = FactorizationMachine(
        TOTAL_CARDINALITY, K, float(np.mean(y_train))
    )
    sparse_optimizer = torch.optim.SparseAdam(
        [model.embedding.weight], lr=LR
    )
    bias_optimizer = torch.optim.Adam([model.bias], lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(y_train.astype(np.float32, copy=False))
    generator = torch.Generator()
    generator.manual_seed(SEED)
    n = len(x_train)

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:min(start + BATCH_SIZE, n)]
            sparse_optimizer.zero_grad(set_to_none=True)
            bias_optimizer.zero_grad(set_to_none=True)

            loss = criterion(model(x_tensor[idx]), y_tensor[idx])
            loss.backward()
            sparse_optimizer.step()
            bias_optimizer.step()

    return model


train = load("train")
valid = load("valid")

x_train = feature_matrix(train)
x_valid = feature_matrix(valid)
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

valid_model, selected_epochs, valid_scores, metrics = fit_with_validation(
    x_train, y_train, x_valid, y_valid, valid.user_id
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64)
    )

# Refit the identical FM recipe through the end of validation, using the
# epoch count selected by the train-only validation experiment.
x_combined = np.concatenate([x_train, x_valid], axis=0)
y_combined = np.concatenate(
    [y_train, y_valid.astype(np.float32, copy=False)], axis=0
)

del valid_model, x_train, x_valid, y_train
refit_model = fit_fixed_epochs(x_combined, y_combined, selected_epochs)

test = load("test")
x_test = feature_matrix(test)
test_scores = predict(refit_model, x_test)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed)
}))