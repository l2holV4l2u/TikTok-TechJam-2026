import os
import time
import json
import gc
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
EMBED_DIM = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 4096
EPOCHS = 8


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))


cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.zeros(len(FIELDS), dtype=np.int64)
if len(FIELDS) > 1:
    offsets[1:] = np.cumsum(cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_matrix(split):
    cols = []
    for j, field in enumerate(FIELDS):
        col = np.asarray(split.X[field], dtype=np.int64)
        cols.append(col + offsets[j])
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


class FactorizationMachine(nn.Module):
    def __init__(self, n_features, embedding_dim, positive_rate):
        super().__init__()
        # Column zero is the first-order coefficient; remaining columns are
        # the latent factors. Sparse updates make full-data CPU training fast.
        self.embedding = nn.Embedding(
            n_features, embedding_dim + 1, sparse=True
        )
        self.bias = nn.Parameter(
            torch.tensor(
                np.log(positive_rate / max(1.0 - positive_rate, 1e-8)),
                dtype=torch.float32,
            )
        )
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)

    def forward(self, x):
        parameters = self.embedding(x)
        linear = parameters[:, :, 0].sum(dim=1)
        latent = parameters[:, :, 1:]
        summed = latent.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - latent.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


def fit_fm(X, y, seed):
    set_seed(seed)

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(
        np.ascontiguousarray(y, dtype=np.float32)
    )
    model = FactorizationMachine(
        total_cardinality,
        EMBED_DIM,
        float(np.mean(y)),
    )

    sparse_optimizer = torch.optim.SparseAdam(
        [model.embedding.weight], lr=LEARNING_RATE
    )
    bias_optimizer = torch.optim.Adam(
        [model.bias], lr=LEARNING_RATE
    )
    loss_fn = nn.BCEWithLogitsLoss()
    n = X_tensor.shape[0]

    model.train()
    generator = torch.Generator()
    generator.manual_seed(seed)

    for _ in range(EPOCHS):
        order = torch.randperm(n, generator=generator)
        for begin in range(0, n, BATCH_SIZE):
            idx = order[begin:begin + BATCH_SIZE]
            xb = X_tensor[idx]
            yb = y_tensor[idx]

            sparse_optimizer.zero_grad(set_to_none=True)
            bias_optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()

            sparse_optimizer.step()
            bias_optimizer.step()

    return model


def predict(model, X):
    X_tensor = torch.from_numpy(X)
    result = np.empty(X.shape[0], dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for begin in range(0, X.shape[0], 32768):
            end = min(begin + 32768, X.shape[0])
            result[begin:end] = (
                model(X_tensor[begin:end]).cpu().numpy()
            )
    return result


# Train-only fit used for the official validation comparison.
train = load("train")
valid = load("valid")

X_train = make_matrix(train)
y_train = np.asarray(train.y, dtype=np.float32)
X_valid = make_matrix(valid)
y_valid = np.asarray(valid.y, dtype=np.int8)

valid_model = fit_fm(X_train, y_train, SEED)
valid_scores = predict(valid_model, X_valid)
metrics = evaluate(valid.user_id, y_valid, valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Release the train-only model before the allowed train+validation refit.
del valid_model
gc.collect()

# Refit the same fixed recipe from scratch on train plus validation.
X_combined = np.ascontiguousarray(
    np.concatenate([X_train, X_valid], axis=0), dtype=np.int64
)
y_combined = np.ascontiguousarray(
    np.concatenate([y_train, y_valid.astype(np.float32)], axis=0),
    dtype=np.float32,
)

del X_train, y_train, X_valid
gc.collect()

test_model = fit_fm(X_combined, y_combined, SEED)

# Test labels are never accessed.
test = load("test")
X_test = make_matrix(test)
test_scores = predict(test_model, X_test)

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
        }
    )
)