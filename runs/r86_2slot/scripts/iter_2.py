import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2022
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
EMBED_DIM = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 4096
EPOCHS = 5

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))


class FactorizationMachine(nn.Module):
    def __init__(self, cardinalities, embedding_dim):
        super().__init__()
        self.offsets = torch.tensor(
            np.cumsum([0] + cardinalities[:-1]), dtype=torch.long
        )
        total_cardinality = int(sum(cardinalities))

        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        offsets = self.offsets.to(x.device)
        x = x + offsets

        linear_term = self.linear(x).sum(dim=1).squeeze(1)
        factors = self.embedding(x)
        summed = factors.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - factors.square().sum(dim=1)
        ).sum(dim=1)

        return self.bias + linear_term + interaction


def make_matrix(split):
    # Explicit whitelist: no auxiliary outcomes, numeric columns, dates, or labels.
    return np.stack(
        [np.asarray(split.X[name], dtype=np.int64) for name in FIELDS],
        axis=1,
    )


def fit_fm(x_np, y_np, seed):
    torch.manual_seed(seed)
    model = FactorizationMachine(
        [int(FEATURE_CARDINALITIES[name]) for name in FIELDS],
        EMBED_DIM,
    )
    model.train()

    x = torch.from_numpy(np.ascontiguousarray(x_np))
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    n = x.shape[0]
    generator = torch.Generator()
    generator.manual_seed(seed)

    for _ in range(EPOCHS):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[idx])
            loss = criterion(logits, y[idx])
            loss.backward()
            optimizer.step()

    return model


def predict(model, x_np):
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(x_np))
    result = np.empty(x.shape[0], dtype=np.float64)

    with torch.inference_mode():
        for start in range(0, x.shape[0], BATCH_SIZE * 2):
            end = min(start + BATCH_SIZE * 2, x.shape[0])
            result[start:end] = (
                model(x[start:end]).detach().cpu().numpy().astype(np.float64)
            )
    return result


# Validation model: fit exclusively on the official training split.
train = load("train")
valid = load("valid")

x_train = make_matrix(train)
x_valid = make_matrix(valid)
y_train = np.asarray(train.y, dtype=np.float32)

validation_model = fit_fm(x_train, y_train, SEED)
valid_scores = predict(validation_model, x_valid)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

del validation_model
gc.collect()

# Permitted final refit using train plus validation, with the identical recipe.
y_valid = np.asarray(valid.y, dtype=np.float32)
x_combined = np.concatenate([x_train, x_valid], axis=0)
y_combined = np.concatenate([y_train, y_valid], axis=0)

del x_train, y_train
gc.collect()

test_model = fit_fm(x_combined, y_combined, SEED)

# Test labels are never accessed.
test = load("test")
x_test = make_matrix(test)
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
        }
    )
)