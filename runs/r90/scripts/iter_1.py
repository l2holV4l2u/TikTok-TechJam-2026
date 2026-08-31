import os
import time
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 42
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
EMBED_DIM = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 2048
EPOCHS = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
device = torch.device("cpu")

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets_np = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_features(split):
    cols = [
        np.asarray(split.X[name], dtype=np.int64) + offsets_np[j]
        for j, name in enumerate(FIELDS)
    ]
    return torch.from_numpy(np.stack(cols, axis=1))


class FactorizationMachine(nn.Module):
    def __init__(self, n_features, embedding_dim):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1)
        self.embedding = nn.Embedding(n_features, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))

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


def fit_model(x, y, seed):
    torch.manual_seed(seed)
    model = FactorizationMachine(total_cardinality, EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    n = x.shape[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1000)

    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb = x[idx].to(device)
            yb = y[idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return model


@torch.inference_mode()
def predict(model, x, batch_size=16384):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        logits = model(x[start:end].to(device))
        result[start:end] = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
    return result


# Train-only fit and validation evaluation.
train = load("train")
valid = load("valid")

x_train = make_features(train)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))
x_valid = make_features(valid)

validation_model = fit_model(x_train, y_train, SEED)
valid_scores = predict(validation_model, x_valid)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Release the train-only model before the allowed train+validation refit.
del validation_model

# Refit the identical recipe on train plus validation, then score test.
y_valid = torch.from_numpy(np.asarray(valid.y, dtype=np.float32))
x_combined = torch.cat([x_train, x_valid], dim=0)
y_combined = torch.cat([y_train, y_valid], dim=0)

test_model = fit_model(x_combined, y_combined, SEED)
test = load("test")
x_test = make_features(test)
test_scores = predict(test_model, x_test)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.6f}'
    % (
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        float(elapsed),
    )
)