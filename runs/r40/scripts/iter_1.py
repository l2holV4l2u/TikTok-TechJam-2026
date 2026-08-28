import os
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
LR = 0.001
BATCH_SIZE = 4096
EPOCHS = 12

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def make_matrix(split, offsets):
    cols = []
    for name, offset in zip(FIELDS, offsets):
        cols.append(np.asarray(split.X[name], dtype=np.int64) + offset)
    return torch.from_numpy(np.stack(cols, axis=1))


cards = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))


class FactorizationMachine(nn.Module):
    def __init__(self, cardinality, rank):
        super().__init__()
        # Column zero is the linear coefficient; remaining columns are factors.
        self.embedding = nn.Embedding(cardinality, rank + 1, sparse=True)
        self.bias = nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)

    def forward(self, x):
        e = self.embedding(x)
        linear = e[:, :, 0].sum(dim=1)
        factors = e[:, :, 1:]
        summed = factors.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1)
            - factors.square().sum(dim=(1, 2))
        )
        return self.bias + linear + interaction


@torch.no_grad()
def predict(model, x, batch_size=32768):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        result[start:end] = model(x[start:end]).cpu().numpy().astype(np.float64)
    return result


train = load("train")
valid = load("valid")

x_train = make_matrix(train, offsets)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))
x_valid = make_matrix(valid, offsets)

model = FactorizationMachine(total_cardinality, K)
embedding_optimizer = torch.optim.SparseAdam(
    [model.embedding.weight], lr=LR
)
bias_optimizer = torch.optim.Adam([model.bias], lr=LR)
criterion = nn.BCEWithLogitsLoss()

n_train = x_train.shape[0]
generator = torch.Generator()
generator.manual_seed(SEED)

best_primary = -np.inf
best_embedding = None
best_bias = None
best_metrics = None

for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(n_train, generator=generator)

    for start in range(0, n_train, BATCH_SIZE):
        idx = permutation[start:min(start + BATCH_SIZE, n_train)]
        xb = x_train[idx]
        yb = y_train[idx]

        embedding_optimizer.zero_grad(set_to_none=True)
        bias_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()

        embedding_optimizer.step()
        bias_optimizer.step()

    valid_scores = predict(model, x_valid)
    metrics = evaluate(valid.user_id, valid.y, valid_scores)

    if float(metrics["primary"]) > best_primary:
        best_primary = float(metrics["primary"])
        best_embedding = model.embedding.weight.detach().clone()
        best_bias = model.bias.detach().clone()
        best_metrics = {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
        }

    print(
        "epoch=%d loss=%.6f primary=%.6f gauc=%.6f ndcg@5=%.6f"
        % (
            epoch + 1,
            float(loss.detach()),
            float(metrics["primary"]),
            float(metrics["gauc"]),
            float(metrics["ndcg@5"]),
        ),
        flush=True,
    )

with torch.no_grad():
    model.embedding.weight.copy_(best_embedding)
    model.bias.copy_(best_bias)

# Generate the required hidden-test scores without inspecting test labels.
out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    test = load("test")
    x_test = make_matrix(test, offsets)
    test_scores = predict(model, x_test)
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

final = {
    "primary": best_metrics["primary"],
    "gauc": best_metrics["gauc"],
    "ndcg@5": best_metrics["ndcg@5"],
    "gpu_seconds": 0.0,
}
print("METRICS " + json.dumps(final))