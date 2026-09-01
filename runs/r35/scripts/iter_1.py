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
EMBED_DIM = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 4096
MAX_EPOCHS = 15
PATIENCE = 3


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def make_offsets(fields):
    offsets = []
    running = 0
    for name in fields:
        offsets.append(running)
        running += int(FEATURE_CARDINALITIES[name])
    return np.asarray(offsets, dtype=np.int64), running


OFFSETS, TOTAL_CARDINALITY = make_offsets(FIELDS)


def encode_split(split):
    columns = []
    for j, name in enumerate(FIELDS):
        values = np.asarray(split.X[name], dtype=np.int64)
        columns.append(values + OFFSETS[j])
    return torch.from_numpy(np.stack(columns, axis=1))


class FactorizationMachine(nn.Module):
    def __init__(self, cardinality, embedding_dim):
        super().__init__()
        self.linear = nn.Embedding(cardinality, 1)
        self.embedding = nn.Embedding(cardinality, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)
        latent = self.embedding(x)
        summed = latent.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1)
            - latent.square().sum(dim=(1, 2))
        )
        return self.bias + linear_term + interaction


def predict(model, x, batch_size=32768):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            result[start:stop] = model(x[start:stop]).cpu().numpy()
    return result


train = load("train")
valid = load("valid")

x_train = encode_split(train)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))
x_valid = encode_split(valid)

model = FactorizationMachine(TOTAL_CARDINALITY, EMBED_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

generator = torch.Generator()
generator.manual_seed(SEED)

best_primary = -np.inf
best_state = None
epochs_without_improvement = 0

for epoch in range(MAX_EPOCHS):
    model.train()
    permutation = torch.randperm(x_train.shape[0], generator=generator)

    for start in range(0, x_train.shape[0], BATCH_SIZE):
        batch_idx = permutation[start:start + BATCH_SIZE]
        logits = model(x_train[batch_idx])
        loss = criterion(logits, y_train[batch_idx])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    valid_scores = predict(model, x_valid)
    epoch_metrics = evaluate(valid.user_id, valid.y, valid_scores)
    epoch_primary = float(epoch_metrics["primary"])

    if epoch_primary > best_primary:
        best_primary = epoch_primary
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch >= 4 and epochs_without_improvement >= PATIENCE:
        break

model.load_state_dict(best_state)

valid_scores = predict(model, x_valid)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    test = load("test")
    x_test = encode_split(test)
    test_scores = predict(model, x_test)
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

final_metrics = {
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": 0.0,
}
print("METRICS " + json.dumps(final_metrics, separators=(",", ":")))