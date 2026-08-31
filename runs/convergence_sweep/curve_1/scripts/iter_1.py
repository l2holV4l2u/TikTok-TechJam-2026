import os
import time
import json
import copy
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 2024

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

cpu_count = os.cpu_count() or 8
torch.set_num_threads(min(16, cpu_count))
try:
    torch.set_num_interop_threads(min(4, cpu_count))
except RuntimeError:
    pass

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
]
EMBED_DIM = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 8192
MAX_EPOCHS = 12
PATIENCE = 3


def make_matrix(split):
    return np.column_stack(
        [np.asarray(split.X[name], dtype=np.int64) for name in FIELDS]
    )


cardinalities = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
offsets_np = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


class FactorizationMachine(nn.Module):
    def __init__(self, total_features, offsets, embedding_dim):
        super().__init__()
        self.register_buffer(
            "offsets", torch.as_tensor(offsets, dtype=torch.long)
        )
        self.linear = nn.Embedding(total_features, 1)
        self.embedding = nn.Embedding(total_features, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        x = x + self.offsets
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)

        emb = self.embedding(x)
        summed = emb.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - emb.square().sum(dim=1)
        ).sum(dim=1)

        return self.bias + linear_term + interaction


@torch.inference_mode()
def predict(model, x_np, batch_size=32768):
    model.eval()
    outputs = np.empty(x_np.shape[0], dtype=np.float64)
    for start in range(0, x_np.shape[0], batch_size):
        end = min(start + batch_size, x_np.shape[0])
        xb = torch.from_numpy(x_np[start:end])
        outputs[start:end] = model(xb).cpu().numpy().astype(np.float64)
    return outputs


train = load("train")
valid = load("valid")

x_train_np = make_matrix(train)
x_valid_np = make_matrix(valid)
y_train_np = np.asarray(train.y, dtype=np.float32)
y_valid_np = np.asarray(valid.y, dtype=np.int8)

x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)

model = FactorizationMachine(
    total_features=total_cardinality,
    offsets=offsets_np,
    embedding_dim=EMBED_DIM,
)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

best_primary = -np.inf
best_metrics = None
best_scores = None
best_state = None
stale_epochs = 0

generator = torch.Generator()
generator.manual_seed(SEED)

n_train = x_train.shape[0]

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    permutation = torch.randperm(n_train, generator=generator)

    total_loss = 0.0
    total_seen = 0

    for start in range(0, n_train, BATCH_SIZE):
        indices = permutation[start:start + BATCH_SIZE]
        xb = x_train[indices]
        yb = y_train[indices]

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        batch_n = int(indices.numel())
        total_loss += float(loss.detach()) * batch_n
        total_seen += batch_n

    valid_scores_epoch = predict(model, x_valid_np)
    metrics_epoch = evaluate(valid.user_id, y_valid_np, valid_scores_epoch)
    primary_epoch = float(metrics_epoch["primary"])

    print(
        "epoch={} loss={:.6f} primary={:.6f} gauc={:.6f} ndcg5={:.6f}".format(
            epoch,
            total_loss / max(total_seen, 1),
            primary_epoch,
            float(metrics_epoch["gauc"]),
            float(metrics_epoch["ndcg@5"]),
        ),
        flush=True,
    )

    if primary_epoch > best_primary + 1e-5:
        best_primary = primary_epoch
        best_metrics = metrics_epoch
        best_scores = valid_scores_epoch.copy()
        best_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        stale_epochs = 0
    else:
        stale_epochs += 1

    if stale_epochs >= PATIENCE:
        break

model.load_state_dict(best_state)
model.eval()

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )

test = load("test")
x_test_np = make_matrix(test)
test_scores = predict(model, x_test_np)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START_TIME
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result, separators=(",", ":")))