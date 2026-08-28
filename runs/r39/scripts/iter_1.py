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
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))


def make_matrix(split):
    return np.stack(
        [np.asarray(split.X[name], dtype=np.int64) for name in FIELDS],
        axis=1,
    )


cards = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))


def add_offsets(x):
    return x + offsets[None, :]


class FactorizationMachine(nn.Module):
    def __init__(self, n_tokens, rank):
        super().__init__()
        # Column 0 is the linear coefficient; the remaining columns are factors.
        self.embedding = nn.Embedding(n_tokens, rank + 1, sparse=True)
        self.bias = nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)

    def forward(self, x):
        z = self.embedding(x)
        linear = z[:, :, 0].sum(dim=1)
        factors = z[:, :, 1:]
        summed = factors.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - factors.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


def predict(model, x_np, batch_size=32768):
    model.eval()
    result = np.empty(x_np.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, x_np.shape[0], batch_size):
            end = min(start + batch_size, x_np.shape[0])
            xb = torch.from_numpy(x_np[start:end])
            result[start:end] = model(xb).cpu().numpy().astype(np.float64)
    return result


train = load("train")
valid = load("valid")

x_train = add_offsets(make_matrix(train))
x_valid = add_offsets(make_matrix(valid))
y_train = np.asarray(train.y, dtype=np.float32)

model = FactorizationMachine(total_cardinality, K)
sparse_optimizer = torch.optim.SparseAdam(
    [model.embedding.weight], lr=LR
)
bias_optimizer = torch.optim.Adam([model.bias], lr=LR)
criterion = nn.BCEWithLogitsLoss()

n_train = len(y_train)
best_primary = -np.inf
best_state = None
best_epoch = -1
best_metrics = None

rng = np.random.default_rng(SEED)

for epoch in range(EPOCHS):
    model.train()
    permutation = rng.permutation(n_train)

    total_loss = 0.0
    total_seen = 0

    for start in range(0, n_train, BATCH_SIZE):
        idx = permutation[start:start + BATCH_SIZE]
        xb = torch.from_numpy(x_train[idx])
        yb = torch.from_numpy(y_train[idx])

        sparse_optimizer.zero_grad(set_to_none=True)
        bias_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()

        sparse_optimizer.step()
        bias_optimizer.step()

        count = len(idx)
        total_loss += float(loss.detach()) * count
        total_seen += count

    valid_scores = predict(model, x_valid)
    metrics = evaluate(valid.user_id, valid.y, valid_scores)

    print(
        "epoch=%d loss=%.6f primary=%.6f gauc=%.6f ndcg5=%.6f"
        % (
            epoch + 1,
            total_loss / max(total_seen, 1),
            float(metrics["primary"]),
            float(metrics["gauc"]),
            float(metrics["ndcg@5"]),
        ),
        flush=True,
    )

    if float(metrics["primary"]) > best_primary:
        best_primary = float(metrics["primary"])
        best_epoch = epoch + 1
        best_metrics = metrics
        best_state = {
            "embedding": model.embedding.weight.detach().clone(),
            "bias": model.bias.detach().clone(),
        }

with torch.no_grad():
    model.embedding.weight.copy_(best_state["embedding"])
    model.bias.copy_(best_state["bias"])

# Recompute validation scores from the exact selected checkpoint.
valid_scores = predict(model, x_valid)
final_metrics = evaluate(valid.user_id, valid.y, valid_scores)

# Produce hidden-test scores without inspecting or using hidden-test labels.
test = load("test")
x_test = add_offsets(make_matrix(test))
test_scores = predict(model, x_test)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print("selected_epoch=%d" % best_epoch, flush=True)
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(final_metrics["primary"]),
            "gauc": float(final_metrics["gauc"]),
            "ndcg@5": float(final_metrics["ndcg@5"]),
            "gpu_seconds": 0.0,
        },
        separators=(",", ":"),
    )
)