import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2024

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1], dtype=np.int64)
TOTAL_CARDINALITY = int(sum(CARDS))


def encoded_features(split):
    cols = []
    for field, offset, card in zip(FIELDS, OFFSETS, CARDS):
        x = np.asarray(split.X[field], dtype=np.int64)
        if x.min() < 0 or x.max() >= card:
            raise ValueError(
                f"{field} has ids outside [0, {card}): "
                f"min={x.min()}, max={x.max()}"
            )
        cols.append(x + offset)
    return torch.from_numpy(np.ascontiguousarray(np.stack(cols, axis=1)))


class FactorizationMachine(nn.Module):
    def __init__(self, n_features, rank):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1, sparse=True)
        self.latent = nn.Embedding(n_features, rank, sparse=True)
        self.bias = nn.Parameter(torch.zeros(1))

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
def predict(model, x, batch_size=65536):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float32)
    for begin in range(0, x.shape[0], batch_size):
        end = min(begin + batch_size, x.shape[0])
        result[begin:end] = model(x[begin:end]).cpu().numpy()
    return result


train = load("train")
valid = load("valid")

x_train = encoded_features(train)
y_train = torch.from_numpy(
    np.ascontiguousarray(np.asarray(train.y, dtype=np.float32))
)
x_valid = encoded_features(valid)
valid_users = np.asarray(valid.user_id)
valid_labels = np.asarray(valid.y)

model = FactorizationMachine(TOTAL_CARDINALITY, rank=16)

# The two embedding tables have sparse gradients, while the scalar intercept
# uses ordinary Adam.
sparse_optimizer = torch.optim.SparseAdam(
    [model.linear.weight, model.latent.weight], lr=0.001
)
bias_optimizer = torch.optim.Adam([model.bias], lr=0.001)
criterion = nn.BCEWithLogitsLoss()

batch_size = 4096
max_epochs = 18
generator = torch.Generator()
generator.manual_seed(SEED)

best_primary = -np.inf
best_metrics = None
best_scores = None
best_state = None
epochs_without_gain = 0

for epoch in range(max_epochs):
    model.train()
    order = torch.randperm(x_train.shape[0], generator=generator)

    for begin in range(0, x_train.shape[0], batch_size):
        idx = order[begin:begin + batch_size]
        xb = x_train[idx]
        yb = y_train[idx]

        sparse_optimizer.zero_grad(set_to_none=True)
        bias_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()

        sparse_optimizer.step()
        bias_optimizer.step()

    valid_scores_epoch = predict(model, x_valid)
    metrics_epoch = evaluate(valid_users, valid_labels, valid_scores_epoch)
    primary_epoch = float(metrics_epoch["primary"])

    if primary_epoch > best_primary:
        best_primary = primary_epoch
        best_metrics = metrics_epoch
        best_scores = valid_scores_epoch.copy()
        best_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        epochs_without_gain = 0
    else:
        epochs_without_gain += 1

    # Permit the baseline to converge, while avoiding unnecessary late
    # overfitting once several complete passes cease helping.
    if epoch >= 7 and epochs_without_gain >= 4:
        break

if best_state is None:
    raise RuntimeError("Training produced no valid checkpoint")

model.load_state_dict(best_state)
valid_scores = best_scores

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

test = load("test")
x_test = encoded_features(test)
test_scores = predict(model, x_test)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

artifacts = os.environ.get("RUN_ARTIFACTS")
if artifacts:
    os.makedirs(artifacts, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "fields": FIELDS,
            "cardinalities": CARDS,
            "offsets": OFFSETS,
            "rank": 16,
            "learning_rate": 0.001,
            "seed": SEED,
            "validation_metrics": {
                "primary": float(best_metrics["primary"]),
                "gauc": float(best_metrics["gauc"]),
                "ndcg@5": float(best_metrics["ndcg@5"]),
            },
        },
        os.path.join(artifacts, "official_fm_k16_seed2024.pt"),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)