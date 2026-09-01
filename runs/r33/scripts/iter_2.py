import os
import json
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
EMBED_DIM = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 2048
EPOCHS = 12

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets_np = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_features(split):
    x = np.column_stack(
        [np.asarray(split.X[f], dtype=np.int64) for f in FIELDS]
    )
    x += offsets_np[None, :]
    return torch.from_numpy(np.ascontiguousarray(x))


class FactorizationMachine(nn.Module):
    def __init__(self, n_categories, dim, zero_indices):
        super().__init__()
        self.linear = nn.Embedding(n_categories, 1, sparse=True)
        self.factors = nn.Embedding(n_categories, dim, sparse=True)
        self.bias = nn.Parameter(torch.zeros(()))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)

        with torch.no_grad():
            self.linear.weight[zero_indices].zero_()
            self.factors.weight[zero_indices].zero_()

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.factors(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear_term + interaction


@torch.inference_mode()
def predict(model, x, batch_size=16384):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        logits = model(x[start:end])
        result[start:end] = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
    return result


train = load("train")
valid = load("valid")

x_train = make_features(train)
y_train = torch.from_numpy(
    np.ascontiguousarray(np.asarray(train.y, dtype=np.float32))
)
x_valid = make_features(valid)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)

zero_indices = torch.tensor(offsets_np, dtype=torch.long)
model = FactorizationMachine(
    total_cardinality,
    EMBED_DIM,
    zero_indices,
)

sparse_optimizer = torch.optim.SparseAdam(
    [model.linear.weight, model.factors.weight],
    lr=LEARNING_RATE,
)
bias_optimizer = torch.optim.Adam(
    [model.bias],
    lr=LEARNING_RATE,
)

best_primary = -np.inf
best_metrics = None
best_state = None
n_train = x_train.shape[0]

for epoch in range(EPOCHS):
    model.train()

    generator = torch.Generator()
    generator.manual_seed(SEED + epoch)
    order = torch.randperm(n_train, generator=generator)

    for start in range(0, n_train, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        xb = x_train[idx]
        yb = y_train[idx]

        sparse_optimizer.zero_grad(set_to_none=True)
        bias_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()

        sparse_optimizer.step()
        bias_optimizer.step()

    valid_scores = predict(model, x_valid)
    metrics = evaluate(valid_users, y_valid, valid_scores)

    if float(metrics["primary"]) > best_primary:
        best_primary = float(metrics["primary"])
        best_metrics = {
            key: float(value)
            for key, value in metrics.items()
        }
        best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)

final_valid_scores = predict(model, x_valid)
best_metrics = {
    key: float(value)
    for key, value in evaluate(
        valid_users,
        y_valid,
        final_valid_scores,
    ).items()
}

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    test = load("test")
    x_test = make_features(test)
    test_scores = predict(model, x_test)
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print(
    "METRICS "
    + json.dumps(
        {
            "primary": best_metrics["primary"],
            "gauc": best_metrics["gauc"],
            "ndcg@5": best_metrics["ndcg@5"],
            "gpu_seconds": 0.0,
        }
    )
)