import os
import json
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


torch.set_num_threads(min(8, os.cpu_count() or 1))
torch.manual_seed(2024)
np.random.seed(2024)

FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
CARDINALITIES = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]

offsets = np.zeros(len(FIELDS), dtype=np.int64)
offsets[1:] = np.cumsum(CARDINALITIES[:-1], dtype=np.int64)
TOTAL_CARDINALITY = int(sum(CARDINALITIES))


def make_features(split):
    x = np.column_stack([split.X[name] for name in FIELDS]).astype(np.int64, copy=False)
    x = x + offsets[None, :]
    return torch.from_numpy(np.ascontiguousarray(x))


class FactorizationMachine(nn.Module):
    def __init__(self, num_categories, num_factors=16):
        super().__init__()
        self.linear = nn.Embedding(num_categories, 1, sparse=True)
        self.embedding = nn.Embedding(num_categories, num_factors, sparse=True)
        self.bias = nn.Parameter(torch.zeros(()))

        nn.init.zeros_(self.linear.weight)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1) - v.square().sum(dim=(1, 2))
        )
        return self.bias + linear_term + interaction


@torch.inference_mode()
def predict(model, x, batch_size=65536):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        result[start:end] = model(x[start:end]).double().numpy()
    return result


train = load("train")
valid = load("valid")

x_train = make_features(train)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))
x_valid = make_features(valid)
y_valid_np = np.asarray(valid.y)

model = FactorizationMachine(TOTAL_CARDINALITY, num_factors=16)

sparse_optimizer = torch.optim.SparseAdam(
    [model.linear.weight, model.embedding.weight], lr=0.001
)
bias_optimizer = torch.optim.Adam([model.bias], lr=0.001)

batch_size = 8192
num_epochs = 10
generator = torch.Generator()
generator.manual_seed(2024)

best_primary = -np.inf
best_metrics = None
best_state = None

model.train()
n = x_train.shape[0]

for epoch in range(num_epochs):
    permutation = torch.randperm(n, generator=generator)

    for start in range(0, n, batch_size):
        idx = permutation[start:start + batch_size]
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
    metrics = evaluate(valid.user_id, y_valid_np, valid_scores)
    primary = float(metrics["primary"])

    if primary > best_primary:
        best_primary = primary
        best_metrics = {k: float(v) for k, v in metrics.items()}
        best_state = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }

model.load_state_dict(best_state)
valid_scores = predict(model, x_valid)
best_metrics = {
    k: float(v)
    for k, v in evaluate(valid.user_id, y_valid_np, valid_scores).items()
}

test = load("test")
x_test = make_features(test)
test_scores = predict(model, x_test)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": 0.0,
        },
        separators=(",", ":"),
    )
)