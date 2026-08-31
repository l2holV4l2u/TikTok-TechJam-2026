import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
LR = 0.001
BATCH_SIZE = 2048
EPOCHS = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))


cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_matrix(split):
    cols = []
    for field, offset, cardinality in zip(FIELDS, offsets, cardinalities):
        x = np.asarray(split.X[field], dtype=np.int64)
        if x.size and (x.min() < 0 or x.max() >= cardinality):
            raise ValueError(
                f"{field} contains IDs outside [0, {cardinality - 1}]"
            )
        cols.append(x + offset)
    return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.int64)


class FactorizationMachine(nn.Module):
    def __init__(self, n_features, rank):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1, sparse=True)
        self.latent = nn.Embedding(n_features, rank, sparse=True)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).squeeze(-1).sum(dim=1)

        v = self.latent(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)

        return self.bias + linear_term + interaction


train = load("train")
x_train = make_matrix(train)
y_train = np.asarray(train.y, dtype=np.float32)

model = FactorizationMachine(total_cardinality, K)
embedding_optimizer = torch.optim.SparseAdam(
    [model.linear.weight, model.latent.weight], lr=LR
)
bias_optimizer = torch.optim.Adam([model.bias], lr=LR)

n_train = len(y_train)
rng = np.random.default_rng(SEED)
model.train()

for epoch in range(EPOCHS):
    order = rng.permutation(n_train)

    for start in range(0, n_train, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        xb = torch.from_numpy(x_train[idx])
        yb = torch.from_numpy(y_train[idx])

        embedding_optimizer.zero_grad(set_to_none=True)
        bias_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()

        embedding_optimizer.step()
        bias_optimizer.step()


@torch.inference_mode()
def predict_matrix(x):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)

    for start in range(0, x.shape[0], 16384):
        end = min(start + 16384, x.shape[0])
        xb = torch.from_numpy(x[start:end])
        result[start:end] = model(xb).cpu().numpy().astype(np.float64)

    return result


valid = load("valid")
x_valid = make_matrix(valid)
valid_scores = predict_matrix(x_valid)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

test = load("test")
x_test = make_matrix(test)
test_scores = predict_matrix(x_test)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.3f}'
    % (
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        elapsed,
    )
)