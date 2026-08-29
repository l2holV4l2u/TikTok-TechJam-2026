import os
import time
import math
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
EPOCHS = 5
BATCH_SIZE = 4096

torch.set_num_threads(min(16, os.cpu_count() or 1))
torch.manual_seed(SEED)
np.random.seed(SEED)


def make_matrix(split):
    n = len(split.user_id)
    x = np.empty((n, len(FIELDS)), dtype=np.int64)
    offset = 0
    for j, name in enumerate(FIELDS):
        x[:, j] = split.X[name] + offset
        offset += int(FEATURE_CARDINALITIES[name])
    return x


TOTAL_CARDINALITY = sum(int(FEATURE_CARDINALITIES[name]) for name in FIELDS)


class FactorizationMachine(nn.Module):
    def __init__(self, cardinality, rank, intercept):
        super().__init__()
        # The first coordinate is the linear term; the rest are FM factors.
        self.embedding = nn.Embedding(cardinality, rank + 1, sparse=True)
        self.register_buffer("intercept", torch.tensor(float(intercept),
                                                        dtype=torch.float32))
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)

    def forward(self, x):
        e = self.embedding(x)
        linear = e[:, :, 0].sum(dim=1)
        v = e[:, :, 1:]
        summed = v.sum(dim=1)
        interactions = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.intercept + linear + interactions


def fit_fm(x_np, y_np, seed):
    torch.manual_seed(seed)
    n = x_np.shape[0]
    prevalence = float(np.mean(y_np))
    prevalence = min(max(prevalence, 1e-6), 1.0 - 1e-6)
    intercept = math.log(prevalence / (1.0 - prevalence))

    x = torch.from_numpy(np.ascontiguousarray(x_np, dtype=np.int64))
    y = torch.from_numpy(np.ascontiguousarray(y_np, dtype=np.float32))

    model = FactorizationMachine(TOTAL_CARDINALITY, K, intercept)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=LR)
    generator = torch.Generator()
    generator.manual_seed(seed + 7919)

    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(x[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


@torch.no_grad()
def predict(model, x_np):
    model.eval()
    n = x_np.shape[0]
    scores = np.empty(n, dtype=np.float64)
    x = torch.from_numpy(np.ascontiguousarray(x_np, dtype=np.int64))
    for start in range(0, n, 32768):
        end = min(start + 32768, n)
        scores[start:end] = model(x[start:end]).cpu().numpy().astype(np.float64)
    return scores


# Train-only fit used for the reported validation metric.
train = load("train")
valid = load("valid")

x_train = make_matrix(train)
x_valid = make_matrix(valid)
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

valid_model = fit_fm(x_train, y_train, SEED)
valid_scores = predict(valid_model, x_valid)
metrics = evaluate(valid.user_id, y_valid, valid_scores)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Refit the identical recipe on train + validation for the hidden test.
x_joint = np.concatenate((x_train, x_valid), axis=0)
y_joint = np.concatenate(
    (y_train, np.asarray(valid.y, dtype=np.float32)), axis=0
)

del valid_model, x_train, x_valid, y_train
joint_model = fit_fm(x_joint, y_joint, SEED)
del x_joint, y_joint

test = load("test")
x_test = make_matrix(test)
test_scores = predict(joint_model, x_test)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
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