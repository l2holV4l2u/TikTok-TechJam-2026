import os
import gc
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2022
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
LR = 0.001
EPOCHS = 8
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 65536

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))


def field_array(split, name):
    if name in split.X:
        return split.X[name]
    if name == "user_id":
        return split.user_id
    if name == "video_id":
        return split.video_id
    raise KeyError("Required baseline field not found: %s" % name)


cardinalities = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
offsets_np = np.zeros(len(FIELDS), dtype=np.int64)
offsets_np[1:] = np.cumsum(np.asarray(cardinalities[:-1], dtype=np.int64))
total_cardinality = int(sum(cardinalities))


class FactorizationMachine(nn.Module):
    def __init__(self, total_features, rank, initial_bias):
        super().__init__()
        # Column zero stores first-order weights; remaining columns are FM factors.
        self.embedding = nn.Embedding(total_features, rank + 1, sparse=True)
        self.bias = nn.Parameter(torch.tensor(float(initial_bias), dtype=torch.float32))

        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)

    def forward(self, ids):
        e = self.embedding(ids)
        linear = e[:, :, 0].sum(dim=1)
        v = e[:, :, 1:]
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


def make_feature_matrix(split):
    # Shape is [n_rows, 5]. Offsets make IDs from different fields disjoint.
    cols = []
    for j, name in enumerate(FIELDS):
        a = np.asarray(field_array(split, name), dtype=np.int64)
        cols.append(a + offsets_np[j])
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


def predict(model, split):
    model.eval()
    n = len(split.y)
    result = np.empty(n, dtype=np.float32)
    offsets = torch.from_numpy(offsets_np)

    # Construct each prediction batch directly from the source arrays to avoid
    # retaining another multi-million-row feature matrix.
    arrays = [
        np.asarray(field_array(split, name), dtype=np.int64)
        for name in FIELDS
    ]

    with torch.inference_mode():
        for start in range(0, n, PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, n)
            batch_np = np.empty((end - start, len(FIELDS)), dtype=np.int64)
            for j, a in enumerate(arrays):
                batch_np[:, j] = a[start:end]
            ids = torch.from_numpy(batch_np)
            ids.add_(offsets)
            result[start:end] = model(ids).cpu().numpy()
    return result


train = load("train")
x_train_np = make_feature_matrix(train)
y_train_np = np.asarray(train.y, dtype=np.float32)

x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)

positive_rate = float(y_train_np.mean())
initial_bias = math.log(positive_rate / (1.0 - positive_rate))

model = FactorizationMachine(total_cardinality, K, initial_bias)

# SparseAdam updates only category rows touched by a batch. The scalar bias is
# optimized separately because SparseAdam accepts sparse parameters only.
embedding_optimizer = torch.optim.SparseAdam(
    [model.embedding.weight], lr=LR
)
bias_optimizer = torch.optim.Adam([model.bias], lr=LR)

n_train = y_train.shape[0]
generator = torch.Generator()
generator.manual_seed(SEED)

model.train()
for epoch in range(EPOCHS):
    permutation = torch.randperm(n_train, generator=generator)

    loss_sum = 0.0
    rows_seen = 0

    for start in range(0, n_train, BATCH_SIZE):
        idx = permutation[start:min(start + BATCH_SIZE, n_train)]
        ids = x_train.index_select(0, idx)
        labels = y_train.index_select(0, idx)

        embedding_optimizer.zero_grad(set_to_none=True)
        bias_optimizer.zero_grad(set_to_none=True)

        logits = model(ids)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()

        embedding_optimizer.step()
        bias_optimizer.step()

        b = labels.numel()
        loss_sum += float(loss.detach()) * b
        rows_seen += b

    print(
        "epoch=%d train_logloss=%.6f bias=%.6f"
        % (epoch + 1, loss_sum / rows_seen, float(model.bias.detach())),
        flush=True,
    )

# Optimizer moments are no longer required and can be much larger than the FM.
del embedding_optimizer, bias_optimizer, x_train, y_train
del x_train_np, y_train_np, permutation
del train
gc.collect()

valid = load("valid")
valid_scores = predict(model, valid)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    test = load("test")
    test_scores = predict(model, test)
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    del test, test_scores
    gc.collect()

print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": 0.0,
        },
        separators=(",", ":"),
    )
)