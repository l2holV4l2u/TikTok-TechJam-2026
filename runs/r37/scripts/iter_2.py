import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
LR = 0.001
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 32768
EPOCHS = 8

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))


def make_offsets():
    offsets = {}
    cur = 0
    for name in FIELDS:
        offsets[name] = cur
        cur += int(FEATURE_CARDINALITIES[name])
    return offsets, cur


OFFSETS, TOTAL_CARDINALITY = make_offsets()


def make_matrix(split):
    cols = [
        np.asarray(split.X[name], dtype=np.int64) + OFFSETS[name]
        for name in FIELDS
    ]
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


class FactorizationMachine(nn.Module):
    def __init__(self, n_categories, rank):
        super().__init__()
        self.embedding = nn.Embedding(
            n_categories, rank + 1, sparse=True
        )
        self.bias = nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)

    def forward(self, x):
        e = self.embedding(x)
        linear = e[:, :, 0].sum(dim=1)
        v = e[:, :, 1:]
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


@torch.no_grad()
def predict(model, x_np):
    model.eval()
    n = len(x_np)
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, n)
        xb = torch.from_numpy(x_np[start:end])
        out[start:end] = model(xb).cpu().numpy().astype(np.float64)
    return out


train = load("train")
valid = load("valid")

x_train_np = make_matrix(train)
x_valid_np = make_matrix(valid)
y_train_np = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y)

# Metric-aligned activity weighting: interpolate between ordinary row weighting
# and equal-user weighting using the inverse square root of user frequency.
train_user_ids = np.asarray(train.X["user_id"], dtype=np.int64)
user_counts = np.bincount(
    train_user_ids,
    minlength=int(FEATURE_CARDINALITIES["user_id"]),
).astype(np.float32)

row_counts = user_counts[train_user_ids]
mean_user_count = float(len(train_user_ids)) / float(np.count_nonzero(user_counts))
sample_weights_np = np.sqrt(mean_user_count / np.maximum(row_counts, 1.0))
sample_weights_np = np.clip(sample_weights_np, 0.25, 4.0)
sample_weights_np /= sample_weights_np.mean()
sample_weights_np = np.asarray(sample_weights_np, dtype=np.float32)

print(
    "FINDINGS user_weight_min=%.4f median=%.4f max=%.4f"
    % (
        float(sample_weights_np.min()),
        float(np.median(sample_weights_np)),
        float(sample_weights_np.max()),
    ),
    flush=True,
)

x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)
sample_weights = torch.from_numpy(sample_weights_np)

model = FactorizationMachine(TOTAL_CARDINALITY, K)

embedding_optimizer = torch.optim.SparseAdam(
    [model.embedding.weight], lr=LR
)
bias_optimizer = torch.optim.Adam([model.bias], lr=LR)

generator = torch.Generator()
generator.manual_seed(SEED)

best_primary = -np.inf
best_state = None
best_metrics = None

n_train = len(y_train_np)

for epoch in range(EPOCHS):
    model.train()
    order = torch.randperm(n_train, generator=generator)
    total_loss = 0.0

    for start in range(0, n_train, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        xb = x_train[idx]
        yb = y_train[idx]
        wb = sample_weights[idx]

        embedding_optimizer.zero_grad(set_to_none=True)
        bias_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        per_row_loss = F.binary_cross_entropy_with_logits(
            logits, yb, reduction="none"
        )
        loss = (per_row_loss * wb).sum() / wb.sum().clamp_min(1e-12)
        loss.backward()

        embedding_optimizer.step()
        bias_optimizer.step()
        total_loss += float(loss.detach()) * len(idx)

    valid_scores = predict(model, x_valid_np)
    metrics = evaluate(valid.user_id, y_valid, valid_scores)
    primary = float(metrics["primary"])

    print(
        "epoch=%d loss=%.6f primary=%.6f gauc=%.6f ndcg5=%.6f"
        % (
            epoch + 1,
            total_loss / n_train,
            primary,
            float(metrics["gauc"]),
            float(metrics["ndcg@5"]),
        ),
        flush=True,
    )

    if primary > best_primary:
        best_primary = primary
        best_metrics = metrics
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

model.load_state_dict(best_state)
valid_scores = predict(model, x_valid_np)
final_metrics = evaluate(valid.user_id, y_valid, valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    test = load("test")
    x_test_np = make_matrix(test)
    test_scores = predict(model, x_test_np)
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

result = {
    "primary": float(final_metrics["primary"]),
    "gauc": float(final_metrics["gauc"]),
    "ndcg@5": float(final_metrics["ndcg@5"]),
    "gpu_seconds": 0.0,
}
print("METRICS " + json.dumps(result, separators=(",", ":")))