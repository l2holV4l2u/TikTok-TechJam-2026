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
EMBED_DIM = 16
LR = 0.001
BATCH_SIZE = 4096
MAX_EPOCHS = 10


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
set_seed(SEED)

field_cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + field_cards[:-1], dtype=np.int64)
total_cardinality = int(sum(field_cards))


def encode(split):
    cols = []
    for j, name in enumerate(FIELDS):
        x = np.asarray(split.X[name], dtype=np.int64)
        cols.append(x + offsets[j])
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


class FactorizationMachine(nn.Module):
    def __init__(self, n_features, dim, initial_rate):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1, sparse=True)
        self.factors = nn.Embedding(n_features, dim, sparse=True)
        self.bias = nn.Parameter(
            torch.tensor(np.log(initial_rate / (1.0 - initial_rate)), dtype=torch.float32)
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).squeeze(-1).sum(dim=1)
        v = self.factors(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear_term + interaction


def make_model(label_rate):
    model = FactorizationMachine(total_cardinality, EMBED_DIM, label_rate)
    sparse_optimizer = torch.optim.SparseAdam(
        [model.linear.weight, model.factors.weight], lr=LR
    )
    dense_optimizer = torch.optim.Adam([model.bias], lr=LR)
    return model, sparse_optimizer, dense_optimizer


def train_one_epoch(model, sparse_optimizer, dense_optimizer, x, y, rng):
    model.train()
    n = len(y)
    permutation = rng.permutation(n)

    total_loss = 0.0
    total_rows = 0
    for start in range(0, n, BATCH_SIZE):
        idx_np = permutation[start:start + BATCH_SIZE]
        idx = torch.from_numpy(idx_np)
        xb = x.index_select(0, idx)
        yb = y.index_select(0, idx)

        sparse_optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()

        sparse_optimizer.step()
        dense_optimizer.step()

        batch_n = len(idx_np)
        total_loss += float(loss.detach()) * batch_n
        total_rows += batch_n

    return total_loss / total_rows


@torch.no_grad()
def predict(model, x):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)
    prediction_batch = 32768
    for start in range(0, x.shape[0], prediction_batch):
        end = min(start + prediction_batch, x.shape[0])
        logits = model(x[start:end])
        result[start:end] = logits.cpu().numpy().astype(np.float64, copy=False)
    return result


# Train-only fit and validation evaluation.
tr = load("train")
va = load("valid")

x_train_np = encode(tr)
x_valid_np = encode(va)
y_train_np = np.asarray(tr.y, dtype=np.float32)
y_valid_np = np.asarray(va.y, dtype=np.int8)

x_train = torch.from_numpy(x_train_np)
x_valid = torch.from_numpy(x_valid_np)
y_train = torch.from_numpy(y_train_np)

model, sparse_optimizer, dense_optimizer = make_model(float(y_train_np.mean()))
rng = np.random.default_rng(SEED)

best_epoch = 1
best_primary = -np.inf
best_metrics = None
best_valid_scores = None

for epoch in range(1, MAX_EPOCHS + 1):
    loss = train_one_epoch(
        model, sparse_optimizer, dense_optimizer, x_train, y_train, rng
    )
    epoch_scores = predict(model, x_valid)
    epoch_metrics = evaluate(va.user_id, y_valid_np, epoch_scores)

    print(
        "epoch=%d loss=%.6f primary=%.6f gauc=%.6f ndcg5=%.6f"
        % (
            epoch,
            loss,
            epoch_metrics["primary"],
            epoch_metrics["gauc"],
            epoch_metrics["ndcg@5"],
        ),
        flush=True,
    )

    if epoch_metrics["primary"] > best_primary:
        best_epoch = epoch
        best_primary = float(epoch_metrics["primary"])
        best_metrics = epoch_metrics
        best_valid_scores = epoch_scores.copy()

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )

# Release the train-only model before the train+validation refit.
del model, sparse_optimizer, dense_optimizer
del x_train, x_valid
del x_train_np, x_valid_np, y_train
torch.cuda.empty_cache()

# Refit the identical recipe on train + validation for the hidden test.
set_seed(SEED)

x_combined_np = np.ascontiguousarray(
    np.column_stack(
        [
            np.concatenate(
                [
                    np.asarray(tr.X[name], dtype=np.int64),
                    np.asarray(va.X[name], dtype=np.int64),
                ]
            )
            + offsets[j]
            for j, name in enumerate(FIELDS)
        ]
    ),
    dtype=np.int64,
)
y_combined_np = np.concatenate(
    [
        np.asarray(tr.y, dtype=np.float32),
        np.asarray(va.y, dtype=np.float32),
    ]
)

x_combined = torch.from_numpy(x_combined_np)
y_combined = torch.from_numpy(y_combined_np)

final_model, sparse_optimizer, dense_optimizer = make_model(
    float(y_combined_np.mean())
)
refit_rng = np.random.default_rng(SEED)

for epoch in range(1, best_epoch + 1):
    refit_loss = train_one_epoch(
        final_model,
        sparse_optimizer,
        dense_optimizer,
        x_combined,
        y_combined,
        refit_rng,
    )
    print(
        "refit_epoch=%d/%d loss=%.6f" % (epoch, best_epoch, refit_loss),
        flush=True,
    )

te = load("test")
x_test_np = encode(te)
x_test = torch.from_numpy(x_test_np)
test_scores = predict(final_model, x_test)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.6f}'
    % (
        float(best_metrics["primary"]),
        float(best_metrics["gauc"]),
        float(best_metrics["ndcg@5"]),
        elapsed,
    )
)