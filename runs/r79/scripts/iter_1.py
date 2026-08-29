import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
LR = 0.001
BATCH_SIZE = 4096
MAX_EPOCHS = 8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, max(1, os.cpu_count() or 1)))


def make_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([split.X[name] for name in FIELDS]),
        dtype=np.int64,
    )


CARDINALITIES = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
OFFSETS = np.cumsum([0] + CARDINALITIES[:-1], dtype=np.int64)
TOTAL_CARDINALITY = int(sum(CARDINALITIES))
OFFSETS_T = torch.from_numpy(OFFSETS)


class FactorizationMachine(nn.Module):
    def __init__(self, total_cardinality, embedding_dim):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1, sparse=True)
        self.latent = nn.Embedding(
            total_cardinality, embedding_dim, sparse=True
        )
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, mean=0.0, std=0.01)

    def forward(self, local_ids):
        global_ids = local_ids + OFFSETS_T
        linear_term = self.linear(global_ids).squeeze(-1).sum(dim=1)

        v = self.latent(global_ids)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)

        return self.bias + linear_term + interaction


def predict(model, x_np, batch_size=65536):
    model.eval()
    x = torch.from_numpy(x_np)
    out = np.empty(len(x_np), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(x_np), batch_size):
            stop = min(start + batch_size, len(x_np))
            out[start:stop] = (
                model(x[start:stop]).detach().cpu().numpy().astype(np.float64)
            )
    return out


def train_one_epoch(model, x, y, sparse_optimizer, bias_optimizer, generator):
    model.train()
    order = torch.randperm(len(x), generator=generator)
    total_loss = 0.0

    for start in range(0, len(x), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        xb = x[idx]
        yb = y[idx]

        sparse_optimizer.zero_grad(set_to_none=True)
        bias_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()

        sparse_optimizer.step()
        bias_optimizer.step()
        total_loss += float(loss.detach()) * len(idx)

    return total_loss / len(x)


def new_model_and_optimizers(seed):
    torch.manual_seed(seed)
    model = FactorizationMachine(TOTAL_CARDINALITY, K)
    sparse_optimizer = torch.optim.SparseAdam(
        [model.linear.weight, model.latent.weight], lr=LR
    )
    bias_optimizer = torch.optim.Adam([model.bias], lr=LR)
    return model, sparse_optimizer, bias_optimizer


train = load("train")
valid = load("valid")

x_train_np = make_matrix(train)
x_valid_np = make_matrix(valid)
y_train_np = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)

model, sparse_opt, bias_opt = new_model_and_optimizers(SEED)
generator = torch.Generator()
generator.manual_seed(SEED + 1)

best_primary = -np.inf
best_epoch = 1
best_state = None
best_valid_scores = None
best_metrics = None

for epoch in range(1, MAX_EPOCHS + 1):
    loss = train_one_epoch(
        model, x_train, y_train, sparse_opt, bias_opt, generator
    )
    epoch_scores = predict(model, x_valid_np)
    epoch_metrics = evaluate(valid.user_id, y_valid, epoch_scores)

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
        best_primary = float(epoch_metrics["primary"])
        best_epoch = epoch
        best_metrics = epoch_metrics
        best_valid_scores = epoch_scores.copy()
        best_state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

model.load_state_dict(best_state)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )

# Refit the identical selected recipe on train + validation for test scoring.
x_combined_np = np.ascontiguousarray(
    np.concatenate([x_train_np, x_valid_np], axis=0),
    dtype=np.int64,
)
y_combined_np = np.ascontiguousarray(
    np.concatenate(
        [y_train_np, np.asarray(valid.y, dtype=np.float32)], axis=0
    ),
    dtype=np.float32,
)

del x_train, y_train, model, sparse_opt, bias_opt
x_combined = torch.from_numpy(x_combined_np)
y_combined = torch.from_numpy(y_combined_np)

final_model, final_sparse_opt, final_bias_opt = new_model_and_optimizers(SEED)
final_generator = torch.Generator()
final_generator.manual_seed(SEED + 1)

for epoch in range(1, best_epoch + 1):
    refit_loss = train_one_epoch(
        final_model,
        x_combined,
        y_combined,
        final_sparse_opt,
        final_bias_opt,
        final_generator,
    )
    print(
        "refit_epoch=%d/%d loss=%.6f"
        % (epoch, best_epoch, refit_loss),
        flush=True,
    )

test = load("test")
x_test_np = make_matrix(test)
test_scores = predict(final_model, x_test_np)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
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