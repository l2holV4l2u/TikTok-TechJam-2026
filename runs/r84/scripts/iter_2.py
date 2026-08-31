import os
import time
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
LR = 0.001
BATCH_SIZE = 4096
MAX_EPOCHS = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))
device = torch.device("cpu")

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets_np = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_x(split):
    x = np.column_stack([split.X[f] for f in FIELDS]).astype(np.int64, copy=False)
    x = x + offsets_np[None, :]
    return np.ascontiguousarray(x)


class FactorizationMachine(nn.Module):
    def __init__(self, n_features, rank):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1)
        self.embedding = nn.Embedding(n_features, rank)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear_term + interaction


def train_one_epoch(model, optimizer, x, y, generator):
    model.train()
    n = x.shape[0]
    order = torch.randperm(n, generator=generator)
    for start in range(0, n, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        xb = x[idx]
        yb = y[idx]
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def predict(model, x):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, x.shape[0])
        result[start:end] = model(x[start:end]).cpu().numpy().astype(np.float64)
    return result


# Train-only fit and validation evaluation.
train = load("train")
valid = load("valid")

x_train = torch.from_numpy(make_x(train)).to(device)
y_train_np = np.asarray(train.y, dtype=np.float32)
y_train = torch.from_numpy(y_train_np).to(device)
x_valid = torch.from_numpy(make_x(valid)).to(device)
y_valid_np = np.asarray(valid.y, dtype=np.int8)

torch.manual_seed(SEED)
model = FactorizationMachine(total_cardinality, K).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
generator = torch.Generator(device="cpu")
generator.manual_seed(SEED)

best_primary = -np.inf
best_epoch = 1
best_scores = None
best_metrics = None
best_state = None
epochs_without_gain = 0

for epoch in range(1, MAX_EPOCHS + 1):
    train_one_epoch(model, optimizer, x_train, y_train, generator)
    valid_scores_epoch = predict(model, x_valid)
    metrics_epoch = evaluate(valid.user_id, y_valid_np, valid_scores_epoch)

    if metrics_epoch["primary"] > best_primary:
        best_primary = float(metrics_epoch["primary"])
        best_epoch = epoch
        best_scores = valid_scores_epoch.copy()
        best_metrics = metrics_epoch
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    if metrics_epoch["primary"] > best_primary - 1e-12:
        epochs_without_gain = 0
    else:
        epochs_without_gain += 1

# Restore the exact train-only model whose validation scores are reported.
model.load_state_dict(best_state)
valid_scores = best_scores
metrics = best_metrics

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Refit the identical recipe for the selected epoch count on train + validation.
x_valid_np = x_valid.numpy()
x_combined_np = np.concatenate([x_train.numpy(), x_valid_np], axis=0)
y_combined_np = np.concatenate(
    [y_train_np, y_valid_np.astype(np.float32)], axis=0
)
x_combined = torch.from_numpy(np.ascontiguousarray(x_combined_np)).to(device)
y_combined = torch.from_numpy(np.ascontiguousarray(y_combined_np)).to(device)

del model, optimizer, x_train, y_train, x_valid, x_valid_np, x_combined_np
torch.manual_seed(SEED)
final_model = FactorizationMachine(total_cardinality, K).to(device)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=LR)
final_generator = torch.Generator(device="cpu")
final_generator.manual_seed(SEED)

for _ in range(best_epoch):
    train_one_epoch(
        final_model, final_optimizer, x_combined, y_combined, final_generator
    )

test = load("test")
x_test = torch.from_numpy(make_x(test)).to(device)
test_scores = predict(final_model, x_test)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.6f}'
    % (
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        float(elapsed),
    )
)