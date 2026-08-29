import os
import time
import json
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2024
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
]
K = 16
LR = 0.001
BATCH_SIZE = 4096
MAX_EPOCHS = 12
PATIENCE = 3

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_matrix(split):
    x = np.empty((len(split.user_id), len(FIELDS)), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:, j] = split.X[field]
        x[:, j] += offsets[j]
    return x


class FactorizationMachine(nn.Module):
    def __init__(self, num_categories, rank, initial_logit=0.0):
        super().__init__()
        self.linear = nn.Embedding(num_categories, 1)
        self.factors = nn.Embedding(num_categories, rank)
        self.bias = nn.Parameter(
            torch.tensor(float(initial_logit), dtype=torch.float32)
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)
        factors = self.factors(x)
        summed = factors.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - factors.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear_term + interaction


def predict(model, x, batch_size=32768):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            xb = torch.from_numpy(x[start:end])
            result[start:end] = model(xb).cpu().numpy()
    return result


def fit_one_epoch(model, optimizer, criterion, x, y, rng):
    model.train()
    order = rng.permutation(x.shape[0])
    total_loss = 0.0

    for start in range(0, x.shape[0], BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        xb = torch.from_numpy(x[idx])
        yb = torch.from_numpy(y[idx])

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach()) * len(idx)

    return total_loss / x.shape[0]


train = load("train")
valid = load("valid")

x_train = make_matrix(train)
y_train = np.asarray(train.y, dtype=np.float32)
x_valid = make_matrix(valid)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

positive_rate = float(y_train.mean())
initial_logit = np.log(
    np.clip(positive_rate, 1e-6, 1.0 - 1e-6)
    / np.clip(1.0 - positive_rate, 1e-6, 1.0 - 1e-6)
)

model = FactorizationMachine(total_cardinality, K, initial_logit)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()
rng = np.random.default_rng(SEED)

best_primary = -np.inf
best_epoch = 0
best_state = None
best_valid_scores = None
best_metrics = None
epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):
    fit_one_epoch(model, optimizer, criterion, x_train, y_train, rng)

    epoch_scores = predict(model, x_valid)
    epoch_metrics = evaluate(valid_users, y_valid, epoch_scores)
    primary = float(epoch_metrics["primary"])

    if primary > best_primary:
        best_primary = primary
        best_epoch = epoch
        best_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        best_valid_scores = epoch_scores.copy()
        best_metrics = epoch_metrics
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        break

model.load_state_dict(best_state)
valid_scores = np.asarray(best_valid_scores, dtype=np.float64)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "scores_valid.npy"), valid_scores)

# Refit the identical selected recipe on train plus validation for test scoring.
x_combined = np.concatenate((x_train, x_valid), axis=0)
y_combined = np.concatenate((y_train, y_valid.astype(np.float32)), axis=0)

del model, optimizer, x_train, x_valid
del train, valid

combined_rate = float(y_combined.mean())
combined_logit = np.log(
    np.clip(combined_rate, 1e-6, 1.0 - 1e-6)
    / np.clip(1.0 - combined_rate, 1e-6, 1.0 - 1e-6)
)

torch.manual_seed(SEED)
final_model = FactorizationMachine(total_cardinality, K, combined_logit)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=LR)
final_rng = np.random.default_rng(SEED)

for _ in range(best_epoch):
    fit_one_epoch(
        final_model,
        final_optimizer,
        criterion,
        x_combined,
        y_combined,
        final_rng,
    )

del x_combined, y_combined

test = load("test")
x_test = make_matrix(test)
test_scores = predict(final_model, x_test).astype(np.float64)

if out_dir:
    np.save(os.path.join(out_dir, "scores_test.npy"), test_scores)

elapsed = time.time() - START
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result))