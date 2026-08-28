import os
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
LR = 0.001
BATCH_SIZE = 4096
EPOCHS = 12

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))


def make_matrix(split):
    return np.stack(
        [np.asarray(split.X[name], dtype=np.int64) for name in FIELDS],
        axis=1,
    )


cards = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))


def add_offsets(x):
    return x + offsets[None, :]


def make_metric_aligned_weights(user_ids, labels):
    """
    Construct a 50/50 blend of:
      1. Macro-user weighting: every user has equal total training weight,
         corresponding to the macro aggregation used by nDCG.
      2. GAUC weighting: every eligible user's total weight is proportional
         to its positive count, matching the organizer's GAUC aggregation.

    Each component is separately normalized to mean row weight one before
    blending, so the optimizer's effective learning-rate scale is preserved.
    """
    user_ids = np.asarray(user_ids)
    labels = np.asarray(labels, dtype=np.float64)
    n_rows = len(labels)

    _, inverse = np.unique(user_ids, return_inverse=True)
    n_users = int(inverse.max()) + 1

    user_count = np.bincount(inverse, minlength=n_users).astype(np.float64)
    user_pos = np.bincount(
        inverse, weights=labels, minlength=n_users
    ).astype(np.float64)

    macro_weight = (
        float(n_rows) / (float(n_users) * user_count[inverse])
    )

    eligible = (user_pos > 0.0) & (user_pos < user_count)
    eligible_positive_total = float(user_pos[eligible].sum())

    gauc_user_mass = np.zeros(n_users, dtype=np.float64)
    if eligible_positive_total > 0.0:
        gauc_user_mass[eligible] = (
            float(n_rows) * user_pos[eligible] / eligible_positive_total
        )
    gauc_weight = gauc_user_mass[inverse] / user_count[inverse]

    weights = 0.5 * macro_weight + 0.5 * gauc_weight
    weights /= weights.mean()

    return weights.astype(np.float32), {
        "users": n_users,
        "eligible_users": int(eligible.sum()),
        "min": float(weights.min()),
        "median": float(np.median(weights)),
        "p99": float(np.quantile(weights, 0.99)),
        "max": float(weights.max()),
    }


class FactorizationMachine(nn.Module):
    def __init__(self, n_tokens, rank):
        super().__init__()
        # Column 0 is the linear coefficient; remaining columns are factors.
        self.embedding = nn.Embedding(n_tokens, rank + 1, sparse=True)
        self.bias = nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)

    def forward(self, x):
        z = self.embedding(x)
        linear = z[:, :, 0].sum(dim=1)
        factors = z[:, :, 1:]
        summed = factors.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - factors.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


def predict(model, x_np, batch_size=32768):
    model.eval()
    result = np.empty(x_np.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, x_np.shape[0], batch_size):
            end = min(start + batch_size, x_np.shape[0])
            xb = torch.from_numpy(x_np[start:end])
            result[start:end] = (
                model(xb).cpu().numpy().astype(np.float64)
            )
    return result


train = load("train")
valid = load("valid")

x_train = add_offsets(make_matrix(train))
x_valid = add_offsets(make_matrix(valid))
y_train = np.asarray(train.y, dtype=np.float32)

train_weights, weight_stats = make_metric_aligned_weights(
    train.user_id, train.y
)
print(
    "FINDINGS metric_weight_users=%d eligible=%d "
    "min=%.6f median=%.6f p99=%.6f max=%.6f"
    % (
        weight_stats["users"],
        weight_stats["eligible_users"],
        weight_stats["min"],
        weight_stats["median"],
        weight_stats["p99"],
        weight_stats["max"],
    ),
    flush=True,
)

model = FactorizationMachine(total_cardinality, K)
sparse_optimizer = torch.optim.SparseAdam(
    [model.embedding.weight], lr=LR
)
bias_optimizer = torch.optim.Adam([model.bias], lr=LR)

n_train = len(y_train)
best_primary = -np.inf
best_state = None
best_epoch = -1

rng = np.random.default_rng(SEED)

for epoch in range(EPOCHS):
    model.train()
    permutation = rng.permutation(n_train)

    total_loss = 0.0
    total_weight = 0.0

    for start in range(0, n_train, BATCH_SIZE):
        idx = permutation[start:start + BATCH_SIZE]
        xb = torch.from_numpy(x_train[idx])
        yb = torch.from_numpy(y_train[idx])
        wb = torch.from_numpy(train_weights[idx])

        sparse_optimizer.zero_grad(set_to_none=True)
        bias_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        row_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, yb, reduction="none"
        )
        loss = (row_loss * wb).sum() / wb.sum().clamp_min(1e-12)
        loss.backward()

        sparse_optimizer.step()
        bias_optimizer.step()

        total_loss += float((row_loss.detach() * wb).sum())
        total_weight += float(wb.sum())

    valid_scores = predict(model, x_valid)
    metrics = evaluate(valid.user_id, valid.y, valid_scores)

    print(
        "epoch=%d weighted_loss=%.6f primary=%.6f "
        "gauc=%.6f ndcg5=%.6f"
        % (
            epoch + 1,
            total_loss / max(total_weight, 1e-12),
            float(metrics["primary"]),
            float(metrics["gauc"]),
            float(metrics["ndcg@5"]),
        ),
        flush=True,
    )

    if float(metrics["primary"]) > best_primary:
        best_primary = float(metrics["primary"])
        best_epoch = epoch + 1
        best_state = {
            "embedding": model.embedding.weight.detach().clone(),
            "bias": model.bias.detach().clone(),
        }

with torch.no_grad():
    model.embedding.weight.copy_(best_state["embedding"])
    model.bias.copy_(best_state["bias"])

valid_scores = predict(model, x_valid)
final_metrics = evaluate(valid.user_id, valid.y, valid_scores)

# Produce hidden-test scores without reading or using hidden-test labels.
test = load("test")
x_test = add_offsets(make_matrix(test))
test_scores = predict(model, x_test)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print("selected_epoch=%d" % best_epoch, flush=True)
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(final_metrics["primary"]),
            "gauc": float(final_metrics["gauc"]),
            "ndcg@5": float(final_metrics["ndcg@5"]),
            "gpu_seconds": 0.0,
        },
        separators=(",", ":"),
    )
)