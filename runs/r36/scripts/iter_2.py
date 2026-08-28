import os
import json
import random
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
BATCH_SIZE = 8192
EPOCHS = 12

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_matrix(split):
    x = np.stack([split.X[f] for f in FIELDS], axis=1).astype(
        np.int64, copy=False
    )
    x = x + offsets[None, :]
    return np.ascontiguousarray(x)


class FactorizationMachine(nn.Module):
    def __init__(self, n_values, rank):
        super().__init__()
        self.linear = nn.Embedding(n_values, 1, sparse=True)
        self.factors = nn.Embedding(n_values, rank, sparse=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)

        with torch.no_grad():
            for off in offsets:
                self.linear.weight[int(off)].zero_()
                self.factors.weight[int(off)].zero_()

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.factors(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return linear_term + interaction


@torch.inference_mode()
def predict(model, x_np, batch_size=32768):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float64)
    for start in range(0, len(x_np), batch_size):
        end = min(start + batch_size, len(x_np))
        xb = torch.from_numpy(x_np[start:end])
        result[start:end] = (
            model(xb).cpu().numpy().astype(np.float64, copy=False)
        )
    return result


train = load("train")
x_train = make_matrix(train)
y_train = np.asarray(train.y, dtype=np.float32).copy()
train_user_ids = np.asarray(train.X["user_id"], dtype=np.int64)

# Construct a convex blend of the two metric aggregation schemes.
#
# Equal-user component:
#   each user's total training weight is constant, corresponding to nDCG's
#   unweighted mean over users.
#
# Positive-count component:
#   each user's total training weight is proportional to its number of
#   positives, corresponding to GAUC's positive-count weighting.
user_cardinality = int(FEATURE_CARDINALITIES["user_id"])
user_impressions = np.bincount(
    train_user_ids, minlength=user_cardinality
).astype(np.float64)
user_positives = np.bincount(
    train_user_ids,
    weights=y_train.astype(np.float64),
    minlength=user_cardinality,
)

n_users_seen = int(np.count_nonzero(user_impressions))
mean_impressions_per_user = len(y_train) / max(n_users_seen, 1)
label_rate = float(y_train.mean())

row_n = user_impressions[train_user_ids]
row_p = user_positives[train_user_ids]

equal_user_weight = mean_impressions_per_user / np.maximum(row_n, 1.0)
gauc_user_weight = (
    row_p / np.maximum(row_n, 1.0) / max(label_rate, 1e-8)
)
train_weights = 0.5 * equal_user_weight + 0.5 * gauc_user_weight

# Very short histories can otherwise receive extreme noisy gradients.
train_weights = np.clip(train_weights, 0.1, 10.0)
train_weights /= train_weights.mean()
train_weights = train_weights.astype(np.float32)

print(
    "FINDINGS "
    f"metric_weight_mean={train_weights.mean():.6f} "
    f"metric_weight_q10={np.quantile(train_weights, 0.10):.4f} "
    f"metric_weight_q50={np.quantile(train_weights, 0.50):.4f} "
    f"metric_weight_q90={np.quantile(train_weights, 0.90):.4f} "
    f"metric_weight_q99={np.quantile(train_weights, 0.99):.4f}",
    flush=True,
)

del train, train_user_ids
del user_impressions, user_positives, row_n, row_p
del equal_user_weight, gauc_user_weight

valid = load("valid")
x_valid = make_matrix(valid)
y_valid = np.asarray(valid.y, dtype=np.int8).copy()
valid_users = np.asarray(valid.user_id, dtype=np.int64).copy()
del valid

x_train_t = torch.from_numpy(x_train)
y_train_t = torch.from_numpy(y_train)
train_weights_t = torch.from_numpy(train_weights)

model = FactorizationMachine(total_cardinality, K)
optimizer = torch.optim.SparseAdam(model.parameters(), lr=LR)

best_primary = -np.inf
best_metrics = None
best_state = None
best_epoch = -1

n_train = len(x_train)
rng = np.random.default_rng(SEED)

for epoch in range(1, EPOCHS + 1):
    model.train()
    permutation = rng.permutation(n_train)

    for start in range(0, n_train, BATCH_SIZE):
        idx_np = permutation[start:start + BATCH_SIZE]
        idx = torch.from_numpy(idx_np)

        xb = x_train_t.index_select(0, idx)
        yb = y_train_t.index_select(0, idx)
        wb = train_weights_t.index_select(0, idx)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        row_loss = F.binary_cross_entropy_with_logits(
            logits, yb, reduction="none"
        )
        loss = (row_loss * wb).sum() / wb.sum().clamp_min(1e-8)
        loss.backward()
        optimizer.step()

    valid_scores = predict(model, x_valid)
    epoch_metrics = evaluate(valid_users, y_valid, valid_scores)
    print(
        f"epoch={epoch} primary={epoch_metrics['primary']:.6f} "
        f"gauc={epoch_metrics['gauc']:.6f} "
        f"ndcg@5={epoch_metrics['ndcg@5']:.6f}",
        flush=True,
    )

    if float(epoch_metrics["primary"]) > best_primary:
        best_primary = float(epoch_metrics["primary"])
        best_metrics = epoch_metrics
        best_epoch = epoch
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

model.load_state_dict(best_state)
valid_scores = predict(model, x_valid)
final_metrics = evaluate(valid_users, y_valid, valid_scores)
print(f"selected_epoch={best_epoch}", flush=True)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    test = load("test")
    x_test = make_matrix(test)
    del test
    test_scores = predict(model, x_test)
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

payload = {
    "primary": float(final_metrics["primary"]),
    "gauc": float(final_metrics["gauc"]),
    "ndcg@5": float(final_metrics["ndcg@5"]),
    "gpu_seconds": 0.0,
}
print("METRICS " + json.dumps(payload))