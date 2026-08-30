import os
import time
import json
import copy
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
MAX_EPOCHS = 12

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))


cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_matrix(split):
    cols = []
    for j, field in enumerate(FIELDS):
        col = np.asarray(split.X[field], dtype=np.int64)
        cols.append(col + offsets[j])
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


class FactorizationMachine(nn.Module):
    def __init__(self, n_categories, embedding_dim, initial_rate):
        super().__init__()
        self.linear = nn.Embedding(n_categories, 1)
        self.embedding = nn.Embedding(n_categories, embedding_dim)
        initial_rate = float(np.clip(initial_rate, 1e-5, 1.0 - 1e-5))
        self.bias = nn.Parameter(
            torch.tensor(np.log(initial_rate / (1.0 - initial_rate)), dtype=torch.float32)
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(1)
        latent = self.embedding(x)
        summed = latent.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - latent.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear_term + interaction


@torch.inference_mode()
def predict(model, X, batch_size=16384):
    model.eval()
    xt = torch.from_numpy(X)
    result = np.empty(len(X), dtype=np.float32)
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        result[start:end] = model(xt[start:end]).cpu().numpy()
    return result


def fit_model(X, y, epochs, initial_rate, valid_data=None):
    torch.manual_seed(SEED)
    model = FactorizationMachine(total_cardinality, K, initial_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    xt = torch.from_numpy(X)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(SEED)

    best_primary = -np.inf
    best_metrics = None
    best_state = None
    best_epoch = epochs

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(X), generator=generator)

        for start in range(0, len(X), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(xt[idx])
            loss = loss_fn(logits, yt[idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if valid_data is not None:
            X_valid, y_valid, user_valid = valid_data
            valid_scores = predict(model, X_valid)
            metrics = evaluate(user_valid, y_valid, valid_scores)
            primary = float(metrics["primary"])

            if primary > best_primary:
                best_primary = primary
                best_metrics = metrics
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_epoch, best_metrics


# Train-only validation model.
train = load("train")
valid = load("valid")

X_train = make_matrix(train)
X_valid = make_matrix(valid)
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)

validation_model, selected_epochs, _ = fit_model(
    X_train,
    y_train,
    MAX_EPOCHS,
    float(y_train.mean()),
    valid_data=(X_valid, y_valid, valid_users),
)

valid_scores = predict(validation_model, X_valid).astype(np.float64)
metrics = evaluate(valid_users, y_valid, valid_scores)

# Refit the identical recipe for the selected number of epochs on train + validation.
X_combined = np.concatenate([X_train, X_valid], axis=0)
y_combined = np.concatenate(
    [y_train, y_valid.astype(np.float32)], axis=0
)

del validation_model
del X_train
del X_valid

final_model, _, _ = fit_model(
    X_combined,
    y_combined,
    selected_epochs,
    float(y_combined.mean()),
    valid_data=None,
)

test = load("test")
X_test = make_matrix(test)
test_scores = predict(final_model, X_test).astype(np.float64)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "scores_valid.npy"), valid_scores)
    np.save(os.path.join(out, "scores_test.npy"), test_scores)

wall_time = time.time() - START_TIME
print(
    "METRICS " +
    json.dumps({
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(wall_time),
    })
)