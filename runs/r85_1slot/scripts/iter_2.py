import os
import gc
import time
import json
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
BATCH_SIZE = 4096
MAX_EPOCHS = 8
MIN_SELECT_EPOCH = 4

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))


def extract(split_name, with_labels=True):
    s = load(split_name)
    x = np.column_stack([s.X[name] for name in FIELDS]).astype(np.int64, copy=False)
    users = np.asarray(s.user_id, dtype=np.int64)
    if with_labels:
        y = np.asarray(s.y, dtype=np.float32)
        del s
        gc.collect()
        return x, y, users
    del s
    gc.collect()
    return x, users


cardinalities = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
offsets_np = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


class FactorizationMachine(nn.Module):
    def __init__(self, total_features, offsets, rank, base_logit):
        super().__init__()
        self.linear = nn.Embedding(total_features, 1)
        self.factors = nn.Embedding(total_features, rank)
        self.register_buffer("offsets", torch.as_tensor(offsets, dtype=torch.long))
        self.register_buffer(
            "base_logit", torch.tensor(float(base_logit), dtype=torch.float32)
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)

    def forward(self, x):
        ids = x + self.offsets
        linear_term = self.linear(ids).squeeze(-1).sum(dim=1)
        v = self.factors(ids)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.base_logit + linear_term + interaction


@torch.no_grad()
def predict(model, x_np, batch_size=32768):
    model.eval()
    result = np.empty(x_np.shape[0], dtype=np.float32)
    for start in range(0, x_np.shape[0], batch_size):
        end = min(start + batch_size, x_np.shape[0])
        xb = torch.from_numpy(x_np[start:end])
        result[start:end] = model(xb).cpu().numpy()
    return result


def make_model(y, seed):
    torch.manual_seed(seed)
    p = float(np.clip(np.mean(y), 1e-5, 1.0 - 1e-5))
    base_logit = np.log(p / (1.0 - p))
    return FactorizationMachine(
        total_features=total_cardinality,
        offsets=offsets_np,
        rank=K,
        base_logit=base_logit,
    )


def train_with_validation(x_train, y_train, x_valid, y_valid, valid_users):
    model = make_model(y_train, SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(y_train)

    best_primary = -np.inf
    best_scores = None
    best_metrics = None
    best_epoch = MAX_EPOCHS

    model.train()
    for epoch in range(1, MAX_EPOCHS + 1):
        permutation = torch.randperm(x_tensor.shape[0], generator=generator)
        model.train()

        for start in range(0, x_tensor.shape[0], BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            logits = model(x_tensor[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y_tensor[idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if epoch >= MIN_SELECT_EPOCH:
            scores = predict(model, x_valid)
            metrics = evaluate(valid_users, y_valid.astype(np.int8), scores)
            if metrics["primary"] > best_primary:
                best_primary = float(metrics["primary"])
                best_scores = scores.copy()
                best_metrics = metrics
                best_epoch = epoch

    return model, best_scores, best_metrics, best_epoch


def fit_fixed_epochs(x, y, epochs):
    model = make_model(y, SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)

    for _ in range(epochs):
        permutation = torch.randperm(x_tensor.shape[0], generator=generator)
        model.train()
        for start in range(0, x_tensor.shape[0], BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            logits = model(x_tensor[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y_tensor[idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


x_train, y_train, _ = extract("train", with_labels=True)
x_valid, y_valid, valid_users = extract("valid", with_labels=True)

valid_model, valid_scores, metrics, selected_epochs = train_with_validation(
    x_train, y_train, x_valid, y_valid, valid_users
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

del valid_model
gc.collect()

x_refit = np.concatenate([x_train, x_valid], axis=0)
y_refit = np.concatenate([y_train, y_valid], axis=0)
del x_train, y_train, x_valid, y_valid
gc.collect()

test_model = fit_fixed_epochs(x_refit, y_refit, selected_epochs)
del x_refit, y_refit
gc.collect()

x_test, _ = extract("test", with_labels=False)
test_scores = predict(test_model, x_test)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS " + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)