import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2024

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
NUMERIC_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1], dtype=np.int64)
TOTAL_CARDINALITY = int(sum(CARDS))


def encoded_features(split):
    cols = []
    for field, offset, card in zip(FIELDS, OFFSETS, CARDS):
        x = np.asarray(split.X[field], dtype=np.int64)
        if x.min() < 0 or x.max() >= card:
            raise ValueError(
                f"{field} has ids outside [0, {card}): "
                f"min={x.min()}, max={x.max()}"
            )
        cols.append(x + offset)
    return torch.from_numpy(
        np.ascontiguousarray(np.stack(cols, axis=1))
    )


def raw_dense_sources(split_name, split):
    sources = {}

    for name in NUMERIC_FIELDS:
        sources["num:" + name] = np.asarray(split.num[name], dtype=np.float32)

    video_hist = historical_features(split_name, key="video_id")
    author_hist = historical_features(split_name, key="author_id")

    for name in sorted(video_hist):
        arr = np.asarray(video_hist[name], dtype=np.float32)
        if arr.ndim == 1 and len(arr) == len(split.user_id):
            sources["video_hist:" + name] = arr

    for name in sorted(author_hist):
        arr = np.asarray(author_hist[name], dtype=np.float32)
        if arr.ndim == 1 and len(arr) == len(split.user_id):
            sources["author_hist:" + name] = arr

    return sources


def fit_dense_preprocessor(sources):
    specs = []
    for name in sorted(sources):
        x = np.asarray(sources[name], dtype=np.float64)
        finite = np.isfinite(x)

        if finite.any():
            observed = x[finite]
            median = float(np.median(observed))
            minimum = float(np.min(observed))
            q001, q999 = np.quantile(observed, [0.001, 0.999])
            use_log1p = bool(minimum >= 0.0 and q999 > 10.0)
        else:
            median = 0.0
            q001 = 0.0
            q999 = 0.0
            use_log1p = False

        filled = np.where(finite, x, median)
        filled = np.clip(filled, q001, q999)
        if use_log1p:
            filled = np.log1p(np.maximum(filled, 0.0))

        mean = float(filled.mean())
        std = float(filled.std())
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0

        specs.append(
            {
                "name": name,
                "median": median,
                "low": float(q001),
                "high": float(q999),
                "log1p": use_log1p,
                "mean": mean,
                "std": std,
                "missing_indicator": bool((~finite).any()),
            }
        )
    return specs


def transform_dense(sources, specs):
    columns = []

    for spec in specs:
        name = spec["name"]
        if name not in sources:
            raise KeyError(f"Missing dense feature {name}")

        x = np.asarray(sources[name], dtype=np.float64)
        missing = ~np.isfinite(x)

        z = np.where(missing, x * 0.0 + spec["median"], x)
        z = np.clip(z, spec["low"], spec["high"])

        if spec["log1p"]:
            z = np.log1p(np.maximum(z, 0.0))

        z = (z - spec["mean"]) / spec["std"]
        z = np.clip(z, -8.0, 8.0)
        columns.append(z.astype(np.float32))

        if spec["missing_indicator"]:
            columns.append(missing.astype(np.float32))

    if not columns:
        return torch.empty((len(next(iter(sources.values()))), 0), dtype=torch.float32)

    matrix = np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.float32)
    return torch.from_numpy(matrix)


class FactorizationMachineWithDenseHead(nn.Module):
    def __init__(self, n_features, rank, n_dense):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1, sparse=True)
        self.latent = nn.Embedding(n_features, rank, sparse=True)
        self.dense_linear = nn.Linear(n_dense, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.dense_linear.weight)

    def forward(self, x, dense):
        linear_term = self.linear(x).sum(dim=1).squeeze(1)

        v = self.latent(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)

        dense_term = self.dense_linear(dense).squeeze(1)
        return self.bias + linear_term + interaction + dense_term


@torch.no_grad()
def predict(model, x, dense, batch_size=65536):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float32)

    for begin in range(0, x.shape[0], batch_size):
        end = min(begin + batch_size, x.shape[0])
        result[begin:end] = model(
            x[begin:end], dense[begin:end]
        ).cpu().numpy()

    return result


train = load("train")
valid = load("valid")

x_train = encoded_features(train)
x_valid = encoded_features(valid)

train_dense_sources = raw_dense_sources("train", train)
dense_specs = fit_dense_preprocessor(train_dense_sources)
dense_train = transform_dense(train_dense_sources, dense_specs)
del train_dense_sources

valid_dense_sources = raw_dense_sources("valid", valid)
dense_valid = transform_dense(valid_dense_sources, dense_specs)
del valid_dense_sources

y_train = torch.from_numpy(
    np.ascontiguousarray(np.asarray(train.y, dtype=np.float32))
)
valid_users = np.asarray(valid.user_id)
valid_labels = np.asarray(valid.y)

model = FactorizationMachineWithDenseHead(
    TOTAL_CARDINALITY,
    rank=16,
    n_dense=dense_train.shape[1],
)

sparse_optimizer = torch.optim.SparseAdam(
    [model.linear.weight, model.latent.weight],
    lr=0.001,
)
dense_optimizer = torch.optim.Adam(
    [
        {"params": [model.bias], "lr": 0.001},
        {"params": model.dense_linear.parameters(), "lr": 0.01},
    ]
)
criterion = nn.BCEWithLogitsLoss()

batch_size = 4096
max_epochs = 18
generator = torch.Generator()
generator.manual_seed(SEED)

best_primary = -np.inf
best_metrics = None
best_scores = None
best_state = None
epochs_without_gain = 0

for epoch in range(max_epochs):
    model.train()
    order = torch.randperm(x_train.shape[0], generator=generator)

    for begin in range(0, x_train.shape[0], batch_size):
        idx = order[begin:begin + batch_size]
        xb = x_train[idx]
        db = dense_train[idx]
        yb = y_train[idx]

        sparse_optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)

        logits = model(xb, db)
        loss = criterion(logits, yb)
        loss.backward()

        sparse_optimizer.step()
        dense_optimizer.step()

    valid_scores_epoch = predict(model, x_valid, dense_valid)
    metrics_epoch = evaluate(
        valid_users,
        valid_labels,
        valid_scores_epoch,
    )
    primary_epoch = float(metrics_epoch["primary"])

    if primary_epoch > best_primary:
        best_primary = primary_epoch
        best_metrics = metrics_epoch
        best_scores = valid_scores_epoch.copy()
        best_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        epochs_without_gain = 0
    else:
        epochs_without_gain += 1

    if epoch >= 7 and epochs_without_gain >= 4:
        break

if best_state is None:
    raise RuntimeError("Training produced no valid checkpoint")

model.load_state_dict(best_state)
valid_scores = best_scores

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

test = load("test")
x_test = encoded_features(test)
test_dense_sources = raw_dense_sources("test", test)
dense_test = transform_dense(test_dense_sources, dense_specs)
del test_dense_sources

test_scores = predict(model, x_test, dense_test)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

artifacts = os.environ.get("RUN_ARTIFACTS")
if artifacts:
    os.makedirs(artifacts, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "fields": FIELDS,
            "cardinalities": CARDS,
            "offsets": OFFSETS,
            "rank": 16,
            "dense_specs": dense_specs,
            "dense_dimension": int(dense_train.shape[1]),
            "seed": SEED,
            "validation_metrics": {
                "primary": float(best_metrics["primary"]),
                "gauc": float(best_metrics["gauc"]),
                "ndcg@5": float(best_metrics["ndcg@5"]),
            },
        },
        os.path.join(artifacts, "fm_k16_dense_history_seed2024.pt"),
    )

elapsed = time.time() - START
print(
    "FINDINGS "
    + json.dumps(
        {
            "dense_dimension": int(dense_train.shape[1]),
            "epochs_trained": int(epoch + 1),
            "best_primary": float(best_primary),
        }
    )
)
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)