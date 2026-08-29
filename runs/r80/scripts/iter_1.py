import os
import time
import copy
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

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def field_offsets():
    offsets = []
    cur = 0
    for name in FIELDS:
        offsets.append(cur)
        cur += int(FEATURE_CARDINALITIES[name])
    return np.asarray(offsets, dtype=np.int64), cur


OFFSETS, TOTAL_CARDINALITY = field_offsets()


def make_matrix(split):
    return np.ascontiguousarray(
        np.column_stack(
            [np.asarray(split.X[name], dtype=np.int64) + OFFSETS[j]
             for j, name in enumerate(FIELDS)]
        ),
        dtype=np.int64,
    )


class FactorizationMachine(nn.Module):
    def __init__(self, cardinality, rank, prevalence):
        super().__init__()
        self.embedding = nn.Embedding(cardinality, rank + 1, sparse=True)
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)

        p = float(np.clip(prevalence, 1e-5, 1.0 - 1e-5))
        self.bias = nn.Parameter(
            torch.tensor(np.log(p / (1.0 - p)), dtype=torch.float32)
        )

    def forward(self, x):
        e = self.embedding(x)
        linear = e[:, :, 0].sum(dim=1)
        v = e[:, :, 1:]
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


def predict(model, x_np):
    model.eval()
    out = np.empty(len(x_np), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(x_np), 32768):
            end = min(start + 32768, len(x_np))
            xb = torch.from_numpy(x_np[start:end])
            out[start:end] = model(xb).cpu().numpy().astype(np.float64)
    return out


def train_one_epoch(model, x_np, y_np, sparse_opt, bias_opt, rng):
    model.train()
    order = rng.permutation(len(y_np))
    loss_sum = 0.0

    for start in range(0, len(order), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        xb = torch.from_numpy(x_np[idx])
        yb = torch.from_numpy(y_np[idx])

        sparse_opt.zero_grad(set_to_none=True)
        bias_opt.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()

        sparse_opt.step()
        bias_opt.step()
        loss_sum += float(loss.detach()) * len(idx)

    return loss_sum / len(y_np)


def fit_with_validation(x_train, y_train, x_valid, y_valid, valid_users):
    model = FactorizationMachine(
        TOTAL_CARDINALITY, K, float(y_train.mean())
    )
    sparse_opt = torch.optim.SparseAdam(
        [model.embedding.weight], lr=LR
    )
    bias_opt = torch.optim.Adam([model.bias], lr=LR)
    rng = np.random.default_rng(SEED)

    best_primary = -np.inf
    best_epoch = 1
    best_state = None
    best_scores = None
    best_metrics = None

    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_one_epoch(
            model, x_train, y_train, sparse_opt, bias_opt, rng
        )
        scores = predict(model, x_valid)
        metrics = evaluate(valid_users, y_valid.astype(np.int8), scores)
        print(
            "epoch=%d loss=%.6f primary=%.6f gauc=%.6f ndcg5=%.6f"
            % (
                epoch,
                loss,
                metrics["primary"],
                metrics["gauc"],
                metrics["ndcg@5"],
            ),
            flush=True,
        )

        if metrics["primary"] > best_primary:
            best_primary = float(metrics["primary"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_scores = scores.copy()
            best_metrics = dict(metrics)

    model.load_state_dict(best_state)
    return model, best_epoch, best_scores, best_metrics


def fit_fixed_epochs(x_train, y_train, epochs):
    torch.manual_seed(SEED)
    model = FactorizationMachine(
        TOTAL_CARDINALITY, K, float(y_train.mean())
    )
    sparse_opt = torch.optim.SparseAdam(
        [model.embedding.weight], lr=LR
    )
    bias_opt = torch.optim.Adam([model.bias], lr=LR)
    rng = np.random.default_rng(SEED)

    for epoch in range(epochs):
        train_one_epoch(
            model, x_train, y_train, sparse_opt, bias_opt, rng
        )
    return model


tr = load("train")
va = load("valid")

x_tr = make_matrix(tr)
x_va = make_matrix(va)
y_tr = np.asarray(tr.y, dtype=np.float32)
y_va_float = np.asarray(va.y, dtype=np.float32)
y_va = np.asarray(va.y, dtype=np.int8)

_, best_epoch, valid_scores, metrics = fit_with_validation(
    x_tr, y_tr, x_va, y_va_float, np.asarray(va.user_id)
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Refit the same recipe for the selected number of epochs on train + validation.
x_combined = np.concatenate([x_tr, x_va], axis=0)
y_combined = np.concatenate([y_tr, y_va_float], axis=0)

del x_tr, x_va, y_tr, y_va_float

test_model = fit_fixed_epochs(x_combined, y_combined, best_epoch)

te = load("test")
x_te = make_matrix(te)
test_scores = predict(test_model, x_te)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(", ", ": "),
    )
)