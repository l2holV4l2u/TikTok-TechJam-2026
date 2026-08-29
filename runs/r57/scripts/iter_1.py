import os
import time
import json
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
EMBED_DIM = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 65536
EPOCHS = 12


def set_reproducible(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_matrix(split, offsets):
    cols = []
    for field, offset in zip(FIELDS, offsets):
        col = np.asarray(split.X[field], dtype=np.int64)
        cols.append(col + np.int64(offset))
    return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.int64)


class FactorizationMachine(nn.Module):
    def __init__(self, total_cardinality, embedding_dim):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        first_order = self.linear(x).sum(dim=1).squeeze(-1)

        v = self.embedding(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)

        return self.bias + first_order + interaction


@torch.no_grad()
def predict(model, x_np, device):
    model.eval()
    n = x_np.shape[0]
    result = np.empty(n, dtype=np.float32)

    for start in range(0, n, PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, n)
        xb = torch.from_numpy(x_np[start:end]).to(device=device)
        logits = model(xb)
        result[start:end] = torch.sigmoid(logits).cpu().numpy()

    return result


def main():
    wall_start = time.time()

    requested_device = os.environ.get("AGENT_DEVICE", "cpu").lower()
    if requested_device != "cpu":
        raise RuntimeError(
            "This benchmark run is CPU-selected; refusing to use a non-CPU device."
        )
    device = torch.device("cpu")

    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(min(8, cpu_count))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    set_reproducible(SEED)

    cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
    offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
    total_cardinality = int(sum(cardinalities))

    train = load("train")
    valid = load("valid")

    x_train_np = make_matrix(train, offsets)
    y_train_np = np.ascontiguousarray(train.y.astype(np.float32, copy=False))
    x_valid_np = make_matrix(valid, offsets)

    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(y_train_np)

    model = FactorizationMachine(total_cardinality, EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)

    best_primary = -np.inf
    best_epoch = -1
    best_state = None
    best_valid_scores = None
    best_metrics = None

    n_train = x_train.shape[0]

    for epoch in range(1, EPOCHS + 1):
        model.train()
        permutation = torch.randperm(n_train, generator=generator)
        running_loss = 0.0
        seen = 0

        for start in range(0, n_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_train)
            idx = permutation[start:end]

            xb = x_train[idx].to(device=device)
            yb = y_train[idx].to(device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()

            batch_n = end - start
            running_loss += float(loss.detach()) * batch_n
            seen += batch_n

        valid_scores = predict(model, x_valid_np, device)
        metrics = evaluate(valid.user_id, valid.y, valid_scores)
        epoch_primary = float(metrics["primary"])

        print(
            "epoch={} loss={:.6f} primary={:.6f} gauc={:.6f} ndcg@5={:.6f}".format(
                epoch,
                running_loss / max(seen, 1),
                epoch_primary,
                float(metrics["gauc"]),
                float(metrics["ndcg@5"]),
            ),
            flush=True,
        )

        if epoch_primary > best_primary:
            best_primary = epoch_primary
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_valid_scores = valid_scores.copy()
            best_metrics = {
                "primary": float(metrics["primary"]),
                "gauc": float(metrics["gauc"]),
                "ndcg@5": float(metrics["ndcg@5"]),
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # Recompute from the pinned best checkpoint so saved validation and test
    # predictions are generated through the identical inference path.
    final_valid_scores = predict(model, x_valid_np, device)
    final_metrics = evaluate(valid.user_id, valid.y, final_valid_scores)

    print(
        "selected_epoch={} primary={:.6f}".format(
            best_epoch, float(final_metrics["primary"])
        ),
        flush=True,
    )

    out_dir = os.environ.get("ITER_OUT")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(
            os.path.join(out_dir, "scores_valid.npy"),
            np.asarray(final_valid_scores, dtype=np.float64),
        )

    # Test labels are never accessed or used for fitting/model selection.
    test = load("test")
    x_test_np = make_matrix(test, offsets)
    test_scores = predict(model, x_test_np, device)

    if out_dir:
        np.save(
            os.path.join(out_dir, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

    elapsed = time.time() - wall_start
    result = {
        "primary": float(final_metrics["primary"]),
        "gauc": float(final_metrics["gauc"]),
        "ndcg@5": float(final_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
        "device": "cpu",
    }
    print("METRICS " + json.dumps(result, separators=(", ", ": ")))


if __name__ == "__main__":
    main()