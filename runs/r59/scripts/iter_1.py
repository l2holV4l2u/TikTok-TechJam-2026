import os
import time
import json
import copy
import random

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
BATCH_SIZE = 1024
EPOCHS = 5
PRED_BATCH_SIZE = 32768


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_offsets():
    cardinalities = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
    offsets = np.zeros(len(cardinalities), dtype=np.int64)
    if len(cardinalities) > 1:
        offsets[1:] = np.cumsum(cardinalities[:-1], dtype=np.int64)
    return offsets, int(sum(cardinalities))


def encode_split(split, offsets):
    cols = [
        np.asarray(split.X[name], dtype=np.int64) + offsets[j]
        for j, name in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


class FactorizationMachine(nn.Module):
    def __init__(self, num_categories, embed_dim):
        super().__init__()
        self.linear = nn.Embedding(num_categories, 1)
        self.embedding = nn.Embedding(num_categories, embed_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)

        v = self.embedding(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1) -
            v.square().sum(dim=(1, 2))
        )
        return self.bias + linear_term + interaction


@torch.no_grad()
def predict(model, x_np, device):
    model.eval()
    result = np.empty(x_np.shape[0], dtype=np.float64)
    for start in range(0, x_np.shape[0], PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, x_np.shape[0])
        xb = torch.from_numpy(x_np[start:end]).to(device=device)
        logits = model(xb)
        result[start:end] = logits.detach().cpu().numpy().astype(np.float64)
    return result


def main():
    start_time = time.perf_counter()
    seed_everything(SEED)

    requested_device = os.environ["AGENT_DEVICE"].lower()
    device = torch.device(requested_device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))))

    train = load("train")
    valid = load("valid")

    offsets, num_categories = make_offsets()
    x_train_np = encode_split(train, offsets)
    x_valid_np = encode_split(valid, offsets)
    y_train_np = np.ascontiguousarray(train.y, dtype=np.float32)

    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(y_train_np)

    model = FactorizationMachine(num_categories, EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    rng = np.random.default_rng(SEED)
    best_primary = -np.inf
    best_state = None

    n_train = x_train.shape[0]

    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(n_train)
        total_loss = 0.0

        for start in range(0, n_train, BATCH_SIZE):
            idx_np = order[start:start + BATCH_SIZE]
            idx = torch.from_numpy(idx_np)

            xb = x_train.index_select(0, idx).to(device=device)
            yb = y_train.index_select(0, idx).to(device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach()) * len(idx_np)

        valid_scores_epoch = predict(model, x_valid_np, device)
        metrics_epoch = evaluate(valid.user_id, valid.y, valid_scores_epoch)
        print(
            "epoch=%d loss=%.6f primary=%.6f gauc=%.6f ndcg@5=%.6f"
            % (
                epoch + 1,
                total_loss / n_train,
                metrics_epoch["primary"],
                metrics_epoch["gauc"],
                metrics_epoch["ndcg@5"],
            ),
            flush=True,
        )

        if metrics_epoch["primary"] > best_primary:
            best_primary = float(metrics_epoch["primary"])
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    valid_scores = predict(model, x_valid_np, device)
    metrics = evaluate(valid.user_id, valid.y, valid_scores)

    out_dir = os.environ.get("ITER_OUT")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(
            os.path.join(out_dir, "scores_valid.npy"),
            np.asarray(valid_scores, dtype=np.float64),
        )

    del x_train, y_train, x_train_np, y_train_np, x_valid_np
    del train

    test = load("test")
    x_test_np = encode_split(test, offsets)
    test_scores = predict(model, x_test_np, device)

    if out_dir:
        np.save(
            os.path.join(out_dir, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start_time
    final = {
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
        "device": device.type,
    }
    print("METRICS " + json.dumps(final, separators=(", ", ": ")))


if __name__ == "__main__":
    main()