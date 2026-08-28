import os
import json
import math
import random
import numpy as np
import torch
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
RANK = 16
LR = 0.001
BATCH_SIZE = 1024
EPOCHS = 7


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_indices(split, offsets):
    cols = []
    for name, offset in zip(FIELDS, offsets):
        x = np.asarray(split.X[name], dtype=np.int64)
        cols.append(x + np.int64(offset))
    x = np.stack(cols, axis=1)
    return torch.from_numpy(x)


class FactorizationMachine(torch.nn.Module):
    def __init__(self, num_categories, rank, initial_bias):
        super().__init__()
        # Column zero is the linear coefficient; remaining columns are factors.
        self.embedding = torch.nn.Embedding(
            num_categories, rank + 1, sparse=True
        )
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)

        # A constant bias affects optimization calibration but not within-user ranking.
        self.register_buffer(
            "intercept", torch.tensor(float(initial_bias), dtype=torch.float32)
        )

    def forward(self, x):
        z = self.embedding(x)
        linear = z[:, :, 0].sum(dim=1)
        factors = z[:, :, 1:]
        summed = factors.sum(dim=1)
        interactions = 0.5 * (
            summed.square().sum(dim=1)
            - factors.square().sum(dim=(1, 2))
        )
        return self.intercept + linear + interactions


@torch.no_grad()
def predict(model, x, batch_size=32768):
    model.eval()
    out = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        scores = model(x[start:end]).cpu().numpy()
        out[start:end] = scores.astype(np.float64, copy=False)
    return out


def main():
    seed_everything(SEED)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    train = load("train")
    valid = load("valid")

    offsets = []
    total_categories = 0
    for field in FIELDS:
        offsets.append(total_categories)
        total_categories += int(FEATURE_CARDINALITIES[field])

    x_train = make_indices(train, offsets)
    x_valid = make_indices(valid, offsets)
    y_train = torch.from_numpy(
        np.asarray(train.y, dtype=np.float32)
    )

    positive_rate = float(np.clip(y_train.numpy().mean(), 1e-6, 1.0 - 1e-6))
    initial_bias = math.log(positive_rate / (1.0 - positive_rate))

    model = FactorizationMachine(total_categories, RANK, initial_bias)
    optimizer = torch.optim.SparseAdam(model.embedding.parameters(), lr=LR)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    best_primary = -np.inf
    best_metrics = None
    best_weights = None

    n = x_train.shape[0]
    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        loss_sum = 0.0

        for start in range(0, n, BATCH_SIZE):
            ids = permutation[start:start + BATCH_SIZE]
            xb = x_train[ids]
            yb = y_train[ids]

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss) * len(ids)

        valid_scores = predict(model, x_valid)
        metrics = evaluate(
            np.asarray(valid.user_id),
            np.asarray(valid.y),
            valid_scores,
        )
        print(
            "epoch=%d loss=%.6f primary=%.6f gauc=%.6f ndcg5=%.6f"
            % (
                epoch + 1,
                loss_sum / n,
                metrics["primary"],
                metrics["gauc"],
                metrics["ndcg@5"],
            ),
            flush=True,
        )

        if metrics["primary"] > best_primary:
            best_primary = float(metrics["primary"])
            best_metrics = dict(metrics)
            best_weights = model.embedding.weight.detach().cpu().clone()

    with torch.no_grad():
        model.embedding.weight.copy_(best_weights)

    # Recompute validation scores from exactly the selected checkpoint.
    valid_scores = predict(model, x_valid)
    best_metrics = evaluate(
        np.asarray(valid.user_id),
        np.asarray(valid.y),
        valid_scores,
    )

    out = os.environ.get("ITER_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        test = load("test")
        x_test = make_indices(test, offsets)
        test_scores = predict(model, x_test)
        np.save(
            os.path.join(out, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

    result = {
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": 0.0,
    }
    print("METRICS " + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()