import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
EMBED_DIM = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 8192
EPOCHS = 5


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FactorizationMachine(nn.Module):
    def __init__(self, cardinalities, embedding_dim):
        super().__init__()
        total_cardinality = int(sum(cardinalities))
        offsets = np.cumsum([0] + list(cardinalities[:-1]), dtype=np.int64)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        x = x + self.offsets
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)

        latent = self.embedding(x)
        summed = latent.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - latent.square().sum(dim=1)
        ).sum(dim=1)

        return self.bias + linear_term + interaction


def make_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([split.X[name] for name in FIELDS]),
        dtype=np.int64,
    )


def fit_model(x_np, y_np, seed):
    seed_everything(seed)

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))

    cardinalities = [FEATURE_CARDINALITIES[name] for name in FIELDS]
    model = FactorizationMachine(cardinalities, EMBED_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    n = x.shape[0]
    generator = torch.Generator()
    generator.manual_seed(seed)

    model.train()
    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            logits = model(x[idx])
            loss = criterion(logits, y[idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


def predict(model, x_np):
    x = torch.from_numpy(x_np)
    result = np.empty(x.shape[0], dtype=np.float32)

    model.eval()
    with torch.inference_mode():
        for start in range(0, x.shape[0], BATCH_SIZE * 2):
            end = min(start + BATCH_SIZE * 2, x.shape[0])
            result[start:end] = model(x[start:end]).cpu().numpy()

    return result


def main():
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    seed_everything(SEED)

    train = load("train")
    valid = load("valid")

    x_train = make_matrix(train)
    x_valid = make_matrix(valid)
    y_train = np.asarray(train.y, dtype=np.float32)
    y_valid = np.asarray(valid.y, dtype=np.int8)

    validation_model = fit_model(x_train, y_train, SEED)
    valid_scores = predict(validation_model, x_valid)
    metrics = evaluate(valid.user_id, y_valid, valid_scores)

    out_dir = os.environ.get("ITER_OUT")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(
            os.path.join(out_dir, "scores_valid.npy"),
            np.asarray(valid_scores, dtype=np.float64),
        )

    del validation_model

    # Refit the identical recipe on all labels available before the test period.
    x_refit = np.concatenate([x_train, x_valid], axis=0)
    y_refit = np.concatenate(
        [y_train, np.asarray(y_valid, dtype=np.float32)], axis=0
    )

    refit_model = fit_model(x_refit, y_refit, SEED)

    test = load("test")
    x_test = make_matrix(test)
    test_scores = predict(refit_model, x_test)

    if out_dir:
        np.save(
            os.path.join(out_dir, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

    elapsed = time.time() - START_TIME
    payload = {
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    }
    print("METRICS " + json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()