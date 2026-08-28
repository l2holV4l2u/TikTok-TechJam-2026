import os
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
EMBED_DIM = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 2048
MAX_EPOCHS = 15
USER_WEIGHT_EXPONENT = 0.5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_matrix(split, offsets):
    cols = []
    for name, offset in zip(FIELDS, offsets):
        x = np.asarray(split.X[name], dtype=np.int64)
        cols.append(x + offset)
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


def make_user_weights(split):
    user_ids = np.asarray(split.X["user_id"], dtype=np.int64)
    counts = np.bincount(
        user_ids,
        minlength=int(FEATURE_CARDINALITIES["user_id"]),
    )
    row_counts = counts[user_ids].astype(np.float32)
    weights = np.power(np.maximum(row_counts, 1.0), -USER_WEIGHT_EXPONENT)
    weights /= weights.mean()
    return np.ascontiguousarray(weights, dtype=np.float32)


class FactorizationMachine(nn.Module):
    def __init__(self, total_cardinality, embedding_dim, initial_bias):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, embedding_dim)
        self.bias = nn.Parameter(
            torch.tensor(float(initial_bias), dtype=torch.float32)
        )

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).squeeze(-1).sum(dim=1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear_term + interaction


@torch.inference_mode()
def predict(model, matrix, batch_size=32768):
    model.eval()
    n = matrix.shape[0]
    scores = np.empty(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = torch.from_numpy(matrix[start:end])
        scores[start:end] = model(xb).cpu().numpy()

    return scores


def main():
    set_seed(SEED)
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

    train = load("train")
    valid = load("valid")

    cardinalities = [
        int(FEATURE_CARDINALITIES[name])
        for name in FIELDS
    ]
    offsets = np.cumsum(
        [0] + cardinalities[:-1],
        dtype=np.int64,
    )
    total_cardinality = int(sum(cardinalities))

    x_train_np = make_matrix(train, offsets)
    x_valid_np = make_matrix(valid, offsets)
    y_train_np = np.asarray(train.y, dtype=np.float32)
    train_weights_np = make_user_weights(train)

    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(y_train_np)
    train_weights = torch.from_numpy(train_weights_np)

    positive_rate = float(
        np.clip(
            np.average(y_train_np, weights=train_weights_np),
            1e-6,
            1.0 - 1e-6,
        )
    )
    initial_bias = math.log(positive_rate / (1.0 - positive_rate))

    model = FactorizationMachine(
        total_cardinality=total_cardinality,
        embedding_dim=EMBED_DIM,
        initial_bias=initial_bias,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    generator = torch.Generator()
    generator.manual_seed(SEED)

    best_primary = -np.inf
    best_state = None
    best_epoch = 0
    n = len(y_train_np)

    print(
        "FINDINGS "
        f"user_weight_min={float(train_weights_np.min()):.6f} "
        f"user_weight_median={float(np.median(train_weights_np)):.6f} "
        f"user_weight_max={float(train_weights_np.max()):.6f} "
        f"weighted_label_rate={positive_rate:.6f}",
        flush=True,
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        weighted_loss_sum = 0.0
        weight_sum = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            xb = x_train[idx]
            yb = y_train[idx]
            wb = train_weights[idx]

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            row_losses = criterion(logits, yb)
            loss = (row_losses * wb).sum() / wb.sum()
            loss.backward()
            optimizer.step()

            weighted_loss_sum += float(
                (row_losses.detach() * wb).sum()
            )
            weight_sum += float(wb.sum())

        valid_scores = predict(model, x_valid_np)
        metrics = evaluate(valid.user_id, valid.y, valid_scores)
        primary = float(metrics["primary"])

        print(
            f"epoch={epoch} "
            f"weighted_loss={weighted_loss_sum / weight_sum:.6f} "
            f"primary={primary:.6f} "
            f"gauc={float(metrics['gauc']):.6f} "
            f"ndcg5={float(metrics['ndcg@5']):.6f}",
            flush=True,
        )

        if primary > best_primary:
            best_primary = primary
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    valid_scores = predict(model, x_valid_np)
    final_metrics = evaluate(
        valid.user_id,
        valid.y,
        valid_scores,
    )

    artifacts = os.environ.get("RUN_ARTIFACTS")
    if artifacts:
        os.makedirs(artifacts, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "fields": FIELDS,
                "cardinalities": cardinalities,
                "offsets": offsets,
                "embedding_dim": EMBED_DIM,
                "user_weight_exponent": USER_WEIGHT_EXPONENT,
                "best_epoch": best_epoch,
                "validation_metrics": {
                    "primary": float(final_metrics["primary"]),
                    "gauc": float(final_metrics["gauc"]),
                    "ndcg@5": float(final_metrics["ndcg@5"]),
                },
            },
            os.path.join(
                artifacts,
                "official_fm_k16_user_sqrt_weighted.pt",
            ),
        )

    out = os.environ.get("ITER_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        test = load("test")
        x_test_np = make_matrix(test, offsets)
        test_scores = predict(model, x_test_np)
        np.save(
            os.path.join(out, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

    result = {
        "primary": float(final_metrics["primary"]),
        "gauc": float(final_metrics["gauc"]),
        "ndcg@5": float(final_metrics["ndcg@5"]),
        "gpu_seconds": 0.0,
    }
    print("METRICS " + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()