import gc
import json
import os
import numpy as np
import torch
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
]
K = 16
LR = 0.001
EPOCHS = 10
TRAIN_BATCH_SIZE = 8192
PRED_BATCH_SIZE = 131072
USER_WEIGHT_POWER = 0.5

np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


class FactorizationMachine(torch.nn.Module):
    def __init__(self, cardinalities, k):
        super().__init__()
        offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
        self.register_buffer("offsets", torch.from_numpy(offsets))
        self.embedding = torch.nn.Embedding(
            int(sum(cardinalities)), k + 1, sparse=True
        )

        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)
            # Local category zero is the unseen category for each field.
            self.embedding.weight[self.offsets].zero_()

    def forward(self, x):
        z = self.embedding(x + self.offsets)
        linear = z[:, :, 0].sum(dim=1)
        factors = z[:, :, 1:]
        summed = factors.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - factors.square().sum(dim=1)
        ).sum(dim=1)
        return linear + interaction


def matrix_from_split(split):
    missing = [name for name in FIELDS if name not in split.X]
    if missing:
        raise KeyError(
            "Required FM feature fields are missing from s.X: "
            + repr(missing)
        )
    return np.column_stack(
        [np.asarray(split.X[name], dtype=np.int64) for name in FIELDS]
    )


def make_user_weights(user_ids):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    counts = np.bincount(
        user_ids,
        minlength=int(FEATURE_CARDINALITIES["user_id"]),
    ).astype(np.float64)

    row_counts = counts[user_ids]
    if np.any(row_counts <= 0):
        raise RuntimeError("Encountered a training row with zero user frequency.")

    weights = np.power(row_counts, -USER_WEIGHT_POWER)
    weights /= weights.mean()
    return np.asarray(weights, dtype=np.float32)


@torch.no_grad()
def predict(model, x, intercept):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float32)
    for start in range(0, x.shape[0], PRED_BATCH_SIZE):
        stop = min(start + PRED_BATCH_SIZE, x.shape[0])
        xb = torch.from_numpy(x[start:stop])
        logits = model(xb) + intercept
        result[start:stop] = logits.cpu().numpy()
    return result


def main():
    train = load("train")
    valid = load("valid")

    cardinalities = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
    x_train = matrix_from_split(train)
    x_valid = matrix_from_split(valid)
    y_train = np.asarray(train.y, dtype=np.float32)
    train_weights = make_user_weights(train.X["user_id"])

    if x_train.shape[1] != len(FIELDS):
        raise RuntimeError("FM feature matrix has an unexpected field count.")
    if not np.all((y_train == 0.0) | (y_train == 1.0)):
        raise RuntimeError("train.y is not the native binary long_view label.")
    if not np.all(np.isfinite(train_weights)):
        raise RuntimeError("Non-finite user-frequency training weights.")

    model = FactorizationMachine(cardinalities, K)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=LR)

    positive_rate = float(np.clip(y_train.mean(), 1e-6, 1.0 - 1e-6))
    intercept = float(np.log(positive_rate / (1.0 - positive_rate)))

    n = x_train.shape[0]
    best_primary = -np.inf
    best_weight = None
    best_metrics = None

    model.train()
    for epoch in range(EPOCHS):
        order = np.random.permutation(n)

        for start in range(0, n, TRAIN_BATCH_SIZE):
            idx = order[start:min(start + TRAIN_BATCH_SIZE, n)]
            xb = torch.from_numpy(x_train[idx])
            yb = torch.from_numpy(y_train[idx])
            wb = torch.from_numpy(train_weights[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb) + intercept
            row_loss = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = torch.sum(row_loss * wb) / torch.sum(wb)
            loss.backward()
            optimizer.step()

        del order

        valid_scores = predict(model, x_valid, intercept)
        metrics = evaluate(valid.user_id, valid.y, valid_scores)

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_metrics = metrics
            if best_weight is not None:
                del best_weight
                gc.collect()
            best_weight = model.embedding.weight.detach().clone()

        model.train()

    if best_weight is None:
        raise RuntimeError("Training failed to produce a validation checkpoint.")

    with torch.no_grad():
        model.embedding.weight.copy_(best_weight)
    model.eval()

    valid_scores = predict(model, x_valid, intercept)
    final_metrics = evaluate(valid.user_id, valid.y, valid_scores)

    del optimizer, best_weight
    del x_train, y_train, train_weights, train
    del x_valid, valid_scores, valid
    gc.collect()

    out = os.environ.get("ITER_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        test = load("test")
        x_test = matrix_from_split(test)
        test_scores = predict(model, x_test, intercept)
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
    print("METRICS " + json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()