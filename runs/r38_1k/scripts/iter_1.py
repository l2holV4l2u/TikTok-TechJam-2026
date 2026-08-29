import os
import gc
import json
import math
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 20220408
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
LR = 0.001
EPOCHS = 4
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 131072

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, os.cpu_count() or 1))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


class FactorizationMachine(nn.Module):
    def __init__(self, cardinalities, k, base_logit):
        super().__init__()
        offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
        self.register_buffer("offsets", torch.from_numpy(offsets))
        total_cardinality = int(sum(cardinalities))

        self.linear = nn.Embedding(total_cardinality, 1, sparse=True)
        self.factors = nn.Embedding(total_cardinality, k, sparse=True)

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)

        # ID 0 denotes unseen. Keep each field's unseen linear and latent
        # representation neutral unless it actually occurs in training.
        with torch.no_grad():
            unseen_rows = torch.from_numpy(offsets)
            self.linear.weight[unseen_rows].zero_()
            self.factors.weight[unseen_rows].zero_()

        self.register_buffer(
            "base_logit", torch.tensor(float(base_logit), dtype=torch.float32)
        )

    def forward(self, x):
        x = x + self.offsets
        linear_term = self.linear(x).squeeze(-1).sum(dim=1)

        v = self.factors(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)

        return self.base_logit + linear_term + interaction


def get_feature_arrays(split):
    arrays = []
    for name in FIELDS:
        if name in split.X:
            arrays.append(split.X[name])
        elif name == "user_id":
            arrays.append(split.user_id)
        elif name == "video_id":
            arrays.append(split.video_id)
        else:
            raise KeyError(f"Required baseline feature not found: {name}")
    return arrays


def make_batch(arrays, indices):
    x = np.empty((indices.size, len(arrays)), dtype=np.int64)
    for j, arr in enumerate(arrays):
        x[:, j] = arr[indices]
    return torch.from_numpy(x)


def predict(model, split):
    arrays = get_feature_arrays(split)
    n = len(split.user_id)
    scores = np.empty(n, dtype=np.float32)

    model.eval()
    with torch.inference_mode():
        for start in range(0, n, PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, n)
            idx = np.arange(start, end, dtype=np.int64)
            xb = make_batch(arrays, idx)
            scores[start:end] = model(xb).cpu().numpy()
    return scores


def main():
    train = load("train")
    train_arrays = get_feature_arrays(train)
    y_train = np.asarray(train.y, dtype=np.float32)
    n_train = y_train.size

    cardinalities = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
    positive_rate = float(y_train.mean())
    base_logit = math.log(positive_rate / (1.0 - positive_rate))

    model = FactorizationMachine(cardinalities, K, base_logit)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    rng = np.random.default_rng(SEED)
    model.train()

    for _ in range(EPOCHS):
        order = rng.permutation(n_train)
        for start in range(0, n_train, BATCH_SIZE):
            idx = order[start:min(start + BATCH_SIZE, n_train)]
            xb = make_batch(train_arrays, idx)
            yb = torch.from_numpy(y_train[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        del order
        gc.collect()

    del optimizer, criterion, train_arrays, y_train, train
    gc.collect()

    valid = load("valid")
    valid_scores = predict(model, valid)
    metrics = evaluate(valid.user_id, valid.y, valid_scores)

    del valid_scores, valid
    gc.collect()

    out = os.environ.get("ITER_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        test = load("test")
        test_scores = predict(model, test)
        np.save(
            os.path.join(out, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )
        del test_scores, test
        gc.collect()

    result = {
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": 0.0,
    }
    print("METRICS " + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()