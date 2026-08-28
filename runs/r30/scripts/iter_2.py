import os
import json
import math
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
BATCH_SIZE = 2048
EPOCHS = 6

AFFINITY_FIELD = "tag"
AFFINITY_SMOOTHING = 8.0
AFFINITY_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


class FactorizationMachine(nn.Module):
    def __init__(self, total_cardinality, embedding_dim, initial_rate):
        super().__init__()
        self.embedding = nn.Embedding(
            total_cardinality, embedding_dim + 1, sparse=True
        )
        self.bias = nn.Parameter(
            torch.tensor(
                math.log(initial_rate / (1.0 - initial_rate)),
                dtype=torch.float32
            )
        )
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)

    def forward(self, x):
        parameters = self.embedding(x)
        linear = parameters[:, :, 0].sum(dim=1)
        factors = parameters[:, :, 1:]
        summed = factors.sum(dim=1)
        interactions = 0.5 * (
            summed.square() - factors.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interactions


def make_offsets():
    offsets = []
    current = 0
    for name in FIELDS:
        offsets.append(current)
        current += int(FEATURE_CARDINALITIES[name])
    return np.asarray(offsets, dtype=np.int64), current


def encode_split(split, offsets):
    return np.column_stack([
        np.asarray(split.X[name], dtype=np.int64) + offsets[i]
        for i, name in enumerate(FIELDS)
    ])


def logit(probability):
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability) - np.log1p(-probability)


def build_user_tag_affinity(train, y_train):
    user_cardinality = int(FEATURE_CARDINALITIES["user_id"])
    tag_cardinality = int(FEATURE_CARDINALITIES[AFFINITY_FIELD])

    users = np.asarray(train.X["user_id"], dtype=np.int64)
    tags = np.asarray(train.X[AFFINITY_FIELD], dtype=np.int64)
    keys = users * tag_cardinality + tags

    table_size = user_cardinality * tag_cardinality
    pair_counts = np.bincount(keys, minlength=table_size).astype(np.float32)
    pair_positives = np.bincount(
        keys, weights=y_train, minlength=table_size
    ).astype(np.float32)

    tag_counts = np.bincount(
        tags, minlength=tag_cardinality
    ).astype(np.float64)
    tag_positives = np.bincount(
        tags, weights=y_train, minlength=tag_cardinality
    ).astype(np.float64)

    global_rate = float(np.mean(y_train))
    tag_rates = (
        tag_positives + 20.0 * global_rate
    ) / (
        tag_counts + 20.0
    )

    repeated_tag_rates = np.tile(
        tag_rates.astype(np.float32), user_cardinality
    )
    pair_rates = (
        pair_positives + AFFINITY_SMOOTHING * repeated_tag_rates
    ) / (
        pair_counts + AFFINITY_SMOOTHING
    )

    residual = (
        logit(pair_rates.astype(np.float64))
        - logit(repeated_tag_rates.astype(np.float64))
    ).astype(np.float32)

    residual[pair_counts == 0] = 0.0
    residual = residual.reshape(user_cardinality, tag_cardinality)
    return residual


def lookup_affinity(split, affinity_table):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    tags = np.asarray(split.X[AFFINITY_FIELD], dtype=np.int64)

    valid_ids = (
        (users >= 0)
        & (users < affinity_table.shape[0])
        & (tags >= 0)
        & (tags < affinity_table.shape[1])
    )

    result = np.zeros(users.shape[0], dtype=np.float32)
    result[valid_ids] = affinity_table[
        users[valid_ids], tags[valid_ids]
    ]
    return result


@torch.inference_mode()
def predict(model, x_np, batch_size=32768):
    model.eval()
    result = np.empty(x_np.shape[0], dtype=np.float32)
    for start in range(0, x_np.shape[0], batch_size):
        end = min(start + batch_size, x_np.shape[0])
        xb = torch.from_numpy(x_np[start:end])
        result[start:end] = model(xb).cpu().numpy()
    return result


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    train = load("train")
    valid = load("valid")

    offsets, total_cardinality = make_offsets()
    x_train = encode_split(train, offsets)
    x_valid = encode_split(valid, offsets)
    y_train = np.asarray(train.y, dtype=np.float32)

    affinity_table = build_user_tag_affinity(train, y_train)
    valid_affinity = lookup_affinity(valid, affinity_table)

    model = FactorizationMachine(
        total_cardinality=total_cardinality,
        embedding_dim=EMBED_DIM,
        initial_rate=float(y_train.mean())
    )

    sparse_optimizer = torch.optim.SparseAdam(
        [model.embedding.weight], lr=LEARNING_RATE
    )
    bias_optimizer = torch.optim.Adam(
        [model.bias], lr=LEARNING_RATE
    )

    n = x_train.shape[0]
    rng = np.random.default_rng(SEED)
    best_primary = -np.inf
    best_state = None

    model.train()
    for _ in range(EPOCHS):
        order = rng.permutation(n)

        for start in range(0, n, BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            xb = torch.from_numpy(x_train[indices])
            yb = torch.from_numpy(y_train[indices])

            sparse_optimizer.zero_grad(set_to_none=True)
            bias_optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()

            sparse_optimizer.step()
            bias_optimizer.step()

        valid_scores = predict(model, x_valid)
        metrics = evaluate(valid.user_id, valid.y, valid_scores)

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

        model.train()

    model.load_state_dict(best_state)
    valid_fm_scores = predict(model, x_valid)

    best_weight = 0.0
    final_metrics = None
    final_valid_scores = None
    best_blended_primary = -np.inf

    for weight in AFFINITY_WEIGHTS:
        candidate_scores = valid_fm_scores + weight * valid_affinity
        candidate_metrics = evaluate(
            valid.user_id, valid.y, candidate_scores
        )
        candidate_primary = float(candidate_metrics["primary"])

        if candidate_primary > best_blended_primary:
            best_blended_primary = candidate_primary
            best_weight = float(weight)
            final_metrics = dict(candidate_metrics)
            final_valid_scores = candidate_scores

    out = os.environ.get("ITER_OUT")
    if out:
        test = load("test")
        x_test = encode_split(test, offsets)
        test_fm_scores = predict(model, x_test)
        test_affinity = lookup_affinity(test, affinity_table)
        test_scores = test_fm_scores + best_weight * test_affinity

        os.makedirs(out, exist_ok=True)
        np.save(
            os.path.join(out, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64)
        )

    payload = {
        "primary": float(final_metrics["primary"]),
        "gauc": float(final_metrics["gauc"]),
        "ndcg@5": float(final_metrics["ndcg@5"]),
        "gpu_seconds": 0.0
    }
    print("METRICS " + json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()