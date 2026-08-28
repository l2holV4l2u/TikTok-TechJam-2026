import os
import json
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
LR = 0.001
BATCH_SIZE = 4096
EPOCHS = 10
EVAL_FROM_EPOCH = 4


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FactorizationMachine(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.linear = nn.Embedding(num_features, 1, sparse=True)
        self.embedding = nn.Embedding(num_features, embed_dim, sparse=True)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1) - v.square().sum(dim=(1, 2))
        )
        return self.bias + linear_term + interaction


def make_offsets():
    offsets = []
    total = 0
    for field in FIELDS:
        offsets.append(total)
        total += int(FEATURE_CARDINALITIES[field])
    return np.asarray(offsets, dtype=np.int64), total


def make_matrix(split, offsets):
    columns = [
        np.asarray(split.X[field], dtype=np.int64) + offsets[j]
        for j, field in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(np.stack(columns, axis=1))


def prepare_pair_sampler(train):
    user_ids = np.asarray(train.X["user_id"], dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.int8)
    num_users = int(FEATURE_CARDINALITIES["user_id"])

    negative_rows = np.flatnonzero(labels == 0).astype(np.int64)
    negative_order = np.argsort(
        user_ids[negative_rows], kind="stable"
    )
    negative_rows = negative_rows[negative_order]

    negative_counts = np.bincount(
        user_ids[negative_rows], minlength=num_users
    ).astype(np.int64)

    negative_starts = np.empty(num_users, dtype=np.int64)
    negative_starts[0] = 0
    np.cumsum(
        negative_counts[:-1],
        out=negative_starts[1:]
    )

    positive_rows = np.flatnonzero(labels == 1).astype(np.int64)
    positive_users = user_ids[positive_rows]
    usable = negative_counts[positive_users] > 0

    return (
        user_ids,
        positive_rows[usable],
        negative_rows,
        negative_counts,
        negative_starts
    )


def sample_pairs(
    rng,
    user_ids,
    positive_rows,
    negative_rows,
    negative_counts,
    negative_starts
):
    positive_users = user_ids[positive_rows]
    counts = negative_counts[positive_users]

    offsets = (
        rng.random(positive_rows.shape[0]) * counts
    ).astype(np.int64)

    sampled_negative_rows = negative_rows[
        negative_starts[positive_users] + offsets
    ]

    permutation = rng.permutation(positive_rows.shape[0])
    return (
        positive_rows[permutation],
        sampled_negative_rows[permutation]
    )


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
    seed_everything(SEED)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    rng = np.random.default_rng(SEED)

    train = load("train")
    valid = load("valid")

    offsets, num_features = make_offsets()
    x_train_np = make_matrix(train, offsets)
    x_valid_np = make_matrix(valid, offsets)
    x_train = torch.from_numpy(x_train_np)

    (
        train_user_ids,
        positive_rows,
        negative_rows,
        negative_counts,
        negative_starts
    ) = prepare_pair_sampler(train)

    model = FactorizationMachine(num_features, EMBED_DIM)

    sparse_optimizer = torch.optim.SparseAdam(
        [model.linear.weight, model.embedding.weight], lr=LR
    )
    bias_optimizer = torch.optim.Adam([model.bias], lr=LR)

    best_primary = -np.inf
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()

        epoch_positive_rows, epoch_negative_rows = sample_pairs(
            rng,
            train_user_ids,
            positive_rows,
            negative_rows,
            negative_counts,
            negative_starts
        )

        num_pairs = epoch_positive_rows.shape[0]

        for start in range(0, num_pairs, BATCH_SIZE):
            end = min(start + BATCH_SIZE, num_pairs)

            pos_idx = torch.from_numpy(epoch_positive_rows[start:end])
            neg_idx = torch.from_numpy(epoch_negative_rows[start:end])

            positive_x = x_train[pos_idx]
            negative_x = x_train[neg_idx]

            sparse_optimizer.zero_grad(set_to_none=True)
            bias_optimizer.zero_grad(set_to_none=True)

            positive_logits = model(positive_x)
            negative_logits = model(negative_x)

            loss = F.softplus(
                negative_logits - positive_logits
            ).mean()
            loss.backward()

            sparse_optimizer.step()
            bias_optimizer.step()

        if epoch >= EVAL_FROM_EPOCH:
            valid_scores = predict(model, x_valid_np)
            epoch_metrics = evaluate(
                valid.user_id,
                valid.y,
                valid_scores
            )
            if float(epoch_metrics["primary"]) > best_primary:
                best_primary = float(epoch_metrics["primary"])
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }

    if best_state is not None:
        model.load_state_dict(best_state)

    valid_scores = predict(model, x_valid_np)
    metrics = evaluate(valid.user_id, valid.y, valid_scores)

    out = os.environ.get("ITER_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        test = load("test")
        x_test_np = make_matrix(test, offsets)
        test_scores = predict(model, x_test_np)
        np.save(
            os.path.join(out, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64)
        )

    final_metrics = {
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": 0.0
    }
    print("METRICS " + json.dumps(final_metrics))


if __name__ == "__main__":
    main()