import os
import json
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
BATCH_SIZE = 2048
EPOCHS = 7
NEGATIVES_PER_POSITIVE = 3


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_indices(split, offsets):
    cols = []
    for name, offset in zip(FIELDS, offsets):
        x = np.asarray(split.X[name], dtype=np.int64)
        cols.append(x + np.int64(offset))
    return torch.from_numpy(np.stack(cols, axis=1))


class FactorizationMachine(torch.nn.Module):
    def __init__(self, num_categories, rank):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_categories, rank + 1, sparse=True
        )
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)

    def forward(self, x):
        z = self.embedding(x)
        linear = z[:, :, 0].sum(dim=1)
        factors = z[:, :, 1:]
        summed = factors.sum(dim=1)
        interactions = 0.5 * (
            summed.square().sum(dim=1)
            - factors.square().sum(dim=(1, 2))
        )
        return linear + interactions


@torch.no_grad()
def predict(model, x, batch_size=32768):
    model.eval()
    out = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        out[start:end] = (
            model(x[start:end]).cpu().numpy().astype(np.float64, copy=False)
        )
    return out


def prepare_pair_sampler(train):
    labels = np.asarray(train.y, dtype=np.int8)
    users = np.asarray(train.user_id, dtype=np.int64)

    user_cardinality = int(FEATURE_CARDINALITIES["user_id"])
    negative_rows = np.flatnonzero(labels == 0).astype(np.int64)
    negative_users = users[negative_rows]

    order = np.argsort(negative_users, kind="stable")
    negative_rows = negative_rows[order]
    negative_users = negative_users[order]

    negative_counts = np.bincount(
        negative_users, minlength=user_cardinality
    ).astype(np.int64)
    negative_starts = np.empty(user_cardinality, dtype=np.int64)
    negative_starts[0] = 0
    np.cumsum(negative_counts[:-1], out=negative_starts[1:])

    positive_rows = np.flatnonzero(labels == 1).astype(np.int64)
    positive_users = users[positive_rows]
    eligible = negative_counts[positive_users] > 0
    positive_rows = positive_rows[eligible]
    positive_users = positive_users[eligible]

    return (
        positive_rows,
        positive_users,
        negative_rows,
        negative_starts,
        negative_counts,
    )


def sample_pairs(
    rng,
    positive_rows,
    positive_users,
    negative_rows,
    negative_starts,
    negative_counts,
):
    pair_positive_rows = np.repeat(
        positive_rows, NEGATIVES_PER_POSITIVE
    )
    pair_users = np.repeat(
        positive_users, NEGATIVES_PER_POSITIVE
    )

    counts = negative_counts[pair_users]
    offsets = np.floor(rng.random(pair_users.shape[0]) * counts).astype(
        np.int64
    )
    positions = negative_starts[pair_users] + offsets
    pair_negative_rows = negative_rows[positions]

    return (
        torch.from_numpy(pair_positive_rows),
        torch.from_numpy(pair_negative_rows),
    )


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

    sampler_data = prepare_pair_sampler(train)

    model = FactorizationMachine(total_categories, RANK)
    optimizer = torch.optim.SparseAdam(
        model.embedding.parameters(), lr=LR
    )

    rng = np.random.default_rng(SEED)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    best_primary = -np.inf
    best_metrics = None
    best_weights = None

    for epoch in range(EPOCHS):
        positive_rows, negative_rows = sample_pairs(rng, *sampler_data)
        number_of_pairs = positive_rows.shape[0]
        permutation = torch.randperm(number_of_pairs, generator=generator)

        model.train()
        loss_sum = 0.0

        for start in range(0, number_of_pairs, BATCH_SIZE):
            pair_ids = permutation[start:start + BATCH_SIZE]
            pos_ids = positive_rows[pair_ids]
            neg_ids = negative_rows[pair_ids]

            pair_x = torch.cat(
                (x_train[pos_ids], x_train[neg_ids]), dim=0
            )

            optimizer.zero_grad(set_to_none=True)
            pair_scores = model(pair_x)
            batch_size = pos_ids.shape[0]
            positive_scores = pair_scores[:batch_size]
            negative_scores = pair_scores[batch_size:]

            loss = F.softplus(
                negative_scores - positive_scores
            ).mean()
            loss.backward()
            optimizer.step()

            loss_sum += float(loss) * batch_size

        valid_scores = predict(model, x_valid)
        metrics = evaluate(
            np.asarray(valid.user_id),
            np.asarray(valid.y),
            valid_scores,
        )

        print(
            "epoch=%d bpr_loss=%.6f pairs=%d primary=%.6f "
            "gauc=%.6f ndcg5=%.6f"
            % (
                epoch + 1,
                loss_sum / number_of_pairs,
                number_of_pairs,
                metrics["primary"],
                metrics["gauc"],
                metrics["ndcg@5"],
            ),
            flush=True,
        )

        if metrics["primary"] > best_primary:
            best_primary = float(metrics["primary"])
            best_metrics = dict(metrics)
            best_weights = (
                model.embedding.weight.detach().cpu().clone()
            )

    with torch.no_grad():
        model.embedding.weight.copy_(best_weights)

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