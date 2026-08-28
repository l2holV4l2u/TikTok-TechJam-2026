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
CLICK_LOSS_WEIGHT = 0.25
PAIR_PRIOR_STRENGTH = 20.0
BLEND_WEIGHTS = np.asarray(
    [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0],
    dtype=np.float64
)


class MultiTaskFactorizationMachine(nn.Module):
    def __init__(
        self,
        total_cardinality,
        embedding_dim,
        long_view_rate,
        click_rate
    ):
        super().__init__()

        # Two task-specific linear coordinates followed by shared factors.
        self.embedding = nn.Embedding(
            total_cardinality, embedding_dim + 2, sparse=True
        )
        initial_biases = [
            math.log(long_view_rate / (1.0 - long_view_rate)),
            math.log(click_rate / (1.0 - click_rate))
        ]
        self.bias = nn.Parameter(
            torch.tensor(initial_biases, dtype=torch.float32)
        )
        self.interaction_scale = nn.Parameter(
            torch.ones(2, dtype=torch.float32)
        )

        with torch.no_grad():
            self.embedding.weight[:, :2].zero_()
            self.embedding.weight[:, 2:].normal_(mean=0.0, std=0.01)

    def forward(self, x):
        parameters = self.embedding(x)

        # Dedicated linear terms let click and long_view have different
        # marginal feature effects.
        linear = parameters[:, :, :2].sum(dim=1)

        # The latent factors are shared, so auxiliary click supervision
        # regularizes the interaction representation.
        factors = parameters[:, :, 2:]
        summed = factors.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - factors.square().sum(dim=1)
        ).sum(dim=1, keepdim=True)

        return self.bias + linear + interaction * self.interaction_scale


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


@torch.inference_mode()
def predict(model, x_np, batch_size=32768):
    model.eval()
    result = np.empty(x_np.shape[0], dtype=np.float32)

    for start in range(0, x_np.shape[0], batch_size):
        end = min(start + batch_size, x_np.shape[0])
        xb = torch.from_numpy(x_np[start:end])
        result[start:end] = model(xb)[:, 0].cpu().numpy()

    return result


def fit_user_author_affinity(train):
    user = np.asarray(train.X["user_id"], dtype=np.int64)
    author = np.asarray(train.X["author_id"], dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.float64)

    author_cardinality = int(FEATURE_CARDINALITIES["author_id"])

    author_count = np.bincount(
        author, minlength=author_cardinality
    ).astype(np.float64)
    author_sum = np.bincount(
        author, weights=labels, minlength=author_cardinality
    ).astype(np.float64)

    global_rate = float(labels.mean())
    author_rate = (
        author_sum + PAIR_PRIOR_STRENGTH * global_rate
    ) / (
        author_count + PAIR_PRIOR_STRENGTH
    )

    pair_keys = user * np.int64(author_cardinality) + author
    unique_keys, inverse = np.unique(pair_keys, return_inverse=True)

    pair_count = np.bincount(inverse).astype(np.float64)
    pair_sum = np.bincount(
        inverse, weights=labels
    ).astype(np.float64)

    pair_author = unique_keys % np.int64(author_cardinality)
    pair_prior = author_rate[pair_author]
    pair_rate = (
        pair_sum + PAIR_PRIOR_STRENGTH * pair_prior
    ) / (
        pair_count + PAIR_PRIOR_STRENGTH
    )

    eps = 1e-5
    pair_rate = np.clip(pair_rate, eps, 1.0 - eps)
    pair_prior = np.clip(pair_prior, eps, 1.0 - eps)

    pair_residual = (
        np.log(pair_rate) - np.log1p(-pair_rate)
        - np.log(pair_prior) + np.log1p(-pair_prior)
    ).astype(np.float32)

    return unique_keys, pair_residual, author_cardinality


def affinity_scores(split, unique_keys, pair_residual, author_cardinality):
    user = np.asarray(split.X["user_id"], dtype=np.int64)
    author = np.asarray(split.X["author_id"], dtype=np.int64)
    keys = user * np.int64(author_cardinality) + author

    positions = np.searchsorted(unique_keys, keys)
    valid_position = positions < unique_keys.size

    result = np.zeros(keys.shape[0], dtype=np.float32)
    if np.any(valid_position):
        row_indices = np.flatnonzero(valid_position)
        candidate_positions = positions[valid_position]
        matched = (
            unique_keys[candidate_positions] == keys[valid_position]
        )
        if np.any(matched):
            matched_rows = row_indices[matched]
            result[matched_rows] = pair_residual[
                candidate_positions[matched]
            ]

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
    click_train = np.asarray(train.aux["is_click"], dtype=np.float32)

    long_view_rate = float(np.clip(y_train.mean(), 1e-5, 1.0 - 1e-5))
    click_rate = float(np.clip(click_train.mean(), 1e-5, 1.0 - 1e-5))

    model = MultiTaskFactorizationMachine(
        total_cardinality=total_cardinality,
        embedding_dim=EMBED_DIM,
        long_view_rate=long_view_rate,
        click_rate=click_rate
    )

    sparse_optimizer = torch.optim.SparseAdam(
        [model.embedding.weight], lr=LEARNING_RATE
    )
    dense_optimizer = torch.optim.Adam(
        [model.bias, model.interaction_scale], lr=LEARNING_RATE
    )

    n = x_train.shape[0]
    rng = np.random.default_rng(SEED)
    best_primary = -np.inf
    best_state = None

    for _ in range(EPOCHS):
        model.train()
        order = rng.permutation(n)

        for start in range(0, n, BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            xb = torch.from_numpy(x_train[indices])
            long_yb = torch.from_numpy(y_train[indices])
            click_yb = torch.from_numpy(click_train[indices])

            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            long_loss = F.binary_cross_entropy_with_logits(
                logits[:, 0], long_yb
            )
            click_loss = F.binary_cross_entropy_with_logits(
                logits[:, 1], click_yb
            )
            loss = long_loss + CLICK_LOSS_WEIGHT * click_loss
            loss.backward()

            sparse_optimizer.step()
            dense_optimizer.step()

        valid_scores = predict(model, x_valid)
        metrics = evaluate(valid.user_id, valid.y, valid_scores)
        primary = float(metrics["primary"])

        if primary > best_primary:
            best_primary = primary
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    model.load_state_dict(best_state)

    unique_pair_keys, pair_residual, author_cardinality = (
        fit_user_author_affinity(train)
    )

    base_valid_scores = predict(model, x_valid).astype(np.float64)
    valid_affinity = affinity_scores(
        valid,
        unique_pair_keys,
        pair_residual,
        author_cardinality
    ).astype(np.float64)

    best_blend = 0.0
    final_metrics = None
    selected_primary = -np.inf

    for blend in BLEND_WEIGHTS:
        candidate_scores = (
            base_valid_scores + float(blend) * valid_affinity
        )
        candidate_metrics = evaluate(
            valid.user_id, valid.y, candidate_scores
        )
        candidate_primary = float(candidate_metrics["primary"])

        if candidate_primary > selected_primary:
            selected_primary = candidate_primary
            best_blend = float(blend)
            final_metrics = dict(candidate_metrics)

    out = os.environ.get("ITER_OUT")
    if out:
        test = load("test")
        x_test = encode_split(test, offsets)
        base_test_scores = predict(model, x_test).astype(np.float64)
        test_affinity = affinity_scores(
            test,
            unique_pair_keys,
            pair_residual,
            author_cardinality
        ).astype(np.float64)
        test_scores = (
            base_test_scores + best_blend * test_affinity
        )

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