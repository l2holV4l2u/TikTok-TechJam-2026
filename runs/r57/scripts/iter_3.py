import os
import time
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2026
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "hour",
    "tag",
    "upload_type",
    "music_type",
    "video_type",
    "user_active_degree",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "register_days_bucket",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

EMBED_DIM = 12
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 65536
EPOCHS = 10
LEARNING_RATE = 1.2e-3


def set_reproducible(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_cat_matrix(split, offsets):
    columns = []
    for field, offset in zip(FIELDS, offsets):
        values = np.asarray(split.X[field], dtype=np.int64)
        columns.append(values + np.int64(offset))
    return np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.int64)


def fit_numeric_transform(train):
    means = []
    scales = []
    for field in NUM_FIELDS:
        x = np.asarray(train.num[field], dtype=np.float64)
        x = np.log1p(np.maximum(np.nan_to_num(x, nan=0.0), 0.0))
        mean = float(x.mean())
        scale = float(x.std())
        means.append(mean)
        scales.append(max(scale, 1e-6))
    return np.asarray(means, dtype=np.float32), np.asarray(scales, dtype=np.float32)


def make_numeric_matrix(split, means, scales):
    columns = []
    for j, field in enumerate(NUM_FIELDS):
        x = np.asarray(split.num[field], dtype=np.float32)
        x = np.log1p(np.maximum(np.nan_to_num(x, nan=0.0), 0.0))
        x = (x - means[j]) / scales[j]
        x = np.clip(x, -6.0, 6.0)
        columns.append(x)
    return np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.float32)


class DeepFM(nn.Module):
    def __init__(self, total_cardinality, n_fields, n_numeric, embedding_dim):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        deep_input = n_fields * embedding_dim + n_numeric
        self.deep = nn.Sequential(
            nn.Linear(deep_input, 256),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Begin close to an FM and let the nonlinear residual enter gradually.
        nn.init.normal_(self.deep[-1].weight, mean=0.0, std=0.005)

    def forward(self, x_cat, x_num):
        first_order = self.linear(x_cat).sum(dim=1).squeeze(-1)

        embeddings = self.embedding(x_cat)
        summed = embeddings.sum(dim=1)
        fm_interaction = 0.5 * (
            summed.square() - embeddings.square().sum(dim=1)
        ).sum(dim=1)

        deep_input = torch.cat(
            [embeddings.flatten(start_dim=1), x_num], dim=1
        )
        deep_score = self.deep(deep_input).squeeze(-1)

        return self.bias + first_order + fm_interaction + deep_score


@torch.no_grad()
def predict(model, x_cat, x_num, device):
    model.eval()
    result = np.empty(x_cat.shape[0], dtype=np.float32)

    for start in range(0, x_cat.shape[0], PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, x_cat.shape[0])
        cat_batch = torch.from_numpy(x_cat[start:end]).to(device=device)
        num_batch = torch.from_numpy(x_num[start:end]).to(device=device)
        logits = model(cat_batch, num_batch)
        result[start:end] = torch.sigmoid(logits).cpu().numpy()

    return result


def within_user_ranks(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    row_id = np.arange(n, dtype=np.int64)

    order = np.lexsort((row_id, scores, user_ids))
    ordered_users = user_ids[order]

    boundary = np.empty(n, dtype=bool)
    boundary[0] = True
    boundary[1:] = ordered_users[1:] != ordered_users[:-1]

    positions = np.arange(n, dtype=np.int64)
    group_starts = np.maximum.accumulate(np.where(boundary, positions, 0))
    ordinal_rank = positions - group_starts

    _, first, counts = np.unique(
        ordered_users, return_index=True, return_counts=True
    )
    ordered_denominator = np.repeat(np.maximum(counts - 1, 1), counts)

    ordered_rank = ordinal_rank.astype(np.float64) / ordered_denominator
    result = np.empty(n, dtype=np.float64)
    result[order] = ordered_rank
    return result


def candidate_blends(user_ids, incumbent, candidate):
    incumbent = np.asarray(incumbent, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)

    incumbent_rank = within_user_ranks(user_ids, incumbent)
    candidate_rank = within_user_ranks(user_ids, candidate)

    alphas = np.linspace(0.0, 1.0, 21)
    for alpha in alphas:
        yield (
            "direct",
            float(alpha),
            (1.0 - alpha) * incumbent + alpha * candidate,
        )
    for alpha in alphas:
        yield (
            "rank",
            float(alpha),
            (1.0 - alpha) * incumbent_rank + alpha * candidate_rank,
        )


def apply_blend(user_ids, incumbent, candidate, mode, alpha):
    if mode == "direct":
        return (
            (1.0 - alpha) * np.asarray(incumbent, dtype=np.float64)
            + alpha * np.asarray(candidate, dtype=np.float64)
        )

    incumbent_rank = within_user_ranks(user_ids, incumbent)
    candidate_rank = within_user_ranks(user_ids, candidate)
    return (1.0 - alpha) * incumbent_rank + alpha * candidate_rank


def main():
    wall_start = time.time()

    requested_device = os.environ.get("AGENT_DEVICE", "cpu").lower()
    if requested_device != "cpu":
        raise RuntimeError("This script is configured for the CPU-selected run.")
    device = torch.device("cpu")

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    set_reproducible(SEED)

    artifacts = os.environ["RUN_ARTIFACTS"]
    incumbent_valid_path = os.path.join(
        artifacts, "incumbent_valid_scores.npy"
    )
    incumbent_test_path = os.path.join(
        artifacts, "incumbent_test_scores.npy"
    )
    if not os.path.exists(incumbent_valid_path):
        raise FileNotFoundError(incumbent_valid_path)
    if not os.path.exists(incumbent_test_path):
        raise FileNotFoundError(incumbent_test_path)

    incumbent_valid = np.load(incumbent_valid_path).astype(
        np.float64, copy=False
    )
    incumbent_test = np.load(incumbent_test_path).astype(
        np.float64, copy=False
    )

    train = load("train")
    valid = load("valid")

    cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
    offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
    total_cardinality = int(sum(cardinalities))

    numeric_means, numeric_scales = fit_numeric_transform(train)

    x_train_cat_np = make_cat_matrix(train, offsets)
    x_valid_cat_np = make_cat_matrix(valid, offsets)
    x_train_num_np = make_numeric_matrix(
        train, numeric_means, numeric_scales
    )
    x_valid_num_np = make_numeric_matrix(
        valid, numeric_means, numeric_scales
    )
    y_train_np = np.ascontiguousarray(
        train.y.astype(np.float32, copy=False)
    )

    if incumbent_valid.shape[0] != valid.y.shape[0]:
        raise ValueError("Incumbent validation prediction length mismatch.")

    incumbent_metrics = evaluate(
        valid.user_id, valid.y, incumbent_valid
    )
    print(
        "FINDINGS incumbent primary={:.6f} gauc={:.6f} ndcg@5={:.6f}".format(
            float(incumbent_metrics["primary"]),
            float(incumbent_metrics["gauc"]),
            float(incumbent_metrics["ndcg@5"]),
        ),
        flush=True,
    )

    model = DeepFM(
        total_cardinality=total_cardinality,
        n_fields=len(FIELDS),
        n_numeric=len(NUM_FIELDS),
        embedding_dim=EMBED_DIM,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-6,
    )

    x_train_cat = torch.from_numpy(x_train_cat_np)
    x_train_num = torch.from_numpy(x_train_num_np)
    y_train = torch.from_numpy(y_train_np)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)

    best_primary = float(incumbent_metrics["primary"])
    best_metrics = incumbent_metrics
    best_epoch = 0
    best_mode = "direct"
    best_alpha = 0.0
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    best_valid_scores = np.asarray(incumbent_valid, dtype=np.float64).copy()

    n_train = x_train_cat.shape[0]

    for epoch in range(1, EPOCHS + 1):
        model.train()
        permutation = torch.randperm(n_train, generator=generator)
        running_loss = 0.0
        seen = 0

        for start in range(0, n_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_train)
            idx = permutation[start:end]

            cat_batch = x_train_cat[idx].to(device=device)
            num_batch = x_train_num[idx].to(device=device)
            label_batch = y_train[idx].to(device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(cat_batch, num_batch)
            loss = F.binary_cross_entropy_with_logits(logits, label_batch)
            loss.backward()
            optimizer.step()

            batch_size = end - start
            running_loss += float(loss.detach()) * batch_size
            seen += batch_size

        deep_scores = predict(
            model, x_valid_cat_np, x_valid_num_np, device
        )
        raw_metrics = evaluate(valid.user_id, valid.y, deep_scores)

        epoch_best_primary = -np.inf
        epoch_best_mode = None
        epoch_best_alpha = None
        epoch_best_metrics = None
        epoch_best_scores = None

        for mode, alpha, scores in candidate_blends(
            valid.user_id, incumbent_valid, deep_scores
        ):
            metrics = evaluate(valid.user_id, valid.y, scores)
            primary = float(metrics["primary"])
            if primary > epoch_best_primary:
                epoch_best_primary = primary
                epoch_best_mode = mode
                epoch_best_alpha = alpha
                epoch_best_metrics = metrics
                epoch_best_scores = np.asarray(
                    scores, dtype=np.float64
                ).copy()

        print(
            "epoch={} loss={:.6f} raw_primary={:.6f} "
            "blend_primary={:.6f} mode={} alpha={:.2f}".format(
                epoch,
                running_loss / max(seen, 1),
                float(raw_metrics["primary"]),
                epoch_best_primary,
                epoch_best_mode,
                epoch_best_alpha,
            ),
            flush=True,
        )

        if epoch_best_primary > best_primary:
            best_primary = epoch_best_primary
            best_metrics = epoch_best_metrics
            best_epoch = epoch
            best_mode = epoch_best_mode
            best_alpha = epoch_best_alpha
            best_valid_scores = epoch_best_scores
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    if best_epoch == 0:
        final_valid_scores = np.asarray(
            incumbent_valid, dtype=np.float64
        ).copy()
    else:
        selected_deep_valid = predict(
            model, x_valid_cat_np, x_valid_num_np, device
        )
        final_valid_scores = apply_blend(
            valid.user_id,
            incumbent_valid,
            selected_deep_valid,
            best_mode,
            best_alpha,
        )

    final_metrics = evaluate(
        valid.user_id, valid.y, final_valid_scores
    )

    print(
        "CANDIDATES "
        + json.dumps(
            {
                "incumbent": float(incumbent_metrics["primary"]),
                "selected": float(final_metrics["primary"]),
                "selected_epoch": int(best_epoch),
                "selected_alpha": float(best_alpha),
            },
            separators=(", ", ": "),
        ),
        flush=True,
    )
    print(
        "FINDINGS selected epoch={} mode={} alpha={:.2f}".format(
            best_epoch, best_mode, best_alpha
        ),
        flush=True,
    )

    out_dir = os.environ.get("ITER_OUT")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(
            os.path.join(out_dir, "scores_valid.npy"),
            np.asarray(final_valid_scores, dtype=np.float64),
        )

    test = load("test")
    if incumbent_test.shape[0] != test.user_id.shape[0]:
        raise ValueError("Incumbent test prediction length mismatch.")

    if best_epoch == 0:
        final_test_scores = np.asarray(
            incumbent_test, dtype=np.float64
        ).copy()
    else:
        x_test_cat_np = make_cat_matrix(test, offsets)
        x_test_num_np = make_numeric_matrix(
            test, numeric_means, numeric_scales
        )
        selected_deep_test = predict(
            model, x_test_cat_np, x_test_num_np, device
        )
        final_test_scores = apply_blend(
            test.user_id,
            incumbent_test,
            selected_deep_test,
            best_mode,
            best_alpha,
        )

    if out_dir:
        np.save(
            os.path.join(out_dir, "scores_test.npy"),
            np.asarray(final_test_scores, dtype=np.float64),
        )

    elapsed = time.time() - wall_start
    result = {
        "primary": float(final_metrics["primary"]),
        "gauc": float(final_metrics["gauc"]),
        "ndcg@5": float(final_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
        "device": "cpu",
    }
    print("METRICS " + json.dumps(result, separators=(", ", ": ")))


if __name__ == "__main__":
    main()