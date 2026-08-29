import os
import time
import json
import copy
import random

import numpy as np
import torch
import torch.nn as nn
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
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]
EMBED_DIM = 12
DEEP_DIMS = (64, 32)
DROPOUT = 0.10
LEARNING_RATE = 0.001
BATCH_SIZE = 4096
EPOCHS = 5
PRED_BATCH_SIZE = 32768


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_offsets():
    cardinalities = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
    offsets = np.zeros(len(cardinalities), dtype=np.int64)
    if len(cardinalities) > 1:
        offsets[1:] = np.cumsum(cardinalities[:-1], dtype=np.int64)
    return offsets, int(sum(cardinalities))


def encode_split(split, offsets):
    columns = [
        np.asarray(split.X[name], dtype=np.int64) + offsets[j]
        for j, name in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.int64)


class DeepFM(nn.Module):
    def __init__(self, num_categories, num_fields, embed_dim, deep_dims, dropout):
        super().__init__()
        self.num_fields = num_fields
        self.embed_dim = embed_dim

        self.linear = nn.Embedding(num_categories, 1)
        self.embedding = nn.Embedding(num_categories, embed_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        layers = []
        input_dim = num_fields * embed_dim
        for output_dim in deep_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim
        self.deep = nn.Sequential(*layers)
        self.deep_output = nn.Linear(input_dim, 1)

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.deep_output.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.deep_output.bias)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)

        embeddings = self.embedding(x)
        summed = embeddings.sum(dim=1)
        fm_term = 0.5 * (
            summed.square().sum(dim=1)
            - embeddings.square().sum(dim=(1, 2))
        )

        deep_input = embeddings.reshape(embeddings.shape[0], -1)
        deep_term = self.deep_output(self.deep(deep_input)).squeeze(-1)

        return self.bias + linear_term + fm_term + deep_term


@torch.no_grad()
def predict(model, x_np, device):
    model.eval()
    result = np.empty(x_np.shape[0], dtype=np.float64)
    for start in range(0, x_np.shape[0], PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, x_np.shape[0])
        xb = torch.from_numpy(x_np[start:end]).to(device=device)
        logits = model(xb)
        result[start:end] = logits.cpu().numpy().astype(np.float64)
    return result


def standardization(scores):
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return mean, std


def main():
    start_time = time.perf_counter()
    seed_everything(SEED)

    requested_device = os.environ["AGENT_DEVICE"].lower()
    device = torch.device(requested_device)
    if device.type == "cpu":
        torch.set_num_threads(
            max(1, int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))
        )

    train = load("train")
    valid = load("valid")

    offsets, num_categories = make_offsets()
    x_train_np = encode_split(train, offsets)
    x_valid_np = encode_split(valid, offsets)
    y_train_np = np.ascontiguousarray(train.y, dtype=np.float32)

    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(y_train_np)

    model = DeepFM(
        num_categories=num_categories,
        num_fields=len(FIELDS),
        embed_dim=EMBED_DIM,
        deep_dims=DEEP_DIMS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    rng = np.random.default_rng(SEED)
    n_train = x_train.shape[0]
    best_primary = -np.inf
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(n_train)
        total_loss = 0.0

        for start in range(0, n_train, BATCH_SIZE):
            idx_np = order[start:start + BATCH_SIZE]
            idx = torch.from_numpy(idx_np)

            xb = x_train.index_select(0, idx).to(device=device)
            yb = y_train.index_select(0, idx).to(device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach()) * len(idx_np)

        epoch_scores = predict(model, x_valid_np, device)
        epoch_metrics = evaluate(valid.user_id, valid.y, epoch_scores)
        print(
            "epoch=%d loss=%.6f primary=%.6f gauc=%.6f ndcg@5=%.6f"
            % (
                epoch + 1,
                total_loss / n_train,
                epoch_metrics["primary"],
                epoch_metrics["gauc"],
                epoch_metrics["ndcg@5"],
            ),
            flush=True,
        )

        if epoch_metrics["primary"] > best_primary:
            best_primary = float(epoch_metrics["primary"])
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    deep_valid_scores = predict(model, x_valid_np, device)
    deep_metrics = evaluate(valid.user_id, valid.y, deep_valid_scores)

    artifacts = os.environ.get("RUN_ARTIFACTS", "")
    incumbent_valid_path = os.path.join(
        artifacts, "incumbent_valid_scores.npy"
    )
    incumbent_test_path = os.path.join(
        artifacts, "incumbent_test_scores.npy"
    )

    candidate_results = {
        "deepfm": float(deep_metrics["primary"])
    }
    selected_alpha = 1.0
    valid_scores = deep_valid_scores
    deep_mean, deep_std = standardization(deep_valid_scores)
    incumbent_mean = 0.0
    incumbent_std = 1.0
    use_incumbent = False

    if (
        os.path.isfile(incumbent_valid_path)
        and os.path.isfile(incumbent_test_path)
    ):
        incumbent_valid = np.asarray(
            np.load(incumbent_valid_path), dtype=np.float64
        )
        if incumbent_valid.shape == deep_valid_scores.shape:
            use_incumbent = True
            incumbent_mean, incumbent_std = standardization(incumbent_valid)

            deep_z = (deep_valid_scores - deep_mean) / deep_std
            incumbent_z = (
                incumbent_valid - incumbent_mean
            ) / incumbent_std

            best_blend_primary = -np.inf
            for alpha in np.linspace(0.0, 1.0, 11):
                blended = alpha * deep_z + (1.0 - alpha) * incumbent_z
                blend_metrics = evaluate(valid.user_id, valid.y, blended)
                name = "blend_deep_%0.1f" % alpha
                candidate_results[name] = float(blend_metrics["primary"])

                if blend_metrics["primary"] > best_blend_primary:
                    best_blend_primary = float(blend_metrics["primary"])
                    selected_alpha = float(alpha)
                    valid_scores = blended.copy()

    metrics = evaluate(valid.user_id, valid.y, valid_scores)

    print(
        "FINDINGS selected_deep_weight=%.1f deepfm_primary=%.6f selected_primary=%.6f"
        % (
            selected_alpha,
            deep_metrics["primary"],
            metrics["primary"],
        ),
        flush=True,
    )
    print(
        "CANDIDATES " + json.dumps(candidate_results, sort_keys=True),
        flush=True,
    )

    out_dir = os.environ.get("ITER_OUT")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(
            os.path.join(out_dir, "scores_valid.npy"),
            np.asarray(valid_scores, dtype=np.float64),
        )

    del x_train, y_train, x_train_np, y_train_np, x_valid_np
    del train

    test = load("test")
    x_test_np = encode_split(test, offsets)
    deep_test_scores = predict(model, x_test_np, device)

    if use_incumbent:
        incumbent_test = np.asarray(
            np.load(incumbent_test_path), dtype=np.float64
        )
        if incumbent_test.shape == deep_test_scores.shape:
            deep_test_z = (deep_test_scores - deep_mean) / deep_std
            incumbent_test_z = (
                incumbent_test - incumbent_mean
            ) / incumbent_std
            test_scores = (
                selected_alpha * deep_test_z
                + (1.0 - selected_alpha) * incumbent_test_z
            )
        else:
            test_scores = deep_test_scores
    else:
        test_scores = deep_test_scores

    if out_dir:
        np.save(
            os.path.join(out_dir, "scores_test.npy"),
            np.asarray(test_scores, dtype=np.float64),
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start_time
    final = {
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
        "device": device.type,
    }
    print("METRICS " + json.dumps(final, separators=(", ", ": ")))


if __name__ == "__main__":
    main()