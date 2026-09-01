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
FIELDS = [
    "author_id",
    "duration_bucket",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "hour",
    "is_live_streamer",
    "is_lowactive_period",
    "is_video_author",
    "music_type",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat4",
    "onehot_feat5",
    "onehot_feat6",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat9",
    "onehot_feat10",
    "onehot_feat11",
    "onehot_feat12",
    "onehot_feat13",
    "onehot_feat14",
    "onehot_feat15",
    "onehot_feat16",
    "onehot_feat17",
    "register_days_bucket",
    "register_days_range",
    "tab",
    "tag",
    "upload_type",
    "user_active_degree",
    "user_id",
    "video_id",
]
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

    train = load("train")
    valid = load("valid")

    offsets, num_features = make_offsets()
    x_train_np = make_matrix(train, offsets)
    x_valid_np = make_matrix(valid, offsets)

    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))

    model = FactorizationMachine(num_features, EMBED_DIM)

    sparse_optimizer = torch.optim.SparseAdam(
        [model.linear.weight, model.embedding.weight], lr=LR
    )
    bias_optimizer = torch.optim.Adam([model.bias], lr=LR)

    n = x_train.shape[0]
    best_primary = -np.inf
    best_state = None

    model.train()
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(n)

        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb = x_train[idx]
            yb = y_train[idx]

            sparse_optimizer.zero_grad(set_to_none=True)
            bias_optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
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

        model.train()

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