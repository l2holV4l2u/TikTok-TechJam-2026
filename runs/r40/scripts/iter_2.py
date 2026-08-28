import os
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
NUM_FIELDS = [
    "collect_cnt",
    "comment_cnt",
    "complete_play_cnt",
    "counts",
    "download_cnt",
    "duration_ms",
    "follow_cnt",
    "like_cnt",
    "long_time_play_cnt",
    "play_cnt",
    "play_duration",
    "play_progress",
    "play_user_num",
    "share_cnt",
    "short_time_play_cnt",
    "show_cnt",
    "show_user_num",
    "valid_play_cnt",
]
K = 16
LR = 0.001
BATCH_SIZE = 4096
EPOCHS = 12
ALPHAS = [0.5, 0.75, 1.0, 1.25, 1.5]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def make_categorical_matrix(split, offsets):
    cols = []
    for name, offset in zip(FIELDS, offsets):
        cols.append(np.asarray(split.X[name], dtype=np.int64) + offset)
    return torch.from_numpy(np.stack(cols, axis=1))


def fit_numeric_transform(split):
    centers = []
    scales = []
    for name in NUM_FIELDS:
        raw = np.asarray(split.num[name], dtype=np.float64)
        finite = np.isfinite(raw)
        median = float(np.median(raw[finite])) if finite.any() else 0.0
        filled = np.where(finite, raw, median)
        transformed = np.sign(filled) * np.log1p(np.abs(filled))
        center = float(np.mean(transformed))
        scale = float(np.std(transformed))
        if not np.isfinite(scale) or scale < 1e-5:
            scale = 1.0
        centers.append(center)
        scales.append(scale)
    return (
        np.asarray(centers, dtype=np.float32),
        np.asarray(scales, dtype=np.float32),
    )


def make_numeric_matrix(split, centers, scales):
    result = np.empty((len(split.y), len(NUM_FIELDS)), dtype=np.float32)
    for j, name in enumerate(NUM_FIELDS):
        raw = np.asarray(split.num[name], dtype=np.float32)
        raw = np.where(np.isfinite(raw), raw, 0.0)
        transformed = np.sign(raw) * np.log1p(np.abs(raw))
        result[:, j] = np.clip(
            (transformed - centers[j]) / scales[j], -8.0, 8.0
        )
    return torch.from_numpy(result)


cards = [int(FEATURE_CARDINALITIES[name]) for name in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))


class NumericAugmentedFM(nn.Module):
    def __init__(self, cardinality, rank, num_numeric):
        super().__init__()
        self.embedding = nn.Embedding(cardinality, rank + 1, sparse=True)
        self.bias = nn.Parameter(torch.zeros(()))
        self.numeric_tower = nn.Sequential(
            nn.Linear(num_numeric, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)

        for module in self.numeric_tower:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.numeric_tower[-1].weight, mean=0.0, std=0.01)

    def components(self, x_cat, x_num):
        e = self.embedding(x_cat)
        linear = e[:, :, 0].sum(dim=1)
        factors = e[:, :, 1:]
        summed = factors.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1)
            - factors.square().sum(dim=(1, 2))
        )
        categorical_logit = self.bias + linear + interaction
        numeric_logit = self.numeric_tower(x_num).squeeze(1)
        return categorical_logit, numeric_logit

    def forward(self, x_cat, x_num):
        categorical_logit, numeric_logit = self.components(x_cat, x_num)
        return categorical_logit + numeric_logit


@torch.no_grad()
def predict_components(model, x_cat, x_num, batch_size=32768):
    model.eval()
    categorical = np.empty(x_cat.shape[0], dtype=np.float64)
    numeric = np.empty(x_cat.shape[0], dtype=np.float64)

    for start in range(0, x_cat.shape[0], batch_size):
        end = min(start + batch_size, x_cat.shape[0])
        cat_part, num_part = model.components(
            x_cat[start:end], x_num[start:end]
        )
        categorical[start:end] = cat_part.cpu().numpy().astype(np.float64)
        numeric[start:end] = num_part.cpu().numpy().astype(np.float64)

    return categorical, numeric


train = load("train")
valid = load("valid")

numeric_centers, numeric_scales = fit_numeric_transform(train)

x_train_cat = make_categorical_matrix(train, offsets)
x_valid_cat = make_categorical_matrix(valid, offsets)
x_train_num = make_numeric_matrix(train, numeric_centers, numeric_scales)
x_valid_num = make_numeric_matrix(valid, numeric_centers, numeric_scales)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))

model = NumericAugmentedFM(
    total_cardinality, K, len(NUM_FIELDS)
)

embedding_optimizer = torch.optim.SparseAdam(
    [model.embedding.weight], lr=LR
)
dense_parameters = [
    model.bias,
    *list(model.numeric_tower.parameters()),
]
dense_optimizer = torch.optim.Adam(dense_parameters, lr=LR)
criterion = nn.BCEWithLogitsLoss()

n_train = x_train_cat.shape[0]
generator = torch.Generator()
generator.manual_seed(SEED)

best_primary = -np.inf
best_state = None
best_alpha = 1.0
best_metrics = None
candidate_best = {str(alpha): -np.inf for alpha in ALPHAS}

for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(n_train, generator=generator)

    for start in range(0, n_train, BATCH_SIZE):
        idx = permutation[start:min(start + BATCH_SIZE, n_train)]
        xb_cat = x_train_cat[idx]
        xb_num = x_train_num[idx]
        yb = y_train[idx]

        embedding_optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)

        logits = model(xb_cat, xb_num)
        loss = criterion(logits, yb)
        loss.backward()

        embedding_optimizer.step()
        dense_optimizer.step()

    valid_cat_part, valid_num_part = predict_components(
        model, x_valid_cat, x_valid_num
    )

    epoch_best_primary = -np.inf
    epoch_best_metrics = None
    epoch_best_alpha = 1.0

    for alpha in ALPHAS:
        valid_scores = valid_cat_part + alpha * valid_num_part
        metrics = evaluate(valid.user_id, valid.y, valid_scores)
        primary = float(metrics["primary"])
        candidate_best[str(alpha)] = max(
            candidate_best[str(alpha)], primary
        )

        if primary > epoch_best_primary:
            epoch_best_primary = primary
            epoch_best_alpha = alpha
            epoch_best_metrics = {
                "primary": primary,
                "gauc": float(metrics["gauc"]),
                "ndcg@5": float(metrics["ndcg@5"]),
            }

    if epoch_best_primary > best_primary:
        best_primary = epoch_best_primary
        best_alpha = epoch_best_alpha
        best_metrics = epoch_best_metrics
        best_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    print(
        "epoch=%d loss=%.6f alpha=%.2f primary=%.6f gauc=%.6f ndcg@5=%.6f"
        % (
            epoch + 1,
            float(loss.detach()),
            epoch_best_alpha,
            epoch_best_metrics["primary"],
            epoch_best_metrics["gauc"],
            epoch_best_metrics["ndcg@5"],
        ),
        flush=True,
    )

model.load_state_dict(best_state)

print(
    "CANDIDATES "
    + json.dumps(
        {
            "numeric_alpha_" + key: float(value)
            for key, value in candidate_best.items()
        },
        sort_keys=True,
    )
)

print(
    "FINDINGS "
    + "selected numeric-tower alpha=%.2f across %d robustly scaled video features"
    % (best_alpha, len(NUM_FIELDS))
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    test = load("test")
    x_test_cat = make_categorical_matrix(test, offsets)
    x_test_num = make_numeric_matrix(
        test, numeric_centers, numeric_scales
    )
    test_cat_part, test_num_part = predict_components(
        model, x_test_cat, x_test_num
    )
    test_scores = test_cat_part + best_alpha * test_num_part
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

final = {
    "primary": best_metrics["primary"],
    "gauc": best_metrics["gauc"],
    "ndcg@5": best_metrics["ndcg@5"],
    "gpu_seconds": 0.0,
}
print("METRICS " + json.dumps(final))