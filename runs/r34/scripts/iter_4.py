import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


torch.set_num_threads(min(8, os.cpu_count() or 1))
torch.manual_seed(2024)
np.random.seed(2024)

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "onehot_feat3",
    "upload_type",
    "duration_bucket",
    "onehot_feat1",
    "onehot_feat8",
    "user_active_degree",
    "hour",
]

CARDINALITIES = np.asarray(
    [int(FEATURE_CARDINALITIES[name]) for name in FIELDS],
    dtype=np.int64,
)
OFFSETS = np.zeros(len(FIELDS), dtype=np.int64)
OFFSETS[1:] = np.cumsum(CARDINALITIES[:-1], dtype=np.int64)
TOTAL_CARDINALITY = int(CARDINALITIES.sum())


def make_features(split):
    x = np.column_stack([split.X[name] for name in FIELDS]).astype(
        np.int64, copy=False
    )
    x = x + OFFSETS[None, :]
    return torch.from_numpy(np.ascontiguousarray(x))


class DeepFM(nn.Module):
    def __init__(self, num_categories, num_fields, embedding_dim=16):
        super().__init__()
        self.linear = nn.Embedding(num_categories, 1, sparse=True)
        self.embedding = nn.Embedding(
            num_categories, embedding_dim, sparse=True
        )
        self.bias = nn.Parameter(torch.zeros(()))

        deep_input = num_fields * embedding_dim
        self.deep = nn.Sequential(
            nn.Linear(deep_input, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1),
        )

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        for layer in self.deep:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.deep[-1].weight)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)

        embeddings = self.embedding(x)
        summed = embeddings.sum(dim=1)
        fm_term = 0.5 * (
            summed.square().sum(dim=1)
            - embeddings.square().sum(dim=(1, 2))
        )

        deep_term = self.deep(
            embeddings.reshape(embeddings.shape[0], -1)
        ).squeeze(-1)

        return self.bias + linear_term + fm_term + deep_term


@torch.inference_mode()
def predict(model, x, batch_size=65536):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        result[start:end] = (
            model(x[start:end]).detach().double().cpu().numpy()
        )
    return result


def safe_logit(prob):
    prob = np.clip(prob, 1e-5, 1.0 - 1e-5)
    return np.log(prob) - np.log1p(-prob)


def make_rate_tables(train, field, weights, smoothing):
    ids = np.asarray(train.X[field], dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.float64)
    cardinality = int(FEATURE_CARDINALITIES[field])

    counts = np.bincount(
        ids, weights=weights, minlength=cardinality
    ).astype(np.float64, copy=False)
    positives = np.bincount(
        ids, weights=weights * labels, minlength=cardinality
    ).astype(np.float64, copy=False)

    global_rate = float(np.sum(weights * labels) / np.sum(weights))
    rates = (positives + smoothing * global_rate) / (
        counts + smoothing
    )
    return safe_logit(rates), safe_logit(global_rate)


def build_temporal_tables(train, half_life=3.0):
    dates = np.asarray(train.date)
    unique_dates = np.unique(dates)
    day_index = np.searchsorted(unique_dates, dates)
    age = (len(unique_dates) - 1 - day_index).astype(np.float64)
    recent_weights = np.exp2(-age / half_life)
    all_weights = np.ones(len(dates), dtype=np.float64)

    tables = {}
    for field, smoothing in (
        ("video_id", 20.0),
        ("author_id", 25.0),
        ("tag", 80.0),
    ):
        all_logits, all_global = make_rate_tables(
            train, field, all_weights, smoothing
        )
        recent_logits, recent_global = make_rate_tables(
            train, field, recent_weights, smoothing
        )
        tables[field] = {
            "all": all_logits,
            "recent": recent_logits,
            "all_global": all_global,
            "recent_global": recent_global,
        }
    return tables


def temporal_signals(split, tables):
    level_parts = []
    residual_parts = []

    for field in ("video_id", "author_id"):
        ids = np.asarray(split.X[field], dtype=np.int64)
        table = tables[field]

        level = table["recent"][ids] - table["recent_global"]
        residual = (
            table["recent"][ids]
            - table["all"][ids]
            - (table["recent_global"] - table["all_global"])
        )
        level_parts.append(level)
        residual_parts.append(residual)

    tag_ids = np.asarray(split.X["tag"], dtype=np.int64)
    tag_table = tables["tag"]
    tag_level = (
        tag_table["recent"][tag_ids] - tag_table["recent_global"]
    )
    tag_residual = (
        tag_table["recent"][tag_ids]
        - tag_table["all"][tag_ids]
        - (tag_table["recent_global"] - tag_table["all_global"])
    )

    level_signal = (
        0.45 * level_parts[0]
        + 0.45 * level_parts[1]
        + 0.10 * tag_level
    )
    residual_signal = (
        0.45 * residual_parts[0]
        + 0.45 * residual_parts[1]
        + 0.10 * tag_residual
    )

    level_signal = np.clip(level_signal, -3.0, 3.0)
    residual_signal = np.clip(residual_signal, -2.0, 2.0)
    return {
        "raw": np.zeros(len(split.y), dtype=np.float64),
        "recent_level": level_signal.astype(np.float64, copy=False),
        "recent_residual": residual_signal.astype(np.float64, copy=False),
    }


train = load("train")
valid = load("valid")

x_train = make_features(train)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))
x_valid = make_features(valid)
y_valid_np = np.asarray(valid.y)

temporal_tables = build_temporal_tables(train, half_life=3.0)
valid_temporal = temporal_signals(valid, temporal_tables)

model = DeepFM(
    TOTAL_CARDINALITY,
    num_fields=len(FIELDS),
    embedding_dim=16,
)

sparse_optimizer = torch.optim.SparseAdam(
    [model.linear.weight, model.embedding.weight],
    lr=0.002,
)
dense_parameters = [
    model.bias,
    *list(model.deep.parameters()),
]
dense_optimizer = torch.optim.Adam(
    dense_parameters,
    lr=0.001,
    weight_decay=1e-6,
)

batch_size = 8192
num_epochs = 9
n = x_train.shape[0]
generator = torch.Generator()
generator.manual_seed(2024)

blend_grid = {
    "raw": [0.0],
    "recent_residual": [0.10, 0.20, 0.35, 0.50, 0.70, 1.00],
    "recent_level": [0.03, 0.06, 0.10, 0.15, 0.22, 0.30],
}

best_primary = -np.inf
best_metrics = None
best_state = None
best_mode = "raw"
best_alpha = 0.0
candidate_best = {}

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(n, generator=generator)

    for start in range(0, n, batch_size):
        idx = permutation[start:start + batch_size]
        xb = x_train[idx]
        yb = y_train[idx]

        sparse_optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(dense_parameters, max_norm=5.0)
        sparse_optimizer.step()
        dense_optimizer.step()

    base_scores = predict(model, x_valid)

    for mode, alphas in blend_grid.items():
        signal = valid_temporal[mode]
        for alpha in alphas:
            scores = base_scores + alpha * signal
            metrics = evaluate(
                valid.user_id,
                y_valid_np,
                scores,
            )
            primary = float(metrics["primary"])
            candidate_name = (
                f"epoch{epoch + 1}_{mode}_a{alpha:.2f}"
            )
            candidate_best[candidate_name] = primary

            if primary > best_primary:
                best_primary = primary
                best_metrics = {
                    key: float(value)
                    for key, value in metrics.items()
                }
                best_state = {
                    name: tensor.detach().clone()
                    for name, tensor in model.state_dict().items()
                }
                best_mode = mode
                best_alpha = float(alpha)

model.load_state_dict(best_state)
valid_base_scores = predict(model, x_valid)
valid_scores = (
    valid_base_scores
    + best_alpha * valid_temporal[best_mode]
)
best_metrics = {
    key: float(value)
    for key, value in evaluate(
        valid.user_id,
        y_valid_np,
        valid_scores,
    ).items()
}

test = load("test")
x_test = make_features(test)
test_temporal = temporal_signals(test, temporal_tables)
test_scores = (
    predict(model, x_test)
    + best_alpha * test_temporal[best_mode]
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

top_candidates = dict(
    sorted(
        candidate_best.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:12]
)
print(
    "FINDINGS "
    + json.dumps(
        {
            "selected_mode": best_mode,
            "selected_alpha": best_alpha,
            "selected_epoch_candidate": max(
                candidate_best,
                key=candidate_best.get,
            ),
            "temporal_residual_std": float(
                np.std(valid_temporal["recent_residual"])
            ),
            "temporal_level_std": float(
                np.std(valid_temporal["recent_level"])
            ),
        },
        separators=(",", ":"),
    )
)
print(
    "CANDIDATES "
    + json.dumps(top_candidates, separators=(",", ":"))
)
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": 0.0,
        },
        separators=(",", ":"),
    )
)