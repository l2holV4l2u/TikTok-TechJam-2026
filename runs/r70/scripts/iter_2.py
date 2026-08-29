import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 2024
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "onehot_feat3",
    "upload_type",
    "onehot_feat8",
    "duration_bucket",
    "onehot_feat1",
    "music_type",
    "user_active_degree",
]
K = 16
HIDDEN_DIMS = (64, 32)
DROPOUT = 0.10
LR = 0.001
BATCH_SIZE = 4096
MAX_EPOCHS = 8
BLEND_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
num_features = int(sum(cardinalities))


def make_matrix(split):
    columns = [
        np.asarray(split.X[name], dtype=np.int64) + offsets[j]
        for j, name in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.int64)


class DeepFM(nn.Module):
    def __init__(self, n_features, n_fields, rank, initial_rate):
        super().__init__()
        self.n_fields = n_fields
        self.rank = rank

        self.linear = nn.Embedding(n_features, 1)
        self.embedding = nn.Embedding(n_features, rank)
        self.bias = nn.Parameter(
            torch.tensor(
                np.log(initial_rate / max(1.0 - initial_rate, 1e-7)),
                dtype=torch.float32,
            )
        )

        layers = []
        input_dim = n_fields * rank
        for hidden_dim in HIDDEN_DIMS:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(DROPOUT))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.deep = nn.Sequential(*layers)

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

        final_layer = self.deep[-1]
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(final_layer.bias)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(1)

        latent = self.embedding(x)
        summed = latent.sum(dim=1)
        fm_term = 0.5 * (
            summed.square() - latent.square().sum(dim=1)
        ).sum(dim=1)

        deep_input = latent.reshape(latent.shape[0], -1)
        deep_term = self.deep(deep_input).squeeze(1)

        return self.bias + linear_term + fm_term + deep_term


def predict(model, x_np):
    model.eval()
    x = torch.from_numpy(x_np)
    result = np.empty(len(x_np), dtype=np.float64)

    with torch.no_grad():
        for start in range(0, len(x_np), BATCH_SIZE * 2):
            end = min(start + BATCH_SIZE * 2, len(x_np))
            result[start:end] = (
                model(x[start:end]).cpu().numpy().astype(np.float64)
            )

    return result


def train_one_epoch(model, optimizer, loss_fn, x, y, generator):
    model.train()
    order = torch.randperm(len(x), generator=generator)
    total_loss = 0.0

    for start in range(0, len(x), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        logits = model(x[idx])
        loss = loss_fn(logits, y[idx])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach()) * len(idx)

    return total_loss / len(x)


def new_model(initial_rate):
    return DeepFM(
        n_features=num_features,
        n_fields=len(FIELDS),
        rank=K,
        initial_rate=initial_rate,
    )


def fit_with_validation(x_train_np, y_train_np, x_valid_np, valid):
    torch.manual_seed(SEED)
    model = new_model(float(np.mean(y_train_np)))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(np.asarray(y_train_np, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(SEED)

    best_primary = -np.inf
    best_epoch = 1
    best_scores = None
    best_metrics = None

    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_one_epoch(
            model, optimizer, loss_fn, x_train, y_train, generator
        )
        scores = predict(model, x_valid_np)
        metrics = evaluate(valid.user_id, valid.y, scores)

        print(
            "epoch=%d loss=%.6f primary=%.6f gauc=%.6f ndcg5=%.6f"
            % (
                epoch,
                loss,
                float(metrics["primary"]),
                float(metrics["gauc"]),
                float(metrics["ndcg@5"]),
            ),
            flush=True,
        )

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_epoch = epoch
            best_scores = scores.copy()
            best_metrics = metrics

    return best_epoch, best_scores, best_metrics


def fit_fixed_epochs(x_np, y_np, epochs):
    torch.manual_seed(SEED)
    model = new_model(float(np.mean(y_np)))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(SEED)

    for _ in range(epochs):
        train_one_epoch(model, optimizer, loss_fn, x, y, generator)

    return model


train = load("train")
valid = load("valid")

x_train = make_matrix(train)
x_valid = make_matrix(valid)
y_train = np.asarray(train.y, dtype=np.float32)

selected_epochs, deep_valid_scores, deep_metrics = fit_with_validation(
    x_train, y_train, x_valid, valid
)

artifacts_dir = os.environ.get("RUN_ARTIFACTS", "")
incumbent_valid_path = os.path.join(
    artifacts_dir, "incumbent_valid_scores.npy"
)
incumbent_test_path = os.path.join(
    artifacts_dir, "incumbent_test_scores.npy"
)

candidate_metrics = {}
chosen_weight = 1.0
valid_scores = deep_valid_scores
metrics = deep_metrics

if os.path.exists(incumbent_valid_path):
    incumbent_valid = np.asarray(
        np.load(incumbent_valid_path), dtype=np.float64
    )

    if len(incumbent_valid) == len(deep_valid_scores):
        best_primary = -np.inf

        for weight in BLEND_WEIGHTS:
            blended = (
                weight * deep_valid_scores
                + (1.0 - weight) * incumbent_valid
            )
            blend_metrics = evaluate(valid.user_id, valid.y, blended)
            name = "deep_weight_%.2f" % weight
            candidate_metrics[name] = float(blend_metrics["primary"])

            if float(blend_metrics["primary"]) > best_primary:
                best_primary = float(blend_metrics["primary"])
                chosen_weight = float(weight)
                valid_scores = blended.copy()
                metrics = blend_metrics
else:
    candidate_metrics["deep_only"] = float(deep_metrics["primary"])

print(
    "CANDIDATES "
    + json.dumps(candidate_metrics, sort_keys=True, separators=(",", ":")),
    flush=True,
)
print(
    "FINDINGS selected_epoch=%d selected_deep_weight=%.2f deep_primary=%.6f"
    % (
        selected_epochs,
        chosen_weight,
        float(deep_metrics["primary"]),
    ),
    flush=True,
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

test = load("test")
x_test = make_matrix(test)

x_combined = np.ascontiguousarray(
    np.concatenate([x_train, x_valid], axis=0),
    dtype=np.int64,
)
y_combined = np.ascontiguousarray(
    np.concatenate(
        [
            y_train,
            np.asarray(valid.y, dtype=np.float32),
        ],
        axis=0,
    ),
    dtype=np.float32,
)

final_model = fit_fixed_epochs(
    x_combined,
    y_combined,
    selected_epochs,
)
deep_test_scores = predict(final_model, x_test)
test_scores = deep_test_scores

if (
    chosen_weight < 1.0
    and os.path.exists(incumbent_test_path)
):
    incumbent_test = np.asarray(
        np.load(incumbent_test_path), dtype=np.float64
    )
    if len(incumbent_test) == len(deep_test_scores):
        test_scores = (
            chosen_weight * deep_test_scores
            + (1.0 - chosen_weight) * incumbent_test
        )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START_TIME
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(",", ":"),
    )
)