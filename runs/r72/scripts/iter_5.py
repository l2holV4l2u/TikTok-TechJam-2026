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


START = time.time()
ARTIFACTS = os.environ["RUN_ARTIFACTS"]
OUT_DIR = os.environ.get("ITER_OUT")

SEED = 20260829
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
]
AUX_CANDIDATES = [
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
]
EMBED_DIM = 16
EPOCHS = 5
BATCH_SIZE = 8192
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
AUX_WEIGHT = 0.12
BLEND_ALPHAS = [
    0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
    0.80, 0.90, 1.00, 1.15, 1.30, 1.50,
]

BASE_VALID_PATH = os.path.join(
    ARTIFACTS, "incumbent_valid_scores.npy"
)
BASE_TEST_PATH = os.path.join(
    ARTIFACTS, "incumbent_test_scores.npy"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))


def standardize(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(values)
    return (values - mean) / std


CARDINALITIES = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
OFFSETS = np.cumsum([0] + CARDINALITIES[:-1]).astype(np.int64)
TOTAL_CARDINALITY = int(sum(CARDINALITIES))


def encode_split(split):
    columns = []
    for field, offset, cardinality in zip(
        FIELDS, OFFSETS, CARDINALITIES
    ):
        values = np.asarray(split.X[field], dtype=np.int64)
        if np.any(values < 0) or np.any(values >= cardinality):
            raise ValueError("Categorical id outside declared cardinality")
        columns.append(values + offset)
    return np.ascontiguousarray(
        np.stack(columns, axis=1), dtype=np.int64
    )


def usable_auxiliary_names(split):
    names = []
    for name in AUX_CANDIDATES:
        if name not in split.aux:
            continue
        values = np.asarray(split.aux[name])
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        binary = finite & (values >= 0.0) & (values <= 1.0)
        if np.sum(binary) == 0:
            continue
        selected = values[binary]
        if np.min(selected) == np.max(selected):
            continue
        names.append(name)
    return names


def make_targets(split, aux_names):
    main = np.asarray(split.y, dtype=np.float32)
    n = len(main)
    targets = np.zeros(
        (n, 1 + len(aux_names)), dtype=np.float32
    )
    masks = np.zeros_like(targets, dtype=np.float32)

    targets[:, 0] = main
    masks[:, 0] = 1.0

    for j, name in enumerate(aux_names, start=1):
        raw = np.asarray(split.aux[name], dtype=np.float32)
        valid = (
            np.isfinite(raw)
            & (raw >= 0.0)
            & (raw <= 1.0)
        )
        targets[valid, j] = raw[valid]
        masks[valid, j] = 1.0

    return targets, masks


class MultiTaskFM(nn.Module):
    def __init__(
        self,
        total_cardinality,
        embedding_dim,
        num_tasks,
    ):
        super().__init__()
        self.num_tasks = num_tasks

        self.linear = nn.Embedding(
            total_cardinality, num_tasks
        )
        self.embedding = nn.Embedding(
            total_cardinality, embedding_dim
        )

        # Each outcome gets its own projection of the shared vector of
        # pairwise FM interactions. This preserves task-specific ranking
        # while forcing entity factors to learn from all outcomes.
        self.interaction_heads = nn.Parameter(
            torch.ones(embedding_dim, num_tasks)
        )
        self.bias = nn.Parameter(torch.zeros(num_tasks))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1)

        embeddings = self.embedding(x)
        summed = embeddings.sum(dim=1)
        interaction_vector = 0.5 * (
            summed.square()
            - embeddings.square().sum(dim=1)
        )
        interaction_term = (
            interaction_vector @ self.interaction_heads
        )
        return linear_term + interaction_term + self.bias


def train_model(x, targets, masks, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_tasks = targets.shape[1]
    model = MultiTaskFM(
        TOTAL_CARDINALITY,
        EMBED_DIM,
        num_tasks,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    task_weights = torch.full(
        (num_tasks,), AUX_WEIGHT, dtype=torch.float32
    )
    task_weights[0] = 1.0

    n = len(x)
    rng = np.random.default_rng(seed)
    model.train()

    epoch_losses = []
    for epoch in range(EPOCHS):
        permutation = rng.permutation(n)
        running_loss = 0.0
        running_rows = 0

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]

            xb = torch.from_numpy(x[idx])
            yb = torch.from_numpy(targets[idx])
            mb = torch.from_numpy(masks[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)

            element_loss = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            weighted_mask = mb * task_weights.unsqueeze(0)
            denominator = weighted_mask.sum().clamp_min(1.0)
            loss = (
                element_loss * weighted_mask
            ).sum() / denominator

            loss.backward()
            optimizer.step()

            rows = len(idx)
            running_loss += float(loss.detach()) * rows
            running_rows += rows

        epoch_losses.append(
            running_loss / max(1, running_rows)
        )

    return model, epoch_losses


def predict_main(model, x):
    model.eval()
    output = np.empty(len(x), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE * 2):
            end = min(len(x), start + BATCH_SIZE * 2)
            xb = torch.from_numpy(x[start:end])
            logits = model(xb)[:, 0]
            output[start:end] = (
                logits.cpu().numpy().astype(np.float64)
            )
    return output


if not os.path.exists(BASE_VALID_PATH):
    raise FileNotFoundError(BASE_VALID_PATH)
if not os.path.exists(BASE_TEST_PATH):
    raise FileNotFoundError(BASE_TEST_PATH)

train = load("train")
valid = load("valid")

aux_names = usable_auxiliary_names(train)
if not aux_names:
    raise RuntimeError("No usable auxiliary training outcomes found")

x_train = encode_split(train)
x_valid = encode_split(valid)
train_targets, train_masks = make_targets(train, aux_names)

model, train_losses = train_model(
    x_train,
    train_targets,
    train_masks,
    SEED,
)
raw_valid = predict_main(model, x_valid)

valid_users = np.asarray(valid.user_id, dtype=np.int64)
valid_labels = np.asarray(valid.y, dtype=np.int8)

base_valid_raw = np.load(BASE_VALID_PATH)
if len(base_valid_raw) != len(valid_users):
    raise ValueError("Incumbent validation length mismatch")

base_valid = standardize(base_valid_raw)
candidate_valid = standardize(raw_valid)

base_metrics = evaluate(
    valid_users, valid_labels, base_valid
)
raw_metrics = evaluate(
    valid_users, valid_labels, candidate_valid
)

best_alpha = 0.0
best_scores = base_valid.copy()
best_metrics = base_metrics
candidate_results = {
    "incumbent": float(base_metrics["primary"]),
    "multitask_fm_raw": float(raw_metrics["primary"]),
}

for alpha in BLEND_ALPHAS:
    alpha = float(alpha)
    scores = (
        (1.0 - alpha) * base_valid
        + alpha * candidate_valid
    )
    metrics = evaluate(
        valid_users, valid_labels, scores
    )
    primary = float(metrics["primary"])
    candidate_results[
        f"blend_{alpha:.2f}"
    ] = primary

    if primary > float(best_metrics["primary"]):
        best_alpha = alpha
        best_scores = scores.copy()
        best_metrics = metrics

valid_scores = np.asarray(best_scores, dtype=np.float64)

if OUT_DIR:
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(
        os.path.join(OUT_DIR, "scores_valid.npy"),
        valid_scores,
    )

aux_rates = {}
aux_coverage = {}
for j, name in enumerate(aux_names, start=1):
    mask = train_masks[:, j] > 0
    aux_coverage[name] = float(np.mean(mask))
    aux_rates[name] = float(
        np.mean(train_targets[mask, j])
    )

print(
    "FINDINGS "
    + json.dumps(
        {
            "auxiliary_tasks": aux_names,
            "auxiliary_rates": aux_rates,
            "auxiliary_coverage": aux_coverage,
            "epoch_losses": [
                round(float(v), 6) for v in train_losses
            ],
            "raw_primary": float(raw_metrics["primary"]),
            "incumbent_primary": float(
                base_metrics["primary"]
            ),
            "selected_alpha": best_alpha,
        },
        sort_keys=True,
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        {
            key: round(float(value), 6)
            for key, value in sorted(
                candidate_results.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        },
        sort_keys=True,
    )
)

base_test_raw = np.load(BASE_TEST_PATH)

if best_alpha == 0.0:
    test_scores = standardize(base_test_raw)
else:
    # Refit the identical architecture and objective on train+validation.
    # Validation outcomes, including auxiliary outcomes, are used only here
    # after validation selection, to make the submitted model less stale.
    test = load("test")
    x_test = encode_split(test)

    x_combined = np.concatenate(
        (x_train, x_valid), axis=0
    )

    valid_targets, valid_masks = make_targets(
        valid, aux_names
    )
    combined_targets = np.concatenate(
        (train_targets, valid_targets), axis=0
    )
    combined_masks = np.concatenate(
        (train_masks, valid_masks), axis=0
    )

    refit_model, refit_losses = train_model(
        x_combined,
        combined_targets,
        combined_masks,
        SEED,
    )
    raw_test = predict_main(refit_model, x_test)

    base_test = standardize(base_test_raw)
    candidate_test = standardize(raw_test)
    test_scores = (
        (1.0 - best_alpha) * base_test
        + best_alpha * candidate_test
    )

    print(
        "FINDINGS "
        + json.dumps(
            {
                "refit_epoch_losses": [
                    round(float(v), 6)
                    for v in refit_losses
                ],
                "refit_rows": int(len(x_combined)),
            },
            sort_keys=True,
        )
    )

if OUT_DIR:
    np.save(
        os.path.join(OUT_DIR, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result))