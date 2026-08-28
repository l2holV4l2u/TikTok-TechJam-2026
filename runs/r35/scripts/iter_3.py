import os
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


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
    "onehot_feat7",
    "user_active_degree",
    "register_days_bucket",
    "register_days_range",
]
QUALITY_FIELDS = [
    ("video_id", 30.0),
    ("author_id", 60.0),
    ("tag", 200.0),
    ("onehot_feat3", 120.0),
]
AUX_NAMES = [
    "is_click",
    "is_like",
    "is_comment",
    "is_follow",
    "is_forward",
    "is_hate",
    "is_profile_enter",
    "play_time_ms",
    "comment_stay_time",
    "profile_stay_time",
]
CONTINUOUS_AUX = {
    "play_time_ms",
    "comment_stay_time",
    "profile_stay_time",
}

EMBED_DIM = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 4096
PREDICT_BATCH_SIZE = 32768
MAX_EPOCHS = 15
PATIENCE = 3
RIDGE = 10.0
PSEUDO_CUTOFF = 20220418
FUSION_BETAS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def make_offsets(fields):
    offsets = []
    running = 0
    for name in fields:
        offsets.append(running)
        running += int(FEATURE_CARDINALITIES[name])
    return np.asarray(offsets, dtype=np.int64), running


OFFSETS, TOTAL_CARDINALITY = make_offsets(FIELDS)


def encode_split(split):
    columns = []
    for j, name in enumerate(FIELDS):
        values = np.asarray(split.X[name], dtype=np.int64)
        columns.append(values + OFFSETS[j])
    return torch.from_numpy(np.stack(columns, axis=1))


class DeepFM(nn.Module):
    def __init__(self, cardinality, num_fields, embedding_dim):
        super().__init__()
        self.linear = nn.Embedding(cardinality, 1)
        self.embedding = nn.Embedding(cardinality, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        deep_input_dim = num_fields * embedding_dim
        self.deep = nn.Sequential(
            nn.Linear(deep_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.normal_(self.deep[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.deep[-1].bias)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)

        latent = self.embedding(x)
        summed = latent.sum(dim=1)
        fm_interaction = 0.5 * (
            summed.square().sum(dim=1)
            - latent.square().sum(dim=(1, 2))
        )

        deep_term = self.deep(latent.flatten(start_dim=1)).squeeze(-1)
        return self.bias + linear_term + fm_interaction + deep_term


def predict(model, x, batch_size=PREDICT_BATCH_SIZE):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            result[start:stop] = model(x[start:stop]).cpu().numpy()
    return result


def outcome_arrays(split, mask=None):
    arrays = [np.asarray(split.y, dtype=np.float64)]
    for name in AUX_NAMES:
        values = np.asarray(split.aux[name], dtype=np.float64)
        if name in CONTINUOUS_AUX:
            values = np.log1p(np.maximum(values, 0.0))
        arrays.append(values)

    if mask is not None:
        arrays = [values[mask] for values in arrays]
    return arrays


def build_quality_features(source, target, source_mask=None, target_mask=None):
    source_outcomes = outcome_arrays(source, source_mask)

    if source_mask is None:
        source_size = len(source.y)
    else:
        source_size = int(np.count_nonzero(source_mask))

    if target_mask is None:
        target_size = len(target.y)
    else:
        target_size = int(np.count_nonzero(target_mask))

    result = np.empty(
        (target_size, len(QUALITY_FIELDS) * len(source_outcomes)),
        dtype=np.float32,
    )

    column = 0
    for field, prior_strength in QUALITY_FIELDS:
        source_ids = np.asarray(source.X[field], dtype=np.int64)
        target_ids = np.asarray(target.X[field], dtype=np.int64)

        if source_mask is not None:
            source_ids = source_ids[source_mask]
        if target_mask is not None:
            target_ids = target_ids[target_mask]

        cardinality = int(FEATURE_CARDINALITIES[field])
        counts = np.bincount(source_ids, minlength=cardinality).astype(np.float64)

        safe_target_ids = np.where(
            (target_ids >= 0) & (target_ids < cardinality),
            target_ids,
            0,
        )

        for values in source_outcomes:
            global_mean = float(values.mean()) if source_size else 0.0
            sums = np.bincount(
                source_ids,
                weights=values,
                minlength=cardinality,
            ).astype(np.float64)

            smoothed = (
                sums + prior_strength * global_mean
            ) / (counts + prior_strength)

            result[:, column] = smoothed[safe_target_ids].astype(np.float32)
            column += 1

    return result


def fit_quality_model(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0)
    z = (x - mean) / std

    design = np.empty((z.shape[0], z.shape[1] + 1), dtype=np.float64)
    design[:, 0] = 1.0
    design[:, 1:] = z

    gram = design.T @ design
    penalty = np.eye(gram.shape[0], dtype=np.float64) * RIDGE
    penalty[0, 0] = 0.0
    rhs = design.T @ y
    weights = np.linalg.solve(gram + penalty, rhs)

    fitted = design @ weights
    score_mean = float(fitted.mean())
    score_std = float(fitted.std())
    if score_std < 1e-8:
        score_std = 1.0

    return mean, std, weights, score_mean, score_std


def quality_predict(x, mean, std, weights, score_mean, score_std):
    z = (np.asarray(x, dtype=np.float64) - mean) / std
    raw = weights[0] + z @ weights[1:]
    return (raw - score_mean) / score_std


train = load("train")
valid = load("valid")

# Learn how multiple historical engagement rates predict future long-view
# using a strictly forward-in-time pseudo-validation slice inside train.
history_mask = np.asarray(train.date) <= PSEUDO_CUTOFF
future_mask = np.asarray(train.date) > PSEUDO_CUTOFF

pseudo_features = build_quality_features(
    train,
    train,
    source_mask=history_mask,
    target_mask=future_mask,
)
pseudo_labels = np.asarray(train.y, dtype=np.float64)[future_mask]

quality_mean, quality_std, quality_weights, quality_score_mean, quality_score_std = (
    fit_quality_model(pseudo_features, pseudo_labels)
)
del pseudo_features

valid_quality_features = build_quality_features(train, valid)
valid_quality_scores = quality_predict(
    valid_quality_features,
    quality_mean,
    quality_std,
    quality_weights,
    quality_score_mean,
    quality_score_std,
)
del valid_quality_features

x_train = encode_split(train)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))
x_valid = encode_split(valid)

model = DeepFM(
    cardinality=TOTAL_CARDINALITY,
    num_fields=len(FIELDS),
    embedding_dim=EMBED_DIM,
)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

generator = torch.Generator()
generator.manual_seed(SEED)

best_primary = -np.inf
best_state = None
epochs_without_improvement = 0

for epoch in range(MAX_EPOCHS):
    model.train()
    permutation = torch.randperm(x_train.shape[0], generator=generator)

    for start in range(0, x_train.shape[0], BATCH_SIZE):
        batch_idx = permutation[start:start + BATCH_SIZE]
        logits = model(x_train[batch_idx])
        loss = criterion(logits, y_train[batch_idx])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    epoch_scores = predict(model, x_valid)
    epoch_metrics = evaluate(valid.user_id, valid.y, epoch_scores)
    epoch_primary = float(epoch_metrics["primary"])

    if epoch_primary > best_primary:
        best_primary = epoch_primary
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch >= 4 and epochs_without_improvement >= PATIENCE:
        break

model.load_state_dict(best_state)
valid_deep_scores = predict(model, x_valid)

candidate_metrics = {}
best_beta = 0.0
best_metrics = None
best_valid_scores = None
best_fused_primary = -np.inf

for beta in FUSION_BETAS:
    scores = valid_deep_scores + beta * valid_quality_scores
    metrics = evaluate(valid.user_id, valid.y, scores)
    primary = float(metrics["primary"])
    candidate_metrics[f"beta_{beta:.2f}"] = primary

    if primary > best_fused_primary:
        best_fused_primary = primary
        best_beta = beta
        best_metrics = metrics
        best_valid_scores = scores

print(
    "FINDINGS "
    + json.dumps(
        {
            "pseudo_history_rows": int(np.count_nonzero(history_mask)),
            "pseudo_future_rows": int(np.count_nonzero(future_mask)),
            "quality_features": int(len(QUALITY_FIELDS) * (1 + len(AUX_NAMES))),
            "selected_beta": float(best_beta),
            "quality_valid_std": float(np.std(valid_quality_scores)),
        },
        separators=(",", ":"),
    )
)
print("CANDIDATES " + json.dumps(candidate_metrics, separators=(",", ":")))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    test = load("test")
    x_test = encode_split(test)
    test_deep_scores = predict(model, x_test)

    test_quality_features = build_quality_features(train, test)
    test_quality_scores = quality_predict(
        test_quality_features,
        quality_mean,
        quality_std,
        quality_weights,
        quality_score_mean,
        quality_score_std,
    )
    test_scores = test_deep_scores + best_beta * test_quality_scores

    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

final_metrics = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": 0.0,
}
print("METRICS " + json.dumps(final_metrics, separators=(",", ":")))