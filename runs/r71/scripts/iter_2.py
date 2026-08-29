import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2024
K = 16
LR = 0.001
BATCH_SIZE = 4096
MAX_EPOCHS = 6

# Retain the baseline fields and add informative user, item, and context fields
# whose higher-order interactions can be represented by the deep tower.
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "video_type",
    "music_type",
    "hour",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_bucket",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_matrix(split):
    columns = [
        np.asarray(split.X[name], dtype=np.int64) + offsets[j]
        for j, name in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.int64)


class DeepFM(nn.Module):
    def __init__(self, n_categories, rank, n_fields, initial_bias):
        super().__init__()
        # Column zero is the wide/FM linear coefficient; the remaining columns
        # are shared by the FM interaction and nonlinear tower.
        self.embedding = nn.Embedding(
            n_categories, rank + 1, sparse=True
        )
        self.bias = nn.Parameter(
            torch.tensor(float(initial_bias), dtype=torch.float32)
        )
        self.deep = nn.Sequential(
            nn.Linear(n_fields * rank, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1),
        )

        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(mean=0.0, std=0.01)

        for layer in self.deep:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5.0))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Begin with a small nonlinear residual rather than overwhelming the
        # well-behaved FM component at initialization.
        with torch.no_grad():
            self.deep[-1].weight.mul_(0.05)

    def forward(self, x):
        embedded = self.embedding(x)
        linear = embedded[:, :, 0].sum(dim=1)
        vectors = embedded[:, :, 1:]

        summed = vectors.sum(dim=1)
        fm_interaction = 0.5 * (
            summed.square() - vectors.square().sum(dim=1)
        ).sum(dim=1)

        deep_logit = self.deep(
            vectors.reshape(vectors.shape[0], -1)
        ).squeeze(1)

        return self.bias + linear + fm_interaction + deep_logit


def fit_model(X, y, epochs, seed):
    torch.manual_seed(seed)

    y = np.asarray(y, dtype=np.float32)
    positive_rate = float(np.mean(y))
    initial_bias = np.log(
        np.clip(positive_rate, 1e-6, 1.0 - 1e-6)
        / np.clip(1.0 - positive_rate, 1e-6, 1.0)
    )

    model = DeepFM(
        total_cardinality,
        K,
        len(FIELDS),
        initial_bias,
    )

    sparse_optimizer = torch.optim.SparseAdam(
        [model.embedding.weight], lr=LR
    )
    dense_parameters = [
        p for name, p in model.named_parameters()
        if name != "embedding.weight"
    ]
    dense_optimizer = torch.optim.Adam(
        dense_parameters, lr=LR, weight_decay=1e-6
    )

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    generator = torch.Generator()
    generator.manual_seed(seed + 17)
    n = len(y)

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(n, generator=generator)
        running_loss = 0.0
        seen = 0

        for begin in range(0, n, BATCH_SIZE):
            idx = permutation[begin:begin + BATCH_SIZE]
            xb = X_tensor[idx]
            yb = y_tensor[idx]

            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(dense_parameters, max_norm=5.0)
            sparse_optimizer.step()
            dense_optimizer.step()

            running_loss += float(loss.detach()) * len(idx)
            seen += len(idx)

        print(
            "FINDINGS "
            + json.dumps(
                {
                    "phase": "fit",
                    "epoch": epoch + 1,
                    "loss": running_loss / max(seen, 1),
                }
            )
        )

    return model


def predict(model, X):
    model.eval()
    X_tensor = torch.from_numpy(X)
    scores = np.empty(len(X), dtype=np.float32)

    with torch.no_grad():
        for begin in range(0, len(X), 16384):
            end = min(begin + 16384, len(X))
            scores[begin:end] = (
                model(X_tensor[begin:end]).cpu().numpy()
            )

    return scores


def load_incumbent_predictions(valid_length):
    artifact_dir = os.environ.get("RUN_ARTIFACTS")
    if not artifact_dir:
        return None, None

    valid_path = os.path.join(
        artifact_dir, "incumbent_valid_scores.npy"
    )
    test_path = os.path.join(
        artifact_dir, "incumbent_test_scores.npy"
    )

    if not (os.path.exists(valid_path) and os.path.exists(test_path)):
        return None, None

    incumbent_valid = np.asarray(
        np.load(valid_path), dtype=np.float64
    )
    incumbent_test = np.asarray(
        np.load(test_path), dtype=np.float64
    )

    if len(incumbent_valid) != valid_length:
        return None, None

    return incumbent_valid, incumbent_test


# Train only on the official training split for validation measurement.
train = load("train")
valid = load("valid")

X_train = make_matrix(train)
X_valid = make_matrix(valid)
y_train = np.asarray(train.y, dtype=np.float32)

model = fit_model(X_train, y_train, MAX_EPOCHS, SEED)
deep_valid_scores = predict(model, X_valid).astype(np.float64)

incumbent_valid, incumbent_test = load_incumbent_predictions(
    len(deep_valid_scores)
)

candidate_scores = {}
candidate_metrics = {}
candidate_alphas = {}

if incumbent_valid is None:
    candidate_scores["deepfm"] = deep_valid_scores
    candidate_alphas["deepfm"] = 1.0
else:
    # Alpha is the DeepFM weight. Selection uses validation only, and exactly
    # the selected alpha is subsequently applied to test predictions.
    for alpha in np.linspace(0.0, 1.0, 11):
        name = "deepfm_blend_{:.1f}".format(alpha)
        scores = (
            alpha * deep_valid_scores
            + (1.0 - alpha) * incumbent_valid
        )
        candidate_scores[name] = scores
        candidate_alphas[name] = float(alpha)

for name, scores in candidate_scores.items():
    candidate_metrics[name] = evaluate(
        valid.user_id, valid.y, scores
    )

best_name = max(
    candidate_metrics,
    key=lambda name: candidate_metrics[name]["primary"],
)
best_alpha = candidate_alphas[best_name]
valid_scores = candidate_scores[best_name]
metrics = candidate_metrics[best_name]

print(
    "CANDIDATES "
    + json.dumps(
        {
            name: float(result["primary"])
            for name, result in candidate_metrics.items()
        },
        sort_keys=True,
    )
)
print(
    "FINDINGS "
    + json.dumps(
        {
            "selected": best_name,
            "deepfm_weight": best_alpha,
            "selected_primary": float(metrics["primary"]),
        }
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Refit the identical DeepFM recipe on train + validation and score test.
test = load("test")
X_test = make_matrix(test)

X_combined = np.ascontiguousarray(
    np.concatenate([X_train, X_valid], axis=0),
    dtype=np.int64,
)
y_combined = np.concatenate(
    [
        np.asarray(train.y, dtype=np.float32),
        np.asarray(valid.y, dtype=np.float32),
    ]
)

del model
combined_model = fit_model(
    X_combined, y_combined, MAX_EPOCHS, SEED
)
deep_test_scores = predict(
    combined_model, X_test
).astype(np.float64)

if (
    incumbent_test is not None
    and len(incumbent_test) == len(deep_test_scores)
):
    test_scores = (
        best_alpha * deep_test_scores
        + (1.0 - best_alpha) * incumbent_test
    )
else:
    # This branch can only occur if reusable artifacts are unavailable or
    # malformed; validation then selected the standalone model.
    test_scores = deep_test_scores

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)