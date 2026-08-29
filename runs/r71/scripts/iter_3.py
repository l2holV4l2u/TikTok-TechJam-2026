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
CROSS_RANK = 32
N_CROSS_LAYERS = 2

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
    return np.ascontiguousarray(
        np.column_stack(columns), dtype=np.int64
    )


class LowRankCrossLayer(nn.Module):
    def __init__(self, input_dim, rank):
        super().__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, input_dim, bias=True)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.up.bias)

    def forward(self, x0, x):
        cross_transform = self.up(self.down(x))
        return x + x0 * cross_transform


class CrossDeepFM(nn.Module):
    def __init__(
        self,
        n_categories,
        rank,
        n_fields,
        initial_bias,
        cross_rank,
        n_cross_layers,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            n_categories, rank + 1, sparse=True
        )
        self.bias = nn.Parameter(
            torch.tensor(float(initial_bias), dtype=torch.float32)
        )

        input_dim = n_fields * rank

        self.deep = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1),
        )

        self.cross_layers = nn.ModuleList(
            [
                LowRankCrossLayer(input_dim, cross_rank)
                for _ in range(n_cross_layers)
            ]
        )
        self.cross_output = nn.Linear(input_dim, 1, bias=False)

        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(
                mean=0.0, std=0.01
            )

        for layer in self.deep:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(
                    layer.weight, a=np.sqrt(5.0)
                )
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        with torch.no_grad():
            self.deep[-1].weight.mul_(0.05)
            self.cross_output.weight.normal_(
                mean=0.0, std=0.01
            )

    def forward(self, x):
        embedded = self.embedding(x)
        linear = embedded[:, :, 0].sum(dim=1)
        vectors = embedded[:, :, 1:]

        summed = vectors.sum(dim=1)
        fm_interaction = 0.5 * (
            summed.square() - vectors.square().sum(dim=1)
        ).sum(dim=1)

        flat_vectors = vectors.reshape(vectors.shape[0], -1)

        deep_logit = self.deep(flat_vectors).squeeze(1)

        crossed = flat_vectors
        for cross_layer in self.cross_layers:
            crossed = cross_layer(flat_vectors, crossed)
        cross_logit = self.cross_output(crossed).squeeze(1)

        return (
            self.bias
            + linear
            + fm_interaction
            + deep_logit
            + cross_logit
        )


def fit_model(X, y, epochs, seed):
    torch.manual_seed(seed)

    y = np.asarray(y, dtype=np.float32)
    positive_rate = float(np.mean(y))
    initial_bias = np.log(
        np.clip(positive_rate, 1e-6, 1.0 - 1e-6)
        / np.clip(1.0 - positive_rate, 1e-6, 1.0)
    )

    model = CrossDeepFM(
        total_cardinality,
        K,
        len(FIELDS),
        initial_bias,
        CROSS_RANK,
        N_CROSS_LAYERS,
    )

    sparse_optimizer = torch.optim.SparseAdam(
        [model.embedding.weight], lr=LR
    )
    dense_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name != "embedding.weight"
    ]
    dense_optimizer = torch.optim.Adam(
        dense_parameters,
        lr=LR,
        weight_decay=1e-6,
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
            loss = F.binary_cross_entropy_with_logits(
                logits, yb
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                dense_parameters, max_norm=5.0
            )
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
                    "cross_layers": N_CROSS_LAYERS,
                    "cross_rank": CROSS_RANK,
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

    if not (
        os.path.exists(valid_path)
        and os.path.exists(test_path)
    ):
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


train = load("train")
valid = load("valid")

X_train = make_matrix(train)
X_valid = make_matrix(valid)
y_train = np.asarray(train.y, dtype=np.float32)

model = fit_model(
    X_train,
    y_train,
    MAX_EPOCHS,
    SEED,
)
cross_valid_scores = predict(
    model, X_valid
).astype(np.float64)

incumbent_valid, incumbent_test = load_incumbent_predictions(
    len(cross_valid_scores)
)

candidate_scores = {}
candidate_metrics = {}
candidate_alphas = {}

if incumbent_valid is None:
    candidate_scores["cross_deepfm"] = cross_valid_scores
    candidate_alphas["cross_deepfm"] = 1.0
else:
    for alpha in np.linspace(0.0, 1.0, 11):
        name = "cross_deepfm_blend_{:.1f}".format(alpha)
        scores = (
            alpha * cross_valid_scores
            + (1.0 - alpha) * incumbent_valid
        )
        candidate_scores[name] = scores
        candidate_alphas[name] = float(alpha)

for name, scores in candidate_scores.items():
    candidate_metrics[name] = evaluate(
        valid.user_id,
        valid.y,
        scores,
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
            "cross_deepfm_weight": best_alpha,
            "standalone_primary": float(
                evaluate(
                    valid.user_id,
                    valid.y,
                    cross_valid_scores,
                )["primary"]
            ),
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
    X_combined,
    y_combined,
    MAX_EPOCHS,
    SEED,
)
cross_test_scores = predict(
    combined_model, X_test
).astype(np.float64)

if (
    incumbent_test is not None
    and len(incumbent_test) == len(cross_test_scores)
):
    test_scores = (
        best_alpha * cross_test_scores
        + (1.0 - best_alpha) * incumbent_test
    )
else:
    test_scores = cross_test_scores

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