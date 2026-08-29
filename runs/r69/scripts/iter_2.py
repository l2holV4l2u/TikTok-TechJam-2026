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


START = time.time()
SEED = 2024
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "user_active_degree",
    "register_days_range",
    "register_days_bucket",
    "is_live_streamer",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat3",
    "onehot_feat6",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat11",
    "onehot_feat12",
]
EMBED_DIM = 16
HIDDEN_DIMS = (128, 64)
DROPOUT = 0.10
LR = 0.001
BATCH_SIZE = 4096
MAX_EPOCHS = 12
PATIENCE = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_matrix(split):
    columns = [
        np.asarray(split.X[field], dtype=np.int64) + offsets[j]
        for j, field in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.int64)


class DeepFM(nn.Module):
    def __init__(
        self,
        num_embeddings,
        num_fields,
        embedding_dim,
        hidden_dims,
        dropout,
        initial_bias,
    ):
        super().__init__()
        self.linear = nn.Embedding(num_embeddings, 1, sparse=True)
        self.latent = nn.Embedding(
            num_embeddings, embedding_dim, sparse=True
        )
        self.bias = nn.Parameter(
            torch.tensor(float(initial_bias), dtype=torch.float32)
        )

        layers = []
        input_dim = num_fields * embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.deep = nn.Sequential(*layers)

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, mean=0.0, std=0.01)

        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

        final_layer = self.deep[-1]
        nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(final_layer.bias)

    def forward(self, x):
        linear_term = self.linear(x).squeeze(-1).sum(dim=1)

        embeddings = self.latent(x)
        summed = embeddings.sum(dim=1)
        fm_interaction = 0.5 * (
            summed.square() - embeddings.square().sum(dim=1)
        ).sum(dim=1)

        deep_input = embeddings.reshape(embeddings.shape[0], -1)
        deep_term = self.deep(deep_input).squeeze(-1)

        return self.bias + linear_term + fm_interaction + deep_term


@torch.no_grad()
def predict(model, x_np, batch_size=32768):
    model.eval()
    result = np.empty(x_np.shape[0], dtype=np.float64)
    for begin in range(0, x_np.shape[0], batch_size):
        end = min(begin + batch_size, x_np.shape[0])
        xb = torch.from_numpy(x_np[begin:end])
        result[begin:end] = (
            model(xb).cpu().numpy().astype(np.float64)
        )
    return result


train = load("train")
valid = load("valid")

x_train_np = make_matrix(train)
x_valid_np = make_matrix(valid)
y_train_np = np.asarray(train.y, dtype=np.float32)
y_valid_np = np.asarray(valid.y, dtype=np.int8)

x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)

positive_rate = float(
    np.clip(y_train_np.mean(), 1e-6, 1.0 - 1e-6)
)
initial_bias = np.log(positive_rate / (1.0 - positive_rate))

model = DeepFM(
    num_embeddings=total_cardinality,
    num_fields=len(FIELDS),
    embedding_dim=EMBED_DIM,
    hidden_dims=HIDDEN_DIMS,
    dropout=DROPOUT,
    initial_bias=initial_bias,
)

sparse_optimizer = torch.optim.SparseAdam(
    [model.linear.weight, model.latent.weight],
    lr=LR,
)
dense_parameters = [
    parameter
    for name, parameter in model.named_parameters()
    if name not in {"linear.weight", "latent.weight"}
]
dense_optimizer = torch.optim.Adam(
    dense_parameters,
    lr=LR,
    weight_decay=1e-6,
)

best_primary = -np.inf
best_state = None
best_deepfm_valid_scores = None
best_deepfm_metrics = None
stale_epochs = 0
n_train = x_train.shape[0]

for epoch in range(MAX_EPOCHS):
    model.train()
    permutation = torch.randperm(n_train)

    for begin in range(0, n_train, BATCH_SIZE):
        idx = permutation[begin:min(begin + BATCH_SIZE, n_train)]
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

    epoch_scores = predict(model, x_valid_np)
    epoch_metrics = evaluate(
        valid.user_id, y_valid_np, epoch_scores
    )
    epoch_primary = float(epoch_metrics["primary"])

    if epoch_primary > best_primary + 1e-5:
        best_primary = epoch_primary
        best_state = copy.deepcopy(model.state_dict())
        best_deepfm_valid_scores = epoch_scores.copy()
        best_deepfm_metrics = epoch_metrics
        stale_epochs = 0
    else:
        stale_epochs += 1

    if stale_epochs >= PATIENCE:
        break

model.load_state_dict(best_state)

# Compare the new representation directly and in a validation-selected
# residual blend with the trusted incumbent FM. The same selected weight
# is subsequently applied to test predictions.
artifacts = os.environ.get("RUN_ARTIFACTS", "")
inc_valid_path = os.path.join(
    artifacts, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    artifacts, "incumbent_test_scores.npy"
)

candidate_metrics = {
    "deepfm": best_deepfm_metrics
}
selected_alpha = 1.0
valid_scores = best_deepfm_valid_scores
final_metrics = best_deepfm_metrics

if os.path.exists(inc_valid_path):
    incumbent_valid = np.asarray(
        np.load(inc_valid_path), dtype=np.float64
    )
    if incumbent_valid.shape == best_deepfm_valid_scores.shape:
        incumbent_metrics = evaluate(
            valid.user_id, y_valid_np, incumbent_valid
        )
        candidate_metrics["incumbent"] = incumbent_metrics

        for alpha in np.linspace(0.0, 1.0, 11):
            blended = (
                (1.0 - alpha) * incumbent_valid
                + alpha * best_deepfm_valid_scores
            )
            metrics = evaluate(
                valid.user_id, y_valid_np, blended
            )
            name = "blend_{:.1f}".format(alpha)
            candidate_metrics[name] = metrics

            if float(metrics["primary"]) > float(
                final_metrics["primary"]
            ):
                selected_alpha = float(alpha)
                valid_scores = blended.copy()
                final_metrics = metrics

candidate_report = {
    name: float(metrics["primary"])
    for name, metrics in candidate_metrics.items()
}
print("CANDIDATES " + json.dumps(candidate_report, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "selected_deepfm_weight": selected_alpha,
            "best_epoch_deepfm_primary": float(
                best_deepfm_metrics["primary"]
            ),
            "epochs_run": epoch + 1,
        },
        sort_keys=True,
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
x_test_np = make_matrix(test)
deepfm_test_scores = predict(model, x_test_np)
test_scores = deepfm_test_scores

if (
    selected_alpha < 1.0
    and os.path.exists(inc_test_path)
):
    incumbent_test = np.asarray(
        np.load(inc_test_path), dtype=np.float64
    )
    if incumbent_test.shape == deepfm_test_scores.shape:
        test_scores = (
            (1.0 - selected_alpha) * incumbent_test
            + selected_alpha * deepfm_test_scores
        )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
report = {
    "primary": float(final_metrics["primary"]),
    "gauc": float(final_metrics["gauc"]),
    "ndcg@5": float(final_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(report))