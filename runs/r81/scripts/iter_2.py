import os
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2025
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
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 32768

FAMILY_CONFIGS = {
    "additive": {"epochs": 4, "lr": 0.003},
    "expanded_fm": {"epochs": 8, "lr": 0.001},
    "deepfm": {"epochs": 4, "lr": 0.0015},
    "dcn": {"epochs": 4, "lr": 0.0015},
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    torch.set_num_threads(min(12, max(1, os.cpu_count() or 1)))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))
n_fields = len(FIELDS)


def make_matrix(split):
    cols = []
    for field, offset in zip(FIELDS, offsets):
        col = np.asarray(split.X[field], dtype=np.int64)
        cols.append(col + offset)
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


class BaseSparseModel(torch.nn.Module):
    def sparse_parameters(self):
        raise NotImplementedError

    def dense_parameters(self):
        sparse_ids = {id(p) for p in self.sparse_parameters()}
        return [p for p in self.parameters() if id(p) not in sparse_ids]


class AdditiveModel(BaseSparseModel):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Embedding(
            total_cardinality, 1, sparse=True
        )
        self.bias = torch.nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            self.linear.weight.zero_()

    def sparse_parameters(self):
        return [self.linear.weight]

    def forward(self, x):
        return self.bias + self.linear(x).squeeze(-1).sum(dim=1)


class ExpandedFM(BaseSparseModel):
    def __init__(self, rank=16):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            total_cardinality, rank + 1, sparse=True
        )
        self.bias = torch.nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(0.0, 0.01)

    def sparse_parameters(self):
        return [self.embedding.weight]

    def forward(self, x):
        z = self.embedding(x)
        linear = z[:, :, 0].sum(dim=1)
        v = z[:, :, 1:]
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1)
            - v.square().sum(dim=(1, 2))
        )
        return self.bias + linear + interaction


class DeepFM(BaseSparseModel):
    def __init__(self, rank=12):
        super().__init__()
        self.linear = torch.nn.Embedding(
            total_cardinality, 1, sparse=True
        )
        self.embedding = torch.nn.Embedding(
            total_cardinality, rank, sparse=True
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_fields * rank, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        self.bias = torch.nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            self.linear.weight.zero_()
            self.embedding.weight.normal_(0.0, 0.01)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def sparse_parameters(self):
        return [self.linear.weight, self.embedding.weight]

    def forward(self, x):
        linear = self.linear(x).squeeze(-1).sum(dim=1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        fm = 0.5 * (
            summed.square().sum(dim=1)
            - v.square().sum(dim=(1, 2))
        )
        deep = self.mlp(v.reshape(v.shape[0], -1)).squeeze(-1)
        return self.bias + linear + fm + deep


class CrossLayer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x0, x):
        projection = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x + x0 * projection + self.bias


class DCN(BaseSparseModel):
    def __init__(self, rank=8):
        super().__init__()
        dim = n_fields * rank
        self.linear = torch.nn.Embedding(
            total_cardinality, 1, sparse=True
        )
        self.embedding = torch.nn.Embedding(
            total_cardinality, rank, sparse=True
        )
        self.cross1 = CrossLayer(dim)
        self.cross2 = CrossLayer(dim)
        self.output = torch.nn.Linear(dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            self.linear.weight.zero_()
            self.embedding.weight.normal_(0.0, 0.01)
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def sparse_parameters(self):
        return [self.linear.weight, self.embedding.weight]

    def forward(self, x):
        first_order = self.linear(x).squeeze(-1).sum(dim=1)
        x0 = self.embedding(x).reshape(x.shape[0], -1)
        z = self.cross1(x0, x0)
        z = self.cross2(x0, z)
        crossed = self.output(z).squeeze(-1)
        return self.bias + first_order + crossed


def create_model(name, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if name == "additive":
        return AdditiveModel()
    if name == "expanded_fm":
        return ExpandedFM(rank=16)
    if name == "deepfm":
        return DeepFM(rank=12)
    if name == "dcn":
        return DCN(rank=8)
    raise ValueError("Unknown family: " + name)


def create_optimizers(model, lr):
    sparse_params = list(model.sparse_parameters())
    dense_params = list(model.dense_parameters())
    sparse_optimizer = torch.optim.SparseAdam(sparse_params, lr=lr)
    dense_optimizer = (
        torch.optim.Adam(dense_params, lr=lr)
        if dense_params else None
    )
    return sparse_optimizer, dense_optimizer


def train_epoch(model, sparse_optimizer, dense_optimizer,
                x, y, generator):
    model.train()
    permutation = torch.randperm(len(y), generator=generator)
    total_loss = 0.0

    for begin in range(0, len(y), BATCH_SIZE):
        idx = permutation[begin:begin + BATCH_SIZE]
        xb = x[idx]
        yb = y[idx]

        sparse_optimizer.zero_grad(set_to_none=True)
        if dense_optimizer is not None:
            dense_optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()

        sparse_optimizer.step()
        if dense_optimizer is not None:
            dense_optimizer.step()

        total_loss += float(loss.detach()) * len(idx)

    return total_loss / len(y)


def predict(model, x_np):
    model.eval()
    x = torch.from_numpy(x_np)
    scores = np.empty(len(x_np), dtype=np.float64)
    with torch.no_grad():
        for begin in range(0, len(x_np), PRED_BATCH_SIZE):
            end = min(begin + PRED_BATCH_SIZE, len(x_np))
            scores[begin:end] = (
                model(x[begin:end]).detach().cpu().numpy()
                .astype(np.float64, copy=False)
            )
    return scores


train = load("train")
valid = load("valid")

x_train_np = make_matrix(train)
x_valid_np = make_matrix(valid)
y_train_np = np.asarray(train.y, dtype=np.float32)
y_valid_np = np.asarray(valid.y, dtype=np.int8)

x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)

artifacts = os.environ.get("RUN_ARTIFACTS", "")
inc_valid_path = os.path.join(artifacts, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(artifacts, "incumbent_test_scores.npy")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if len(inc_valid) != len(y_valid_np):
    raise RuntimeError("Trusted incumbent validation predictions have wrong length")

inc_metrics = evaluate(valid.user_id, y_valid_np, inc_valid)

candidate_primary = {
    "trusted_incumbent": float(inc_metrics["primary"])
}
family_best_scores = {}
family_best_epochs = {}
family_metrics = {}

global_name = "trusted_incumbent"
global_scores = inc_valid.copy()
global_metrics = inc_metrics
global_family = None
global_alpha = 0.0

blend_alphas = np.linspace(0.0, 1.0, 11)

for family_index, (family, config) in enumerate(FAMILY_CONFIGS.items()):
    family_seed = SEED + 100 * family_index
    model = create_model(family, family_seed)
    sparse_optimizer, dense_optimizer = create_optimizers(
        model, config["lr"]
    )
    generator = torch.Generator()
    generator.manual_seed(family_seed + 1)

    best_scores = None
    best_metrics = None
    best_epoch = 1

    for epoch in range(1, config["epochs"] + 1):
        loss = train_epoch(
            model, sparse_optimizer, dense_optimizer,
            x_train, y_train, generator
        )
        valid_scores = predict(model, x_valid_np)
        metrics = evaluate(valid.user_id, y_valid_np, valid_scores)

        print(
            "family=%s epoch=%d loss=%.6f primary=%.6f "
            "gauc=%.6f ndcg@5=%.6f"
            % (
                family,
                epoch,
                loss,
                metrics["primary"],
                metrics["gauc"],
                metrics["ndcg@5"],
            ),
            flush=True,
        )

        if best_metrics is None or metrics["primary"] > best_metrics["primary"]:
            best_metrics = metrics
            best_scores = valid_scores.copy()
            best_epoch = epoch

    family_best_scores[family] = best_scores
    family_best_epochs[family] = best_epoch
    family_metrics[family] = best_metrics
    candidate_primary[family] = float(best_metrics["primary"])

    best_blend_metrics = None
    best_blend_scores = None
    best_alpha = 1.0

    for alpha in blend_alphas:
        blended = alpha * best_scores + (1.0 - alpha) * inc_valid
        blend_metrics = evaluate(
            valid.user_id, y_valid_np, blended
        )
        if (
            best_blend_metrics is None
            or blend_metrics["primary"] > best_blend_metrics["primary"]
        ):
            best_blend_metrics = blend_metrics
            best_blend_scores = blended.copy()
            best_alpha = float(alpha)

    blend_name = family + "_incumbent_blend"
    candidate_primary[blend_name] = float(
        best_blend_metrics["primary"]
    )

    print(
        "blend family=%s alpha_new=%.2f primary=%.6f "
        "gauc=%.6f ndcg@5=%.6f"
        % (
            family,
            best_alpha,
            best_blend_metrics["primary"],
            best_blend_metrics["gauc"],
            best_blend_metrics["ndcg@5"],
        ),
        flush=True,
    )

    if best_metrics["primary"] > global_metrics["primary"]:
        global_name = family
        global_scores = best_scores.copy()
        global_metrics = best_metrics
        global_family = family
        global_alpha = 1.0

    if best_blend_metrics["primary"] > global_metrics["primary"]:
        global_name = blend_name
        global_scores = best_blend_scores.copy()
        global_metrics = best_blend_metrics
        global_family = family
        global_alpha = best_alpha

    del model, sparse_optimizer, dense_optimizer
    gc.collect()


print(
    "CANDIDATES " + json.dumps(
        candidate_primary, sort_keys=True, separators=(", ", ": ")
    ),
    flush=True,
)
print(
    "FINDINGS selected=%s family=%s alpha_new=%.2f epoch=%s"
    % (
        global_name,
        str(global_family),
        global_alpha,
        str(
            family_best_epochs.get(global_family)
            if global_family is not None else None
        ),
    ),
    flush=True,
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(global_scores, dtype=np.float64),
    )


# Produce test predictions. If no new family beat the incumbent, reuse its
# already valid test predictions. Otherwise refit only the selected family
# using the same recipe and selected epoch on train + validation.
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if global_family is None or global_alpha == 0.0:
    test_scores = inc_test.copy()
else:
    x_combined_np = np.ascontiguousarray(
        np.concatenate([x_train_np, x_valid_np], axis=0),
        dtype=np.int64,
    )
    y_combined_np = np.ascontiguousarray(
        np.concatenate([
            y_train_np,
            y_valid_np.astype(np.float32, copy=False),
        ]),
        dtype=np.float32,
    )
    x_combined = torch.from_numpy(x_combined_np)
    y_combined = torch.from_numpy(y_combined_np)

    family_index = list(FAMILY_CONFIGS.keys()).index(global_family)
    final_seed = SEED + 100 * family_index
    final_model = create_model(global_family, final_seed)
    final_sparse_optimizer, final_dense_optimizer = create_optimizers(
        final_model, FAMILY_CONFIGS[global_family]["lr"]
    )
    final_generator = torch.Generator()
    final_generator.manual_seed(final_seed + 1)

    selected_epoch = family_best_epochs[global_family]
    for epoch in range(1, selected_epoch + 1):
        loss = train_epoch(
            final_model,
            final_sparse_optimizer,
            final_dense_optimizer,
            x_combined,
            y_combined,
            final_generator,
        )
        print(
            "refit family=%s epoch=%d/%d loss=%.6f"
            % (global_family, epoch, selected_epoch, loss),
            flush=True,
        )

    test = load("test")
    x_test_np = make_matrix(test)
    family_test_scores = predict(final_model, x_test_np)

    if len(inc_test) != len(family_test_scores):
        raise RuntimeError("Trusted incumbent test predictions have wrong length")

    test_scores = (
        global_alpha * family_test_scores
        + (1.0 - global_alpha) * inc_test
    )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
result = {
    "primary": float(global_metrics["primary"]),
    "gauc": float(global_metrics["gauc"]),
    "ndcg@5": float(global_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result, separators=(", ", ": ")))