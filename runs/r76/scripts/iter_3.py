import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 2026
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 32768
MAX_EPOCHS = 7

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

torch.set_num_threads(min(8, os.cpu_count() or 1))
torch.manual_seed(SEED)
np.random.seed(SEED)

CARDINALITIES = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
OFFSETS = np.asarray(
    [0] + list(np.cumsum(CARDINALITIES[:-1], dtype=np.int64)),
    dtype=np.int64,
)
TOTAL_CARDINALITY = int(sum(CARDINALITIES))


def make_local_matrix(split):
    return np.ascontiguousarray(
        np.column_stack(
            [np.asarray(split.X[field], dtype=np.int64) for field in FIELDS]
        ),
        dtype=np.int64,
    )


def offset_matrix(x):
    return np.ascontiguousarray(x + OFFSETS[None, :], dtype=np.int64)


def initial_logit(y):
    p = float(np.clip(np.mean(y), 1e-5, 1.0 - 1e-5))
    return float(np.log(p / (1.0 - p)))


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    mean = float(np.mean(x))
    scale = float(np.std(x))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return (x - mean) / scale


class VectorCrossLayer(nn.Module):
    """Original DCN rank-one/vector cross layer."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x0, x):
        scalar = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * scalar + self.bias + x


class LowRankMatrixCrossLayer(nn.Module):
    """DCN-V2 matrix cross, factorized to rank << input dimension."""

    def __init__(self, dim, rank):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)

    def forward(self, x0, x):
        crossed = self.up(self.down(x))
        return x + x0 * (crossed + self.bias)


class OriginalDCN(nn.Module):
    def __init__(self, bias, embedding_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, embedding_dim)
        dim = len(FIELDS) * embedding_dim

        self.cross1 = VectorCrossLayer(dim)
        self.cross2 = VectorCrossLayer(dim)

        self.deep = nn.Sequential(
            nn.Linear(dim, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
        )
        self.output = nn.Linear(dim + 32, 1)
        self.global_bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x0 = self.embedding(x).flatten(start_dim=1)
        cross = self.cross1(x0, x0)
        cross = self.cross2(x0, cross)
        deep = self.deep(x0)
        joined = torch.cat([cross, deep], dim=1)
        return self.global_bias + self.output(joined).squeeze(1)


class DCNV2(nn.Module):
    def __init__(self, bias, embedding_dim=8, cross_rank=32):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, embedding_dim)
        dim = len(FIELDS) * embedding_dim

        self.cross_layers = nn.ModuleList(
            [
                LowRankMatrixCrossLayer(dim, cross_rank),
                LowRankMatrixCrossLayer(dim, cross_rank),
                LowRankMatrixCrossLayer(dim, cross_rank),
            ]
        )

        self.deep = nn.Sequential(
            nn.Linear(dim, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
        )
        self.output = nn.Linear(dim + 32, 1)
        self.global_bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x0 = self.embedding(x).flatten(start_dim=1)
        cross = x0
        for layer in self.cross_layers:
            cross = layer(x0, cross)
        deep = self.deep(x0)
        joined = torch.cat([cross, deep], dim=1)
        return self.global_bias + self.output(joined).squeeze(1)


MODEL_BUILDERS = {
    "original_dcn": lambda bias: OriginalDCN(bias, embedding_dim=8),
    "dcn_v2_rank32": lambda bias: DCNV2(
        bias, embedding_dim=8, cross_rank=32
    ),
}


@torch.no_grad()
def predict(model, x_np):
    model.eval()
    x = torch.from_numpy(x_np)
    scores = np.empty(len(x_np), dtype=np.float64)

    for start in range(0, len(x_np), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(x_np))
        scores[start:end] = (
            model(x[start:end])
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )

    return scores


def fit_with_validation(
    model_name,
    x_train,
    y_train,
    x_valid,
    y_valid,
    valid_users,
):
    seed = SEED + 101 * list(MODEL_BUILDERS).index(model_name)
    torch.manual_seed(seed)

    model = MODEL_BUILDERS[model_name](initial_logit(y_train))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    xt = torch.from_numpy(x_train)
    yt = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32))

    generator = torch.Generator()
    generator.manual_seed(seed + 19)

    n = len(y_train)
    best_primary = -np.inf
    best_epoch = 1
    best_scores = None
    epoch_metrics = {}

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        total_loss = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            xb = xt.index_select(0, idx)
            yb = yt.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach()) * len(idx)

        scores = predict(model, x_valid)
        metrics = evaluate(valid_users, y_valid, scores)
        primary = float(metrics["primary"])
        epoch_metrics[epoch] = primary

        print(
            "family=%s epoch=%d loss=%.6f primary=%.6f "
            "gauc=%.6f ndcg5=%.6f"
            % (
                model_name,
                epoch,
                total_loss / n,
                primary,
                float(metrics["gauc"]),
                float(metrics["ndcg@5"]),
            ),
            flush=True,
        )

        if primary > best_primary:
            best_primary = primary
            best_epoch = epoch
            best_scores = scores.copy()

    del model, optimizer, xt, yt
    gc.collect()

    return best_scores, best_epoch, epoch_metrics


def fit_fixed_epochs(model_name, x_train, y_train, epochs):
    seed = SEED + 101 * list(MODEL_BUILDERS).index(model_name)
    torch.manual_seed(seed)

    model = MODEL_BUILDERS[model_name](initial_logit(y_train))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    xt = torch.from_numpy(x_train)
    yt = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32))

    generator = torch.Generator()
    generator.manual_seed(seed + 19)

    n = len(y_train)

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        total_loss = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            xb = xt.index_select(0, idx)
            yb = yt.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach()) * len(idx)

        print(
            "refit_family=%s epoch=%d loss=%.6f"
            % (model_name, epoch, total_loss / n),
            flush=True,
        )

    del optimizer, xt, yt
    gc.collect()
    return model


train = load("train")
valid = load("valid")

x_train_local = make_local_matrix(train)
x_valid_local = make_local_matrix(valid)
x_train = offset_matrix(x_train_local)
x_valid = offset_matrix(x_valid_local)

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

artifact_dir = os.environ["RUN_ARTIFACTS"]
inc_valid = np.asarray(
    np.load(os.path.join(artifact_dir, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_valid_z = standardize(inc_valid)

family_valid = {}
family_epochs = {}
family_epoch_metrics = {}

for family in MODEL_BUILDERS:
    scores, best_epoch, epoch_metrics = fit_with_validation(
        family,
        x_train,
        y_train,
        x_valid,
        y_valid,
        valid_users,
    )
    family_valid[family] = scores
    family_epochs[family] = best_epoch
    family_epoch_metrics[family] = epoch_metrics

candidate_scores = {}
candidate_arrays = {}
candidate_recipes = {}

inc_metrics = evaluate(valid_users, y_valid, inc_valid)
candidate_scores["incumbent"] = float(inc_metrics["primary"])
candidate_arrays["incumbent"] = inc_valid.copy()
candidate_recipes["incumbent"] = ("incumbent", 0.0)

blend_weights = np.linspace(0.0, 1.0, 21)

for family, raw_scores in family_valid.items():
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    raw_metrics = evaluate(valid_users, y_valid, raw_scores)

    candidate_scores[family] = float(raw_metrics["primary"])
    candidate_arrays[family] = raw_scores.copy()
    candidate_recipes[family] = (family, 1.0)

    family_z = standardize(raw_scores)
    best_blend_primary = -np.inf
    best_blend_scores = None
    best_weight = 1.0

    for weight in blend_weights:
        blended = weight * family_z + (1.0 - weight) * inc_valid_z
        metrics = evaluate(valid_users, y_valid, blended)
        primary = float(metrics["primary"])

        if primary > best_blend_primary:
            best_blend_primary = primary
            best_blend_scores = blended.copy()
            best_weight = float(weight)

    blend_name = "blend_" + family
    candidate_scores[blend_name] = best_blend_primary
    candidate_arrays[blend_name] = best_blend_scores
    candidate_recipes[blend_name] = (family, best_weight)

winner = max(candidate_scores, key=candidate_scores.get)
winner_family, winner_weight = candidate_recipes[winner]
valid_scores = np.asarray(candidate_arrays[winner], dtype=np.float64)
valid_metrics = evaluate(valid_users, y_valid, valid_scores)

print(
    "FINDINGS original_best_epoch=%d original_primary=%.6f "
    "dcnv2_best_epoch=%d dcnv2_primary=%.6f"
    % (
        family_epochs["original_dcn"],
        candidate_scores["original_dcn"],
        family_epochs["dcn_v2_rank32"],
        candidate_scores["dcn_v2_rank32"],
    ),
    flush=True,
)
print(
    "FINDINGS winner=%s family=%s model_weight=%.2f epoch=%d"
    % (
        winner,
        winner_family,
        winner_weight,
        family_epochs.get(winner_family, 0),
    ),
    flush=True,
)
print(
    "CANDIDATES "
    + json.dumps(
        {key: float(value) for key, value in candidate_scores.items()},
        sort_keys=True,
        separators=(", ", ": "),
    ),
    flush=True,
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        valid_scores.astype(np.float64, copy=False),
    )

inc_test = np.asarray(
    np.load(os.path.join(artifact_dir, "incumbent_test_scores.npy")),
    dtype=np.float64,
)

if winner_family == "incumbent":
    test_scores = inc_test.copy()
else:
    test = load("test")
    x_test_local = make_local_matrix(test)
    x_test = offset_matrix(x_test_local)

    y_combined = np.ascontiguousarray(
        np.concatenate(
            [y_train, np.asarray(valid.y, dtype=np.float32)]
        ),
        dtype=np.float32,
    )
    x_combined_local = np.ascontiguousarray(
        np.concatenate([x_train_local, x_valid_local], axis=0),
        dtype=np.int64,
    )
    x_combined = offset_matrix(x_combined_local)

    selected_epoch = family_epochs[winner_family]
    final_model = fit_fixed_epochs(
        winner_family,
        x_combined,
        y_combined,
        selected_epoch,
    )
    candidate_test = predict(final_model, x_test)

    if winner_weight >= 1.0 - 1e-12:
        test_scores = candidate_test
    else:
        test_scores = (
            winner_weight * standardize(candidate_test)
            + (1.0 - winner_weight) * standardize(inc_test)
        )

    del final_model
    gc.collect()

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(valid_metrics["primary"]),
            "gauc": float(valid_metrics["gauc"]),
            "ndcg@5": float(valid_metrics["ndcg@5"]),
            "gpu_seconds": elapsed,
        },
        separators=(", ", ": "),
    ),
    flush=True,
)