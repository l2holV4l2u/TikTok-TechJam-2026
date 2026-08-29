import os
import time
import json
import copy
import gc
import numpy as np
import torch
import torch.nn as nn

_start_time = time.time()

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


SEED = 20260828
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

N_THREADS = min(16, max(1, os.cpu_count() or 1))
torch.set_num_threads(N_THREADS)
try:
    torch.set_num_interop_threads(min(4, N_THREADS))
except RuntimeError:
    pass


CAT_FIELDS = [
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
    "fans_user_num_range",
    "hour",
    "is_video_author",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

EMBED_DIM = 8
CROSS_RANK = 32
N_CROSS_LAYERS = 3
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 16384
EPOCHS = 5


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    positions = (
        np.arange(n, dtype=np.float64)
        - np.repeat(starts, lengths).astype(np.float64)
    )
    denominators = np.maximum(np.repeat(lengths, lengths) - 1, 1)

    ranked = positions / denominators
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def z_params(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    std = max(float(np.std(values)), 1e-8)
    return mean, std


def apply_z(values, mean, std):
    return (np.asarray(values, dtype=np.float64) - mean) / std


def categorical_matrix(split):
    columns = []
    for name in CAT_FIELDS:
        values = np.asarray(split.X[name], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[name])
        values = np.clip(values, 0, card - 1)
        columns.append(values.astype(np.int32))
    return np.column_stack(columns).astype(np.int32, copy=False)


def raw_numeric_matrix(split):
    columns = []

    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.log1p(np.maximum(x, 0.0)).astype(np.float32)
        columns.append(x)

    for key in ("video_id", "author_id"):
        hist = historical_features(split_name_for_history[split], key=key)
        for name in sorted(hist.keys()):
            x = np.asarray(hist[name], dtype=np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            columns.append(x)

    return np.column_stack(columns).astype(np.float32, copy=False)


def normalize_numeric(train_matrix, other_matrix=None):
    means = np.mean(train_matrix, axis=0, dtype=np.float64).astype(np.float32)
    stds = np.std(train_matrix, axis=0, dtype=np.float64).astype(np.float32)
    stds = np.maximum(stds, 1e-4)

    normalized_train = (train_matrix - means) / stds
    normalized_train = np.clip(normalized_train, -8.0, 8.0).astype(
        np.float32
    )

    if other_matrix is None:
        return normalized_train, means, stds

    normalized_other = (other_matrix - means) / stds
    normalized_other = np.clip(normalized_other, -8.0, 8.0).astype(
        np.float32
    )
    return normalized_train, normalized_other, means, stds


class LowRankCrossLayer(nn.Module):
    def __init__(self, dimension, rank):
        super().__init__()
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        self.bias = nn.Parameter(torch.zeros(dimension))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)

    def forward(self, x0, x):
        projected = self.up(torch.tanh(self.down(x)))
        return x + x0 * (projected + self.bias)


class DCNV2Ranker(nn.Module):
    def __init__(self, cardinalities, num_numeric):
        super().__init__()

        offsets = np.cumsum([0] + cardinalities[:-1]).astype(np.int64)
        self.register_buffer(
            "offsets", torch.as_tensor(offsets, dtype=torch.long)
        )

        total_cardinality = int(sum(cardinalities))
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)
        self.linear_embedding = nn.Embedding(total_cardinality, 1)

        input_dim = len(cardinalities) * EMBED_DIM + num_numeric
        self.numeric_projection = nn.Sequential(
            nn.LayerNorm(num_numeric),
            nn.Linear(num_numeric, 32),
            nn.SiLU(),
            nn.Linear(32, num_numeric),
        )

        self.cross_layers = nn.ModuleList(
            [
                LowRankCrossLayer(input_dim, CROSS_RANK)
                for _ in range(N_CROSS_LAYERS)
            ]
        )

        self.deep = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(192, 96),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 48),
            nn.SiLU(),
        )

        self.cross_head = nn.Linear(input_dim, 1)
        self.deep_head = nn.Linear(48, 1)
        self.numeric_linear = nn.Linear(num_numeric, 1, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(()))

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear_embedding.weight)

    def forward(self, cat, numeric):
        indices = cat.long() + self.offsets
        embeddings = self.embedding(indices).flatten(1)

        numeric_enhanced = numeric + 0.20 * self.numeric_projection(numeric)
        x0 = torch.cat([embeddings, numeric_enhanced], dim=1)

        crossed = x0
        for layer in self.cross_layers:
            crossed = layer(x0, crossed)

        deep = self.deep(x0)
        wide = self.linear_embedding(indices).sum(dim=1)

        logits = (
            wide
            + self.cross_head(crossed).squeeze(1)
            + self.deep_head(deep).squeeze(1)
            + self.numeric_linear(numeric).squeeze(1)
            + self.output_bias
        )
        return logits


@torch.no_grad()
def predict_model(model, cat_tensor, num_tensor):
    model.eval()
    outputs = []
    n = cat_tensor.shape[0]

    for start in range(0, n, PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, n)
        logits = model(
            cat_tensor[start:end],
            num_tensor[start:end],
        )
        outputs.append(logits.cpu().numpy())

    return np.concatenate(outputs).astype(np.float64)


artifacts = os.environ.get("RUN_ARTIFACTS", "")
inc_valid_path = os.path.join(artifacts, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(artifacts, "incumbent_test_scores.npy")

if not os.path.isfile(inc_valid_path):
    raise FileNotFoundError("Missing trusted incumbent validation scores")
if not os.path.isfile(inc_test_path):
    raise FileNotFoundError("Missing trusted incumbent test scores")

train = load("train")
valid = load("valid")

split_name_for_history = {
    train: "train",
    valid: "valid",
}

incumbent_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if len(incumbent_valid) != len(valid.y):
    raise ValueError("Incumbent validation length mismatch")

incumbent_metrics = evaluate(
    valid.user_id,
    valid.y,
    incumbent_valid,
)

print(
    "FINDINGS incumbent primary=%.6f gauc=%.6f ndcg5=%.6f"
    % (
        float(incumbent_metrics["primary"]),
        float(incumbent_metrics["gauc"]),
        float(incumbent_metrics["ndcg@5"]),
    )
)

cat_train_np = categorical_matrix(train)
cat_valid_np = categorical_matrix(valid)

raw_num_train = raw_numeric_matrix(train)
raw_num_valid = raw_numeric_matrix(valid)

num_train_np, num_valid_np, num_means, num_stds = normalize_numeric(
    raw_num_train,
    raw_num_valid,
)

del raw_num_train, raw_num_valid
gc.collect()

print(
    "FINDINGS dcn_input categorical_fields=%d numeric_history_features=%d "
    "cross_layers=%d cross_rank=%d"
    % (
        len(CAT_FIELDS),
        num_train_np.shape[1],
        N_CROSS_LAYERS,
        CROSS_RANK,
    )
)

cat_train = torch.from_numpy(cat_train_np)
cat_valid = torch.from_numpy(cat_valid_np)
num_train = torch.from_numpy(num_train_np)
num_valid = torch.from_numpy(num_valid_np)
labels_train = torch.from_numpy(
    np.asarray(train.y, dtype=np.float32)
)

train_dates = np.asarray(train.date, dtype=np.int64)
days_old = np.maximum(int(train_dates.max()) - train_dates, 0)
row_weights_np = np.exp(-0.018 * days_old).astype(np.float32)
row_weights = torch.from_numpy(row_weights_np)

cardinalities = [int(FEATURE_CARDINALITIES[x]) for x in CAT_FIELDS]
model = DCNV2Ranker(cardinalities, num_train_np.shape[1])

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1.4e-3,
    weight_decay=2.0e-5,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=3.5e-4,
)

loss_function = nn.BCEWithLogitsLoss(reduction="none")

inc_mean, inc_std = z_params(incumbent_valid)
inc_z_valid = apply_z(incumbent_valid, inc_mean, inc_std)
inc_rank_valid = within_user_rank(valid.user_id, incumbent_valid)

best = {
    "primary": float(incumbent_metrics["primary"]),
    "metrics": incumbent_metrics,
    "scores": incumbent_valid.copy(),
    "mode": "incumbent",
    "alpha": 0.0,
    "epoch": 0,
    "candidate_mean": 0.0,
    "candidate_std": 1.0,
    "state": None,
    "name": "incumbent",
}

candidate_log = {
    "incumbent": float(incumbent_metrics["primary"])
}

n_train = len(train.y)

for epoch in range(1, EPOCHS + 1):
    model.train()
    permutation = rng.permutation(n_train)
    total_loss = 0.0
    total_weight = 0.0

    for start in range(0, n_train, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_train)
        idx_np = permutation[start:end]
        idx = torch.from_numpy(idx_np)

        batch_cat = cat_train[idx]
        batch_num = num_train[idx]
        batch_y = labels_train[idx]
        batch_w = row_weights[idx]

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_cat, batch_num)
        losses = loss_function(logits, batch_y)
        loss = torch.sum(losses * batch_w) / torch.sum(batch_w)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += float(torch.sum(losses * batch_w).detach())
        total_weight += float(torch.sum(batch_w))

    scheduler.step()

    candidate_valid = predict_model(model, cat_valid, num_valid)
    raw_metrics = evaluate(valid.user_id, valid.y, candidate_valid)
    raw_name = "dcn_epoch%d_raw" % epoch
    candidate_log[raw_name] = float(raw_metrics["primary"])

    candidate_mean, candidate_std = z_params(candidate_valid)
    candidate_z = apply_z(
        candidate_valid, candidate_mean, candidate_std
    )
    candidate_rank = within_user_rank(valid.user_id, candidate_valid)

    epoch_options = [
        (
            raw_name,
            "raw",
            1.0,
            candidate_valid,
            raw_metrics,
        )
    ]

    for alpha in np.arange(0.05, 0.651, 0.05):
        alpha = float(alpha)

        z_scores = (
            (1.0 - alpha) * inc_z_valid
            + alpha * candidate_z
        )
        z_metrics = evaluate(valid.user_id, valid.y, z_scores)
        z_name = "dcn_epoch%d_z_%.2f" % (epoch, alpha)
        candidate_log[z_name] = float(z_metrics["primary"])
        epoch_options.append(
            (z_name, "zblend", alpha, z_scores, z_metrics)
        )

        rank_scores = (
            (1.0 - alpha) * inc_rank_valid
            + alpha * candidate_rank
        )
        rank_metrics = evaluate(valid.user_id, valid.y, rank_scores)
        rank_name = "dcn_epoch%d_rank_%.2f" % (epoch, alpha)
        candidate_log[rank_name] = float(rank_metrics["primary"])
        epoch_options.append(
            (
                rank_name,
                "rankblend",
                alpha,
                rank_scores,
                rank_metrics,
            )
        )

    epoch_best = max(
        epoch_options,
        key=lambda item: float(item[4]["primary"]),
    )

    if float(epoch_best[4]["primary"]) > best["primary"]:
        best = {
            "primary": float(epoch_best[4]["primary"]),
            "metrics": epoch_best[4],
            "scores": np.asarray(epoch_best[3], dtype=np.float64).copy(),
            "mode": epoch_best[1],
            "alpha": float(epoch_best[2]),
            "epoch": epoch,
            "candidate_mean": candidate_mean,
            "candidate_std": candidate_std,
            "state": copy.deepcopy(model.state_dict()),
            "name": epoch_best[0],
        }

    print(
        "FINDINGS epoch=%d train_loss=%.6f raw_primary=%.6f "
        "epoch_best=%s epoch_best_primary=%.6f global_best=%.6f"
        % (
            epoch,
            total_loss / max(total_weight, 1e-8),
            float(raw_metrics["primary"]),
            epoch_best[0],
            float(epoch_best[4]["primary"]),
            float(best["primary"]),
        )
    )

top_candidates = sorted(
    candidate_log.items(),
    key=lambda item: item[1],
    reverse=True,
)[:20]

print(
    "CANDIDATES "
    + json.dumps(
        {name: score for name, score in top_candidates},
        separators=(", ", ": "),
    )
)

print(
    "FINDINGS selected=%s mode=%s epoch=%d alpha=%.3f "
    "primary=%.6f incumbent=%.6f"
    % (
        best["name"],
        best["mode"],
        best["epoch"],
        best["alpha"],
        best["primary"],
        float(incumbent_metrics["primary"]),
    )
)

valid_scores = np.asarray(best["scores"], dtype=np.float64)
final_metrics = evaluate(valid.user_id, valid.y, valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        valid_scores,
    )

# Validation selection is now complete. Test labels are never accessed.
incumbent_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
test = load("test")

if len(incumbent_test) != len(test.user_id):
    raise ValueError("Incumbent test length mismatch")

if best["mode"] == "incumbent":
    test_scores = incumbent_test.copy()
else:
    model.load_state_dict(best["state"])
    model.eval()

    split_name_for_history[test] = "test"
    cat_test_np = categorical_matrix(test)
    raw_num_test = raw_numeric_matrix(test)
    num_test_np = (raw_num_test - num_means) / num_stds
    num_test_np = np.clip(num_test_np, -8.0, 8.0).astype(np.float32)

    cat_test = torch.from_numpy(cat_test_np)
    num_test = torch.from_numpy(num_test_np)

    candidate_test = predict_model(model, cat_test, num_test)

    if best["mode"] == "raw":
        test_scores = candidate_test
    elif best["mode"] == "zblend":
        incumbent_test_z = apply_z(
            incumbent_test, inc_mean, inc_std
        )
        candidate_test_z = apply_z(
            candidate_test,
            best["candidate_mean"],
            best["candidate_std"],
        )
        test_scores = (
            (1.0 - best["alpha"]) * incumbent_test_z
            + best["alpha"] * candidate_test_z
        )
    elif best["mode"] == "rankblend":
        incumbent_test_rank = within_user_rank(
            test.user_id, incumbent_test
        )
        candidate_test_rank = within_user_rank(
            test.user_id, candidate_test
        )
        test_scores = (
            (1.0 - best["alpha"]) * incumbent_test_rank
            + best["alpha"] * candidate_test_rank
        )
    else:
        raise ValueError("Unknown selected mode: %s" % best["mode"])

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - _start_time
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.4f}'
    % (
        float(final_metrics["primary"]),
        float(final_metrics["gauc"]),
        float(final_metrics["ndcg@5"]),
        elapsed,
    )
)