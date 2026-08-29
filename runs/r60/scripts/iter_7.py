import os
import time
import json
import copy
import gc
import random
import numpy as np
import torch
import torch.nn as nn

_start_time = time.time()

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


SEED = 20260828
DEVICE = torch.device("cpu")
BATCH_SIZE = 8192
EPOCHS = 4
EMBED_DIM = 6
CROSS_RANK = 12
NUM_CROSS_LAYERS = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(12, max(1, os.cpu_count() or 1)))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat1",
    "onehot_feat7",
    "music_type",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "hour",
    "is_video_author",
    "video_type",
]

RAW_NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    positions = np.arange(n, dtype=np.float64) - np.repeat(starts, lengths)
    denominators = np.maximum(np.repeat(lengths, lengths) - 1, 1)

    ranked_sorted = positions / denominators
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def z_parameters(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    std = max(float(np.std(values)), 1e-8)
    return mean, std


def apply_z(values, mean, std):
    return (np.asarray(values, dtype=np.float64) - mean) / std


def build_categorical(split):
    return np.column_stack(
        [np.asarray(split.X[name], dtype=np.int64) for name in CAT_FIELDS]
    ).astype(np.int64, copy=False)


def raw_numeric_columns(split):
    columns = []
    for name in RAW_NUM_FIELDS:
        values = np.asarray(split.num[name], dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = np.sign(values) * np.log1p(np.abs(values))
        columns.append(values.astype(np.float32))
        columns.append((~np.isfinite(np.asarray(split.num[name]))).astype(np.float32))

    # Coarse time/context quantities that are available at scoring time.
    columns.append(np.asarray(split.X["hour"], dtype=np.float32) / 23.0)
    columns.append(np.asarray(split.X["duration_bucket"], dtype=np.float32) / 10.0)
    columns.append(np.asarray(split.X["tab"], dtype=np.float32) / 14.0)
    return columns


def history_columns(split_name, expected_names=None):
    """
    The previous failed attempt used Split objects as dictionary keys. This
    version passes the literal split-name string required by the history API.
    """
    video_hist = historical_features(split_name, key="video_id")
    author_hist = historical_features(split_name, key="author_id")

    available = {}
    for name, values in video_hist.items():
        available["video__" + name] = np.asarray(values, dtype=np.float32)
    for name, values in author_hist.items():
        available["author__" + name] = np.asarray(values, dtype=np.float32)

    if expected_names is None:
        names = sorted(available.keys())
    else:
        names = list(expected_names)

    columns = []
    for name in names:
        if name not in available:
            raise KeyError("Missing historical feature %s for %s" % (name, split_name))
        values = np.asarray(available[name], dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(values)

    return columns, names


def build_numeric(split, split_name, history_names=None):
    raw_columns = raw_numeric_columns(split)
    hist_columns, names = history_columns(split_name, history_names)
    matrix = np.column_stack(raw_columns + hist_columns).astype(np.float32)
    return matrix, names


def fit_numeric_scaler(matrix):
    matrix64 = np.asarray(matrix, dtype=np.float64)
    mean = np.mean(matrix64, axis=0)
    std = np.std(matrix64, axis=0)
    std = np.where(std < 1e-5, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_numeric_scaler(matrix, mean, std):
    result = (matrix - mean[None, :]) / std[None, :]
    result = np.clip(result, -8.0, 8.0)
    return result.astype(np.float32, copy=False)


class LowRankCrossLayer(nn.Module):
    def __init__(self, input_dim, rank):
        super().__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, input_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)

    def forward(self, x0, xl):
        crossed = self.up(self.down(xl)) + self.bias
        return xl + x0 * crossed


class DCNV2Ranker(nn.Module):
    def __init__(self, cardinalities, num_numeric):
        super().__init__()
        self.cardinalities = list(cardinalities)
        self.offsets = []
        running = 0
        for cardinality in self.cardinalities:
            self.offsets.append(running)
            running += int(cardinality)

        self.register_buffer(
            "field_offsets",
            torch.tensor(self.offsets, dtype=torch.long),
        )

        self.embedding = nn.Embedding(running, EMBED_DIM)
        self.linear_embedding = nn.Embedding(running, 1)

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.linear_embedding.weight)

        input_dim = len(self.cardinalities) * EMBED_DIM + num_numeric
        self.cross_layers = nn.ModuleList(
            [
                LowRankCrossLayer(input_dim, CROSS_RANK)
                for _ in range(NUM_CROSS_LAYERS)
            ]
        )

        self.deep = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.LayerNorm(96),
            nn.Dropout(0.08),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.05),
        )

        self.output = nn.Linear(input_dim + 48, 1)
        self.fm_scale = nn.Parameter(torch.tensor(0.15))
        self.global_bias = nn.Parameter(torch.zeros(1))

        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, categorical, numeric):
        indexed = categorical + self.field_offsets[None, :]
        embeddings = self.embedding(indexed)

        flattened = embeddings.reshape(embeddings.shape[0], -1)
        x0 = torch.cat([flattened, numeric], dim=1)

        crossed = x0
        for layer in self.cross_layers:
            crossed = layer(x0, crossed)

        deep = self.deep(x0)
        network_logit = self.output(torch.cat([crossed, deep], dim=1)).squeeze(1)

        linear_logit = self.linear_embedding(indexed).sum(dim=1).squeeze(1)
        summed = embeddings.sum(dim=1)
        fm_vector = 0.5 * (
            summed.square() - embeddings.square().sum(dim=1)
        )
        fm_logit = fm_vector.sum(dim=1) / np.sqrt(float(EMBED_DIM))

        return (
            network_logit
            + linear_logit
            + self.fm_scale * fm_logit
            + self.global_bias
        )


def predict_model(model, categorical, numeric):
    model.eval()
    n = len(categorical)
    predictions = np.empty(n, dtype=np.float64)

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE * 2):
            end = min(start + BATCH_SIZE * 2, n)
            cat_batch = torch.from_numpy(categorical[start:end]).to(DEVICE)
            num_batch = torch.from_numpy(numeric[start:end]).to(DEVICE)
            logits = model(cat_batch, num_batch)
            predictions[start:end] = logits.cpu().numpy().astype(np.float64)

    return predictions


def clone_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


artifacts = os.environ.get("RUN_ARTIFACTS", "")
incumbent_valid_path = os.path.join(artifacts, "incumbent_valid_scores.npy")
incumbent_test_path = os.path.join(artifacts, "incumbent_test_scores.npy")

if not (
    os.path.isfile(incumbent_valid_path)
    and os.path.isfile(incumbent_test_path)
):
    raise FileNotFoundError("Trusted incumbent validation/test scores are required")

train = load("train")
valid = load("valid")

incumbent_valid = np.asarray(
    np.load(incumbent_valid_path), dtype=np.float64
)
if len(incumbent_valid) != len(valid.y):
    raise ValueError("Incumbent validation prediction length mismatch")

incumbent_metrics = evaluate(valid.user_id, valid.y, incumbent_valid)
inc_mean, inc_std = z_parameters(incumbent_valid)
inc_valid_z = apply_z(incumbent_valid, inc_mean, inc_std)
inc_valid_rank = within_user_rank(valid.user_id, incumbent_valid)

Xcat_train = build_categorical(train)
Xcat_valid = build_categorical(valid)

Xnum_train_raw, history_names = build_numeric(train, "train")
Xnum_valid_raw, _ = build_numeric(valid, "valid", history_names)

num_mean, num_std = fit_numeric_scaler(Xnum_train_raw)
Xnum_train = apply_numeric_scaler(Xnum_train_raw, num_mean, num_std)
Xnum_valid = apply_numeric_scaler(Xnum_valid_raw, num_mean, num_std)

del Xnum_train_raw, Xnum_valid_raw
gc.collect()

cardinalities = [int(FEATURE_CARDINALITIES[name]) for name in CAT_FIELDS]
model = DCNV2Ranker(cardinalities, Xnum_train.shape[1]).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1.2e-3,
    weight_decay=2.0e-6,
)
criterion = nn.BCEWithLogitsLoss(reduction="none")

train_labels = np.asarray(train.y, dtype=np.float32)
train_dates = np.asarray(train.date, dtype=np.int64)
days_old = np.maximum(int(train_dates.max()) - train_dates, 0)
row_weights = np.exp(-0.018 * days_old).astype(np.float32)
row_weights /= max(float(np.mean(row_weights)), 1e-8)

n_train = len(train_labels)
candidate_scores = {
    "incumbent": float(incumbent_metrics["primary"])
}

best = {
    "primary": float(incumbent_metrics["primary"]),
    "name": "incumbent",
    "epoch": 0,
    "mode": "incumbent",
    "alpha": 0.0,
    "metrics": incumbent_metrics,
    "scores": incumbent_valid.copy(),
    "source_mean": 0.0,
    "source_std": 1.0,
    "state": None,
}

epoch_findings = []

for epoch in range(1, EPOCHS + 1):
    model.train()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + epoch * 101)
    permutation = torch.randperm(n_train, generator=generator).numpy()

    loss_sum = 0.0
    weight_sum = 0.0

    for start in range(0, n_train, BATCH_SIZE):
        batch_indices = permutation[start:start + BATCH_SIZE]

        cat_batch = torch.from_numpy(Xcat_train[batch_indices]).to(DEVICE)
        num_batch = torch.from_numpy(Xnum_train[batch_indices]).to(DEVICE)
        y_batch = torch.from_numpy(train_labels[batch_indices]).to(DEVICE)
        w_batch = torch.from_numpy(row_weights[batch_indices]).to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(cat_batch, num_batch)
        element_loss = criterion(logits, y_batch)
        loss = torch.sum(element_loss * w_batch) / torch.sum(w_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        loss_sum += float(torch.sum(element_loss * w_batch).detach())
        weight_sum += float(torch.sum(w_batch))

    valid_raw = predict_model(model, Xcat_valid, Xnum_valid)
    raw_metrics = evaluate(valid.user_id, valid.y, valid_raw)
    raw_name = "dcn_epoch_%d_raw" % epoch
    candidate_scores[raw_name] = float(raw_metrics["primary"])

    source_mean, source_std = z_parameters(valid_raw)
    source_z = apply_z(valid_raw, source_mean, source_std)
    source_rank = within_user_rank(valid.user_id, valid_raw)

    epoch_best_primary = float(raw_metrics["primary"])
    epoch_best_name = raw_name

    local_candidates = [
        (
            raw_name,
            "raw",
            1.0,
            valid_raw,
            raw_metrics,
        )
    ]

    for alpha in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        z_scores = (1.0 - alpha) * inc_valid_z + alpha * source_z
        z_metrics = evaluate(valid.user_id, valid.y, z_scores)
        z_name = "dcn_epoch_%d_z_%.2f" % (epoch, alpha)
        candidate_scores[z_name] = float(z_metrics["primary"])
        local_candidates.append(
            (z_name, "zblend", alpha, z_scores, z_metrics)
        )

        rank_scores = (
            (1.0 - alpha) * inc_valid_rank + alpha * source_rank
        )
        rank_metrics = evaluate(valid.user_id, valid.y, rank_scores)
        rank_name = "dcn_epoch_%d_rank_%.2f" % (epoch, alpha)
        candidate_scores[rank_name] = float(rank_metrics["primary"])
        local_candidates.append(
            (rank_name, "rankblend", alpha, rank_scores, rank_metrics)
        )

    improved_this_epoch = False
    for name, mode, alpha, scores, metrics in local_candidates:
        primary = float(metrics["primary"])
        if primary > epoch_best_primary:
            epoch_best_primary = primary
            epoch_best_name = name

        if primary > best["primary"]:
            best = {
                "primary": primary,
                "name": name,
                "epoch": epoch,
                "mode": mode,
                "alpha": float(alpha),
                "metrics": metrics,
                "scores": np.asarray(scores, dtype=np.float64).copy(),
                "source_mean": source_mean,
                "source_std": source_std,
                "state": clone_state_dict(model),
            }
            improved_this_epoch = True

    epoch_findings.append(
        {
            "epoch": epoch,
            "loss": loss_sum / max(weight_sum, 1e-8),
            "raw_primary": float(raw_metrics["primary"]),
            "best_primary": epoch_best_primary,
            "best_name": epoch_best_name,
            "improved_global": improved_this_epoch,
        }
    )

    print(
        "FINDINGS epoch=%d train_loss=%.6f raw_primary=%.6f "
        "epoch_best=%.6f epoch_selected=%s"
        % (
            epoch,
            loss_sum / max(weight_sum, 1e-8),
            float(raw_metrics["primary"]),
            epoch_best_primary,
            epoch_best_name,
        )
    )

top_candidates = sorted(
    candidate_scores.items(), key=lambda item: item[1], reverse=True
)[:20]
print(
    "CANDIDATES "
    + json.dumps(
        {name: score for name, score in top_candidates},
        separators=(", ", ": "),
    )
)

print(
    "FINDINGS dcn_fields=%d numeric_features=%d history_features=%d "
    "selected=%s selected_epoch=%d mode=%s alpha=%.2f "
    "incumbent=%.6f selected_primary=%.6f"
    % (
        len(CAT_FIELDS),
        Xnum_train.shape[1],
        len(history_names),
        best["name"],
        int(best["epoch"]),
        best["mode"],
        float(best["alpha"]),
        float(incumbent_metrics["primary"]),
        float(best["primary"]),
    )
)

valid_scores = np.asarray(best["scores"], dtype=np.float64)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        valid_scores,
    )

# Validation selection is now complete. Test labels are never read or used.
test = load("test")
incumbent_test = np.asarray(
    np.load(incumbent_test_path), dtype=np.float64
)
if len(incumbent_test) != len(test.user_id):
    raise ValueError("Incumbent test prediction length mismatch")

if best["mode"] == "incumbent":
    test_scores = incumbent_test.copy()
else:
    if best["state"] is None:
        raise RuntimeError("Selected DCN candidate has no saved state")

    model.load_state_dict(best["state"])
    model.eval()

    Xcat_test = build_categorical(test)
    Xnum_test_raw, _ = build_numeric(test, "test", history_names)
    Xnum_test = apply_numeric_scaler(Xnum_test_raw, num_mean, num_std)
    del Xnum_test_raw

    test_raw = predict_model(model, Xcat_test, Xnum_test)

    if best["mode"] == "raw":
        test_scores = test_raw
    elif best["mode"] == "zblend":
        incumbent_test_z = apply_z(incumbent_test, inc_mean, inc_std)
        source_test_z = apply_z(
            test_raw, best["source_mean"], best["source_std"]
        )
        alpha = float(best["alpha"])
        test_scores = (
            (1.0 - alpha) * incumbent_test_z + alpha * source_test_z
        )
    elif best["mode"] == "rankblend":
        incumbent_test_rank = within_user_rank(test.user_id, incumbent_test)
        source_test_rank = within_user_rank(test.user_id, test_raw)
        alpha = float(best["alpha"])
        test_scores = (
            (1.0 - alpha) * incumbent_test_rank + alpha * source_test_rank
        )
    else:
        raise ValueError("Unknown selected mode %s" % best["mode"])

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

final_metrics = evaluate(valid.user_id, valid.y, valid_scores)
elapsed = time.time() - _start_time

print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.4f}'
    % (
        float(final_metrics["primary"]),
        float(final_metrics["gauc"]),
        float(final_metrics["ndcg@5"]),
        float(elapsed),
    )
)