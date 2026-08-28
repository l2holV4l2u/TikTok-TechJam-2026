import os
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 7319
N_THREADS = max(1, min(16, os.cpu_count() or 1))

FM_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
]

FIBI_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
]

AUTO_CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "hour",
    "music_type",
    "user_active_degree",
    "register_days_bucket",
]

AUTO_NUM_FIELDS = [
    "long_time_play_cnt",
    "play_progress",
    "valid_play_cnt",
    "complete_play_cnt",
    "play_duration",
    "play_cnt",
    "play_user_num",
    "like_cnt",
    "show_cnt",
    "duration_ms",
]

FM_RANK = 16
FM_BATCH = 4096
FM_EPOCHS = 6
FM_LR = 0.001

FIBI_DIM = 8
FIBI_BATCH = 8192
FIBI_EPOCHS = 5
FIBI_LR = 0.001

AUTO_DIM = 8
AUTO_HEADS = 2
AUTO_BATCH = 8192
AUTO_EPOCHS = 5
AUTO_LR = 0.001
NUM_BINS = 32

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(N_THREADS)


def make_offset_matrix(split, fields):
    cards = [int(FEATURE_CARDINALITIES[name]) for name in fields]
    offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
    x = np.stack(
        [np.asarray(split.X[name], dtype=np.int64) for name in fields],
        axis=1,
    )
    x = np.ascontiguousarray(x + offsets[None, :], dtype=np.int64)
    return x, int(sum(cards))


def fit_numeric_edges(split, fields, n_bins):
    edges = {}
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    for name in fields:
        values = np.asarray(split.num[name], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            edges[name] = np.empty(0, dtype=np.float64)
            continue

        e = np.quantile(finite, quantiles)
        e = np.unique(e[np.isfinite(e)])
        edges[name] = np.asarray(e, dtype=np.float64)
    return edges


def make_autoint_matrix(split, numeric_edges):
    cat_cards = [
        int(FEATURE_CARDINALITIES[name]) for name in AUTO_CAT_FIELDS
    ]
    num_cards = [
        len(numeric_edges[name]) + 2 for name in AUTO_NUM_FIELDS
    ]
    cards = cat_cards + num_cards
    offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)

    columns = []
    for name in AUTO_CAT_FIELDS:
        columns.append(np.asarray(split.X[name], dtype=np.int64))

    for name in AUTO_NUM_FIELDS:
        values = np.asarray(split.num[name], dtype=np.float64)
        finite = np.isfinite(values)
        bins = np.zeros(len(values), dtype=np.int64)
        if finite.any():
            bins[finite] = (
                np.searchsorted(
                    numeric_edges[name],
                    values[finite],
                    side="right",
                )
                + 1
            )
        columns.append(bins)

    x = np.stack(columns, axis=1)
    x = np.ascontiguousarray(x + offsets[None, :], dtype=np.int64)
    return x, int(sum(cards))


class FactorizationMachine(nn.Module):
    def __init__(self, n_tokens, rank):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, rank + 1, sparse=True)
        self.bias = nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(0.0, 0.01)

    def forward(self, x):
        z = self.embedding(x)
        linear = z[:, :, 0].sum(dim=1)
        factors = z[:, :, 1:]
        summed = factors.sum(dim=1)
        interactions = 0.5 * (
            summed.square() - factors.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interactions


class FiBiNET(nn.Module):
    def __init__(self, n_tokens, n_fields, dim):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, dim)
        self.wide = nn.Embedding(n_tokens, 1)
        self.bias = nn.Parameter(torch.zeros(()))

        reduction = max(3, n_fields // 3)
        self.senet = nn.Sequential(
            nn.Linear(n_fields, reduction),
            nn.ReLU(),
            nn.Linear(reduction, n_fields),
            nn.Sigmoid(),
        )
        self.bilinear = nn.Linear(dim, dim, bias=False)

        left = []
        right = []
        for i in range(n_fields):
            for j in range(i + 1, n_fields):
                left.append(i)
                right.append(j)

        self.register_buffer(
            "pair_left", torch.tensor(left, dtype=torch.long)
        )
        self.register_buffer(
            "pair_right", torch.tensor(right, dtype=torch.long)
        )

        input_dim = (n_fields + len(left)) * dim
        self.tower = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(96, 40),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(40, 1),
        )

        nn.init.normal_(self.embedding.weight, 0.0, 0.03)
        nn.init.zeros_(self.wide.weight)
        nn.init.xavier_uniform_(self.bilinear.weight)

    def forward(self, x):
        e = self.embedding(x)
        field_scale = 2.0 * self.senet(e.mean(dim=2))
        se = e * field_scale.unsqueeze(2)

        transformed = self.bilinear(se)
        pairwise = (
            transformed.index_select(1, self.pair_left)
            * se.index_select(1, self.pair_right)
        )

        features = torch.cat(
            [se.flatten(start_dim=1), pairwise.flatten(start_dim=1)],
            dim=1,
        )
        return (
            self.bias
            + self.wide(x).squeeze(2).sum(dim=1)
            + self.tower(features).squeeze(1)
        )


class AutoInt(nn.Module):
    def __init__(self, n_tokens, n_fields, dim, n_heads):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, dim)
        self.wide = nn.Embedding(n_tokens, 1)
        self.bias = nn.Parameter(torch.zeros(()))

        self.attention1 = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=0.05,
            batch_first=True,
        )
        self.attention2 = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=0.05,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
        )
        self.norm3 = nn.LayerNorm(dim)

        input_dim = 2 * n_fields * dim
        self.tower = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(96, 40),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(40, 1),
        )

        nn.init.normal_(self.embedding.weight, 0.0, 0.03)
        nn.init.zeros_(self.wide.weight)

    def forward(self, x):
        base = self.embedding(x)

        attended1, _ = self.attention1(
            base, base, base, need_weights=False
        )
        h = self.norm1(base + attended1)

        attended2, _ = self.attention2(
            h, h, h, need_weights=False
        )
        h = self.norm2(h + attended2)
        h = self.norm3(h + self.ffn(h))

        features = torch.cat(
            [base.flatten(start_dim=1), h.flatten(start_dim=1)],
            dim=1,
        )
        return (
            self.bias
            + self.wide(x).squeeze(2).sum(dim=1)
            + self.tower(features).squeeze(1)
        )


def train_fm(model, x, y):
    sparse_optimizer = torch.optim.SparseAdam(
        [model.embedding.weight], lr=FM_LR
    )
    bias_optimizer = torch.optim.Adam([model.bias], lr=FM_LR)
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED + 1)

    for epoch in range(FM_EPOCHS):
        permutation = rng.permutation(len(y))
        model.train()
        total_loss = 0.0

        for start in range(0, len(y), FM_BATCH):
            idx = permutation[start:start + FM_BATCH]
            xb = torch.from_numpy(x[idx])
            yb = torch.from_numpy(y[idx])

            sparse_optimizer.zero_grad(set_to_none=True)
            bias_optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            sparse_optimizer.step()
            bias_optimizer.step()
            total_loss += float(loss.detach()) * len(idx)

        print(
            "train model=FM epoch=%d loss=%.6f"
            % (epoch + 1, total_loss / len(y)),
            flush=True,
        )


def train_dense_model(
    model,
    x,
    y,
    epochs,
    batch_size,
    learning_rate,
    seed,
    name,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-6,
    )
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        permutation = rng.permutation(len(y))
        model.train()
        total_loss = 0.0

        for start in range(0, len(y), batch_size):
            idx = permutation[start:start + batch_size]
            xb = torch.from_numpy(x[idx])
            yb = torch.from_numpy(y[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(idx)

        print(
            "train model=%s epoch=%d loss=%.6f"
            % (name, epoch + 1, total_loss / len(y)),
            flush=True,
        )


def predict_model(model, x, batch_size=32768):
    model.eval()
    predictions = np.empty(len(x), dtype=np.float64)

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            predictions[start:end] = (
                model(torch.from_numpy(x[start:end]))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
    return predictions


def standardize_for_blend(train_reference, values):
    center = float(np.mean(train_reference))
    scale = float(np.std(train_reference))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return (values - center) / scale


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

x_fm_train, n_fm_tokens = make_offset_matrix(train, FM_FIELDS)
x_fm_valid, _ = make_offset_matrix(valid, FM_FIELDS)

x_fibi_train, n_fibi_tokens = make_offset_matrix(train, FIBI_FIELDS)
x_fibi_valid, _ = make_offset_matrix(valid, FIBI_FIELDS)

numeric_edges = fit_numeric_edges(train, AUTO_NUM_FIELDS, NUM_BINS)
x_auto_train, n_auto_tokens = make_autoint_matrix(
    train, numeric_edges
)
x_auto_valid, _ = make_autoint_matrix(valid, numeric_edges)

fm = FactorizationMachine(n_fm_tokens, FM_RANK)
train_fm(fm, x_fm_train, y_train)

fibi = FiBiNET(
    n_tokens=n_fibi_tokens,
    n_fields=len(FIBI_FIELDS),
    dim=FIBI_DIM,
)
train_dense_model(
    fibi,
    x_fibi_train,
    y_train,
    epochs=FIBI_EPOCHS,
    batch_size=FIBI_BATCH,
    learning_rate=FIBI_LR,
    seed=SEED + 17,
    name="FiBiNET",
)

autoint = AutoInt(
    n_tokens=n_auto_tokens,
    n_fields=len(AUTO_CAT_FIELDS) + len(AUTO_NUM_FIELDS),
    dim=AUTO_DIM,
    n_heads=AUTO_HEADS,
)
train_dense_model(
    autoint,
    x_auto_train,
    y_train,
    epochs=AUTO_EPOCHS,
    batch_size=AUTO_BATCH,
    learning_rate=AUTO_LR,
    seed=SEED + 29,
    name="AutoInt",
)

fm_valid_raw = predict_model(fm, x_fm_valid)
fibi_valid_raw = predict_model(fibi, x_fibi_valid)
auto_valid_raw = predict_model(autoint, x_auto_valid)

fm_valid = standardize_for_blend(fm_valid_raw, fm_valid_raw)
fibi_valid = standardize_for_blend(fibi_valid_raw, fibi_valid_raw)
auto_valid = standardize_for_blend(auto_valid_raw, auto_valid_raw)

valid_components = [fm_valid, fibi_valid, auto_valid]
component_names = ["fm", "fibi", "autoint"]

candidate_summary = {}
best_primary = -np.inf
best_metrics = None
best_weights = None
best_scores = None

# Coarse simplex search limits validation-selection variance while allowing
# AutoInt to contribute only when it supplies complementary ranking signal.
for fm_units in range(0, 11):
    for fibi_units in range(0, 11 - fm_units):
        auto_units = 10 - fm_units - fibi_units
        weights = np.asarray(
            [fm_units, fibi_units, auto_units],
            dtype=np.float64,
        ) / 10.0

        scores = (
            weights[0] * valid_components[0]
            + weights[1] * valid_components[1]
            + weights[2] * valid_components[2]
        )
        metrics = evaluate(valid.user_id, y_valid, scores)
        primary = float(metrics["primary"])

        if primary > best_primary:
            best_primary = primary
            best_metrics = metrics
            best_weights = weights.copy()
            best_scores = scores.copy()

for i, name in enumerate(component_names):
    metrics = evaluate(
        valid.user_id,
        y_valid,
        valid_components[i],
    )
    candidate_summary[name] = float(metrics["primary"])

base_scores = 0.5 * fm_valid + 0.5 * fibi_valid
base_metrics = evaluate(valid.user_id, y_valid, base_scores)
candidate_summary["fm_fibi_equal"] = float(base_metrics["primary"])
candidate_summary["selected"] = float(best_primary)

print(
    "FINDINGS selected_weights fm=%.2f fibi=%.2f autoint=%.2f"
    % tuple(best_weights),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_summary, sort_keys=True),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    test = load("test")

    x_fm_test, _ = make_offset_matrix(test, FM_FIELDS)
    x_fibi_test, _ = make_offset_matrix(test, FIBI_FIELDS)
    x_auto_test, _ = make_autoint_matrix(test, numeric_edges)

    fm_test_raw = predict_model(fm, x_fm_test)
    fibi_test_raw = predict_model(fibi, x_fibi_test)
    auto_test_raw = predict_model(autoint, x_auto_test)

    fm_center = float(np.mean(fm_valid_raw))
    fm_scale = max(float(np.std(fm_valid_raw)), 1e-8)
    fibi_center = float(np.mean(fibi_valid_raw))
    fibi_scale = max(float(np.std(fibi_valid_raw)), 1e-8)
    auto_center = float(np.mean(auto_valid_raw))
    auto_scale = max(float(np.std(auto_valid_raw)), 1e-8)

    fm_test = (fm_test_raw - fm_center) / fm_scale
    fibi_test = (fibi_test_raw - fibi_center) / fibi_scale
    auto_test = (auto_test_raw - auto_center) / auto_scale

    test_scores = (
        best_weights[0] * fm_test
        + best_weights[1] * fibi_test
        + best_weights[2] * auto_test
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": 0.0,
        }
    ),
    flush=True,
)