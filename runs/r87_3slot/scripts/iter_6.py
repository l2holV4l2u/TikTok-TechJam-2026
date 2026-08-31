import os
import time
import json
import gc
import random
import datetime
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7331
THREADS = max(1, min(8, os.cpu_count() or 1))
torch.set_num_threads(THREADS)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat1",
    "onehot_feat7",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "hour",
]
EMBED_DIM = 8
BATCH_SIZE = 4096
PRED_BATCH = 16384
EPOCHS = 2
HALF_LIFE_DAYS = 5.0

CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1], dtype=np.int64)
TOTAL_CARD = int(sum(CARDS))
N_FIELDS = len(FIELDS)
FLAT_DIM = N_FIELDS * EMBED_DIM


def make_matrix(parts):
    cols = []
    for field, offset in zip(FIELDS, OFFSETS):
        if len(parts) == 1:
            x = np.asarray(parts[0].X[field], dtype=np.int64)
        else:
            x = np.concatenate([
                np.asarray(part.X[field], dtype=np.int64)
                for part in parts
            ])
        cols.append(x + offset)
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


def dates_to_ordinals(date_values):
    date_values = np.asarray(date_values, dtype=np.int64)
    unique_dates, inverse = np.unique(date_values, return_inverse=True)
    ordinal_values = np.empty(len(unique_dates), dtype=np.int32)
    for i, value in enumerate(unique_dates):
        value = int(value)
        year = value // 10000
        month = (value // 100) % 100
        day = value % 100
        ordinal_values[i] = datetime.date(year, month, day).toordinal()
    return ordinal_values[inverse]


def recency_weights(date_values):
    ordinals = dates_to_ordinals(date_values)
    age = ordinals.max() - ordinals
    weights = np.power(0.5, age.astype(np.float64) / HALF_LIFE_DAYS)
    weights = weights.astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return weights


class FeatureEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x):
        return self.embedding(x)


class DCNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = FeatureEmbedding()
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.empty(FLAT_DIM)) for _ in range(3)
        ])
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(FLAT_DIM)) for _ in range(3)
        ])
        for weight in self.cross_weights:
            nn.init.normal_(weight, mean=0.0, std=0.015)

        self.deep = nn.Sequential(
            nn.Linear(FLAT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(FLAT_DIM + 64, 1)

    def forward(self, x):
        x0 = self.features(x).reshape(x.shape[0], -1)
        cross = x0
        for weight, bias in zip(self.cross_weights, self.cross_biases):
            scalar = torch.sum(cross * weight, dim=1, keepdim=True)
            cross = cross + x0 * scalar + bias
        deep = self.deep(x0)
        return self.output(torch.cat([cross, deep], dim=1)).squeeze(1)


class PNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = FeatureEmbedding()
        pair_i, pair_j = np.triu_indices(N_FIELDS, k=1)
        self.register_buffer(
            "pair_i", torch.from_numpy(pair_i.astype(np.int64))
        )
        self.register_buffer(
            "pair_j", torch.from_numpy(pair_j.astype(np.int64))
        )
        pair_count = len(pair_i)
        self.network = nn.Sequential(
            nn.Linear(FLAT_DIM + pair_count, 160),
            nn.ReLU(),
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        emb = self.features(x)
        products = torch.sum(
            emb[:, self.pair_i, :] * emb[:, self.pair_j, :],
            dim=2,
        )
        flat = emb.reshape(x.shape[0], -1)
        return self.network(torch.cat([flat, products], dim=1)).squeeze(1)


class AutoIntModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = FeatureEmbedding()
        self.attention1 = nn.MultiheadAttention(
            EMBED_DIM, num_heads=2, dropout=0.0, batch_first=True
        )
        self.attention2 = nn.MultiheadAttention(
            EMBED_DIM, num_heads=2, dropout=0.0, batch_first=True
        )
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        self.output = nn.Sequential(
            nn.Linear(FLAT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        emb = self.features(x)
        attn1, _ = self.attention1(
            emb, emb, emb, need_weights=False
        )
        z = self.norm1(emb + attn1)
        attn2, _ = self.attention2(
            z, z, z, need_weights=False
        )
        z = self.norm2(z + attn2)
        return self.output(z.reshape(x.shape[0], -1)).squeeze(1)


def create_model(family):
    if family == "dcn":
        return DCNModel()
    if family == "pnn":
        return PNNModel()
    if family == "autoint":
        return AutoIntModel()
    raise ValueError("unknown family: " + str(family))


def fit_model(family, x_np, y_np, weights_np, seed):
    torch.manual_seed(seed)
    model = create_model(family)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0015, weight_decay=2e-6
    )

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    weights = torch.from_numpy(np.asarray(weights_np, dtype=np.float32))
    n = len(x_np)

    for epoch in range(EPOCHS):
        model.train()
        generator = torch.Generator()
        generator.manual_seed(seed + 1009 * (epoch + 1))
        order = torch.randperm(n, generator=generator)

        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[idx])
            row_losses = nn.functional.binary_cross_entropy_with_logits(
                logits, y[idx], reduction="none"
            )
            batch_weights = weights[idx]
            loss = torch.sum(row_losses * batch_weights) / torch.sum(
                batch_weights
            ).clamp_min(1e-8)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def predict_model(model, x_np):
    model.eval()
    x = torch.from_numpy(x_np)
    result = np.empty(len(x_np), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(x_np), PRED_BATCH):
            end = min(start + PRED_BATCH, len(x_np))
            result[start:end] = (
                model(x[start:end]).cpu().numpy().astype(np.float64)
            )
    return result


def metric(users, labels, scores):
    return evaluate(
        users,
        labels,
        np.asarray(scores, dtype=np.float64),
    )


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

x_train = make_matrix([train])
x_valid = make_matrix([valid])
train_weights = recency_weights(train.date)

candidate_values = {}
inc_metrics = metric(valid.user_id, y_valid, inc_valid)
candidate_values["incumbent"] = float(inc_metrics["primary"])

best_name = "incumbent"
best_scores = inc_valid.copy()
best_metrics = inc_metrics
best_family = None
best_alpha = 0.0
best_scale = 1.0
best_raw = None

inc_std = max(float(np.std(inc_valid)), 1e-8)
families = ["dcn", "pnn", "autoint"]
blend_alphas = [0.10, 0.20, 0.30, 0.45, 0.60]
raw_predictions = {}

for family_index, family in enumerate(families):
    model = fit_model(
        family,
        x_train,
        y_train,
        train_weights,
        SEED + 100 * family_index,
    )
    raw = predict_model(model, x_valid)
    raw_predictions[family] = raw
    del model
    gc.collect()

    raw_metrics = metric(valid.user_id, y_valid, raw)
    candidate_values[family] = float(raw_metrics["primary"])

    if raw_metrics["primary"] > best_metrics["primary"]:
        best_name = family
        best_scores = raw.copy()
        best_metrics = raw_metrics
        best_family = family
        best_alpha = 1.0
        best_scale = 1.0
        best_raw = raw.copy()

    scale = inc_std / max(float(np.std(raw)), 1e-8)
    scaled_raw = scale * raw

    for alpha in blend_alphas:
        blended = (1.0 - alpha) * inc_valid + alpha * scaled_raw
        blended_metrics = metric(valid.user_id, y_valid, blended)
        name = "%s_blend_%.2f" % (family, alpha)
        candidate_values[name] = float(blended_metrics["primary"])

        if blended_metrics["primary"] > best_metrics["primary"]:
            best_name = name
            best_scores = blended.copy()
            best_metrics = blended_metrics
            best_family = family
            best_alpha = alpha
            best_scale = scale
            best_raw = raw.copy()

print(
    "CANDIDATES " + json.dumps(candidate_values, sort_keys=True),
    flush=True,
)
print(
    "FINDINGS selected=%s incumbent=%.6f dcn=%.6f pnn=%.6f autoint=%.6f"
    % (
        best_name,
        candidate_values["incumbent"],
        candidate_values["dcn"],
        candidate_values["pnn"],
        candidate_values["autoint"],
    ),
    flush=True,
)

test = load("test")
inc_test = np.load(inc_test_path).astype(np.float64)

if best_family is None:
    test_scores = inc_test.copy()
else:
    y_train_valid = np.concatenate([y_train, y_valid]).astype(np.int8)
    dates_train_valid = np.concatenate([
        np.asarray(train.date),
        np.asarray(valid.date),
    ])
    weights_train_valid = recency_weights(dates_train_valid)

    del x_train, x_valid
    gc.collect()

    x_train_valid = make_matrix([train, valid])
    x_test = make_matrix([test])

    family_index = families.index(best_family)
    refit_model = fit_model(
        best_family,
        x_train_valid,
        y_train_valid,
        weights_train_valid,
        SEED + 100 * family_index,
    )
    test_raw = predict_model(refit_model, x_test)

    if best_alpha >= 1.0:
        test_scores = test_raw
    else:
        test_scores = (
            (1.0 - best_alpha) * inc_test
            + best_alpha * best_scale * test_raw
        )

    del refit_model, x_train_valid, x_test
    gc.collect()

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if best_family is not None and best_alpha < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.3f}'
    % (
        best_metrics["primary"],
        best_metrics["gauc"],
        best_metrics["ndcg@5"],
        elapsed,
    )
)