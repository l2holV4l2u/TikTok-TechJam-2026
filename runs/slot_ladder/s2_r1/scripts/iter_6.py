import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 27183
EMBED_DIM = 12
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768
EPOCHS = 3
HALF_LIFE_DAYS = 4.0

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "user_active_degree",
    "hour",
]
NUMERIC_FIELDS = [
    "duration_ms",
    "user_follow_user_num",
    "user_fans_user_num",
    "user_friend_user_num",
    "user_register_days",
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def make_offsets():
    offsets = []
    total = 0
    for name in FIELDS:
        offsets.append(total)
        total += int(FEATURE_CARDINALITIES[name])
    return np.asarray(offsets, dtype=np.int64), total


OFFSETS, TOTAL_CARDINALITY = make_offsets()
N_FIELDS = len(FIELDS)
N_NUMERIC = len(NUMERIC_FIELDS)


def encode_categorical(split):
    n = len(split.user_id)
    result = np.empty((n, N_FIELDS), dtype=np.int64)
    for j, name in enumerate(FIELDS):
        values = np.asarray(split.X[name], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[name])
        if values.size and (values.min() < 0 or values.max() >= card):
            raise ValueError("Out-of-range categorical value in " + name)
        result[:, j] = values + OFFSETS[j]
    return result


def fit_numeric_transform(train):
    centers = np.zeros(N_NUMERIC, dtype=np.float32)
    scales = np.ones(N_NUMERIC, dtype=np.float32)
    for j, name in enumerate(NUMERIC_FIELDS):
        raw = np.asarray(train.num[name], dtype=np.float64)
        transformed = np.log1p(np.maximum(raw, 0.0))
        finite = np.isfinite(transformed)
        if finite.any():
            centers[j] = np.float32(np.median(transformed[finite]))
            q25, q75 = np.percentile(transformed[finite], [25.0, 75.0])
            scale = (q75 - q25) / 1.349
            if not np.isfinite(scale) or scale < 1e-3:
                scale = np.std(transformed[finite])
            scales[j] = np.float32(max(float(scale), 1e-3))
    return centers, scales


def transform_numeric(split, centers, scales):
    n = len(split.user_id)
    result = np.empty((n, N_NUMERIC), dtype=np.float32)
    for j, name in enumerate(NUMERIC_FIELDS):
        raw = np.asarray(split.num[name], dtype=np.float64)
        values = np.log1p(np.maximum(raw, 0.0))
        values[~np.isfinite(values)] = float(centers[j])
        values = (values - float(centers[j])) / float(scales[j])
        result[:, j] = np.clip(values, -8.0, 8.0).astype(np.float32)
    return result


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int64)
    day = dates % 100
    newest = int(day.max())
    age = newest - day
    weights = np.exp2(-age.astype(np.float64) / HALF_LIFE_DAYS)
    weights /= weights.mean()
    return weights.astype(np.float32)


class BaseCTR(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, EMBED_DIM)
        self.wide = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.numeric_wide = nn.Linear(N_NUMERIC, 1, bias=False)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.wide.weight)
        nn.init.zeros_(self.numeric_wide.weight)

    def linear_term(self, x, numeric):
        return (
            self.wide(x).sum(dim=1).squeeze(1)
            + self.numeric_wide(numeric).squeeze(1)
        )


class DeepFM(BaseCTR):
    def __init__(self):
        super().__init__()
        width = N_FIELDS * EMBED_DIM + N_NUMERIC
        self.mlp = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, numeric):
        emb = self.embedding(x)
        summed = emb.sum(dim=1)
        fm = 0.5 * (
            summed.square() - emb.square().sum(dim=1)
        ).sum(dim=1)
        deep_input = torch.cat([emb.flatten(1), numeric], dim=1)
        deep = self.mlp(deep_input).squeeze(1)
        return self.linear_term(x, numeric) + fm + deep


class CrossNetwork(nn.Module):
    def __init__(self, width, depth=3):
        super().__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(width)) for _ in range(depth)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(width)) for _ in range(depth)
        ])
        for weight in self.weights:
            nn.init.normal_(weight, mean=0.0, std=0.02)

    def forward(self, x0):
        x = x0
        for weight, bias in zip(self.weights, self.biases):
            scalar = (x * weight).sum(dim=1, keepdim=True)
            x = x0 * scalar + bias + x
        return x


class DCN(BaseCTR):
    def __init__(self):
        super().__init__()
        width = N_FIELDS * EMBED_DIM + N_NUMERIC
        self.cross = CrossNetwork(width, depth=3)
        self.deep = nn.Sequential(
            nn.Linear(width, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
        )
        self.output = nn.Linear(width + 48, 1)

    def forward(self, x, numeric):
        emb = self.embedding(x).flatten(1)
        z = torch.cat([emb, numeric], dim=1)
        crossed = self.cross(z)
        deep = self.deep(z)
        return (
            self.linear_term(x, numeric)
            + self.output(torch.cat([crossed, deep], dim=1)).squeeze(1)
        )


class AutoInt(BaseCTR):
    def __init__(self):
        super().__init__()
        self.numeric_tokens = nn.ModuleList([
            nn.Linear(1, EMBED_DIM) for _ in range(N_NUMERIC)
        ])
        self.attn1 = nn.MultiheadAttention(
            EMBED_DIM, num_heads=3, batch_first=True, dropout=0.0
        )
        self.attn2 = nn.MultiheadAttention(
            EMBED_DIM, num_heads=3, batch_first=True, dropout=0.0
        )
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        token_count = N_FIELDS + N_NUMERIC
        self.output = nn.Sequential(
            nn.Linear(token_count * EMBED_DIM, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

    def forward(self, x, numeric):
        cat_tokens = self.embedding(x)
        num_tokens = torch.stack([
            layer(numeric[:, j:j + 1])
            for j, layer in enumerate(self.numeric_tokens)
        ], dim=1)
        z = torch.cat([cat_tokens, num_tokens], dim=1)
        attended, _ = self.attn1(z, z, z, need_weights=False)
        z = self.norm1(z + attended)
        attended, _ = self.attn2(z, z, z, need_weights=False)
        z = self.norm2(z + attended)
        return (
            self.linear_term(x, numeric)
            + self.output(z.flatten(1)).squeeze(1)
        )


class PNN(BaseCTR):
    def __init__(self):
        super().__init__()
        left, right = np.triu_indices(N_FIELDS, k=1)
        self.register_buffer("pair_left", torch.from_numpy(left.astype(np.int64)))
        self.register_buffer("pair_right", torch.from_numpy(right.astype(np.int64)))
        pair_count = len(left)
        width = N_FIELDS * EMBED_DIM + pair_count + N_NUMERIC
        self.network = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Linear(128, 56),
            nn.ReLU(),
            nn.Linear(56, 1),
        )

    def forward(self, x, numeric):
        emb = self.embedding(x)
        products = (
            emb[:, self.pair_left, :] * emb[:, self.pair_right, :]
        ).sum(dim=2)
        z = torch.cat([emb.flatten(1), products, numeric], dim=1)
        return self.linear_term(x, numeric) + self.network(z).squeeze(1)


class FiBiNET(BaseCTR):
    def __init__(self):
        super().__init__()
        hidden = max(4, N_FIELDS // 3)
        self.squeeze = nn.Sequential(
            nn.Linear(N_FIELDS, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_FIELDS),
            nn.Sigmoid(),
        )
        left, right = np.triu_indices(N_FIELDS, k=1)
        self.register_buffer("pair_left", torch.from_numpy(left.astype(np.int64)))
        self.register_buffer("pair_right", torch.from_numpy(right.astype(np.int64)))
        pair_count = len(left)
        width = pair_count * EMBED_DIM + N_FIELDS * EMBED_DIM + N_NUMERIC
        self.network = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x, numeric):
        emb = self.embedding(x)
        pooled = emb.mean(dim=2)
        importance = self.squeeze(pooled)
        reweighted = emb * importance.unsqueeze(2)
        bilinear = (
            reweighted[:, self.pair_left, :]
            * emb[:, self.pair_right, :]
        )
        z = torch.cat(
            [reweighted.flatten(1), bilinear.flatten(1), numeric], dim=1
        )
        return self.linear_term(x, numeric) + self.network(z).squeeze(1)


def fit_model(model, x, numeric, labels, weights, seed):
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0014, weight_decay=1e-6
    )
    n = x.shape[0]
    generator = torch.Generator()
    generator.manual_seed(seed)

    for _ in range(EPOCHS):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb = x[idx]
            nb = numeric[idx]
            yb = labels[idx]
            wb = weights[idx]

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, nb)
            row_loss = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = (row_loss * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def predict(model, x, numeric):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, x.shape[0], PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, x.shape[0])
            logits = model(
                torch.from_numpy(x[start:end]),
                torch.from_numpy(numeric[start:end]),
            )
            result[start:end] = logits.cpu().numpy().astype(np.float64)
    return result


train = load("train")
valid = load("valid")
test = load("test")

x_train_np = encode_categorical(train)
x_valid_np = encode_categorical(valid)
x_test_np = encode_categorical(test)

numeric_centers, numeric_scales = fit_numeric_transform(train)
n_train_np = transform_numeric(train, numeric_centers, numeric_scales)
n_valid_np = transform_numeric(valid, numeric_centers, numeric_scales)
n_test_np = transform_numeric(test, numeric_centers, numeric_scales)

y_train_np = np.asarray(train.y, dtype=np.float32)
weights_np = recency_weights(train.date)

x_train = torch.from_numpy(x_train_np)
n_train = torch.from_numpy(n_train_np)
y_train = torch.from_numpy(y_train_np)
train_weights = torch.from_numpy(weights_np)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if inc_valid.shape[0] != len(valid.user_id):
    raise ValueError("Incumbent validation prediction length mismatch")
if inc_test.shape[0] != len(test.user_id):
    raise ValueError("Incumbent test prediction length mismatch")

families = [
    ("deepfm", DeepFM),
    ("dcn", DCN),
    ("autoint", AutoInt),
    ("pnn", PNN),
    ("fibinet", FiBiNET),
]

alphas = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]
candidate_scores = {}
winner_primary = -np.inf
winner_valid = None
winner_test = None
winner_raw_valid = None
winner_name = None
winner_metrics = None

inc_metrics = evaluate(valid.user_id, valid.y, inc_valid)
candidate_scores["incumbent"] = float(inc_metrics["primary"])

for family_index, (name, constructor) in enumerate(families):
    torch.manual_seed(SEED + 1009 * family_index)
    model = constructor()
    model = fit_model(
        model,
        x_train,
        n_train,
        y_train,
        train_weights,
        SEED + 7919 * family_index,
    )

    raw_valid = predict(model, x_valid_np, n_valid_np)
    raw_test = predict(model, x_test_np, n_test_np)

    raw_metrics = evaluate(valid.user_id, valid.y, raw_valid)
    candidate_scores[name + "_raw"] = float(raw_metrics["primary"])

    for alpha in alphas:
        if alpha == 1.0:
            blended_valid = raw_valid
            blended_test = raw_test
        else:
            blended_valid = alpha * raw_valid + (1.0 - alpha) * inc_valid
            blended_test = alpha * raw_test + (1.0 - alpha) * inc_test

        metrics = evaluate(valid.user_id, valid.y, blended_valid)
        key = name + "_blend_" + str(alpha)
        candidate_scores[key] = float(metrics["primary"])

        if float(metrics["primary"]) > winner_primary:
            winner_primary = float(metrics["primary"])
            winner_valid = np.asarray(blended_valid, dtype=np.float64).copy()
            winner_test = np.asarray(blended_test, dtype=np.float64).copy()
            winner_raw_valid = np.asarray(raw_valid, dtype=np.float64).copy()
            winner_name = key
            winner_metrics = metrics

    del model

# The trusted incumbent itself is a valid fallback if every new family hurts.
if float(inc_metrics["primary"]) > winner_primary:
    winner_primary = float(inc_metrics["primary"])
    winner_valid = inc_valid.copy()
    winner_test = inc_test.copy()
    winner_raw_valid = inc_valid.copy()
    winner_name = "incumbent"
    winner_metrics = inc_metrics

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(winner_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(winner_test, dtype=np.float64),
    )
    if winner_name != "incumbent" and "_blend_" in winner_name:
        alpha_text = winner_name.rsplit("_", 1)[-1]
        if float(alpha_text) < 1.0:
            np.save(
                os.path.join(out, "scores_valid_raw.npy"),
                np.asarray(winner_raw_valid, dtype=np.float64),
            )

print("FINDINGS " + json.dumps({
    "winner": winner_name,
    "half_life_days": HALF_LIFE_DAYS,
    "train_weight_min": float(weights_np.min()),
    "train_weight_max": float(weights_np.max()),
    "incumbent_std": float(np.std(inc_valid)),
    "winner_std": float(np.std(winner_valid)),
}, sort_keys=True))
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(winner_metrics["primary"]),
    "gauc": float(winner_metrics["gauc"]),
    "ndcg@5": float(winner_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))