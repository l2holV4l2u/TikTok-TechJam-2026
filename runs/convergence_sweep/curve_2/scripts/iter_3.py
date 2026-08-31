import os
import time
import json
import math
import random
import copy
import gc

import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7319
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "user_active_degree",
    "onehot_feat3",
    "onehot_feat8",
    "hour",
    "register_days_bucket",
    "music_type",
    "video_type",
]
RAW_NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)


def make_categorical(split):
    return np.ascontiguousarray(
        np.stack(
            [np.asarray(split.X[name], dtype=np.int64) for name in CAT_FIELDS],
            axis=1,
        ),
        dtype=np.int64,
    )


xcat_train = make_categorical(train)
xcat_valid = make_categorical(valid)
xcat_test = make_categorical(test)


def make_raw_numeric(split):
    cols = []
    for name in RAW_NUM_FIELDS:
        a = np.asarray(split.num[name], dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(np.log1p(np.maximum(a, 0.0)))
    return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.float32)


raw_train = make_raw_numeric(train)
raw_valid = make_raw_numeric(valid)
raw_test = make_raw_numeric(test)


def get_histories(split_name):
    ans = {}
    for entity in ("video_id", "author_id"):
        d = historical_features(split_name, key=entity)
        for key, value in d.items():
            ans[entity + "__" + key] = np.asarray(value, dtype=np.float32)
    return ans


hist_train_dict = get_histories("train")
hist_valid_dict = get_histories("valid")
hist_test_dict = get_histories("test")

common_history_keys = sorted(
    set(hist_train_dict) & set(hist_valid_dict) & set(hist_test_dict)
)

# Retain stable support and relevance statistics plus a small number of
# engagement priors. This keeps attention sequence length manageable.
selected_history_keys = [
    k for k in common_history_keys
    if (
        "train_count" in k
        or "long_view_rate" in k
        or "is_like_rate" in k
        or "is_follow_rate" in k
    )
]
if not selected_history_keys:
    selected_history_keys = common_history_keys[:8]


def history_matrix(d):
    return np.ascontiguousarray(
        np.stack(
            [
                np.nan_to_num(
                    np.asarray(d[k], dtype=np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                for k in selected_history_keys
            ],
            axis=1,
        ),
        dtype=np.float32,
    )


hist_train = history_matrix(hist_train_dict)
hist_valid = history_matrix(hist_valid_dict)
hist_test = history_matrix(hist_test_dict)

xnum_train = np.ascontiguousarray(
    np.concatenate([raw_train, hist_train], axis=1), dtype=np.float32
)
xnum_valid = np.ascontiguousarray(
    np.concatenate([raw_valid, hist_valid], axis=1), dtype=np.float32
)
xnum_test = np.ascontiguousarray(
    np.concatenate([raw_test, hist_test], axis=1), dtype=np.float32
)

num_mean = xnum_train.mean(axis=0, dtype=np.float64).astype(np.float32)
num_std = xnum_train.std(axis=0, dtype=np.float64).astype(np.float32)
num_std = np.maximum(num_std, 1e-3)

xnum_train = np.ascontiguousarray(
    np.clip((xnum_train - num_mean) / num_std, -8.0, 8.0),
    dtype=np.float32,
)
xnum_valid = np.ascontiguousarray(
    np.clip((xnum_valid - num_mean) / num_std, -8.0, 8.0),
    dtype=np.float32,
)
xnum_test = np.ascontiguousarray(
    np.clip((xnum_test - num_mean) / num_std, -8.0, 8.0),
    dtype=np.float32,
)

# Emphasize observations nearest the date boundary, where the logging and
# target distributions most closely resemble validation and test.
age = np.asarray(train.date.max() - train.date, dtype=np.float32)
train_weights = np.exp(-math.log(2.0) * age / 6.0).astype(np.float32)
train_weights /= train_weights.mean()

CARDINALITIES = [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS]
OFFSETS = np.cumsum([0] + CARDINALITIES[:-1], dtype=np.int64)
TOTAL_CATEGORIES = int(sum(CARDINALITIES))
N_CAT = len(CAT_FIELDS)
N_NUM = xnum_train.shape[1]
EMBED_DIM = 8


class FieldEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("offsets", torch.from_numpy(OFFSETS.copy()))
        self.embedding = nn.Embedding(TOTAL_CATEGORIES, EMBED_DIM)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, xcat):
        return self.embedding(xcat + self.offsets)


class CrossLayer(nn.Module):
    def __init__(self, dim, rank=32):
        super().__init__()
        self.u = nn.Linear(dim, rank, bias=False)
        self.v = nn.Linear(rank, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.u.weight)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, x0, x):
        return x + x0 * (self.v(torch.tanh(self.u(x))) + self.bias)


class DCNv2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FieldEncoder()
        dim = N_CAT * EMBED_DIM + N_NUM
        self.cross_layers = nn.ModuleList(
            [CrossLayer(dim, rank=32) for _ in range(3)]
        )
        self.deep = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(dim + 64, 1)

    def forward(self, xcat, xnum):
        emb = self.encoder(xcat).flatten(1)
        x0 = torch.cat([emb, xnum], dim=1)
        cross = x0
        for layer in self.cross_layers:
            cross = layer(x0, cross)
        deep = self.deep(x0)
        return self.output(torch.cat([cross, deep], dim=1)).squeeze(1)


class AutoIntModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FieldEncoder()
        self.numeric_weight = nn.Parameter(
            torch.empty(N_NUM, EMBED_DIM)
        )
        self.numeric_bias = nn.Parameter(
            torch.zeros(N_NUM, EMBED_DIM)
        )
        nn.init.normal_(self.numeric_weight, std=0.05)

        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=2,
            dim_feedforward=32,
            dropout=0.05,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.attention = nn.TransformerEncoder(layer, num_layers=2)
        n_fields = N_CAT + N_NUM
        self.output = nn.Sequential(
            nn.Linear(n_fields * EMBED_DIM + N_NUM, 96),
            nn.ReLU(),
            nn.Dropout(0.06),
            nn.Linear(96, 1),
        )

    def forward(self, xcat, xnum):
        cat_tokens = self.encoder(xcat)
        num_tokens = (
            xnum.unsqueeze(-1) * self.numeric_weight.unsqueeze(0)
            + self.numeric_bias.unsqueeze(0)
        )
        tokens = torch.cat([cat_tokens, num_tokens], dim=1)
        attended = self.attention(tokens)
        z = torch.cat([attended.flatten(1), xnum], dim=1)
        return self.output(z).squeeze(1)


class PNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FieldEncoder()
        self.numeric_weight = nn.Parameter(
            torch.empty(N_NUM, EMBED_DIM)
        )
        self.numeric_bias = nn.Parameter(
            torch.zeros(N_NUM, EMBED_DIM)
        )
        nn.init.normal_(self.numeric_weight, std=0.05)

        n_fields = N_CAT + N_NUM
        pair_i, pair_j = np.triu_indices(n_fields, k=1)
        self.register_buffer("pair_i", torch.from_numpy(pair_i.astype(np.int64)))
        self.register_buffer("pair_j", torch.from_numpy(pair_j.astype(np.int64)))
        n_pairs = len(pair_i)

        self.network = nn.Sequential(
            nn.Linear(n_fields * EMBED_DIM + n_pairs + N_NUM, 160),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, xcat, xnum):
        cat_tokens = self.encoder(xcat)
        num_tokens = (
            xnum.unsqueeze(-1) * self.numeric_weight.unsqueeze(0)
            + self.numeric_bias.unsqueeze(0)
        )
        fields = torch.cat([cat_tokens, num_tokens], dim=1)
        products = (
            fields[:, self.pair_i, :] * fields[:, self.pair_j, :]
        ).sum(dim=2)
        z = torch.cat([fields.flatten(1), products, xnum], dim=1)
        return self.network(z).squeeze(1)


@torch.no_grad()
def predict(model, xcat, xnum, batch_size=16384):
    model.eval()
    result = np.empty(xcat.shape[0], dtype=np.float32)
    for start in range(0, xcat.shape[0], batch_size):
        end = min(start + batch_size, xcat.shape[0])
        logits = model(
            torch.from_numpy(xcat[start:end]),
            torch.from_numpy(xnum[start:end]),
        )
        result[start:end] = logits.cpu().numpy()
    return result


txcat = torch.from_numpy(xcat_train)
txnum = torch.from_numpy(xnum_train)
ty = torch.from_numpy(y_train)
tw = torch.from_numpy(train_weights)

pred_valid = {}
pred_test = {}
epoch_findings = {}


def fit_family(name, constructor, epochs=3, lr=1.1e-3, batch_size=4096):
    model = constructor()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=2e-6
    )
    generator = torch.Generator()
    generator.manual_seed(SEED + sum(ord(c) for c in name))

    best_primary = -np.inf
    best_state = None
    best_valid = None
    scores_by_epoch = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(y_train), generator=generator)

        for start in range(0, len(y_train), batch_size):
            idx = permutation[start:start + batch_size]
            logits = model(txcat[idx], txnum[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, ty[idx], reduction="none"
            )
            loss = (losses * tw[idx]).sum() / tw[idx].sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        va_scores = predict(model, xcat_valid, xnum_valid)
        metrics = evaluate(valid.user_id, valid.y, va_scores)
        primary = float(metrics["primary"])
        scores_by_epoch.append(primary)

        if primary > best_primary:
            best_primary = primary
            best_valid = va_scores.copy()
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    te_scores = predict(model, xcat_test, xnum_test)

    pred_valid[name] = best_valid
    pred_test[name] = te_scores
    epoch_findings[name] = scores_by_epoch

    del optimizer, model, best_state
    gc.collect()


fit_family("dcnv2", DCNv2Model, epochs=4, lr=1.0e-3)
fit_family("autoint", AutoIntModel, epochs=3, lr=8.0e-4, batch_size=3072)
fit_family("pnn", PNNModel, epochs=4, lr=1.0e-3)

del txcat, txnum, ty, tw
gc.collect()


def standardized(a):
    a = np.asarray(a, dtype=np.float64)
    mean = float(np.mean(a))
    std = float(np.std(a))
    if not np.isfinite(std) or std < 1e-10:
        std = 1.0
    return np.clip((a - mean) / std, -8.0, 8.0)


shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = None
inc_test = None
if shared:
    valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
    test_path = os.path.join(shared, "incumbent_test_scores.npy")
    if os.path.exists(valid_path) and os.path.exists(test_path):
        inc_valid = np.asarray(np.load(valid_path), dtype=np.float64)
        inc_test = np.asarray(np.load(test_path), dtype=np.float64)

candidates = {}
payloads = {}

for name in list(pred_valid):
    va = np.asarray(pred_valid[name], dtype=np.float64)
    te = np.asarray(pred_test[name], dtype=np.float64)
    metrics = evaluate(valid.user_id, valid.y, va)
    candidates[name] = float(metrics["primary"])
    payloads[name] = {
        "valid": va,
        "test": te,
        "raw": va,
        "metrics": metrics,
    }

# A model-agnostic consensus is also useful if the three interaction mechanisms
# make partially independent errors.
ensemble_valid = np.mean(
    np.stack([standardized(pred_valid[k]) for k in pred_valid], axis=0),
    axis=0,
)
ensemble_test = np.mean(
    np.stack([standardized(pred_test[k]) for k in pred_test], axis=0),
    axis=0,
)
ensemble_metrics = evaluate(valid.user_id, valid.y, ensemble_valid)
candidates["interaction_ensemble"] = float(ensemble_metrics["primary"])
payloads["interaction_ensemble"] = {
    "valid": ensemble_valid,
    "test": ensemble_test,
    "raw": ensemble_valid,
    "metrics": ensemble_metrics,
}

if inc_valid is not None:
    zinc_valid = standardized(inc_valid)
    zinc_test = standardized(inc_test)

    base_names = list(payloads.keys())
    for name in base_names:
        own_valid = payloads[name]["valid"]
        own_test = payloads[name]["test"]
        zown_valid = standardized(own_valid)
        zown_test = standardized(own_test)

        best_score = -np.inf
        best_alpha = None
        best_metrics = None
        best_valid_scores = None

        for alpha in np.linspace(0.10, 0.90, 17):
            blend_valid = (
                (1.0 - float(alpha)) * zinc_valid
                + float(alpha) * zown_valid
            )
            metrics = evaluate(valid.user_id, valid.y, blend_valid)
            if float(metrics["primary"]) > best_score:
                best_score = float(metrics["primary"])
                best_alpha = float(alpha)
                best_metrics = metrics
                best_valid_scores = blend_valid.copy()

        blend_name = name + "_inc_blend"
        blend_test = (
            (1.0 - best_alpha) * zinc_test
            + best_alpha * zown_test
        )
        candidates[blend_name] = best_score
        payloads[blend_name] = {
            "valid": best_valid_scores,
            "test": blend_test,
            "raw": own_valid,
            "metrics": best_metrics,
            "alpha": best_alpha,
        }

winner = max(candidates, key=candidates.get)
winner_payload = payloads[winner]
winner_metrics = winner_payload["metrics"]
valid_scores = np.asarray(winner_payload["valid"], dtype=np.float64)
test_scores = np.asarray(winner_payload["test"], dtype=np.float64)

print(
    "FINDINGS "
    + json.dumps(
        {
            "epoch_primary": epoch_findings,
            "history_features": selected_history_keys,
            "winner": winner,
            "winner_blend_alpha": winner_payload.get("alpha", None),
        },
        sort_keys=True,
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        {k: round(float(v), 8) for k, v in candidates.items()},
        sort_keys=True,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        valid_scores,
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        test_scores,
    )
    if "_inc_blend" in winner:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(winner_payload["raw"], dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(winner_metrics["primary"]),
            "gauc": float(winner_metrics["gauc"]),
            "ndcg@5": float(winner_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)