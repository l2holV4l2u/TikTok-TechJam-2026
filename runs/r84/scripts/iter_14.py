import os
import time
import json
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 41873
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(min(12, max(1, os.cpu_count() or 1)))
DEVICE = torch.device("cpu")

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
    "user_active_degree",
    "onehot_feat3",
    "onehot_feat8",
]
NUM_NAMES = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

CARDS = np.asarray(
    [int(FEATURE_CARDINALITIES[name]) for name in FIELDS],
    dtype=np.int64
)
OFFSETS = np.concatenate([
    np.zeros(1, dtype=np.int64),
    np.cumsum(CARDS[:-1], dtype=np.int64)
])
TOTAL_CARD = int(CARDS.sum())

EMBED_DIM = 8
BATCH_SIZE = 8192
EPOCHS = 3
HALF_LIFE_DAYS = 7.0

train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)


def make_categorical(split):
    x = np.column_stack([
        np.asarray(split.X[name], dtype=np.int64) + OFFSETS[j]
        for j, name in enumerate(FIELDS)
    ])
    return np.ascontiguousarray(x, dtype=np.int64)


def fit_numeric_stats(splits):
    joined = []
    for name in NUM_NAMES:
        values = np.concatenate([
            np.asarray(s.num[name], dtype=np.float64) for s in splits
        ])
        finite = np.isfinite(values)
        median = float(np.median(values[finite])) if finite.any() else 0.0
        values = np.where(finite, values, median)
        values = np.log1p(np.maximum(values, 0.0))
        mean = float(values.mean())
        std = max(float(values.std()), 1e-6)
        joined.append((median, mean, std))
    return joined


def make_numeric(split, stats):
    cols = []
    for name, (median, mean, std) in zip(NUM_NAMES, stats):
        values = np.asarray(split.num[name], dtype=np.float64)
        values = np.where(np.isfinite(values), values, median)
        values = np.log1p(np.maximum(values, 0.0))
        values = np.clip((values - mean) / std, -6.0, 6.0)
        cols.append(values.astype(np.float32))
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)


def concatenate_inputs(splits, labels, stats):
    cats = np.concatenate([make_categorical(s) for s in splits], axis=0)
    nums = np.concatenate([make_numeric(s, stats) for s in splits], axis=0)
    ys = np.concatenate([
        np.asarray(y, dtype=np.float32) for y in labels
    ])
    dates = np.concatenate([
        np.asarray(s.date, dtype=np.int32) for s in splits
    ])
    return cats, nums, ys, dates


def recency_weights(dates):
    unique_dates = np.unique(dates)
    date_rank = {
        int(day): i for i, day in enumerate(unique_dates.tolist())
    }
    ordinal = np.fromiter(
        (date_rank[int(day)] for day in dates),
        dtype=np.int16,
        count=len(dates)
    )
    age = int(ordinal.max()) - ordinal.astype(np.int32)
    weights = np.exp2(-age.astype(np.float32) / HALF_LIFE_DAYS)
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


class AutoIntModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM)
        self.linear_embedding = nn.Embedding(TOTAL_CARD, 1)

        self.q = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.k = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.v = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.attn_norm = nn.LayerNorm(EMBED_DIM)

        dense_in = len(FIELDS) * EMBED_DIM + len(NUM_NAMES)
        self.deep = nn.Sequential(
            nn.Linear(dense_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.numeric_linear = nn.Linear(len(NUM_NAMES), 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear_embedding.weight)

    def forward(self, cats, nums):
        emb = self.embedding(cats)
        q = self.q(emb)
        k = self.k(emb)
        v = self.v(emb)

        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / np.sqrt(float(EMBED_DIM))
        attention = torch.softmax(attention, dim=-1)
        interacted = torch.matmul(attention, v)
        interacted = self.attn_norm(interacted + emb)

        first_order = self.linear_embedding(cats).sum(dim=1)
        dense_input = torch.cat([
            interacted.flatten(start_dim=1), nums
        ], dim=1)
        return (
            first_order +
            self.deep(dense_input) +
            self.numeric_linear(nums) +
            self.bias
        ).squeeze(1)


class FiBiNETModel(nn.Module):
    def __init__(self):
        super().__init__()
        n_fields = len(FIELDS)
        n_pairs = n_fields * (n_fields - 1) // 2

        self.embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM)
        self.linear_embedding = nn.Embedding(TOTAL_CARD, 1)

        bottleneck = max(4, n_fields // 3)
        self.senet = nn.Sequential(
            nn.Linear(n_fields, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, n_fields),
            nn.Sigmoid()
        )

        pair_i, pair_j = np.triu_indices(n_fields, k=1)
        self.register_buffer(
            "pair_i", torch.as_tensor(pair_i, dtype=torch.long)
        )
        self.register_buffer(
            "pair_j", torch.as_tensor(pair_j, dtype=torch.long)
        )

        dense_in = n_pairs * EMBED_DIM + len(NUM_NAMES)
        self.deep = nn.Sequential(
            nn.Linear(dense_in, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.numeric_linear = nn.Linear(len(NUM_NAMES), 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear_embedding.weight)

    def forward(self, cats, nums):
        emb = self.embedding(cats)

        squeeze = emb.mean(dim=2)
        gates = 0.5 + self.senet(squeeze)
        gated = emb * gates.unsqueeze(2)

        left = gated[:, self.pair_i, :]
        right = gated[:, self.pair_j, :]
        products = (left * right).flatten(start_dim=1)

        first_order = self.linear_embedding(cats).sum(dim=1)
        dense_input = torch.cat([products, nums], dim=1)
        return (
            first_order +
            self.deep(dense_input) +
            self.numeric_linear(nums) +
            self.bias
        ).squeeze(1)


def build_model(name):
    torch.manual_seed(SEED)
    if name == "autoint":
        return AutoIntModel()
    if name == "fibinet":
        return FiBiNETModel()
    raise ValueError(name)


def fit_model(model_name, splits, labels):
    stats = fit_numeric_stats(splits)
    cats, nums, ys, dates = concatenate_inputs(
        splits, labels, stats
    )
    weights = recency_weights(dates)

    cat_tensor = torch.from_numpy(cats)
    num_tensor = torch.from_numpy(nums)
    y_tensor = torch.from_numpy(ys)
    weight_tensor = torch.from_numpy(weights)

    model = build_model(model_name).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.003, weight_decay=2e-6
    )

    n = len(ys)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    model.train()
    for epoch in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            batch_cat = cat_tensor[idx].to(DEVICE)
            batch_num = num_tensor[idx].to(DEVICE)
            batch_y = y_tensor[idx].to(DEVICE)
            batch_weight = weight_tensor[idx].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_cat, batch_num)
            losses = F.binary_cross_entropy_with_logits(
                logits, batch_y, reduction="none"
            )
            loss = (losses * batch_weight).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    del cats, nums, ys, dates, weights
    del cat_tensor, num_tensor, y_tensor, weight_tensor
    gc.collect()
    return model, stats


@torch.no_grad()
def predict(model, stats, split):
    cats = make_categorical(split)
    nums = make_numeric(split, stats)
    cat_tensor = torch.from_numpy(cats)
    num_tensor = torch.from_numpy(nums)

    output = np.empty(len(cats), dtype=np.float64)
    model.eval()
    for start in range(0, len(cats), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(cats))
        logits = model(
            cat_tensor[start:end].to(DEVICE),
            num_tensor[start:end].to(DEVICE)
        )
        output[start:end] = logits.cpu().numpy().astype(np.float64)

    del cats, nums, cat_tensor, num_tensor
    gc.collect()
    return output


def standardized_blend(own, incumbent, own_weight):
    own = np.asarray(own, dtype=np.float64)
    incumbent = np.asarray(incumbent, dtype=np.float64)
    own_z = (own - own.mean()) / max(float(own.std()), 1e-8)
    inc_z = (
        incumbent - incumbent.mean()
    ) / max(float(incumbent.std()), 1e-8)
    return own_weight * own_z + (1.0 - own_weight) * inc_z


shared = os.environ["SHARED_ARTIFACTS"]
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)

family_predictions = {}
family_correlations = {}
candidate_results = {}

for family in ["autoint", "fibinet"]:
    model, stats = fit_model(family, [train], [y_train])
    raw_valid = predict(model, stats, valid)
    family_predictions[family] = raw_valid

    raw_metrics = evaluate(valid.user_id, y_valid, raw_valid)
    candidate_results[f"{family}_raw"] = float(
        raw_metrics["primary"]
    )
    family_correlations[f"{family}_inc_corr"] = float(
        np.corrcoef(raw_valid, inc_valid)[0, 1]
    )

    del model, stats
    gc.collect()

family_correlations["autoint_fibinet_corr"] = float(
    np.corrcoef(
        family_predictions["autoint"],
        family_predictions["fibinet"]
    )[0, 1]
)

blend_weights = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0]
best_primary = -np.inf
best_family = None
best_weight = None
best_scores = None
best_raw = None
best_metrics = None

for family, raw_valid in family_predictions.items():
    for weight in blend_weights:
        scores = standardized_blend(
            raw_valid, inc_valid, weight
        )
        metrics = evaluate(valid.user_id, y_valid, scores)
        name = f"{family}_blend_{weight:.2f}"
        primary = float(metrics["primary"])
        candidate_results[name] = primary

        if primary > best_primary:
            best_primary = primary
            best_family = family
            best_weight = float(weight)
            best_scores = scores.copy()
            best_raw = raw_valid.copy()
            best_metrics = metrics

print("CANDIDATES " + json.dumps(
    candidate_results, sort_keys=True
))
print("FINDINGS " + json.dumps({
    "winner_family": best_family,
    "winner_own_weight": best_weight,
    **family_correlations
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64)
    )
    if best_weight < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64)
        )

# Refit the selected architecture with the identical weighting/training
# recipe on all labels available before the hidden test window.
te = load("test")
final_model, final_stats = fit_model(
    best_family,
    [train, valid],
    [y_train, y_valid.astype(np.float32)]
)
raw_test = predict(final_model, final_stats, te)

if best_weight < 1.0:
    inc_test = np.load(inc_test_path).astype(np.float64)
    test_scores = standardized_blend(
        raw_test, inc_test, best_weight
    )
else:
    test_scores = raw_test

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed)
}))