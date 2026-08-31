import os
import time
import json
import gc
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 48173
THREADS = max(1, min(8, os.cpu_count() or 1))
torch.set_num_threads(THREADS)
torch.set_num_interop_threads(1)
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
    "user_active_degree",
    "hour",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

EMBED_DIM = 8
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 16384
EPOCHS = 2
HALF_LIFE_DAYS = 8.0

cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))
n_fields = len(FIELDS)

pair_i_np, pair_j_np = np.triu_indices(n_fields, k=1)
pair_i = torch.from_numpy(pair_i_np.astype(np.int64))
pair_j = torch.from_numpy(pair_j_np.astype(np.int64))
n_pairs = len(pair_i_np)


def make_cat(parts):
    columns = []
    for field, offset in zip(FIELDS, offsets):
        if len(parts) == 1:
            x = np.asarray(parts[0].X[field], dtype=np.int64)
        else:
            x = np.concatenate([
                np.asarray(part.X[field], dtype=np.int64)
                for part in parts
            ])
        columns.append(x + offset)
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.int64)


def numeric_statistics(parts):
    centers = []
    scales = []
    for field in NUM_FIELDS:
        if len(parts) == 1:
            x = np.asarray(parts[0].num[field], dtype=np.float64)
        else:
            x = np.concatenate([
                np.asarray(part.num[field], dtype=np.float64)
                for part in parts
            ])
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.log1p(np.maximum(x, 0.0))
        center = float(np.median(x))
        q10, q90 = np.percentile(x, [10.0, 90.0])
        centers.append(center)
        scales.append(max(float(q90 - q10), 0.5))
    return np.asarray(centers, np.float32), np.asarray(scales, np.float32)


def make_num(parts, center, scale):
    columns = []
    for j, field in enumerate(NUM_FIELDS):
        if len(parts) == 1:
            x = np.asarray(parts[0].num[field], dtype=np.float32)
        else:
            x = np.concatenate([
                np.asarray(part.num[field], dtype=np.float32)
                for part in parts
            ])
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.log1p(np.maximum(x, 0.0))
        x = np.clip((x - center[j]) / scale[j], -5.0, 5.0)
        columns.append(x.astype(np.float32))
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def date_ordinals(parts):
    if len(parts) == 1:
        dates = np.asarray(parts[0].date, dtype=np.int32)
    else:
        dates = np.concatenate([
            np.asarray(part.date, dtype=np.int32)
            for part in parts
        ])

    unique_dates, inverse = np.unique(dates, return_inverse=True)
    ordinal_values = np.empty(len(unique_dates), dtype=np.float32)
    for i, date_value in enumerate(unique_dates):
        dt = datetime.strptime(str(int(date_value)), "%Y%m%d")
        ordinal_values[i] = float(dt.toordinal())
    return ordinal_values[inverse]


def recency_weights(parts):
    ordinals = date_ordinals(parts)
    age = np.max(ordinals) - ordinals
    weights = np.exp(-np.log(2.0) * age / HALF_LIFE_DAYS)
    weights = weights / np.mean(weights)
    return np.asarray(weights, dtype=np.float32)


class CommonEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)
        self.wide = nn.Embedding(total_cardinality, 1)
        self.numeric_wide = nn.Linear(len(NUM_FIELDS), 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.wide.weight)
        nn.init.zeros_(self.numeric_wide.weight)

    def components(self, cat, num):
        emb = self.embedding(cat)
        wide = self.wide(cat).squeeze(-1).sum(dim=1)
        wide = wide + self.numeric_wide(num).squeeze(1) + self.bias
        return emb, wide


class FieldWeightedFM(nn.Module):
    """FM whose pair interactions have separately learned field-pair weights."""
    def __init__(self):
        super().__init__()
        self.common = CommonEmbedding()
        self.register_buffer("pair_i", pair_i.clone())
        self.register_buffer("pair_j", pair_j.clone())
        self.field_weights = nn.Parameter(torch.ones(n_pairs))
        self.numeric_interaction = nn.Sequential(
            nn.Linear(len(NUM_FIELDS), 16),
            nn.Tanh(),
            nn.Linear(16, 1, bias=False),
        )
        nn.init.zeros_(self.numeric_interaction[-1].weight)

    def forward(self, cat, num):
        emb, wide = self.common.components(cat, num)
        dots = (
            emb[:, self.pair_i, :] * emb[:, self.pair_j, :]
        ).sum(dim=2)
        interaction = (dots * self.field_weights).sum(dim=1)
        return wide + interaction + self.numeric_interaction(num).squeeze(1)


class NFM(nn.Module):
    """Neural FM using vector-valued second-order bi-interaction pooling."""
    def __init__(self):
        super().__init__()
        self.common = CommonEmbedding()
        self.network = nn.Sequential(
            nn.Linear(EMBED_DIM + len(NUM_FIELDS), 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(self, cat, num):
        emb, wide = self.common.components(cat, num)
        summed = emb.sum(dim=1)
        bi = 0.5 * (
            summed.square() - emb.square().sum(dim=1)
        )
        hidden = torch.cat([bi, num], dim=1)
        return wide + self.network(hidden).squeeze(1)


class CINLayer(nn.Module):
    def __init__(self, previous_fields, output_fields):
        super().__init__()
        self.previous_fields = previous_fields
        self.output_fields = output_fields
        self.projection = nn.Linear(
            previous_fields * n_fields,
            output_fields,
        )

    def forward(self, previous, original):
        products = torch.einsum(
            "bhd,bfd->bhfd", previous, original
        )
        products = products.flatten(1, 2).transpose(1, 2)
        projected = self.projection(products)
        return projected.transpose(1, 2)


class XDeepFM(nn.Module):
    """Compressed Interaction Network plus a nonlinear deep branch."""
    def __init__(self):
        super().__init__()
        self.common = CommonEmbedding()
        self.cin1 = CINLayer(n_fields, 18)
        self.cin2 = CINLayer(18, 18)
        self.cin_output = nn.Linear(36, 1, bias=False)
        self.deep = nn.Sequential(
            nn.Linear(n_fields * EMBED_DIM + len(NUM_FIELDS), 96),
            nn.ReLU(),
            nn.LayerNorm(96),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, cat, num):
        emb, wide = self.common.components(cat, num)
        h1 = torch.relu(self.cin1(emb, emb))
        h2 = torch.relu(self.cin2(h1, emb))
        cin_features = torch.cat([
            h1.sum(dim=2),
            h2.sum(dim=2),
        ], dim=1)
        cin_score = self.cin_output(cin_features).squeeze(1)

        deep_input = torch.cat([emb.flatten(1), num], dim=1)
        deep_score = self.deep(deep_input).squeeze(1)
        return wide + cin_score + deep_score


def make_model(family):
    if family == "fwfm":
        return FieldWeightedFM()
    if family == "nfm":
        return NFM()
    if family == "xdeepfm":
        return XDeepFM()
    raise ValueError(f"Unknown family: {family}")


def fit_model(family, cat, num, labels, weights, seed):
    torch.manual_seed(seed)
    model = make_model(family)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0018,
        weight_decay=2.0e-6,
    )

    cat_t = torch.from_numpy(cat)
    num_t = torch.from_numpy(num)
    y_t = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    weight_t = torch.from_numpy(np.asarray(weights, dtype=np.float32))

    n = len(labels)
    for epoch in range(EPOCHS):
        generator = torch.Generator()
        generator.manual_seed(seed + 7919 * epoch)
        order = torch.randperm(n, generator=generator)

        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(cat_t[idx], num_t[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, y_t[idx], reduction="none"
            )
            batch_weights = weight_t[idx]
            loss = (losses * batch_weights).sum() / batch_weights.sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def predict(model, cat, num):
    model.eval()
    cat_t = torch.from_numpy(cat)
    num_t = torch.from_numpy(num)
    result = np.empty(len(cat), dtype=np.float64)

    with torch.no_grad():
        for start in range(0, len(cat), PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, len(cat))
            result[start:end] = model(
                cat_t[start:end],
                num_t[start:end],
            ).cpu().numpy()
    return result


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    positions = np.arange(n, dtype=np.int64)
    start_positions = np.where(starts, positions, 0)
    start_positions = np.maximum.accumulate(start_positions)
    local_positions = positions - start_positions

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_positions = np.where(ends, positions + 1, n)
    end_positions = np.minimum.accumulate(end_positions[::-1])[::-1]
    sizes = end_positions - start_positions

    normalized = local_positions / np.maximum(sizes - 1, 1)
    output = np.empty(n, dtype=np.float64)
    output[order] = normalized
    return output


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

center, scale = numeric_statistics([train])
cat_train = make_cat([train])
num_train = make_num([train], center, scale)
train_weights = recency_weights([train])
cat_valid = make_cat([valid])
num_valid = make_num([valid], center, scale)

families = ["fwfm", "nfm", "xdeepfm"]
family_seeds = {
    "fwfm": SEED + 101,
    "nfm": SEED + 202,
    "xdeepfm": SEED + 303,
}

raw_predictions = {}
candidate_results = {}

inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)
candidate_results["incumbent"] = float(inc_metrics["primary"])

best_name = "incumbent"
best_metrics = inc_metrics
best_scores = inc_valid.copy()
best_spec = {
    "family": "incumbent",
    "alpha": 0.0,
    "type": "incumbent",
}
best_raw = None

inc_rank = within_user_rank(valid.user_id, inc_valid)
blend_alphas = [0.15, 0.25, 0.35, 0.50, 0.65]

for family in families:
    model = fit_model(
        family,
        cat_train,
        num_train,
        y_train,
        train_weights,
        family_seeds[family],
    )
    raw = predict(model, cat_valid, num_valid)
    raw_predictions[family] = raw
    del model
    gc.collect()

    raw_metrics = evaluate(valid.user_id, y_valid, raw)
    candidate_results[family] = float(raw_metrics["primary"])

    if raw_metrics["primary"] > best_metrics["primary"]:
        best_name = family
        best_metrics = raw_metrics
        best_scores = raw.copy()
        best_raw = raw.copy()
        best_spec = {
            "family": family,
            "alpha": 1.0,
            "type": "raw",
        }

    raw_rank = within_user_rank(valid.user_id, raw)

    for alpha in blend_alphas:
        blended = (1.0 - alpha) * inc_rank + alpha * raw_rank
        metrics = evaluate(valid.user_id, y_valid, blended)
        name = f"{family}_rankblend_{alpha:.2f}"
        candidate_results[name] = float(metrics["primary"])

        if metrics["primary"] > best_metrics["primary"]:
            best_name = name
            best_metrics = metrics
            best_scores = blended.copy()
            best_raw = raw.copy()
            best_spec = {
                "family": family,
                "alpha": float(alpha),
                "type": "rankblend",
            }

print("CANDIDATES " + json.dumps(candidate_results, sort_keys=True), flush=True)
print(
    "FINDINGS selected=%s type=%s family=%s alpha=%.2f "
    "fwfm=%.6f nfm=%.6f xdeepfm=%.6f"
    % (
        best_name,
        best_spec["type"],
        best_spec["family"],
        best_spec["alpha"],
        candidate_results["fwfm"],
        candidate_results["nfm"],
        candidate_results["xdeepfm"],
    ),
    flush=True,
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_spec["type"] == "rankblend":
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

test = load("test")
inc_test = np.load(inc_test_path).astype(np.float64)

if best_spec["type"] == "incumbent":
    test_scores = inc_test
else:
    selected_family = best_spec["family"]

    del cat_train, num_train, train_weights, cat_valid, num_valid
    del raw_predictions
    gc.collect()

    combined_y = np.concatenate([
        np.asarray(train.y, dtype=np.float32),
        np.asarray(valid.y, dtype=np.float32),
    ])
    combined_center, combined_scale = numeric_statistics([train, valid])
    combined_cat = make_cat([train, valid])
    combined_num = make_num(
        [train, valid], combined_center, combined_scale
    )
    combined_weights = recency_weights([train, valid])

    test_cat = make_cat([test])
    test_num = make_num([test], combined_center, combined_scale)

    final_model = fit_model(
        selected_family,
        combined_cat,
        combined_num,
        combined_y,
        combined_weights,
        family_seeds[selected_family],
    )
    raw_test = predict(final_model, test_cat, test_num)

    if best_spec["type"] == "raw":
        test_scores = raw_test
    else:
        alpha = best_spec["alpha"]
        inc_test_rank = within_user_rank(test.user_id, inc_test)
        raw_test_rank = within_user_rank(test.user_id, raw_test)
        test_scores = (
            (1.0 - alpha) * inc_test_rank
            + alpha * raw_test_rank
        )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    }),
    flush=True,
)