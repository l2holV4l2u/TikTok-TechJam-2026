import os
import time
import json
import gc
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 93217
THREADS = max(1, min(8, os.cpu_count() or 1))
torch.set_num_threads(THREADS)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "duration_bucket", "upload_type", "music_type",
    "onehot_feat3", "onehot_feat8", "onehot_feat1",
    "onehot_feat7", "user_active_degree",
    "register_days_bucket", "fans_user_num_range", "hour",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

EMBED_DIM = 8
BATCH_SIZE = 8192
EPOCHS = 2

cards = [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))
n_fields = len(CAT_FIELDS)

pair_i, pair_j = np.triu_indices(n_fields, k=1)
pair_i = torch.from_numpy(pair_i.astype(np.int64))
pair_j = torch.from_numpy(pair_j.astype(np.int64))
n_pairs = len(pair_i)


def make_cat(parts):
    cols = []
    for field, offset in zip(CAT_FIELDS, offsets):
        if len(parts) == 1:
            x = np.asarray(parts[0].X[field], dtype=np.int64)
        else:
            x = np.concatenate([
                np.asarray(part.X[field], dtype=np.int64)
                for part in parts
            ])
        cols.append(x + offset)
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


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
        x = np.log1p(np.maximum(
            np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), 0.0
        ))
        med = float(np.median(x))
        q25, q75 = np.percentile(x, [25.0, 75.0])
        centers.append(med)
        scales.append(max(float(q75 - q25), 0.25))
    return np.asarray(centers, np.float32), np.asarray(scales, np.float32)


def make_num(parts, center, scale):
    cols = []
    for j, field in enumerate(NUM_FIELDS):
        if len(parts) == 1:
            x = np.asarray(parts[0].num[field], dtype=np.float32)
        else:
            x = np.concatenate([
                np.asarray(part.num[field], dtype=np.float32)
                for part in parts
            ])
        x = np.log1p(np.maximum(
            np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), 0.0
        ))
        x = np.clip((x - center[j]) / scale[j], -6.0, 6.0)
        cols.append(x.astype(np.float32))
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)


class CommonEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)
        self.wide = nn.Embedding(total_cardinality, 1)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.018)
        nn.init.zeros_(self.wide.weight)

    def forward(self, cat):
        emb = self.embedding(cat)
        wide = self.wide(cat).squeeze(-1).sum(dim=1)
        return emb, wide


class PNN(nn.Module):
    """Product Neural Network using explicit pairwise inner products."""
    def __init__(self):
        super().__init__()
        self.common = CommonEmbedding()
        self.register_buffer("pair_i", pair_i.clone())
        self.register_buffer("pair_j", pair_j.clone())
        input_dim = n_fields * EMBED_DIM + n_pairs + len(NUM_FIELDS)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 160),
            nn.ReLU(),
            nn.LayerNorm(160),
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, cat, num):
        emb, wide = self.common(cat)
        products = (
            emb[:, self.pair_i, :] * emb[:, self.pair_j, :]
        ).sum(dim=2)
        x = torch.cat([emb.flatten(1), products, num], dim=1)
        return self.mlp(x).squeeze(1) + wide + self.bias


class FiBiNET(nn.Module):
    """Field squeeze/excitation followed by explicit bilinear interactions."""
    def __init__(self):
        super().__init__()
        self.common = CommonEmbedding()
        self.register_buffer("pair_i", pair_i.clone())
        self.register_buffer("pair_j", pair_j.clone())
        reduction = max(4, n_fields // 3)
        self.squeeze = nn.Sequential(
            nn.Linear(n_fields, reduction),
            nn.ReLU(),
            nn.Linear(reduction, n_fields),
            nn.Sigmoid(),
        )
        self.bilinear = nn.Parameter(
            torch.empty(n_pairs, EMBED_DIM)
        )
        nn.init.normal_(self.bilinear, mean=1.0, std=0.05)

        input_dim = n_fields * EMBED_DIM + n_pairs * EMBED_DIM + len(NUM_FIELDS)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.ReLU(),
            nn.LayerNorm(192),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, cat, num):
        emb, wide = self.common(cat)
        field_summary = emb.mean(dim=2)
        gates = 2.0 * self.squeeze(field_summary)
        gated = emb * gates.unsqueeze(2)

        interactions = (
            gated[:, self.pair_i, :]
            * gated[:, self.pair_j, :]
            * self.bilinear.unsqueeze(0)
        )
        x = torch.cat([
            gated.flatten(1),
            interactions.flatten(1),
            num,
        ], dim=1)
        return self.mlp(x).squeeze(1) + wide + self.bias


class AutoInt(nn.Module):
    """Two residual self-attention blocks over categorical field tokens."""
    def __init__(self):
        super().__init__()
        self.common = CommonEmbedding()
        self.attn1 = nn.MultiheadAttention(
            EMBED_DIM, num_heads=2, dropout=0.0, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            EMBED_DIM, num_heads=2, dropout=0.0, batch_first=True
        )
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        input_dim = n_fields * EMBED_DIM + len(NUM_FIELDS)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, cat, num):
        emb, wide = self.common(cat)
        a1, _ = self.attn1(
            emb, emb, emb, need_weights=False
        )
        h = self.norm1(emb + a1)
        a2, _ = self.attn2(
            h, h, h, need_weights=False
        )
        h = self.norm2(h + a2)
        x = torch.cat([h.flatten(1), num], dim=1)
        return self.mlp(x).squeeze(1) + wide + self.bias


def make_model(family):
    if family == "pnn":
        return PNN()
    if family == "fibinet":
        return FiBiNET()
    if family == "autoint":
        return AutoInt()
    raise ValueError(family)


def fit_model(family, cat, num, labels, seed):
    torch.manual_seed(seed)
    model = make_model(family)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0020,
        weight_decay=2.0e-6,
    )

    cat_t = torch.from_numpy(cat)
    num_t = torch.from_numpy(num)
    y_t = torch.from_numpy(np.asarray(labels, dtype=np.float32))

    n = len(labels)
    for epoch in range(EPOCHS):
        generator = torch.Generator()
        generator.manual_seed(seed + 1009 * epoch)
        order = torch.randperm(n, generator=generator)

        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(cat_t[idx], num_t[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y_t[idx]
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def predict(model, cat, num):
    model.eval()
    cat_t = torch.from_numpy(cat)
    num_t = torch.from_numpy(num)
    scores = np.empty(len(cat), dtype=np.float64)

    with torch.no_grad():
        step = BATCH_SIZE * 2
        for start in range(0, len(cat), step):
            end = min(start + step, len(cat))
            scores[start:end] = model(
                cat_t[start:end], num_t[start:end]
            ).cpu().numpy()
    return scores


def within_user_rank(user_ids, scores):
    """Normalized ordinal rank in [0,1], computed without user loops."""
    users = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, users))
    sorted_users = users[order]

    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = sorted_users[1:] != sorted_users[:-1]

    positions = np.arange(n, dtype=np.int64)
    start_positions = np.where(starts_flag, positions, 0)
    start_positions = np.maximum.accumulate(start_positions)
    local_rank = positions - start_positions

    end_flag = np.empty(n, dtype=bool)
    end_flag[-1] = True
    end_flag[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_positions = np.where(end_flag, positions + 1, n)
    end_positions = np.minimum.accumulate(end_positions[::-1])[::-1]
    group_size = end_positions - start_positions

    normalized = local_rank / np.maximum(group_size - 1, 1)
    result = np.empty(n, dtype=np.float64)
    result[order] = normalized
    return result


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
cat_valid = make_cat([valid])
num_valid = make_num([valid], center, scale)

families = ["pnn", "fibinet", "autoint"]
family_seed = {
    "pnn": SEED + 101,
    "fibinet": SEED + 202,
    "autoint": SEED + 303,
}

raw_predictions = {}
for family in families:
    model = fit_model(
        family,
        cat_train,
        num_train,
        y_train,
        family_seed[family],
    )
    raw_predictions[family] = predict(model, cat_valid, num_valid)
    del model
    gc.collect()

candidate_results = {}
inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)
candidate_results["incumbent"] = float(inc_metrics["primary"])

best_name = "incumbent"
best_scores = inc_valid.copy()
best_metrics = inc_metrics
best_spec = {
    "family": "incumbent",
    "blend_type": "none",
    "alpha": 0.0,
    "scale": 1.0,
}
best_raw = None

inc_std = max(float(np.std(inc_valid)), 1.0e-8)
inc_rank = within_user_rank(valid.user_id, inc_valid)
alphas = [0.15, 0.25, 0.35, 0.50, 0.65]

for family in families:
    raw = raw_predictions[family]
    raw_metrics = evaluate(valid.user_id, y_valid, raw)
    candidate_results[family] = float(raw_metrics["primary"])

    if raw_metrics["primary"] > best_metrics["primary"]:
        best_name = family
        best_scores = raw.copy()
        best_metrics = raw_metrics
        best_raw = raw.copy()
        best_spec = {
            "family": family,
            "blend_type": "raw",
            "alpha": 1.0,
            "scale": 1.0,
        }

    raw_std = max(float(np.std(raw)), 1.0e-8)
    score_scale = inc_std / raw_std
    scaled_raw = raw * score_scale
    raw_rank = within_user_rank(valid.user_id, raw)

    for alpha in alphas:
        score_blend = (
            (1.0 - alpha) * inc_valid + alpha * scaled_raw
        )
        score_metrics = evaluate(valid.user_id, y_valid, score_blend)
        score_name = "%s_scoreblend_%.2f" % (family, alpha)
        candidate_results[score_name] = float(score_metrics["primary"])

        if score_metrics["primary"] > best_metrics["primary"]:
            best_name = score_name
            best_scores = score_blend.copy()
            best_metrics = score_metrics
            best_raw = raw.copy()
            best_spec = {
                "family": family,
                "blend_type": "score",
                "alpha": alpha,
                "scale": score_scale,
            }

        rank_blend = (
            (1.0 - alpha) * inc_rank + alpha * raw_rank
        )
        rank_metrics = evaluate(valid.user_id, y_valid, rank_blend)
        rank_name = "%s_rankblend_%.2f" % (family, alpha)
        candidate_results[rank_name] = float(rank_metrics["primary"])

        if rank_metrics["primary"] > best_metrics["primary"]:
            best_name = rank_name
            best_scores = rank_blend.copy()
            best_metrics = rank_metrics
            best_raw = raw.copy()
            best_spec = {
                "family": family,
                "blend_type": "rank",
                "alpha": alpha,
                "scale": 1.0,
            }

print(
    "CANDIDATES " + json.dumps(candidate_results, sort_keys=True),
    flush=True,
)
print(
    "FINDINGS structurally_distinct=true "
    "pnn=%.6f fibinet=%.6f autoint=%.6f selected=%s selected_type=%s"
    % (
        candidate_results["pnn"],
        candidate_results["fibinet"],
        candidate_results["autoint"],
        best_name,
        best_spec["blend_type"],
    ),
    flush=True,
)

test = load("test")
inc_test = np.load(inc_test_path).astype(np.float64)

if best_spec["family"] == "incumbent":
    test_scores = inc_test.copy()
else:
    selected_family = best_spec["family"]

    y_tv = np.concatenate([
        y_train,
        y_valid.astype(np.float32),
    ])
    center_tv, scale_tv = numeric_statistics([train, valid])
    cat_tv = make_cat([train, valid])
    num_tv = make_num([train, valid], center_tv, scale_tv)
    cat_test = make_cat([test])
    num_test = make_num([test], center_tv, scale_tv)

    refit = fit_model(
        selected_family,
        cat_tv,
        num_tv,
        y_tv,
        family_seed[selected_family],
    )
    test_raw = predict(refit, cat_test, num_test)

    blend_type = best_spec["blend_type"]
    alpha = float(best_spec["alpha"])

    if blend_type == "raw":
        test_scores = test_raw
    elif blend_type == "score":
        test_scores = (
            (1.0 - alpha) * inc_test
            + alpha * float(best_spec["scale"]) * test_raw
        )
    elif blend_type == "rank":
        inc_test_rank = within_user_rank(test.user_id, inc_test)
        raw_test_rank = within_user_rank(test.user_id, test_raw)
        test_scores = (
            (1.0 - alpha) * inc_test_rank
            + alpha * raw_test_rank
        )
    else:
        raise ValueError(blend_type)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if best_raw is not None and best_spec["blend_type"] in ("score", "rank"):
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.4f}'
    % (
        float(best_metrics["primary"]),
        float(best_metrics["gauc"]),
        float(best_metrics["ndcg@5"]),
        elapsed,
    )
)