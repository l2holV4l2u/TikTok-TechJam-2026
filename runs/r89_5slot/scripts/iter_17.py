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
SEED = 91427
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

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
NB_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "video_type",
    "hour",
]
CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]


def extract_cats(s, fields=FIELDS):
    return np.column_stack([
        np.asarray(s.X[f], dtype=np.int32) for f in fields
    ]).astype(np.int32, copy=False)


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x))
    if sd < 1e-12:
        sd = 1.0
    return (x - float(np.mean(x))) / sd


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, users))
    sorted_users = users[order]

    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(first)
    counts = np.diff(np.r_[starts, n])
    positions = np.arange(n, dtype=np.float64) - np.repeat(starts, counts)
    repeated = np.repeat(counts, counts)
    ranks = positions / np.maximum(repeated - 1, 1)
    ranks[repeated == 1] = 0.5

    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


class FirstOrder(nn.Module):
    def __init__(self, cards):
        super().__init__()
        self.tables = nn.ModuleList([nn.Embedding(c, 1) for c in cards])
        for table in self.tables:
            nn.init.zeros_(table.weight)

    def forward(self, cats):
        out = 0.0
        for j, table in enumerate(self.tables):
            out = out + table(cats[:, j]).squeeze(-1)
        return out


class AutoIntModel(nn.Module):
    def __init__(self, cards, dim=8):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c, dim) for c in cards])
        self.first = FirstOrder(cards)
        self.attn1 = nn.MultiheadAttention(
            dim, num_heads=2, dropout=0.05, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            dim, num_heads=2, dropout=0.05, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.output = nn.Linear(len(cards) * dim, 1)
        self.bias = nn.Parameter(torch.zeros(()))
        for table in self.emb:
            nn.init.normal_(table.weight, std=0.02)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, cats):
        x = torch.stack(
            [table(cats[:, j]) for j, table in enumerate(self.emb)], dim=1
        )
        a, _ = self.attn1(x, x, x, need_weights=False)
        x = self.norm1(x + F.silu(a))
        a, _ = self.attn2(x, x, x, need_weights=False)
        x = self.norm2(x + F.silu(a))
        return (
            self.output(x.flatten(1)).squeeze(1)
            + self.first(cats)
            + self.bias
        )


class CINLayer(nn.Module):
    def __init__(self, in_fields, base_fields, out_fields):
        super().__init__()
        self.in_fields = in_fields
        self.base_fields = base_fields
        self.conv = nn.Conv1d(
            in_fields * base_fields, out_fields, kernel_size=1
        )

    def forward(self, x0, x):
        interactions = torch.einsum("bhd,bfd->bhfd", x, x0)
        b, h, f, d = interactions.shape
        interactions = interactions.reshape(b, h * f, d)
        return F.silu(self.conv(interactions))


class XDeepFMModel(nn.Module):
    def __init__(self, cards, dim=8):
        super().__init__()
        nf = len(cards)
        self.emb = nn.ModuleList([nn.Embedding(c, dim) for c in cards])
        self.first = FirstOrder(cards)
        self.cin1 = CINLayer(nf, nf, 16)
        self.cin2 = CINLayer(16, nf, 16)
        self.cin_out = nn.Linear(32, 1)
        self.deep = nn.Sequential(
            nn.Linear(nf * dim, 64),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 24),
            nn.SiLU(),
            nn.Linear(24, 1),
        )
        self.bias = nn.Parameter(torch.zeros(()))
        for table in self.emb:
            nn.init.normal_(table.weight, std=0.02)

    def forward(self, cats):
        x0 = torch.stack(
            [table(cats[:, j]) for j, table in enumerate(self.emb)], dim=1
        )
        h1 = self.cin1(x0, x0)
        h2 = self.cin2(x0, h1)
        cin = torch.cat([h1.sum(dim=2), h2.sum(dim=2)], dim=1)
        return (
            self.first(cats)
            + self.cin_out(cin).squeeze(1)
            + self.deep(x0.flatten(1)).squeeze(1)
            + self.bias
        )


class TwoTowerModel(nn.Module):
    def __init__(self, cards, dim=24):
        super().__init__()
        self.user = nn.Embedding(cards[0], dim)
        self.video = nn.Embedding(cards[1], dim)
        self.author = nn.Embedding(cards[2], dim)
        self.tag = nn.Embedding(cards[4], dim)
        self.context = nn.ModuleList([
            nn.Embedding(cards[j], dim)
            for j in (3, 5, 6, 7, 8, 9, 10, 11)
        ])
        self.user_bias = nn.Embedding(cards[0], 1)
        self.video_bias = nn.Embedding(cards[1], 1)
        self.author_bias = nn.Embedding(cards[2], 1)
        self.global_bias = nn.Parameter(torch.zeros(()))

        for module in [self.user, self.video, self.author, self.tag] + list(self.context):
            nn.init.normal_(module.weight, std=0.025)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.video_bias.weight)
        nn.init.zeros_(self.author_bias.weight)

    def forward(self, cats):
        u = self.user(cats[:, 0])
        item = (
            self.video(cats[:, 1])
            + 0.55 * self.author(cats[:, 2])
            + 0.35 * self.tag(cats[:, 4])
        )
        context_fields = (3, 5, 6, 7, 8, 9, 10, 11)
        context = 0.0
        for table, j in zip(self.context, context_fields):
            context = context + table(cats[:, j])
        context = context / np.sqrt(len(context_fields))
        u = F.normalize(u + 0.20 * context, dim=1)
        item = F.normalize(item, dim=1)
        match = 4.0 * torch.sum(u * item, dim=1)
        return (
            match
            + self.user_bias(cats[:, 0]).squeeze(1)
            + self.video_bias(cats[:, 1]).squeeze(1)
            + 0.5 * self.author_bias(cats[:, 2]).squeeze(1)
            + self.global_bias
        )


def build_model(name):
    if name == "autoint":
        return AutoIntModel(CARDS)
    if name == "xdeepfm":
        return XDeepFMModel(CARDS)
    if name == "two_tower":
        return TwoTowerModel(CARDS)
    raise ValueError(name)


def train_neural(name, cats, y, dates, epochs=2):
    torch.manual_seed(SEED + {
        "autoint": 11,
        "xdeepfm": 23,
        "two_tower": 37,
    }[name])
    model = build_model(name)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0022 if name != "two_tower" else 0.0030,
        weight_decay=2e-6 if name != "two_tower" else 2e-5,
    )

    cats_t = torch.from_numpy(cats.astype(np.int64, copy=False))
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
    dates = np.asarray(dates, dtype=np.int64)
    max_date = int(dates.max())
    age = (max_date - dates).astype(np.float32)
    weights = np.exp(-np.maximum(age, 0.0) / 18.0).astype(np.float32)
    weights *= 1.0 / max(float(weights.mean()), 1e-6)
    w_t = torch.from_numpy(weights)

    n = len(y)
    batch_size = 8192 if name != "autoint" else 6144
    rng = np.random.default_rng(SEED + 101)

    for epoch in range(epochs):
        order = rng.permutation(n)
        model.train()
        for st in range(0, n, batch_size):
            idx_np = order[st:st + batch_size]
            idx = torch.from_numpy(idx_np)
            logits = model(cats_t[idx])
            losses = F.binary_cross_entropy_with_logits(
                logits, y_t[idx], reduction="none"
            )
            loss = (losses * w_t[idx]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


@torch.no_grad()
def predict_neural(model, cats):
    model.eval()
    result = np.empty(len(cats), dtype=np.float32)
    batch_size = 32768
    cats_t = torch.from_numpy(cats.astype(np.int64, copy=False))
    for st in range(0, len(cats), batch_size):
        en = min(st + batch_size, len(cats))
        result[st:en] = model(cats_t[st:en]).cpu().numpy()
    return result


def fit_naive_bayes(split_list, y_list):
    total_y = np.concatenate([
        np.asarray(y, dtype=np.float64) for y in y_list
    ])
    global_rate = float(np.clip(total_y.mean(), 1e-5, 1.0 - 1e-5))
    global_logit = np.log(global_rate / (1.0 - global_rate))
    tables = {}

    for field in NB_FIELDS:
        card = int(FEATURE_CARDINALITIES[field])
        count = np.zeros(card, dtype=np.float64)
        positive = np.zeros(card, dtype=np.float64)
        for s, y in zip(split_list, y_list):
            ids = np.asarray(s.X[field], dtype=np.int64)
            yy = np.asarray(y, dtype=np.float64)
            count += np.bincount(ids, minlength=card)
            positive += np.bincount(ids, weights=yy, minlength=card)

        strength = 35.0 if field in ("video_id", "author_id") else 80.0
        rate = (positive + strength * global_rate) / (count + strength)
        rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
        evidence = np.log(rate / (1.0 - rate)) - global_logit

        reliability = count / (count + strength)
        tables[field] = (evidence * reliability).astype(np.float32)

    return global_logit, tables


def predict_naive_bayes(s, fitted):
    global_logit, tables = fitted
    score = np.full(len(s.user_id), global_logit, dtype=np.float32)
    norm = 0.0
    weights = {
        "video_id": 1.4,
        "author_id": 1.1,
        "tab": 0.9,
        "tag": 0.8,
        "duration_bucket": 0.5,
        "upload_type": 0.7,
        "music_type": 0.4,
        "onehot_feat3": 0.8,
        "onehot_feat7": 0.4,
        "onehot_feat8": 0.7,
        "video_type": 0.3,
        "hour": 0.35,
    }
    for field in NB_FIELDS:
        w = weights[field]
        score += w * tables[field][np.asarray(s.X[field], dtype=np.int64)]
        norm += w
    return (global_logit + (score - global_logit) / max(norm / 3.0, 1.0)).astype(
        np.float32
    )


def apply_combination(mode, alpha, users, incumbent, raw):
    if mode == "raw":
        return np.asarray(raw, dtype=np.float64)
    if mode == "zblend":
        return (
            (1.0 - alpha) * zscore(incumbent)
            + alpha * zscore(raw)
        )
    if mode == "rankblend":
        return (
            (1.0 - alpha) * within_user_rank(users, incumbent)
            + alpha * within_user_rank(users, raw)
        )
    if mode == "incumbent":
        return np.asarray(incumbent, dtype=np.float64)
    raise ValueError(mode)


train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
train_users = np.asarray(train.user_id, dtype=np.int64)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

train_cats = extract_cats(train)
valid_cats = extract_cats(valid)

shared = os.environ["SHARED_ARTIFACTS"]
inc_valid = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(shared, "incumbent_test_scores.npy")),
    dtype=np.float64,
)

candidate_scores = {}
candidate_specs = {}
raw_predictions = {}

inc_metrics = evaluate(valid_users, valid_y, inc_valid)
candidate_scores["incumbent"] = float(inc_metrics["primary"])
candidate_specs["incumbent"] = ("incumbent", "incumbent", 0.0)

nb_fit = fit_naive_bayes([train], [train_y])
raw_predictions["naive_bayes"] = predict_naive_bayes(valid, nb_fit)

for family in ("two_tower", "autoint", "xdeepfm"):
    model = train_neural(
        family,
        train_cats,
        train_y,
        np.asarray(train.date),
        epochs=2,
    )
    raw_predictions[family] = predict_neural(model, valid_cats)
    del model

best_name = "incumbent"
best_metrics = inc_metrics
best_scores = inc_valid.copy()
best_spec = candidate_specs["incumbent"]
best_primary = float(inc_metrics["primary"])

for family, raw in raw_predictions.items():
    raw_met = evaluate(valid_users, valid_y, raw)
    raw_name = family + "_raw"
    candidate_scores[raw_name] = float(raw_met["primary"])
    candidate_specs[raw_name] = (family, "raw", 1.0)

    if float(raw_met["primary"]) > best_primary:
        best_primary = float(raw_met["primary"])
        best_name = raw_name
        best_metrics = raw_met
        best_scores = np.asarray(raw, dtype=np.float64)
        best_spec = candidate_specs[raw_name]

    for mode in ("zblend", "rankblend"):
        for alpha in (0.08, 0.14, 0.20, 0.28, 0.36, 0.46):
            combined = apply_combination(
                mode, alpha, valid_users, inc_valid, raw
            )
            met = evaluate(valid_users, valid_y, combined)
            name = f"{family}_{mode}_{alpha:.2f}"
            candidate_scores[name] = float(met["primary"])
            candidate_specs[name] = (family, mode, alpha)
            if float(met["primary"]) > best_primary:
                best_primary = float(met["primary"])
                best_name = name
                best_metrics = met
                best_scores = combined
                best_spec = candidate_specs[name]

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps({
        "winner": best_name,
        "winner_spec": list(best_spec),
        "raw_primaries": {
            k: candidate_scores[k + "_raw"] for k in raw_predictions
        },
    }, sort_keys=True)
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_spec[0] in raw_predictions and best_spec[1] != "raw":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_predictions[best_spec[0]], dtype=np.float64),
        )

test = load("test")
test_users = np.asarray(test.user_id, dtype=np.int64)

winner_family, winner_mode, winner_alpha = best_spec

if winner_family == "incumbent":
    test_scores = inc_test.copy()
else:
    combined_cats = np.concatenate(
        [train_cats, valid_cats], axis=0
    )
    combined_y = np.concatenate(
        [train_y, valid_y.astype(np.float32)], axis=0
    )
    combined_dates = np.concatenate([
        np.asarray(train.date),
        np.asarray(valid.date),
    ])

    if winner_family == "naive_bayes":
        final_fit = fit_naive_bayes(
            [train, valid], [train_y, valid_y.astype(np.float32)]
        )
        raw_test = predict_naive_bayes(test, final_fit)
    else:
        test_cats = extract_cats(test)
        final_model = train_neural(
            winner_family,
            combined_cats,
            combined_y,
            combined_dates,
            epochs=2,
        )
        raw_test = predict_neural(final_model, test_cats)

    test_scores = apply_combination(
        winner_mode,
        winner_alpha,
        test_users,
        inc_test,
        raw_test,
    )

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
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
    })
)