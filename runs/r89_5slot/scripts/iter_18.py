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
SEED = 94037
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
]
CHUNK_LEN = 16
BATCH_SEQS = 512
EPOCHS = 2


def extract_cats(s):
    return np.column_stack([
        np.asarray(s.X[f], dtype=np.int64) for f in FIELDS
    ])


def make_sequence_pack(users, times, chunk_len=CHUNK_LEN):
    users = np.asarray(users, dtype=np.int64)
    times = np.asarray(times, dtype=np.int64)
    n = users.size
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, times, users))
    us = users[order]
    ts = times[order]

    new_user = np.empty(n, dtype=bool)
    new_user[0] = True
    new_user[1:] = us[1:] != us[:-1]
    user_starts = np.flatnonzero(new_user)
    user_counts = np.diff(np.r_[user_starts, n])
    repeated_starts = np.repeat(user_starts, user_counts)

    pos_sorted = np.arange(n, dtype=np.int64) - repeated_starts
    count_sorted = np.repeat(user_counts, user_counts)
    rev_sorted = count_sorted - 1 - pos_sorted

    new_chunk = new_user | ((pos_sorted % chunk_len) == 0)
    chunk_id_sorted = np.cumsum(new_chunk).astype(np.int64) - 1
    within_sorted = pos_sorted % chunk_len
    n_chunks = int(chunk_id_sorted[-1]) + 1

    padded_rows = np.full((n_chunks, chunk_len), -1, dtype=np.int64)
    padded_rows[chunk_id_sorted, within_sorted] = order

    pos = np.empty(n, dtype=np.float32)
    rev = np.empty(n, dtype=np.float32)
    count = np.empty(n, dtype=np.float32)
    prev_gap = np.empty(n, dtype=np.float32)

    denom = np.maximum(count_sorted - 1, 1)
    pos[order] = (pos_sorted / denom).astype(np.float32)
    rev[order] = (rev_sorted / denom).astype(np.float32)
    count[order] = np.log1p(count_sorted).astype(np.float32)

    gap_sorted = np.zeros(n, dtype=np.float32)
    if n > 1:
        same_user = us[1:] == us[:-1]
        gaps = np.maximum(ts[1:] - ts[:-1], 0).astype(np.float64)
        gap_sorted[1:] = np.where(
            same_user, np.log1p(gaps) / 20.0, 0.0
        ).astype(np.float32)
    prev_gap[order] = gap_sorted

    row_features = np.column_stack([
        pos,
        rev,
        count / 7.0,
        np.clip(prev_gap, 0.0, 2.0),
    ]).astype(np.float32)

    return {
        "rows": padded_rows,
        "features": row_features,
        "n_rows": n,
    }


def embedding_dims():
    dims = []
    for field in FIELDS:
        card = int(FEATURE_CARDINALITIES[field])
        if card >= 5000:
            dim = 12
        elif card >= 500:
            dim = 10
        elif card >= 50:
            dim = 8
        else:
            dim = 5
        dims.append(dim)
    return dims


class CategoricalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        dims = embedding_dims()
        self.output_dim = int(sum(dims))
        self.tables = nn.ModuleList([
            nn.Embedding(int(FEATURE_CARDINALITIES[f]), d)
            for f, d in zip(FIELDS, dims)
        ])
        for table in self.tables:
            nn.init.normal_(table.weight, std=0.025)

    def forward(self, cats):
        return torch.cat([
            table(cats[:, :, j])
            for j, table in enumerate(self.tables)
        ], dim=-1)


class BiGRUReranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CategoricalEncoder()
        input_dim = self.encoder.output_dim + 4
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=28,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(56 + input_dim, 48),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(48, 1),
        )

    def forward(self, cats, feats, mask):
        base = torch.cat([self.encoder(cats), feats], dim=-1)
        contextual, _ = self.gru(base)
        logits = self.head(torch.cat([base, contextual], dim=-1)).squeeze(-1)
        return logits


class DeepSetsReranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CategoricalEncoder()
        d = self.encoder.output_dim + 4
        self.phi = nn.Sequential(
            nn.Linear(d, 48),
            nn.SiLU(),
            nn.Linear(48, 40),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(d + 40 + 40, 64),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1),
        )

    def forward(self, cats, feats, mask):
        base = torch.cat([self.encoder(cats), feats], dim=-1)
        latent = self.phi(base)
        mf = mask.unsqueeze(-1).float()
        pooled = (latent * mf).sum(dim=1, keepdim=True)
        pooled = pooled / torch.clamp(mf.sum(dim=1, keepdim=True), min=1.0)
        pooled = pooled.expand_as(latent)
        logits = self.head(
            torch.cat([base, latent, pooled], dim=-1)
        ).squeeze(-1)
        return logits


def build_model(family):
    torch.manual_seed(SEED + (11 if family == "bigru" else 29))
    if family == "bigru":
        return BiGRUReranker()
    if family == "deepsets":
        return DeepSetsReranker()
    raise ValueError(family)


def gather_batch(cats, feats, labels, rows):
    mask_np = rows >= 0
    safe_rows = np.maximum(rows, 0)
    bc = torch.from_numpy(cats[safe_rows])
    bf = torch.from_numpy(feats[safe_rows])
    bm = torch.from_numpy(mask_np)
    by = None if labels is None else torch.from_numpy(labels[safe_rows])
    return bc, bf, bm, by


def train_model(family, cats, pack, labels, dates, epochs=EPOCHS):
    model = build_model(family)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0025, weight_decay=2e-6
    )

    labels = np.asarray(labels, dtype=np.float32)
    dates = np.asarray(dates, dtype=np.int64)
    max_date = int(dates.max())
    day_values = np.unique(dates)
    day_index = np.searchsorted(day_values, dates)
    max_day_index = int(day_index.max())
    recency = np.exp(
        -np.log(2.0) * (max_day_index - day_index) / 6.0
    ).astype(np.float32)
    recency /= float(recency.mean())

    chunk_rows = pack["rows"]
    n_chunks = chunk_rows.shape[0]
    rng = np.random.default_rng(SEED + (101 if family == "bigru" else 203))

    for epoch in range(epochs):
        sequence_ids = rng.permutation(n_chunks)
        model.train()

        for st in range(0, n_chunks, BATCH_SEQS):
            chosen = sequence_ids[st:st + BATCH_SEQS]
            rows = chunk_rows[chosen]
            bc, bf, bm, by = gather_batch(
                cats, pack["features"], labels, rows
            )
            safe_rows = np.maximum(rows, 0)
            bw = torch.from_numpy(recency[safe_rows])

            logits = model(bc, bf, bm)
            loss_rows = F.binary_cross_entropy_with_logits(
                logits, by, reduction="none"
            )

            # A modest per-sequence normalization prevents the long training
            # histories from completely dominating sparse evaluation-like lists.
            lengths = bm.sum(dim=1, keepdim=True).float()
            sequence_weight = torch.rsqrt(torch.clamp(lengths, min=1.0))
            weights = bw * sequence_weight * bm.float()
            loss = (loss_rows * weights).sum() / torch.clamp(
                weights.sum(), min=1.0
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.no_grad()
def predict_model(model, cats, pack):
    model.eval()
    scores = np.empty(pack["n_rows"], dtype=np.float32)
    rows_all = pack["rows"]

    for st in range(0, rows_all.shape[0], BATCH_SEQS * 2):
        rows = rows_all[st:st + BATCH_SEQS * 2]
        bc, bf, bm, _ = gather_batch(
            cats, pack["features"], None, rows
        )
        pred = model(bc, bf, bm).numpy()
        valid = rows >= 0
        scores[rows[valid]] = pred[valid]
    return scores


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, users))
    sorted_users = users[order]

    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(first)
    counts = np.diff(np.r_[starts, n])
    positions = np.arange(n, dtype=np.float64) - np.repeat(starts, counts)
    repeated_counts = np.repeat(counts, counts)

    ranks = positions / np.maximum(repeated_counts - 1, 1)
    ranks[repeated_counts == 1] = 0.5

    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def combine_splits(a, b, cats_a, cats_b):
    cats = np.concatenate([cats_a, cats_b], axis=0)
    labels = np.concatenate([
        np.asarray(a.y, dtype=np.float32),
        np.asarray(b.y, dtype=np.float32),
    ])
    dates = np.concatenate([
        np.asarray(a.date, dtype=np.int64),
        np.asarray(b.date, dtype=np.int64),
    ])
    times = np.concatenate([
        np.asarray(a.time_ms, dtype=np.int64),
        np.asarray(b.time_ms, dtype=np.int64),
    ])

    ua = np.asarray(a.user_id, dtype=np.int64)
    ub = np.asarray(b.user_id, dtype=np.int64)
    offset = int(max(ua.max(initial=0), ub.max(initial=0))) + 1
    grouped_users = np.concatenate([ua, ub + offset])
    return cats, labels, dates, times, grouped_users


train = load("train")
valid = load("valid")

train_cats = extract_cats(train)
valid_cats = extract_cats(valid)

train_users = np.asarray(train.user_id, dtype=np.int64)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)

train_pack = make_sequence_pack(train_users, train.time_ms)
valid_pack = make_sequence_pack(valid_users, valid.time_ms)

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

candidate_scores = {}
candidate_values = {}
raw_predictions = {}

inc_metrics = evaluate(valid_users, valid_y, inc_valid)
candidate_scores["incumbent"] = inc_valid
candidate_values["incumbent"] = float(inc_metrics["primary"])

best_name = "incumbent"
best_scores = inc_valid
best_metrics = inc_metrics
best_family = None
best_weight = 0.0
best_is_blend = False

inc_rank = within_user_rank(valid_users, inc_valid)

for family in ("deepsets", "bigru"):
    model = train_model(
        family,
        train_cats,
        train_pack,
        train_y,
        np.asarray(train.date, dtype=np.int64),
    )
    pred = predict_model(model, valid_cats, valid_pack).astype(np.float64)
    raw_predictions[family] = pred

    raw_met = evaluate(valid_users, valid_y, pred)
    candidate_values[family] = float(raw_met["primary"])
    candidate_scores[family] = pred

    if float(raw_met["primary"]) > float(best_metrics["primary"]):
        best_name = family
        best_scores = pred
        best_metrics = raw_met
        best_family = family
        best_weight = 1.0
        best_is_blend = False

    model_rank = within_user_rank(valid_users, pred)
    for weight in (0.10, 0.20, 0.30, 0.40, 0.50):
        name = f"{family}_rankblend_{weight:.2f}"
        blended = (1.0 - weight) * inc_rank + weight * model_rank
        met = evaluate(valid_users, valid_y, blended)
        candidate_values[name] = float(met["primary"])
        candidate_scores[name] = blended

        if float(met["primary"]) > float(best_metrics["primary"]):
            best_name = name
            best_scores = blended
            best_metrics = met
            best_family = family
            best_weight = weight
            best_is_blend = True

corr = float(np.corrcoef(
    within_user_rank(valid_users, raw_predictions["deepsets"]),
    within_user_rank(valid_users, raw_predictions["bigru"]),
)[0, 1])
print("FINDINGS " + json.dumps({
    "deepsets_bigru_within_user_rank_corr": corr,
    "train_chunks": int(train_pack["rows"].shape[0]),
    "valid_chunks": int(valid_pack["rows"].shape[0]),
    "selected": best_name,
}, sort_keys=True))
print("CANDIDATES " + json.dumps(candidate_values, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_is_blend and best_family is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_predictions[best_family], dtype=np.float64),
        )

test = load("test")
test_users = np.asarray(test.user_id, dtype=np.int64)

if best_family is None:
    test_scores = inc_test
else:
    combined_cats, combined_y, combined_dates, combined_times, combined_users = (
        combine_splits(train, valid, train_cats, valid_cats)
    )
    combined_pack = make_sequence_pack(combined_users, combined_times)

    final_model = train_model(
        best_family,
        combined_cats,
        combined_pack,
        combined_y,
        combined_dates,
    )

    test_cats = extract_cats(test)
    test_pack = make_sequence_pack(test_users, test.time_ms)
    raw_test = predict_model(
        final_model, test_cats, test_pack
    ).astype(np.float64)

    if best_is_blend:
        test_scores = (
            (1.0 - best_weight) * within_user_rank(test_users, inc_test)
            + best_weight * within_user_rank(test_users, raw_test)
        )
    else:
        test_scores = raw_test

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))