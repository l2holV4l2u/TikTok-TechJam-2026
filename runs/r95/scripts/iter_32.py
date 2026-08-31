import os
import time
import json
import math
import gc
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

START = time.time()
SEED = 20260831
THREADS = min(8, os.cpu_count() or 8)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.float32)
yva = np.asarray(valid.y, dtype=np.int8)
uva = np.asarray(valid.user_id, dtype=np.int64)
ute = np.asarray(test.user_id, dtype=np.int64)

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "music_type",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

offsets = []
total_cardinality = 0
for field in CAT_FIELDS:
    offsets.append(total_cardinality)
    total_cardinality += int(FEATURE_CARDINALITIES[field])
offsets = np.asarray(offsets, dtype=np.int64)


def make_cat(sample):
    x = np.column_stack([
        np.asarray(sample.X[f], dtype=np.int64) for f in CAT_FIELDS
    ])
    x += offsets[None, :]
    return x.astype(np.int64, copy=False)


def make_raw_num(sample):
    cols = []
    for field in NUM_FIELDS:
        v = np.asarray(sample.num[field], dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(np.log1p(np.maximum(v, 0.0)))
    return np.column_stack(cols).astype(np.float32, copy=False)


def make_history(split_name):
    cols = []
    names = []
    for key in ("video_id", "author_id"):
        h = historical_features(split_name, key=key)
        for name in sorted(h):
            v = np.asarray(h[name], dtype=np.float32)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            if (
                "count" in name.lower()
                or "num" in name.lower()
                or (v.size and np.max(np.abs(v)) > 50.0)
            ):
                v = np.sign(v) * np.log1p(np.abs(v))
            cols.append(v)
            names.append(key + ":" + name)
    if not cols:
        length = (
            len(train.user_id) if split_name == "train"
            else len(valid.user_id) if split_name == "valid"
            else len(test.user_id)
        )
        return np.zeros((length, 0), dtype=np.float32), names
    return np.column_stack(cols).astype(np.float32, copy=False), names


SLATE_SIZE = 24


def make_slates(sample):
    """
    Consecutive chunks within user-day sequences. Context features use only
    exposure metadata available before outcomes: chunk position, chunk size,
    and elapsed logged time from the beginning of the chunk.
    """
    n = len(sample.user_id)
    users = np.asarray(sample.user_id, dtype=np.int64)
    dates = np.asarray(sample.date, dtype=np.int64)
    times = np.asarray(sample.time_ms, dtype=np.int64)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, times, dates, users))
    su = users[order]
    sd = dates[order]

    starts = np.flatnonzero(
        np.r_[True, (su[1:] != su[:-1]) | (sd[1:] != sd[:-1])]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    within = np.arange(n, dtype=np.int64) - np.repeat(starts, lengths)
    base_group = np.repeat(np.arange(len(starts), dtype=np.int64), lengths)
    chunk_number = within // SLATE_SIZE
    multiplier = int(chunk_number.max()) + 1 if n else 1
    keys = base_group * multiplier + chunk_number

    boundary = np.r_[True, keys[1:] != keys[:-1]]
    chunk_starts = np.flatnonzero(boundary)
    chunk_lengths = np.diff(np.r_[chunk_starts, n]).astype(np.int64)
    chunk_id = np.cumsum(boundary, dtype=np.int64) - 1
    position = np.arange(n, dtype=np.int64) - np.repeat(
        chunk_starts, chunk_lengths
    )

    slates = np.full(
        (len(chunk_starts), SLATE_SIZE), -1, dtype=np.int64
    )
    slates[chunk_id, position] = order

    contextual = np.zeros((n, 4), dtype=np.float32)
    ordered_times = times[order]
    first_times = ordered_times[np.repeat(chunk_starts, chunk_lengths)]
    elapsed_s = np.maximum(
        (ordered_times - first_times).astype(np.float64) / 1000.0, 0.0
    )

    contextual[order, 0] = (
        position.astype(np.float32)
        / np.maximum(np.repeat(chunk_lengths, chunk_lengths) - 1, 1)
    )
    contextual[order, 1] = np.log1p(
        np.repeat(chunk_lengths, chunk_lengths).astype(np.float32)
    )
    contextual[order, 2] = np.log1p(elapsed_s).astype(np.float32)

    same_timestamp_start = np.r_[
        True,
        (ordered_times[1:] != ordered_times[:-1])
        | (su[1:] != su[:-1])
        | (sd[1:] != sd[:-1]),
    ]
    timestamp_starts = np.flatnonzero(same_timestamp_start)
    timestamp_lengths = np.diff(np.r_[timestamp_starts, n])
    timestamp_pos = np.arange(n) - np.repeat(
        timestamp_starts, timestamp_lengths
    )
    contextual[order, 3] = (
        timestamp_pos.astype(np.float32)
        / np.maximum(np.repeat(timestamp_lengths, timestamp_lengths) - 1, 1)
    )

    return slates, contextual


xtr_cat = make_cat(train)
xva_cat = make_cat(valid)
xte_cat = make_cat(test)

htr, history_names = make_history("train")
hva, _ = make_history("valid")
hte, _ = make_history("test")

tr_slates, ctr = make_slates(train)
va_slates, cva = make_slates(valid)
te_slates, cte = make_slates(test)

xtr_num = np.column_stack([make_raw_num(train), htr, ctr]).astype(
    np.float32, copy=False
)
xva_num = np.column_stack([make_raw_num(valid), hva, cva]).astype(
    np.float32, copy=False
)
xte_num = np.column_stack([make_raw_num(test), hte, cte]).astype(
    np.float32, copy=False
)

# Train-only robust normalization.
center = np.median(xtr_num, axis=0).astype(np.float32)
q25 = np.quantile(xtr_num, 0.25, axis=0).astype(np.float32)
q75 = np.quantile(xtr_num, 0.75, axis=0).astype(np.float32)
scale = np.maximum(q75 - q25, 1.0e-3).astype(np.float32)

xtr_num = np.clip((xtr_num - center) / scale, -8.0, 8.0).astype(
    np.float32, copy=False
)
xva_num = np.clip((xva_num - center) / scale, -8.0, 8.0).astype(
    np.float32, copy=False
)
xte_num = np.clip((xte_num - center) / scale, -8.0, 8.0).astype(
    np.float32, copy=False
)

# Fixed four-day recency weighting, selected a priori rather than on valid.
unique_dates = np.unique(np.asarray(train.date, dtype=np.int64))
day_index = np.searchsorted(
    unique_dates, np.asarray(train.date, dtype=np.int64)
)
age = (len(unique_dates) - 1 - day_index).astype(np.float32)
train_weights = np.exp2(-age / 4.0).astype(np.float32)
train_weights /= train_weights.mean()

print(
    "FINDINGS history_features=%d train_slates=%d valid_slates=%d "
    "mean_train_slate=%.3f"
    % (
        len(history_names),
        len(tr_slates),
        len(va_slates),
        float(np.mean(np.sum(tr_slates >= 0, axis=1))),
    )
)


class RowEncoder(nn.Module):
    def __init__(self, cardinality, fields, numeric_dim):
        super().__init__()
        emb_dim = 8
        hidden = 72
        self.embedding = nn.Embedding(cardinality, emb_dim)
        self.input = nn.Sequential(
            nn.Linear(fields * emb_dim + numeric_dim, 128),
            nn.SiLU(),
            nn.Linear(128, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        nn.init.normal_(self.embedding.weight, std=0.018)

    def forward(self, cat, num):
        emb = self.embedding(cat).flatten(2)
        return self.input(torch.cat([emb, num], dim=2))


class PointwiseScorer(nn.Module):
    """Independent candidate utility; serves as the no-slate control."""

    def __init__(self, cardinality, fields, numeric_dim):
        super().__init__()
        self.encoder = RowEncoder(cardinality, fields, numeric_dim)
        self.head = nn.Sequential(
            nn.Linear(72, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )

    def forward(self, cat, num, mask):
        h = self.encoder(cat, num)
        return self.head(h).squeeze(-1)


class DeepSetScorer(nn.Module):
    """
    Permutation-equivariant set scorer. Each row is compared with pooled
    first- and second-order statistics of the currently exposed slate.
    """

    def __init__(self, cardinality, fields, numeric_dim):
        super().__init__()
        self.encoder = RowEncoder(cardinality, fields, numeric_dim)
        self.head = nn.Sequential(
            nn.Linear(72 * 4, 128),
            nn.SiLU(),
            nn.Linear(128, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )

    def forward(self, cat, num, mask):
        h = self.encoder(cat, num)
        m = mask.unsqueeze(-1).float()
        count = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (h * m).sum(dim=1, keepdim=True) / count
        second = (h.square() * m).sum(dim=1, keepdim=True) / count
        std = (second - mean.square()).clamp_min(0.0).sqrt()
        mean = mean.expand_as(h)
        std = std.expand_as(h)
        z = torch.cat([h, h - mean, h * mean, std], dim=-1)
        return self.head(z).squeeze(-1)


class SetAttentionScorer(nn.Module):
    """
    Cross-impression self-attention forms each utility from candidate-specific
    interactions with all other candidates in the same logged slate.
    """

    def __init__(self, cardinality, fields, numeric_dim):
        super().__init__()
        self.encoder = RowEncoder(cardinality, fields, numeric_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=72, num_heads=4, dropout=0.05, batch_first=True
        )
        self.norm = nn.LayerNorm(72)
        self.ff = nn.Sequential(
            nn.Linear(72, 144),
            nn.SiLU(),
            nn.Linear(144, 72),
        )
        self.norm2 = nn.LayerNorm(72)
        self.head = nn.Sequential(
            nn.Linear(144, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, cat, num, mask):
        h = self.encoder(cat, num)
        attended, _ = self.attention(
            h, h, h, key_padding_mask=~mask, need_weights=False
        )
        a = self.norm(h + attended)
        a = self.norm2(a + self.ff(a))
        return self.head(torch.cat([h, a], dim=-1)).squeeze(-1)


def fit_model(model, model_seed):
    rng = np.random.default_rng(model_seed)
    torch.manual_seed(model_seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0018, weight_decay=2.0e-6
    )
    batch_size = 192

    model.train()
    epoch_losses = []
    for epoch in range(2):
        permutation = rng.permutation(len(tr_slates))
        total_loss = 0.0
        total_rows = 0

        for begin in range(0, len(permutation), batch_size):
            chosen = permutation[begin:begin + batch_size]
            rows = tr_slates[chosen]
            mask_np = rows >= 0
            safe = np.maximum(rows, 0)

            cat = torch.from_numpy(xtr_cat[safe])
            num = torch.from_numpy(xtr_num[safe])
            labels = torch.from_numpy(ytr[safe])
            weights = torch.from_numpy(train_weights[safe])
            mask = torch.from_numpy(mask_np)

            logits = model(cat, num, mask)
            element = nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )
            effective_weight = weights * mask.float()
            loss = (element * effective_weight).sum() / (
                effective_weight.sum().clamp_min(1.0)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss.detach()) * int(mask_np.sum())
            total_rows += int(mask_np.sum())

        epoch_losses.append(total_loss / max(total_rows, 1))

    return epoch_losses


@torch.no_grad()
def predict_model(model, slates, xcat, xnum):
    model.eval()
    result = np.zeros(len(xcat), dtype=np.float32)
    batch_size = 256

    for begin in range(0, len(slates), batch_size):
        rows = slates[begin:begin + batch_size]
        mask_np = rows >= 0
        safe = np.maximum(rows, 0)

        cat = torch.from_numpy(xcat[safe])
        num = torch.from_numpy(xnum[safe])
        mask = torch.from_numpy(mask_np)

        logits = model(cat, num, mask).cpu().numpy()
        result[rows[mask_np]] = logits[mask_np]

    return result


model_specs = [
    (
        "pointwise_context_control",
        PointwiseScorer,
        SEED + 11,
    ),
    (
        "deepsets_slate",
        DeepSetScorer,
        SEED + 29,
    ),
    (
        "set_attention_slate",
        SetAttentionScorer,
        SEED + 47,
    ),
]

valid_predictions = {}
test_predictions = {}
candidate_scores = {}

for name, model_class, model_seed in model_specs:
    model = model_class(
        total_cardinality, len(CAT_FIELDS), xtr_num.shape[1]
    )
    losses = fit_model(model, model_seed)
    pv = predict_model(model, va_slates, xva_cat, xva_num)
    pt = predict_model(model, te_slates, xte_cat, xte_num)

    valid_predictions[name] = pv
    test_predictions[name] = pt
    metric = evaluate(uva, yva, pv)
    candidate_scores[name] = float(metric["primary"])
    print(
        "FINDINGS %s train_losses=%s valid_gauc=%.6f valid_ndcg5=%.6f"
        % (
            name,
            ",".join("%.5f" % x for x in losses),
            float(metric["gauc"]),
            float(metric["ndcg@5"]),
        )
    )

    del model
    gc.collect()


def within_user_rank(user_ids, scores):
    """
    Scale-free rank representation for combining heterogeneous model logits.
    Transformation is monotone within each user and thus leaves a standalone
    model's evaluated ordering unchanged.
    """
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    lengths = np.diff(np.r_[starts, n])
    positions = np.arange(n, dtype=np.int64) - np.repeat(starts, lengths)
    denominators = np.maximum(np.repeat(lengths, lengths) - 1, 1)

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = positions.astype(np.float64) / denominators
    return ranked


shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

winner_name = max(candidate_scores, key=candidate_scores.get)
winner_valid = valid_predictions[winner_name].astype(np.float64)
winner_test = test_predictions[winner_name].astype(np.float64)
winner_raw_valid = winner_valid.copy()
winner_metric = evaluate(uva, yva, winner_valid)

if os.path.exists(inc_valid_path) and os.path.exists(inc_test_path):
    inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
    inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

    inc_valid_rank = within_user_rank(uva, inc_valid)
    inc_test_rank = within_user_rank(ute, inc_test)

    # Fixed grid, with alpha denoting the contribution of the new family.
    blend_alphas = (0.15, 0.30, 0.50, 0.70)

    for name in list(valid_predictions):
        own_valid_rank = within_user_rank(uva, valid_predictions[name])
        own_test_rank = within_user_rank(ute, test_predictions[name])

        for alpha in blend_alphas:
            bv = (
                (1.0 - alpha) * inc_valid_rank
                + alpha * own_valid_rank
            )
            bt = (
                (1.0 - alpha) * inc_test_rank
                + alpha * own_test_rank
            )
            blend_name = "%s_blend_%.2f" % (name, alpha)
            bm = evaluate(uva, yva, bv)
            candidate_scores[blend_name] = float(bm["primary"])

            if float(bm["primary"]) > float(winner_metric["primary"]):
                winner_name = blend_name
                winner_valid = bv
                winner_test = bt
                winner_raw_valid = np.asarray(
                    valid_predictions[name], dtype=np.float64
                )
                winner_metric = bm

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print("FINDINGS selected=" + winner_name)

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
    if "_blend_" in winner_name:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(winner_raw_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(winner_metric["primary"]),
            "gauc": float(winner_metric["gauc"]),
            "ndcg@5": float(winner_metric["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)