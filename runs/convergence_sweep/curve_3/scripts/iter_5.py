import os
import time
import json
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 19437
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "duration_bucket",
    "music_type",
    "user_active_degree",
    "register_days_bucket",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

HISTORY_SUFFIXES = [
    "train_count_log1p",
    "long_view_rate",
    "is_click_rate",
    "play_time_ms_logmean",
]

SEQ_LEN = 10
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 16384
EPOCHS = 2
HALF_LIFE_DAYS = 7.0


def load_histories(split_name):
    return {
        "video_id": historical_features(split_name, key="video_id"),
        "author_id": historical_features(split_name, key="author_id"),
    }


def make_cat_matrix(split):
    cards = [int(FEATURE_CARDINALITIES[name]) for name in CAT_FIELDS]
    offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
    x = np.empty((len(split.user_id), len(CAT_FIELDS)), dtype=np.int64)
    for j, name in enumerate(CAT_FIELDS):
        x[:, j] = np.asarray(split.X[name], dtype=np.int64) + offsets[j]
    return np.ascontiguousarray(x), cards, offsets


def raw_numeric_matrix(split, histories):
    cols = []
    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))

    for entity in ("video_id", "author_id"):
        h = histories[entity]
        for suffix in HISTORY_SUFFIXES:
            key = entity + "_" + suffix
            x = np.asarray(h[key], dtype=np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            cols.append(x)

    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)


def fit_numeric_transform(x):
    lo = np.quantile(x, 0.002, axis=0).astype(np.float32)
    hi = np.quantile(x, 0.998, axis=0).astype(np.float32)
    clipped = np.clip(x, lo, hi)
    mean = clipped.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = clipped.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-4] = 1.0
    return lo, hi, mean, std


def apply_numeric_transform(x, transform):
    lo, hi, mean, std = transform
    return np.ascontiguousarray(
        (np.clip(x, lo, hi) - mean) / std,
        dtype=np.float32,
    )


def build_positive_histories(train):
    users = np.asarray(train.user_id, dtype=np.int64)
    videos = np.asarray(train.video_id, dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.int8)
    times = np.asarray(train.time_ms, dtype=np.int64)
    rows = np.arange(len(users), dtype=np.int64)

    order = np.lexsort((rows, times, users))
    su = users[order]
    sv = videos[order]
    sy = labels[order]

    positive = sy > 0
    global_before = np.cumsum(positive, dtype=np.int64)
    global_before -= positive.astype(np.int64)

    starts = np.empty(len(order), dtype=bool)
    starts[0] = True
    starts[1:] = su[1:] != su[:-1]
    start_indices = np.maximum.accumulate(
        np.where(starts, np.arange(len(order), dtype=np.int64), 0)
    )
    user_positive_base = global_before[start_indices]
    prior_count = global_before - user_positive_base
    positive_videos = sv[positive]

    seq_sorted = np.zeros((len(order), SEQ_LEN), dtype=np.int32)
    mask_sorted = np.zeros((len(order), SEQ_LEN), dtype=np.bool_)
    for lag in range(1, SEQ_LEN + 1):
        valid = prior_count >= lag
        indices = user_positive_base[valid] + prior_count[valid] - lag
        seq_sorted[valid, lag - 1] = positive_videos[indices].astype(np.int32)
        mask_sorted[valid, lag - 1] = True

    seq_train = np.empty_like(seq_sorted)
    mask_train = np.empty_like(mask_sorted)
    seq_train[order] = seq_sorted
    mask_train[order] = mask_sorted

    n_users = int(FEATURE_CARDINALITIES["user_id"])
    user_counts = np.bincount(
        su[positive], minlength=n_users
    ).astype(np.int64)
    ends = np.cumsum(user_counts, dtype=np.int64)

    final_seq = np.zeros((n_users, SEQ_LEN), dtype=np.int32)
    final_mask = np.zeros((n_users, SEQ_LEN), dtype=np.bool_)
    user_axis = np.arange(n_users, dtype=np.int64)

    for lag in range(1, SEQ_LEN + 1):
        valid_users = user_counts >= lag
        indices = ends[valid_users] - lag
        final_seq[user_axis[valid_users], lag - 1] = (
            positive_videos[indices].astype(np.int32)
        )
        final_mask[user_axis[valid_users], lag - 1] = True

    return seq_train, mask_train, final_seq, final_mask


def histories_for_split(split, final_seq, final_mask):
    users = np.asarray(split.user_id, dtype=np.int64)
    return (
        np.ascontiguousarray(final_seq[users]),
        np.ascontiguousarray(final_mask[users]),
    )


def centered_scaled(scores, user_ids, scale=None):
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(user_ids, dtype=np.int64)
    _, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=scores)
    centered = scores - (sums / np.maximum(counts, 1))[inverse]

    if scale is None:
        scale = float(np.std(centered))
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
    return centered / float(scale), float(scale)


class WideBase(nn.Module):
    def __init__(self, total_categories, n_num, k):
        super().__init__()
        self.embedding = nn.Embedding(total_categories, k)
        self.linear = nn.Embedding(total_categories, 1)
        self.num_linear = nn.Linear(n_num, 1)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def wide(self, cats, nums):
        return (
            self.linear(cats).sum(dim=1).squeeze(1)
            + self.num_linear(nums).squeeze(1)
        )


class AutoIntModel(WideBase):
    def __init__(self, total_categories, n_fields, n_num, k=12):
        super().__init__(total_categories, n_num, k)
        self.attn1 = nn.MultiheadAttention(
            k, num_heads=3, dropout=0.05, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            k, num_heads=3, dropout=0.05, batch_first=True
        )
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.deep = nn.Sequential(
            nn.Linear(n_fields * k + n_num, 96),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(96, 1),
        )

    def forward(self, cats, nums, seq=None, seq_mask=None):
        e = self.embedding(cats)
        a, _ = self.attn1(e, e, e, need_weights=False)
        e = self.norm1(e + a)
        a, _ = self.attn2(e, e, e, need_weights=False)
        e = self.norm2(e + a)
        z = torch.cat([e.flatten(start_dim=1), nums], dim=1)
        return self.wide(cats, nums) + self.deep(z).squeeze(1)


class FiBiNETModel(WideBase):
    def __init__(self, total_categories, n_fields, n_num, k=8):
        super().__init__(total_categories, n_num, k)
        self.n_fields = n_fields
        hidden = max(4, n_fields // 3)
        self.squeeze = nn.Sequential(
            nn.Linear(n_fields, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_fields),
            nn.Sigmoid(),
        )
        pi, pj = np.triu_indices(n_fields, k=1)
        self.register_buffer("pair_i", torch.from_numpy(pi.astype(np.int64)))
        self.register_buffer("pair_j", torch.from_numpy(pj.astype(np.int64)))
        pair_dim = len(pi) * k
        self.deep = nn.Sequential(
            nn.Linear(pair_dim + n_fields * k + n_num, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, cats, nums, seq=None, seq_mask=None):
        e = self.embedding(cats)
        field_summary = e.mean(dim=2)
        weights = self.squeeze(field_summary).unsqueeze(2)
        recalibrated = e * weights
        products = (
            recalibrated[:, self.pair_i, :]
            * recalibrated[:, self.pair_j, :]
        )
        z = torch.cat(
            [
                recalibrated.flatten(start_dim=1),
                products.flatten(start_dim=1),
                nums,
            ],
            dim=1,
        )
        return self.wide(cats, nums) + self.deep(z).squeeze(1)


class PNNModel(WideBase):
    def __init__(self, total_categories, n_fields, n_num, k=10):
        super().__init__(total_categories, n_num, k)
        pi, pj = np.triu_indices(n_fields, k=1)
        self.register_buffer("pair_i", torch.from_numpy(pi.astype(np.int64)))
        self.register_buffer("pair_j", torch.from_numpy(pj.astype(np.int64)))
        dim = n_fields * k + len(pi) + n_num
        self.deep = nn.Sequential(
            nn.Linear(dim, 128),
            nn.PReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 48),
            nn.PReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, cats, nums, seq=None, seq_mask=None):
        e = self.embedding(cats)
        inner_products = (
            e[:, self.pair_i, :] * e[:, self.pair_j, :]
        ).sum(dim=2)
        z = torch.cat(
            [e.flatten(start_dim=1), inner_products, nums], dim=1
        )
        return self.wide(cats, nums) + self.deep(z).squeeze(1)


class BSTModel(WideBase):
    def __init__(
        self,
        total_categories,
        n_fields,
        n_num,
        video_offset,
        k=12,
    ):
        super().__init__(total_categories, n_num, k)
        self.video_offset = int(video_offset)
        self.position = nn.Embedding(SEQ_LEN + 1, k)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=k,
            nhead=3,
            dim_feedforward=48,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=1
        )
        self.deep = nn.Sequential(
            nn.Linear(n_fields * k + 2 * k + n_num, 112),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(112, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )

    def forward(self, cats, nums, seq, seq_mask):
        fields = self.embedding(cats)
        target = fields[:, 1:2, :]
        history = self.embedding(seq.long() + self.video_offset)
        tokens = torch.cat([target, history], dim=1)

        positions = torch.arange(
            SEQ_LEN + 1, device=cats.device
        ).unsqueeze(0)
        tokens = tokens + self.position(positions)

        target_valid = torch.ones(
            (cats.shape[0], 1), dtype=torch.bool, device=cats.device
        )
        valid = torch.cat([target_valid, seq_mask], dim=1)
        contextual = self.transformer(
            tokens, src_key_padding_mask=~valid
        )
        target_context = contextual[:, 0, :]

        history_float = seq_mask.float().unsqueeze(2)
        history_mean = (
            history * history_float
        ).sum(dim=1) / (history_float.sum(dim=1) + 1e-6)

        z = torch.cat(
            [
                fields.flatten(start_dim=1),
                target_context,
                history_mean,
                nums,
            ],
            dim=1,
        )
        return self.wide(cats, nums) + self.deep(z).squeeze(1)


def train_model(
    model,
    cats,
    nums,
    labels,
    weights,
    seed,
    seq=None,
    seq_mask=None,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0025, weight_decay=2e-5
    )
    rng = np.random.default_rng(seed)
    epoch_losses = []

    model.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(len(labels))
        loss_sum = 0.0
        weight_sum = 0.0

        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            c = torch.from_numpy(cats[idx])
            x = torch.from_numpy(nums[idx])
            y = torch.from_numpy(labels[idx])
            w = torch.from_numpy(weights[idx])

            optimizer.zero_grad(set_to_none=True)
            if seq is None:
                logits = model(c, x)
            else:
                q = torch.from_numpy(seq[idx])
                qm = torch.from_numpy(seq_mask[idx])
                logits = model(c, x, q, qm)

            row_loss = F.binary_cross_entropy_with_logits(
                logits, y, reduction="none"
            )
            loss = torch.sum(row_loss * w) / torch.sum(w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loss_sum += float(torch.sum(row_loss * w).detach())
            weight_sum += float(torch.sum(w))

        epoch_losses.append(loss_sum / max(weight_sum, 1.0))

    return epoch_losses


@torch.no_grad()
def predict_model(model, cats, nums, seq=None, seq_mask=None):
    model.eval()
    result = np.empty(len(cats), dtype=np.float32)

    for start in range(0, len(cats), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(cats))
        c = torch.from_numpy(cats[start:end])
        x = torch.from_numpy(nums[start:end])

        if seq is None:
            logits = model(c, x)
        else:
            q = torch.from_numpy(seq[start:end])
            qm = torch.from_numpy(seq_mask[start:end])
            logits = model(c, x, q, qm)

        result[start:end] = logits.cpu().numpy().astype(np.float32)

    return result


train = load("train")
valid = load("valid")

train_histories = load_histories("train")
valid_histories = load_histories("valid")

train_cats, cards, offsets = make_cat_matrix(train)
valid_cats, _, _ = make_cat_matrix(valid)

train_num_raw = raw_numeric_matrix(train, train_histories)
valid_num_raw = raw_numeric_matrix(valid, valid_histories)
numeric_transform = fit_numeric_transform(train_num_raw)
train_nums = apply_numeric_transform(train_num_raw, numeric_transform)
valid_nums = apply_numeric_transform(valid_num_raw, numeric_transform)

del train_num_raw, valid_num_raw, train_histories, valid_histories
gc.collect()

labels = np.asarray(train.y, dtype=np.float32)
dates = np.asarray(train.date, dtype=np.int32)
date_values = np.unique(dates)
date_rank = {int(d): i for i, d in enumerate(date_values)}
age = np.array(
    [len(date_values) - 1 - date_rank[int(d)] for d in dates],
    dtype=np.float32,
)
weights = np.exp2(-age / HALF_LIFE_DAYS).astype(np.float32)
weights /= weights.mean()

train_seq, train_seq_mask, final_seq, final_seq_mask = (
    build_positive_histories(train)
)
valid_seq, valid_seq_mask = histories_for_split(
    valid, final_seq, final_seq_mask
)

total_categories = int(sum(cards))
n_fields = len(CAT_FIELDS)
n_num = train_nums.shape[1]
video_offset = int(offsets[CAT_FIELDS.index("video_id")])

model_specs = [
    (
        "autoint",
        lambda: AutoIntModel(total_categories, n_fields, n_num),
        False,
    ),
    (
        "fibinet",
        lambda: FiBiNETModel(total_categories, n_fields, n_num),
        False,
    ),
    (
        "pnn",
        lambda: PNNModel(total_categories, n_fields, n_num),
        False,
    ),
    (
        "bst",
        lambda: BSTModel(
            total_categories,
            n_fields,
            n_num,
            video_offset,
        ),
        True,
    ),
]

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_valid_z, inc_scale = centered_scaled(
    inc_valid, valid.user_id, scale=None
)

models = {}
valid_predictions = {}
candidate_scores = {}
candidate_details = {}
alphas = [0.25, 0.50, 0.75, 0.90]

best_primary = -np.inf
best_name = None
best_family = None
best_alpha = 1.0
best_valid_scores = None

for model_index, (name, constructor, uses_sequence) in enumerate(model_specs):
    model = constructor()
    if uses_sequence:
        losses = train_model(
            model,
            train_cats,
            train_nums,
            labels,
            weights,
            SEED + 101 * model_index,
            train_seq,
            train_seq_mask,
        )
        pred = predict_model(
            model,
            valid_cats,
            valid_nums,
            valid_seq,
            valid_seq_mask,
        )
    else:
        losses = train_model(
            model,
            train_cats,
            train_nums,
            labels,
            weights,
            SEED + 101 * model_index,
        )
        pred = predict_model(model, valid_cats, valid_nums)

    pred = np.asarray(pred, dtype=np.float64)
    valid_predictions[name] = pred
    models[name] = model

    standalone_metrics = evaluate(valid.user_id, valid.y, pred)
    candidate_scores[name] = float(standalone_metrics["primary"])
    candidate_details[name] = {
        "alpha": 1.0,
        "primary": float(standalone_metrics["primary"]),
    }

    if standalone_metrics["primary"] > best_primary:
        best_primary = float(standalone_metrics["primary"])
        best_name = name
        best_family = name
        best_alpha = 1.0
        best_valid_scores = pred.copy()

    pred_z, own_scale = centered_scaled(pred, valid.user_id, scale=None)
    candidate_details[name]["scale"] = own_scale

    family_best_blend = -np.inf
    family_best_alpha = None
    for alpha in alphas:
        blend = alpha * pred_z + (1.0 - alpha) * inc_valid_z
        blend_metrics = evaluate(valid.user_id, valid.y, blend)
        primary = float(blend_metrics["primary"])
        key = "{}_blend_{:.2f}".format(name, alpha)
        candidate_scores[key] = primary

        if primary > family_best_blend:
            family_best_blend = primary
            family_best_alpha = alpha

        if primary > best_primary:
            best_primary = primary
            best_name = key
            best_family = name
            best_alpha = alpha
            best_valid_scores = blend.copy()

    print(
        "FINDINGS "
        + json.dumps(
            {
                "family": name,
                "losses": [round(x, 6) for x in losses],
                "standalone": round(
                    float(standalone_metrics["primary"]), 6
                ),
                "best_blend": round(float(family_best_blend), 6),
                "best_blend_alpha": family_best_alpha,
            },
            sort_keys=True,
        )
    )

print(
    "CANDIDATES "
    + json.dumps(
        {k: round(v, 7) for k, v in candidate_scores.items()},
        sort_keys=True,
    )
)

winner_model = models[best_family]
winner_scale = float(candidate_details[best_family]["scale"])

te = load("test")
test_histories = load_histories("test")
test_cats, _, _ = make_cat_matrix(te)
test_num_raw = raw_numeric_matrix(te, test_histories)
test_nums = apply_numeric_transform(test_num_raw, numeric_transform)
test_seq, test_seq_mask = histories_for_split(
    te, final_seq, final_seq_mask
)

if best_family == "bst":
    own_test_scores = predict_model(
        winner_model,
        test_cats,
        test_nums,
        test_seq,
        test_seq_mask,
    ).astype(np.float64)
else:
    own_test_scores = predict_model(
        winner_model, test_cats, test_nums
    ).astype(np.float64)

own_valid_scores = valid_predictions[best_family]

if best_alpha < 1.0:
    inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
    own_test_z, _ = centered_scaled(
        own_test_scores, te.user_id, scale=winner_scale
    )
    inc_test_z, _ = centered_scaled(
        inc_test, te.user_id, scale=inc_scale
    )
    test_scores = (
        best_alpha * own_test_z + (1.0 - best_alpha) * inc_test_z
    )
else:
    test_scores = own_test_scores
    best_valid_scores = own_valid_scores

final_metrics = evaluate(valid.user_id, valid.y, best_valid_scores)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if best_alpha < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(own_valid_scores, dtype=np.float64),
        )

print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": best_name,
            "family": best_family,
            "own_weight": best_alpha,
            "recency_half_life_days": HALF_LIFE_DAYS,
        },
        sort_keys=True,
    )
)

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(final_metrics["primary"]),
            "gauc": float(final_metrics["gauc"]),
            "ndcg@5": float(final_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)