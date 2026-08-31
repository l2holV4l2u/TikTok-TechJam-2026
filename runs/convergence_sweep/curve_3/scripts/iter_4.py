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
SEED = 7319
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
BATCH_SIZE = 6144
EPOCHS = 3


def load_histories(split_name):
    return {
        "video_id": historical_features(split_name, key="video_id"),
        "author_id": historical_features(split_name, key="author_id"),
    }


def make_cat_matrix(split):
    cards = [int(FEATURE_CARDINALITIES[name]) for name in CAT_FIELDS]
    offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
    out = np.empty((len(split.user_id), len(CAT_FIELDS)), dtype=np.int64)
    for j, name in enumerate(CAT_FIELDS):
        out[:, j] = np.asarray(split.X[name], dtype=np.int64) + offsets[j]
    return np.ascontiguousarray(out), cards, offsets


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


def center_scale(scores, user_ids, scale=None):
    scores = np.asarray(scores, dtype=np.float64)
    _, inv = np.unique(np.asarray(user_ids), return_inverse=True)
    sums = np.bincount(inv, weights=scores)
    counts = np.bincount(inv)
    centered = scores - (sums / np.maximum(counts, 1))[inv]
    if scale is None:
        scale = float(np.std(centered))
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
    return centered / float(scale), float(scale)


def build_positive_histories(train):
    """
    Construct a causal sequence for each training row and the final train-only
    positive history for every user. Sorting by (user, time_ms, row position)
    is the benchmark-defined interaction order.
    """
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
    global_before = np.cumsum(positive, dtype=np.int64) - positive.astype(np.int64)

    starts = np.empty(len(order), dtype=bool)
    starts[0] = True
    starts[1:] = su[1:] != su[:-1]
    start_idx = np.maximum.accumulate(
        np.where(starts, np.arange(len(order), dtype=np.int64), 0)
    )
    base_positive = global_before[start_idx]
    prior_count = global_before - base_positive

    positive_videos = sv[positive]
    seq_sorted = np.zeros((len(order), SEQ_LEN), dtype=np.int32)
    mask_sorted = np.zeros((len(order), SEQ_LEN), dtype=np.bool_)

    for lag in range(1, SEQ_LEN + 1):
        valid = prior_count >= lag
        positive_index = base_positive[valid] + prior_count[valid] - lag
        seq_sorted[valid, lag - 1] = positive_videos[positive_index].astype(np.int32)
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
        idx = ends[valid_users] - lag
        final_seq[user_axis[valid_users], lag - 1] = (
            positive_videos[idx].astype(np.int32)
        )
        final_mask[user_axis[valid_users], lag - 1] = True

    return seq_train, mask_train, final_seq, final_mask


def histories_for_split(split, final_seq, final_mask):
    users = np.asarray(split.user_id, dtype=np.int64)
    return (
        np.ascontiguousarray(final_seq[users]),
        np.ascontiguousarray(final_mask[users]),
    )


class XDeepFM(nn.Module):
    def __init__(self, total_categories, n_fields, n_num, k=8):
        super().__init__()
        self.n_fields = n_fields
        self.embedding = nn.Embedding(total_categories, k)
        self.linear = nn.Embedding(total_categories, 1)
        self.num_linear = nn.Linear(n_num, 1)

        self.cin1 = nn.Conv1d(n_fields * n_fields, 16, kernel_size=1)
        self.cin2 = nn.Conv1d(n_fields * 16, 16, kernel_size=1)

        deep_dim = n_fields * k + n_num
        self.deep = nn.Sequential(
            nn.Linear(deep_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.output = nn.Linear(32, 1)

        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def forward(self, cats, nums):
        x0 = self.embedding(cats)
        wide = self.linear(cats).sum(dim=1).squeeze(1)
        wide = wide + self.num_linear(nums).squeeze(1)

        z1 = torch.einsum("bfk,bhk->bfhk", x0, x0)
        z1 = z1.reshape(
            z1.shape[0], self.n_fields * self.n_fields, z1.shape[-1]
        )
        x1 = F.relu(self.cin1(z1))

        z2 = torch.einsum("bfk,bhk->bfhk", x0, x1)
        z2 = z2.reshape(
            z2.shape[0], self.n_fields * 16, z2.shape[-1]
        )
        x2 = F.relu(self.cin2(z2))

        cin_features = torch.cat(
            [x1.sum(dim=2), x2.sum(dim=2)], dim=1
        )
        cin_logit = self.output(cin_features).squeeze(1)

        deep_logit = self.deep(
            torch.cat([x0.flatten(start_dim=1), nums], dim=1)
        ).squeeze(1)
        return wide + cin_logit + deep_logit


class DIN(nn.Module):
    def __init__(
        self,
        total_categories,
        n_fields,
        n_num,
        video_offset,
        k=12,
    ):
        super().__init__()
        self.video_offset = int(video_offset)
        self.embedding = nn.Embedding(total_categories, k)
        self.linear = nn.Embedding(total_categories, 1)
        self.num_linear = nn.Linear(n_num, 1)

        self.attention = nn.Sequential(
            nn.Linear(4 * k, 48),
            nn.PReLU(),
            nn.Linear(48, 16),
            nn.PReLU(),
            nn.Linear(16, 1),
        )

        input_dim = n_fields * k + k + n_num
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.PReLU(),
            nn.Dropout(0.10),
            nn.Linear(96, 48),
            nn.PReLU(),
            nn.Linear(48, 1),
        )

        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def forward(self, cats, nums, seq_video, seq_mask):
        fields = self.embedding(cats)
        target = fields[:, 1, :]
        seq_global = seq_video.long() + self.video_offset
        history = self.embedding(seq_global)

        target_expanded = target.unsqueeze(1).expand_as(history)
        att_input = torch.cat(
            [
                target_expanded,
                history,
                target_expanded - history,
                target_expanded * history,
            ],
            dim=2,
        )
        att_logits = self.attention(att_input).squeeze(2)
        att_logits = att_logits.masked_fill(~seq_mask, -1e4)
        att_weight = torch.softmax(att_logits, dim=1)
        att_weight = att_weight * seq_mask.float()
        att_weight = att_weight / (
            att_weight.sum(dim=1, keepdim=True) + 1e-8
        )
        interest = torch.sum(history * att_weight.unsqueeze(2), dim=1)

        wide = self.linear(cats).sum(dim=1).squeeze(1)
        wide = wide + self.num_linear(nums).squeeze(1)
        deep_input = torch.cat(
            [fields.flatten(start_dim=1), interest, nums], dim=1
        )
        return wide + self.deep(deep_input).squeeze(1)


class MMoE(nn.Module):
    def __init__(self, total_categories, n_fields, n_num, k=8):
        super().__init__()
        self.embedding = nn.Embedding(total_categories, k)
        dim = n_fields * k + n_num
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            for _ in range(4)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(dim, 4) for _ in range(3)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(3)
        ])
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, cats, nums):
        embedded = self.embedding(cats).flatten(start_dim=1)
        x = torch.cat([embedded, nums], dim=1)
        experts = torch.stack([expert(x) for expert in self.experts], dim=1)

        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            weights = torch.softmax(gate(x), dim=1)
            mixed = torch.sum(experts * weights.unsqueeze(2), dim=1)
            outputs.append(tower(mixed).squeeze(1))
        return outputs


def train_single(model, cats, nums, labels, weights, seed, seq=None, mask=None):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0025, weight_decay=2e-5
    )
    rng = np.random.default_rng(seed)
    losses = []

    model.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(len(labels))
        total = 0.0
        total_weight = 0.0

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
                qm = torch.from_numpy(mask[idx])
                logits = model(c, x, q, qm)

            row_loss = F.binary_cross_entropy_with_logits(
                logits, y, reduction="none"
            )
            loss = torch.sum(row_loss * w) / torch.sum(w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total += float(torch.sum(row_loss * w).detach())
            total_weight += float(torch.sum(w))

        losses.append(total / max(total_weight, 1.0))

    return losses


def train_mmoe(model, cats, nums, targets, weights, seed):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0025, weight_decay=2e-5
    )
    rng = np.random.default_rng(seed)
    losses = []
    task_coefficients = [1.0, 0.25, 0.10]

    model.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(len(weights))
        total = 0.0
        total_weight = 0.0

        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            c = torch.from_numpy(cats[idx])
            x = torch.from_numpy(nums[idx])
            w = torch.from_numpy(weights[idx])

            optimizer.zero_grad(set_to_none=True)
            outputs = model(c, x)
            loss = 0.0
            logged_long_loss = None

            for task_index, (logits, target, coefficient) in enumerate(
                zip(outputs, targets, task_coefficients)
            ):
                y = torch.from_numpy(target[idx])
                row_loss = F.binary_cross_entropy_with_logits(
                    logits, y, reduction="none"
                )
                weighted = torch.sum(row_loss * w) / torch.sum(w)
                loss = loss + coefficient * weighted
                if task_index == 0:
                    logged_long_loss = row_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total += float(torch.sum(logged_long_loss * w).detach())
            total_weight += float(torch.sum(w))

        losses.append(total / max(total_weight, 1.0))

    return losses


@torch.no_grad()
def predict_single(model, cats, nums, seq=None, mask=None):
    model.eval()
    predictions = np.empty(len(cats), dtype=np.float64)
    prediction_batch = 12288

    for start in range(0, len(cats), prediction_batch):
        stop = min(start + prediction_batch, len(cats))
        c = torch.from_numpy(cats[start:stop])
        x = torch.from_numpy(nums[start:stop])
        if seq is None:
            logits = model(c, x)
        else:
            q = torch.from_numpy(seq[start:stop])
            qm = torch.from_numpy(mask[start:stop])
            logits = model(c, x, q, qm)
        predictions[start:stop] = logits.cpu().numpy()

    return predictions


@torch.no_grad()
def predict_mmoe(model, cats, nums):
    model.eval()
    predictions = np.empty(len(cats), dtype=np.float64)
    prediction_batch = 12288

    for start in range(0, len(cats), prediction_batch):
        stop = min(start + prediction_batch, len(cats))
        c = torch.from_numpy(cats[start:stop])
        x = torch.from_numpy(nums[start:stop])
        predictions[start:stop] = model(c, x)[0].cpu().numpy()

    return predictions


def primary(split, labels, scores):
    return float(evaluate(split.user_id, labels, scores)["primary"])


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

hist_train = load_histories("train")
hist_valid = load_histories("valid")

cat_train, cards, offsets = make_cat_matrix(train)
cat_valid, _, _ = make_cat_matrix(valid)

num_train_raw = raw_numeric_matrix(train, hist_train)
num_valid_raw = raw_numeric_matrix(valid, hist_valid)
num_transform = fit_numeric_transform(num_train_raw)
num_train = apply_numeric_transform(num_train_raw, num_transform)
num_valid = apply_numeric_transform(num_valid_raw, num_transform)

del num_train_raw, num_valid_raw, hist_train, hist_valid
gc.collect()

seq_train, mask_train, final_seq, final_mask = build_positive_histories(train)
seq_valid, mask_valid = histories_for_split(valid, final_seq, final_mask)

last_date = int(np.max(np.asarray(train.date)))
days_old = last_date - np.asarray(train.date, dtype=np.int64)
train_weights = np.exp2(-days_old / 5.0).astype(np.float32)
train_weights /= np.mean(train_weights)

total_categories = int(sum(cards))
n_fields = len(CAT_FIELDS)
n_num = num_train.shape[1]
video_offset = int(offsets[CAT_FIELDS.index("video_id")])

xdeepfm = XDeepFM(total_categories, n_fields, n_num, k=8)
xdeepfm_losses = train_single(
    xdeepfm,
    cat_train,
    num_train,
    y_train,
    train_weights,
    SEED + 1,
)
xdeepfm_valid = predict_single(xdeepfm, cat_valid, num_valid)

din = DIN(
    total_categories,
    n_fields,
    n_num,
    video_offset=video_offset,
    k=12,
)
din_losses = train_single(
    din,
    cat_train,
    num_train,
    y_train,
    train_weights,
    SEED + 2,
    seq=seq_train,
    mask=mask_train,
)
din_valid = predict_single(
    din,
    cat_valid,
    num_valid,
    seq=seq_valid,
    mask=mask_valid,
)

click_target = np.asarray(train.aux["is_click"], dtype=np.float32)
like_target = np.asarray(train.aux["is_like"], dtype=np.float32)
click_target = np.nan_to_num(click_target, nan=0.0)
like_target = np.nan_to_num(like_target, nan=0.0)
click_target = np.clip(click_target, 0.0, 1.0)
like_target = np.clip(like_target, 0.0, 1.0)

mmoe = MMoE(total_categories, n_fields, n_num, k=8)
mmoe_losses = train_mmoe(
    mmoe,
    cat_train,
    num_train,
    [y_train, click_target, like_target],
    train_weights,
    SEED + 3,
)
mmoe_valid = predict_mmoe(mmoe, cat_valid, num_valid)

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)

families = {
    "xdeepfm": xdeepfm_valid,
    "din_positive_history": din_valid,
    "mmoe_feedback_regularized": mmoe_valid,
}

candidate_scores = {
    "trusted_incumbent": primary(valid, y_valid, inc_valid)
}

inc_norm, inc_scale = center_scale(inc_valid, valid.user_id)
alpha_grid = [0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90]

best_name = "trusted_incumbent"
best_primary = candidate_scores["trusted_incumbent"]
best_scores = inc_valid.copy()
best_family = None
best_alpha = None
best_is_blend = False

best_own_name = None
best_own_primary = -np.inf
best_own_scores = None

family_scales = {}
blend_alphas = {}

for name, scores in families.items():
    raw_primary = primary(valid, y_valid, scores)
    candidate_scores[name] = raw_primary

    if raw_primary > best_own_primary:
        best_own_primary = raw_primary
        best_own_name = name
        best_own_scores = scores.copy()

    if raw_primary > best_primary:
        best_name = name
        best_primary = raw_primary
        best_scores = scores.copy()
        best_family = name
        best_alpha = None
        best_is_blend = False

    normalized, scale = center_scale(scores, valid.user_id)
    family_scales[name] = scale

    local_primary = -np.inf
    local_alpha = None
    local_scores = None

    for alpha in alpha_grid:
        blend = alpha * normalized + (1.0 - alpha) * inc_norm
        blend_primary = primary(valid, y_valid, blend)
        if blend_primary > local_primary:
            local_primary = blend_primary
            local_alpha = alpha
            local_scores = blend.copy()

    blend_name = name + "_incumbent_blend"
    candidate_scores[blend_name] = local_primary
    blend_alphas[name] = local_alpha

    if local_primary > best_primary:
        best_name = blend_name
        best_primary = local_primary
        best_scores = local_scores
        best_family = name
        best_alpha = local_alpha
        best_is_blend = True

metrics = evaluate(valid.user_id, y_valid, best_scores)

history_lengths = mask_valid.sum(axis=1)
print(
    "FINDINGS valid_positive_history_nonempty=%.4f mean_length=%.3f"
    % (
        float(np.mean(history_lengths > 0)),
        float(np.mean(history_lengths)),
    )
)
print(
    "FINDINGS losses="
    + json.dumps(
        {
            "xdeepfm": [float(x) for x in xdeepfm_losses],
            "din_positive_history": [float(x) for x in din_losses],
            "mmoe_feedback_regularized": [float(x) for x in mmoe_losses],
        },
        sort_keys=True,
    )
)
print(
    "FINDINGS winner=%s best_own=%s blend_alphas=%s"
    % (
        best_name,
        best_own_name,
        json.dumps(
            {k: float(v) for k, v in blend_alphas.items()},
            sort_keys=True,
        ),
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_scores.items()},
        sort_keys=True,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_is_blend or best_family is None:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(best_own_scores, dtype=np.float64),
        )

# Model fitting and all validation-based selection are complete before test load.
test = load("test")
hist_test = load_histories("test")
cat_test, _, _ = make_cat_matrix(test)
num_test_raw = raw_numeric_matrix(test, hist_test)
num_test = apply_numeric_transform(num_test_raw, num_transform)
seq_test, mask_test = histories_for_split(test, final_seq, final_mask)

del num_test_raw, hist_test
gc.collect()

inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

if best_family is None:
    test_scores = inc_test
else:
    if best_family == "xdeepfm":
        raw_test = predict_single(xdeepfm, cat_test, num_test)
    elif best_family == "din_positive_history":
        raw_test = predict_single(
            din,
            cat_test,
            num_test,
            seq=seq_test,
            mask=mask_test,
        )
    elif best_family == "mmoe_feedback_regularized":
        raw_test = predict_mmoe(mmoe, cat_test, num_test)
    else:
        raise RuntimeError("Unknown selected family: " + str(best_family))

    if best_is_blend:
        own_test_norm, _ = center_scale(
            raw_test,
            test.user_id,
            scale=family_scales[best_family],
        )
        inc_test_norm, _ = center_scale(
            inc_test,
            test.user_id,
            scale=inc_scale,
        )
        test_scores = (
            float(best_alpha) * own_test_norm
            + (1.0 - float(best_alpha)) * inc_test_norm
        )
    else:
        test_scores = raw_test

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)