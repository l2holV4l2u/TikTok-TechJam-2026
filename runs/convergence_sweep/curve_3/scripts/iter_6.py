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
SEED = 27183
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


def make_cat_matrix(split):
    cards = [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS]
    offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
    x = np.empty((len(split.user_id), len(CAT_FIELDS)), dtype=np.int64)
    for j, field in enumerate(CAT_FIELDS):
        x[:, j] = np.asarray(split.X[field], dtype=np.int64) + offsets[j]
    return np.ascontiguousarray(x), cards, offsets


def load_histories(split_name):
    return {
        "video_id": historical_features(split_name, key="video_id"),
        "author_id": historical_features(split_name, key="author_id"),
    }


def raw_numeric_matrix(split, histories):
    cols = []
    for field in NUM_FIELDS:
        x = np.asarray(split.num[field], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))

    for entity in ("video_id", "author_id"):
        for suffix in HISTORY_SUFFIXES:
            key = entity + "_" + suffix
            x = np.asarray(histories[entity][key], dtype=np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            cols.append(x)

    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)


def fit_numeric_transform(x):
    lo = np.quantile(x, 0.002, axis=0).astype(np.float32)
    hi = np.quantile(x, 0.998, axis=0).astype(np.float32)
    z = np.clip(x, lo, hi)
    mean = z.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = z.std(axis=0, dtype=np.float64).astype(np.float32)
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

    starts = np.empty(len(order), dtype=np.bool_)
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

    # Chronological layout: oldest retained item first, newest last.
    for position in range(SEQ_LEN):
        lag = SEQ_LEN - position
        valid = prior_count >= lag
        indices = user_positive_base[valid] + prior_count[valid] - lag
        seq_sorted[valid, position] = positive_videos[indices].astype(np.int32)
        mask_sorted[valid, position] = True

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

    for position in range(SEQ_LEN):
        lag = SEQ_LEN - position
        valid = user_counts >= lag
        indices = ends[valid] - lag
        final_seq[user_axis[valid], position] = (
            positive_videos[indices].astype(np.int32)
        )
        final_mask[user_axis[valid], position] = True

    return seq_train, mask_train, final_seq, final_mask


def histories_for_split(split, final_seq, final_mask):
    users = np.asarray(split.user_id, dtype=np.int64)
    return (
        np.ascontiguousarray(final_seq[users]),
        np.ascontiguousarray(final_mask[users]),
    )


def training_weights(train, half_life, user_balance_power=0.0):
    dates = np.asarray(train.date, dtype=np.int64)
    delta = dates.max() - dates
    weights = np.power(0.5, delta.astype(np.float32) / float(half_life))

    if user_balance_power > 0:
        users = np.asarray(train.user_id, dtype=np.int64)
        counts = np.bincount(
            users,
            minlength=int(FEATURE_CARDINALITIES["user_id"]),
        ).astype(np.float32)
        weights *= np.power(np.maximum(counts[users], 1.0), -user_balance_power)

    weights /= max(float(weights.mean()), 1e-8)
    return np.ascontiguousarray(weights, dtype=np.float32)


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


class BaseCTR(nn.Module):
    def __init__(self, total_categories, n_num, k):
        super().__init__()
        self.embedding = nn.Embedding(total_categories, k)
        self.linear = nn.Embedding(total_categories, 1)
        self.num_linear = nn.Linear(n_num, 1)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.num_linear.bias)

    def wide(self, cats, nums):
        return (
            self.linear(cats).sum(dim=1).squeeze(1)
            + self.num_linear(nums).squeeze(1)
        )


class FieldWeightedFM(BaseCTR):
    def __init__(self, total_categories, n_fields, n_num, k=12):
        super().__init__(total_categories, n_num, k)
        pi, pj = np.triu_indices(n_fields, k=1)
        self.register_buffer("pair_i", torch.from_numpy(pi.astype(np.int64)))
        self.register_buffer("pair_j", torch.from_numpy(pj.astype(np.int64)))
        self.pair_weights = nn.Parameter(torch.ones(len(pi)))
        self.num_head = nn.Sequential(
            nn.Linear(n_num, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(self, cats, nums, seq=None, seq_mask=None):
        e = self.embedding(cats)
        pair_dot = (
            e[:, self.pair_i, :] * e[:, self.pair_j, :]
        ).sum(dim=2)
        interaction = (pair_dot * self.pair_weights).sum(dim=1)
        return (
            self.wide(cats, nums)
            + interaction
            + self.num_head(nums).squeeze(1)
        )


class DeepFM(BaseCTR):
    def __init__(self, total_categories, n_fields, n_num, k=12):
        super().__init__(total_categories, n_num, k)
        self.deep = nn.Sequential(
            nn.Linear(n_fields * k + n_num, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, cats, nums, seq=None, seq_mask=None):
        e = self.embedding(cats)
        summed = e.sum(dim=1)
        fm = 0.5 * (
            summed.square() - e.square().sum(dim=1)
        ).sum(dim=1)
        deep = self.deep(
            torch.cat([e.flatten(start_dim=1), nums], dim=1)
        ).squeeze(1)
        return self.wide(cats, nums) + fm + deep


class WideAndDeep(BaseCTR):
    def __init__(self, total_categories, n_fields, n_num, k=10):
        super().__init__(total_categories, n_num, k)
        self.deep = nn.Sequential(
            nn.Linear(n_fields * k + n_num, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, cats, nums, seq=None, seq_mask=None):
        e = self.embedding(cats)
        z = torch.cat([e.flatten(start_dim=1), nums], dim=1)
        return self.wide(cats, nums) + self.deep(z).squeeze(1)


class RecurrentInterestModel(BaseCTR):
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
        self.gru_cell = nn.GRUCell(k, k)
        self.deep = nn.Sequential(
            nn.Linear(n_fields * k + 3 * k + n_num + 1, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )

    def forward(self, cats, nums, seq, seq_mask):
        fields = self.embedding(cats)
        target = fields[:, 1, :]
        history = self.embedding(seq.long() + self.video_offset)

        hidden = torch.zeros_like(target)
        for t in range(SEQ_LEN):
            proposed = self.gru_cell(history[:, t, :], hidden)
            valid = seq_mask[:, t].unsqueeze(1)
            hidden = torch.where(valid, proposed, hidden)

        mask_float = seq_mask.float().unsqueeze(2)
        history_mean = (
            (history * mask_float).sum(dim=1)
            / (mask_float.sum(dim=1) + 1e-6)
        )
        target_match = (target * hidden).sum(dim=1, keepdim=True)

        z = torch.cat(
            [
                fields.flatten(start_dim=1),
                target,
                hidden,
                history_mean,
                nums,
                target_match,
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
        model.parameters(),
        lr=0.0025,
        weight_decay=2e-5,
    )
    rng = np.random.default_rng(seed)
    losses = []

    model.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(len(labels))
        weighted_loss_sum = 0.0
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

            weighted_loss_sum += float(torch.sum(row_loss * w).detach())
            weight_sum += float(torch.sum(w))

        losses.append(weighted_loss_sum / max(weight_sum, 1.0))

    return losses


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

        result[start:end] = logits.detach().cpu().numpy()

    return result


train = load("train")
valid = load("valid")
test = load("test")

train_cats, cards, offsets = make_cat_matrix(train)
valid_cats, _, _ = make_cat_matrix(valid)
test_cats, _, _ = make_cat_matrix(test)

train_hist = load_histories("train")
valid_hist = load_histories("valid")
test_hist = load_histories("test")

train_num_raw = raw_numeric_matrix(train, train_hist)
valid_num_raw = raw_numeric_matrix(valid, valid_hist)
test_num_raw = raw_numeric_matrix(test, test_hist)

num_transform = fit_numeric_transform(train_num_raw)
train_num = apply_numeric_transform(train_num_raw, num_transform)
valid_num = apply_numeric_transform(valid_num_raw, num_transform)
test_num = apply_numeric_transform(test_num_raw, num_transform)

del train_num_raw, valid_num_raw, test_num_raw
del train_hist, valid_hist, test_hist
gc.collect()

train_seq, train_seq_mask, final_seq, final_seq_mask = (
    build_positive_histories(train)
)
valid_seq, valid_seq_mask = histories_for_split(
    valid, final_seq, final_seq_mask
)
test_seq, test_seq_mask = histories_for_split(
    test, final_seq, final_seq_mask
)

labels = np.asarray(train.y, dtype=np.float32)
valid_labels = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)

total_categories = int(sum(cards))
n_fields = len(CAT_FIELDS)
n_num = train_num.shape[1]
video_offset = int(offsets[CAT_FIELDS.index("video_id")])

weight_schemes = {
    "short_recency": training_weights(train, half_life=3.0),
    "medium_recency": training_weights(train, half_life=7.0),
    "long_recency": training_weights(train, half_life=14.0),
    "user_balanced": training_weights(
        train, half_life=7.0, user_balance_power=0.5
    ),
}

model_specs = [
    (
        "field_weighted_fm_hl3",
        lambda: FieldWeightedFM(
            total_categories, n_fields, n_num, k=12
        ),
        "short_recency",
        False,
    ),
    (
        "deepfm_hl14",
        lambda: DeepFM(
            total_categories, n_fields, n_num, k=12
        ),
        "long_recency",
        False,
    ),
    (
        "wide_deep_user_balanced",
        lambda: WideAndDeep(
            total_categories, n_fields, n_num, k=10
        ),
        "user_balanced",
        False,
    ),
    (
        "recurrent_interest_hl7",
        lambda: RecurrentInterestModel(
            total_categories,
            n_fields,
            n_num,
            video_offset,
            k=12,
        ),
        "medium_recency",
        True,
    ),
]

valid_predictions = {}
test_predictions = {}
loss_report = {}

for model_index, (name, constructor, weight_name, uses_sequence) in enumerate(
    model_specs
):
    torch.manual_seed(SEED + 101 * model_index)
    model = constructor()

    losses = train_model(
        model,
        train_cats,
        train_num,
        labels,
        weight_schemes[weight_name],
        seed=SEED + 1009 * model_index,
        seq=train_seq if uses_sequence else None,
        seq_mask=train_seq_mask if uses_sequence else None,
    )
    loss_report[name] = [round(float(v), 6) for v in losses]

    valid_predictions[name] = predict_model(
        model,
        valid_cats,
        valid_num,
        seq=valid_seq if uses_sequence else None,
        seq_mask=valid_seq_mask if uses_sequence else None,
    ).astype(np.float64)

    test_predictions[name] = predict_model(
        model,
        test_cats,
        test_num,
        seq=test_seq if uses_sequence else None,
        seq_mask=test_seq_mask if uses_sequence else None,
    ).astype(np.float64)

    del model
    gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

inc_valid_z, _ = centered_scaled(inc_valid, valid_users)
inc_test_z, _ = centered_scaled(inc_test, test_users)

candidate_scores = {}
candidate_valid_arrays = {}
candidate_test_arrays = {}
candidate_raw_arrays = {}

for name in valid_predictions:
    raw_valid = valid_predictions[name]
    raw_test = test_predictions[name]
    raw_metrics = evaluate(valid_users, valid_labels, raw_valid)
    candidate_scores[name] = float(raw_metrics["primary"])
    candidate_valid_arrays[name] = raw_valid
    candidate_test_arrays[name] = raw_test
    candidate_raw_arrays[name] = None

    valid_z, _ = centered_scaled(raw_valid, valid_users)
    test_z, _ = centered_scaled(raw_test, test_users)

    for alpha in (0.25, 0.50, 0.75):
        blend_name = name + "_inc" + str(int(100 * alpha))
        blend_valid = alpha * valid_z + (1.0 - alpha) * inc_valid_z
        blend_test = alpha * test_z + (1.0 - alpha) * inc_test_z
        metrics = evaluate(valid_users, valid_labels, blend_valid)

        candidate_scores[blend_name] = float(metrics["primary"])
        candidate_valid_arrays[blend_name] = blend_valid
        candidate_test_arrays[blend_name] = blend_test
        candidate_raw_arrays[blend_name] = raw_valid

# Rank-normalized averaging is avoided; z-score averaging preserves more
# confidence information while equalizing incompatible model logit scales.
new_valid_z = []
new_test_z = []
for name in valid_predictions:
    vz, _ = centered_scaled(valid_predictions[name], valid_users)
    tz, _ = centered_scaled(test_predictions[name], test_users)
    new_valid_z.append(vz)
    new_test_z.append(tz)

ensemble_valid = np.mean(np.stack(new_valid_z, axis=0), axis=0)
ensemble_test = np.mean(np.stack(new_test_z, axis=0), axis=0)
ensemble_metrics = evaluate(valid_users, valid_labels, ensemble_valid)

candidate_scores["new_family_ensemble"] = float(
    ensemble_metrics["primary"]
)
candidate_valid_arrays["new_family_ensemble"] = ensemble_valid
candidate_test_arrays["new_family_ensemble"] = ensemble_test
candidate_raw_arrays["new_family_ensemble"] = None

for alpha in (0.25, 0.50, 0.75):
    name = "new_ensemble_inc" + str(int(100 * alpha))
    blend_valid = alpha * ensemble_valid + (1.0 - alpha) * inc_valid_z
    blend_test = alpha * ensemble_test + (1.0 - alpha) * inc_test_z
    metrics = evaluate(valid_users, valid_labels, blend_valid)

    candidate_scores[name] = float(metrics["primary"])
    candidate_valid_arrays[name] = blend_valid
    candidate_test_arrays[name] = blend_test
    candidate_raw_arrays[name] = ensemble_valid

inc_metrics = evaluate(valid_users, valid_labels, inc_valid)
candidate_scores["trusted_incumbent"] = float(inc_metrics["primary"])
candidate_valid_arrays["trusted_incumbent"] = inc_valid
candidate_test_arrays["trusted_incumbent"] = inc_test
candidate_raw_arrays["trusted_incumbent"] = None

winner_name = max(candidate_scores, key=candidate_scores.get)
winner_valid = np.asarray(
    candidate_valid_arrays[winner_name], dtype=np.float64
)
winner_test = np.asarray(
    candidate_test_arrays[winner_name], dtype=np.float64
)
winner_metrics = evaluate(valid_users, valid_labels, winner_valid)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        winner_valid,
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        winner_test,
    )
    if candidate_raw_arrays[winner_name] is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(
                candidate_raw_arrays[winner_name],
                dtype=np.float64,
            ),
        )

print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner_name,
            "training_losses": loss_report,
            "ensemble_primary": float(
                ensemble_metrics["primary"]
            ),
        },
        sort_keys=True,
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        {
            k: round(float(v), 6)
            for k, v in sorted(candidate_scores.items())
        },
        sort_keys=True,
    )
)
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(winner_metrics["primary"]),
            "gauc": float(winner_metrics["gauc"]),
            "ndcg@5": float(winner_metrics["ndcg@5"]),
            "gpu_seconds": float(time.time() - START),
        }
    )
)