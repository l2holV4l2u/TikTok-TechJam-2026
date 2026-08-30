import os
import time
import json
import gc
import math
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 82417
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "duration_bucket", "upload_type", "music_type",
    "hour", "user_active_degree",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
HISTORY_LENGTH = 12
EMBED_DIM = 12
BATCH_SIZE = 8192

FIELD_OFFSETS = {}
cursor = 1
for field in CAT_FIELDS:
    FIELD_OFFSETS[field] = cursor
    cursor += int(FEATURE_CARDINALITIES[field])
TOTAL_TOKENS = cursor
VIDEO_OFFSET = FIELD_OFFSETS["video_id"]
USER_CARD = int(FEATURE_CARDINALITIES["user_id"])


def ordinal_day(dates):
    d = np.asarray(dates, dtype=np.int64)
    month = (d // 100) % 100
    day = d % 100
    return day + (month == 5) * 30


def recency_weights(dates, half_life=6.0):
    day = ordinal_day(dates)
    age = day.max() - day
    w = np.exp2(-age.astype(np.float32) / float(half_life))
    return (w / np.maximum(w.mean(), 1e-8)).astype(np.float32)


def encoded_categories(split):
    return np.column_stack([
        np.asarray(split.X[field], dtype=np.int64)
        + FIELD_OFFSETS[field]
        for field in CAT_FIELDS
    ]).astype(np.int64, copy=False)


def raw_numeric(split):
    cols = []
    for field in NUM_FIELDS:
        x = np.asarray(split.num[field], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(np.log1p(np.maximum(x, 0.0)))
    return np.column_stack(cols).astype(np.float32, copy=False)


def fit_numeric_stats(split):
    x = raw_numeric(split)
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.maximum(std, 0.1)
    return mean, std


def normalized_numeric(split, stats):
    mean, std = stats
    return ((raw_numeric(split) - mean) / std).astype(np.float32)


def ordered_prior_positive_histories(split, labels, history_length):
    """
    For every fitting row, construct only the positive-video history strictly
    before that row in (user_id, time_ms, row_position) order.
    """
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int8)
    n = len(users)
    row_id = np.arange(n, dtype=np.int64)

    order = np.lexsort((row_id, times, users))
    su = users[order]
    sy = (y[order] > 0).astype(np.int64)

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = su[1:] != su[:-1]
    starts = np.flatnonzero(starts_mask)

    cumulative = np.cumsum(sy, dtype=np.int64)
    baseline = np.zeros(len(starts), dtype=np.int64)
    has_previous = starts > 0
    baseline[has_previous] = cumulative[starts[has_previous] - 1]
    group_index = np.cumsum(starts_mask) - 1

    before_sorted = cumulative - sy - baseline[group_index]
    before = np.empty(n, dtype=np.int64)
    before[order] = before_sorted

    total_per_group = np.add.reduceat(sy, starts)
    max_positive = int(total_per_group.max(initial=0))
    positive_table = np.zeros(
        (USER_CARD, max_positive + 1), dtype=np.int32
    )

    positive_rows = order[sy > 0]
    positive_positions = before[positive_rows] + 1
    positive_table[
        users[positive_rows], positive_positions
    ] = videos[positive_rows].astype(np.int32)

    offsets = np.arange(
        history_length - 1, -1, -1, dtype=np.int64
    )
    positions = before[:, None] - offsets[None, :]
    valid = positions >= 1
    safe_positions = np.maximum(positions, 0)
    histories = positive_table[users[:, None], safe_positions]
    histories[~valid] = 0

    return histories.astype(np.int32, copy=False), positive_table


def full_positive_table(split, labels):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int8)
    row_id = np.arange(len(users), dtype=np.int64)

    mask = y > 0
    positive_rows = row_id[mask]
    if len(positive_rows) == 0:
        return np.zeros((USER_CARD, 1), dtype=np.int32)

    order = np.lexsort((
        positive_rows,
        times[mask],
        users[mask],
    ))
    pu = users[mask][order]
    pv = videos[mask][order]

    starts_mask = np.empty(len(pu), dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = pu[1:] != pu[:-1]
    starts = np.flatnonzero(starts_mask)
    group_index = np.cumsum(starts_mask) - 1
    position = np.arange(len(pu)) - starts[group_index] + 1
    max_positive = int(position.max(initial=0))

    table = np.zeros((USER_CARD, max_positive + 1), dtype=np.int32)
    table[pu, position] = pv.astype(np.int32)
    return table


def inference_histories(split, positive_table, history_length):
    users = np.asarray(split.user_id, dtype=np.int64)
    counts = np.count_nonzero(positive_table, axis=1).astype(np.int64)
    offsets = np.arange(
        history_length - 1, -1, -1, dtype=np.int64
    )
    positions = counts[users, None] - offsets[None, :]
    valid = positions >= 1
    safe_positions = np.maximum(positions, 0)
    histories = positive_table[users[:, None], safe_positions]
    histories[~valid] = 0
    return histories.astype(np.int32, copy=False)


def encoded_history_tensor(history):
    h = np.asarray(history, dtype=np.int64)
    out = np.zeros_like(h, dtype=np.int64)
    mask = h > 0
    out[mask] = h[mask] + VIDEO_OFFSET
    return out


class SequenceCTR(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.embedding = nn.Embedding(
            TOTAL_TOKENS, EMBED_DIM, padding_idx=0
        )
        nn.init.normal_(self.embedding.weight, std=0.025)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

        if mode == "gru":
            self.gru = nn.GRU(
                EMBED_DIM, EMBED_DIM, batch_first=True
            )
        else:
            self.attention = nn.Sequential(
                nn.Linear(EMBED_DIM * 4, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        input_dim = (
            len(CAT_FIELDS) * EMBED_DIM
            + EMBED_DIM
            + EMBED_DIM
            + len(NUM_FIELDS)
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, cats, nums, history):
        field_embeddings = self.embedding(cats)
        candidate_video = field_embeddings[:, 1, :]
        history_embeddings = self.embedding(history)
        mask = history.ne(0)

        if self.mode == "din":
            expanded_candidate = candidate_video[:, None, :].expand(
                -1, history_embeddings.shape[1], -1
            )
            interaction = torch.cat([
                history_embeddings,
                expanded_candidate,
                history_embeddings * expanded_candidate,
                history_embeddings - expanded_candidate,
            ], dim=-1)
            logits = self.attention(interaction).squeeze(-1)
            logits = logits.masked_fill(~mask, -1e4)
            attention = torch.softmax(logits, dim=1)
            attention = attention * mask.float()
            attention = attention / attention.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-6)
            context = torch.sum(
                attention[:, :, None] * history_embeddings, dim=1
            )
        else:
            output, _ = self.gru(history_embeddings)
            last_index = mask.long().sum(dim=1).sub(1).clamp_min(0)
            context = output[
                torch.arange(len(output), device=output.device),
                last_index,
            ]
            context = context * mask.any(dim=1, keepdim=True).float()

        product = candidate_video * context
        features = torch.cat([
            field_embeddings.flatten(1),
            context,
            product,
            nums,
        ], dim=1)
        return self.mlp(features).squeeze(1)


def fit_sequence_model(
    split, labels, histories, mode, stats, epochs, seed
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    cats = encoded_categories(split)
    nums = normalized_numeric(split, stats)
    hist = encoded_history_tensor(histories)
    labels = np.asarray(labels, dtype=np.float32)
    weights = recency_weights(split.date, half_life=6.0)

    model = SequenceCTR(mode)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0022, weight_decay=2e-6
    )

    n = len(labels)
    for epoch in range(epochs):
        permutation = rng.permutation(n)
        running_loss = 0.0
        seen = 0
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            tc = torch.from_numpy(cats[idx])
            tn = torch.from_numpy(nums[idx])
            th = torch.from_numpy(hist[idx])
            ty = torch.from_numpy(labels[idx])
            tw = torch.from_numpy(weights[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(tc, tn, th)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, ty, reduction="none"
            )
            loss = (losses * tw).sum() / tw.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += float(loss.detach()) * len(idx)
            seen += len(idx)

        print(
            "FINDINGS %s_epoch=%d weighted_logloss=%.6f"
            % (mode, epoch + 1, running_loss / max(seen, 1))
        )

    del cats, nums, hist
    return model


def predict_sequence_model(model, split, histories, stats):
    cats = encoded_categories(split)
    nums = normalized_numeric(split, stats)
    hist = encoded_history_tensor(histories)
    pred = np.empty(len(cats), dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(cats), BATCH_SIZE * 2):
            end = min(start + BATCH_SIZE * 2, len(cats))
            logits = model(
                torch.from_numpy(cats[start:end]),
                torch.from_numpy(nums[start:end]),
                torch.from_numpy(hist[start:end]),
            )
            pred[start:end] = logits.numpy().astype(np.float64)

    del cats, nums, hist
    return pred


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n), scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    group_index = np.cumsum(starts_mask) - 1
    position = np.arange(n) - starts[group_index]
    sizes = np.diff(np.r_[starts, n])
    denominator = np.maximum(sizes[group_index] - 1, 1)

    sorted_rank = position.astype(np.float64) / denominator
    sorted_rank[sizes[group_index] == 1] = 0.5
    rank = np.empty(n, dtype=np.float64)
    rank[order] = sorted_rank
    return rank


def best_blend(user_ids, labels, own, incumbent):
    own_rank = within_user_rank(user_ids, own)
    incumbent_rank = within_user_rank(user_ids, incumbent)
    best = None
    for alpha in [
        0.0, 0.10, 0.20, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.80, 0.90, 1.0,
    ]:
        scores = alpha * own_rank + (1.0 - alpha) * incumbent_rank
        metrics = evaluate(user_ids, labels, scores)
        candidate = (
            float(metrics["primary"]), float(alpha), scores, metrics
        )
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best


class JoinedSplit:
    pass


def join_splits(a, b):
    out = JoinedSplit()
    out.X = {
        field: np.concatenate([
            np.asarray(a.X[field]), np.asarray(b.X[field])
        ])
        for field in CAT_FIELDS
    }
    out.num = {
        field: np.concatenate([
            np.asarray(a.num[field]), np.asarray(b.num[field])
        ])
        for field in NUM_FIELDS
    }
    out.user_id = np.concatenate([
        np.asarray(a.user_id), np.asarray(b.user_id)
    ])
    out.video_id = np.concatenate([
        np.asarray(a.video_id), np.asarray(b.video_id)
    ])
    out.time_ms = np.concatenate([
        np.asarray(a.time_ms), np.asarray(b.time_ms)
    ])
    out.date = np.concatenate([
        np.asarray(a.date), np.asarray(b.date)
    ])
    return out


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)

stats_train = fit_numeric_stats(train)
train_histories, train_positive_table = (
    ordered_prior_positive_histories(
        train, y_train, HISTORY_LENGTH
    )
)
valid_histories = inference_histories(
    valid, train_positive_table, HISTORY_LENGTH
)

history_nonempty_train = float(
    np.mean(np.any(train_histories > 0, axis=1))
)
history_nonempty_valid = float(
    np.mean(np.any(valid_histories > 0, axis=1))
)
print(
    "FINDINGS history_nonempty_train=%.4f history_nonempty_valid=%.4f"
    % (history_nonempty_train, history_nonempty_valid)
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)

family_specs = [
    ("din", 2, SEED + 1),
    ("gru", 1, SEED + 2),
]
family_predictions = {}
candidate_summary = {}
raw_summary = {}
selection = None

for mode, epochs, seed in family_specs:
    model = fit_sequence_model(
        train, y_train, train_histories,
        mode, stats_train, epochs, seed
    )
    prediction = predict_sequence_model(
        model, valid, valid_histories, stats_train
    )
    family_predictions[mode] = prediction

    raw_metrics = evaluate(valid.user_id, valid.y, prediction)
    raw_summary[mode] = float(raw_metrics["primary"])
    candidate_summary[mode + "_raw"] = float(
        raw_metrics["primary"]
    )

    blend = best_blend(
        valid.user_id, valid.y, prediction, inc_valid
    )
    candidate_summary[mode + "_blend"] = float(blend[0])

    if selection is None or blend[0] > selection["primary"]:
        selection = {
            "name": mode,
            "epochs": epochs,
            "seed": seed,
            "primary": blend[0],
            "alpha": blend[1],
            "scores": blend[2],
            "metrics": blend[3],
            "raw": prediction,
        }

    del model
    gc.collect()

print(
    "FINDINGS raw_sequence_primary="
    + json.dumps(raw_summary, sort_keys=True)
)
print(
    "FINDINGS selected_family=%s own_rank_weight=%.2f"
    % (selection["name"], selection["alpha"])
)
print("CANDIDATES " + json.dumps(
    candidate_summary, sort_keys=True
))

valid_scores = np.asarray(selection["scores"], dtype=np.float64)
metrics = selection["metrics"]

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        valid_scores,
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(selection["raw"], dtype=np.float64),
    )

# Permitted identical-recipe refit on train + validation.
combined = join_splits(train, valid)
combined_y = np.concatenate([y_train, y_valid])
stats_combined = fit_numeric_stats(combined)

del train_histories, valid_histories, train_positive_table
gc.collect()

combined_histories, combined_positive_table = (
    ordered_prior_positive_histories(
        combined, combined_y, HISTORY_LENGTH
    )
)

final_model = fit_sequence_model(
    combined,
    combined_y,
    combined_histories,
    selection["name"],
    stats_combined,
    selection["epochs"],
    selection["seed"],
)

test = load("test")
test_histories = inference_histories(
    test, combined_positive_table, HISTORY_LENGTH
)
own_test = predict_sequence_model(
    final_model, test, test_histories, stats_combined
)

inc_test = np.load(inc_test_path).astype(np.float64)
test_scores = (
    selection["alpha"] * within_user_rank(test.user_id, own_test)
    + (1.0 - selection["alpha"])
    * within_user_rank(test.user_id, inc_test)
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": elapsed,
}))