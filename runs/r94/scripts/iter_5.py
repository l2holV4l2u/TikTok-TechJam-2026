import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7319
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
]
N_FIELDS = len(FIELDS)
CAT_DIM = 8
SEQ_DIM = 16
SEQ_LEN = 10
BATCH_SIZE = 8192
EPOCHS = 2
LR = 0.0015
HALF_LIFE_DAYS = 4.0

OFFSETS = np.zeros(N_FIELDS, dtype=np.int64)
total_cardinality = 0
for j, field in enumerate(FIELDS):
    OFFSETS[j] = total_cardinality
    total_cardinality += int(FEATURE_CARDINALITIES[field])

VIDEO_CARDINALITY = int(FEATURE_CARDINALITIES["video_id"])
USER_CARDINALITY = int(FEATURE_CARDINALITIES["user_id"])


def build_categorical(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[field], dtype=np.int64) + OFFSETS[j]
            for j, field in enumerate(FIELDS)
        ]),
        dtype=np.int64,
    )


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int32)
    latest = int(dates.max())
    w = np.power(
        2.0,
        (dates.astype(np.float64) - latest) / HALF_LIFE_DAYS,
    )
    w /= w.mean()
    return w.astype(np.float32)


def build_positive_history(train):
    """
    Construct each training row's last SEQ_LEN positive videos strictly before
    that row in (user_id, time_ms, row_position) order. Also return a compact
    bank from which evaluation histories using the complete training split can
    be gathered without reading evaluation outcomes.
    """
    n = len(train.user_id)
    row = np.arange(n, dtype=np.int64)
    raw_users = np.asarray(train.user_id, dtype=np.int64)
    user_ids = np.asarray(train.X["user_id"], dtype=np.int64)
    video_ids = np.asarray(train.X["video_id"], dtype=np.int64)
    times = np.asarray(train.time_ms, dtype=np.int64)
    labels = np.asarray(train.y, dtype=np.int8)

    order = np.lexsort((row, times, raw_users))
    sorted_users = user_ids[order]
    sorted_videos = video_ids[order]
    sorted_labels = labels[order].astype(np.int64)

    group_start = np.empty(n, dtype=bool)
    group_start[0] = True
    group_start[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(group_start)
    ends = np.append(starts[1:], n)
    sizes = ends - starts

    global_cumulative = np.cumsum(sorted_labels, dtype=np.int64)
    cumulative_before_group = (
        global_cumulative[starts] - sorted_labels[starts]
    )
    group_base = np.repeat(cumulative_before_group, sizes)
    positive_count_before = (
        global_cumulative - sorted_labels - group_base
    )

    positive_counts = np.bincount(
        user_ids,
        weights=labels.astype(np.int64),
        minlength=USER_CARDINALITY,
    ).astype(np.int64)

    segment_sizes = positive_counts + SEQ_LEN
    bank_starts = np.zeros(USER_CARDINALITY, dtype=np.int64)
    if USER_CARDINALITY > 1:
        bank_starts[1:] = np.cumsum(segment_sizes[:-1], dtype=np.int64)

    bank = np.zeros(int(segment_sizes.sum()), dtype=np.int64)
    positive_mask = sorted_labels == 1
    positive_users = sorted_users[positive_mask]
    positive_ordinals = positive_count_before[positive_mask]
    positive_bank_positions = (
        bank_starts[positive_users] + SEQ_LEN + positive_ordinals
    )
    # Zero is padding; native video ids are shifted by one.
    bank[positive_bank_positions] = sorted_videos[positive_mask] + 1

    # Oldest-to-newest layout. Missing history occupies leading zero slots.
    lags = np.arange(SEQ_LEN, 0, -1, dtype=np.int64)
    gather = (
        bank_starts[sorted_users, None]
        + SEQ_LEN
        + positive_count_before[:, None]
        - lags[None, :]
    )
    sorted_history = bank[gather]
    train_history = np.empty_like(sorted_history)
    train_history[order] = sorted_history

    return (
        np.ascontiguousarray(train_history, dtype=np.int64),
        bank,
        bank_starts,
        positive_counts,
    )


def build_eval_history(split, bank, bank_starts, positive_counts):
    user_ids = np.asarray(split.X["user_id"], dtype=np.int64)
    available = positive_counts[user_ids]
    lags = np.arange(SEQ_LEN, 0, -1, dtype=np.int64)
    gather = (
        bank_starts[user_ids, None]
        + SEQ_LEN
        + available[:, None]
        - lags[None, :]
    )
    return np.ascontiguousarray(bank[gather], dtype=np.int64)


class SequenceScorer(nn.Module):
    def __init__(self, family):
        super().__init__()
        self.family = family
        self.cat_embedding = nn.Embedding(total_cardinality, CAT_DIM)
        self.video_embedding = nn.Embedding(
            VIDEO_CARDINALITY + 1,
            SEQ_DIM,
            padding_idx=0,
        )

        if family == "din":
            self.att_hist = nn.Linear(SEQ_DIM, 32, bias=False)
            self.att_candidate = nn.Linear(SEQ_DIM, 32, bias=True)
            self.att_output = nn.Linear(32, 1, bias=False)
        elif family == "gru":
            self.gru = nn.GRU(
                input_size=SEQ_DIM,
                hidden_size=SEQ_DIM,
                batch_first=True,
            )
        elif family != "mean":
            raise ValueError(f"Unknown family: {family}")

        input_dim = N_FIELDS * CAT_DIM + 2 * SEQ_DIM
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        nn.init.normal_(self.cat_embedding.weight, std=0.02)
        nn.init.normal_(self.video_embedding.weight, std=0.02)
        with torch.no_grad():
            self.video_embedding.weight[0].zero_()

    def encode_history(self, history_ids, candidate):
        history = self.video_embedding(history_ids)
        mask = history_ids.ne(0)

        if self.family == "mean":
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
            interest = (
                history * mask.unsqueeze(-1)
            ).sum(dim=1) / denom

        elif self.family == "din":
            hidden = torch.tanh(
                self.att_hist(history)
                + self.att_candidate(candidate).unsqueeze(1)
            )
            attention_logits = self.att_output(hidden).squeeze(-1)
            attention_logits = attention_logits.masked_fill(
                ~mask, -1.0e4
            )
            attention = torch.softmax(attention_logits, dim=1)
            attention = attention * mask
            attention = attention / attention.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0e-8)
            interest = (history * attention.unsqueeze(-1)).sum(dim=1)

        else:
            # Histories are oldest-to-newest with leading zero padding, so the
            # final recurrent state represents the most recent positive event.
            _, final = self.gru(history)
            interest = final.squeeze(0)
            no_history = ~mask.any(dim=1)
            if no_history.any():
                interest = interest.masked_fill(
                    no_history.unsqueeze(1), 0.0
                )

        return interest

    def forward(self, categorical, history_ids, candidate_video):
        cat = self.cat_embedding(categorical).flatten(1)
        candidate = self.video_embedding(candidate_video)
        interest = self.encode_history(history_ids, candidate)
        features = torch.cat([cat, candidate, interest], dim=1)
        return self.head(features).squeeze(1)


def train_model(
    family,
    categorical,
    history,
    candidate_video,
    labels,
    weights,
    seed,
):
    torch.manual_seed(seed)
    model = SequenceScorer(family)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1.0e-5,
    )
    n = categorical.shape[0]
    generator = torch.Generator()
    generator.manual_seed(seed + 91)

    x_cat = torch.from_numpy(categorical)
    x_hist = torch.from_numpy(history)
    x_video = torch.from_numpy(candidate_video)
    y = torch.from_numpy(labels)
    w = torch.from_numpy(weights)

    model.train()
    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            logits = model(
                x_cat[idx],
                x_hist[idx],
                x_video[idx],
            )
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits,
                y[idx],
                reduction="none",
            )
            loss = (losses * w[idx]).sum() / w[idx].sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def predict_model(model, categorical, history, candidate_video):
    model.eval()
    n = categorical.shape[0]
    result = np.empty(n, dtype=np.float64)

    with torch.inference_mode():
        for start in range(0, n, 32768):
            end = min(start + 32768, n)
            cat = torch.from_numpy(categorical[start:end])
            hist = torch.from_numpy(history[start:end])
            video = torch.from_numpy(candidate_video[start:end])
            result[start:end] = (
                model(cat, hist, video).cpu().numpy().astype(np.float64)
            )
    return result


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    group_index = np.cumsum(starts_mask) - 1
    positions = np.arange(n, dtype=np.int64) - starts[group_index]
    sizes = np.diff(np.append(starts, n))
    denominators = np.maximum(sizes[group_index] - 1, 1)

    ranked_sorted = positions.astype(np.float64) / denominators
    ranked_sorted[sizes[group_index] == 1] = 0.5
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


train = load("train")
valid = load("valid")

train_categorical = build_categorical(train)
valid_categorical = build_categorical(valid)

(
    train_history,
    history_bank,
    history_bank_starts,
    train_positive_counts,
) = build_positive_history(train)

valid_history = build_eval_history(
    valid,
    history_bank,
    history_bank_starts,
    train_positive_counts,
)

train_video = (
    np.asarray(train.X["video_id"], dtype=np.int64) + 1
)
valid_video = (
    np.asarray(valid.X["video_id"], dtype=np.int64) + 1
)
train_labels = np.asarray(train.y, dtype=np.float32)
train_weights = recency_weights(train.date)

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared_dir, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared_dir, "incumbent_test_scores.npy"
)
if not os.path.exists(inc_valid_path):
    raise FileNotFoundError("Trusted validation incumbent is unavailable")
if not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted test incumbent is unavailable")

inc_valid = np.asarray(
    np.load(inc_valid_path), dtype=np.float64
)
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

families = ["mean", "din", "gru"]
models = {}
valid_raw = {}
candidate_values = {}
candidate_arrays = {}
candidate_owner = {}
candidate_weights = {}

for family_index, family in enumerate(families):
    model = train_model(
        family=family,
        categorical=train_categorical,
        history=train_history,
        candidate_video=train_video,
        labels=train_labels,
        weights=train_weights,
        seed=SEED + 1000 * family_index,
    )
    raw_scores = predict_model(
        model,
        valid_categorical,
        valid_history,
        valid_video,
    )
    models[family] = model
    valid_raw[family] = raw_scores

    standalone = evaluate(valid.user_id, valid.y, raw_scores)
    standalone_name = f"{family}_standalone"
    candidate_values[standalone_name] = float(
        standalone["primary"]
    )
    candidate_arrays[standalone_name] = raw_scores
    candidate_owner[standalone_name] = family
    candidate_weights[standalone_name] = None

    own_rank = within_user_rank(valid.user_id, raw_scores)
    for own_weight in (0.20, 0.35, 0.50, 0.65, 0.80):
        blended = (
            own_weight * own_rank
            + (1.0 - own_weight) * inc_valid_rank
        )
        name = f"{family}_blend_{own_weight:.2f}"
        result = evaluate(valid.user_id, valid.y, blended)
        candidate_values[name] = float(result["primary"])
        candidate_arrays[name] = blended
        candidate_owner[name] = family
        candidate_weights[name] = own_weight

winner_name = max(candidate_values, key=candidate_values.get)
winner_family = candidate_owner[winner_name]
winner_weight = candidate_weights[winner_name]
valid_scores = candidate_arrays[winner_name]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(
    candidate_values, sort_keys=True
))
history_coverage = float(
    np.mean(np.any(valid_history != 0, axis=1))
)
mean_history_length = float(
    np.mean(np.sum(valid_history != 0, axis=1))
)
print(
    "FINDINGS "
    + json.dumps({
        "valid_rows_with_positive_train_history": history_coverage,
        "valid_mean_clipped_history_length": mean_history_length,
        "winner": winner_name,
    }, sort_keys=True)
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if winner_weight is not None:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(
                valid_raw[winner_family], dtype=np.float64
            ),
        )

# Generate test features and scores only after all choices have been made from
# training and public validation. Test labels are never accessed.
test = load("test")
test_categorical = build_categorical(test)
test_history = build_eval_history(
    test,
    history_bank,
    history_bank_starts,
    train_positive_counts,
)
test_video = (
    np.asarray(test.X["video_id"], dtype=np.int64) + 1
)
test_raw = predict_model(
    models[winner_family],
    test_categorical,
    test_history,
    test_video,
)

if winner_weight is None:
    test_scores = test_raw
else:
    inc_test = np.asarray(
        np.load(inc_test_path), dtype=np.float64
    )
    own_test_rank = within_user_rank(test.user_id, test_raw)
    inc_test_rank = within_user_rank(test.user_id, inc_test)
    test_scores = (
        winner_weight * own_test_rank
        + (1.0 - winner_weight) * inc_test_rank
    )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))