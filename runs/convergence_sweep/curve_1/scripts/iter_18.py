import os
import time
import json
import warnings

import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
warnings.filterwarnings("ignore")

SEED = 314159
HISTORY_LEN = 12
HAWKES_LEN = 48
DIM = 24
BATCH_SIZE = 4096
EPOCHS = 2
MAX_TRAIN_ROWS = 620000

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

DEVICE = torch.device("cpu")


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores.copy()

    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    ranked = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ranked[multi] = positions[multi] / (
        repeated_lengths[multi].astype(np.float64) - 1.0
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def recency_weights(dates, half_life=3.0):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    positions = np.searchsorted(unique_dates, dates)
    ages = len(unique_dates) - 1 - positions
    w = np.exp2(-ages.astype(np.float64) / float(half_life))
    w /= np.mean(w)
    return w.astype(np.float32)


class OrderedHistory:
    def __init__(self, train):
        n = len(train.user_id)
        row = np.arange(n, dtype=np.int64)
        self.order = np.lexsort(
            (row, np.asarray(train.time_ms), np.asarray(train.user_id))
        )
        self.inverse = np.empty(n, dtype=np.int64)
        self.inverse[self.order] = np.arange(n, dtype=np.int64)

        self.sorted_users = np.asarray(
            train.user_id, dtype=np.int64
        )[self.order]
        self.sorted_video = np.asarray(
            train.video_id, dtype=np.int64
        )[self.order]
        self.sorted_author = np.asarray(
            train.X["author_id"], dtype=np.int64
        )[self.order]
        self.sorted_tag = np.asarray(
            train.X["tag"], dtype=np.int64
        )[self.order]
        self.sorted_label = np.asarray(
            train.y, dtype=np.int64
        )[self.order]

        new_user = np.empty(n, dtype=bool)
        new_user[0] = True
        new_user[1:] = self.sorted_users[1:] != self.sorted_users[:-1]
        starts = np.flatnonzero(new_user)
        ends = np.r_[starts[1:], n]

        self.sorted_start = np.repeat(starts, ends - starts)

        user_card = FEATURE_CARDINALITIES["user_id"]
        self.user_end = np.full(user_card, -1, dtype=np.int64)
        ending_users = self.sorted_users[ends - 1]
        valid = (ending_users >= 0) & (ending_users < user_card)
        self.user_end[ending_users[valid]] = ends[valid]

    def for_training_rows(self, rows, length):
        rows = np.asarray(rows, dtype=np.int64)
        pos = self.inverse[rows]
        starts = self.sorted_start[pos]

        offsets = np.arange(length, 0, -1, dtype=np.int64)
        hist_pos = pos[:, None] - offsets[None, :]
        valid = hist_pos >= starts[:, None]
        safe = np.maximum(hist_pos, 0)

        return (
            np.where(valid, self.sorted_video[safe] + 1, 0).astype(np.int64),
            np.where(valid, self.sorted_author[safe] + 1, 0).astype(np.int64),
            np.where(valid, self.sorted_tag[safe] + 1, 0).astype(np.int64),
            np.where(valid, self.sorted_label[safe] + 1, 0).astype(np.int64),
            valid,
        )

    def for_split(self, split, length):
        users = np.asarray(split.user_id, dtype=np.int64)
        safe_users = np.clip(users, 0, len(self.user_end) - 1)
        ends = self.user_end[safe_users]
        known = (
            (users >= 0) &
            (users < len(self.user_end)) &
            (ends >= 0)
        )

        offsets = np.arange(length, 0, -1, dtype=np.int64)
        hist_pos = ends[:, None] - offsets[None, :]
        hist_pos = np.maximum(hist_pos, 0)

        valid = known[:, None]
        valid &= self.sorted_users[hist_pos] == users[:, None]

        return (
            np.where(valid, self.sorted_video[hist_pos] + 1, 0).astype(np.int64),
            np.where(valid, self.sorted_author[hist_pos] + 1, 0).astype(np.int64),
            np.where(valid, self.sorted_tag[hist_pos] + 1, 0).astype(np.int64),
            np.where(valid, self.sorted_label[hist_pos] + 1, 0).astype(np.int64),
            valid,
        )


CANDIDATE_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
]


def candidate_arrays(split, rows=None):
    if rows is None:
        rows = slice(None)
    return [
        np.asarray(split.X[name], dtype=np.int64)[rows]
        for name in CANDIDATE_FIELDS
    ]


class SequenceBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.video_emb = nn.Embedding(
            FEATURE_CARDINALITIES["video_id"] + 1, DIM, padding_idx=0
        )
        self.author_emb = nn.Embedding(
            FEATURE_CARDINALITIES["author_id"] + 1, DIM, padding_idx=0
        )
        self.tag_emb = nn.Embedding(
            FEATURE_CARDINALITIES["tag"] + 1, DIM, padding_idx=0
        )
        self.label_emb = nn.Embedding(3, DIM, padding_idx=0)
        self.position_emb = nn.Embedding(HISTORY_LEN, DIM)

        self.candidate_embeddings = nn.ModuleList([
            nn.Embedding(FEATURE_CARDINALITIES[name] + 1, 8)
            for name in CANDIDATE_FIELDS
        ])

        layer = nn.TransformerEncoderLayer(
            d_model=DIM,
            nhead=2,
            dim_feedforward=64,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(DIM)

    def encode_history(self, hv, ha, ht, hy, mask):
        pos = torch.arange(
            HISTORY_LEN, device=hv.device, dtype=torch.long
        )[None, :]
        token = (
            self.video_emb(hv) +
            self.author_emb(ha) +
            self.tag_emb(ht) +
            self.label_emb(hy) +
            self.position_emb(pos)
        )
        token = token * mask.unsqueeze(-1).float()

        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, -1] = True

        encoded = self.encoder(
            token,
            src_key_padding_mask=~safe_mask,
        )
        return self.norm(encoded), safe_mask, empty

    def candidate_context(self, candidates):
        pieces = []
        for emb, values in zip(self.candidate_embeddings, candidates):
            pieces.append(emb(values + 1))
        return torch.cat(pieces, dim=1)


class BSTModel(SequenceBase):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(2 * DIM, DIM)
        self.output = nn.Sequential(
            nn.Linear(DIM + 8 * len(CANDIDATE_FIELDS), 96),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(96, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, candidates, hv, ha, ht, hy, mask):
        encoded, safe_mask, _ = self.encode_history(
            hv, ha, ht, hy, mask
        )
        current_video = self.video_emb(candidates[1] + 1)
        current_author = self.author_emb(candidates[2] + 1)
        query = self.query(
            torch.cat([current_video, current_author], dim=1)
        )

        attention = torch.sum(
            encoded * query[:, None, :], dim=2
        ) / np.sqrt(DIM)
        attention = attention.masked_fill(~safe_mask, -1e4)
        attention = torch.softmax(attention, dim=1)
        interest = torch.sum(
            encoded * attention.unsqueeze(-1), dim=1
        )

        context = self.candidate_context(candidates)
        return self.output(
            torch.cat([interest, context], dim=1)
        ).squeeze(1)


class SASRecStateModel(SequenceBase):
    def __init__(self):
        super().__init__()
        self.state_projection = nn.Linear(DIM, DIM)
        self.output = nn.Sequential(
            nn.Linear(DIM + 8 * len(CANDIDATE_FIELDS) + 1, 80),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(80, 1),
        )

    def forward(self, candidates, hv, ha, ht, hy, mask):
        encoded, safe_mask, empty = self.encode_history(
            hv, ha, ht, hy, mask
        )
        lengths = safe_mask.sum(dim=1)
        last_index = torch.clamp(lengths - 1, min=0)
        state = encoded[
            torch.arange(encoded.shape[0], device=encoded.device),
            last_index,
        ]
        state = self.state_projection(state)
        state[empty] = 0.0

        target = self.video_emb(candidates[1] + 1)
        compatibility = torch.sum(state * target, dim=1, keepdim=True)
        compatibility = compatibility / np.sqrt(DIM)

        context = self.candidate_context(candidates)
        return self.output(
            torch.cat([state, context, compatibility], dim=1)
        ).squeeze(1)


def to_long(x):
    return torch.as_tensor(x, dtype=torch.long, device=DEVICE)


def fit_model(model, train, history, training_rows):
    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-3, weight_decay=2e-5
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    row_weights = recency_weights(train.date, half_life=3.0)
    labels = np.asarray(train.y, dtype=np.float32)

    rng = np.random.RandomState(SEED + model.__class__.__name__.__len__())

    for epoch in range(EPOCHS):
        shuffled = training_rows.copy()
        rng.shuffle(shuffled)

        epoch_loss = 0.0
        epoch_weight = 0.0

        for start in range(0, len(shuffled), BATCH_SIZE):
            rows = shuffled[start:start + BATCH_SIZE]
            hv, ha, ht, hy, mask = history.for_training_rows(
                rows, HISTORY_LEN
            )
            cats = candidate_arrays(train, rows)

            candidates_t = [to_long(x) for x in cats]
            logits = model(
                candidates_t,
                to_long(hv),
                to_long(ha),
                to_long(ht),
                to_long(hy),
                torch.as_tensor(mask, dtype=torch.bool, device=DEVICE),
            )

            target = torch.as_tensor(
                labels[rows], dtype=torch.float32, device=DEVICE
            )
            weight = torch.as_tensor(
                row_weights[rows], dtype=torch.float32, device=DEVICE
            )
            loss_vec = loss_fn(logits, target)
            loss = torch.sum(loss_vec * weight) / torch.sum(weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += float(torch.sum(loss_vec * weight).detach())
            epoch_weight += float(torch.sum(weight))

        print(
            "FINDINGS {}_epoch{}_weighted_loss={:.6f}".format(
                model.__class__.__name__,
                epoch + 1,
                epoch_loss / max(epoch_weight, 1.0),
            ),
            flush=True,
        )

    model.eval()
    return model


@torch.no_grad()
def predict_model(model, split, history):
    hv, ha, ht, hy, mask = history.for_split(split, HISTORY_LEN)
    cats = candidate_arrays(split)
    result = np.empty(len(split.user_id), dtype=np.float64)

    model.eval()
    for start in range(0, len(result), BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, len(result))
        candidates_t = [to_long(x[start:end]) for x in cats]
        logits = model(
            candidates_t,
            to_long(hv[start:end]),
            to_long(ha[start:end]),
            to_long(ht[start:end]),
            to_long(hy[start:end]),
            torch.as_tensor(
                mask[start:end], dtype=torch.bool, device=DEVICE
            ),
        )
        result[start:end] = logits.cpu().numpy().astype(np.float64)

    return result


def hawkes_scores(split, history):
    hv, ha, ht, hy, mask = history.for_split(split, HAWKES_LEN)

    current_video = np.asarray(split.video_id, dtype=np.int64)[:, None] + 1
    current_author = (
        np.asarray(split.X["author_id"], dtype=np.int64)[:, None] + 1
    )
    current_tag = np.asarray(split.X["tag"], dtype=np.int64)[:, None] + 1

    labels = np.where(hy == 2, 1.0, -0.55)
    labels *= mask

    ages = np.arange(
        HAWKES_LEN - 1, -1, -1, dtype=np.float64
    )[None, :]
    decay_fast = np.exp2(-ages / 3.0)
    decay_slow = np.exp2(-ages / 10.0)

    video_match = (hv == current_video) & mask
    author_match = (ha == current_author) & mask
    tag_match = (ht == current_tag) & mask

    video_signal = np.sum(
        labels * video_match * decay_fast, axis=1
    )
    author_signal = np.sum(
        labels * author_match * decay_slow, axis=1
    )
    tag_signal = np.sum(
        labels * tag_match * decay_slow, axis=1
    )

    exposure = np.sum(mask * decay_slow, axis=1)
    positive_memory = np.sum(
        (hy == 2) * mask * decay_slow, axis=1
    )
    user_tendency = (
        positive_memory + 1.5
    ) / (
        exposure + 4.5
    )

    return (
        1.50 * video_signal +
        0.65 * author_signal +
        0.40 * tag_signal +
        0.20 * np.log(np.clip(user_tendency, 1e-5, 1.0))
    ).astype(np.float64)


train = load("train")
valid = load("valid")
history = OrderedHistory(train)

all_rows = np.arange(len(train.user_id), dtype=np.int64)
dates = np.asarray(train.date, dtype=np.int32)

recent_rows = all_rows[dates >= 20220413]
if len(recent_rows) > MAX_TRAIN_ROWS:
    recent_rows = recent_rows[-MAX_TRAIN_ROWS:]

print(
    "FINDINGS sequence_training_rows={} date_min={} date_max={}".format(
        len(recent_rows),
        int(dates[recent_rows].min()),
        int(dates[recent_rows].max()),
    ),
    flush=True,
)

bst = fit_model(BSTModel(), train, history, recent_rows)
sasrec = fit_model(SASRecStateModel(), train, history, recent_rows)

valid_bst = predict_model(bst, valid, history)
valid_sasrec = predict_model(sasrec, valid, history)
valid_hawkes = hawkes_scores(valid, history)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

inc_valid = np.load(inc_valid_path).astype(np.float64)
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

raw_valid = {
    "bst_target_attention": valid_bst,
    "sasrec_state": valid_sasrec,
    "hawkes_memory": valid_hawkes,
}

candidate_log = {}
best_primary = -np.inf
best_name = None
best_alpha = None
best_valid = None
best_metrics = None
best_raw_valid = None

inc_metrics = evaluate(valid.user_id, valid.y, inc_valid)
candidate_log["incumbent"] = float(inc_metrics["primary"])

blend_weights = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0]

for name, raw_score in raw_valid.items():
    raw_metrics = evaluate(valid.user_id, valid.y, raw_score)
    candidate_log[name + "_raw"] = float(raw_metrics["primary"])

    raw_rank = within_user_rank(valid.user_id, raw_score)
    local_best = -np.inf

    for alpha in blend_weights:
        if alpha == 1.0:
            blended = raw_rank
        else:
            blended = (
                (1.0 - alpha) * inc_valid_rank +
                alpha * raw_rank
            )

        metrics = evaluate(valid.user_id, valid.y, blended)
        key = "{}_blend_{:.2f}".format(name, alpha)
        candidate_log[key] = float(metrics["primary"])
        local_best = max(local_best, float(metrics["primary"]))

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_name = name
            best_alpha = alpha
            best_valid = blended.copy()
            best_metrics = metrics
            best_raw_valid = raw_score.copy()

# The exact trusted incumbent remains eligible rather than its rank transform.
if float(inc_metrics["primary"]) >= best_primary:
    best_primary = float(inc_metrics["primary"])
    best_name = "incumbent"
    best_alpha = 0.0
    best_valid = inc_valid.copy()
    best_metrics = inc_metrics

print(
    "CANDIDATES " + json.dumps(
        candidate_log, sort_keys=True
    ),
    flush=True,
)
print(
    "FINDINGS selected_family={} selected_alpha={:.2f}".format(
        best_name, float(best_alpha)
    ),
    flush=True,
)

te = load("test")
inc_test = np.load(inc_test_path).astype(np.float64)

if best_name == "incumbent":
    test_scores = inc_test
else:
    if best_name == "bst_target_attention":
        raw_test = predict_model(bst, te, history)
    elif best_name == "sasrec_state":
        raw_test = predict_model(sasrec, te, history)
    else:
        raw_test = hawkes_scores(te, history)

    raw_test_rank = within_user_rank(te.user_id, raw_test)
    inc_test_rank = within_user_rank(te.user_id, inc_test)

    if best_alpha == 1.0:
        test_scores = raw_test_rank
    else:
        test_scores = (
            (1.0 - best_alpha) * inc_test_rank +
            best_alpha * raw_test_rank
        )

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if best_name != "incumbent":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS " + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    }),
    flush=True,
)