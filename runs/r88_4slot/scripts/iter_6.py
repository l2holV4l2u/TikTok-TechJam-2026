import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 19427
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
]
VIDEO_FIELD = FIELDS.index("video_id")
CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1], dtype=np.int64)
TOTAL_CARD = int(sum(CARDS))
VIDEO_OFFSET = int(OFFSETS[VIDEO_FIELD])

HISTORY_LEN = 12
EMBED_DIM = 12
BATCH_SIZE = 8192
LR = 0.002
CHECKPOINTS = (2, 4)
MAX_EPOCHS = max(CHECKPOINTS)


def encode(split):
    return np.stack(
        [
            np.asarray(split.X[f], dtype=np.int64) + OFFSETS[j]
            for j, f in enumerate(FIELDS)
        ],
        axis=1,
    )


def chronological_histories(parts, target_part):
    users = np.concatenate(
        [np.asarray(p.user_id, dtype=np.int64) for p in parts]
    )
    times = np.concatenate(
        [np.asarray(p.time_ms, dtype=np.int64) for p in parts]
    )
    videos = np.concatenate(
        [np.asarray(p.X["video_id"], dtype=np.int64) for p in parts]
    )
    n_each = [len(np.asarray(p.user_id)) for p in parts]
    n = len(users)

    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, times, users))
    sorted_users = users[order]
    sorted_videos = videos[order]

    hist_sorted = np.zeros((n, HISTORY_LEN), dtype=np.int32)
    for lag in range(1, HISTORY_LEN + 1):
        if lag >= n:
            break
        valid = sorted_users[lag:] == sorted_users[:-lag]
        dest = np.arange(lag, n, dtype=np.int64)[valid]
        src = np.arange(0, n - lag, dtype=np.int64)[valid]
        hist_sorted[dest, lag - 1] = sorted_videos[src].astype(
            np.int32, copy=False
        )

    hist = np.empty_like(hist_sorted)
    hist[order] = hist_sorted

    begin = int(sum(n_each[:target_part]))
    end = begin + n_each[target_part]
    return hist[begin:end]


def recency_weights(dates, half_life=7.0):
    dates = np.asarray(dates, dtype=np.int64)
    unique_dates = np.unique(dates)
    day_index = np.searchsorted(unique_dates, dates).astype(np.float32)
    age = float(day_index.max()) - day_index
    weights = np.exp2(-age / half_life).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-6)
    return weights


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.r_[
        0,
        np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1,
    ]
    ends = np.r_[starts[1:], n]
    counts = ends - starts

    repeated_starts = np.repeat(starts, counts)
    repeated_counts = np.repeat(counts, counts)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    ranked_sorted = np.full(n, 0.5, dtype=np.float64)
    mask = repeated_counts > 1
    ranked_sorted[mask] = (
        positions[mask] / (repeated_counts[mask] - 1.0)
    )

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


class SequenceBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            TOTAL_CARD,
            EMBED_DIM,
            padding_idx=VIDEO_OFFSET,
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.025)
        with torch.no_grad():
            self.embedding.weight[VIDEO_OFFSET].zero_()

    def static_embeddings(self, x):
        return self.embedding(x)

    def history_embeddings(self, history):
        return self.embedding(history)


class DINModel(SequenceBase):
    def __init__(self):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(4 * EMBED_DIM, 48),
            nn.PReLU(),
            nn.Linear(48, 16),
            nn.PReLU(),
            nn.Linear(16, 1),
        )
        input_dim = len(FIELDS) * EMBED_DIM + 3 * EMBED_DIM
        self.output = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.PReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 48),
            nn.PReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x, history):
        static = self.static_embeddings(x)
        candidate = static[:, VIDEO_FIELD]
        hist = self.history_embeddings(history)
        query = candidate.unsqueeze(1).expand_as(hist)

        attention_input = torch.cat(
            [query, hist, query - hist, query * hist],
            dim=2,
        )
        logits = self.attention(attention_input).squeeze(2)
        mask = history != VIDEO_OFFSET
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        interest = torch.sum(hist * weights.unsqueeze(2), dim=1)

        z = torch.cat(
            [
                static.reshape(x.shape[0], -1),
                interest,
                candidate * interest,
                candidate - interest,
            ],
            dim=1,
        )
        return self.output(z).squeeze(1)


class GRUHistoryModel(SequenceBase):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=EMBED_DIM,
            hidden_size=24,
            num_layers=1,
            batch_first=True,
        )
        input_dim = len(FIELDS) * EMBED_DIM + 24 + EMBED_DIM
        self.output = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x, history):
        static = self.static_embeddings(x)
        candidate = static[:, VIDEO_FIELD]

        chronological = torch.flip(history, dims=[1])
        hist = self.history_embeddings(chronological)
        sequence_output, _ = self.gru(hist)

        valid = chronological != VIDEO_OFFSET
        positions = torch.arange(
            chronological.shape[1],
            device=chronological.device,
        ).unsqueeze(0)
        last_positions = torch.where(
            valid,
            positions,
            torch.full_like(positions, -1),
        ).max(dim=1).values
        safe_positions = last_positions.clamp_min(0)

        batch_positions = torch.arange(
            x.shape[0], device=x.device
        )
        state = sequence_output[batch_positions, safe_positions]
        state = state * (last_positions >= 0).float().unsqueeze(1)

        projected_state = state[:, :EMBED_DIM]
        z = torch.cat(
            [
                static.reshape(x.shape[0], -1),
                state,
                candidate * projected_state,
            ],
            dim=1,
        )
        return self.output(z).squeeze(1)


def predict_neural(model, x_np, h_np):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float32)
    with torch.inference_mode():
        for lo in range(0, len(x_np), BATCH_SIZE * 2):
            hi = min(lo + BATCH_SIZE * 2, len(x_np))
            xb = torch.from_numpy(x_np[lo:hi])
            hb = torch.from_numpy(
                h_np[lo:hi].astype(np.int64, copy=False) + VIDEO_OFFSET
            )
            result[lo:hi] = model(xb, hb).cpu().numpy()
    return result


def train_select_neural(
    model_class,
    x_train,
    h_train,
    y_train,
    dates_train,
    x_valid,
    h_valid,
    y_valid,
    valid_users,
):
    torch.manual_seed(SEED)
    model = model_class()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=2e-6,
    )

    x_t = torch.from_numpy(x_train)
    h_t = torch.from_numpy(
        h_train.astype(np.int64, copy=False) + VIDEO_OFFSET
    )
    y_t = torch.from_numpy(y_train.astype(np.float32, copy=False))
    w_t = torch.from_numpy(recency_weights(dates_train))
    n = len(x_train)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    best_epoch = None
    best_scores = None
    best_metrics = None
    best_primary = -np.inf

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(n, generator=generator)

        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:min(lo + BATCH_SIZE, n)]
            logits = model(x_t[idx], h_t[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits,
                y_t[idx],
                reduction="none",
            )
            loss = (losses * w_t[idx]).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if epoch in CHECKPOINTS:
            scores = predict_neural(model, x_valid, h_valid)
            metrics = evaluate(valid_users, y_valid, scores)
            primary = float(metrics["primary"])
            if primary > best_primary:
                best_primary = primary
                best_epoch = epoch
                best_scores = scores.copy()
                best_metrics = metrics

    return best_epoch, best_scores, best_metrics


def fit_fixed_neural(
    model_class,
    x_fit,
    h_fit,
    y_fit,
    dates_fit,
    epochs,
):
    torch.manual_seed(SEED)
    model = model_class()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=2e-6,
    )

    x_t = torch.from_numpy(x_fit)
    h_t = torch.from_numpy(
        h_fit.astype(np.int64, copy=False) + VIDEO_OFFSET
    )
    y_t = torch.from_numpy(y_fit.astype(np.float32, copy=False))
    w_t = torch.from_numpy(recency_weights(dates_fit))
    n = len(x_fit)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    for _ in range(int(epochs)):
        model.train()
        order = torch.randperm(n, generator=generator)
        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:min(lo + BATCH_SIZE, n)]
            logits = model(x_t[idx], h_t[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits,
                y_t[idx],
                reduction="none",
            )
            loss = (losses * w_t[idx]).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def grouped_sum_count(keys, labels):
    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
    sorted_y = labels[order].astype(np.float64, copy=False)

    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    unique_keys = sorted_keys[starts]
    counts = np.diff(np.r_[starts, len(sorted_keys)]).astype(np.float64)
    sums = np.add.reduceat(sorted_y, starts)
    return unique_keys, sums, counts


def lookup_stats(query_keys, unique_keys, sums, counts):
    pos = np.searchsorted(unique_keys, query_keys)
    present = pos < len(unique_keys)
    safe = np.minimum(pos, max(len(unique_keys) - 1, 0))
    if len(unique_keys):
        present &= unique_keys[safe] == query_keys
    else:
        present[:] = False

    out_sum = np.zeros(len(query_keys), dtype=np.float64)
    out_count = np.zeros(len(query_keys), dtype=np.float64)
    if len(unique_keys):
        out_sum[present] = sums[safe[present]]
        out_count[present] = counts[safe[present]]
    return out_sum, out_count


def transition_scores(
    fit_video,
    fit_author,
    fit_history,
    fit_y,
    query_video,
    query_author,
    query_history,
):
    fit_video = np.asarray(fit_video, dtype=np.int64)
    query_video = np.asarray(query_video, dtype=np.int64)
    fit_author = np.asarray(fit_author, dtype=np.int64)
    query_author = np.asarray(query_author, dtype=np.int64)
    fit_y = np.asarray(fit_y, dtype=np.float64)

    video_card = int(FEATURE_CARDINALITIES["video_id"])
    author_card = int(FEATURE_CARDINALITIES["author_id"])

    video_keys, video_sums, video_counts = grouped_sum_count(
        fit_video, fit_y
    )
    q_video_sum, q_video_count = lookup_stats(
        query_video, video_keys, video_sums, video_counts
    )

    global_rate = float(fit_y.mean())
    video_rate = (
        q_video_sum + 25.0 * global_rate
    ) / (q_video_count + 25.0)

    previous_fit = fit_history[:, 0].astype(np.int64, copy=False)
    previous_query = query_history[:, 0].astype(np.int64, copy=False)

    pair_fit_key = previous_fit * video_card + fit_video
    pair_query_key = previous_query * video_card + query_video
    pair_keys, pair_sums, pair_counts = grouped_sum_count(
        pair_fit_key, fit_y
    )
    pair_sum, pair_count = lookup_stats(
        pair_query_key, pair_keys, pair_sums, pair_counts
    )
    pair_rate = (
        pair_sum + 8.0 * video_rate
    ) / (pair_count + 8.0)

    fit_prev_author = np.zeros(len(fit_y), dtype=np.int64)
    valid_prev = previous_fit > 0
    if np.any(valid_prev):
        video_to_author = np.zeros(video_card, dtype=np.int64)
        np.maximum.at(video_to_author, fit_video, fit_author)
        clipped = np.minimum(previous_fit, video_card - 1)
        fit_prev_author[valid_prev] = video_to_author[
            clipped[valid_prev]
        ]

    query_prev_author = np.zeros(len(query_video), dtype=np.int64)
    if np.any(previous_query > 0):
        video_to_author = np.zeros(video_card, dtype=np.int64)
        np.maximum.at(video_to_author, fit_video, fit_author)
        clipped = np.minimum(previous_query, video_card - 1)
        qmask = previous_query > 0
        query_prev_author[qmask] = video_to_author[clipped[qmask]]

    author_pair_fit = fit_prev_author * author_card + fit_author
    author_pair_query = query_prev_author * author_card + query_author
    ap_keys, ap_sums, ap_counts = grouped_sum_count(
        author_pair_fit, fit_y
    )
    ap_sum, ap_count = lookup_stats(
        author_pair_query, ap_keys, ap_sums, ap_counts
    )
    author_pair_rate = (
        ap_sum + 15.0 * video_rate
    ) / (ap_count + 15.0)

    return (
        0.60 * pair_rate
        + 0.25 * author_pair_rate
        + 0.15 * video_rate
    ).astype(np.float32)


train = load("train")
valid = load("valid")

x_train = encode(train)
x_valid = encode(valid)
h_train = chronological_histories([train], 0)
h_valid = chronological_histories([train, valid], 1)

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)
dates_train = np.asarray(train.date)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.load(inc_valid_path)
inc_valid_rank = within_user_rank(valid_users, inc_valid)

families = {
    "din_attention": DINModel,
    "gru_history": GRUHistoryModel,
}

raw_candidates = {}
candidate_epochs = {}
standalone_metrics = {}

for name, model_class in families.items():
    epoch, scores, metrics = train_select_neural(
        model_class=model_class,
        x_train=x_train,
        h_train=h_train,
        y_train=y_train,
        dates_train=dates_train,
        x_valid=x_valid,
        h_valid=h_valid,
        y_valid=y_valid,
        valid_users=valid_users,
    )
    raw_candidates[name] = scores
    candidate_epochs[name] = int(epoch)
    standalone_metrics[name] = metrics

transition_valid = transition_scores(
    fit_video=np.asarray(train.X["video_id"]),
    fit_author=np.asarray(train.X["author_id"]),
    fit_history=h_train,
    fit_y=y_train,
    query_video=np.asarray(valid.X["video_id"]),
    query_author=np.asarray(valid.X["author_id"]),
    query_history=h_valid,
)
raw_candidates["transition_empirical_bayes"] = transition_valid
candidate_epochs["transition_empirical_bayes"] = 0
standalone_metrics["transition_empirical_bayes"] = evaluate(
    valid_users, y_valid, transition_valid
)

alphas = np.linspace(0.0, 1.0, 11)
recorded = {}
best_primary = -np.inf
best_name = None
best_alpha = None
best_valid_scores = None
best_raw_scores = None
best_metrics = None

for name, raw_scores in raw_candidates.items():
    recorded[name + "_standalone"] = float(
        standalone_metrics[name]["primary"]
    )
    model_rank = within_user_rank(valid_users, raw_scores)

    local_best = -np.inf
    local_alpha = None
    for alpha in alphas:
        blended = (
            (1.0 - float(alpha)) * inc_valid_rank
            + float(alpha) * model_rank
        )
        metrics = evaluate(valid_users, y_valid, blended)
        primary = float(metrics["primary"])

        if primary > local_best:
            local_best = primary
            local_alpha = float(alpha)

        if primary > best_primary:
            best_primary = primary
            best_name = name
            best_alpha = float(alpha)
            best_valid_scores = blended.copy()
            best_raw_scores = raw_scores.copy()
            best_metrics = metrics

    recorded[name + "_best_blend"] = float(local_best)
    recorded[name + "_blend_alpha"] = float(local_alpha)
    recorded[name + "_epoch"] = int(candidate_epochs[name])

print("CANDIDATES " + json.dumps(recorded, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(best_raw_scores, dtype=np.float64),
    )

test = load("test")

x_fit = np.concatenate([x_train, x_valid], axis=0)
y_fit = np.concatenate(
    [y_train, np.asarray(valid.y, dtype=np.float32)],
    axis=0,
)
dates_fit = np.concatenate(
    [np.asarray(train.date), np.asarray(valid.date)],
    axis=0,
)
h_fit = chronological_histories([train, valid], 0)
# The function above with target_part=0 returns only train, so construct the
# combined causal history directly through a lightweight concatenated proxy.
class CombinedSplit:
    pass

combined = CombinedSplit()
combined.user_id = np.concatenate(
    [np.asarray(train.user_id), np.asarray(valid.user_id)]
)
combined.time_ms = np.concatenate(
    [np.asarray(train.time_ms), np.asarray(valid.time_ms)]
)
combined.X = {
    "video_id": np.concatenate(
        [
            np.asarray(train.X["video_id"]),
            np.asarray(valid.X["video_id"]),
        ]
    )
}
h_fit = chronological_histories([combined], 0)

x_test = encode(test)
h_test = chronological_histories([combined, test], 1)

if best_name in families:
    selected_model = fit_fixed_neural(
        model_class=families[best_name],
        x_fit=x_fit,
        h_fit=h_fit,
        y_fit=y_fit,
        dates_fit=dates_fit,
        epochs=candidate_epochs[best_name],
    )
    raw_test = predict_neural(selected_model, x_test, h_test)
else:
    raw_test = transition_scores(
        fit_video=np.concatenate(
            [
                np.asarray(train.X["video_id"]),
                np.asarray(valid.X["video_id"]),
            ]
        ),
        fit_author=np.concatenate(
            [
                np.asarray(train.X["author_id"]),
                np.asarray(valid.X["author_id"]),
            ]
        ),
        fit_history=h_fit,
        fit_y=y_fit,
        query_video=np.asarray(test.X["video_id"]),
        query_author=np.asarray(test.X["author_id"]),
        query_history=h_test,
    )

inc_test = np.load(inc_test_path)
test_users = np.asarray(test.user_id)
inc_test_rank = within_user_rank(test_users, inc_test)
raw_test_rank = within_user_rank(test_users, raw_test)
test_scores = (
    (1.0 - best_alpha) * inc_test_rank
    + best_alpha * raw_test_rank
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print(
    "FINDINGS "
    + json.dumps(
        {
            "selected_family": best_name,
            "selected_epoch": int(candidate_epochs[best_name]),
            "selected_model_weight": float(best_alpha),
            "history_length": int(HISTORY_LEN),
        },
        sort_keys=True,
    )
)

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(", ", ": "),
    )
)