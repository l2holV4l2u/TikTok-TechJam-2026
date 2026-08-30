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
SEED = 73421
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
DEVICE = torch.device("cpu")

FIELDS = [
    "user_id", "video_id", "author_id", "tag",
    "tab", "duration_bucket", "hour",
]
CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1], dtype=np.int64)
TOTAL_CARD = int(sum(CARDS))
VIDEO_FIELD = FIELDS.index("video_id")
VIDEO_OFFSET = int(OFFSETS[VIDEO_FIELD])

EMBED_DIM = 12
HISTORY_LEN = 8
BATCH_SIZE = 4096
PRED_BATCH = 16384
EPOCHS = 2
HARD_CANDIDATES = 3


def make_x(split):
    return np.stack(
        [
            np.asarray(split.X[f], dtype=np.int64) + OFFSETS[j]
            for j, f in enumerate(FIELDS)
        ],
        axis=1,
    )


def concatenate_x(a, b):
    return np.concatenate([make_x(a), make_x(b)], axis=0)


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates, dtype=np.float64)
    age = float(dates.max()) - dates
    w = np.exp2(-age / float(half_life))
    w /= max(float(w.mean()), 1e-12)
    return w.astype(np.float32)


def causal_positive_histories(user_ids, time_ms, video_ids, labels, history_len):
    """
    For every training row, return the previous K positive videos belonging
    to the same user. The current row is always excluded.
    """
    user_ids = np.asarray(user_ids, dtype=np.int64)
    time_ms = np.asarray(time_ms, dtype=np.int64)
    video_ids = np.asarray(video_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    n = len(user_ids)

    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, time_ms, user_ids))
    su = user_ids[order]
    sy = labels[order]
    sv = video_ids[order]

    change = np.r_[True, su[1:] != su[:-1]]
    group = np.cumsum(change, dtype=np.int64) - 1
    starts = np.maximum.accumulate(
        np.where(change, np.arange(n, dtype=np.int64), 0)
    )
    local = np.arange(n, dtype=np.int64) - starts

    # Encoding by group makes one global maximum-accumulate equivalent to
    # a separate cumulative maximum for every user.
    stride = n + 1
    base = group * stride
    encoded_positive = base + np.where(sy == 1, local + 1, 0)
    cumulative = np.maximum.accumulate(encoded_positive)

    previous_encoded = np.empty(n, dtype=np.int64)
    previous_encoded[0] = base[0]
    previous_encoded[1:] = cumulative[:-1]
    previous_encoded[change] = base[change]

    previous_local = previous_encoded - base - 1
    previous_ptr = np.where(
        previous_local >= 0, starts + previous_local, -1
    ).astype(np.int64)

    hist_sorted = np.zeros((n, history_len), dtype=np.int32)
    ptr = previous_ptr.copy()
    for k in range(history_len):
        valid = ptr >= 0
        if not valid.any():
            break
        hist_sorted[valid, k] = sv[ptr[valid]].astype(np.int32)
        next_ptr = np.full(n, -1, dtype=np.int64)
        next_ptr[valid] = previous_ptr[ptr[valid]]
        ptr = next_ptr

    hist = np.empty_like(hist_sorted)
    hist[order] = hist_sorted

    # Last positive pointer for every user, used to construct evaluation
    # histories from the fitted split only.
    user_card = int(FEATURE_CARDINALITIES["user_id"])
    tails = np.full(user_card, -1, dtype=np.int64)
    positive_positions = np.flatnonzero(sy == 1)
    if len(positive_positions):
        np.maximum.at(tails, su[positive_positions], positive_positions)

    static = np.zeros((user_card, history_len), dtype=np.int32)
    ptr = tails.copy()
    for k in range(history_len):
        valid = ptr >= 0
        if not valid.any():
            break
        static[valid, k] = sv[ptr[valid]].astype(np.int32)
        next_ptr = np.full(user_card, -1, dtype=np.int64)
        next_ptr[valid] = previous_ptr[ptr[valid]]
        ptr = next_ptr

    return hist, static


def static_histories(static_by_user, user_ids):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    out = np.zeros((len(user_ids), static_by_user.shape[1]), dtype=np.int32)
    valid = (user_ids >= 0) & (user_ids < len(static_by_user))
    out[valid] = static_by_user[user_ids[valid]]
    return out


def pair_sampling_state(user_ids, labels):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    _, inverse = np.unique(user_ids, return_inverse=True)
    inverse = inverse.astype(np.int64)
    n_groups = int(inverse.max()) + 1

    counts = np.bincount(inverse, minlength=n_groups).astype(np.int64)
    positives = np.bincount(
        inverse, weights=labels, minlength=n_groups
    ).astype(np.int64)
    negatives = counts - positives

    row_order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts[:-1])].astype(np.int64)
    anchors = np.flatnonzero(
        (labels == 1) & (negatives[inverse] > 0)
    ).astype(np.int64)
    return inverse, counts, starts, row_order, anchors


def sample_negative_matrix(state, labels, rng, n_candidates):
    inverse, counts, starts, row_order, anchors = state
    labels = np.asarray(labels, dtype=np.int8)
    groups = inverse[anchors]
    result = np.empty((len(anchors), n_candidates), dtype=np.int64)

    for c in range(n_candidates):
        offsets = (rng.random(len(anchors)) * counts[groups]).astype(np.int64)
        sampled = row_order[starts[groups] + offsets]
        bad = labels[sampled] != 0
        while bad.any():
            gb = groups[bad]
            offsets = (
                rng.random(int(bad.sum())) * counts[gb]
            ).astype(np.int64)
            sampled[bad] = row_order[starts[gb] + offsets]
            bad = labels[sampled] != 0
        result[:, c] = sampled

    return anchors, result


class BaseSequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARD, EMBED_DIM)
        self.linear = nn.Embedding(TOTAL_CARD, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def base_parts(self, x, hist):
        field_embeddings = self.embedding(x)
        hist_global = hist.long() + VIDEO_OFFSET
        hist_embeddings = self.embedding(hist_global)
        mask = hist.ne(0)
        wide = self.linear(x).sum(dim=1).squeeze(-1) + self.bias
        return field_embeddings, hist_embeddings, mask, wide


class DINModel(BaseSequenceModel):
    def __init__(self):
        super().__init__()
        deep_in = len(FIELDS) * EMBED_DIM + 4 * EMBED_DIM
        self.deep = nn.Sequential(
            nn.Linear(deep_in, 112),
            nn.ReLU(),
            nn.Linear(112, 40),
            nn.ReLU(),
            nn.Linear(40, 1),
        )

    def forward(self, x, hist):
        fields, he, mask, wide = self.base_parts(x, hist)
        target = fields[:, VIDEO_FIELD]

        logits = (he * target[:, None, :]).sum(dim=2)
        logits = logits / np.sqrt(float(EMBED_DIM))
        logits = logits.masked_fill(~mask, -1e4)
        attention = torch.softmax(logits, dim=1)
        attention = attention * mask.float()
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-6)
        context = (attention[:, :, None] * he).sum(dim=1)

        count = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        mean_context = (he * mask[:, :, None]).sum(dim=1) / count

        interaction = torch.cat(
            [
                context,
                mean_context,
                context * target,
                torch.abs(context - target),
            ],
            dim=1,
        )
        deep_input = torch.cat([fields.flatten(1), interaction], dim=1)
        return wide + self.deep(deep_input).squeeze(-1)


class GRUHistoryModel(BaseSequenceModel):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            EMBED_DIM, 24, num_layers=1, batch_first=True
        )
        deep_in = len(FIELDS) * EMBED_DIM + 24 + 2 * EMBED_DIM
        self.deep = nn.Sequential(
            nn.Linear(deep_in, 112),
            nn.ReLU(),
            nn.Linear(112, 40),
            nn.ReLU(),
            nn.Linear(40, 1),
        )

    def forward(self, x, hist):
        fields, he, mask, wide = self.base_parts(x, hist)
        # Histories are newest-first; GRU consumes oldest-first.
        sequence = torch.flip(he * mask[:, :, None], dims=[1])
        _, hidden = self.gru(sequence)
        state = hidden[-1]
        target = fields[:, VIDEO_FIELD]
        mean_context = (
            (he * mask[:, :, None]).sum(dim=1)
            / mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        )
        deep_input = torch.cat(
            [fields.flatten(1), state, target * mean_context,
             torch.abs(target - mean_context)],
            dim=1,
        )
        return wide + self.deep(deep_input).squeeze(-1)


class DCNHistoryModel(BaseSequenceModel):
    def __init__(self):
        super().__init__()
        self.input_dim = (len(FIELDS) + 1) * EMBED_DIM
        self.cross_w = nn.ParameterList([
            nn.Parameter(torch.empty(self.input_dim)) for _ in range(3)
        ])
        self.cross_b = nn.ParameterList([
            nn.Parameter(torch.zeros(self.input_dim)) for _ in range(3)
        ])
        for w in self.cross_w:
            nn.init.normal_(w, std=0.02)

        self.deep = nn.Sequential(
            nn.Linear(self.input_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
        )
        self.output = nn.Linear(self.input_dim + 32, 1)

    def forward(self, x, hist):
        fields, he, mask, wide = self.base_parts(x, hist)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        pooled = (he * mask[:, :, None]).sum(dim=1) / count
        x0 = torch.cat([fields.flatten(1), pooled], dim=1)

        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scalar = (cross * w).sum(dim=1, keepdim=True)
            cross = x0 * scalar + b + cross

        deep = self.deep(x0)
        return wide + self.output(torch.cat([cross, deep], dim=1)).squeeze(-1)


MODEL_CLASSES = {
    "din_hard_pairwise": DINModel,
    "gru_hard_pairwise": GRUHistoryModel,
    "dcn_history_cross": DCNHistoryModel,
}


@torch.inference_mode()
def predict_model(model, x_np, hist_np, batch_size=PRED_BATCH):
    model.eval()
    out = np.empty(len(x_np), dtype=np.float64)
    for start in range(0, len(x_np), batch_size):
        end = min(start + batch_size, len(x_np))
        x = torch.from_numpy(x_np[start:end]).to(DEVICE)
        h = torch.from_numpy(hist_np[start:end].astype(
            np.int64, copy=False
        )).to(DEVICE)
        out[start:end] = model(x, h).cpu().numpy().astype(np.float64)
    return out


def fit_model(model_name, x_np, hist_np, labels, users, dates, seed):
    torch.manual_seed(seed)
    model = MODEL_CLASSES[model_name]().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0012, weight_decay=1e-6
    )

    labels = np.asarray(labels, dtype=np.int8)
    weights = recency_weights(dates, half_life=4.0)
    state = pair_sampling_state(users, labels)
    rng = np.random.default_rng(seed + 101)
    shuffle_gen = torch.Generator(device="cpu")
    shuffle_gen.manual_seed(seed + 303)

    model.train()
    for epoch in range(EPOCHS):
        n_candidates = 1 if epoch == 0 else HARD_CANDIDATES
        anchors, negative_matrix = sample_negative_matrix(
            state, labels, rng, n_candidates
        )

        if n_candidates > 1:
            candidate_scores = np.empty(
                (len(anchors), n_candidates), dtype=np.float32
            )
            model.eval()
            with torch.inference_mode():
                for c in range(n_candidates):
                    candidate_scores[:, c] = predict_model(
                        model,
                        x_np[negative_matrix[:, c]],
                        hist_np[negative_matrix[:, c]],
                        batch_size=PRED_BATCH,
                    ).astype(np.float32)
            choice = np.argmax(candidate_scores, axis=1)
            negatives = negative_matrix[
                np.arange(len(anchors), dtype=np.int64), choice
            ]
            model.train()
        else:
            negatives = negative_matrix[:, 0]

        permutation = torch.randperm(
            len(anchors), generator=shuffle_gen
        ).numpy()

        for start in range(0, len(permutation), BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            pidx = anchors[idx]
            nidx = negatives[idx]

            xp = torch.from_numpy(x_np[pidx]).to(DEVICE)
            hp = torch.from_numpy(
                hist_np[pidx].astype(np.int64, copy=False)
            ).to(DEVICE)
            xn = torch.from_numpy(x_np[nidx]).to(DEVICE)
            hn = torch.from_numpy(
                hist_np[nidx].astype(np.int64, copy=False)
            ).to(DEVICE)
            wb = torch.from_numpy(weights[pidx]).to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            positive = model(xp, hp)
            negative = model(xn, hn)
            loss = (
                nn.functional.softplus(-(positive - negative)) * wb
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def within_user_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((
        np.arange(n, dtype=np.int64), scores, user_ids
    ))
    sorted_users = user_ids[order]

    change = np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    starts = np.maximum.accumulate(
        np.where(change, np.arange(n, dtype=np.int64), 0)
    )
    positions = np.arange(n, dtype=np.int64) - starts

    end_change = np.r_[sorted_users[:-1] != sorted_users[1:], True]
    ends = np.minimum.accumulate(
        np.where(
            end_change, np.arange(n, dtype=np.int64), n - 1
        )[::-1]
    )[::-1]
    lengths = ends - starts + 1

    ranked = np.where(
        lengths > 1,
        positions.astype(np.float64) / np.maximum(lengths - 1, 1),
        0.5,
    )
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


train = load("train")
valid = load("valid")

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_rank = within_user_percentile(valid.user_id, inc_valid)

x_train = make_x(train)
x_valid = make_x(valid)
y_train = np.asarray(train.y, dtype=np.int8)

hist_train, static_train = causal_positive_histories(
    train.user_id,
    train.time_ms,
    train.video_id,
    y_train,
    HISTORY_LEN,
)
hist_valid = static_histories(static_train, valid.user_id)

candidate_scores = {}
candidate_predictions = {}
candidate_recipes = {}

inc_metric = evaluate(valid.user_id, valid.y, inc_valid)
candidate_scores["incumbent"] = float(inc_metric["primary"])
candidate_predictions["incumbent"] = inc_valid
candidate_recipes["incumbent"] = ("incumbent", 0.0)

standalone_results = {}

for j, family in enumerate(MODEL_CLASSES):
    model = fit_model(
        family,
        x_train,
        hist_train,
        y_train,
        train.user_id,
        train.date,
        SEED + 1000 * (j + 1),
    )
    raw = predict_model(model, x_valid, hist_valid)
    rank = within_user_percentile(valid.user_id, raw)

    standalone_metric = evaluate(valid.user_id, valid.y, raw)
    standalone_results[family] = float(standalone_metric["primary"])
    candidate_scores[family] = float(standalone_metric["primary"])
    candidate_predictions[family] = raw
    candidate_recipes[family] = (family, 1.0)

    for alpha in (0.10, 0.20, 0.35, 0.50, 0.70):
        name = "%s_rankblend_%.2f" % (family, alpha)
        blended = alpha * rank + (1.0 - alpha) * inc_rank
        metric = evaluate(valid.user_id, valid.y, blended)
        candidate_scores[name] = float(metric["primary"])
        candidate_predictions[name] = blended
        candidate_recipes[name] = (family, alpha)

    del model

winner = max(candidate_scores, key=candidate_scores.get)
winner_family, winner_alpha = candidate_recipes[winner]
valid_scores = np.asarray(candidate_predictions[winner], dtype=np.float64)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS standalone DIN=%.6f GRU=%.6f DCN=%.6f winner=%s"
    % (
        standalone_results["din_hard_pairwise"],
        standalone_results["gru_hard_pairwise"],
        standalone_results["dcn_history_cross"],
        winner,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if winner_family == "incumbent":
    test_scores = inc_test.copy()
else:
    y_valid = np.asarray(valid.y, dtype=np.int8)
    y_combined = np.concatenate([y_train, y_valid])
    users_combined = np.concatenate([
        np.asarray(train.user_id, dtype=np.int64),
        np.asarray(valid.user_id, dtype=np.int64),
    ])
    times_combined = np.concatenate([
        np.asarray(train.time_ms, dtype=np.int64),
        np.asarray(valid.time_ms, dtype=np.int64),
    ])
    videos_combined = np.concatenate([
        np.asarray(train.video_id, dtype=np.int64),
        np.asarray(valid.video_id, dtype=np.int64),
    ])
    dates_combined = np.concatenate([
        np.asarray(train.date),
        np.asarray(valid.date),
    ])
    x_combined = concatenate_x(train, valid)
    x_test = make_x(test)

    hist_combined, static_combined = causal_positive_histories(
        users_combined,
        times_combined,
        videos_combined,
        y_combined,
        HISTORY_LEN,
    )
    hist_test = static_histories(static_combined, test.user_id)

    selected_index = list(MODEL_CLASSES.keys()).index(winner_family)
    final_model = fit_model(
        winner_family,
        x_combined,
        hist_combined,
        y_combined,
        users_combined,
        dates_combined,
        SEED + 1000 * (selected_index + 1),
    )
    test_raw = predict_model(final_model, x_test, hist_test)

    if winner_alpha >= 0.999:
        test_scores = test_raw
    else:
        test_rank = within_user_percentile(test.user_id, test_raw)
        inc_test_rank = within_user_percentile(test.user_id, inc_test)
        test_scores = (
            winner_alpha * test_rank
            + (1.0 - winner_alpha) * inc_test_rank
        )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)