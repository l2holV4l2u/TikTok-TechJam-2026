import os
import gc
import json
import time
import math
import warnings
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")

START = time.time()
OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")

if OUT:
    os.makedirs(OUT, exist_ok=True)

torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
torch.manual_seed(1729)
np.random.seed(1729)

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
]
CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
FIELD_INDEX = {f: i for i, f in enumerate(FIELDS)}
HISTORY_LENGTH = 4
BATCH_SIZE = 32768
EPOCHS = 2


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    start_positions = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local_positions = (
        np.arange(n, dtype=np.float32)
        - start_positions.astype(np.float32)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(np.r_[-1, end_positions]).astype(np.float32)

    group_indices = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[group_indices] - 1.0, 1.0)

    result = np.empty(n, dtype=np.float32)
    result[order] = local_positions / denom
    return result


def clipped_field(split, name):
    card = int(FEATURE_CARDINALITIES[name])
    values = np.asarray(split.X[name], dtype=np.int64)
    # The failed predecessor allocated embeddings from observed training
    # maxima. Using the declared cardinality and mapping every invalid value
    # to the unseen id prevents validation/test index errors.
    valid = (values >= 0) & (values < card)
    return np.where(valid, values, 0).astype(np.int32)


def build_current_features(split):
    return [clipped_field(split, f) for f in FIELDS]


def build_causal_context(split, current):
    """
    Construct label-free, causal histories in (user_id, time_ms, row) order.
    Rows at the same timestamp retain original row order.
    """
    uid = np.asarray(split.user_id, dtype=np.int64)
    tm = np.asarray(split.time_ms, dtype=np.int64)
    n = len(uid)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, tm, uid))

    su = uid[order]
    st = tm[order]
    video_sorted = current[FIELD_INDEX["video_id"]][order]
    author_sorted = current[FIELD_INDEX["author_id"]][order]

    hist_video_sorted = np.zeros(
        (n, HISTORY_LENGTH), dtype=np.int32
    )
    hist_author_sorted = np.zeros(
        (n, HISTORY_LENGTH), dtype=np.int32
    )

    for lag in range(1, HISTORY_LENGTH + 1):
        if lag >= n:
            break
        same_user = su[lag:] == su[:-lag]
        dest = np.arange(lag, n, dtype=np.int64)
        valid_dest = dest[same_user]
        hist_video_sorted[valid_dest, lag - 1] = (
            video_sorted[:-lag][same_user]
        )
        hist_author_sorted[valid_dest, lag - 1] = (
            author_sorted[:-lag][same_user]
        )

    user_start = np.empty(n, dtype=bool)
    user_start[0] = True
    user_start[1:] = su[1:] != su[:-1]

    gap_ms = np.zeros(n, dtype=np.int64)
    gap_ms[1:] = np.maximum(st[1:] - st[:-1], 0)
    gap_ms[user_start] = 0

    session_start = user_start.copy()
    session_start[1:] |= gap_ms[1:] > 30 * 60 * 1000

    session_start_pos = np.maximum.accumulate(
        np.where(session_start, np.arange(n, dtype=np.int64), 0)
    )
    session_position = (
        np.arange(n, dtype=np.int64) - session_start_pos
    ).astype(np.float32)

    same_video = (
        (hist_video_sorted[:, 0] != 0)
        & (hist_video_sorted[:, 0] == video_sorted)
    ).astype(np.float32)
    same_author = (
        (hist_author_sorted[:, 0] != 0)
        & (hist_author_sorted[:, 0] == author_sorted)
    ).astype(np.float32)

    log_gap = np.log1p(
        np.minimum(gap_ms.astype(np.float64), 24 * 3600 * 1000)
        / 1000.0
    ).astype(np.float32)
    log_gap /= np.float32(np.log1p(24 * 3600))

    log_session_position = np.log1p(
        np.minimum(session_position, 1024.0)
    ).astype(np.float32)
    log_session_position /= np.float32(np.log1p(1024.0))

    numeric_sorted = np.column_stack([
        log_gap,
        log_session_position,
        same_video,
        same_author,
    ]).astype(np.float32)

    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n, dtype=np.int64)

    hist_video = hist_video_sorted[inverse]
    hist_author = hist_author_sorted[inverse]
    numeric = numeric_sorted[inverse]

    return hist_video, hist_author, numeric


class TransitionMLP(nn.Module):
    """
    Ordered-history prediction: current categorical embeddings and the last
    four video/author embeddings are concatenated, so each lag has a distinct
    role in forming the score.
    """
    def __init__(self):
        super().__init__()
        dims = [12, 16, 16, 8, 5, 5, 6]
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim, padding_idx=0)
            for card, dim in zip(CARDS, dims)
        ])
        self.video_embedding = self.embeddings[FIELD_INDEX["video_id"]]
        self.author_embedding = self.embeddings[FIELD_INDEX["author_id"]]

        input_dim = (
            sum(dims)
            + HISTORY_LENGTH * dims[FIELD_INDEX["video_id"]]
            + HISTORY_LENGTH * dims[FIELD_INDEX["author_id"]]
            + 4
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.SiLU(),
            nn.LayerNorm(192),
            nn.Dropout(0.08),
            nn.Linear(192, 96),
            nn.SiLU(),
            nn.Linear(96, 1),
        )
        self._initialize()

    def _initialize(self):
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.025)
            with torch.no_grad():
                emb.weight[0].zero_()

    def forward(self, categorical, hist_video, hist_author, numeric):
        current_parts = [
            emb(categorical[:, j])
            for j, emb in enumerate(self.embeddings)
        ]
        hv = self.video_embedding(hist_video).reshape(
            categorical.shape[0], -1
        )
        ha = self.author_embedding(hist_author).reshape(
            categorical.shape[0], -1
        )
        x = torch.cat(current_parts + [hv, ha, numeric], dim=1)
        return self.network(x).squeeze(1)


class DINAttention(nn.Module):
    """
    A different score-forming family: current video/author embeddings are a
    query over causal history, and a learned attention-weighted context enters
    the scoring tower. Unlike concatenation, history contributions depend on
    query-history compatibility.
    """
    def __init__(self):
        super().__init__()
        self.user_embedding = nn.Embedding(CARDS[0], 12, padding_idx=0)
        self.video_embedding = nn.Embedding(CARDS[1], 16, padding_idx=0)
        self.author_embedding = nn.Embedding(CARDS[2], 16, padding_idx=0)
        self.tag_embedding = nn.Embedding(CARDS[3], 8, padding_idx=0)
        self.tab_embedding = nn.Embedding(CARDS[4], 5, padding_idx=0)
        self.duration_embedding = nn.Embedding(CARDS[5], 5, padding_idx=0)
        self.upload_embedding = nn.Embedding(CARDS[6], 6, padding_idx=0)

        self.attention = nn.Sequential(
            nn.Linear(32 * 4, 96),
            nn.SiLU(),
            nn.Linear(96, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        input_dim = 12 + 16 + 16 + 8 + 5 + 5 + 6 + 32 + 32 + 4
        self.network = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.SiLU(),
            nn.LayerNorm(192),
            nn.Dropout(0.08),
            nn.Linear(192, 96),
            nn.SiLU(),
            nn.Linear(96, 1),
        )
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.025)
                with torch.no_grad():
                    module.weight[0].zero_()

    def forward(self, categorical, hist_video, hist_author, numeric):
        user = self.user_embedding(categorical[:, 0])
        video = self.video_embedding(categorical[:, 1])
        author = self.author_embedding(categorical[:, 2])
        tag = self.tag_embedding(categorical[:, 3])
        tab = self.tab_embedding(categorical[:, 4])
        duration = self.duration_embedding(categorical[:, 5])
        upload = self.upload_embedding(categorical[:, 6])

        query = torch.cat([video, author], dim=1)
        history = torch.cat([
            self.video_embedding(hist_video),
            self.author_embedding(hist_author),
        ], dim=2)

        expanded_query = query[:, None, :].expand(
            -1, HISTORY_LENGTH, -1
        )
        attention_input = torch.cat([
            expanded_query,
            history,
            expanded_query - history,
            expanded_query * history,
        ], dim=2)

        logits = self.attention(attention_input).squeeze(2)
        mask = (hist_video != 0) | (hist_author != 0)
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1)
        weights = weights * mask.float()
        weights = weights / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        context = torch.sum(history * weights[:, :, None], dim=1)

        x = torch.cat([
            user,
            video,
            author,
            tag,
            tab,
            duration,
            upload,
            query,
            context,
            numeric,
        ], dim=1)
        return self.network(x).squeeze(1)


def batch_tensors(current, hist_video, hist_author, numeric, indices):
    cat = np.column_stack([x[indices] for x in current]).astype(
        np.int64, copy=False
    )
    return (
        torch.from_numpy(cat),
        torch.from_numpy(
            np.asarray(hist_video[indices], dtype=np.int64)
        ),
        torch.from_numpy(
            np.asarray(hist_author[indices], dtype=np.int64)
        ),
        torch.from_numpy(
            np.asarray(numeric[indices], dtype=np.float32)
        ),
    )


def train_model(model, current, hist_video, hist_author, numeric,
                labels, sample_weight, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0022, weight_decay=2e-6
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    n = len(labels)

    for epoch in range(EPOCHS):
        permutation = rng.permutation(n)
        total_loss = 0.0
        total_weight = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            cat, hv, ha, num = batch_tensors(
                current, hist_video, hist_author, numeric, idx
            )
            yb = torch.from_numpy(
                np.asarray(labels[idx], dtype=np.float32)
            )
            wb = torch.from_numpy(
                np.asarray(sample_weight[idx], dtype=np.float32)
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(cat, hv, ha, num)
            losses = criterion(logits, yb)
            loss = torch.sum(losses * wb) / torch.sum(wb).clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(torch.sum(losses * wb).detach())
            total_weight += float(torch.sum(wb))

        print(
            "FINDINGS model=%s epoch=%d weighted_logloss=%.6f"
            % (
                model.__class__.__name__,
                epoch + 1,
                total_loss / max(total_weight, 1.0),
            ),
            flush=True,
        )
    return model


def predict_model(model, current, hist_video, hist_author, numeric):
    model.eval()
    n = len(numeric)
    output = np.empty(n, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            idx = slice(start, end)
            cat = torch.from_numpy(
                np.column_stack(
                    [x[idx] for x in current]
                ).astype(np.int64, copy=False)
            )
            hv = torch.from_numpy(
                np.asarray(hist_video[idx], dtype=np.int64)
            )
            ha = torch.from_numpy(
                np.asarray(hist_author[idx], dtype=np.int64)
            )
            num = torch.from_numpy(
                np.asarray(numeric[idx], dtype=np.float32)
            )
            output[start:end] = torch.sigmoid(
                model(cat, hv, ha, num)
            ).numpy().astype(np.float32)
    return output


def make_model(name):
    if name == "transition_mlp":
        return TransitionMLP()
    if name == "din_attention":
        return DINAttention()
    raise ValueError(name)


inc_valid_path = (
    os.path.join(SHARED, "incumbent_valid_scores.npy")
    if SHARED else ""
)
inc_test_path = (
    os.path.join(SHARED, "incumbent_test_scores.npy")
    if SHARED else ""
)
if not (
    inc_valid_path
    and inc_test_path
    and os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise RuntimeError("Trusted incumbent artifacts are required")

train = load("train")
valid = load("valid")

train_current = build_current_features(train)
valid_current = build_current_features(valid)

train_hv, train_ha, train_num = build_causal_context(
    train, train_current
)
valid_hv, valid_ha, valid_num = build_causal_context(
    valid, valid_current
)

train_y = np.asarray(train.y, dtype=np.float32)
train_date = np.asarray(train.date, dtype=np.int32)

# Main-model sample weighting, rather than weighting only a side component.
age_days = int(train_date.max()) - train_date
sample_weight = np.exp(
    -np.log(2.0) * age_days.astype(np.float32) / 4.0
).astype(np.float32)
sample_weight /= np.mean(sample_weight)

valid_uid = np.asarray(valid.user_id, dtype=np.int64)
valid_y = np.asarray(valid.y)
inc_valid = np.asarray(
    np.load(inc_valid_path, mmap_mode="r"), dtype=np.float32
)
inc_rank = within_user_rank(valid_uid, inc_valid)
control_metrics = evaluate(valid_uid, valid_y, inc_rank)

candidate_results = {
    "trusted_incumbent": float(control_metrics["primary"])
}
best_primary = float(control_metrics["primary"])
best_scores = inc_rank.copy()
best_raw_scores = None
best_model_name = None
best_alpha = 0.0
best_state = None

model_specs = [
    ("transition_mlp", 1801),
    ("din_attention", 2903),
]
blend_alphas = [0.10, 0.25, 0.50, 0.75, 1.00]

for model_name, seed in model_specs:
    model = make_model(model_name)
    model = train_model(
        model,
        train_current,
        train_hv,
        train_ha,
        train_num,
        train_y,
        sample_weight,
        seed,
    )

    raw_valid = predict_model(
        model,
        valid_current,
        valid_hv,
        valid_ha,
        valid_num,
    )
    raw_rank = within_user_rank(valid_uid, raw_valid)
    raw_metrics = evaluate(valid_uid, valid_y, raw_rank)

    candidate_results[model_name + "_standalone"] = float(
        raw_metrics["primary"]
    )
    print(
        "FINDINGS family=%s standalone_primary=%.6f "
        "standalone_gauc=%.6f standalone_ndcg5=%.6f"
        % (
            model_name,
            float(raw_metrics["primary"]),
            float(raw_metrics["gauc"]),
            float(raw_metrics["ndcg@5"]),
        ),
        flush=True,
    )

    for alpha in blend_alphas:
        blended = (
            (1.0 - alpha) * inc_rank + alpha * raw_rank
        ).astype(np.float32)
        metrics = evaluate(valid_uid, valid_y, blended)
        candidate_name = "%s_blend_%.2f" % (model_name, alpha)
        candidate_results[candidate_name] = float(metrics["primary"])

        print(
            "FINDINGS candidate=%s primary=%.6f gauc=%.6f "
            "ndcg5=%.6f delta=%+.6f"
            % (
                candidate_name,
                float(metrics["primary"]),
                float(metrics["gauc"]),
                float(metrics["ndcg@5"]),
                float(metrics["primary"])
                - float(control_metrics["primary"]),
            ),
            flush=True,
        )

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_scores = blended.copy()
            best_raw_scores = raw_rank.copy()
            best_model_name = model_name
            best_alpha = float(alpha)
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    del model
    del raw_valid
    del raw_rank
    gc.collect()

final_metrics = evaluate(valid_uid, valid_y, best_scores)

print(
    "FINDINGS embedding_failure_diagnosis=observed-max-sized "
    "embeddings allowed unseen validation ids to exceed the table; "
    "declared cardinalities plus unseen-id clipping fixed the crash",
    flush=True,
)
print(
    "FINDINGS winner=%s alpha=%.2f control_primary=%.6f "
    "winner_primary=%.6f"
    % (
        best_model_name if best_model_name is not None
        else "trusted_incumbent",
        best_alpha,
        float(control_metrics["primary"]),
        float(final_metrics["primary"]),
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_results, sort_keys=True),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_raw_scores is not None:
        np.save(
            os.path.join(OUT, "scores_valid_raw.npy"),
            np.asarray(best_raw_scores, dtype=np.float64),
        )

del train
del valid
del train_current
del valid_current
del train_hv
del train_ha
del train_num
del valid_hv
del valid_ha
del valid_num
del train_y
del train_date
del sample_weight
del inc_valid
gc.collect()

test = load("test")
inc_test = np.asarray(
    np.load(inc_test_path, mmap_mode="r"), dtype=np.float32
)
inc_test_rank = within_user_rank(test.user_id, inc_test)

if best_model_name is None:
    test_scores = inc_test_rank
else:
    test_current = build_current_features(test)
    test_hv, test_ha, test_num = build_causal_context(
        test, test_current
    )

    selected_model = make_model(best_model_name)
    selected_model.load_state_dict(best_state)
    selected_raw = predict_model(
        selected_model,
        test_current,
        test_hv,
        test_ha,
        test_num,
    )
    selected_rank = within_user_rank(test.user_id, selected_raw)
    test_scores = (
        (1.0 - best_alpha) * inc_test_rank
        + best_alpha * selected_rank
    ).astype(np.float32)

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print(
    "METRICS "
    + json.dumps({
        "primary": float(final_metrics["primary"]),
        "gauc": float(final_metrics["gauc"]),
        "ndcg@5": float(final_metrics["ndcg@5"]),
        "gpu_seconds": elapsed,
    })
)