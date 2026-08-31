import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 73129
HIST_LEN = 12
BATCH = 4096
PRED_BATCH = 16384
EPOCHS = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "hour",
    "upload_type",
    "user_active_degree",
]
CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
VIDEO_POS = FIELDS.index("video_id")
USER_CARD = int(FEATURE_CARDINALITIES["user_id"])
VIDEO_CARD = int(FEATURE_CARDINALITIES["video_id"])

train = load("train")
valid = load("valid")
test = load("test")


def make_current(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[f], dtype=np.int64) for f in FIELDS
        ]),
        dtype=np.int64,
    )


xtr_np = make_current(train)
xva_np = make_current(valid)
xte_np = make_current(test)
ytr_np = np.asarray(train.y, dtype=np.float32)
utr_np = np.asarray(train.user_id, dtype=np.int64)
uva_np = np.asarray(valid.user_id, dtype=np.int64)
ute_np = np.asarray(test.user_id, dtype=np.int64)
vtr_np = np.asarray(train.video_id, dtype=np.int64)

# Build strictly causal positive-video histories for train rows. Sorting by
# (user_id, time_ms, row_position) is the only valid interaction ordering.
ntr = len(ytr_np)
row_position = np.arange(ntr, dtype=np.int64)
order = np.lexsort((
    row_position,
    np.asarray(train.time_ms, dtype=np.int64),
    utr_np,
))
sorted_users = utr_np[order]
sorted_y = ytr_np[order].astype(np.int64)

starts = np.flatnonzero(
    np.r_[True, sorted_users[1:] != sorted_users[:-1]]
)
ends = np.r_[starts[1:], ntr]
group_lengths = ends - starts

global_cs = np.cumsum(sorted_y, dtype=np.int64)
group_base_values = global_cs[starts] - sorted_y[starts]
group_base = np.repeat(group_base_values, group_lengths)
prior_positive_count_sorted = global_cs - group_base - sorted_y

prior_positive_count = np.empty(ntr, dtype=np.int64)
prior_positive_count[order] = prior_positive_count_sorted

positive_rows_sorted = order[sorted_y == 1]
positive_counts = np.bincount(
    utr_np[positive_rows_sorted], minlength=USER_CARD
).astype(np.int64)
positive_offsets = np.zeros(USER_CARD + 1, dtype=np.int64)
np.cumsum(positive_counts, out=positive_offsets[1:])

# Histories are left padded and chronological, oldest to newest.
htr_np = np.zeros((ntr, HIST_LEN), dtype=np.int32)
for col in range(HIST_LEN):
    lag = HIST_LEN - col
    usable = prior_positive_count >= lag
    rows = np.flatnonzero(usable)
    pos_index = (
        positive_offsets[utr_np[rows]]
        + prior_positive_count[rows]
        - lag
    )
    source_rows = positive_rows_sorted[pos_index]
    htr_np[rows, col] = vtr_np[source_rows].astype(np.int32) + 1


def static_train_history(split_users):
    split_users = np.asarray(split_users, dtype=np.int64)
    result = np.zeros((len(split_users), HIST_LEN), dtype=np.int32)
    counts = positive_counts[split_users]

    for col in range(HIST_LEN):
        lag = HIST_LEN - col
        usable = counts >= lag
        rows = np.flatnonzero(usable)
        pos_index = positive_offsets[split_users[rows]] + counts[rows] - lag
        source_rows = positive_rows_sorted[pos_index]
        result[rows, col] = vtr_np[source_rows].astype(np.int32) + 1
    return result


hva_np = static_train_history(uva_np)
hte_np = static_train_history(ute_np)

hist_count_tr_np = np.minimum(prior_positive_count, HIST_LEN).astype(np.float32)
hist_count_va_np = np.minimum(positive_counts[uva_np], HIST_LEN).astype(np.float32)
hist_count_te_np = np.minimum(positive_counts[ute_np], HIST_LEN).astype(np.float32)

# Main-model sample weights combine temporal proximity with a partial
# user-balanced correction. The latter prevents prolific train users from
# completely determining a metric whose nDCG component averages over users.
last_date = int(np.max(np.asarray(train.date, dtype=np.int64)))
day_age = (
    last_date - np.asarray(train.date, dtype=np.int64)
).astype(np.float32)
recency = np.exp2(-day_age / 4.0).astype(np.float32)

rows_per_user = np.bincount(
    utr_np, minlength=USER_CARD
).astype(np.float32)
user_balance = 1.0 / np.sqrt(np.maximum(rows_per_user[utr_np], 1.0))
user_balance /= np.mean(user_balance)
sample_weight_np = recency * (0.5 + 0.5 * user_balance)
sample_weight_np /= np.mean(sample_weight_np)
sample_weight_np = sample_weight_np.astype(np.float32)

xtr = torch.from_numpy(xtr_np)
htr = torch.from_numpy(htr_np.astype(np.int64, copy=False))
ytr = torch.from_numpy(ytr_np)
wtr = torch.from_numpy(sample_weight_np)
ctr = torch.from_numpy(hist_count_tr_np)


class ContextBackbone(nn.Module):
    def __init__(self, emb_dim=12):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, emb_dim) for card in CARDS
        ])
        self.biases = nn.ModuleList([
            nn.Embedding(card, 1) for card in CARDS
        ])
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.025)
        for bias in self.biases:
            nn.init.zeros_(bias.weight)

        self.context_net = nn.Sequential(
            nn.Linear(len(FIELDS) * emb_dim + 1, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.global_bias = nn.Parameter(torch.zeros(()))

    def field_embeddings(self, x):
        return torch.stack(
            [self.embeddings[j](x[:, j]) for j in range(len(FIELDS))],
            dim=1,
        )

    def context_score(self, x, hist_count):
        e = self.field_embeddings(x)
        log_count = torch.log1p(hist_count).unsqueeze(1) / np.log1p(HIST_LEN)
        deep = self.context_net(
            torch.cat([e.flatten(1), log_count], dim=1)
        ).squeeze(1)
        wide = torch.stack(
            [
                self.biases[j](x[:, j]).squeeze(1)
                for j in range(len(FIELDS))
            ],
            dim=1,
        ).sum(dim=1)
        return deep + wide + self.global_bias, e


class CandidateTransformer(nn.Module):
    """
    The candidate token and causal train-only history tokens are contextualized
    jointly. A deterministic frequency gate suppresses sequence residuals when
    little evidence exists, leaving the stationary context predictor intact.
    """
    def __init__(self, emb_dim=12, seq_dim=32):
        super().__init__()
        self.context = ContextBackbone(emb_dim)
        self.history_video = nn.Embedding(
            VIDEO_CARD + 1, seq_dim, padding_idx=0
        )
        nn.init.normal_(self.history_video.weight, std=0.025)
        with torch.no_grad():
            self.history_video.weight[0].zero_()

        self.candidate_projection = nn.Sequential(
            nn.Linear(emb_dim * 3, seq_dim),
            nn.ReLU(),
        )
        self.position = nn.Parameter(
            torch.zeros(1, HIST_LEN + 1, seq_dim)
        )
        nn.init.normal_(self.position, std=0.015)

        layer = nn.TransformerEncoderLayer(
            d_model=seq_dim,
            nhead=4,
            dim_feedforward=64,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.sequence_head = nn.Sequential(
            nn.LayerNorm(seq_dim),
            nn.Linear(seq_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.sequence_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, history, hist_count):
        base, fields = self.context.context_score(x, hist_count)
        candidate = self.candidate_projection(
            torch.cat([
                fields[:, VIDEO_POS, :],
                fields[:, FIELDS.index("author_id"), :],
                fields[:, FIELDS.index("tag"), :],
            ], dim=1)
        ).unsqueeze(1)

        history_tokens = self.history_video(history)
        tokens = torch.cat([candidate, history_tokens], dim=1)
        tokens = tokens + self.position

        padding = history.eq(0)
        token_padding = torch.cat([
            torch.zeros(
                (len(x), 1), dtype=torch.bool, device=x.device
            ),
            padding,
        ], dim=1)
        encoded = self.encoder(
            tokens, src_key_padding_mask=token_padding
        )
        residual = self.sequence_head(encoded[:, 0, :]).squeeze(1)

        # Approximately 0, .33, .55, .70, .80 as evidence grows.
        gate = hist_count / (hist_count + 2.0)
        return base + self.sequence_scale * gate * residual


class GRUInterestEvolution(nn.Module):
    """
    A distinct recurrent family evolves the ordered positive-video sequence.
    Its final state is matched to the current video/author/tag representation.
    """
    def __init__(self, emb_dim=12, seq_dim=32):
        super().__init__()
        self.context = ContextBackbone(emb_dim)
        self.history_video = nn.Embedding(
            VIDEO_CARD + 1, seq_dim, padding_idx=0
        )
        nn.init.normal_(self.history_video.weight, std=0.025)
        with torch.no_grad():
            self.history_video.weight[0].zero_()

        self.gru = nn.GRU(
            input_size=seq_dim,
            hidden_size=seq_dim,
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(emb_dim * 3, seq_dim),
            nn.Tanh(),
        )
        self.residual_net = nn.Sequential(
            nn.Linear(seq_dim * 3, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.sequence_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, history, hist_count):
        base, fields = self.context.context_score(x, hist_count)
        seq = self.history_video(history)
        outputs, _ = self.gru(seq)

        # Histories are left padded, so the final recurrent output always
        # follows the newest available positive event.
        state = outputs[:, -1, :]
        candidate = self.candidate_projection(
            torch.cat([
                fields[:, VIDEO_POS, :],
                fields[:, FIELDS.index("author_id"), :],
                fields[:, FIELDS.index("tag"), :],
            ], dim=1)
        )
        residual = self.residual_net(
            torch.cat([
                state,
                candidate,
                state * candidate,
            ], dim=1)
        ).squeeze(1)
        gate = hist_count / (hist_count + 2.0)
        return base + self.sequence_scale * gate * residual


def fit_model(model, name, seed_offset):
    generator = torch.Generator().manual_seed(SEED + seed_offset)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.3e-3, weight_decay=1e-6
    )
    n = len(ytr)

    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        total_loss = 0.0

        for st in range(0, n, BATCH):
            idx = permutation[st:min(st + BATCH, n)]
            xb = xtr.index_select(0, idx)
            hb = htr.index_select(0, idx)
            cb = ctr.index_select(0, idx)
            yb = ytr.index_select(0, idx)
            wb = wtr.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, hb, cb)
            row_loss = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = (row_loss * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(idx)

        print(
            "TRAIN %s epoch=%d loss=%.6f" %
            (name, epoch + 1, total_loss / n),
            flush=True,
        )


def predict(model, x_np, history_np, count_np):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float32)
    with torch.inference_mode():
        for st in range(0, len(x_np), PRED_BATCH):
            en = min(st + PRED_BATCH, len(x_np))
            xb = torch.from_numpy(x_np[st:en])
            hb = torch.from_numpy(
                history_np[st:en].astype(np.int64, copy=False)
            )
            cb = torch.from_numpy(count_np[st:en])
            result[st:en] = (
                model(xb, hb, cb).detach().cpu().numpy()
            )
    return result


transformer = CandidateTransformer()
fit_model(transformer, "candidate_transformer", 101)
trf_valid = predict(transformer, xva_np, hva_np, hist_count_va_np)
trf_test = predict(transformer, xte_np, hte_np, hist_count_te_np)
del transformer

gru = GRUInterestEvolution()
fit_model(gru, "gru_interest_evolution", 503)
gru_valid = predict(gru, xva_np, hva_np, hist_count_va_np)
gru_test = predict(gru, xte_np, hte_np, hist_count_te_np)
del gru

valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(shared, "incumbent_test_scores.npy")),
    dtype=np.float64,
)


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    # Row position resolves exact score ties deterministically.
    rows = np.arange(len(scores), dtype=np.int64)
    order = np.lexsort((rows, scores, users))
    sorted_users_local = users[order]
    starts_local = np.flatnonzero(
        np.r_[True, sorted_users_local[1:] != sorted_users_local[:-1]]
    )
    ends_local = np.r_[starts_local[1:], len(order)]
    lengths = ends_local - starts_local

    positions = (
        np.arange(len(order), dtype=np.float64)
        - np.repeat(starts_local, lengths)
    )
    denominators = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    ranked_sorted = positions / denominators

    ranked = np.empty(len(order), dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


models_valid = {
    "transformer": np.asarray(trf_valid, dtype=np.float64),
    "gru": np.asarray(gru_valid, dtype=np.float64),
}
models_test = {
    "transformer": np.asarray(trf_test, dtype=np.float64),
    "gru": np.asarray(gru_test, dtype=np.float64),
}

inc_rank_valid = within_user_rank(valid_users, inc_valid)
inc_rank_test = within_user_rank(test_users, inc_test)

candidate_scores = {}
best_primary = -np.inf
best_name = None
best_valid = None
best_test = None
best_raw = None
best_metrics = None

for family in ["transformer", "gru"]:
    own_valid = models_valid[family]
    own_test = models_test[family]
    own_rank_valid = within_user_rank(valid_users, own_valid)
    own_rank_test = within_user_rank(test_users, own_test)

    raw_metrics = evaluate(valid_users, valid_y, own_valid)
    raw_name = family + "_standalone"
    candidate_scores[raw_name] = float(raw_metrics["primary"])

    if float(raw_metrics["primary"]) > best_primary:
        best_primary = float(raw_metrics["primary"])
        best_name = raw_name
        best_valid = own_valid.copy()
        best_test = own_test.copy()
        best_raw = own_valid.copy()
        best_metrics = raw_metrics

    # Borda blending removes arbitrary logit-scale differences and optimizes
    # only the within-user ordering used by both target metrics.
    for alpha in [0.15, 0.30, 0.50, 0.70]:
        blend_valid = (
            (1.0 - alpha) * inc_rank_valid
            + alpha * own_rank_valid
        )
        blend_test = (
            (1.0 - alpha) * inc_rank_test
            + alpha * own_rank_test
        )
        name = family + "_borda_%.2f" % alpha
        metrics = evaluate(valid_users, valid_y, blend_valid)
        primary = float(metrics["primary"])
        candidate_scores[name] = primary

        if primary > best_primary:
            best_primary = primary
            best_name = name
            best_valid = blend_valid.copy()
            best_test = blend_test.copy()
            best_raw = own_valid.copy()
            best_metrics = metrics

print(
    "FINDINGS history_coverage valid_zero=%.4f valid_ge3=%.4f "
    "test_zero=%.4f test_ge3=%.4f winner=%s" %
    (
        float(np.mean(hist_count_va_np == 0)),
        float(np.mean(hist_count_va_np >= 3)),
        float(np.mean(hist_count_te_np == 0)),
        float(np.mean(hist_count_te_np >= 3)),
        best_name,
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(
        candidate_scores, sort_keys=True
    ),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    if "_borda_" in best_name:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
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