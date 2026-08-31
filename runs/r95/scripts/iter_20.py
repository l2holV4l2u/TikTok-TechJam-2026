import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 73129
BATCH = 8192
PRED_BATCH = 32768
HIST_LEN = 6

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

train = load("train")
valid = load("valid")
test = load("test")

ytr_np = np.asarray(train.y, dtype=np.float32)

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "user_active_degree",
    "onehot_feat3",
    "onehot_feat8",
]
CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
N_FIELDS = len(FIELDS)
VIDEO_INDEX = FIELDS.index("video_id")
AUTHOR_INDEX = FIELDS.index("author_id")


def current_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[f], dtype=np.int64) for f in FIELDS
        ]),
        dtype=np.int64,
    )


def causal_context(split):
    """
    Every contextual value for a row is constructed only from impressions
    earlier in that user's time ordering. Labels and auxiliary outcomes are
    never consulted.
    """
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    videos = np.asarray(split.X["video_id"], dtype=np.int64)
    authors = np.asarray(split.X["author_id"], dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    n = len(users)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, times, users))
    su = users[order]
    sv = videos[order]
    sa = authors[order]
    st = times[order]

    new_user = np.r_[True, su[1:] != su[:-1]]
    starts = np.flatnonzero(new_user)
    sizes = np.diff(np.r_[starts, n])
    user_position = np.arange(n, dtype=np.int64) - np.repeat(starts, sizes)

    prev_gap = np.zeros(n, dtype=np.float32)
    same = ~new_user
    prev_gap[same] = np.clip(
        (st[same] - st[np.flatnonzero(same) - 1]) / 1000.0,
        0.0,
        86400.0 * 7.0,
    ).astype(np.float32)

    session_reset = new_user.copy()
    session_reset[1:] |= (
        (st[1:] - st[:-1]) > 30 * 60 * 1000
    )
    reset_indices = np.where(session_reset, np.arange(n), 0)
    last_reset = np.maximum.accumulate(reset_indices)
    session_position = np.arange(n, dtype=np.int64) - last_reset

    hv = np.zeros((n, HIST_LEN), dtype=np.int64)
    ha = np.zeros((n, HIST_LEN), dtype=np.int64)
    for lag in range(1, HIST_LEN + 1):
        dst = np.arange(lag, n)
        ok = su[dst] == su[dst - lag]
        d = dst[ok]
        hv[d, lag - 1] = sv[d - lag]
        ha[d, lag - 1] = sa[d - lag]

    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n, dtype=np.int64)

    return {
        "hist_video": np.ascontiguousarray(hv[inv]),
        "hist_author": np.ascontiguousarray(ha[inv]),
        "log_gap": np.log1p(prev_gap[inv]).astype(np.float32),
        "user_position": np.log1p(user_position[inv]).astype(np.float32),
        "session_position": np.log1p(session_position[inv]).astype(np.float32),
    }


xtr_np = current_matrix(train)
xva_np = current_matrix(valid)
xte_np = current_matrix(test)

ctr = causal_context(train)
cva = causal_context(valid)
cte = causal_context(test)

print(
    "FINDINGS exposure_history_nonempty=" +
    json.dumps({
        "train": float(np.mean(np.any(ctr["hist_video"] != 0, axis=1))),
        "valid": float(np.mean(np.any(cva["hist_video"] != 0, axis=1))),
        "test": float(np.mean(np.any(cte["hist_video"] != 0, axis=1))),
    }, sort_keys=True),
    flush=True,
)

last_date = int(np.max(np.asarray(train.date, dtype=np.int64)))
age = (
    last_date - np.asarray(train.date, dtype=np.int64)
).astype(np.float32)
recency_weight = np.exp2(-age / 4.0).astype(np.float32)
recency_weight /= float(recency_weight.mean())


def tree_matrix(x, context):
    # Candidate fields, three immediate exposure transitions, and causal
    # session state. LightGBM can form discontinuous transition rules.
    cols = [x[:, j].astype(np.float32) for j in range(N_FIELDS)]
    for lag in range(3):
        cols.append(context["hist_video"][:, lag].astype(np.float32))
        cols.append(context["hist_author"][:, lag].astype(np.float32))
    cols.extend([
        context["log_gap"],
        context["user_position"],
        context["session_position"],
    ])
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)


tree_tr = tree_matrix(xtr_np, ctr)
tree_va = tree_matrix(xva_np, cva)
tree_te = tree_matrix(xte_np, cte)

categorical_indices = list(range(N_FIELDS + 6))
dtrain = lgb.Dataset(
    tree_tr,
    label=ytr_np,
    weight=recency_weight,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)

tree_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "min_data_in_leaf": 700,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.86,
    "bagging_freq": 1,
    "lambda_l1": 0.15,
    "lambda_l2": 3.0,
    "max_bin": 63,
    "max_cat_to_onehot": 16,
    "cat_smooth": 30.0,
    "cat_l2": 15.0,
    "verbosity": -1,
    "verbose": -1,
    "seed": SEED,
    "num_threads": max(1, min(8, os.cpu_count() or 1)),
}

tree_model = lgb.train(
    tree_params,
    dtrain,
    num_boost_round=190,
)

tree_valid = tree_model.predict(tree_va).astype(np.float64)
tree_test = tree_model.predict(tree_te).astype(np.float64)

del tree_tr, tree_va, tree_te, dtrain


class ExposureTransformer(nn.Module):
    """
    Encodes prior displayed candidates rather than prior positive outcomes.
    Thus it can represent within-session topic continuity and fatigue even for
    users whose train history contains no positive long-view event.
    """
    def __init__(self, dim=16):
        super().__init__()
        self.current_embeddings = nn.ModuleList([
            nn.Embedding(card, dim, padding_idx=0) for card in CARDS
        ])
        self.linear_embeddings = nn.ModuleList([
            nn.Embedding(card, 1, padding_idx=0) for card in CARDS
        ])

        self.video_embedding = self.current_embeddings[VIDEO_INDEX]
        self.author_embedding = self.current_embeddings[AUTHOR_INDEX]
        self.position = nn.Parameter(torch.randn(HIST_LEN, dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=2,
            dim_feedforward=48,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.attention = nn.Sequential(
            nn.Linear(4 * dim, 40),
            nn.ReLU(),
            nn.Linear(40, 1),
        )

        deep_width = N_FIELDS * dim + 3 * dim + 3
        self.deep = nn.Sequential(
            nn.Linear(deep_width, 112),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(112, 40),
            nn.ReLU(),
            nn.Linear(40, 1),
        )
        self.bias = nn.Parameter(torch.zeros(()))

        for emb in self.current_embeddings:
            nn.init.normal_(emb.weight, std=0.025)
            with torch.no_grad():
                emb.weight[0].zero_()
        for emb in self.linear_embeddings:
            nn.init.zeros_(emb.weight)

    def forward(self, x, hist_video, hist_author, numeric_context):
        current = torch.stack(
            [self.current_embeddings[j](x[:, j])
             for j in range(N_FIELDS)],
            dim=1,
        )
        wide = torch.stack(
            [self.linear_embeddings[j](x[:, j]).squeeze(-1)
             for j in range(N_FIELDS)],
            dim=1,
        ).sum(dim=1) + self.bias

        # Context arrays are newest-first; reverse them for chronological
        # positional encoding.
        hv = torch.flip(hist_video, dims=[1])
        ha = torch.flip(hist_author, dims=[1])
        mask = hv == 0

        tokens = (
            self.video_embedding(hv)
            + self.author_embedding(ha)
            + self.position.unsqueeze(0)
        )
        encoded = self.encoder(tokens, src_key_padding_mask=mask)

        candidate = (
            current[:, VIDEO_INDEX] + current[:, AUTHOR_INDEX]
        )
        q = candidate.unsqueeze(1).expand_as(encoded)
        att_input = torch.cat(
            [q, encoded, q - encoded, q * encoded],
            dim=2,
        )
        att_logits = self.attention(att_input).squeeze(-1)
        att_logits = att_logits.masked_fill(mask, -1.0e9)
        att_weight = torch.softmax(att_logits, dim=1)
        att_weight = att_weight.masked_fill(mask, 0.0)
        att_weight = att_weight / att_weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-8)
        pooled = (
            encoded * att_weight.unsqueeze(-1)
        ).sum(dim=1)

        last_state = encoded[:, -1]
        no_history = mask.all(dim=1, keepdim=True)
        last_state = last_state.masked_fill(no_history, 0.0)

        deep_input = torch.cat([
            current.flatten(1),
            pooled,
            last_state,
            candidate * pooled,
            numeric_context,
        ], dim=1)
        return wide + self.deep(deep_input).squeeze(-1)


xtr = torch.from_numpy(xtr_np)
hvt = torch.from_numpy(ctr["hist_video"])
hat = torch.from_numpy(ctr["hist_author"])
nct = torch.from_numpy(np.column_stack([
    ctr["log_gap"],
    ctr["user_position"],
    ctr["session_position"],
]).astype(np.float32))
yt = torch.from_numpy(ytr_np)
wt = torch.from_numpy(recency_weight)

sequence_model = ExposureTransformer(dim=16)
optimizer = torch.optim.AdamW(
    sequence_model.parameters(),
    lr=1.4e-3,
    weight_decay=1.0e-5,
)
generator = torch.Generator().manual_seed(SEED + 19)
ntrain = len(ytr_np)

for epoch in range(2):
    sequence_model.train()
    permutation = torch.randperm(ntrain, generator=generator)
    total_loss = 0.0
    for start in range(0, ntrain, BATCH):
        idx = permutation[start:min(start + BATCH, ntrain)]
        logits = sequence_model(
            xtr.index_select(0, idx),
            hvt.index_select(0, idx),
            hat.index_select(0, idx),
            nct.index_select(0, idx),
        )
        target = yt.index_select(0, idx)
        weight = wt.index_select(0, idx)
        per_row = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        loss = (per_row * weight).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            sequence_model.parameters(), 5.0
        )
        optimizer.step()
        total_loss += float(loss.detach()) * len(idx)

    print(
        "TRAIN exposure_transformer epoch=%d loss=%.6f" %
        (epoch + 1, total_loss / ntrain),
        flush=True,
    )


def neural_predict(model, x_np, context):
    result = np.empty(len(x_np), dtype=np.float32)
    numeric = np.column_stack([
        context["log_gap"],
        context["user_position"],
        context["session_position"],
    ]).astype(np.float32)

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x_np), PRED_BATCH):
            end = min(start + PRED_BATCH, len(x_np))
            result[start:end] = model(
                torch.from_numpy(x_np[start:end]),
                torch.from_numpy(context["hist_video"][start:end]),
                torch.from_numpy(context["hist_author"][start:end]),
                torch.from_numpy(numeric[start:end]),
            ).cpu().numpy()
    return result.astype(np.float64)


sequence_valid = neural_predict(sequence_model, xva_np, cva)
sequence_test = neural_predict(sequence_model, xte_np, cte)


def within_user_rank(user_ids, scores):
    """
    Scale-free rank aggregation. Scores are effectively continuous, so stable
    row-order tie breaking affects only otherwise unresolved ties.
    """
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    rows = np.arange(len(users), dtype=np.int64)
    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]

    starts = np.r_[0, np.flatnonzero(
        sorted_users[1:] != sorted_users[:-1]
    ) + 1]
    ends = np.r_[starts[1:], len(users)]
    sizes = ends - starts

    rank_sorted = (
        np.arange(len(users), dtype=np.float64)
        - np.repeat(starts, sizes)
    )
    denom = np.maximum(np.repeat(sizes, sizes) - 1, 1)
    rank_sorted /= denom

    ranked = np.empty(len(users), dtype=np.float64)
    ranked[order] = rank_sorted
    return ranked


shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not (
    os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(
    np.load(inc_valid_path), dtype=np.float64
)
inc_test = np.asarray(
    np.load(inc_test_path), dtype=np.float64
)

valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)
valid_labels = np.asarray(valid.y, dtype=np.int8)

rv_inc = within_user_rank(valid_users, inc_valid)
rv_tree = within_user_rank(valid_users, tree_valid)
rv_seq = within_user_rank(valid_users, sequence_valid)

rt_inc = within_user_rank(test_users, inc_test)
rt_tree = within_user_rank(test_users, tree_test)
rt_seq = within_user_rank(test_users, sequence_test)

candidate_scores = {
    "incumbent": inc_valid,
    "context_lgbm": tree_valid,
    "exposure_transformer": sequence_valid,
    "new_models_rank_mean": 0.5 * rv_tree + 0.5 * rv_seq,
}

candidate_test = {
    "incumbent": inc_test,
    "context_lgbm": tree_test,
    "exposure_transformer": sequence_test,
    "new_models_rank_mean": 0.5 * rt_tree + 0.5 * rt_seq,
}

# Fixed convex candidates. Validation chooses among complete trained systems;
# exactly the corresponding fixed weights are applied to hidden test.
for alpha in (0.25, 0.50, 0.75):
    name = "inc_tree_a%.2f" % alpha
    candidate_scores[name] = alpha * rv_inc + (1.0 - alpha) * rv_tree
    candidate_test[name] = alpha * rt_inc + (1.0 - alpha) * rt_tree

    name = "inc_seq_a%.2f" % alpha
    candidate_scores[name] = alpha * rv_inc + (1.0 - alpha) * rv_seq
    candidate_test[name] = alpha * rt_inc + (1.0 - alpha) * rt_seq

for wi, wt2, ws in (
    (0.50, 0.25, 0.25),
    (0.60, 0.20, 0.20),
    (0.40, 0.30, 0.30),
    (0.50, 0.35, 0.15),
    (0.50, 0.15, 0.35),
):
    name = "tri_%.2f_%.2f_%.2f" % (wi, wt2, ws)
    candidate_scores[name] = (
        wi * rv_inc + wt2 * rv_tree + ws * rv_seq
    )
    candidate_test[name] = (
        wi * rt_inc + wt2 * rt_tree + ws * rt_seq
    )

candidate_metrics = {}
best_name = None
best_metric = None
for name, scores in candidate_scores.items():
    metric = evaluate(valid_users, valid_labels, scores)
    candidate_metrics[name] = float(metric["primary"])
    if (
        best_metric is None
        or metric["primary"] > best_metric["primary"]
    ):
        best_name = name
        best_metric = metric

valid_scores = candidate_scores[best_name]
test_scores = candidate_test[best_name]

print(
    "CANDIDATES " + json.dumps(
        candidate_metrics, sort_keys=True
    ),
    flush=True,
)
print(
    "FINDINGS selected=%s tree_sequence_corr=%.6f" % (
        best_name,
        float(np.corrcoef(rv_tree, rv_seq)[0, 1]),
    ),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

    # The equal-rank aggregate is the script's own prediction without the
    # externally supplied incumbent.
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(
            0.5 * rv_tree + 0.5 * rv_seq,
            dtype=np.float64,
        ),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.4f}' %
    (
        float(best_metric["primary"]),
        float(best_metric["gauc"]),
        float(best_metric["ndcg@5"]),
        float(elapsed),
    )
)