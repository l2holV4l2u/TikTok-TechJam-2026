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
SEED = 19427
BATCH = 8192
PAIR_BATCH = 4096
PRED_BATCH = 32768

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

# Both trained families receive the same overall inputs. The two-tower model
# partitions them according to which side of the interaction they describe.
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "hour",
    "duration_bucket",
    "tag",
    "upload_type",
    "user_active_degree",
    "onehot_feat3",
]
QUERY_POS = [0, 3, 4, 8]
ITEM_POS = [1, 2, 5, 6, 7, 9]
CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
N_FIELDS = len(FIELDS)

train = load("train")
valid = load("valid")
test = load("test")


def make_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[f], dtype=np.int64) for f in FIELDS
        ]),
        dtype=np.int64,
    )


xtr_np = make_matrix(train)
xva_np = make_matrix(valid)
xte_np = make_matrix(test)
ytr_np = np.asarray(train.y, dtype=np.float32)
utr_np = np.asarray(train.user_id, dtype=np.int64)

xtr = torch.from_numpy(xtr_np)
ytr = torch.from_numpy(ytr_np)

# Main-model recency weighting: a four-day half-life emphasizes the portion of
# training nearest the evaluation period while retaining all train examples.
last_date = int(np.max(np.asarray(train.date, dtype=np.int64)))
day_age = (last_date - np.asarray(train.date, dtype=np.int64)).astype(np.float32)
recency_np = np.exp2(-day_age / 4.0).astype(np.float32)
recency_np /= float(np.mean(recency_np))
recency = torch.from_numpy(recency_np)


class CategoricalBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(c, dim) for c in CARDS
        ])
        self.biases = nn.ModuleList([
            nn.Embedding(c, 1) for c in CARDS
        ])
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.025)
        for bias in self.biases:
            nn.init.zeros_(bias.weight)

    def dense(self, x):
        return torch.stack(
            [self.embeddings[j](x[:, j]) for j in range(N_FIELDS)],
            dim=1,
        )

    def wide(self, x):
        return torch.stack(
            [self.biases[j](x[:, j]).squeeze(-1)
             for j in range(N_FIELDS)],
            dim=1,
        ).sum(dim=1)


class TwoTowerMetric(nn.Module):
    """
    Query and item/context representations are formed independently and scored
    by cosine-like dot-product similarity. Additive field biases retain strong
    marginal effects while the normalized vectors focus training on geometry.
    """

    def __init__(self, emb_dim=12, tower_dim=48):
        super().__init__()
        self.base = CategoricalBlock(emb_dim)
        self.query_net = nn.Sequential(
            nn.Linear(len(QUERY_POS) * emb_dim, 96),
            nn.ReLU(),
            nn.Linear(96, tower_dim),
        )
        self.item_net = nn.Sequential(
            nn.Linear(len(ITEM_POS) * emb_dim, 96),
            nn.ReLU(),
            nn.Linear(96, tower_dim),
        )
        self.log_scale = nn.Parameter(torch.tensor(2.0))
        self.global_bias = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        e = self.base.dense(x)
        q = self.query_net(e[:, QUERY_POS, :].flatten(1))
        v = self.item_net(e[:, ITEM_POS, :].flatten(1))
        q = F.normalize(q, p=2, dim=1)
        v = F.normalize(v, p=2, dim=1)
        scale = self.log_scale.exp().clamp(max=30.0)
        interaction = scale * (q * v).sum(dim=1)
        return interaction + self.base.wide(x) + self.global_bias


class ProductNeuralNetwork(nn.Module):
    """
    PNN exposes all pairwise embedding inner products directly to an MLP.
    This differs from metric learning because interactions are jointly formed
    across every field and optimized with pointwise logistic supervision.
    """

    def __init__(self, emb_dim=10):
        super().__init__()
        self.base = CategoricalBlock(emb_dim)
        pair_i = []
        pair_j = []
        for i in range(N_FIELDS):
            for j in range(i + 1, N_FIELDS):
                pair_i.append(i)
                pair_j.append(j)
        self.register_buffer("pair_i", torch.tensor(pair_i, dtype=torch.long))
        self.register_buffer("pair_j", torch.tensor(pair_j, dtype=torch.long))
        width = N_FIELDS * emb_dim + len(pair_i)
        self.net = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.global_bias = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        e = self.base.dense(x)
        products = (
            e.index_select(1, self.pair_i) *
            e.index_select(1, self.pair_j)
        ).sum(dim=2)
        deep_input = torch.cat([e.flatten(1), products], dim=1)
        return (
            self.base.wide(x)
            + self.net(deep_input).squeeze(-1)
            + self.global_bias
        )


def fit_pointwise(model, name, epochs=3, lr=1.2e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(ytr)
    generator = torch.Generator().manual_seed(SEED + 311)

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        loss_total = 0.0

        for st in range(0, n, BATCH):
            idx = permutation[st:min(st + BATCH, n)]
            xb = xtr.index_select(0, idx)
            yb = ytr.index_select(0, idx)
            wb = recency.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            row_loss = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = (row_loss * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += float(loss.detach()) * len(idx)

        print(
            "TRAIN %s epoch=%d loss=%.6f" %
            (name, epoch + 1, loss_total / n),
            flush=True,
        )


def make_pair_sampler():
    """
    Build vectorized group offsets once. Each epoch samples several negatives
    per positive from the same user; no Python loop over users or rows is used.
    """
    pos_rows = np.flatnonzero(ytr_np > 0.5).astype(np.int64)
    neg_rows = np.flatnonzero(ytr_np < 0.5).astype(np.int64)

    neg_order = np.argsort(utr_np[neg_rows], kind="stable")
    neg_rows = neg_rows[neg_order]

    user_card = int(FEATURE_CARDINALITIES["user_id"])
    neg_counts = np.bincount(
        utr_np[neg_rows], minlength=user_card
    ).astype(np.int64)
    neg_offsets = np.zeros(user_card + 1, dtype=np.int64)
    np.cumsum(neg_counts, out=neg_offsets[1:])

    pos_users = utr_np[pos_rows]
    keep = neg_counts[pos_users] > 0
    pos_rows = pos_rows[keep]
    pos_users = pos_users[keep]
    return pos_rows, pos_users, neg_rows, neg_counts, neg_offsets


PAIR_POS, PAIR_USERS, NEG_ROWS, NEG_COUNTS, NEG_OFFSETS = make_pair_sampler()


def fit_metric(model, epochs_bce=1, epochs_pair=3, lr=1.5e-3, k_neg=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(ytr)
    generator = torch.Generator().manual_seed(SEED + 701)

    # A short pointwise warm start prevents arbitrary tower geometry and learns
    # useful marginal biases before hard-negative ranking begins.
    for epoch in range(epochs_bce):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        loss_total = 0.0
        for st in range(0, n, BATCH):
            idx = permutation[st:min(st + BATCH, n)]
            xb = xtr.index_select(0, idx)
            yb = ytr.index_select(0, idx)
            wb = recency.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            row_loss = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = (row_loss * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += float(loss.detach()) * len(idx)

        print(
            "TRAIN two_tower warmup=%d loss=%.6f" %
            (epoch + 1, loss_total / n),
            flush=True,
        )

    rng = np.random.default_rng(SEED + 991)
    n_pos = len(PAIR_POS)

    for epoch in range(epochs_pair):
        # Sample K same-user negatives for every positive entirely in numpy.
        counts = NEG_COUNTS[PAIR_USERS]
        draws = rng.random((n_pos, k_neg))
        local = (draws * counts[:, None]).astype(np.int64)
        neg_idx = NEG_OFFSETS[PAIR_USERS, None] + local
        sampled_neg = NEG_ROWS[neg_idx]

        permutation = rng.permutation(n_pos)
        loss_total = 0.0

        model.train()
        for st in range(0, n_pos, PAIR_BATCH):
            psel_np = permutation[st:min(st + PAIR_BATCH, n_pos)]
            p_rows_np = PAIR_POS[psel_np]
            n_rows_np = sampled_neg[psel_np]

            p_rows = torch.from_numpy(np.ascontiguousarray(p_rows_np))
            n_rows_flat = torch.from_numpy(
                np.ascontiguousarray(n_rows_np.reshape(-1))
            )

            pos_x = xtr.index_select(0, p_rows)
            neg_x = xtr.index_select(0, n_rows_flat)
            pair_w = recency.index_select(0, p_rows)

            optimizer.zero_grad(set_to_none=True)
            pos_score = model(pos_x)
            neg_score = model(neg_x).reshape(len(p_rows_np), k_neg)

            # Only the currently most confusing sampled negative contributes.
            hard_neg = neg_score.max(dim=1).values
            rank_loss = F.softplus(hard_neg - pos_score)
            loss = (rank_loss * pair_w).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += float(loss.detach()) * len(p_rows_np)

        print(
            "TRAIN two_tower pair_epoch=%d loss=%.6f pairs=%d" %
            (epoch + 1, loss_total / n_pos, n_pos),
            flush=True,
        )


def predict(model, x_np):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float32)
    with torch.inference_mode():
        for st in range(0, len(x_np), PRED_BATCH):
            en = min(st + PRED_BATCH, len(x_np))
            xb = torch.from_numpy(x_np[st:en])
            result[st:en] = model(xb).detach().cpu().numpy()
    return result


two_tower = TwoTowerMetric(emb_dim=12, tower_dim=48)
fit_metric(two_tower, epochs_bce=1, epochs_pair=3, lr=1.5e-3, k_neg=5)
tt_valid = predict(two_tower, xva_np)
tt_test = predict(two_tower, xte_np)
del two_tower

pnn = ProductNeuralNetwork(emb_dim=10)
fit_pointwise(pnn, "pnn", epochs=3, lr=1.2e-3)
pnn_valid = predict(pnn, xva_np)
pnn_test = predict(pnn, xte_np)
del pnn

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(shared, "incumbent_test_scores.npy")),
    dtype=np.float64,
)

valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)
valid_y = np.asarray(valid.y, dtype=np.int8)


def within_user_rank(users, scores):
    """
    Convert scores to normalized within-user ordinal positions. This is a
    label-free Borda representation and therefore transfers identically to test.
    """
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((scores, users))
    sorted_users = users[order]
    n = len(order)

    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    position = np.arange(n, dtype=np.float64) - np.repeat(starts, lengths)
    denom = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    ranked = position / denom

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


raw_valid = {
    "two_tower": np.asarray(tt_valid, dtype=np.float64),
    "pnn": np.asarray(pnn_valid, dtype=np.float64),
}
raw_test = {
    "two_tower": np.asarray(tt_test, dtype=np.float64),
    "pnn": np.asarray(pnn_test, dtype=np.float64),
}

inc_rank_valid = within_user_rank(valid_users, inc_valid)
inc_rank_test = within_user_rank(test_users, inc_test)

candidate_primary = {}
best_primary = -1.0
best_name = None
best_valid = None
best_test = None
best_raw = None
best_metrics = None

# Include standalone models, direct-logit blends, and Borda blends. The latter
# remove calibration/scale mismatch and preserve only within-user order.
for name in ["two_tower", "pnn"]:
    own_va = raw_valid[name]
    own_te = raw_test[name]

    direct_alphas = [0.15, 0.30, 0.50, 0.70, 1.0]
    for alpha in direct_alphas:
        cname = (
            name + "_raw" if alpha == 1.0
            else name + "_direct_%.2f" % alpha
        )
        va_score = alpha * own_va + (1.0 - alpha) * inc_valid
        te_score = alpha * own_te + (1.0 - alpha) * inc_test
        metrics = evaluate(valid_users, valid_y, va_score)
        primary = float(metrics["primary"])
        candidate_primary[cname] = primary

        if primary > best_primary:
            best_primary = primary
            best_name = cname
            best_valid = va_score.copy()
            best_test = te_score.copy()
            best_raw = own_va.copy()
            best_metrics = metrics

    own_rank_valid = within_user_rank(valid_users, own_va)
    own_rank_test = within_user_rank(test_users, own_te)
    for alpha in [0.25, 0.50, 0.75]:
        cname = name + "_borda_%.2f" % alpha
        va_score = (
            alpha * own_rank_valid + (1.0 - alpha) * inc_rank_valid
        )
        te_score = (
            alpha * own_rank_test + (1.0 - alpha) * inc_rank_test
        )
        metrics = evaluate(valid_users, valid_y, va_score)
        primary = float(metrics["primary"])
        candidate_primary[cname] = primary

        if primary > best_primary:
            best_primary = primary
            best_name = cname
            best_valid = va_score.copy()
            best_test = te_score.copy()
            best_raw = own_va.copy()
            best_metrics = metrics

# Explicit incumbent candidate ensures a weak new family cannot lower the run's
# submitted score.
inc_metrics = evaluate(valid_users, valid_y, inc_valid)
candidate_primary["incumbent"] = float(inc_metrics["primary"])
if float(inc_metrics["primary"]) > best_primary:
    best_primary = float(inc_metrics["primary"])
    best_name = "incumbent"
    best_valid = inc_valid.copy()
    best_test = inc_test.copy()
    best_raw = raw_valid["two_tower"].copy()
    best_metrics = inc_metrics

tt_metric = evaluate(valid_users, valid_y, raw_valid["two_tower"])
pnn_metric = evaluate(valid_users, valid_y, raw_valid["pnn"])
rank_corr = float(np.corrcoef(
    within_user_rank(valid_users, raw_valid["two_tower"]),
    within_user_rank(valid_users, raw_valid["pnn"]),
)[0, 1])

print("CANDIDATES " + json.dumps(candidate_primary, sort_keys=True))
print(
    "FINDINGS two_tower_raw=%.6f pnn_raw=%.6f "
    "within_user_rank_corr=%.6f selected=%s" %
    (
        float(tt_metric["primary"]),
        float(pnn_metric["primary"]),
        rank_corr,
        best_name,
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
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.6f}' %
    (
        float(best_metrics["primary"]),
        float(best_metrics["gauc"]),
        float(best_metrics["ndcg@5"]),
        elapsed,
    )
)