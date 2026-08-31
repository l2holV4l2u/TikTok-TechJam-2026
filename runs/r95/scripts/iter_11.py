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
BATCH = 8192
PAIR_BATCH = 4096
PRED_BATCH = 32768

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

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
    "register_days_bucket",
    "video_type",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]
CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
N_FIELDS = len(FIELDS)

train = load("train")
valid = load("valid")
test = load("test")


def make_categorical(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[f], dtype=np.int64) for f in FIELDS
        ]),
        dtype=np.int64,
    )


def fit_numeric_transform(split):
    raw = np.column_stack([
        np.asarray(split.num[f], dtype=np.float32) for f in NUM_FIELDS
    ]).astype(np.float32)

    finite = np.isfinite(raw)
    medians = np.zeros(raw.shape[1], dtype=np.float32)
    means = np.zeros(raw.shape[1], dtype=np.float32)
    scales = np.ones(raw.shape[1], dtype=np.float32)

    transformed = np.empty_like(raw)
    for j in range(raw.shape[1]):
        values = raw[finite[:, j], j]
        med = float(np.median(values)) if len(values) else 0.0
        medians[j] = med
        filled = np.where(finite[:, j], raw[:, j], med)
        signed_log = np.sign(filled) * np.log1p(np.abs(filled))
        means[j] = float(np.mean(signed_log))
        scales[j] = max(float(np.std(signed_log)), 1e-3)
        transformed[:, j] = (signed_log - means[j]) / scales[j]

    return transformed, medians, means, scales


def transform_numeric(split, medians, means, scales):
    raw = np.column_stack([
        np.asarray(split.num[f], dtype=np.float32) for f in NUM_FIELDS
    ]).astype(np.float32)
    finite = np.isfinite(raw)
    filled = np.where(finite, raw, medians[None, :])
    signed_log = np.sign(filled) * np.log1p(np.abs(filled))
    return np.ascontiguousarray(
        (signed_log - means[None, :]) / scales[None, :],
        dtype=np.float32,
    )


xtr_np = make_categorical(train)
xva_np = make_categorical(valid)
xte_np = make_categorical(test)

ntr_np, num_medians, num_means, num_scales = fit_numeric_transform(train)
nva_np = transform_numeric(valid, num_medians, num_means, num_scales)
nte_np = transform_numeric(test, num_medians, num_means, num_scales)

ytr_np = np.asarray(train.y, dtype=np.float32)
utr_np = np.asarray(train.user_id, dtype=np.int64)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)
valid_y = np.asarray(valid.y, dtype=np.int8)

xtr = torch.from_numpy(xtr_np)
ntr = torch.from_numpy(ntr_np)
ytr = torch.from_numpy(ytr_np)

# Four-day temporal half-life, previously found substantially more suitable
# than uniform fitting under this date split.
last_date = int(np.max(np.asarray(train.date, dtype=np.int64)))
day_age = (
    last_date - np.asarray(train.date, dtype=np.int64)
).astype(np.float32)
recency_np = np.exp2(-day_age / 4.0).astype(np.float32)

# Row-level BCE heavily favors prolific train users, whereas nDCG averages
# users equally and validation has far fewer rows per user. Square-root inverse
# frequency is a compromise between row weighting and exactly equal users.
user_card = int(FEATURE_CARDINALITIES["user_id"])
user_counts = np.bincount(
    utr_np, minlength=user_card
).astype(np.float32)
user_balance_np = 1.0 / np.sqrt(np.maximum(user_counts[utr_np], 1.0))

sample_weight_np = recency_np * user_balance_np
sample_weight_np /= float(np.mean(sample_weight_np))
sample_weight = torch.from_numpy(sample_weight_np.astype(np.float32))

print(
    "FINDINGS sample_weight_p10_p50_p90=%s" %
    np.array2string(
        np.quantile(sample_weight_np, [0.1, 0.5, 0.9]),
        precision=4,
        separator=",",
    ),
    flush=True,
)


class FeatureEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, emb_dim) for card in CARDS
        ])
        self.biases = nn.ModuleList([
            nn.Embedding(card, 1) for card in CARDS
        ])
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.025)
        for bias in self.biases:
            nn.init.zeros_(bias.weight)

    def embed(self, x):
        return torch.stack(
            [self.embeddings[j](x[:, j]) for j in range(N_FIELDS)],
            dim=1,
        )

    def wide(self, x):
        return torch.stack(
            [
                self.biases[j](x[:, j]).squeeze(-1)
                for j in range(N_FIELDS)
            ],
            dim=1,
        ).sum(dim=1)


class FiBiNET(nn.Module):
    """
    Squeeze-excitation learns impression-dependent field importance. A shared
    bilinear projection then forms vector-valued pair interactions instead of
    reducing every interaction immediately to one scalar as an FM does.
    """

    def __init__(self, emb_dim=10):
        super().__init__()
        self.encoder = FeatureEncoder(emb_dim)

        squeeze_width = max(4, N_FIELDS // 3)
        self.se = nn.Sequential(
            nn.Linear(N_FIELDS, squeeze_width),
            nn.ReLU(),
            nn.Linear(squeeze_width, N_FIELDS),
            nn.Sigmoid(),
        )
        self.bilinear = nn.Linear(emb_dim, emb_dim, bias=False)

        pair_i = []
        pair_j = []
        for i in range(N_FIELDS):
            for j in range(i + 1, N_FIELDS):
                pair_i.append(i)
                pair_j.append(j)
        self.register_buffer(
            "pair_i", torch.tensor(pair_i, dtype=torch.long)
        )
        self.register_buffer(
            "pair_j", torch.tensor(pair_j, dtype=torch.long)
        )

        interaction_width = len(pair_i) * emb_dim
        input_width = N_FIELDS * emb_dim + interaction_width + len(NUM_FIELDS)
        self.deep = nn.Sequential(
            nn.Linear(input_width, 160),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.global_bias = nn.Parameter(torch.zeros(()))

    def forward(self, x, numeric):
        emb = self.encoder.embed(x)
        squeeze = emb.mean(dim=2)
        gates = self.se(squeeze).unsqueeze(2)
        selected = emb * gates

        left = self.bilinear(
            selected.index_select(1, self.pair_i)
        )
        right = selected.index_select(1, self.pair_j)
        pair_vectors = left * right

        deep_input = torch.cat(
            [
                selected.flatten(1),
                pair_vectors.flatten(1),
                numeric,
            ],
            dim=1,
        )
        return (
            self.encoder.wide(x)
            + self.deep(deep_input).squeeze(1)
            + self.global_bias
        )


class MMoE(nn.Module):
    """
    Several experts are shared, while each task has a separate soft gate and
    tower. Auxiliary outcomes are training targets only and are never supplied
    as features or consulted when predicting validation/test rows.
    """

    def __init__(self, n_tasks, emb_dim=10, n_experts=4):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_experts = n_experts
        self.encoder = FeatureEncoder(emb_dim)
        input_width = N_FIELDS * emb_dim + len(NUM_FIELDS)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_width, 96),
                nn.ReLU(),
                nn.Linear(96, 48),
                nn.ReLU(),
            )
            for _ in range(n_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_width, n_experts) for _ in range(n_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(48, 24),
                nn.ReLU(),
                nn.Linear(24, 1),
            )
            for _ in range(n_tasks)
        ])
        self.task_bias = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, x, numeric):
        emb = self.encoder.embed(x)
        base = torch.cat([emb.flatten(1), numeric], dim=1)
        expert_values = torch.stack(
            [expert(base) for expert in self.experts],
            dim=1,
        )

        outputs = []
        wide = self.encoder.wide(x)
        for task in range(self.n_tasks):
            gate = F.softmax(self.gates[task](base), dim=1).unsqueeze(2)
            mixed = (gate * expert_values).sum(dim=1)
            value = self.towers[task](mixed).squeeze(1)
            if task == 0:
                value = value + wide
            outputs.append(value + self.task_bias[task])
        return torch.stack(outputs, dim=1)


def make_pair_sampler():
    pos_rows = np.flatnonzero(ytr_np > 0.5).astype(np.int64)
    neg_rows = np.flatnonzero(ytr_np < 0.5).astype(np.int64)

    neg_order = np.argsort(utr_np[neg_rows], kind="stable")
    neg_rows = neg_rows[neg_order]

    neg_counts = np.bincount(
        utr_np[neg_rows], minlength=user_card
    ).astype(np.int64)
    neg_offsets = np.zeros(user_card + 1, dtype=np.int64)
    np.cumsum(neg_counts, out=neg_offsets[1:])

    pos_users = utr_np[pos_rows]
    keep = neg_counts[pos_users] > 0
    return (
        pos_rows[keep],
        pos_users[keep],
        neg_rows,
        neg_counts,
        neg_offsets,
    )


PAIR_POS, PAIR_USERS, NEG_ROWS, NEG_COUNTS, NEG_OFFSETS = (
    make_pair_sampler()
)


def fit_fibinet(model, bce_epochs=1, pair_epochs=2, k_neg=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1.25e-3)
    generator = torch.Generator().manual_seed(SEED + 101)
    n = len(ytr)

    for epoch in range(bce_epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        total = 0.0

        for st in range(0, n, BATCH):
            idx = permutation[st:min(st + BATCH, n)]
            xb = xtr.index_select(0, idx)
            nb = ntr.index_select(0, idx)
            yb = ytr.index_select(0, idx)
            wb = sample_weight.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, nb)
            row_loss = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = (row_loss * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(idx)

        print(
            "TRAIN fibinet_bce epoch=%d loss=%.6f" %
            (epoch + 1, total / n),
            flush=True,
        )

    rng = np.random.default_rng(SEED + 202)
    n_pos = len(PAIR_POS)

    for epoch in range(pair_epochs):
        counts = NEG_COUNTS[PAIR_USERS]
        local = (
            rng.random((n_pos, k_neg)) * counts[:, None]
        ).astype(np.int64)
        sampled_neg = NEG_ROWS[
            NEG_OFFSETS[PAIR_USERS, None] + local
        ]
        permutation = rng.permutation(n_pos)
        total = 0.0

        model.train()
        for st in range(0, n_pos, PAIR_BATCH):
            selected = permutation[st:min(st + PAIR_BATCH, n_pos)]
            pos_np = np.ascontiguousarray(PAIR_POS[selected])
            neg_np = np.ascontiguousarray(
                sampled_neg[selected].reshape(-1)
            )

            pos_rows = torch.from_numpy(pos_np)
            neg_rows = torch.from_numpy(neg_np)

            optimizer.zero_grad(set_to_none=True)
            pos_score = model(
                xtr.index_select(0, pos_rows),
                ntr.index_select(0, pos_rows),
            )
            neg_score = model(
                xtr.index_select(0, neg_rows),
                ntr.index_select(0, neg_rows),
            ).reshape(len(pos_np), k_neg)

            # The most confusing sampled same-user negative drives each update.
            hard_negative = neg_score.max(dim=1).values
            pair_loss = F.softplus(hard_negative - pos_score)
            pair_weight = sample_weight.index_select(0, pos_rows)
            loss = (pair_loss * pair_weight).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(pos_np)

        print(
            "TRAIN fibinet_pair epoch=%d loss=%.6f pairs=%d" %
            (epoch + 1, total / n_pos, n_pos),
            flush=True,
        )


aux_names = []
aux_arrays = []
for name in ["is_click", "is_like"]:
    if name in train.aux:
        values = np.asarray(train.aux[name], dtype=np.float32)
        if len(values) == len(ytr_np):
            values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
            values = np.clip(values, 0.0, 1.0)
            aux_names.append(name)
            aux_arrays.append(values)

if aux_arrays:
    multitask_np = np.column_stack([ytr_np] + aux_arrays).astype(np.float32)
else:
    multitask_np = ytr_np[:, None].astype(np.float32)
multitask_targets = torch.from_numpy(multitask_np)

print(
    "FINDINGS mmoe_auxiliary_tasks=%s" %
    json.dumps(aux_names),
    flush=True,
)


def fit_mmoe(model, epochs=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=1.1e-3)
    generator = torch.Generator().manual_seed(SEED + 303)
    n = len(ytr)
    task_weights = torch.ones(multitask_np.shape[1], dtype=torch.float32)
    if len(task_weights) > 1:
        task_weights[1:] = 0.18

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        total = 0.0

        for st in range(0, n, BATCH):
            idx = permutation[st:min(st + BATCH, n)]
            xb = xtr.index_select(0, idx)
            nb = ntr.index_select(0, idx)
            targets = multitask_targets.index_select(0, idx)
            wb = sample_weight.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, nb)
            task_losses = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            combined = (
                task_losses * task_weights.unsqueeze(0)
            ).sum(dim=1) / task_weights.sum()
            loss = (combined * wb).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(idx)

        print(
            "TRAIN mmoe epoch=%d loss=%.6f" %
            (epoch + 1, total / n),
            flush=True,
        )


def predict_fibinet(model, categorical, numeric):
    result = np.empty(len(categorical), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for st in range(0, len(categorical), PRED_BATCH):
            en = min(st + PRED_BATCH, len(categorical))
            xb = torch.from_numpy(categorical[st:en])
            nb = torch.from_numpy(numeric[st:en])
            result[st:en] = model(xb, nb).cpu().numpy()
    return result


def predict_mmoe(model, categorical, numeric):
    result = np.empty(len(categorical), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for st in range(0, len(categorical), PRED_BATCH):
            en = min(st + PRED_BATCH, len(categorical))
            xb = torch.from_numpy(categorical[st:en])
            nb = torch.from_numpy(numeric[st:en])
            result[st:en] = model(xb, nb)[:, 0].cpu().numpy()
    return result


fibinet = FiBiNET(emb_dim=10)
fit_fibinet(fibinet, bce_epochs=1, pair_epochs=2, k_neg=5)
fib_valid = predict_fibinet(fibinet, xva_np, nva_np)
fib_test = predict_fibinet(fibinet, xte_np, nte_np)
del fibinet

mmoe = MMoE(
    n_tasks=multitask_np.shape[1],
    emb_dim=10,
    n_experts=4,
)
fit_mmoe(mmoe, epochs=3)
mmoe_valid = predict_mmoe(mmoe, xva_np, nva_np)
mmoe_test = predict_mmoe(mmoe, xte_np, nte_np)
del mmoe

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
    order = np.lexsort((scores, users))
    sorted_users = users[order]
    n = len(order)

    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    positions = (
        np.arange(n, dtype=np.float64) - np.repeat(starts, lengths)
    )
    denominators = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    rank_values = positions / denominators

    result = np.empty(n, dtype=np.float64)
    result[order] = rank_values
    return result


raw_valid = {
    "fibinet_userbalanced_hardneg": np.asarray(
        fib_valid, dtype=np.float64
    ),
    "mmoe_userbalanced_aux": np.asarray(
        mmoe_valid, dtype=np.float64
    ),
}
raw_test = {
    "fibinet_userbalanced_hardneg": np.asarray(
        fib_test, dtype=np.float64
    ),
    "mmoe_userbalanced_aux": np.asarray(
        mmoe_test, dtype=np.float64
    ),
}

inc_rank_valid = within_user_rank(valid_users, inc_valid)
inc_rank_test = within_user_rank(test_users, inc_test)

candidate_primary = {}
best_primary = -np.inf
best_name = None
best_valid = None
best_test = None
best_raw = None
best_metrics = None

for family in raw_valid:
    own_valid = raw_valid[family]
    own_test = raw_test[family]

    own_metrics = evaluate(valid_users, valid_y, own_valid)
    own_primary = float(own_metrics["primary"])
    candidate_primary[family + "_raw"] = own_primary

    if own_primary > best_primary:
        best_primary = own_primary
        best_name = family + "_raw"
        best_valid = own_valid.copy()
        best_test = own_test.copy()
        best_raw = own_valid.copy()
        best_metrics = own_metrics

    own_rank_valid = within_user_rank(valid_users, own_valid)
    own_rank_test = within_user_rank(test_users, own_test)

    for alpha in [0.15, 0.30, 0.50, 0.70]:
        name = family + "_borda_%.2f" % alpha
        blend_valid = (
            alpha * own_rank_valid
            + (1.0 - alpha) * inc_rank_valid
        )
        blend_test = (
            alpha * own_rank_test
            + (1.0 - alpha) * inc_rank_test
        )
        metrics = evaluate(valid_users, valid_y, blend_valid)
        primary = float(metrics["primary"])
        candidate_primary[name] = primary

        if primary > best_primary:
            best_primary = primary
            best_name = name
            best_valid = blend_valid.copy()
            best_test = blend_test.copy()
            best_raw = own_valid.copy()
            best_metrics = metrics

# Also test whether the two genuinely different new representations complement
# each other before they are blended with the trusted incumbent.
fib_rank_valid = within_user_rank(
    valid_users, raw_valid["fibinet_userbalanced_hardneg"]
)
fib_rank_test = within_user_rank(
    test_users, raw_test["fibinet_userbalanced_hardneg"]
)
mmoe_rank_valid = within_user_rank(
    valid_users, raw_valid["mmoe_userbalanced_aux"]
)
mmoe_rank_test = within_user_rank(
    test_users, raw_test["mmoe_userbalanced_aux"]
)

new_ensemble_valid = 0.5 * fib_rank_valid + 0.5 * mmoe_rank_valid
new_ensemble_test = 0.5 * fib_rank_test + 0.5 * mmoe_rank_test

for alpha in [0.15, 0.30, 0.50, 0.70, 1.0]:
    name = (
        "fibinet_mmoe_ensemble_raw"
        if alpha == 1.0
        else "fibinet_mmoe_inc_borda_%.2f" % alpha
    )
    if alpha == 1.0:
        va_score = new_ensemble_valid
        te_score = new_ensemble_test
    else:
        va_score = (
            alpha * new_ensemble_valid
            + (1.0 - alpha) * inc_rank_valid
        )
        te_score = (
            alpha * new_ensemble_test
            + (1.0 - alpha) * inc_rank_test
        )

    metrics = evaluate(valid_users, valid_y, va_score)
    primary = float(metrics["primary"])
    candidate_primary[name] = primary

    if primary > best_primary:
        best_primary = primary
        best_name = name
        best_valid = va_score.copy()
        best_test = te_score.copy()
        best_raw = new_ensemble_valid.copy()
        best_metrics = metrics

print(
    "CANDIDATES " + json.dumps(
        {k: round(v, 7) for k, v in candidate_primary.items()},
        sort_keys=True,
    ),
    flush=True,
)
print(
    "FINDINGS selected_candidate=%s" % best_name,
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
    if best_name is not None and (
        "borda" in best_name or "ensemble" in best_name
    ):
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.4f}' %
    (
        float(best_metrics["primary"]),
        float(best_metrics["gauc"]),
        float(best_metrics["ndcg@5"]),
        elapsed,
    )
)