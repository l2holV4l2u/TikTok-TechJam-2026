import os
import time
import json
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 43871
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag", "upload_type",
    "onehot_feat3", "onehot_feat8", "duration_bucket", "music_type",
    "user_active_degree", "register_days_bucket",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
HISTORY_SUFFIXES = [
    "train_count_log1p", "long_view_rate", "is_click_rate",
    "play_time_ms_logmean",
]

BATCH_SIZE = 8192
PRED_BATCH_SIZE = 16384
POINT_EPOCHS = 2
BPR_EPOCHS = 2
HALF_LIFE = 5.0


def make_cat_matrix(split):
    cards = [int(FEATURE_CARDINALITIES[x]) for x in CAT_FIELDS]
    offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
    out = np.empty((len(split.user_id), len(CAT_FIELDS)), dtype=np.int64)
    for j, name in enumerate(CAT_FIELDS):
        out[:, j] = np.asarray(split.X[name], dtype=np.int64) + offsets[j]
    return np.ascontiguousarray(out), cards, offsets


def make_numeric_raw(split, split_name):
    cols = []
    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))

    for entity in ("video_id", "author_id"):
        h = historical_features(split_name, key=entity)
        for suffix in HISTORY_SUFFIXES:
            x = np.asarray(h[entity + "_" + suffix], dtype=np.float32)
            cols.append(
                np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                .astype(np.float32)
            )
        del h
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)


def fit_numeric_transform(x):
    lo = np.quantile(x, 0.002, axis=0).astype(np.float32)
    hi = np.quantile(x, 0.998, axis=0).astype(np.float32)
    clipped = np.clip(x, lo, hi)
    mean = clipped.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = clipped.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-4] = 1.0
    return lo, hi, mean, std


def apply_numeric_transform(x, transform):
    lo, hi, mean, std = transform
    return np.ascontiguousarray(
        (np.clip(x, lo, hi) - mean) / std,
        dtype=np.float32,
    )


def recency_weights(dates, half_life=HALF_LIFE):
    d = np.asarray(dates, dtype=np.int64)
    age = d.max() - d
    w = np.power(0.5, age.astype(np.float32) / float(half_life))
    w /= max(float(w.mean()), 1e-8)
    return np.ascontiguousarray(w, dtype=np.float32)


def make_aux_targets(train):
    targets = []
    names = []
    for name in ("is_click", "is_like", "is_follow"):
        if name in train.aux:
            x = np.asarray(train.aux[name], dtype=np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
            x = np.clip(x, 0.0, 1.0)
            targets.append(x)
            names.append(name)
    if targets:
        return np.ascontiguousarray(np.column_stack(targets)), names
    return np.zeros((len(train.user_id), 0), dtype=np.float32), names


def make_bpr_pairs(user_ids, labels):
    users = np.asarray(user_ids, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int8)
    n_users = int(FEATURE_CARDINALITIES["user_id"])

    pos_rows = np.flatnonzero(y > 0)
    neg_rows = np.flatnonzero(y == 0)

    pos_users = users[pos_rows]
    neg_users = users[neg_rows]

    neg_order = np.argsort(neg_users, kind="stable")
    neg_rows_sorted = neg_rows[neg_order]
    neg_counts = np.bincount(neg_users, minlength=n_users).astype(np.int64)
    neg_starts = np.cumsum(
        np.concatenate(([0], neg_counts[:-1]))
    ).astype(np.int64)

    usable = neg_counts[pos_users] > 0
    pos_rows = pos_rows[usable]
    pos_users = pos_users[usable]

    rng = np.random.default_rng(SEED + 13)
    offsets = (
        rng.random(len(pos_rows)) * neg_counts[pos_users]
    ).astype(np.int64)
    neg_choice = neg_rows_sorted[neg_starts[pos_users] + offsets]

    return (
        np.ascontiguousarray(pos_rows, dtype=np.int64),
        np.ascontiguousarray(neg_choice, dtype=np.int64),
    )


class WideBase(nn.Module):
    def __init__(self, total_categories, n_num, k):
        super().__init__()
        self.embedding = nn.Embedding(total_categories, k)
        self.linear = nn.Embedding(total_categories, 1)
        self.num_linear = nn.Linear(n_num, 1)
        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.num_linear.bias)

    def wide(self, cats, nums):
        return (
            self.linear(cats).sum(dim=1).squeeze(1)
            + self.num_linear(nums).squeeze(1)
        )


class FieldAwareFM(nn.Module):
    def __init__(self, total_categories, n_fields, n_num, k=6):
        super().__init__()
        self.n_fields = n_fields
        self.k = k
        self.ffm = nn.Embedding(total_categories, n_fields * k)
        self.linear = nn.Embedding(total_categories, 1)
        self.num_head = nn.Sequential(
            nn.Linear(n_num, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )
        pi, pj = np.triu_indices(n_fields, k=1)
        self.register_buffer("pair_i", torch.from_numpy(pi.astype(np.int64)))
        self.register_buffer("pair_j", torch.from_numpy(pj.astype(np.int64)))
        nn.init.normal_(self.ffm.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def forward(self, cats, nums):
        b = cats.shape[0]
        e = self.ffm(cats).view(
            b, self.n_fields, self.n_fields, self.k
        )
        left = e[:, self.pair_i, self.pair_j, :]
        right = e[:, self.pair_j, self.pair_i, :]
        interactions = (left * right).sum(dim=(1, 2))
        return (
            self.linear(cats).sum(dim=1).squeeze(1)
            + self.num_head(nums).squeeze(1)
            + interactions
        )


class DCNv2(WideBase):
    def __init__(self, total_categories, n_fields, n_num, k=8):
        super().__init__(total_categories, n_num, k)
        dim = n_fields * k + n_num
        rank = 32
        self.cross_u = nn.ModuleList(
            [nn.Linear(dim, rank, bias=False) for _ in range(3)]
        )
        self.cross_v = nn.ModuleList(
            [nn.Linear(rank, dim, bias=True) for _ in range(3)]
        )
        self.deep = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 48),
            nn.GELU(),
        )
        self.out = nn.Linear(dim + 48, 1)

    def forward(self, cats, nums):
        e = self.embedding(cats)
        x0 = torch.cat([e.flatten(start_dim=1), nums], dim=1)
        x = x0
        for u, v in zip(self.cross_u, self.cross_v):
            x = x + x0 * v(torch.tanh(u(x)))
        deep = self.deep(x0)
        return self.wide(cats, nums) + self.out(
            torch.cat([x, deep], dim=1)
        ).squeeze(1)


class PLEModel(WideBase):
    def __init__(
        self, total_categories, n_fields, n_num, n_aux, k=8
    ):
        super().__init__(total_categories, n_num, k)
        self.n_tasks = 1 + n_aux
        dim = n_fields * k + n_num
        hidden = 64

        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 96), nn.ReLU(),
                nn.Dropout(0.08), nn.Linear(96, hidden), nn.ReLU()
            )
            for _ in range(2)
        ])
        self.task_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 96), nn.ReLU(),
                nn.Linear(96, hidden), nn.ReLU()
            )
            for _ in range(self.n_tasks)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(dim, 3) for _ in range(self.n_tasks)
        ])
        self.heads = nn.ModuleList([
            nn.Linear(hidden, 1) for _ in range(self.n_tasks)
        ])

    def forward(self, cats, nums, all_tasks=False):
        e = self.embedding(cats)
        x = torch.cat([e.flatten(start_dim=1), nums], dim=1)
        shared = [expert(x) for expert in self.shared_experts]

        outputs = []
        for t in range(self.n_tasks):
            private = self.task_experts[t](x)
            experts = torch.stack(
                [shared[0], shared[1], private], dim=1
            )
            gate = torch.softmax(self.gates[t](x), dim=1).unsqueeze(2)
            mixed = (experts * gate).sum(dim=1)
            logit = self.heads[t](mixed).squeeze(1)
            if t == 0:
                logit = logit + self.wide(cats, nums)
            outputs.append(logit)

        if all_tasks:
            return torch.stack(outputs, dim=1)
        return outputs[0]


class BPRFM(WideBase):
    def __init__(self, total_categories, n_num, k=12):
        super().__init__(total_categories, n_num, k)

    def forward(self, cats, nums):
        e = self.embedding(cats)
        summed = e.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - e.square().sum(dim=1)
        ).sum(dim=1)
        return self.wide(cats, nums) + interaction


def train_pointwise(
    model, cats, nums, labels, weights, seed,
    aux_targets=None
):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0024, weight_decay=2e-5
    )
    y_all = np.asarray(labels, dtype=np.float32)

    model.train()
    for _ in range(POINT_EPOCHS):
        order = rng.permutation(len(y_all))
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            c = torch.from_numpy(cats[idx])
            x = torch.from_numpy(nums[idx])
            y = torch.from_numpy(y_all[idx])
            w = torch.from_numpy(weights[idx])

            optimizer.zero_grad(set_to_none=True)
            if aux_targets is not None and aux_targets.shape[1] > 0:
                logits = model(c, x, all_tasks=True)
                main_loss = F.binary_cross_entropy_with_logits(
                    logits[:, 0], y, reduction="none"
                )
                loss = (main_loss * w).sum() / w.sum()
                aux = torch.from_numpy(aux_targets[idx])
                aux_loss = F.binary_cross_entropy_with_logits(
                    logits[:, 1:], aux, reduction="none"
                ).mean(dim=1)
                loss = loss + 0.18 * (aux_loss * w).sum() / w.sum()
            else:
                logits = model(c, x)
                row_loss = F.binary_cross_entropy_with_logits(
                    logits, y, reduction="none"
                )
                loss = (row_loss * w).sum() / w.sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


def train_bpr(
    model, cats, nums, pos_rows, neg_rows, weights, seed
):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0022, weight_decay=3e-5
    )
    model.train()

    for _ in range(BPR_EPOCHS):
        order = rng.permutation(len(pos_rows))
        for start in range(0, len(order), BATCH_SIZE // 2):
            q = order[start:start + BATCH_SIZE // 2]
            p = pos_rows[q]
            n = neg_rows[q]

            cp = torch.from_numpy(cats[p])
            xp = torch.from_numpy(nums[p])
            cn = torch.from_numpy(cats[n])
            xn = torch.from_numpy(nums[n])
            w = torch.from_numpy(weights[p])

            optimizer.zero_grad(set_to_none=True)
            pos_score = model(cp, xp)
            neg_score = model(cn, xn)
            row_loss = F.softplus(-(pos_score - neg_score))
            loss = (row_loss * w).sum() / w.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


@torch.no_grad()
def predict(model, cats, nums):
    model.eval()
    result = np.empty(len(cats), dtype=np.float32)
    for start in range(0, len(cats), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(cats))
        result[start:end] = model(
            torch.from_numpy(cats[start:end]),
            torch.from_numpy(nums[start:end]),
        ).cpu().numpy()
    return result


def user_center(scores, users, scale=None):
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users, dtype=np.int64)
    _, inv = np.unique(users, return_inverse=True)
    counts = np.bincount(inv)
    means = np.bincount(inv, weights=scores) / np.maximum(counts, 1)
    centered = scores - means[inv]
    if scale is None:
        scale = float(np.std(centered))
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
    return centered / float(scale), float(scale)


train = load("train")
valid = load("valid")
test = load("test")

train_cats, cards, offsets = make_cat_matrix(train)
valid_cats, _, _ = make_cat_matrix(valid)
test_cats, _, _ = make_cat_matrix(test)

train_raw = make_numeric_raw(train, "train")
valid_raw = make_numeric_raw(valid, "valid")
test_raw = make_numeric_raw(test, "test")

transform = fit_numeric_transform(train_raw)
train_num = apply_numeric_transform(train_raw, transform)
valid_num = apply_numeric_transform(valid_raw, transform)
test_num = apply_numeric_transform(test_raw, transform)
del train_raw, valid_raw, test_raw
gc.collect()

labels = np.asarray(train.y, dtype=np.float32)
weights = recency_weights(train.date)
aux_targets, aux_names = make_aux_targets(train)
pos_rows, neg_rows = make_bpr_pairs(train.user_id, train.y)

total_categories = int(sum(cards))
n_fields = len(CAT_FIELDS)
n_num = train_num.shape[1]

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared_dir, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared_dir, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_norm, inc_scale = user_center(inc_valid, valid.user_id)
inc_test_norm, _ = user_center(inc_test, test.user_id, inc_scale)

families = [
    (
        "ffm",
        FieldAwareFM(total_categories, n_fields, n_num, k=6),
        "point",
    ),
    (
        "dcnv2",
        DCNv2(total_categories, n_fields, n_num, k=8),
        "point",
    ),
    (
        "ple_multitask",
        PLEModel(
            total_categories, n_fields, n_num,
            n_aux=aux_targets.shape[1], k=8
        ),
        "ple",
    ),
    (
        "bpr_fm",
        BPRFM(total_categories, n_num, k=12),
        "bpr",
    ),
]

candidate_scores = {}
details = {}
best_primary = -np.inf
best_valid = None
best_test = None
best_raw_valid = None
best_name = None
best_metrics = None
best_alpha = None

for family_index, (name, model, mode) in enumerate(families):
    if mode == "point":
        train_pointwise(
            model, train_cats, train_num, labels, weights,
            SEED + 100 * family_index,
        )
    elif mode == "ple":
        train_pointwise(
            model, train_cats, train_num, labels, weights,
            SEED + 100 * family_index,
            aux_targets=aux_targets,
        )
    else:
        train_bpr(
            model, train_cats, train_num, pos_rows, neg_rows,
            weights, SEED + 100 * family_index,
        )

    raw_valid = predict(model, valid_cats, valid_num).astype(np.float64)
    raw_test = predict(model, test_cats, test_num).astype(np.float64)

    standalone_metrics = evaluate(
        valid.user_id, valid.y, raw_valid
    )
    candidate_scores[name + "_standalone"] = float(
        standalone_metrics["primary"]
    )

    own_valid_norm, own_scale = user_center(
        raw_valid, valid.user_id
    )
    own_test_norm, _ = user_center(
        raw_test, test.user_id, own_scale
    )

    family_best = -np.inf
    family_best_alpha = None
    for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
        blended_valid = (
            alpha * own_valid_norm
            + (1.0 - alpha) * inc_valid_norm
        )
        metrics = evaluate(
            valid.user_id, valid.y, blended_valid
        )
        primary = float(metrics["primary"])

        if primary > family_best:
            family_best = primary
            family_best_alpha = alpha

        if primary > best_primary:
            best_primary = primary
            best_valid = blended_valid.copy()
            best_test = (
                alpha * own_test_norm
                + (1.0 - alpha) * inc_test_norm
            ).copy()
            best_raw_valid = raw_valid.copy()
            best_name = name
            best_metrics = metrics
            best_alpha = alpha

    candidate_scores[name + "_best_blend"] = family_best
    details[name] = {
        "standalone": float(standalone_metrics["primary"]),
        "blend": float(family_best),
        "alpha": float(family_best_alpha),
        "gauc": float(standalone_metrics["gauc"]),
        "ndcg5": float(standalone_metrics["ndcg@5"]),
    }

    del model, raw_valid, raw_test, own_valid_norm, own_test_norm
    gc.collect()

print(
    "FINDINGS "
    + json.dumps(
        {
            "auxiliary_targets": aux_names,
            "bpr_pairs": int(len(pos_rows)),
            "half_life": HALF_LIFE,
            "details": details,
            "winner": best_name,
            "winner_alpha": float(best_alpha),
        },
        sort_keys=True,
    )
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
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
        }
    )
)