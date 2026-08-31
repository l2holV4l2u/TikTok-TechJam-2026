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
SEED = 18437
BATCH = 16384

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.float32)
train_dates = np.asarray(train.date, dtype=np.int64)

# Four-day half-life is fixed from the previously stated drift hypothesis.
recency = np.exp2(
    (train_dates - train_dates.max()).astype(np.float32) / 4.0
)
recency /= recency.mean()
recency = recency.astype(np.float32)


def make_group_ids(split, train_mode):
    users = np.asarray(split.user_id, dtype=np.int64)
    if train_mode:
        dates = np.asarray(split.date, dtype=np.int64)
        keys = users * np.int64(100000000) + dates
    else:
        # The evaluator treats the full split as each user's impression slate.
        keys = users
    _, inverse = np.unique(keys, return_inverse=True)
    return inverse.astype(np.int64, copy=False)


gtr = make_group_ids(train, True)
gva = make_group_ids(valid, False)
gte = make_group_ids(test, False)


def group_sizes(group_ids):
    return np.bincount(group_ids).astype(np.float32)


tr_group_sizes = group_sizes(gtr)
query_balanced_weight = recency / tr_group_sizes[gtr]
query_balanced_weight /= query_balanced_weight.mean()
query_balanced_weight = query_balanced_weight.astype(np.float32)


def grouped_frequency(group_ids, values, cardinality):
    values = np.asarray(values, dtype=np.int64)
    key = group_ids * np.int64(cardinality) + values
    _, inverse, counts = np.unique(
        key, return_inverse=True, return_counts=True
    )
    return counts[inverse].astype(np.float32)


def grouped_mean_std(group_ids, values):
    values = np.asarray(values, dtype=np.float64)
    ng = int(group_ids.max()) + 1

    count = np.bincount(group_ids, minlength=ng).astype(np.float64)
    total = np.bincount(
        group_ids, weights=values, minlength=ng
    ).astype(np.float64)
    square = np.bincount(
        group_ids, weights=values * values, minlength=ng
    ).astype(np.float64)

    mean = total / np.maximum(count, 1.0)
    variance = square / np.maximum(count, 1.0) - mean * mean
    std = np.sqrt(np.maximum(variance, 1e-6))

    return (
        mean[group_ids].astype(np.float32),
        std[group_ids].astype(np.float32),
    )


CONTEXT_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "upload_type",
]


def make_context_features(split, group_ids):
    n = len(group_ids)
    size = np.bincount(group_ids)[group_ids].astype(np.float32)
    columns = [np.log1p(size)]

    for field in CONTEXT_FIELDS:
        x = np.asarray(split.X[field], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[field])
        frequency = grouped_frequency(group_ids, x, card)
        columns.append(np.log1p(frequency))
        columns.append(frequency / np.maximum(size, 1.0))

    duration = np.log1p(
        np.maximum(
            np.asarray(split.num["duration_ms"], dtype=np.float32),
            0.0,
        )
    )
    duration_mean, duration_std = grouped_mean_std(group_ids, duration)
    columns.append(duration - duration_mean)
    columns.append((duration - duration_mean) / duration_std)

    # Relative values of stable user-side numeric quantities are constant
    # within a query, while raw log values still let the model condition its
    # confidence on the user segment.
    for name in [
        "duration_ms",
        "user_fans_user_num",
        "user_follow_user_num",
        "user_friend_user_num",
        "user_register_days",
    ]:
        value = np.asarray(split.num[name], dtype=np.float32)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(value, 0.0)))

    result = np.stack(columns, axis=1).astype(np.float32, copy=False)
    if result.shape[0] != n:
        raise RuntimeError("Context feature shape mismatch")
    return result


ctx_tr = make_context_features(train, gtr)
ctx_va = make_context_features(valid, gva)
ctx_te = make_context_features(test, gte)

# Normalizers are fit on train only.
ctx_mean = ctx_tr.mean(axis=0, dtype=np.float64).astype(np.float32)
ctx_std = ctx_tr.std(axis=0, dtype=np.float64).astype(np.float32)
ctx_std = np.maximum(ctx_std, 1e-3)

ctx_tr = np.clip((ctx_tr - ctx_mean) / ctx_std, -8.0, 8.0)
ctx_va = np.clip((ctx_va - ctx_mean) / ctx_std, -8.0, 8.0)
ctx_te = np.clip((ctx_te - ctx_mean) / ctx_std, -8.0, 8.0)

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "onehot_feat3",
    "onehot_feat8",
]

CARDS = [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS]


def make_cat_matrix(split):
    return np.ascontiguousarray(
        np.stack(
            [
                np.asarray(split.X[field], dtype=np.int64)
                for field in CAT_FIELDS
            ],
            axis=1,
        )
    )


xtr = make_cat_matrix(train)
xva = make_cat_matrix(valid)
xte = make_cat_matrix(test)


class AdditiveQueryModel(nn.Module):
    """A generalized additive categorical model, without interactions."""

    def __init__(self, cards, n_num):
        super().__init__()
        self.tables = nn.ModuleList(
            [nn.Embedding(card, 1) for card in cards]
        )
        self.numeric = nn.Linear(n_num, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

        for table in self.tables:
            nn.init.zeros_(table.weight)
        nn.init.zeros_(self.numeric.weight)

    def forward(self, cat, num):
        result = self.bias.expand(cat.shape[0])
        for j, table in enumerate(self.tables):
            result = result + table(cat[:, j]).squeeze(-1)
        result = result + self.numeric(num).squeeze(-1)
        return result


class SetwiseInteractionModel(nn.Module):
    """
    Nonlinear scorer whose candidate representation interacts with features
    computed from the complete logged slate.
    """

    def __init__(self, cards, n_num, dim=8):
        super().__init__()
        self.tables = nn.ModuleList(
            [nn.Embedding(card, dim) for card in cards]
        )
        input_dim = dim * len(cards) + n_num

        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 160),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(160, 80),
            nn.SiLU(),
            nn.Linear(80, 1),
        )

        for table in self.tables:
            nn.init.normal_(table.weight, std=0.015)

    def forward(self, cat, num):
        embedded = [
            table(cat[:, j]) for j, table in enumerate(self.tables)
        ]
        z = torch.cat(embedded + [num], dim=1)
        return self.mlp(self.norm(z)).squeeze(-1)


def train_additive():
    model = AdditiveQueryModel(CARDS, ctx_tr.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.018, weight_decay=2e-6
    )
    rng = np.random.default_rng(SEED + 1)

    for _ in range(4):
        order = rng.permutation(len(ytr))
        model.train()
        for lo in range(0, len(order), BATCH):
            idx = order[lo:lo + BATCH]
            cat = torch.from_numpy(xtr[idx])
            num = torch.from_numpy(ctx_tr[idx])
            target = torch.from_numpy(ytr[idx])
            weight = torch.from_numpy(query_balanced_weight[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(cat, num)
            row_loss = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            loss = (row_loss * weight).sum() / weight.sum()
            loss.backward()
            optimizer.step()

    return model


def contiguous_group_batches(group_ids, target_rows=BATCH):
    order = np.argsort(group_ids, kind="stable")
    sorted_groups = group_ids[order]
    n = len(order)
    batches = []
    lo = 0

    while lo < n:
        hi = min(lo + target_rows, n)
        if hi < n:
            current_group = sorted_groups[hi - 1]
            hi = np.searchsorted(
                sorted_groups, current_group, side="right"
            )
        batches.append(order[lo:hi])
        lo = hi

    return batches


list_batches = contiguous_group_batches(gtr)


def listnet_loss(logits, targets, local_groups, group_weight):
    ng = int(local_groups.max().item()) + 1
    ones = torch.ones_like(targets)

    counts = torch.zeros(
        ng, dtype=logits.dtype, device=logits.device
    )
    positives = torch.zeros_like(counts)
    counts.index_add_(0, local_groups, ones)
    positives.index_add_(0, local_groups, targets)

    maxima = torch.full_like(counts, -torch.inf)
    maxima.scatter_reduce_(
        0, local_groups, logits, reduce="amax", include_self=True
    )
    shifted = logits - maxima[local_groups]

    exp_sum = torch.zeros_like(counts)
    exp_sum.index_add_(0, local_groups, torch.exp(shifted))
    log_denom = maxima + torch.log(exp_sum.clamp_min(1e-12))

    mixed = (positives > 0.0) & (positives < counts)
    positive_weight = targets / positives[local_groups].clamp_min(1.0)
    row_loss = -positive_weight * (
        logits - log_denom[local_groups]
    )
    row_loss = row_loss * mixed[local_groups].to(logits.dtype)

    group_losses = torch.zeros_like(counts)
    group_losses.index_add_(0, local_groups, row_loss)

    usable_weight = group_weight * mixed.to(logits.dtype)
    return (
        (group_losses * usable_weight).sum()
        / usable_weight.sum().clamp_min(1.0)
    )


def train_setwise():
    model = SetwiseInteractionModel(
        CARDS, ctx_tr.shape[1], dim=8
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0022, weight_decay=1e-5
    )
    rng = np.random.default_rng(SEED + 2)

    for _ in range(4):
        batch_order = rng.permutation(len(list_batches))
        model.train()

        for batch_number in batch_order:
            idx = list_batches[int(batch_number)]
            batch_groups_np = gtr[idx]
            unique_groups, local_np = np.unique(
                batch_groups_np, return_inverse=True
            )

            cat = torch.from_numpy(xtr[idx])
            num = torch.from_numpy(ctx_tr[idx])
            target = torch.from_numpy(ytr[idx])
            local = torch.from_numpy(
                local_np.astype(np.int64, copy=False)
            )

            # Every train query is a user-day, so recency is constant within
            # a group. Equal query weighting prevents prolific users from
            # dominating the objective.
            first_rows = np.unique(
                local_np, return_index=True
            )[1]
            gw = torch.from_numpy(
                recency[idx[first_rows]].astype(np.float32, copy=False)
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(cat, num)
            loss = listnet_loss(
                logits, target, local, gw
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.inference_mode()
def predict_torch(model, cat_matrix, num_matrix):
    model.eval()
    result = np.empty(len(cat_matrix), dtype=np.float64)
    for lo in range(0, len(cat_matrix), 32768):
        hi = min(lo + 32768, len(cat_matrix))
        logits = model(
            torch.from_numpy(cat_matrix[lo:hi]),
            torch.from_numpy(num_matrix[lo:hi]),
        )
        result[lo:hi] = logits.cpu().numpy().astype(np.float64)
    return result


additive_model = train_additive()
add_valid = predict_torch(additive_model, xva, ctx_va)
add_test = predict_torch(additive_model, xte, ctx_te)
del additive_model

setwise_model = train_setwise()
set_valid = predict_torch(setwise_model, xva, ctx_va)
set_test = predict_torch(setwise_model, xte, ctx_te)
del setwise_model


# Third prediction family: histogram-boosted decision trees on exactly the
# same categorical and candidate-set-relative inputs.
tree_xtr = np.concatenate(
    [xtr.astype(np.float32), ctx_tr], axis=1
)
tree_xva = np.concatenate(
    [xva.astype(np.float32), ctx_va], axis=1
)
tree_xte = np.concatenate(
    [xte.astype(np.float32), ctx_te], axis=1
)

tree_train = lgb.Dataset(
    tree_xtr,
    label=ytr,
    weight=query_balanced_weight,
    categorical_feature=list(range(len(CAT_FIELDS))),
    free_raw_data=False,
)

tree_params = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.045,
    "num_leaves": 63,
    "min_data_in_leaf": 350,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.82,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "num_threads": min(8, os.cpu_count() or 1),
    "seed": SEED + 3,
    "feature_fraction_seed": SEED + 4,
    "bagging_seed": SEED + 5,
    "verbose": -1,
}

tree_model = lgb.train(
    tree_params,
    tree_train,
    num_boost_round=240,
)
tree_valid = tree_model.predict(
    tree_xva, num_iteration=tree_model.current_iteration()
).astype(np.float64)
tree_test = tree_model.predict(
    tree_xte, num_iteration=tree_model.current_iteration()
).astype(np.float64)

del tree_model, tree_train, tree_xtr, tree_xva, tree_xte


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort(
        (np.arange(n, dtype=np.int64), scores, user_ids)
    )
    ordered_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_users[1:] != ordered_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.int64) - repeated_starts

    ranked = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ranked[multi] = (
        positions[multi] / (repeated_lengths[multi] - 1.0)
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

families = {
    "query_additive": (add_valid, add_test),
    "setwise_listnet": (set_valid, set_test),
    "set_context_gbdt": (tree_valid, tree_test),
}

candidate_scores = {}
candidate_arrays = {}
candidate_raw = {}
blend_alphas = [0.15, 0.30, 0.50, 0.70, 1.00]

for name, (raw_valid, raw_test) in families.items():
    raw_metric = evaluate(valid.user_id, valid.y, raw_valid)
    raw_name = name + "_raw"
    candidate_scores[raw_name] = float(raw_metric["primary"])
    candidate_arrays[raw_name] = (raw_valid, raw_test)
    candidate_raw[raw_name] = raw_valid

    own_valid_rank = within_user_rank(valid.user_id, raw_valid)
    own_test_rank = within_user_rank(test.user_id, raw_test)

    for alpha in blend_alphas:
        blend_valid = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * own_valid_rank
        )
        blend_test = (
            (1.0 - alpha) * inc_test_rank
            + alpha * own_test_rank
        )
        blend_name = name + "_blend_" + str(alpha)
        metric = evaluate(valid.user_id, valid.y, blend_valid)

        candidate_scores[blend_name] = float(metric["primary"])
        candidate_arrays[blend_name] = (blend_valid, blend_test)
        candidate_raw[blend_name] = raw_valid

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores, test_scores = candidate_arrays[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner,
            "context_features": int(ctx_tr.shape[1]),
            "train_queries": int(gtr.max() + 1),
            "valid_queries": int(gva.max() + 1),
        },
        sort_keys=True,
    )
)
print(
    "CANDIDATES "
    + json.dumps(candidate_scores, sort_keys=True)
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

    if "_blend_" in winner:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[winner], dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)