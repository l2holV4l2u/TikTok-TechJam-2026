import os
import gc
import json
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")

START = time.time()
SEED = 314159
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")
if OUT:
    os.makedirs(OUT, exist_ok=True)

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "music_type",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

TE_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
]

TE_ALPHA = {
    "video_id": 30.0,
    "author_id": 60.0,
    "tag": 180.0,
    "tab": 250.0,
    "duration_bucket": 250.0,
    "upload_type": 250.0,
}

HALF_LIFE = 4.0
RANK = 8
EPOCHS = 2
BATCH_SIZE = 16384
PRED_BATCH = 131072


def finite32(x, fill=0.0):
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float32), 1e-4, 1.0 - 1e-4)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def select_history_keys(history):
    preferred = (
        "long_view_rate",
        "count_log1p",
        "is_click_rate",
        "play_time_ms_logmean",
        "comment_stay_time_logmean",
    )
    selected = []
    keys = sorted(history.keys())
    for term in preferred:
        matches = [k for k in keys if term in k.lower()]
        if matches and matches[0] not in selected:
            selected.append(matches[0])
    for key in keys:
        if key not in selected:
            selected.append(key)
        if len(selected) >= 5:
            break
    return selected[:5]


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=np.bool_)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    start_pos = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local = np.arange(n, dtype=np.float32) - start_pos.astype(np.float32)

    ends = np.empty(n, dtype=np.bool_)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_pos = np.flatnonzero(ends)
    sizes = np.diff(np.r_[-1, end_pos]).astype(np.float32)

    group = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[group] - 1.0, 1.0)
    ranked_sorted = local / denom

    ranked = np.empty(n, dtype=np.float32)
    ranked[order] = ranked_sorted
    return ranked


train = load("train")
train_y = np.asarray(train.y, dtype=np.float32)
n_train = len(train.user_id)

train_dates = np.asarray(train.date, dtype=np.int32)
last_date = int(np.max(train_dates))
date_age = (last_date - train_dates).astype(np.float32)
train_weight = np.power(0.5, date_age / HALF_LIFE).astype(np.float32)
train_weight /= np.mean(train_weight)

prior = float(np.sum(train_weight * train_y) / np.sum(train_weight))
prior_logit = float(np.log(prior / (1.0 - prior)))

cat_offsets = np.cumsum(
    [0] + [
        int(FEATURE_CARDINALITIES[field])
        for field in CAT_FIELDS[:-1]
    ]
).astype(np.int64)

total_cardinality = int(
    sum(int(FEATURE_CARDINALITIES[field]) for field in CAT_FIELDS)
)

te_maps = {}
train_te = {}

for field in TE_FIELDS:
    ids = np.asarray(train.X[field], dtype=np.int64)
    cardinality = int(FEATURE_CARDINALITIES[field])

    sum_weight = np.bincount(
        ids, weights=train_weight, minlength=cardinality
    ).astype(np.float32)
    sum_positive = np.bincount(
        ids,
        weights=train_weight * train_y,
        minlength=cardinality,
    ).astype(np.float32)

    alpha = float(TE_ALPHA[field])
    loo_count = np.maximum(sum_weight[ids] - train_weight, 0.0)
    loo_positive = sum_positive[ids] - train_weight * train_y
    loo_rate = (
        loo_positive + alpha * prior
    ) / np.maximum(loo_count + alpha, 1e-6)

    train_te[field] = (
        safe_logit(loo_rate),
        np.log1p(loo_count).astype(np.float32),
    )
    te_maps[field] = (sum_weight, sum_positive, alpha)


def external_te(split, field):
    ids = np.asarray(split.X[field], dtype=np.int64)
    sum_weight, sum_positive, alpha = te_maps[field]

    rate = np.full(len(ids), prior, dtype=np.float32)
    count = np.zeros(len(ids), dtype=np.float32)
    ok = (ids >= 0) & (ids < len(sum_weight))
    selected = ids[ok]

    rate[ok] = (
        sum_positive[selected] + alpha * prior
    ) / np.maximum(sum_weight[selected] + alpha, 1e-6)
    count[ok] = np.log1p(sum_weight[selected]).astype(np.float32)
    return safe_logit(rate), count


h_video_train = historical_features("train", key="video_id")
h_author_train = historical_features("train", key="author_id")
VIDEO_HKEYS = select_history_keys(h_video_train)
AUTHOR_HKEYS = select_history_keys(h_author_train)


def build_cat(split):
    return np.column_stack([
        np.asarray(split.X[field], dtype=np.int32) + int(offset)
        for field, offset in zip(CAT_FIELDS, cat_offsets)
    ]).astype(np.int32, copy=False)


def build_numeric(split, h_video, h_author, is_train):
    columns = []

    for key in VIDEO_HKEYS:
        columns.append(finite32(h_video[key]))
    for key in AUTHOR_HKEYS:
        columns.append(finite32(h_author[key]))

    for field in NUM_FIELDS:
        z = np.maximum(finite32(split.num[field]), 0.0)
        columns.append(np.log1p(z).astype(np.float32))

    for field in TE_FIELDS:
        if is_train:
            rate, count = train_te[field]
        else:
            rate, count = external_te(split, field)
        columns.extend([rate, count])

    return np.column_stack(columns).astype(np.float32, copy=False)


cat_train = build_cat(train)
num_train = build_numeric(
    train, h_video_train, h_author_train, is_train=True
)

num_mean = np.mean(num_train, axis=0, dtype=np.float64).astype(np.float32)
num_std = np.std(num_train, axis=0, dtype=np.float64).astype(np.float32)
num_std = np.maximum(num_std, 1e-3)

aux_names = [
    name for name in ("is_click", "is_like", "is_follow")
    if name in train.aux
]
aux_targets = [
    np.asarray(train.aux[name], dtype=np.float32)
    for name in aux_names
]
if aux_targets:
    multitask_targets = np.column_stack(
        [train_y] + aux_targets
    ).astype(np.float32, copy=False)
else:
    multitask_targets = train_y[:, None]

del h_video_train, h_author_train, train_te
gc.collect()


class EmbeddingBackbone(nn.Module):
    def __init__(self, total_card, rank):
        super().__init__()
        self.embedding = nn.Embedding(
            total_card, rank + 1, sparse=True
        )
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(0.0, 0.02)

    def fields(self, cat):
        values = self.embedding(cat)
        linear = values[:, :, 0].sum(dim=1)
        vectors = values[:, :, 1:]
        return linear, vectors


class WideDeep(nn.Module):
    def __init__(self, total_card, n_fields, n_num, rank):
        super().__init__()
        self.backbone = EmbeddingBackbone(total_card, rank)
        dim = n_fields * rank + n_num
        self.deep = nn.Sequential(
            nn.Linear(dim, 192),
            nn.ReLU(),
            nn.Linear(192, 80),
            nn.ReLU(),
            nn.Linear(80, 1),
        )
        self.numeric_wide = nn.Linear(n_num, 1)
        self.bias = nn.Parameter(torch.tensor(prior_logit))

    @property
    def sparse_weight(self):
        return self.backbone.embedding.weight

    def forward(self, cat, num):
        linear, vectors = self.backbone.fields(cat)
        dense = torch.cat([vectors.flatten(1), num], dim=1)
        return (
            self.bias
            + linear
            + self.numeric_wide(num).squeeze(1)
            + self.deep(dense).squeeze(1)
        )


class DeepFM(nn.Module):
    def __init__(self, total_card, n_fields, n_num, rank):
        super().__init__()
        self.backbone = EmbeddingBackbone(total_card, rank)
        dim = n_fields * rank + n_num
        self.deep = nn.Sequential(
            nn.Linear(dim, 192),
            nn.ReLU(),
            nn.Linear(192, 80),
            nn.ReLU(),
            nn.Linear(80, 1),
        )
        self.numeric_wide = nn.Linear(n_num, 1)
        self.bias = nn.Parameter(torch.tensor(prior_logit))

    @property
    def sparse_weight(self):
        return self.backbone.embedding.weight

    def forward(self, cat, num):
        linear, vectors = self.backbone.fields(cat)
        summed = vectors.sum(dim=1)
        fm = 0.5 * (
            summed.square() - vectors.square().sum(dim=1)
        ).sum(dim=1)
        dense = torch.cat([vectors.flatten(1), num], dim=1)
        return (
            self.bias
            + linear
            + fm
            + self.numeric_wide(num).squeeze(1)
            + self.deep(dense).squeeze(1)
        )


class NFM(nn.Module):
    def __init__(self, total_card, n_fields, n_num, rank):
        super().__init__()
        self.backbone = EmbeddingBackbone(total_card, rank)
        self.interaction = nn.Sequential(
            nn.BatchNorm1d(rank),
            nn.Linear(rank, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.numeric = nn.Sequential(
            nn.Linear(n_num, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.bias = nn.Parameter(torch.tensor(prior_logit))

    @property
    def sparse_weight(self):
        return self.backbone.embedding.weight

    def forward(self, cat, num):
        linear, vectors = self.backbone.fields(cat)
        summed = vectors.sum(dim=1)
        bi_interaction = 0.5 * (
            summed.square() - vectors.square().sum(dim=1)
        )
        return (
            self.bias
            + linear
            + self.interaction(bi_interaction).squeeze(1)
            + self.numeric(num).squeeze(1)
        )


class FiBiNET(nn.Module):
    def __init__(self, total_card, n_fields, n_num, rank):
        super().__init__()
        self.n_fields = n_fields
        self.backbone = EmbeddingBackbone(total_card, rank)

        reduction = max(2, n_fields // 3)
        self.se = nn.Sequential(
            nn.Linear(n_fields, reduction),
            nn.ReLU(),
            nn.Linear(reduction, n_fields),
            nn.Sigmoid(),
        )

        pair_i, pair_j = np.triu_indices(n_fields, k=1)
        self.register_buffer(
            "pair_i", torch.from_numpy(pair_i.astype(np.int64))
        )
        self.register_buffer(
            "pair_j", torch.from_numpy(pair_j.astype(np.int64))
        )

        pair_dim = len(pair_i) * rank
        self.deep = nn.Sequential(
            nn.Linear(pair_dim + n_fields * rank + n_num, 192),
            nn.ReLU(),
            nn.Linear(192, 72),
            nn.ReLU(),
            nn.Linear(72, 1),
        )
        self.numeric_wide = nn.Linear(n_num, 1)
        self.bias = nn.Parameter(torch.tensor(prior_logit))

    @property
    def sparse_weight(self):
        return self.backbone.embedding.weight

    def forward(self, cat, num):
        linear, vectors = self.backbone.fields(cat)
        squeeze = vectors.mean(dim=2)
        gates = self.se(squeeze).unsqueeze(2)
        weighted = vectors * (0.5 + gates)

        bilinear = (
            weighted[:, self.pair_i, :]
            * weighted[:, self.pair_j, :]
        ).flatten(1)

        dense = torch.cat(
            [weighted.flatten(1), bilinear, num], dim=1
        )
        return (
            self.bias
            + linear
            + self.numeric_wide(num).squeeze(1)
            + self.deep(dense).squeeze(1)
        )


class PLE(nn.Module):
    def __init__(
        self, total_card, n_fields, n_num, rank, n_tasks
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.backbone = EmbeddingBackbone(total_card, rank)
        input_dim = n_fields * rank + n_num
        expert_dim = 64

        def expert():
            return nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, expert_dim),
                nn.ReLU(),
            )

        self.shared_experts = nn.ModuleList([expert(), expert()])
        self.task_experts = nn.ModuleList([
            expert() for _ in range(n_tasks)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, 3) for _ in range(n_tasks)
        ])
        self.heads = nn.ModuleList([
            nn.Linear(expert_dim, 1) for _ in range(n_tasks)
        ])
        self.task_bias = nn.Parameter(torch.zeros(n_tasks))
        with torch.no_grad():
            self.task_bias[0] = prior_logit

    @property
    def sparse_weight(self):
        return self.backbone.embedding.weight

    def forward(self, cat, num):
        linear, vectors = self.backbone.fields(cat)
        x = torch.cat([vectors.flatten(1), num], dim=1)
        shared = [expert(x) for expert in self.shared_experts]

        outputs = []
        for task in range(self.n_tasks):
            specific = self.task_experts[task](x)
            stack = torch.stack(
                [shared[0], shared[1], specific], dim=1
            )
            gate = torch.softmax(self.gates[task](x), dim=1)
            representation = (
                stack * gate.unsqueeze(2)
            ).sum(dim=1)
            output = (
                self.heads[task](representation).squeeze(1)
                + self.task_bias[task]
            )
            if task == 0:
                output = output + linear
            outputs.append(output)

        return torch.stack(outputs, dim=1)


n_fields = len(CAT_FIELDS)
n_num = num_train.shape[1]

models = {
    "wide_deep": WideDeep(
        total_cardinality, n_fields, n_num, RANK
    ),
    "deepfm": DeepFM(
        total_cardinality, n_fields, n_num, RANK
    ),
    "nfm": NFM(
        total_cardinality, n_fields, n_num, RANK
    ),
    "fibinet": FiBiNET(
        total_cardinality, n_fields, n_num, RANK
    ),
    "ple": PLE(
        total_cardinality,
        n_fields,
        n_num,
        RANK,
        multitask_targets.shape[1],
    ),
}

optimizers = {}
dense_parameters = {}

for name, model in models.items():
    sparse = model.sparse_weight
    dense = [
        p for p in model.parameters()
        if p is not sparse
    ]
    dense_parameters[name] = dense
    optimizers[name] = (
        torch.optim.SparseAdam([sparse], lr=0.003),
        torch.optim.AdamW(
            dense, lr=0.0015, weight_decay=1e-5
        ),
    )

generator = torch.Generator().manual_seed(SEED)

for epoch in range(EPOCHS):
    permutation = torch.randperm(
        n_train, generator=generator
    ).numpy()

    loss_sums = {name: 0.0 for name in models}
    seen = 0

    for start in range(0, n_train, BATCH_SIZE):
        idx = permutation[start:start + BATCH_SIZE]

        cat_batch = torch.from_numpy(
            cat_train[idx].astype(np.int64, copy=False)
        )
        normalized = (num_train[idx] - num_mean) / num_std
        normalized = np.clip(
            normalized, -8.0, 8.0
        ).astype(np.float32, copy=False)
        num_batch = torch.from_numpy(normalized)
        y_batch = torch.from_numpy(train_y[idx])
        mt_batch = torch.from_numpy(multitask_targets[idx])
        w_batch = torch.from_numpy(train_weight[idx])

        for name, model in models.items():
            model.train()
            sparse_optimizer, dense_optimizer = optimizers[name]
            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)

            output = model(cat_batch, num_batch)

            if name == "ple":
                per_task = F.binary_cross_entropy_with_logits(
                    output, mt_batch, reduction="none"
                )
                task_weights = torch.full(
                    (output.shape[1],),
                    0.20,
                    dtype=torch.float32,
                )
                task_weights[0] = 1.0
                row_loss = (
                    per_task * task_weights.unsqueeze(0)
                ).sum(dim=1) / task_weights.sum()
            else:
                row_loss = F.binary_cross_entropy_with_logits(
                    output, y_batch, reduction="none"
                )

            loss = (
                torch.sum(row_loss * w_batch)
                / torch.sum(w_batch)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                dense_parameters[name], 5.0
            )
            sparse_optimizer.step()
            dense_optimizer.step()

            loss_sums[name] += float(loss.detach()) * len(idx)

        seen += len(idx)

    print(
        "FINDINGS epoch=%d %s"
        % (
            epoch + 1,
            " ".join(
                "%s_loss=%.6f"
                % (name, loss_sums[name] / max(seen, 1))
                for name in sorted(models)
            ),
        ),
        flush=True,
    )
    del permutation
    gc.collect()


@torch.inference_mode()
def predict_model(name, model, cat, num):
    model.eval()
    result = np.empty(len(cat), dtype=np.float32)

    for start in range(0, len(cat), PRED_BATCH):
        end = min(start + PRED_BATCH, len(cat))
        cat_batch = torch.from_numpy(
            cat[start:end].astype(np.int64, copy=False)
        )
        normalized = (num[start:end] - num_mean) / num_std
        normalized = np.clip(
            normalized, -8.0, 8.0
        ).astype(np.float32, copy=False)
        num_batch = torch.from_numpy(normalized)

        output = model(cat_batch, num_batch)
        if name == "ple":
            output = output[:, 0]
        result[start:end] = output.cpu().numpy()

    return result


del optimizers, dense_parameters
del cat_train, num_train, train_y, train_weight
del multitask_targets, aux_targets, train_dates, date_age, train
gc.collect()

valid = load("valid")
h_video_valid = historical_features("valid", key="video_id")
h_author_valid = historical_features("valid", key="author_id")
cat_valid = build_cat(valid)
num_valid = build_numeric(
    valid, h_video_valid, h_author_valid, is_train=False
)
del h_video_valid, h_author_valid
gc.collect()

valid_uid = np.asarray(valid.user_id)
valid_y = np.asarray(valid.y)

valid_predictions = {
    name: predict_model(name, model, cat_valid, num_valid)
    for name, model in models.items()
}

inc_valid_path = (
    os.path.join(SHARED, "incumbent_valid_scores.npy")
    if SHARED else ""
)
inc_test_path = (
    os.path.join(SHARED, "incumbent_test_scores.npy")
    if SHARED else ""
)
has_incumbent = bool(
    inc_valid_path
    and inc_test_path
    and os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
)

candidates = {}
best_primary = -np.inf
best_name = None
best_family = None
best_weight = 1.0
best_valid = None
best_raw_valid = None

standalone_ranks = {}

for name, scores in valid_predictions.items():
    metrics = evaluate(valid_uid, valid_y, scores)
    primary = float(metrics["primary"])
    candidates[name] = primary
    standalone_ranks[name] = within_user_rank(valid_uid, scores)

    if primary > best_primary:
        best_primary = primary
        best_name = name
        best_family = name
        best_weight = 1.0
        best_valid = scores.copy()
        best_raw_valid = scores.copy()

if has_incumbent:
    incumbent_valid = np.load(inc_valid_path, mmap_mode="r")
    incumbent_metrics = evaluate(
        valid_uid, valid_y, incumbent_valid
    )
    incumbent_primary = float(incumbent_metrics["primary"])
    candidates["trusted_incumbent"] = incumbent_primary
    incumbent_rank = within_user_rank(valid_uid, incumbent_valid)

    for name, own_rank in standalone_ranks.items():
        local_best = -np.inf
        local_weight = 0.0

        for weight in (
            0.05, 0.10, 0.15, 0.20, 0.25,
            0.30, 0.40, 0.50, 0.60, 0.70,
        ):
            blended = (
                weight * own_rank
                + (1.0 - weight) * incumbent_rank
            ).astype(np.float32)
            metrics = evaluate(valid_uid, valid_y, blended)
            primary = float(metrics["primary"])

            if primary > local_best:
                local_best = primary
                local_weight = float(weight)

            if primary > best_primary:
                best_primary = primary
                best_name = "%s_rankblend_w%.2f" % (
                    name, weight
                )
                best_family = name
                best_weight = float(weight)
                best_valid = blended.copy()
                best_raw_valid = valid_predictions[name].copy()

        candidates[name + "_best_blend"] = local_best
        correlation = float(np.corrcoef(
            incumbent_rank, own_rank
        )[0, 1])
        print(
            "FINDINGS family=%s incumbent_rank_corr=%.6f "
            "best_weight=%.2f blend_primary=%.6f"
            % (name, correlation, local_weight, local_best),
            flush=True,
        )

    if incumbent_primary >= best_primary:
        best_primary = incumbent_primary
        best_name = "trusted_incumbent"
        best_family = "trusted_incumbent"
        best_weight = 0.0
        best_valid = np.asarray(
            incumbent_valid, dtype=np.float32
        ).copy()
        standalone_best = max(
            valid_predictions,
            key=lambda key: candidates[key],
        )
        best_raw_valid = valid_predictions[standalone_best].copy()

final_metrics = evaluate(valid_uid, valid_y, best_valid)

print(
    "FINDINGS winner=%s aux_tasks=%s numeric_dimension=%d"
    % (
        best_name,
        ",".join(["long_view"] + aux_names),
        num_valid.shape[1],
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidates, sort_keys=True),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    if best_weight < 1.0:
        np.save(
            os.path.join(OUT, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

del cat_valid, num_valid, valid_y
del best_valid, best_raw_valid, standalone_ranks
gc.collect()

test = load("test")

if best_family == "trusted_incumbent":
    test_scores = np.asarray(
        np.load(inc_test_path, mmap_mode="r"),
        dtype=np.float32,
    ).copy()
else:
    h_video_test = historical_features("test", key="video_id")
    h_author_test = historical_features("test", key="author_id")
    cat_test = build_cat(test)
    num_test = build_numeric(
        test, h_video_test, h_author_test, is_train=False
    )
    del h_video_test, h_author_test
    gc.collect()

    own_test = predict_model(
        best_family,
        models[best_family],
        cat_test,
        num_test,
    )

    if best_weight < 1.0 and has_incumbent:
        incumbent_test = np.load(inc_test_path, mmap_mode="r")
        own_test_rank = within_user_rank(test.user_id, own_test)
        incumbent_test_rank = within_user_rank(
            test.user_id, incumbent_test
        )
        test_scores = (
            best_weight * own_test_rank
            + (1.0 - best_weight) * incumbent_test_rank
        ).astype(np.float32)
    else:
        test_scores = own_test

    del cat_test, num_test, own_test
    gc.collect()

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