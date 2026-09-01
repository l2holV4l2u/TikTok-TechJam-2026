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
SEED = 2026
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
BATCH_SIZE = 16384
EPOCHS = 2


def finite32(x, fill=0.0):
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float32), 1e-4, 1.0 - 1e-4)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def choose_history_keys(d):
    preferred_terms = (
        "long_view_rate",
        "count_log1p",
        "is_click_rate",
        "play_time_ms_logmean",
        "comment_stay_time_logmean",
    )
    selected = []
    for term in preferred_terms:
        matches = [k for k in sorted(d.keys()) if term in k.lower()]
        if matches:
            selected.append(matches[0])
    for k in sorted(d.keys()):
        if k not in selected:
            selected.append(k)
        if len(selected) >= 5:
            break
    return selected[:5]


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    users_sorted = user_ids[order]

    starts = np.empty(n, dtype=np.bool_)
    starts[0] = True
    starts[1:] = users_sorted[1:] != users_sorted[:-1]

    start_position = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local_rank = (
        np.arange(n, dtype=np.float32) - start_position.astype(np.float32)
    )

    ends = np.empty(n, dtype=np.bool_)
    ends[-1] = True
    ends[:-1] = users_sorted[:-1] != users_sorted[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(np.r_[-1, end_positions]).astype(np.float32)

    group_index = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[group_index] - 1.0, 1.0)
    ranked_sorted = local_rank / denom

    result = np.empty(n, dtype=np.float32)
    result[order] = ranked_sorted
    return result


train = load("train")
train_y = np.asarray(train.y, dtype=np.float32)
n_train = len(train.user_id)

train_dates = np.asarray(train.date, dtype=np.int32)
max_train_date = int(np.max(train_dates))
age = (max_train_date - train_dates).astype(np.float32)
train_weight = np.power(0.5, age / HALF_LIFE).astype(np.float32)
train_weight /= float(np.mean(train_weight))

weighted_prior = float(
    np.sum(train_weight * train_y) / np.sum(train_weight)
)

cat_offsets = np.cumsum(
    [0] + [
        int(FEATURE_CARDINALITIES[f])
        for f in CAT_FIELDS[:-1]
    ]
).astype(np.int64)
total_cardinality = int(
    sum(int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS)
)

te_maps = {}
train_te_columns = {}

for field in TE_FIELDS:
    ids = np.asarray(train.X[field], dtype=np.int64)
    card = int(FEATURE_CARDINALITIES[field])
    sw = np.bincount(
        ids, weights=train_weight, minlength=card
    ).astype(np.float32)
    sy = np.bincount(
        ids, weights=train_weight * train_y, minlength=card
    ).astype(np.float32)
    alpha = float(TE_ALPHA[field])

    loo_weight = np.maximum(sw[ids] - train_weight, 0.0)
    loo_positive = sy[ids] - train_weight * train_y
    rate = (
        loo_positive + alpha * weighted_prior
    ) / np.maximum(loo_weight + alpha, 1e-6)

    train_te_columns[field] = (
        safe_logit(rate),
        np.log1p(loo_weight).astype(np.float32),
    )
    te_maps[field] = (sw, sy, alpha)


def external_te(split, field):
    ids = np.asarray(split.X[field], dtype=np.int64)
    sw, sy, alpha = te_maps[field]

    rate = np.full(len(ids), weighted_prior, dtype=np.float32)
    count = np.zeros(len(ids), dtype=np.float32)
    ok = (ids >= 0) & (ids < len(sw))

    selected = ids[ok]
    rate[ok] = (
        sy[selected] + alpha * weighted_prior
    ) / np.maximum(sw[selected] + alpha, 1e-6)
    count[ok] = np.log1p(sw[selected]).astype(np.float32)
    return safe_logit(rate), count


h_video_train = historical_features("train", key="video_id")
h_author_train = historical_features("train", key="author_id")
VIDEO_HKEYS = choose_history_keys(h_video_train)
AUTHOR_HKEYS = choose_history_keys(h_author_train)


def build_cat(split):
    return np.column_stack([
        np.asarray(split.X[field], dtype=np.int32) + int(offset)
        for field, offset in zip(CAT_FIELDS, cat_offsets)
    ]).astype(np.int32, copy=False)


def build_numeric(split, hv, ha, is_train):
    cols = []

    for key in VIDEO_HKEYS:
        cols.append(finite32(hv[key]))
    for key in AUTHOR_HKEYS:
        cols.append(finite32(ha[key]))

    for field in NUM_FIELDS:
        z = np.maximum(finite32(split.num[field]), 0.0)
        cols.append(np.log1p(z).astype(np.float32))

    for field in TE_FIELDS:
        if is_train:
            rate, count = train_te_columns[field]
        else:
            rate, count = external_te(split, field)
        cols.append(rate)
        cols.append(count)

    return np.column_stack(cols).astype(np.float32, copy=False)


cat_train = build_cat(train)
num_train = build_numeric(
    train, h_video_train, h_author_train, is_train=True
)

num_mean = np.mean(num_train, axis=0, dtype=np.float64).astype(np.float32)
num_std = np.std(num_train, axis=0, dtype=np.float64).astype(np.float32)
num_std = np.maximum(num_std, 1e-3)

del h_video_train, h_author_train, train_te_columns
gc.collect()


class ProductNetwork(nn.Module):
    def __init__(self, total_card, n_fields, n_num, rank):
        super().__init__()
        self.n_fields = n_fields
        self.rank = rank
        self.emb = nn.Embedding(total_card, rank + 1, sparse=True)

        pair_i, pair_j = np.triu_indices(n_fields, k=1)
        self.register_buffer(
            "pair_i", torch.from_numpy(pair_i.astype(np.int64))
        )
        self.register_buffer(
            "pair_j", torch.from_numpy(pair_j.astype(np.int64))
        )

        n_pairs = len(pair_i)
        input_dim = n_fields * rank + n_pairs + n_num
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 160),
            nn.ReLU(),
            nn.LayerNorm(160),
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.num_linear = nn.Linear(n_num, 1)
        self.bias = nn.Parameter(torch.tensor(
            np.log(weighted_prior / (1.0 - weighted_prior)),
            dtype=torch.float32,
        ))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.emb.weight[:, 0].zero_()
            self.emb.weight[:, 1:].normal_(0.0, 0.02)

    def forward(self, cat, num):
        all_emb = self.emb(cat)
        linear = all_emb[:, :, 0].sum(dim=1)
        vectors = all_emb[:, :, 1:]

        left = vectors[:, self.pair_i, :]
        right = vectors[:, self.pair_j, :]
        products = (left * right).sum(dim=2)

        dense_input = torch.cat(
            [vectors.flatten(1), products, num], dim=1
        )
        return (
            self.bias
            + linear
            + self.num_linear(num).squeeze(1)
            + self.mlp(dense_input).squeeze(1)
        )


class LowRankCrossLayer(nn.Module):
    def __init__(self, dim, low_rank=32):
        super().__init__()
        self.down = nn.Linear(dim, low_rank)
        self.up = nn.Linear(low_rank, dim)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x0, x):
        interaction = self.up(torch.tanh(self.down(x))) + self.bias
        return x + x0 * interaction


class DCNv2(nn.Module):
    def __init__(self, total_card, n_fields, n_num, rank):
        super().__init__()
        self.emb = nn.Embedding(total_card, rank + 1, sparse=True)
        dim = n_fields * rank + n_num

        self.cross_layers = nn.ModuleList([
            LowRankCrossLayer(dim, 32),
            LowRankCrossLayer(dim, 32),
            LowRankCrossLayer(dim, 32),
        ])
        self.deep = nn.Sequential(
            nn.Linear(dim, 160),
            nn.ReLU(),
            nn.Linear(160, 80),
            nn.ReLU(),
        )
        self.output = nn.Linear(dim + 80, 1)
        self.num_linear = nn.Linear(n_num, 1)
        self.bias = nn.Parameter(torch.tensor(
            np.log(weighted_prior / (1.0 - weighted_prior)),
            dtype=torch.float32,
        ))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.emb.weight[:, 0].zero_()
            self.emb.weight[:, 1:].normal_(0.0, 0.02)

    def forward(self, cat, num):
        all_emb = self.emb(cat)
        linear = all_emb[:, :, 0].sum(dim=1)
        vectors = all_emb[:, :, 1:]
        x0 = torch.cat([vectors.flatten(1), num], dim=1)

        cross = x0
        for layer in self.cross_layers:
            cross = layer(x0, cross)

        deep = self.deep(x0)
        return (
            self.bias
            + linear
            + self.num_linear(num).squeeze(1)
            + self.output(torch.cat([cross, deep], dim=1)).squeeze(1)
        )


class AutoInt(nn.Module):
    def __init__(self, total_card, n_fields, n_num, rank):
        super().__init__()
        self.emb = nn.Embedding(total_card, rank + 1, sparse=True)
        self.attn1 = nn.MultiheadAttention(
            rank, num_heads=2, dropout=0.0, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            rank, num_heads=2, dropout=0.0, batch_first=True
        )
        self.norm1 = nn.LayerNorm(rank)
        self.norm2 = nn.LayerNorm(rank)
        self.output = nn.Sequential(
            nn.Linear(n_fields * rank + n_num, 128),
            nn.ReLU(),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.num_linear = nn.Linear(n_num, 1)
        self.bias = nn.Parameter(torch.tensor(
            np.log(weighted_prior / (1.0 - weighted_prior)),
            dtype=torch.float32,
        ))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.emb.weight[:, 0].zero_()
            self.emb.weight[:, 1:].normal_(0.0, 0.02)

    def forward(self, cat, num):
        all_emb = self.emb(cat)
        linear = all_emb[:, :, 0].sum(dim=1)
        x = all_emb[:, :, 1:]

        a1, _ = self.attn1(x, x, x, need_weights=False)
        x = self.norm1(x + a1)
        a2, _ = self.attn2(x, x, x, need_weights=False)
        x = self.norm2(x + a2)

        dense_input = torch.cat([x.flatten(1), num], dim=1)
        return (
            self.bias
            + linear
            + self.num_linear(num).squeeze(1)
            + self.output(dense_input).squeeze(1)
        )


models = {
    "pnn": ProductNetwork(
        total_cardinality, len(CAT_FIELDS), num_train.shape[1], RANK
    ),
    "dcnv2": DCNv2(
        total_cardinality, len(CAT_FIELDS), num_train.shape[1], RANK
    ),
    "autoint": AutoInt(
        total_cardinality, len(CAT_FIELDS), num_train.shape[1], RANK
    ),
}

optimizers = {}
dense_parameter_lists = {}

for name, model in models.items():
    dense_params = [
        p for parameter_name, p in model.named_parameters()
        if parameter_name != "emb.weight"
    ]
    dense_parameter_lists[name] = dense_params
    optimizers[name] = (
        torch.optim.SparseAdam([model.emb.weight], lr=0.003),
        torch.optim.AdamW(
            dense_params, lr=0.0015, weight_decay=1e-5
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
        normalized_num = (num_train[idx] - num_mean) / num_std
        normalized_num = np.clip(
            normalized_num, -8.0, 8.0
        ).astype(np.float32, copy=False)

        num_batch = torch.from_numpy(normalized_num)
        y_batch = torch.from_numpy(train_y[idx])
        w_batch = torch.from_numpy(train_weight[idx])

        for name, model in models.items():
            model.train()
            sparse_optimizer, dense_optimizer = optimizers[name]
            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)

            logits = model(cat_batch, num_batch)
            losses = F.binary_cross_entropy_with_logits(
                logits, y_batch, reduction="none"
            )
            loss = torch.sum(losses * w_batch) / torch.sum(w_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                dense_parameter_lists[name], 5.0
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
                "%s_weighted_logloss=%.6f"
                % (name, loss_sums[name] / max(seen, 1))
                for name in sorted(models)
            ),
        ),
        flush=True,
    )
    del permutation
    gc.collect()


@torch.inference_mode()
def predict_model(model, cat, num):
    model.eval()
    result = np.empty(len(cat), dtype=np.float32)
    prediction_batch_size = 131072

    for start in range(0, len(cat), prediction_batch_size):
        end = min(start + prediction_batch_size, len(cat))
        cat_batch = torch.from_numpy(
            cat[start:end].astype(np.int64, copy=False)
        )
        normalized_num = (
            num[start:end] - num_mean
        ) / num_std
        normalized_num = np.clip(
            normalized_num, -8.0, 8.0
        ).astype(np.float32, copy=False)
        num_batch = torch.from_numpy(normalized_num)
        result[start:end] = model(
            cat_batch, num_batch
        ).cpu().numpy()

    return result


del optimizers, dense_parameter_lists
del cat_train, num_train, train_y, train_weight, age, train_dates, train
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

valid_predictions = {
    name: predict_model(model, cat_valid, num_valid)
    for name, model in models.items()
}

valid_uid = np.asarray(valid.user_id)
valid_y = np.asarray(valid.y)

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

candidate_scores = {}
best_primary = -np.inf
best_name = None
best_family = None
best_weight = 1.0
best_valid_scores = None
best_raw_valid = None

for name, scores in valid_predictions.items():
    metrics = evaluate(valid_uid, valid_y, scores)
    primary = float(metrics["primary"])
    candidate_scores[name] = primary

    if primary > best_primary:
        best_primary = primary
        best_name = name
        best_family = name
        best_weight = 1.0
        best_valid_scores = scores.copy()
        best_raw_valid = scores.copy()

if has_incumbent:
    incumbent_valid = np.load(inc_valid_path, mmap_mode="r")
    incumbent_metrics = evaluate(valid_uid, valid_y, incumbent_valid)
    incumbent_primary = float(incumbent_metrics["primary"])
    candidate_scores["trusted_incumbent"] = incumbent_primary

    incumbent_rank = within_user_rank(valid_uid, incumbent_valid)

    best_standalone_family = max(
        valid_predictions,
        key=lambda name: candidate_scores[name]
    )

    for name, scores in valid_predictions.items():
        own_rank = within_user_rank(valid_uid, scores)
        local_best_primary = -np.inf
        local_best_weight = None

        for weight in (
            0.10, 0.20, 0.30, 0.40, 0.50,
            0.60, 0.70, 0.80, 0.90
        ):
            blended = (
                weight * own_rank
                + (1.0 - weight) * incumbent_rank
            ).astype(np.float32)
            metrics = evaluate(valid_uid, valid_y, blended)
            primary = float(metrics["primary"])

            if primary > local_best_primary:
                local_best_primary = primary
                local_best_weight = float(weight)

            if primary > best_primary:
                best_primary = primary
                best_name = "%s_rankblend_w%.2f" % (name, weight)
                best_family = name
                best_weight = float(weight)
                best_valid_scores = blended.copy()
                best_raw_valid = scores.copy()

        candidate_scores[name + "_best_blend"] = local_best_primary
        print(
            "FINDINGS family=%s best_incumbent_rank_weight=%.2f "
            "blend_primary=%.6f"
            % (name, local_best_weight, local_best_primary),
            flush=True,
        )

    if incumbent_primary >= best_primary:
        best_primary = incumbent_primary
        best_name = "trusted_incumbent"
        best_family = "trusted_incumbent"
        best_weight = 0.0
        best_valid_scores = np.asarray(
            incumbent_valid, dtype=np.float32
        ).copy()
        best_raw_valid = valid_predictions[
            best_standalone_family
        ].copy()

final_metrics = evaluate(valid_uid, valid_y, best_valid_scores)

print(
    "FINDINGS winner=%s history_video=%s history_author=%s "
    "numeric_dimension=%d"
    % (
        best_name,
        ",".join(VIDEO_HKEYS),
        ",".join(AUTHOR_HKEYS),
        num_valid.shape[1],
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_scores, sort_keys=True),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    if best_weight < 1.0:
        np.save(
            os.path.join(OUT, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

del cat_valid, num_valid, best_valid_scores, best_raw_valid
del valid_y
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
        models[best_family], cat_test, num_test
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