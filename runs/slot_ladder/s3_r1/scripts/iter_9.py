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
SEED = 314159
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
]
NUMERIC_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1], dtype=np.int64)
TOTAL_CARDINALITY = int(sum(CARDS))
NUM_FIELDS = len(FIELDS)
EMBED_DIM = 10
TRAIN_TARGET_ROWS = 9000
PRED_TARGET_ROWS = 24000
EPOCHS = 3
LR = 0.002
HALF_LIFE_DAYS = 4.0


def make_categorical(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[field], dtype=np.int64) + offset
            for field, offset in zip(FIELDS, OFFSETS)
        ]),
        dtype=np.int64,
    )


def fit_numeric_transform(train):
    centers = []
    scales = []
    for field in NUMERIC_FIELDS:
        x = np.asarray(train.num[field], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.sign(x) * np.log1p(np.abs(x))
        centers.append(float(np.mean(x)))
        scale = float(np.std(x))
        scales.append(scale if scale > 1e-6 else 1.0)
    return np.asarray(centers), np.asarray(scales)


def make_numeric(split, centers, scales):
    columns = []
    for j, field in enumerate(NUMERIC_FIELDS):
        x = np.asarray(split.num[field], dtype=np.float64)
        missing = ~np.isfinite(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.sign(x) * np.log1p(np.abs(x))
        x = (x - centers[j]) / scales[j]
        x[missing] = 0.0
        columns.append(x.astype(np.float32))
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def sigmoid_np(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def probability_scale(x):
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
        return np.clip(x, 1e-7, 1.0 - 1e-7)
    return sigmoid_np(x)


def build_group_layout(user_ids, target_rows, seed):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    unique_users, inverse, counts = np.unique(
        user_ids, return_inverse=True, return_counts=True
    )
    rng = np.random.default_rng(seed)
    user_priority = np.empty(unique_users.size, dtype=np.int64)
    user_priority[rng.permutation(unique_users.size)] = np.arange(
        unique_users.size, dtype=np.int64
    )
    row_order = np.argsort(user_priority[inverse], kind="stable")
    ordered_group = user_priority[inverse[row_order]]

    ordered_counts = counts[np.argsort(user_priority)]
    cumulative = np.concatenate([
        np.zeros(1, dtype=np.int64),
        np.cumsum(ordered_counts, dtype=np.int64),
    ])

    chunks = []
    group_start = 0
    row_start = 0
    n_groups = ordered_counts.size
    while group_start < n_groups:
        desired = row_start + target_rows
        group_end = int(np.searchsorted(cumulative, desired, side="right") - 1)
        group_end = max(group_end, group_start + 1)
        group_end = min(group_end, n_groups)
        row_end = int(cumulative[group_end])
        chunks.append((row_start, row_end, group_start, group_end))
        row_start = row_end
        group_start = group_end

    return {
        "order": row_order,
        "ordered_group": ordered_group,
        "chunks": chunks,
        "n_users": int(unique_users.size),
        "max_group": int(counts.max()),
    }


class BaseCategoricalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, EMBED_DIM)
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.025)
        nn.init.zeros_(self.linear.weight)

    def wide(self, x):
        return self.linear(x).sum(dim=1).squeeze(1) + self.bias


class NFMModel(BaseCategoricalModel):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(EMBED_DIM + len(NUMERIC_FIELDS), 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, numeric, group_ids=None, n_groups=None):
        e = self.embedding(x)
        summed = e.sum(dim=1)
        bi = 0.5 * (
            summed.square() - e.square().sum(dim=1)
        )
        features = torch.cat([bi, numeric], dim=1)
        return self.wide(x) + self.network(features).squeeze(1)


class XDeepFMModel(BaseCategoricalModel):
    def __init__(self):
        super().__init__()
        self.cin1 = nn.Conv1d(NUM_FIELDS * NUM_FIELDS, 14, kernel_size=1)
        self.cin2 = nn.Conv1d(NUM_FIELDS * 14, 10, kernel_size=1)
        self.deep = nn.Sequential(
            nn.Linear(NUM_FIELDS * EMBED_DIM + len(NUMERIC_FIELDS), 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.cin_output = nn.Linear(24, 1)

    def forward(self, x, numeric, group_ids=None, n_groups=None):
        e0 = self.embedding(x)

        z1 = (
            e0.unsqueeze(2) * e0.unsqueeze(1)
        ).reshape(e0.shape[0], NUM_FIELDS * NUM_FIELDS, EMBED_DIM)
        h1 = F.relu(self.cin1(z1))

        z2 = (
            e0.unsqueeze(2) * h1.unsqueeze(1)
        ).reshape(e0.shape[0], NUM_FIELDS * 14, EMBED_DIM)
        h2 = F.relu(self.cin2(z2))

        cin_features = torch.cat([
            h1.sum(dim=2),
            h2.sum(dim=2),
        ], dim=1)
        deep_features = torch.cat([
            e0.reshape(e0.shape[0], -1),
            numeric,
        ], dim=1)

        return (
            self.wide(x)
            + self.deep(deep_features).squeeze(1)
            + self.cin_output(cin_features).squeeze(1)
        )


class SetContextModel(BaseCategoricalModel):
    def __init__(self):
        super().__init__()
        self.context_transform = nn.Sequential(
            nn.Linear(EMBED_DIM, 32),
            nn.Tanh(),
            nn.Linear(32, EMBED_DIM),
        )
        self.local_network = nn.Sequential(
            nn.Linear(NUM_FIELDS * EMBED_DIM + len(NUMERIC_FIELDS), 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.context_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, x, numeric, group_ids=None, n_groups=None):
        e = self.embedding(x)
        local = self.local_network(
            torch.cat([e.reshape(e.shape[0], -1), numeric], dim=1)
        ).squeeze(1)

        item_repr = e[:, 1:4, :].mean(dim=1)
        context_sum = torch.zeros(
            (n_groups, EMBED_DIM),
            dtype=item_repr.dtype,
            device=item_repr.device,
        )
        context_sum.index_add_(0, group_ids, item_repr)

        counts = torch.bincount(
            group_ids, minlength=n_groups
        ).to(item_repr.dtype).clamp_min(1.0)
        context = context_sum / counts.unsqueeze(1)
        context = self.context_transform(context)

        candidate = e[:, 1, :]
        context_score = torch.sum(
            candidate * context.index_select(0, group_ids), dim=1
        ) / np.sqrt(EMBED_DIM)

        return (
            self.wide(x)
            + local
            + self.context_scale * context_score
        )


def grouped_listwise_loss(logits, labels, weights, group_ids, n_groups):
    neg_inf = torch.full(
        (n_groups,), -1e30, dtype=logits.dtype, device=logits.device
    )
    group_max = neg_inf.scatter_reduce(
        0, group_ids, logits.detach(), reduce="amax", include_self=True
    )

    shifted_exp = torch.exp(
        logits - group_max.index_select(0, group_ids)
    )
    group_exp_sum = torch.zeros(
        n_groups, dtype=logits.dtype, device=logits.device
    )
    group_exp_sum.index_add_(0, group_ids, shifted_exp)
    log_prob = (
        logits
        - group_max.index_select(0, group_ids)
        - torch.log(
            group_exp_sum.index_select(0, group_ids).clamp_min(1e-12)
        )
    )

    positive_weight = labels * weights
    group_positive_weight = torch.zeros(
        n_groups, dtype=logits.dtype, device=logits.device
    )
    group_positive_weight.index_add_(0, group_ids, positive_weight)

    normalized_target = positive_weight / group_positive_weight.index_select(
        0, group_ids
    ).clamp_min(1e-12)

    group_ce = torch.zeros(
        n_groups, dtype=logits.dtype, device=logits.device
    )
    group_ce.index_add_(0, group_ids, -normalized_target * log_prob)

    active = group_positive_weight > 0
    if bool(active.any()):
        list_loss = group_ce[active].mean()
    else:
        list_loss = logits.sum() * 0.0

    point_losses = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    point_loss = torch.sum(point_losses * weights) / torch.sum(weights)
    return list_loss + 0.30 * point_loss, list_loss.detach(), point_loss.detach()


def train_model(
    model,
    x_np,
    numeric_np,
    y_np,
    weight_np,
    layout,
    seed,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=2e-6
    )
    order = layout["order"]
    ordered_groups = layout["ordered_group"]

    rng = np.random.default_rng(seed)
    chunk_indices = np.arange(len(layout["chunks"]))

    model.train()
    for epoch in range(EPOCHS):
        rng.shuffle(chunk_indices)
        epoch_list = 0.0
        epoch_point = 0.0
        seen = 0

        for chunk_index in chunk_indices:
            row_start, row_end, group_start, group_end = (
                layout["chunks"][chunk_index]
            )
            idx_np = order[row_start:row_end]
            local_group_np = (
                ordered_groups[row_start:row_end] - group_start
            ).astype(np.int64, copy=False)
            n_groups = group_end - group_start

            xb = torch.from_numpy(x_np[idx_np])
            nb = torch.from_numpy(numeric_np[idx_np])
            yb = torch.from_numpy(y_np[idx_np])
            wb = torch.from_numpy(weight_np[idx_np])
            gb = torch.from_numpy(local_group_np)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, nb, gb, n_groups)
            loss, list_part, point_part = grouped_listwise_loss(
                logits, yb, wb, gb, n_groups
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_list += float(list_part)
            epoch_point += float(point_part)
            seen += 1

        print(
            "FINDINGS "
            + json.dumps({
                "model": model.__class__.__name__,
                "epoch": epoch + 1,
                "list_loss": epoch_list / max(seen, 1),
                "point_loss": epoch_point / max(seen, 1),
            })
        )


def predict_grouped(model, x_np, numeric_np, user_ids, seed):
    layout = build_group_layout(user_ids, PRED_TARGET_ROWS, seed)
    order = layout["order"]
    ordered_groups = layout["ordered_group"]
    ordered_prediction = np.empty(x_np.shape[0], dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for row_start, row_end, group_start, group_end in layout["chunks"]:
            idx_np = order[row_start:row_end]
            local_group_np = (
                ordered_groups[row_start:row_end] - group_start
            ).astype(np.int64, copy=False)
            xb = torch.from_numpy(x_np[idx_np])
            nb = torch.from_numpy(numeric_np[idx_np])
            gb = torch.from_numpy(local_group_np)
            logits = model(
                xb, nb, gb, group_end - group_start
            )
            ordered_prediction[row_start:row_end] = logits.cpu().numpy()

    result = np.empty_like(ordered_prediction)
    result[order] = ordered_prediction
    return result


train = load("train")
x_train = make_categorical(train)
numeric_centers, numeric_scales = fit_numeric_transform(train)
n_train = make_numeric(train, numeric_centers, numeric_scales)
y_train = np.asarray(train.y, dtype=np.float32)

dates = np.asarray(train.date, dtype=np.int64)
unique_dates = np.sort(np.unique(dates))
date_age_table = {
    int(date): len(unique_dates) - 1 - i
    for i, date in enumerate(unique_dates)
}
ages = np.fromiter(
    (date_age_table[int(date)] for date in dates),
    count=dates.size,
    dtype=np.float32,
)
train_weights = np.exp2(-ages / HALF_LIFE_DAYS).astype(np.float32)
train_weights /= float(np.mean(train_weights))

train_layout = build_group_layout(
    np.asarray(train.user_id, dtype=np.int64),
    TRAIN_TARGET_ROWS,
    SEED + 1,
)

print(
    "FINDINGS "
    + json.dumps({
        "train_groups": train_layout["n_users"],
        "train_chunks": len(train_layout["chunks"]),
        "max_group_size": train_layout["max_group"],
        "half_life_days": HALF_LIFE_DAYS,
        "weight_min": float(train_weights.min()),
        "weight_max": float(train_weights.max()),
    })
)

models = {
    "nfm_listnet": NFMModel(),
    "xdeepfm_cin_listnet": XDeepFMModel(),
    "set_context_listnet": SetContextModel(),
}

for model_index, (name, model) in enumerate(models.items()):
    train_model(
        model,
        x_train,
        n_train,
        y_train,
        train_weights,
        train_layout,
        SEED + 100 * (model_index + 1),
    )

valid = load("valid")
x_valid = make_categorical(valid)
n_valid = make_numeric(valid, numeric_centers, numeric_scales)

raw_valid = {}
candidate_values = {}
candidate_specs = {}

for model_index, (name, model) in enumerate(models.items()):
    logits = predict_grouped(
        model,
        x_valid,
        n_valid,
        np.asarray(valid.user_id, dtype=np.int64),
        SEED + 1000 + model_index,
    )
    scores = sigmoid_np(logits)
    raw_valid[name] = scores
    metric = evaluate(valid.user_id, valid.y, scores)
    candidate_values[name] = float(metric["primary"])
    candidate_specs[name] = {
        "family": name,
        "alpha": 1.0,
        "blended": False,
        "scores": scores,
    }

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = None
inc_test_path = None
if shared:
    vp = os.path.join(shared, "incumbent_valid_scores.npy")
    tp = os.path.join(shared, "incumbent_test_scores.npy")
    if os.path.exists(vp) and os.path.exists(tp):
        inc_valid = probability_scale(np.load(vp))
        inc_test_path = tp

if inc_valid is not None:
    for name, own_scores in raw_valid.items():
        for alpha in (0.20, 0.35, 0.50, 0.65, 0.80):
            blended = alpha * own_scores + (1.0 - alpha) * inc_valid
            candidate_name = f"{name}_blend_{alpha:.2f}"
            metric = evaluate(valid.user_id, valid.y, blended)
            candidate_values[candidate_name] = float(metric["primary"])
            candidate_specs[candidate_name] = {
                "family": name,
                "alpha": float(alpha),
                "blended": True,
                "scores": blended,
            }

winner_name = max(candidate_values, key=candidate_values.get)
winner = candidate_specs[winner_name]
valid_scores = winner["scores"]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(candidate_values, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps({
        "winner": winner_name,
        "family": winner["family"],
        "alpha": winner["alpha"],
        "blended": winner["blended"],
    })
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if winner["blended"]:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_valid[winner["family"]], dtype=np.float64),
        )

    test = load("test")
    x_test = make_categorical(test)
    n_test = make_numeric(test, numeric_centers, numeric_scales)
    winner_model = models[winner["family"]]
    test_logits = predict_grouped(
        winner_model,
        x_test,
        n_test,
        np.asarray(test.user_id, dtype=np.int64),
        SEED + 9000,
    )
    own_test_scores = sigmoid_np(test_logits)

    if winner["blended"]:
        incumbent_test = probability_scale(np.load(inc_test_path))
        test_scores = (
            winner["alpha"] * own_test_scores
            + (1.0 - winner["alpha"]) * incumbent_test
        )
    else:
        test_scores = own_test_scores

    np.save(
        os.path.join(out, "scores_test.npy"),
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