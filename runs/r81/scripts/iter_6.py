import os
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 20260829
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(12, max(1, os.cpu_count() or 1)))

DEVICE = torch.device("cpu")
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "user_active_degree",
    "video_type",
]
CARDS = [int(FEATURE_CARDINALITIES[x]) for x in FIELDS]
STAT_KEYS = ["video_id", "author_id", "tag"]
RAW_NUMS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores.copy()

    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    positions = np.arange(n, dtype=np.int64) - np.repeat(starts, sizes)
    ranked_sorted = (positions + 0.5) / np.repeat(sizes, sizes)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


def make_x(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[name], dtype=np.int64) for name in FIELDS
        ]),
        dtype=np.int64,
    )


def concat_x(a, b):
    return np.ascontiguousarray(
        np.column_stack([
            np.concatenate([
                np.asarray(a.X[name], dtype=np.int64),
                np.asarray(b.X[name], dtype=np.int64),
            ])
            for name in FIELDS
        ]),
        dtype=np.int64,
    )


def raw_numeric(split):
    cols = []
    for name in RAW_NUMS:
        x = np.asarray(split.num[name], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.sign(x) * np.log1p(np.abs(x))
        cols.append(x)
    return np.column_stack(cols).astype(np.float32)


def fit_numeric_scaler(x):
    mean = x.mean(axis=0, dtype=np.float64)
    std = x.std(axis=0, dtype=np.float64)
    std = np.maximum(std, 1e-3)
    return mean.astype(np.float32), std.astype(np.float32)


def transform_numeric(x, mean, std):
    z = (x - mean[None, :]) / std[None, :]
    return np.clip(z, -8.0, 8.0).astype(np.float32)


def recency_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.int64)
    latest = int(dates.max())
    age = np.maximum(latest - dates, 0).astype(np.float64)
    return np.exp2(-age / float(half_life))


def target_statistics(fit_split, fit_y, score_split, leave_one_out_fit=True):
    y = np.asarray(fit_y, dtype=np.float64)
    dates = np.asarray(fit_split.date, dtype=np.int64)
    recent_w = recency_weights(dates, half_life=5.0)
    global_rate = float(y.mean())
    recent_global = float(np.sum(recent_w * y) / np.sum(recent_w))

    fit_features = []
    score_features = []
    score_logits = []

    for key in STAT_KEYS:
        card = int(FEATURE_CARDINALITIES[key])
        ids_fit = np.asarray(fit_split.X[key], dtype=np.int64)
        ids_score = np.asarray(score_split.X[key], dtype=np.int64)

        count = np.bincount(ids_fit, minlength=card).astype(np.float64)
        positive = np.bincount(
            ids_fit, weights=y, minlength=card
        ).astype(np.float64)
        rcount = np.bincount(
            ids_fit, weights=recent_w, minlength=card
        ).astype(np.float64)
        rpositive = np.bincount(
            ids_fit, weights=recent_w * y, minlength=card
        ).astype(np.float64)

        if leave_one_out_fit:
            own_count = np.maximum(count[ids_fit] - 1.0, 0.0)
            own_positive = positive[ids_fit] - y
            own_rcount = np.maximum(
                rcount[ids_fit] - recent_w, 0.0
            )
            own_rpositive = rpositive[ids_fit] - recent_w * y
        else:
            own_count = count[ids_fit]
            own_positive = positive[ids_fit]
            own_rcount = rcount[ids_fit]
            own_rpositive = rpositive[ids_fit]

        alpha = 14.0
        recent_alpha = 8.0

        fit_rate = (
            own_positive + alpha * global_rate
        ) / (own_count + alpha)
        fit_recent_rate = (
            own_rpositive + recent_alpha * recent_global
        ) / (own_rcount + recent_alpha)

        score_count = count[ids_score]
        score_rcount = rcount[ids_score]
        score_rate = (
            positive[ids_score] + alpha * global_rate
        ) / (score_count + alpha)
        score_recent_rate = (
            rpositive[ids_score] + recent_alpha * recent_global
        ) / (score_rcount + recent_alpha)

        fit_features.extend([
            np.log1p(own_count),
            np.log1p(own_rcount),
            fit_rate,
            fit_recent_rate,
            fit_recent_rate - fit_rate,
        ])
        score_features.extend([
            np.log1p(score_count),
            np.log1p(score_rcount),
            score_rate,
            score_recent_rate,
            score_recent_rate - score_rate,
        ])

        clipped = np.clip(score_recent_rate, 1e-5, 1.0 - 1e-5)
        reliability = score_rcount / (score_rcount + recent_alpha)
        score_logits.append(
            reliability * np.log(clipped / (1.0 - clipped))
        )

    fit_matrix = np.column_stack(fit_features).astype(np.float32)
    score_matrix = np.column_stack(score_features).astype(np.float32)

    # A direct non-parametric family combines independently smoothed
    # video, author, and tag evidence.
    nonparametric_score = np.mean(
        np.column_stack(score_logits), axis=1
    ).astype(np.float64)

    return fit_matrix, score_matrix, nonparametric_score


def training_weights(split):
    recent = recency_weights(split.date, half_life=6.0)
    user_ids = np.asarray(split.user_id, dtype=np.int64)
    counts = np.bincount(
        user_ids,
        minlength=int(FEATURE_CARDINALITIES["user_id"]),
    ).astype(np.float64)
    user_balance = 1.0 / np.sqrt(np.maximum(counts[user_ids], 1.0))
    w = recent * user_balance
    w /= np.mean(w)
    return np.clip(w, 0.1, 8.0).astype(np.float32)


class FieldEmbeddings(nn.Module):
    def __init__(self, cards, dim):
        super().__init__()
        self.tables = nn.ModuleList([
            nn.Embedding(card, dim) for card in cards
        ])
        for table in self.tables:
            nn.init.normal_(table.weight, mean=0.0, std=0.025)

    def forward(self, x):
        return torch.stack(
            [table(x[:, i]) for i, table in enumerate(self.tables)],
            dim=1,
        )


class FieldWeightedFM(nn.Module):
    def __init__(self, cards, n_num, dim=10):
        super().__init__()
        self.emb = FieldEmbeddings(cards, dim)
        self.linear = nn.ModuleList([
            nn.Embedding(card, 1) for card in cards
        ])
        for table in self.linear:
            nn.init.zeros_(table.weight)

        left, right = [], []
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                left.append(i)
                right.append(j)
        self.register_buffer("left", torch.tensor(left, dtype=torch.long))
        self.register_buffer("right", torch.tensor(right, dtype=torch.long))
        self.pair_weight = nn.Parameter(torch.ones(len(left)))
        self.num_head = nn.Sequential(
            nn.Linear(n_num, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, num):
        e = self.emb(x)
        pair_dot = (
            e[:, self.left, :] * e[:, self.right, :]
        ).sum(dim=2)
        interaction = (pair_dot * self.pair_weight).sum(dim=1)
        linear = torch.stack([
            table(x[:, i]).squeeze(1)
            for i, table in enumerate(self.linear)
        ], dim=1).sum(dim=1)
        return self.bias + linear + interaction + self.num_head(num).squeeze(1)


class DCNv2(nn.Module):
    def __init__(self, cards, n_num, dim=6):
        super().__init__()
        self.emb = FieldEmbeddings(cards, dim)
        d = len(cards) * dim + n_num
        rank = 32
        self.u = nn.ModuleList([nn.Linear(d, rank, bias=False) for _ in range(3)])
        self.v = nn.ModuleList([nn.Linear(rank, d, bias=False) for _ in range(3)])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(d)) for _ in range(3)
        ])
        self.deep = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head = nn.Linear(d + 64, 1)

    def forward(self, x, num):
        z0 = torch.cat([self.emb(x).flatten(1), num], dim=1)
        z = z0
        for u, v, bias in zip(self.u, self.v, self.biases):
            cross = v(F.relu(u(z)))
            z = z + z0 * (cross + bias)
        deep = self.deep(z0)
        return self.head(torch.cat([z, deep], dim=1)).squeeze(1)


class FiBiNET(nn.Module):
    def __init__(self, cards, n_num, dim=8):
        super().__init__()
        self.emb = FieldEmbeddings(cards, dim)
        n_fields = len(cards)
        hidden = max(4, n_fields // 3)
        self.squeeze = nn.Sequential(
            nn.Linear(n_fields, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_fields),
            nn.Sigmoid(),
        )
        self.bilinear = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(n_fields)
        ])

        left, right = [], []
        for i in range(n_fields):
            for j in range(i + 1, n_fields):
                left.append(i)
                right.append(j)
        self.left = left
        self.right = right
        input_dim = len(left) * dim + n_num
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x, num):
        e = self.emb(x)
        field_summary = e.mean(dim=2)
        gate = self.squeeze(field_summary).unsqueeze(2)
        e = e * gate

        products = []
        for i, j in zip(self.left, self.right):
            products.append(self.bilinear[i](e[:, i, :]) * e[:, j, :])
        p = torch.cat(products, dim=1)
        return self.head(torch.cat([p, num], dim=1)).squeeze(1)


def build_model(name, n_num):
    if name == "field_weighted_fm":
        return FieldWeightedFM(CARDS, n_num)
    if name == "dcnv2_numeric":
        return DCNv2(CARDS, n_num)
    if name == "fibinet_numeric":
        return FiBiNET(CARDS, n_num)
    raise KeyError(name)


def train_model(name, model, x, num, y, weights, epochs=3):
    model.to(DEVICE)
    lr = 0.0030 if name == "field_weighted_fm" else 0.0018
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=2e-5
    )
    n = len(y)

    for epoch in range(epochs):
        model.train()
        generator = torch.Generator()
        generator.manual_seed(SEED + epoch * 1009 + len(name))
        order = torch.randperm(n, generator=generator).numpy()
        total = 0.0
        seen = 0

        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb = torch.from_numpy(x[idx]).to(DEVICE)
            nb = torch.from_numpy(num[idx]).to(DEVICE)
            yb = torch.from_numpy(y[idx]).to(DEVICE)
            wb = torch.from_numpy(weights[idx]).to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, nb)
            losses = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = torch.sum(losses * wb) / torch.sum(wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total += float(loss.detach()) * len(idx)
            seen += len(idx)

        print(
            "FINDINGS family=%s epoch=%d loss=%.6f"
            % (name, epoch + 1, total / max(seen, 1)),
            flush=True,
        )
    return model


def predict_model(model, x, num):
    model.eval()
    out = np.empty(len(x), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(x), PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, len(x))
            xb = torch.from_numpy(x[start:end]).to(DEVICE)
            nb = torch.from_numpy(num[start:end]).to(DEVICE)
            out[start:end] = model(xb, nb).cpu().numpy()
    return out


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

x_train = make_x(train)
x_valid = make_x(valid)

train_stats, valid_stats, valid_empirical = target_statistics(
    train, y_train, valid, leave_one_out_fit=True
)

train_raw = raw_numeric(train)
valid_raw = raw_numeric(valid)
raw_mean, raw_std = fit_numeric_scaler(train_raw)
train_raw = transform_numeric(train_raw, raw_mean, raw_std)
valid_raw = transform_numeric(valid_raw, raw_mean, raw_std)

stat_mean, stat_std = fit_numeric_scaler(train_stats)
train_stats_scaled = transform_numeric(train_stats, stat_mean, stat_std)
valid_stats_scaled = transform_numeric(valid_stats, stat_mean, stat_std)

num_train = np.ascontiguousarray(
    np.column_stack([train_raw, train_stats_scaled]), dtype=np.float32
)
num_valid = np.ascontiguousarray(
    np.column_stack([valid_raw, valid_stats_scaled]), dtype=np.float32
)
weights = training_weights(train)

artifacts = os.environ["RUN_ARTIFACTS"]
inc_valid = np.asarray(
    np.load(os.path.join(artifacts, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(artifacts, "incumbent_test_scores.npy")),
    dtype=np.float64,
)

if len(inc_valid) != len(y_valid):
    raise RuntimeError("incumbent validation length mismatch")

candidate_scores = {"trusted_incumbent": inc_valid}
candidate_primary = {
    "trusted_incumbent": float(
        evaluate(valid.user_id, y_valid, inc_valid)["primary"]
    )
}
candidate_meta = {
    "trusted_incumbent": ("trusted_incumbent", 0.0)
}

raw_predictions = {
    "recency_empirical_bayes": valid_empirical
}
trained = {}

for name in ["field_weighted_fm", "dcnv2_numeric", "fibinet_numeric"]:
    model = build_model(name, num_train.shape[1])
    model = train_model(
        name, model, x_train, num_train, y_train, weights, epochs=3
    )
    pred = predict_model(model, x_valid, num_valid)
    trained[name] = model
    raw_predictions[name] = pred

    metric = evaluate(valid.user_id, y_valid, pred)
    print(
        "FINDINGS family=%s primary=%.6f gauc=%.6f ndcg5=%.6f"
        % (
            name,
            metric["primary"],
            metric["gauc"],
            metric["ndcg@5"],
        ),
        flush=True,
    )

inc_rank = within_user_rank(valid.user_id, inc_valid)

for family, pred in raw_predictions.items():
    metric = evaluate(valid.user_id, y_valid, pred)
    candidate_scores[family] = pred
    candidate_primary[family] = float(metric["primary"])
    candidate_meta[family] = (family, 1.0)

    pred_rank = within_user_rank(valid.user_id, pred)
    for alpha in [0.15, 0.30, 0.45, 0.60, 0.75]:
        name = "%s_inc_rankblend_%.2f" % (family, alpha)
        score = (1.0 - alpha) * inc_rank + alpha * pred_rank
        metric = evaluate(valid.user_id, y_valid, score)
        candidate_scores[name] = score
        candidate_primary[name] = float(metric["primary"])
        candidate_meta[name] = (family, float(alpha))

best_name = max(candidate_primary, key=candidate_primary.get)
best_valid = candidate_scores[best_name]
best_family, best_alpha = candidate_meta[best_name]
best_metrics = evaluate(valid.user_id, y_valid, best_valid)

print(
    "CANDIDATES " + json.dumps(
        candidate_primary, sort_keys=True, separators=(", ", ": ")
    ),
    flush=True,
)
print(
    "FINDINGS selected=%s family=%s alpha=%.2f"
    % (best_name, best_family, best_alpha),
    flush=True,
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )

# Refit the selected recipe on train + validation before scoring test.
test = load("test")
if len(inc_test) != len(test.user_id):
    raise RuntimeError("incumbent test length mismatch")

if best_family == "trusted_incumbent" or best_alpha == 0.0:
    test_scores = inc_test.copy()
else:
    class CombinedSplit:
        pass

    combined = CombinedSplit()
    combined.X = {
        name: np.concatenate([
            np.asarray(train.X[name], dtype=np.int64),
            np.asarray(valid.X[name], dtype=np.int64),
        ])
        for name in set(FIELDS + STAT_KEYS)
    }
    combined.num = {
        name: np.concatenate([
            np.asarray(train.num[name], dtype=np.float32),
            np.asarray(valid.num[name], dtype=np.float32),
        ])
        for name in RAW_NUMS
    }
    combined.date = np.concatenate([
        np.asarray(train.date, dtype=np.int32),
        np.asarray(valid.date, dtype=np.int32),
    ])
    combined.user_id = combined.X["user_id"]

    y_combined = np.concatenate([
        np.asarray(train.y, dtype=np.float32),
        np.asarray(valid.y, dtype=np.float32),
    ])
    x_combined = concat_x(train, valid)
    x_test = make_x(test)

    combined_stats, test_stats, test_empirical = target_statistics(
        combined, y_combined, test, leave_one_out_fit=True
    )

    combined_raw = np.column_stack([
        np.sign(np.nan_to_num(
            np.asarray(combined.num[name], dtype=np.float64),
            nan=0.0, posinf=0.0, neginf=0.0
        )) * np.log1p(np.abs(np.nan_to_num(
            np.asarray(combined.num[name], dtype=np.float64),
            nan=0.0, posinf=0.0, neginf=0.0
        )))
        for name in RAW_NUMS
    ]).astype(np.float32)
    test_raw = raw_numeric(test)

    c_raw_mean, c_raw_std = fit_numeric_scaler(combined_raw)
    combined_raw = transform_numeric(
        combined_raw, c_raw_mean, c_raw_std
    )
    test_raw = transform_numeric(test_raw, c_raw_mean, c_raw_std)

    c_stat_mean, c_stat_std = fit_numeric_scaler(combined_stats)
    combined_stats = transform_numeric(
        combined_stats, c_stat_mean, c_stat_std
    )
    test_stats = transform_numeric(
        test_stats, c_stat_mean, c_stat_std
    )

    num_combined = np.ascontiguousarray(
        np.column_stack([combined_raw, combined_stats]),
        dtype=np.float32,
    )
    num_test = np.ascontiguousarray(
        np.column_stack([test_raw, test_stats]),
        dtype=np.float32,
    )

    if best_family == "recency_empirical_bayes":
        family_test = test_empirical
    else:
        del trained
        gc.collect()
        selected_model = build_model(best_family, num_combined.shape[1])
        combined_weights = training_weights(combined)
        selected_model = train_model(
            best_family,
            selected_model,
            x_combined,
            num_combined,
            y_combined,
            combined_weights,
            epochs=3,
        )
        family_test = predict_model(selected_model, x_test, num_test)

    if best_alpha >= 1.0:
        test_scores = family_test
    else:
        test_scores = (
            (1.0 - best_alpha)
            * within_user_rank(test.user_id, inc_test)
            + best_alpha
            * within_user_rank(test.user_id, family_test)
        )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS " + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    }, separators=(", ", ": ")),
    flush=True,
)