import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7319
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "hour", "tag", "upload_type", "music_type", "user_active_degree",
    "is_live_streamer", "is_video_author", "follow_user_num_range",
    "fans_user_num_range", "friend_user_num_range", "register_days_range",
    "onehot_feat3", "onehot_feat7", "onehot_feat8", "video_type",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
SEQ_LEN = 12
EMBED_DIM = 8
BATCH_SIZE = 4096
PRED_BATCH = 16384
DIN_EPOCHS = 2
DIN_HALF_LIFE = 5.0


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    su = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = su[1:] != su[:-1]
    start_pos = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = su[:-1] != su[1:]
    end_idx = np.flatnonzero(ends)
    sizes = np.diff(np.concatenate((np.array([-1]), end_idx)))
    size_rows = np.repeat(sizes, sizes)

    pos = np.arange(n, dtype=np.int64) - start_pos
    ranked = (pos.astype(np.float64) + 0.5) / size_rows
    out = np.empty(n, dtype=np.float64)
    out[order] = ranked
    return out


def recency_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.int32)
    age = dates.max() - dates
    w = np.power(0.5, age.astype(np.float32) / half_life).astype(np.float32)
    return w / max(float(w.mean()), 1e-8)


def categorical_matrix(split):
    cards = [FEATURE_CARDINALITIES[f] for f in FIELDS]
    offsets = np.cumsum(np.asarray([0] + cards[:-1], dtype=np.int64))
    x = np.stack([
        np.asarray(split.X[f], dtype=np.int64) + offsets[j]
        for j, f in enumerate(FIELDS)
    ], axis=1)
    return np.ascontiguousarray(x, dtype=np.int64)


def make_positive_history(train, future_splits):
    u = np.asarray(train.X["user_id"], dtype=np.int64)
    v = np.asarray(train.X["video_id"], dtype=np.int64)
    y = np.asarray(train.y, dtype=np.int8)
    t = np.asarray(train.time_ms, dtype=np.int64)
    n = len(train)
    n_users = FEATURE_CARDINALITIES["user_id"]

    order = np.lexsort((np.arange(n, dtype=np.int64), t, u))
    su = u[order]
    sv = v[order]
    sy = y[order].astype(np.int64)

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = su[1:] != su[:-1]
    start_idx = np.flatnonzero(starts)

    cs = np.cumsum(sy, dtype=np.int64)
    base_at_start = cs[start_idx] - sy[start_idx]
    group_sizes = np.diff(np.append(start_idx, n))
    base = np.repeat(base_at_start, group_sizes)
    positives_before_sorted = cs - base - sy

    positives_before = np.empty(n, dtype=np.int64)
    positives_before[order] = positives_before_sorted

    positive_items = sv[sy == 1]
    pos_counts = np.bincount(su[sy == 1], minlength=n_users).astype(np.int64)
    pos_starts = np.cumsum(
        np.concatenate((np.array([0], dtype=np.int64), pos_counts[:-1]))
    )

    train_hist = np.zeros((n, SEQ_LEN), dtype=np.int32)
    for lag in range(SEQ_LEN):
        target = positives_before - 1 - lag
        ok = target >= 0
        idx = pos_starts[u[ok]] + target[ok]
        train_hist[ok, lag] = positive_items[idx].astype(np.int32)

    future_histories = []
    for split in future_splits:
        fu = np.asarray(split.X["user_id"], dtype=np.int64)
        m = len(split)
        hist = np.zeros((m, SEQ_LEN), dtype=np.int32)
        available = pos_counts[fu]
        for lag in range(SEQ_LEN):
            target = available - 1 - lag
            ok = target >= 0
            idx = pos_starts[fu[ok]] + target[ok]
            hist[ok, lag] = positive_items[idx].astype(np.int32)
        future_histories.append(hist)

    return train_hist, future_histories, pos_counts


class DIN(nn.Module):
    def __init__(self, total_features, n_fields, initial_bias):
        super().__init__()
        self.feature_embedding = nn.Embedding(total_features, EMBED_DIM)
        self.video_embedding = nn.Embedding(
            FEATURE_CARDINALITIES["video_id"], EMBED_DIM, padding_idx=0
        )
        self.position_embedding = nn.Embedding(SEQ_LEN, EMBED_DIM)

        base_dim = n_fields * EMBED_DIM
        self.attention = nn.Sequential(
            nn.Linear(4 * EMBED_DIM, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(base_dim + 3 * EMBED_DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.linear = nn.Embedding(total_features, 1)
        self.bias = nn.Parameter(torch.tensor(initial_bias, dtype=torch.float32))

        nn.init.normal_(self.feature_embedding.weight, std=0.02)
        nn.init.normal_(self.video_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x, candidate_video, history):
        base_emb = self.feature_embedding(x)
        q = self.video_embedding(candidate_video)
        h = self.video_embedding(history)

        positions = torch.arange(
            SEQ_LEN, device=x.device, dtype=torch.long
        ).unsqueeze(0)
        h = h + self.position_embedding(positions)

        q_rep = q.unsqueeze(1).expand(-1, SEQ_LEN, -1)
        attn_input = torch.cat(
            (q_rep, h, q_rep - h, q_rep * h), dim=-1
        )
        logits = self.attention(attn_input).squeeze(-1)
        mask = history != 0
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = (weights.unsqueeze(-1) * h).sum(dim=1)

        deep_input = torch.cat(
            (base_emb.flatten(1), q, pooled, q * pooled), dim=1
        )
        deep = self.head(deep_input).squeeze(-1)
        wide = self.linear(x).sum(dim=1).squeeze(-1)
        return self.bias + wide + deep


def train_din(model, x, videos, history, y, weights):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=1e-6
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")
    n = len(y)
    generator = torch.Generator().manual_seed(SEED + 17)

    model.train()
    for epoch in range(DIN_EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            logits = model(x[idx], videos[idx], history[idx])
            loss = (bce(logits, y[idx]) * weights[idx]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def predict_din(model, x_np, videos_np, history_np):
    model.eval()
    result = np.empty(len(x_np), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(x_np), PRED_BATCH):
            end = min(start + PRED_BATCH, len(x_np))
            xb = torch.from_numpy(x_np[start:end])
            vb = torch.from_numpy(videos_np[start:end])
            hb = torch.from_numpy(history_np[start:end].astype(
                np.int64, copy=False
            ))
            result[start:end] = model(xb, vb, hb).cpu().numpy()
    return result


def find_rate_key(hist, entity):
    preferred = f"{entity}_long_view_rate"
    if preferred in hist:
        return preferred
    matches = [k for k in hist if "long_view_rate" in k]
    if not matches:
        raise RuntimeError("No long_view_rate in historical features")
    return matches[0]


def empirical_score(video_hist, author_hist):
    vk = find_rate_key(video_hist, "video_id")
    ak = find_rate_key(author_hist, "author_id")
    vr = np.nan_to_num(
        np.asarray(video_hist[vk], dtype=np.float64), nan=0.3366
    )
    ar = np.nan_to_num(
        np.asarray(author_hist[ak], dtype=np.float64), nan=0.3366
    )
    vr = np.clip(vr, 1e-4, 1 - 1e-4)
    ar = np.clip(ar, 1e-4, 1 - 1e-4)
    return (
        0.62 * np.log(vr / (1.0 - vr))
        + 0.38 * np.log(ar / (1.0 - ar))
    )


def lgb_matrix(split, video_hist, author_hist):
    columns = []
    categorical_indices = []

    for f in FIELDS:
        categorical_indices.append(len(columns))
        columns.append(np.asarray(split.X[f], dtype=np.float32))

    for f in NUM_FIELDS:
        z = np.asarray(split.num[f], dtype=np.float32)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(z, 0.0)).astype(np.float32))

    for hist in (video_hist, author_hist):
        for key in sorted(hist.keys()):
            z = np.asarray(hist[key], dtype=np.float32)
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            columns.append(z)

    return (
        np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.float32),
        categorical_indices,
    )


train = load("train")
valid = load("valid")
test = load("test")

y_train_np = np.asarray(train.y, dtype=np.float32)
positive_rate = float(y_train_np.mean())
initial_bias = float(np.log(positive_rate / (1.0 - positive_rate)))

# Family 1: chronological positive-history DIN.
x_train_np = categorical_matrix(train)
x_valid_np = categorical_matrix(valid)
x_test_np = categorical_matrix(test)

train_hist_seq, future_histories, pos_counts = make_positive_history(
    train, [valid, test]
)
valid_hist_seq, test_hist_seq = future_histories

train_video = np.asarray(train.X["video_id"], dtype=np.int64)
valid_video = np.asarray(valid.X["video_id"], dtype=np.int64)
test_video = np.asarray(test.X["video_id"], dtype=np.int64)

total_features = int(sum(FEATURE_CARDINALITIES[f] for f in FIELDS))
din = DIN(total_features, len(FIELDS), initial_bias)
din_weights_np = recency_weights(train.date, DIN_HALF_LIFE)

din = train_din(
    din,
    torch.from_numpy(x_train_np),
    torch.from_numpy(train_video),
    torch.from_numpy(train_hist_seq.astype(np.int64, copy=False)),
    torch.from_numpy(y_train_np),
    torch.from_numpy(din_weights_np),
)
din_valid = predict_din(
    din, x_valid_np, valid_video, valid_hist_seq
)
din_test = predict_din(
    din, x_test_np, test_video, test_hist_seq
)

# Free sequence training arrays before constructing the boosted-tree table.
del train_hist_seq, future_histories, x_train_np
gc.collect()

# Families 2 and 3 use leakage-safe train-only historical entity statistics.
vh_train = historical_features("train", key="video_id")
ah_train = historical_features("train", key="author_id")
vh_valid = historical_features("valid", key="video_id")
ah_valid = historical_features("valid", key="author_id")
vh_test = historical_features("test", key="video_id")
ah_test = historical_features("test", key="author_id")

emp_valid = empirical_score(vh_valid, ah_valid)
emp_test = empirical_score(vh_test, ah_test)

# User-grouped LambdaMART directly optimizes top-ranked relevance.
lgb_train_x, cat_indices = lgb_matrix(train, vh_train, ah_train)
lgb_valid_x, _ = lgb_matrix(valid, vh_valid, ah_valid)
lgb_test_x, _ = lgb_matrix(test, vh_test, ah_test)

train_user = np.asarray(train.user_id, dtype=np.int64)
train_order = np.argsort(train_user, kind="stable")
sorted_users = train_user[train_order]
_, group_counts = np.unique(sorted_users, return_counts=True)

rank_weights = recency_weights(train.date, 4.0)
dtrain = lgb.Dataset(
    lgb_train_x[train_order],
    label=y_train_np[train_order],
    weight=rank_weights[train_order],
    group=group_counts,
    categorical_feature=cat_indices,
    free_raw_data=True,
)

params = {
    "objective": "lambdarank",
    "metric": "None",
    "label_gain": [0, 1],
    "lambdarank_truncation_level": 5,
    "learning_rate": 0.045,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 180,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.82,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 1.0,
    "max_bin": 127,
    "num_threads": min(16, os.cpu_count() or 1),
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "verbose": -1,
}
ranker = lgb.train(params, dtrain, num_boost_round=320)
lambda_valid = ranker.predict(lgb_valid_x)
lambda_test = ranker.predict(lgb_test_x)

del lgb_train_x, lgb_valid_x, lgb_test_x, dtrain
gc.collect()

own_valid = {
    "din": np.asarray(din_valid, dtype=np.float64),
    "lambdamart": np.asarray(lambda_valid, dtype=np.float64),
    "empirical_bayes": np.asarray(emp_valid, dtype=np.float64),
}
own_test = {
    "din": np.asarray(din_test, dtype=np.float64),
    "lambdamart": np.asarray(lambda_test, dtype=np.float64),
    "empirical_bayes": np.asarray(emp_test, dtype=np.float64),
}

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = rank_percentile(valid.user_id, inc_valid)
inc_test_rank = rank_percentile(test.user_id, inc_test)

candidate_scores = {}
candidate_metrics = {}
candidate_spec = {}

for family in ("din", "lambdamart", "empirical_bayes"):
    standalone_key = family + "_standalone"
    candidate_scores[standalone_key] = own_valid[family]
    candidate_metrics[standalone_key] = evaluate(
        valid.user_id, valid.y, own_valid[family]
    )
    candidate_spec[standalone_key] = (family, None)

    vr = rank_percentile(valid.user_id, own_valid[family])
    for alpha in (0.20, 0.40, 0.60, 0.80):
        key = f"{family}_blend_{alpha:.2f}"
        score = alpha * vr + (1.0 - alpha) * inc_valid_rank
        candidate_scores[key] = score
        candidate_metrics[key] = evaluate(valid.user_id, valid.y, score)
        candidate_spec[key] = (family, alpha)

best_key = max(
    candidate_metrics,
    key=lambda k: float(candidate_metrics[k]["primary"])
)
best_metrics = candidate_metrics[best_key]
best_valid = candidate_scores[best_key]
best_family, best_alpha = candidate_spec[best_key]

if best_alpha is None:
    best_test = own_test[best_family]
else:
    own_test_rank = rank_percentile(test.user_id, own_test[best_family])
    best_test = (
        best_alpha * own_test_rank
        + (1.0 - best_alpha) * inc_test_rank
    )

summary = {
    k: float(v["primary"]) for k, v in candidate_metrics.items()
}
print("CANDIDATES " + json.dumps(summary, sort_keys=True))
print("FINDINGS " + json.dumps({
    "best_candidate": best_key,
    "din_history_length": SEQ_LEN,
    "din_half_life_days": DIN_HALF_LIFE,
    "lambdamart_half_life_days": 4.0,
    "train_users_with_positive_history": int(np.sum(pos_counts > 0)),
    "mean_train_positive_history_count": float(pos_counts.mean()),
}, sort_keys=True))

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
    if best_alpha is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(own_valid[best_family], dtype=np.float64),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))