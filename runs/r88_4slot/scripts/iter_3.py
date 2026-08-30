import os
import time
import json
import gc
import math
import random
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 314159
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

CAT_FIELDS = [
    "author_id", "duration_bucket", "fans_user_num_range",
    "follow_user_num_range", "friend_user_num_range", "hour",
    "is_live_streamer", "is_lowactive_period", "is_video_author",
    "music_type", "onehot_feat0", "onehot_feat1", "onehot_feat10",
    "onehot_feat11", "onehot_feat12", "onehot_feat13", "onehot_feat14",
    "onehot_feat15", "onehot_feat16", "onehot_feat17", "onehot_feat2",
    "onehot_feat3", "onehot_feat4", "onehot_feat5", "onehot_feat6",
    "onehot_feat7", "onehot_feat8", "onehot_feat9",
    "register_days_bucket", "register_days_range", "tab", "tag",
    "upload_type", "user_active_degree", "user_id", "video_id",
    "video_type",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
STAT_FIELDS = ["video_id", "author_id", "user_id", "tag"]
CAT_IDX = list(range(len(CAT_FIELDS)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def within_user_rank(user_ids, scores):
    """Ascending percentile rank within each logged impression set."""
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, user_ids))
    su = user_ids[order]

    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = su[1:] != su[:-1]
    starts = np.flatnonzero(new_group)
    group_id = np.cumsum(new_group) - 1
    positions = np.arange(n, dtype=np.int64) - starts[group_id]
    ends = np.r_[starts[1:], n]
    sizes = ends - starts
    denom = sizes[group_id] - 1

    ranked_sorted = np.full(n, 0.5, dtype=np.float64)
    mask = denom > 0
    ranked_sorted[mask] = positions[mask] / denom[mask]
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def concatenate_field(splits, field, numeric=False):
    source = "num" if numeric else "X"
    arrays = [np.asarray(getattr(s, source)[field]) for s in splits]
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays)


def make_stat_arrays(fit_splits, fit_y, eval_split, alpha=20.0):
    """
    Leakage-free leave-one-out rates/counts on fit rows and full-fit rates on
    evaluation rows.
    """
    prior = float(np.mean(fit_y))
    fit_stats = []
    eval_stats = []
    rates_for_eb_fit = []
    rates_for_eb_eval = []

    for field in STAT_FIELDS:
        fit_ids = concatenate_field(fit_splits, field).astype(np.int64, copy=False)
        eval_ids = np.asarray(eval_split.X[field], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[field])

        cnt = np.bincount(fit_ids, minlength=card).astype(np.float64)
        pos = np.bincount(
            fit_ids, weights=fit_y.astype(np.float64), minlength=card
        ).astype(np.float64)

        loo_cnt = cnt[fit_ids] - 1.0
        loo_pos = pos[fit_ids] - fit_y
        fit_rate = (loo_pos + alpha * prior) / (loo_cnt + alpha)
        eval_rate = (pos[eval_ids] + alpha * prior) / (cnt[eval_ids] + alpha)

        fit_count = np.log1p(np.maximum(loo_cnt, 0.0))
        eval_count = np.log1p(cnt[eval_ids])

        fit_stats.extend([
            fit_count.astype(np.float32),
            fit_rate.astype(np.float32),
        ])
        eval_stats.extend([
            eval_count.astype(np.float32),
            eval_rate.astype(np.float32),
        ])
        rates_for_eb_fit.append(fit_rate)
        rates_for_eb_eval.append(eval_rate)

    return fit_stats, eval_stats, rates_for_eb_fit, rates_for_eb_eval


def build_features(fit_splits, fit_y, eval_split):
    n_fit = len(fit_y)
    n_eval = len(eval_split.user_id)
    cols_fit = []
    cols_eval = []

    for field in CAT_FIELDS:
        cols_fit.append(
            concatenate_field(fit_splits, field).astype(np.float32, copy=False)
        )
        cols_eval.append(
            np.asarray(eval_split.X[field], dtype=np.float32)
        )

    for field in NUM_FIELDS:
        a = concatenate_field(fit_splits, field, numeric=True).astype(
            np.float64, copy=False
        )
        b = np.asarray(eval_split.num[field], dtype=np.float64)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        cols_fit.append(np.log1p(np.maximum(a, 0.0)).astype(np.float32))
        cols_eval.append(np.log1p(np.maximum(b, 0.0)).astype(np.float32))

    fit_stats, eval_stats, eb_fit_rates, eb_eval_rates = make_stat_arrays(
        fit_splits, fit_y, eval_split
    )
    cols_fit.extend(fit_stats)
    cols_eval.extend(eval_stats)

    X_fit = np.empty((n_fit, len(cols_fit)), dtype=np.float32)
    X_eval = np.empty((n_eval, len(cols_eval)), dtype=np.float32)
    for j, col in enumerate(cols_fit):
        X_fit[:, j] = col
    for j, col in enumerate(cols_eval):
        X_eval[:, j] = col

    # Stable, interpretable non-parametric family.
    eb_weights = np.asarray([0.42, 0.25, 0.23, 0.10], dtype=np.float64)
    eb_eval = np.zeros(n_eval, dtype=np.float64)
    for w, rate in zip(eb_weights, eb_eval_rates):
        eb_eval += w * logit(rate)

    return X_fit, X_eval, eb_eval


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int64)
    unique_dates = np.unique(dates)
    day_index = {int(d): i for i, d in enumerate(unique_dates)}
    age = (len(unique_dates) - 1) - np.asarray(
        [day_index[int(d)] for d in dates], dtype=np.float32
    )
    # Eight-day half-life: material adaptation without discarding early users.
    w = np.exp2(-age / 8.0).astype(np.float32)
    return w / np.mean(w)


def train_binary(X, y, weights, rounds=210):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.055,
        "num_leaves": 39,
        "min_data_in_leaf": 500,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 2.0,
        "max_bin": 63,
        "cat_smooth": 30.0,
        "cat_l2": 15.0,
        "num_threads": min(8, max(1, os.cpu_count() or 1)),
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "verbose": -1,
    }
    ds = lgb.Dataset(
        X, label=y, weight=weights,
        categorical_feature=CAT_IDX, free_raw_data=False
    )
    model = lgb.train(params, ds, num_boost_round=rounds)
    return model


def train_lambdarank(X, y, users, weights=None, rounds=165):
    users = np.asarray(users)
    row = np.arange(len(users), dtype=np.int64)
    order = np.lexsort((row, users))
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    groups = np.diff(np.r_[starts, len(users)]).astype(np.int32)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "lambdarank_truncation_level": 10,
        "label_gain": [0, 1],
        "learning_rate": 0.045,
        "num_leaves": 31,
        "min_data_in_leaf": 400,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.88,
        "bagging_freq": 1,
        "lambda_l2": 3.0,
        "max_bin": 63,
        "cat_smooth": 30.0,
        "cat_l2": 15.0,
        "num_threads": min(8, max(1, os.cpu_count() or 1)),
        "seed": SEED + 20,
        "feature_fraction_seed": SEED + 21,
        "bagging_seed": SEED + 22,
        "verbose": -1,
    }
    sorted_weights = None if weights is None else weights[order]
    ds = lgb.Dataset(
        X[order], label=y[order], weight=sorted_weights, group=groups,
        categorical_feature=CAT_IDX, free_raw_data=False
    )
    model = lgb.train(params, ds, num_boost_round=rounds)
    return model


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, rank, prior):
        super().__init__()
        self.user = nn.Embedding(n_users, rank, sparse=True)
        self.item = nn.Embedding(n_items, rank, sparse=True)
        self.user_bias = nn.Embedding(n_users, 1, sparse=True)
        self.item_bias = nn.Embedding(n_items, 1, sparse=True)
        nn.init.normal_(self.user.weight, std=0.035)
        nn.init.normal_(self.item.weight, std=0.035)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        prior = np.clip(prior, 1e-5, 1.0 - 1e-5)
        self.register_buffer(
            "intercept",
            torch.tensor(math.log(prior / (1.0 - prior)), dtype=torch.float32)
        )

    def forward(self, u, v):
        dot = (self.user(u) * self.item(v)).sum(dim=1)
        return (
            self.intercept + dot
            + self.user_bias(u).squeeze(1)
            + self.item_bias(v).squeeze(1)
        )


def fit_mf(user_ids, video_ids, y, epochs=4):
    torch.manual_seed(SEED + 40)
    model = MatrixFactorization(
        int(FEATURE_CARDINALITIES["user_id"]),
        int(FEATURE_CARDINALITIES["video_id"]),
        24,
        float(np.mean(y)),
    )
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.018)
    criterion = nn.BCEWithLogitsLoss()
    u = torch.from_numpy(np.asarray(user_ids, dtype=np.int64))
    v = torch.from_numpy(np.asarray(video_ids, dtype=np.int64))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    batch = 16384
    n = len(y)

    for _ in range(epochs):
        order = torch.randperm(n)
        model.train()
        for lo in range(0, n, batch):
            idx = order[lo:min(lo + batch, n)]
            pred = model(u[idx], v[idx])
            loss = criterion(pred, yt[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def predict_mf(model, user_ids, video_ids):
    u = np.asarray(user_ids, dtype=np.int64)
    v = np.asarray(video_ids, dtype=np.int64)
    out = np.empty(len(u), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for lo in range(0, len(u), 32768):
            hi = min(lo + 32768, len(u))
            out[lo:hi] = model(
                torch.from_numpy(u[lo:hi]),
                torch.from_numpy(v[lo:hi]),
            ).numpy()
    return out


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)

X_train, X_valid, eb_valid = build_features([train], y_train, valid)
date_weights = recency_weights(train.date)

binary_model = train_binary(X_train, y_train, date_weights)
binary_valid = binary_model.predict(X_valid).astype(np.float64)

lambda_model = train_lambdarank(
    X_train, y_train, np.asarray(train.user_id), weights=None
)
lambda_valid = lambda_model.predict(X_valid).astype(np.float64)

mf_model = fit_mf(train.user_id, train.video_id, y_train, epochs=4)
mf_valid = predict_mf(mf_model, valid.user_id, valid.video_id).astype(np.float64)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_rank_valid = within_user_rank(valid_users, inc_valid)

families_valid = {
    "gbdt_binary_recency": binary_valid,
    "lambdarank": lambda_valid,
    "matrix_factorization": mf_valid,
    "empirical_bayes": eb_valid,
}

candidate_scores = {}
candidate_arrays = {}
candidate_meta = {}

inc_metrics = evaluate(valid_users, y_valid, inc_valid)
candidate_scores["incumbent"] = float(inc_metrics["primary"])
candidate_arrays["incumbent"] = inc_valid
candidate_meta["incumbent"] = ("incumbent", 0.0)

for name, raw in families_valid.items():
    m_raw = evaluate(valid_users, y_valid, raw)
    candidate_scores[name] = float(m_raw["primary"])
    candidate_arrays[name] = raw
    candidate_meta[name] = (name, 1.0)

    own_rank = within_user_rank(valid_users, raw)
    for alpha in (0.20, 0.35, 0.50, 0.65, 0.80):
        blended = (1.0 - alpha) * inc_rank_valid + alpha * own_rank
        key = f"{name}_blend_{alpha:.2f}"
        m = evaluate(valid_users, y_valid, blended)
        candidate_scores[key] = float(m["primary"])
        candidate_arrays[key] = blended
        candidate_meta[key] = (name, alpha)

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = candidate_arrays[winner]
metrics = evaluate(valid_users, y_valid, valid_scores)
winner_family, winner_alpha = candidate_meta[winner]
own_valid_scores = (
    families_valid[winner_family]
    if winner_family != "incumbent"
    else inc_valid
)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS selected=%s family=%s alpha=%.2f"
    % (winner, winner_family, winner_alpha)
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if winner_alpha not in (0.0, 1.0):
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(own_valid_scores, dtype=np.float64),
        )

# Remove validation-fit matrices/models before the required train+validation refit.
del X_train, X_valid, binary_model, lambda_model, mf_model
gc.collect()

test = load("test")
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

if winner_family == "incumbent":
    test_scores = inc_test
else:
    y_fit = np.concatenate([
        y_train,
        np.asarray(valid.y, dtype=np.float32),
    ])
    fit_users = np.concatenate([
        np.asarray(train.user_id),
        np.asarray(valid.user_id),
    ])
    fit_videos = np.concatenate([
        np.asarray(train.video_id),
        np.asarray(valid.video_id),
    ])

    if winner_family in ("gbdt_binary_recency", "lambdarank", "empirical_bayes"):
        X_fit, X_test, eb_test = build_features([train, valid], y_fit, test)

        if winner_family == "gbdt_binary_recency":
            fit_dates = np.concatenate([
                np.asarray(train.date),
                np.asarray(valid.date),
            ])
            model = train_binary(
                X_fit, y_fit, recency_weights(fit_dates), rounds=210
            )
            own_test = model.predict(X_test).astype(np.float64)
        elif winner_family == "lambdarank":
            model = train_lambdarank(
                X_fit, y_fit, fit_users, weights=None, rounds=165
            )
            own_test = model.predict(X_test).astype(np.float64)
        else:
            own_test = eb_test.astype(np.float64)

        del X_fit, X_test
        gc.collect()
    else:
        model = fit_mf(fit_users, fit_videos, y_fit, epochs=4)
        own_test = predict_mf(
            model, test.user_id, test.video_id
        ).astype(np.float64)

    if winner_alpha == 1.0:
        test_scores = own_test
    else:
        test_scores = (
            (1.0 - winner_alpha)
            * within_user_rank(np.asarray(test.user_id), inc_test)
            + winner_alpha
            * within_user_rank(np.asarray(test.user_id), own_test)
        )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS " + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(", ", ": "),
    )
)