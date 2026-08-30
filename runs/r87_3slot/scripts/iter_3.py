import os
import time
import json
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2718
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "duration_bucket", "upload_type", "music_type",
    "onehot_feat3", "onehot_feat8", "onehot_feat1",
    "onehot_feat7", "user_active_degree",
    "register_days_bucket", "register_days_range",
    "fans_user_num_range", "follow_user_num_range",
    "friend_user_num_range", "hour", "is_live_streamer",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
WIDE_FIELDS = list(CAT_FIELDS)
WIDE_EPOCHS = 3
WIDE_BATCH = 8192


def make_tree_matrix(split_name, split):
    cols = []
    for f in CAT_FIELDS:
        cols.append(np.asarray(split.X[f], dtype=np.float32))

    for f in NUM_FIELDS:
        x = np.asarray(split.num[f], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))

    for entity in ("video_id", "author_id"):
        h = historical_features(split_name, key=entity)
        selected = [
            k for k in sorted(h)
            if ("count_log1p" in k or "long_view_rate" in k)
        ]
        # Include historical feedback rates as train-only aggregate features.
        selected += [
            k for k in sorted(h)
            if k not in selected and k.endswith("_rate")
        ]
        for k in selected:
            x = np.asarray(h[k], dtype=np.float32)
            cols.append(np.nan_to_num(x, nan=0.0, posinf=0.0,
                                      neginf=0.0).astype(np.float32))
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)


wide_cards = [int(FEATURE_CARDINALITIES[f]) for f in WIDE_FIELDS]
wide_offsets = np.cumsum([0] + wide_cards[:-1], dtype=np.int64)
wide_total = int(sum(wide_cards))


def make_wide_matrix(parts):
    cols = []
    for f, off in zip(WIDE_FIELDS, wide_offsets):
        if len(parts) == 1:
            x = np.asarray(parts[0].X[f], dtype=np.int64)
        else:
            x = np.concatenate([
                np.asarray(p.X[f], dtype=np.int64) for p in parts
            ])
        cols.append(x + off)
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


class WideAdditive(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Embedding(wide_total, 1, sparse=True)
        self.bias = nn.Parameter(torch.zeros(1))
        with torch.no_grad():
            self.weight.weight.zero_()

    def forward(self, x):
        return self.weight(x).squeeze(-1).sum(dim=1) + self.bias


def wide_predict(model, x_np):
    model.eval()
    ans = np.empty(len(x_np), dtype=np.float64)
    x = torch.from_numpy(x_np)
    with torch.no_grad():
        for st in range(0, len(x_np), WIDE_BATCH * 2):
            en = min(st + WIDE_BATCH * 2, len(x_np))
            ans[st:en] = model(x[st:en]).cpu().numpy()
    return ans


def fit_wide(x_np, y_np, epochs, seed):
    torch.manual_seed(seed)
    model = WideAdditive()
    sparse_opt = torch.optim.SparseAdam(
        [model.weight.weight], lr=0.015
    )
    bias_opt = torch.optim.Adam([model.bias], lr=0.005)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))

    for epoch in range(epochs):
        model.train()
        gen = torch.Generator()
        gen.manual_seed(seed + epoch + 1)
        order = torch.randperm(len(x), generator=gen)
        for st in range(0, len(x), WIDE_BATCH):
            idx = order[st:st + WIDE_BATCH]
            sparse_opt.zero_grad(set_to_none=True)
            bias_opt.zero_grad(set_to_none=True)
            logits = model(x[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y[idx]
            )
            loss.backward()
            sparse_opt.step()
            bias_opt.step()
    return model


def group_order(users):
    users = np.asarray(users, dtype=np.int64)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    _, counts = np.unique(sorted_users, return_counts=True)
    return order, counts.astype(np.int32)


def train_binary(x, y, weights, rounds=150):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.06,
        "num_leaves": 63,
        "min_data_in_leaf": 250,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 3.0,
        "max_bin": 127,
        "num_threads": max(1, min(8, os.cpu_count() or 1)),
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "verbose": -1,
    }
    ds = lgb.Dataset(
        x, label=y, weight=weights,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )
    return lgb.train(params, ds, num_boost_round=rounds)


def train_ranker(x, y, users, rounds=130):
    order, groups = group_order(users)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "learning_rate": 0.055,
        "num_leaves": 47,
        "min_data_in_leaf": 220,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.88,
        "bagging_freq": 1,
        "lambda_l2": 4.0,
        "max_bin": 127,
        "lambdarank_truncation_level": 10,
        "num_threads": max(1, min(8, os.cpu_count() or 1)),
        "seed": SEED + 10,
        "feature_fraction_seed": SEED + 11,
        "bagging_seed": SEED + 12,
        "verbose": -1,
    }
    ds = lgb.Dataset(
        np.ascontiguousarray(x[order]),
        label=np.asarray(y, dtype=np.int8)[order],
        group=groups,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )
    model = lgb.train(params, ds, num_boost_round=rounds)
    del order, groups, ds
    gc.collect()
    return model


def score_metrics(users, labels, scores):
    return evaluate(users, labels, np.asarray(scores, dtype=np.float64))


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

x_train = make_tree_matrix("train", train)
x_valid = make_tree_matrix("valid", valid)

# Recency weighting gives the final training days substantially more influence.
age = np.asarray(train.date.max() - train.date, dtype=np.float64)
# YYYYMMDD subtraction is safe within this single April interval.
recency_weight = np.power(0.5, age / 6.0).astype(np.float32)
recency_weight /= np.mean(recency_weight)

binary_model = train_binary(
    x_train, y_train, recency_weight, rounds=150
)
pred_binary = binary_model.predict(x_valid).astype(np.float64)
del binary_model
gc.collect()

rank_model = train_ranker(
    x_train, y_train, train.user_id, rounds=130
)
pred_rank = rank_model.predict(x_valid).astype(np.float64)
del rank_model
gc.collect()

xw_train = make_wide_matrix([train])
xw_valid = make_wide_matrix([valid])
wide_model = fit_wide(xw_train, y_train, WIDE_EPOCHS, SEED + 20)
pred_wide = wide_predict(wide_model, xw_valid)
del wide_model, xw_train, xw_valid
gc.collect()

raw_predictions = {
    "wide_additive": pred_wide,
    "recency_binary_gbdt": pred_binary,
    "lambdamart": pred_rank,
}

candidate_scores = {}
candidate_specs = {}
inc_metric = score_metrics(valid.user_id, y_valid, inc_valid)
candidate_scores["incumbent"] = float(inc_metric["primary"])
candidate_specs["incumbent"] = ("incumbent", 0.0, 1.0)

# Match validation score scales once, then use the exact fixed multiplier on test.
inc_std = float(np.std(inc_valid))
alphas = [0.15, 0.25, 0.35, 0.50, 0.65, 0.80]

best_name = "incumbent"
best_scores = inc_valid.copy()
best_metrics = inc_metric
best_raw = pred_binary
best_spec = candidate_specs["incumbent"]

for family, raw in raw_predictions.items():
    raw_metric = score_metrics(valid.user_id, y_valid, raw)
    candidate_scores[family] = float(raw_metric["primary"])
    candidate_specs[family] = (family, 1.0, 1.0)

    if raw_metric["primary"] > best_metrics["primary"]:
        best_name = family
        best_scores = raw.copy()
        best_metrics = raw_metric
        best_raw = raw.copy()
        best_spec = candidate_specs[family]

    raw_std = max(float(np.std(raw)), 1e-8)
    scale = inc_std / raw_std
    scaled = raw * scale

    for alpha in alphas:
        blended = (1.0 - alpha) * inc_valid + alpha * scaled
        m = score_metrics(valid.user_id, y_valid, blended)
        name = "%s_blend_%.2f" % (family, alpha)
        candidate_scores[name] = float(m["primary"])
        candidate_specs[name] = (family, alpha, scale)
        if m["primary"] > best_metrics["primary"]:
            best_name = name
            best_scores = blended.copy()
            best_metrics = m
            best_raw = raw.copy()
            best_spec = candidate_specs[name]

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True), flush=True)
print(
    "FINDINGS selected=%s binary=%.6f rank=%.6f wide=%.6f incumbent=%.6f"
    % (
        best_name,
        candidate_scores["recency_binary_gbdt"],
        candidate_scores["lambdamart"],
        candidate_scores["wide_additive"],
        candidate_scores["incumbent"],
    ),
    flush=True,
)

# Refit only the selected new family on all labels available before test.
test = load("test")
inc_test = np.load(inc_test_path).astype(np.float64)
selected_family, selected_alpha, selected_scale = best_spec

if selected_family == "incumbent":
    test_scores = inc_test.copy()
else:
    y_tv = np.concatenate([y_train, y_valid]).astype(np.int8)

    if selected_family == "wide_additive":
        xw_tv = make_wide_matrix([train, valid])
        xw_test = make_wide_matrix([test])
        refit = fit_wide(xw_tv, y_tv, WIDE_EPOCHS, SEED + 20)
        test_raw = wide_predict(refit, xw_test)
        del refit, xw_tv, xw_test

    else:
        x_test = make_tree_matrix("test", test)
        x_tv = np.ascontiguousarray(
            np.concatenate([x_train, x_valid], axis=0),
            dtype=np.float32,
        )

        if selected_family == "recency_binary_gbdt":
            all_dates = np.concatenate([
                np.asarray(train.date), np.asarray(valid.date)
            ])
            all_age = np.asarray(
                all_dates.max() - all_dates, dtype=np.float64
            )
            all_weights = np.power(0.5, all_age / 6.0).astype(np.float32)
            all_weights /= np.mean(all_weights)
            refit = train_binary(x_tv, y_tv, all_weights, rounds=150)
        else:
            users_tv = np.concatenate([
                np.asarray(train.user_id),
                np.asarray(valid.user_id),
            ])
            refit = train_ranker(x_tv, y_tv, users_tv, rounds=130)

        test_raw = refit.predict(x_test).astype(np.float64)
        del refit, x_tv, x_test

    if selected_alpha >= 1.0:
        test_scores = test_raw
    else:
        test_scores = (
            (1.0 - selected_alpha) * inc_test
            + selected_alpha * selected_scale * test_raw
        )

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if selected_family != "incumbent" and selected_alpha < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.3f}'
    % (
        best_metrics["primary"],
        best_metrics["gauc"],
        best_metrics["ndcg@5"],
        elapsed,
    )
)