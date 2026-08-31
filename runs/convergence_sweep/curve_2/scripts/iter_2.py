import os
import time
import json
import math
import random
import copy
import gc

import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "user_active_degree",
    "onehot_feat3",
    "onehot_feat8",
    "hour",
    "register_days_bucket",
    "music_type",
    "video_type",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

train = load("train")
valid = load("valid")
test = load("test")


def cat_matrix(split):
    return np.ascontiguousarray(
        np.stack(
            [np.asarray(split.X[f], dtype=np.int64) for f in CAT_FIELDS],
            axis=1,
        ),
        dtype=np.int64,
    )


xcat_tr = cat_matrix(train)
xcat_va = cat_matrix(valid)
xcat_te = cat_matrix(test)


def transformed_raw_numeric(split):
    cols = []
    for f in NUM_FIELDS:
        a = np.asarray(split.num[f], dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(np.log1p(np.maximum(a, 0.0)))
    return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.float32)


rawnum_tr = transformed_raw_numeric(train)
rawnum_va = transformed_raw_numeric(valid)
rawnum_te = transformed_raw_numeric(test)

raw_mean = rawnum_tr.mean(axis=0, dtype=np.float64).astype(np.float32)
raw_std = rawnum_tr.std(axis=0, dtype=np.float64).astype(np.float32)
raw_std = np.maximum(raw_std, 1e-3)

rawnum_tr = (rawnum_tr - raw_mean) / raw_std
rawnum_va = (rawnum_va - raw_mean) / raw_std
rawnum_te = (rawnum_te - raw_mean) / raw_std


def get_histories(split_name):
    vd = historical_features(split_name, key="video_id")
    au = historical_features(split_name, key="author_id")
    merged = {}
    for k, v in vd.items():
        merged["video__" + k] = np.asarray(v, dtype=np.float32)
    for k, v in au.items():
        merged["author__" + k] = np.asarray(v, dtype=np.float32)
    return merged


hist_tr_dict = get_histories("train")
hist_va_dict = get_histories("valid")
hist_te_dict = get_histories("test")

hist_keys = sorted(
    set(hist_tr_dict.keys())
    & set(hist_va_dict.keys())
    & set(hist_te_dict.keys())
)


def history_matrix(d):
    if not hist_keys:
        return np.empty((len(next(iter(d.values()))), 0), dtype=np.float32)
    ans = np.stack(
        [
            np.nan_to_num(
                np.asarray(d[k], dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            for k in hist_keys
        ],
        axis=1,
    )
    return np.ascontiguousarray(ans, dtype=np.float32)


hist_tr = history_matrix(hist_tr_dict)
hist_va = history_matrix(hist_va_dict)
hist_te = history_matrix(hist_te_dict)

hist_mean = hist_tr.mean(axis=0, dtype=np.float64).astype(np.float32)
hist_std = hist_tr.std(axis=0, dtype=np.float64).astype(np.float32)
hist_std = np.maximum(hist_std, 1e-3)

hist_tr_scaled = np.clip((hist_tr - hist_mean) / hist_std, -8.0, 8.0)
hist_va_scaled = np.clip((hist_va - hist_mean) / hist_std, -8.0, 8.0)
hist_te_scaled = np.clip((hist_te - hist_mean) / hist_std, -8.0, 8.0)

xnum_tr = np.ascontiguousarray(
    np.concatenate([rawnum_tr, hist_tr_scaled], axis=1),
    dtype=np.float32,
)
xnum_va = np.ascontiguousarray(
    np.concatenate([rawnum_va, hist_va_scaled], axis=1),
    dtype=np.float32,
)
xnum_te = np.ascontiguousarray(
    np.concatenate([rawnum_te, hist_te_scaled], axis=1),
    dtype=np.float32,
)

y_tr = np.asarray(train.y, dtype=np.float32)
y_va = np.asarray(valid.y, dtype=np.int8)

# Recent training days receive greater influence because the label rate and
# logging regime move substantially across the date boundary.
age_days = np.asarray(train.date.max() - train.date, dtype=np.float32)
date_weights = np.exp(-math.log(2.0) * age_days / 7.0).astype(np.float32)
date_weights /= date_weights.mean()


def find_long_view_key(prefix):
    matches = [
        k for k in hist_keys
        if k.startswith(prefix) and "long_view_rate" in k
    ]
    if not matches:
        matches = [
            k for k in hist_keys
            if k.startswith(prefix) and "long_view" in k and "rate" in k
        ]
    if not matches:
        raise RuntimeError("No long-view history key found for " + prefix)
    return matches[0]


video_rate_key = find_long_view_key("video__")
author_rate_key = find_long_view_key("author__")


def empirical_scores(hdict):
    vr = np.clip(np.asarray(hdict[video_rate_key], dtype=np.float64), 1e-4, 1 - 1e-4)
    ar = np.clip(np.asarray(hdict[author_rate_key], dtype=np.float64), 1e-4, 1 - 1e-4)
    vlogit = np.log(vr) - np.log1p(-vr)
    alogit = np.log(ar) - np.log1p(-ar)
    return (0.68 * vlogit + 0.32 * alogit).astype(np.float32)


predictions_valid = {}
predictions_test = {}

predictions_valid["empirical_bayes"] = empirical_scores(hist_va_dict)
predictions_test["empirical_bayes"] = empirical_scores(hist_te_dict)


# Gradient-boosted tree families share the same categorical, numeric, and
# train-only historical inputs.
order = np.argsort(np.asarray(train.user_id), kind="stable")
sorted_users = np.asarray(train.user_id)[order]
_, group_counts = np.unique(sorted_users, return_counts=True)

x_lgb_tr = np.ascontiguousarray(
    np.concatenate(
        [xcat_tr[order].astype(np.float32), xnum_tr[order]],
        axis=1,
    ),
    dtype=np.float32,
)
x_lgb_va = np.ascontiguousarray(
    np.concatenate([xcat_va.astype(np.float32), xnum_va], axis=1),
    dtype=np.float32,
)
x_lgb_te = np.ascontiguousarray(
    np.concatenate([xcat_te.astype(np.float32), xnum_te], axis=1),
    dtype=np.float32,
)

categorical_indices = list(range(len(CAT_FIELDS)))
common_lgb = {
    "learning_rate": 0.045,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 120,
    "feature_fraction": 0.86,
    "bagging_fraction": 0.88,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "min_data_per_group": 80,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "max_cat_to_onehot": 16,
    "verbosity": -1,
    "verbose": -1,
    "num_threads": min(8, os.cpu_count() or 1),
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "data_random_seed": SEED,
    "deterministic": True,
    "force_col_wise": True,
}

binary_params = dict(common_lgb)
binary_params.update({
    "objective": "binary",
    "metric": "binary_logloss",
})

d_binary = lgb.Dataset(
    x_lgb_tr,
    label=y_tr[order],
    weight=date_weights[order],
    categorical_feature=categorical_indices,
    free_raw_data=True,
)
binary_model = lgb.train(
    binary_params,
    d_binary,
    num_boost_round=260,
)
predictions_valid["lightgbm_binary"] = binary_model.predict(
    x_lgb_va, num_iteration=binary_model.best_iteration
).astype(np.float32)
predictions_test["lightgbm_binary"] = binary_model.predict(
    x_lgb_te, num_iteration=binary_model.best_iteration
).astype(np.float32)
del d_binary, binary_model
gc.collect()

rank_params = dict(common_lgb)
rank_params.update({
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5],
    "lambdarank_truncation_level": 8,
    "label_gain": [0, 1],
})

d_rank = lgb.Dataset(
    x_lgb_tr,
    label=y_tr[order],
    weight=date_weights[order],
    group=group_counts,
    categorical_feature=categorical_indices,
    free_raw_data=True,
)
rank_model = lgb.train(
    rank_params,
    d_rank,
    num_boost_round=220,
)
predictions_valid["lightgbm_lambdarank"] = rank_model.predict(
    x_lgb_va, num_iteration=rank_model.best_iteration
).astype(np.float32)
predictions_test["lightgbm_lambdarank"] = rank_model.predict(
    x_lgb_te, num_iteration=rank_model.best_iteration
).astype(np.float32)
del d_rank, rank_model, x_lgb_tr, x_lgb_va, x_lgb_te, order, sorted_users
gc.collect()


class DeepFM(nn.Module):
    def __init__(self, cardinalities, n_numeric, embed_dim=8):
        super().__init__()
        offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
        self.register_buffer("offsets", torch.from_numpy(offsets))
        total = int(sum(cardinalities))
        self.linear_cat = nn.Embedding(total, 1)
        self.embedding = nn.Embedding(total, embed_dim)
        self.linear_num = nn.Linear(n_numeric, 1)
        deep_input = len(cardinalities) * embed_dim + n_numeric
        self.deep = nn.Sequential(
            nn.Linear(deep_input, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear_cat.weight)
        nn.init.normal_(self.embedding.weight, std=0.015)
        nn.init.zeros_(self.linear_num.bias)

    def forward(self, xcat, xnum):
        x = xcat + self.offsets
        emb = self.embedding(x)
        linear = self.linear_cat(x).sum(dim=1).squeeze(-1)
        linear = linear + self.linear_num(xnum).squeeze(-1)

        summed = emb.sum(dim=1)
        fm = 0.5 * (
            summed.square() - emb.square().sum(dim=1)
        ).sum(dim=1)

        deep_input = torch.cat([emb.flatten(1), xnum], dim=1)
        deep = self.deep(deep_input).squeeze(-1)
        return self.bias + linear + fm + deep


@torch.no_grad()
def deep_predict(model, xc, xn, batch_size=16384):
    model.eval()
    ans = np.empty(xc.shape[0], dtype=np.float32)
    for start in range(0, xc.shape[0], batch_size):
        end = min(start + batch_size, xc.shape[0])
        logits = model(
            torch.from_numpy(xc[start:end]),
            torch.from_numpy(xn[start:end]),
        )
        ans[start:end] = logits.cpu().numpy()
    return ans


cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS]
deep_model = DeepFM(cardinalities, xnum_tr.shape[1], embed_dim=8)
optimizer = torch.optim.AdamW(
    deep_model.parameters(),
    lr=1.2e-3,
    weight_decay=2e-6,
)

txcat = torch.from_numpy(xcat_tr)
txnum = torch.from_numpy(xnum_tr)
ty = torch.from_numpy(y_tr)
tw = torch.from_numpy(date_weights)

generator = torch.Generator()
generator.manual_seed(SEED)
batch_size = 4096
best_deep_primary = -np.inf
best_deep_state = None
best_deep_valid = None
best_deep_epoch = -1

for epoch in range(5):
    deep_model.train()
    perm = torch.randperm(len(y_tr), generator=generator)
    for start in range(0, len(y_tr), batch_size):
        idx = perm[start:start + batch_size]
        logits = deep_model(txcat[idx], txnum[idx])
        losses = nn.functional.binary_cross_entropy_with_logits(
            logits, ty[idx], reduction="none"
        )
        loss = (losses * tw[idx]).sum() / tw[idx].sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(deep_model.parameters(), 5.0)
        optimizer.step()

    current_valid = deep_predict(deep_model, xcat_va, xnum_va)
    current_metrics = evaluate(valid.user_id, valid.y, current_valid)
    if current_metrics["primary"] > best_deep_primary:
        best_deep_primary = float(current_metrics["primary"])
        best_deep_epoch = epoch + 1
        best_deep_valid = current_valid.copy()
        best_deep_state = {
            k: v.detach().cpu().clone()
            for k, v in deep_model.state_dict().items()
        }

deep_model.load_state_dict(best_deep_state)
predictions_valid["deepfm"] = best_deep_valid
predictions_test["deepfm"] = deep_predict(deep_model, xcat_te, xnum_te)

del txcat, txnum, ty, tw, deep_model, optimizer
gc.collect()


shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = None
inc_test = None
if shared:
    iv = os.path.join(shared, "incumbent_valid_scores.npy")
    it = os.path.join(shared, "incumbent_test_scores.npy")
    if os.path.exists(iv) and os.path.exists(it):
        inc_valid = np.asarray(np.load(iv), dtype=np.float64)
        inc_test = np.asarray(np.load(it), dtype=np.float64)


def standardized(a):
    a = np.asarray(a, dtype=np.float64)
    center = float(np.mean(a))
    scale = float(np.std(a))
    if not np.isfinite(scale) or scale < 1e-10:
        scale = 1.0
    return np.clip((a - center) / scale, -8.0, 8.0)


candidate_scores = {}
candidate_metrics = {}
candidate_payload = {}

for name in predictions_valid:
    va_pred = np.asarray(predictions_valid[name], dtype=np.float64)
    te_pred = np.asarray(predictions_test[name], dtype=np.float64)
    met = evaluate(valid.user_id, valid.y, va_pred)
    candidate_scores[name] = float(met["primary"])
    candidate_metrics[name] = met
    candidate_payload[name] = (va_pred, te_pred, name, None)

    if inc_valid is not None:
        zvi = standardized(inc_valid)
        zti = standardized(inc_test)
        zvn = standardized(va_pred)
        ztn = standardized(te_pred)

        best_blend_score = -np.inf
        best_blend = None
        best_alpha = None
        best_blend_metrics = None
        for alpha in np.linspace(0.10, 0.90, 17):
            blend_valid = (1.0 - alpha) * zvi + alpha * zvn
            blend_metrics = evaluate(valid.user_id, valid.y, blend_valid)
            if blend_metrics["primary"] > best_blend_score:
                best_blend_score = float(blend_metrics["primary"])
                best_alpha = float(alpha)
                best_blend = blend_valid.copy()
                best_blend_metrics = blend_metrics

        blend_name = name + "_inc_blend"
        blend_test = (1.0 - best_alpha) * zti + best_alpha * ztn
        candidate_scores[blend_name] = best_blend_score
        candidate_metrics[blend_name] = best_blend_metrics
        candidate_payload[blend_name] = (
            best_blend,
            blend_test,
            name,
            best_alpha,
        )

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores, test_scores, raw_family, selected_alpha = candidate_payload[winner]
best_metrics = candidate_metrics[winner]

print("FINDINGS " + json.dumps({
    "history_features": len(hist_keys),
    "deepfm_best_epoch": int(best_deep_epoch),
    "winner": winner,
    "winner_raw_family": raw_family,
    "winner_incumbent_alpha_for_new_model": selected_alpha,
}, sort_keys=True))
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if selected_alpha is not None:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(predictions_valid[raw_family], dtype=np.float64),
        )

elapsed = time.time() - START
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result))