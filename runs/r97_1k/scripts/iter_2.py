import os
import gc
import json
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")
START = time.time()
SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")
if OUT:
    os.makedirs(OUT, exist_ok=True)

# Low-cardinality fields are passed natively to LightGBM. High-cardinality
# identities are represented through leakage-safe train histories and
# temporally weighted leave-one-out target statistics.
LOW_CAT = [
    "tab", "tag", "duration_bucket", "upload_type", "hour",
    "music_type", "onehot_feat1", "onehot_feat3", "onehot_feat7",
    "onehot_feat8", "user_active_degree", "video_type",
]
TE_FIELDS = [
    "video_id", "author_id", "tag", "tab", "duration_bucket",
    "upload_type", "onehot_feat3", "onehot_feat8",
]
DEEP_CAT = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "duration_bucket", "upload_type", "onehot_feat3",
    "onehot_feat8", "music_type",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]

HALF_LIFE = 4.0
TE_ALPHA = {
    "video_id": 30.0,
    "author_id": 60.0,
    "tag": 200.0,
    "tab": 300.0,
    "duration_bucket": 300.0,
    "upload_type": 300.0,
    "onehot_feat3": 120.0,
    "onehot_feat8": 120.0,
}


def finite32(x, fill=0.0):
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float32), 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


train = load("train")
n_train = len(train.user_id)
train_y = np.asarray(train.y, dtype=np.float32)
max_date = int(np.max(train.date))

# Proximity weighting is normalized to keep the effective loss scale stable.
age = (max_date - np.asarray(train.date, dtype=np.int32)).astype(np.float32)
train_weight = np.power(0.5, age / HALF_LIFE).astype(np.float32)
train_weight /= float(np.mean(train_weight))
weighted_prior = float(np.sum(train_weight * train_y) / np.sum(train_weight))

# Create train-only weighted entity maps. The train feature itself is
# leave-one-out; validation/test use all train rows.
te_maps = {}
train_te = {}
for field in TE_FIELDS:
    ids = np.asarray(train.X[field], dtype=np.int64)
    card = int(FEATURE_CARDINALITIES[field])
    sw = np.bincount(ids, weights=train_weight, minlength=card).astype(np.float32)
    sy = np.bincount(
        ids, weights=train_weight * train_y, minlength=card
    ).astype(np.float32)
    alpha = float(TE_ALPHA[field])

    den = sw[ids] - train_weight + alpha
    rate = (
        sy[ids] - train_weight * train_y + alpha * weighted_prior
    ) / np.maximum(den, 1e-6)
    cnt = np.log1p(np.maximum(sw[ids] - train_weight, 0.0)).astype(np.float32)

    train_te[field] = (rate.astype(np.float32), cnt)
    te_maps[field] = (sw, sy, alpha)


def external_te(split, field):
    ids = np.asarray(split.X[field], dtype=np.int64)
    sw, sy, alpha = te_maps[field]
    ok = ids < len(sw)
    rate = np.full(len(ids), weighted_prior, dtype=np.float32)
    count = np.zeros(len(ids), dtype=np.float32)
    rate[ok] = (
        sy[ids[ok]] + alpha * weighted_prior
    ) / np.maximum(sw[ids[ok]] + alpha, 1e-6)
    count[ok] = np.log1p(sw[ids[ok]])
    return rate, count


# Select informative histories without assuming their exact dictionary order.
h_video_train = historical_features("train", key="video_id")
h_author_train = historical_features("train", key="author_id")


def choose_history_keys(d):
    preferred = []
    for k in sorted(d.keys()):
        kl = k.lower()
        if (
            "long_view_rate" in kl
            or "count_log1p" in kl
            or "is_click_rate" in kl
            or "play_time_ms_logmean" in kl
            or "comment_stay_time_logmean" in kl
        ):
            preferred.append(k)
    if not preferred:
        preferred = sorted(d.keys())[:5]
    return preferred[:5]


VIDEO_HKEYS = choose_history_keys(h_video_train)
AUTHOR_HKEYS = choose_history_keys(h_author_train)


def build_matrix(split, split_name, hv, ha, is_train=False):
    cols = []

    for field in LOW_CAT:
        cols.append(np.asarray(split.X[field], dtype=np.float32))

    for k in VIDEO_HKEYS:
        cols.append(finite32(hv[k]))
    for k in AUTHOR_HKEYS:
        cols.append(finite32(ha[k]))

    for field in NUM_FIELDS:
        z = finite32(split.num[field])
        z = np.maximum(z, 0.0)
        cols.append(np.log1p(z).astype(np.float32))

    empirical_parts = []
    for field in TE_FIELDS:
        if is_train:
            rate, count = train_te[field]
        else:
            rate, count = external_te(split, field)
        cols.append(logit(rate).astype(np.float32))
        cols.append(count.astype(np.float32))
        if field in ("video_id", "author_id", "tag", "duration_bucket"):
            empirical_parts.append(logit(rate))

    # Relative training date lets trees learn residual short-term drift. Future
    # rows receive the boundary value rather than extrapolated dates.
    if is_train:
        recency = -age / 7.0
    else:
        recency = np.zeros(len(split.user_id), dtype=np.float32)
    cols.append(np.asarray(recency, dtype=np.float32))

    X = np.column_stack(cols).astype(np.float32, copy=False)

    # Fixed, train-independent combination of distinct empirical Bayes rates.
    empirical = (
        0.40 * empirical_parts[0]
        + 0.35 * empirical_parts[1]
        + 0.15 * empirical_parts[2]
        + 0.10 * empirical_parts[3]
    ).astype(np.float32)
    return X, empirical


X_train, empirical_train = build_matrix(
    train, "train", h_video_train, h_author_train, is_train=True
)
del h_video_train, h_author_train, empirical_train, train_te
gc.collect()

deep_offsets = np.cumsum(
    [0] + [int(FEATURE_CARDINALITIES[f]) for f in DEEP_CAT[:-1]]
).astype(np.int64)
deep_total_card = int(sum(int(FEATURE_CARDINALITIES[f]) for f in DEEP_CAT))

deep_cat_train = np.column_stack([
    np.asarray(train.X[f], dtype=np.int32) + int(off)
    for f, off in zip(DEEP_CAT, deep_offsets)
]).astype(np.int32, copy=False)

n_low = len(LOW_CAT)
deep_num_train = X_train[:, n_low:]
num_mean = np.mean(deep_num_train, axis=0, dtype=np.float64).astype(np.float32)
num_std = np.std(deep_num_train, axis=0, dtype=np.float64).astype(np.float32)
num_std = np.maximum(num_std, 1e-3)


class DeepFM(nn.Module):
    def __init__(self, total_card, n_fields, n_num, rank=10):
        super().__init__()
        self.emb = nn.Embedding(total_card, rank + 1, sparse=True)
        with torch.no_grad():
            self.emb.weight[:, 0].zero_()
            self.emb.weight[:, 1:].normal_(0.0, 0.015)
        self.deep = nn.Sequential(
            nn.Linear(n_fields * rank + n_num, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.num_linear = nn.Linear(n_num, 1)
        self.register_buffer(
            "bias",
            torch.tensor(
                np.log(weighted_prior / (1.0 - weighted_prior)),
                dtype=torch.float32,
            ),
        )

    def forward(self, cat, num):
        e = self.emb(cat)
        linear = e[:, :, 0].sum(dim=1) + self.num_linear(num).squeeze(1)
        v = e[:, :, 1:]
        fm = 0.5 * (
            v.sum(dim=1).square() - v.square().sum(dim=1)
        ).sum(dim=1)
        deep = self.deep(
            torch.cat([v.reshape(v.shape[0], -1), num], dim=1)
        ).squeeze(1)
        return self.bias + linear + fm + deep


deep_model = DeepFM(
    deep_total_card,
    len(DEEP_CAT),
    deep_num_train.shape[1],
    rank=10,
)
sparse_opt = torch.optim.SparseAdam([deep_model.emb.weight], lr=0.003)
dense_params = [
    p for name, p in deep_model.named_parameters()
    if name != "emb.weight"
]
dense_opt = torch.optim.AdamW(dense_params, lr=0.0015, weight_decay=1e-5)

batch_size = 16384
generator = torch.Generator().manual_seed(SEED)
deep_model.train()

# Two full passes provide a meaningful deep-family comparison while temporal
# weighting concentrates both passes on rows closest to deployment.
for epoch in range(2):
    perm = torch.randperm(n_train, generator=generator).numpy()
    total_loss = 0.0
    seen = 0
    for start in range(0, n_train, batch_size):
        idx = perm[start:start + batch_size]
        cat = torch.from_numpy(deep_cat_train[idx].astype(np.int64, copy=False))
        num_np = (
            deep_num_train[idx] - num_mean
        ) / num_std
        num_np = np.clip(num_np, -8.0, 8.0).astype(np.float32, copy=False)
        num = torch.from_numpy(num_np)
        yb = torch.from_numpy(train_y[idx])
        wb = torch.from_numpy(train_weight[idx])

        sparse_opt.zero_grad(set_to_none=True)
        dense_opt.zero_grad(set_to_none=True)
        logits = deep_model(cat, num)
        losses = F.binary_cross_entropy_with_logits(
            logits, yb, reduction="none"
        )
        loss = torch.sum(losses * wb) / torch.sum(wb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dense_params, 5.0)
        sparse_opt.step()
        dense_opt.step()

        total_loss += float(loss.detach()) * len(idx)
        seen += len(idx)
    print(
        "deepfm_epoch=%d weighted_logloss=%.6f"
        % (epoch + 1, total_loss / max(seen, 1)),
        flush=True,
    )
    del perm
    gc.collect()


# Main boosted model: temporal sample weighting is applied directly to its loss.
lgb_train = lgb.Dataset(
    X_train,
    label=train_y,
    weight=train_weight,
    categorical_feature=list(range(n_low)),
    free_raw_data=True,
)
lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 1000,
    "max_bin": 127,
    "feature_fraction": 0.90,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 2.0,
    "max_cat_to_onehot": 16,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "num_threads": min(16, os.cpu_count() or 1),
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "verbose": -1,
}
gbm = lgb.train(lgb_params, lgb_train, num_boost_round=280)

del lgb_train, X_train, deep_cat_train, deep_num_train
del train_y, train_weight, age, train
gc.collect()


def build_deep_cat(split):
    return np.column_stack([
        np.asarray(split.X[f], dtype=np.int32) + int(off)
        for f, off in zip(DEEP_CAT, deep_offsets)
    ]).astype(np.int32, copy=False)


@torch.inference_mode()
def predict_deep(split, X):
    deep_model.eval()
    cats = build_deep_cat(split)
    nums = X[:, n_low:]
    result = np.empty(len(split.user_id), dtype=np.float32)
    pred_bs = 131072
    for start in range(0, len(result), pred_bs):
        end = min(start + pred_bs, len(result))
        cat = torch.from_numpy(cats[start:end].astype(np.int64, copy=False))
        num_np = (nums[start:end] - num_mean) / num_std
        num_np = np.clip(num_np, -8.0, 8.0).astype(np.float32, copy=False)
        result[start:end] = deep_model(
            cat, torch.from_numpy(num_np)
        ).cpu().numpy()
    del cats
    return result


valid = load("valid")
h_video_valid = historical_features("valid", key="video_id")
h_author_valid = historical_features("valid", key="author_id")
X_valid, empirical_valid = build_matrix(
    valid, "valid", h_video_valid, h_author_valid, is_train=False
)
del h_video_valid, h_author_valid
gc.collect()

lgb_valid = gbm.predict(X_valid, num_iteration=gbm.current_iteration()).astype(
    np.float32
)
deep_valid = predict_deep(valid, X_valid)
valid_uid = np.asarray(valid.user_id)
valid_y = np.asarray(valid.y)


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
    group_start = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local_rank = np.arange(n, dtype=np.float32) - group_start.astype(np.float32)

    end_flag = np.empty(n, dtype=np.bool_)
    end_flag[-1] = True
    end_flag[:-1] = sorted_users[:-1] != sorted_users[1:]
    ends = np.flatnonzero(end_flag)
    sizes_per_group = np.diff(np.r_[-1, ends]).astype(np.float32)
    group_index = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes_per_group[group_index] - 1.0, 1.0)

    ranked_sorted = local_rank / denom
    result = np.empty(n, dtype=np.float32)
    result[order] = ranked_sorted
    return result


inc_valid_path = (
    os.path.join(SHARED, "incumbent_valid_scores.npy")
    if SHARED else ""
)
inc_test_path = (
    os.path.join(SHARED, "incumbent_test_scores.npy")
    if SHARED else ""
)
has_incumbent = bool(
    inc_valid_path and os.path.exists(inc_valid_path)
    and inc_test_path and os.path.exists(inc_test_path)
)

families_valid = {
    "temporal_lgbm": lgb_valid,
    "temporal_deepfm": deep_valid,
    "empirical_bayes": empirical_valid,
}

candidate_scores = {}
candidate_metrics = {}
best_name = None
best_valid_scores = None
best_own_valid = None
best_family = None
best_weight = 1.0
best_primary = -np.inf

for name, scores in families_valid.items():
    met = evaluate(valid_uid, valid_y, scores)
    candidate_scores[name] = float(met["primary"])
    candidate_metrics[name] = met
    if float(met["primary"]) > best_primary:
        best_primary = float(met["primary"])
        best_name = name
        best_valid_scores = scores.copy()
        best_own_valid = scores.copy()
        best_family = name
        best_weight = 1.0

if has_incumbent:
    incumbent_valid = np.load(inc_valid_path, mmap_mode="r")
    inc_rank_valid = within_user_rank(valid_uid, incumbent_valid)
    incumbent_met = evaluate(valid_uid, valid_y, incumbent_valid)
    candidate_scores["trusted_incumbent"] = float(incumbent_met["primary"])

    for name, scores in families_valid.items():
        own_rank = within_user_rank(valid_uid, scores)
        local_best = -np.inf
        local_w = None
        for w in (0.15, 0.30, 0.45, 0.60, 0.75, 0.90):
            blended = w * own_rank + (1.0 - w) * inc_rank_valid
            met = evaluate(valid_uid, valid_y, blended)
            if float(met["primary"]) > local_best:
                local_best = float(met["primary"])
                local_w = float(w)
            if float(met["primary"]) > best_primary:
                best_primary = float(met["primary"])
                best_name = name + "_rankblend_w%.2f" % w
                best_valid_scores = blended.astype(np.float32)
                best_own_valid = scores.copy()
                best_family = name
                best_weight = float(w)
        candidate_scores[name + "_best_blend"] = local_best
        print(
            "FINDINGS %s best_incumbent_rank_blend_weight=%.2f primary=%.6f"
            % (name, local_w, local_best),
            flush=True,
        )

    # Do not regress if every exploratory family is harmful.
    if float(incumbent_met["primary"]) > best_primary:
        best_primary = float(incumbent_met["primary"])
        best_name = "trusted_incumbent"
        best_valid_scores = np.asarray(incumbent_valid, dtype=np.float32).copy()
        best_own_valid = lgb_valid.copy()
        best_family = "trusted_incumbent"
        best_weight = 0.0

final_metrics = evaluate(valid_uid, valid_y, best_valid_scores)

print(
    "FINDINGS winner=%s lgb_features=%d history_video=%s history_author=%s"
    % (
        best_name,
        X_valid.shape[1],
        ",".join(VIDEO_HKEYS),
        ",".join(AUTHOR_HKEYS),
    ),
    flush=True,
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True), flush=True)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    if best_weight < 1.0:
        np.save(
            os.path.join(OUT, "scores_valid_raw.npy"),
            np.asarray(best_own_valid, dtype=np.float64),
        )

del X_valid, valid_y, best_valid_scores
del empirical_valid, lgb_valid, deep_valid
gc.collect()

test = load("test")

if best_family == "trusted_incumbent":
    test_scores = np.asarray(
        np.load(inc_test_path, mmap_mode="r"), dtype=np.float32
    ).copy()
else:
    h_video_test = historical_features("test", key="video_id")
    h_author_test = historical_features("test", key="author_id")
    X_test, empirical_test = build_matrix(
        test, "test", h_video_test, h_author_test, is_train=False
    )
    del h_video_test, h_author_test
    gc.collect()

    if best_family == "temporal_lgbm":
        own_test = gbm.predict(
            X_test, num_iteration=gbm.current_iteration()
        ).astype(np.float32)
    elif best_family == "temporal_deepfm":
        own_test = predict_deep(test, X_test)
    elif best_family == "empirical_bayes":
        own_test = empirical_test.astype(np.float32)
    else:
        raise RuntimeError("Unknown winning family: " + str(best_family))

    if best_weight < 1.0 and has_incumbent:
        incumbent_test = np.load(inc_test_path, mmap_mode="r")
        own_rank_test = within_user_rank(test.user_id, own_test)
        inc_rank_test = within_user_rank(test.user_id, incumbent_test)
        test_scores = (
            best_weight * own_rank_test
            + (1.0 - best_weight) * inc_rank_test
        ).astype(np.float32)
    else:
        test_scores = own_test

    del X_test, empirical_test
    gc.collect()

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(final_metrics["primary"]),
            "gauc": float(final_metrics["gauc"]),
            "ndcg@5": float(final_metrics["ndcg@5"]),
            "gpu_seconds": elapsed,
        }
    )
)