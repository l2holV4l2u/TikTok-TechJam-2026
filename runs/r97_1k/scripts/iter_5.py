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
SEED = 314159
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")
if OUT:
    os.makedirs(OUT, exist_ok=True)

LOW_CAT = [
    "tab", "tag", "duration_bucket", "upload_type", "hour",
    "music_type", "onehot_feat1", "onehot_feat3", "onehot_feat7",
    "onehot_feat8", "user_active_degree", "video_type",
]
MODEL_CAT = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "duration_bucket", "upload_type", "onehot_feat3",
    "onehot_feat8", "music_type",
]
TE_FIELDS = [
    "video_id", "author_id", "tag", "tab", "duration_bucket",
    "upload_type", "onehot_feat3", "onehot_feat8",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
TE_ALPHA = {
    "video_id": 25.0,
    "author_id": 50.0,
    "tag": 180.0,
    "tab": 250.0,
    "duration_bucket": 250.0,
    "upload_type": 250.0,
    "onehot_feat3": 100.0,
    "onehot_feat8": 100.0,
}
HALF_LIFE = 4.0
SEQ_LEN = 12


def finite32(x):
    return np.nan_to_num(
        np.asarray(x, dtype=np.float32),
        nan=0.0, posinf=0.0, neginf=0.0,
    )


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float32), 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


def choose_history_keys(d):
    preferred = []
    for k in sorted(d):
        kl = k.lower()
        if (
            "long_view_rate" in kl
            or "count_log1p" in kl
            or "is_click_rate" in kl
            or "play_time_ms_logmean" in kl
            or "comment_stay_time_logmean" in kl
        ):
            preferred.append(k)
    return (preferred if preferred else sorted(d))[:5]


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, user_ids))
    su = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = su[1:] != su[:-1]
    group_start = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local = np.arange(n, dtype=np.float32) - group_start.astype(np.float32)

    ends_flag = np.empty(n, dtype=bool)
    ends_flag[-1] = True
    ends_flag[:-1] = su[:-1] != su[1:]
    ends = np.flatnonzero(ends_flag)
    sizes = np.diff(np.r_[-1, ends]).astype(np.float32)
    gid = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[gid] - 1.0, 1.0)

    ranked = np.empty(n, dtype=np.float32)
    ranked[order] = local / denom
    return ranked


train = load("train")
train_y_original = np.asarray(train.y, dtype=np.float32)
max_date = int(np.max(train.date))
age_original = (
    max_date - np.asarray(train.date, dtype=np.int32)
).astype(np.float32)
weight_original = np.power(0.5, age_original / HALF_LIFE).astype(np.float32)
weight_original /= float(np.mean(weight_original))
prior = float(
    np.sum(weight_original * train_y_original) / np.sum(weight_original)
)

# Leakage-safe, temporally weighted target statistics.
te_maps = {}
train_te = {}
for field in TE_FIELDS:
    ids = np.asarray(train.X[field], dtype=np.int64)
    card = int(FEATURE_CARDINALITIES[field])
    sw = np.bincount(
        ids, weights=weight_original, minlength=card
    ).astype(np.float32)
    sy = np.bincount(
        ids, weights=weight_original * train_y_original, minlength=card
    ).astype(np.float32)
    alpha = float(TE_ALPHA[field])

    loo_w = np.maximum(sw[ids] - weight_original, 0.0)
    loo_y = sy[ids] - weight_original * train_y_original
    rate = (loo_y + alpha * prior) / np.maximum(loo_w + alpha, 1e-6)
    train_te[field] = (
        rate.astype(np.float32),
        np.log1p(loo_w).astype(np.float32),
    )
    te_maps[field] = (sw, sy, alpha)


def external_te(split, field):
    ids = np.asarray(split.X[field], dtype=np.int64)
    sw, sy, alpha = te_maps[field]
    ok = (ids >= 0) & (ids < len(sw))
    rate = np.full(len(ids), prior, dtype=np.float32)
    count = np.zeros(len(ids), dtype=np.float32)
    rate[ok] = (
        sy[ids[ok]] + alpha * prior
    ) / np.maximum(sw[ids[ok]] + alpha, 1e-6)
    count[ok] = np.log1p(sw[ids[ok]])
    return rate, count


hv_train = historical_features("train", key="video_id")
ha_train = historical_features("train", key="author_id")
VIDEO_HKEYS = choose_history_keys(hv_train)
AUTHOR_HKEYS = choose_history_keys(ha_train)


def build_matrix(split, hv, ha, is_train=False):
    cols = []
    for field in LOW_CAT:
        cols.append(np.asarray(split.X[field], dtype=np.float32))

    for k in VIDEO_HKEYS:
        cols.append(finite32(hv[k]))
    for k in AUTHOR_HKEYS:
        cols.append(finite32(ha[k]))

    for field in NUM_FIELDS:
        cols.append(np.log1p(np.maximum(finite32(split.num[field]), 0.0)))

    for field in TE_FIELDS:
        if is_train:
            rate, count = train_te[field]
        else:
            rate, count = external_te(split, field)
        cols.append(logit(rate).astype(np.float32))
        cols.append(count.astype(np.float32))

    if is_train:
        cols.append((-age_original / 7.0).astype(np.float32))
    else:
        cols.append(np.zeros(len(split.user_id), dtype=np.float32))

    return np.column_stack(cols).astype(np.float32, copy=False)


X_original = build_matrix(train, hv_train, ha_train, is_train=True)
del hv_train, ha_train, train_te
gc.collect()

# Sort once by user and true chronological order. This supplies valid ranking
# groups and also permits vectorized construction of prior-positive histories.
n_train = len(train.user_id)
row_original = np.arange(n_train, dtype=np.int64)
sort_order = np.lexsort((
    row_original,
    np.asarray(train.time_ms, dtype=np.int64),
    np.asarray(train.user_id, dtype=np.int64),
))
sorted_uid = np.asarray(train.user_id, dtype=np.int64)[sort_order]
train_y = train_y_original[sort_order]
train_weight = weight_original[sort_order]
X_train = X_original[sort_order]
del X_original, train_y_original, weight_original, age_original
gc.collect()

starts = np.empty(n_train, dtype=bool)
starts[0] = True
starts[1:] = sorted_uid[1:] != sorted_uid[:-1]
group_starts = np.flatnonzero(starts)
group_sizes = np.diff(np.r_[group_starts, n_train]).astype(np.int32)
group_id = np.cumsum(starts, dtype=np.int32) - 1
num_groups = len(group_sizes)

sorted_video = np.asarray(train.X["video_id"], dtype=np.int32)[sort_order]
positive_counts = np.bincount(
    group_id, weights=train_y, minlength=num_groups
).astype(np.int64)
positive_starts = np.r_[0, np.cumsum(positive_counts[:-1])].astype(np.int64)
positive_videos = sorted_video[train_y > 0.5]

global_positive_cum = np.cumsum(train_y.astype(np.int64))
prior_positive_count = (
    global_positive_cum
    - positive_starts[group_id]
    - train_y.astype(np.int64)
)

train_history = np.zeros((n_train, SEQ_LEN), dtype=np.int32)
for lag in range(1, SEQ_LEN + 1):
    idx = positive_starts[group_id] + prior_positive_count - lag
    ok = idx >= positive_starts[group_id]
    train_history[ok, lag - 1] = positive_videos[idx[ok]]

# Final train-only user histories are used for both validation and test.
user_card = int(FEATURE_CARDINALITIES["user_id"])
final_history = np.zeros((user_card, SEQ_LEN), dtype=np.int32)
group_users = sorted_uid[group_starts]
for lag in range(1, SEQ_LEN + 1):
    idx = positive_starts + positive_counts - lag
    ok = idx >= positive_starts
    final_history[group_users[ok], lag - 1] = positive_videos[idx[ok]]

cat_offsets = np.cumsum(
    [0] + [int(FEATURE_CARDINALITIES[f]) for f in MODEL_CAT[:-1]]
).astype(np.int64)
total_card = int(sum(int(FEATURE_CARDINALITIES[f]) for f in MODEL_CAT))
video_field_index = MODEL_CAT.index("video_id")
video_offset = int(cat_offsets[video_field_index])

cat_train = np.column_stack([
    np.asarray(train.X[f], dtype=np.int32)[sort_order] + int(off)
    for f, off in zip(MODEL_CAT, cat_offsets)
]).astype(np.int32, copy=False)

n_low = len(LOW_CAT)
nn_train_num = X_train[:, n_low:]
num_mean = np.mean(nn_train_num, axis=0, dtype=np.float64).astype(np.float32)
num_std = np.std(nn_train_num, axis=0, dtype=np.float64).astype(np.float32)
num_std = np.maximum(num_std, 1e-3)

click_key = "is_click" if "is_click" in train.aux else sorted(train.aux.keys())[0]
train_click = np.asarray(train.aux[click_key], dtype=np.float32)[sort_order]

# Family 1: query-aware boosted trees optimized for top-ranked positives.
rank_dataset = lgb.Dataset(
    X_train,
    label=train_y,
    weight=train_weight,
    group=group_sizes,
    categorical_feature=list(range(n_low)),
    free_raw_data=True,
)
rank_params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10],
    "lambdarank_truncation_level": 10,
    "lambdarank_norm": True,
    "label_gain": [0, 1],
    "learning_rate": 0.045,
    "num_leaves": 63,
    "min_data_in_leaf": 800,
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
ranker = lgb.train(rank_params, rank_dataset, num_boost_round=240)
del rank_dataset
gc.collect()


def normalized_num(x):
    z = (x - num_mean) / num_std
    return np.clip(z, -8.0, 8.0).astype(np.float32, copy=False)


# Family 2: DIN-style attention over leakage-safe prior positive videos.
class DIN(nn.Module):
    def __init__(self, card, fields, n_num, dim=10):
        super().__init__()
        self.emb = nn.Embedding(card, dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        self.attn = nn.Sequential(
            nn.Linear(dim * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(fields * dim + dim + n_num, 160),
            nn.ReLU(),
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.bias = nn.Parameter(torch.tensor(
            np.log(prior / (1.0 - prior)), dtype=torch.float32
        ))

    def forward(self, cat, num, history_ids):
        fields = self.emb(cat)
        candidate = fields[:, video_field_index, :]
        hist = self.emb(history_ids)
        cand = candidate.unsqueeze(1).expand_as(hist)
        attn_input = torch.cat(
            [hist, cand, hist - cand, hist * cand], dim=-1
        )
        logits = self.attn(attn_input).squeeze(-1)
        mask = history_ids != video_offset
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        interest = torch.sum(weights.unsqueeze(-1) * hist, dim=1)
        z = torch.cat(
            [fields.flatten(1), interest, num], dim=1
        )
        return self.bias + self.mlp(z).squeeze(1)


# Family 3: MMoE jointly learns long-view and click representations.
class MMoE(nn.Module):
    def __init__(self, card, fields, n_num, dim=8, n_experts=4):
        super().__init__()
        self.emb = nn.Embedding(card, dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        input_dim = fields * dim + n_num
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 48),
                nn.ReLU(),
            )
            for _ in range(n_experts)
        ])
        self.gate_long = nn.Linear(input_dim, n_experts)
        self.gate_click = nn.Linear(input_dim, n_experts)
        self.head_long = nn.Linear(48, 1)
        self.head_click = nn.Linear(48, 1)
        self.long_bias = nn.Parameter(torch.tensor(
            np.log(prior / (1.0 - prior)), dtype=torch.float32
        ))

    def forward(self, cat, num):
        x = torch.cat([self.emb(cat).flatten(1), num], dim=1)
        experts = torch.stack([e(x) for e in self.experts], dim=1)
        gl = torch.softmax(self.gate_long(x), dim=1).unsqueeze(-1)
        gc = torch.softmax(self.gate_click(x), dim=1).unsqueeze(-1)
        long_rep = torch.sum(gl * experts, dim=1)
        click_rep = torch.sum(gc * experts, dim=1)
        long_logit = self.long_bias + self.head_long(long_rep).squeeze(1)
        click_logit = self.head_click(click_rep).squeeze(1)
        return long_logit, click_logit


din = DIN(
    total_card, len(MODEL_CAT), nn_train_num.shape[1], dim=10
)
mmoe = MMoE(
    total_card, len(MODEL_CAT), nn_train_num.shape[1], dim=8, n_experts=4
)
din_opt = torch.optim.AdamW(din.parameters(), lr=0.0015, weight_decay=1e-6)
mmoe_opt = torch.optim.AdamW(mmoe.parameters(), lr=0.0015, weight_decay=1e-6)

batch_size = 16384
generator = torch.Generator().manual_seed(SEED)

# One full, temporally weighted pass per structurally different neural family.
din.train()
perm = torch.randperm(n_train, generator=generator).numpy()
din_loss_sum = 0.0
for start in range(0, n_train, batch_size):
    idx = perm[start:start + batch_size]
    cat = torch.from_numpy(cat_train[idx].astype(np.int64, copy=False))
    num = torch.from_numpy(normalized_num(nn_train_num[idx]))
    hist = torch.from_numpy(
        train_history[idx].astype(np.int64, copy=False) + video_offset
    )
    y = torch.from_numpy(train_y[idx])
    w = torch.from_numpy(train_weight[idx])

    din_opt.zero_grad(set_to_none=True)
    logits = din(cat, num, hist)
    losses = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    loss = torch.sum(losses * w) / torch.sum(w)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(din.parameters(), 5.0)
    din_opt.step()
    din_loss_sum += float(loss.detach()) * len(idx)
print(
    "FINDINGS din_weighted_logloss=%.6f"
    % (din_loss_sum / n_train),
    flush=True,
)
del perm
gc.collect()

mmoe.train()
perm = torch.randperm(n_train, generator=generator).numpy()
mmoe_loss_sum = 0.0
for start in range(0, n_train, batch_size):
    idx = perm[start:start + batch_size]
    cat = torch.from_numpy(cat_train[idx].astype(np.int64, copy=False))
    num = torch.from_numpy(normalized_num(nn_train_num[idx]))
    y_long = torch.from_numpy(train_y[idx])
    y_click = torch.from_numpy(train_click[idx])
    w = torch.from_numpy(train_weight[idx])

    mmoe_opt.zero_grad(set_to_none=True)
    long_logit, click_logit = mmoe(cat, num)
    long_loss = F.binary_cross_entropy_with_logits(
        long_logit, y_long, reduction="none"
    )
    click_loss = F.binary_cross_entropy_with_logits(
        click_logit, y_click, reduction="none"
    )
    loss = torch.sum(w * (long_loss + 0.35 * click_loss)) / torch.sum(w)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mmoe.parameters(), 5.0)
    mmoe_opt.step()
    mmoe_loss_sum += float(loss.detach()) * len(idx)
print(
    "FINDINGS mmoe_weighted_joint_loss=%.6f aux_target=%s"
    % (mmoe_loss_sum / n_train, click_key),
    flush=True,
)

del perm, train_click, train_history, cat_train, nn_train_num
del sorted_video, positive_videos, group_id, prior_positive_count
del train_y, train_weight, sort_order, train
gc.collect()


def build_cat(split):
    return np.column_stack([
        np.asarray(split.X[f], dtype=np.int32) + int(off)
        for f, off in zip(MODEL_CAT, cat_offsets)
    ]).astype(np.int32, copy=False)


def split_history(split):
    u = np.asarray(split.user_id, dtype=np.int64)
    result = np.zeros((len(u), SEQ_LEN), dtype=np.int32)
    ok = (u >= 0) & (u < final_history.shape[0])
    result[ok] = final_history[u[ok]]
    return result


@torch.inference_mode()
def predict_din(split, X):
    din.eval()
    cats = build_cat(split)
    histories = split_history(split)
    nums = X[:, n_low:]
    out = np.empty(len(split.user_id), dtype=np.float32)
    bs = 65536
    for start in range(0, len(out), bs):
        end = min(start + bs, len(out))
        cat = torch.from_numpy(
            cats[start:end].astype(np.int64, copy=False)
        )
        num = torch.from_numpy(normalized_num(nums[start:end]))
        hist = torch.from_numpy(
            histories[start:end].astype(np.int64, copy=False) + video_offset
        )
        out[start:end] = din(cat, num, hist).cpu().numpy()
    del cats, histories
    return out


@torch.inference_mode()
def predict_mmoe(split, X):
    mmoe.eval()
    cats = build_cat(split)
    nums = X[:, n_low:]
    out = np.empty(len(split.user_id), dtype=np.float32)
    bs = 65536
    for start in range(0, len(out), bs):
        end = min(start + bs, len(out))
        cat = torch.from_numpy(
            cats[start:end].astype(np.int64, copy=False)
        )
        num = torch.from_numpy(normalized_num(nums[start:end]))
        out[start:end] = mmoe(cat, num)[0].cpu().numpy()
    del cats
    return out


valid = load("valid")
hv_valid = historical_features("valid", key="video_id")
ha_valid = historical_features("valid", key="author_id")
X_valid = build_matrix(valid, hv_valid, ha_valid, is_train=False)
del hv_valid, ha_valid
gc.collect()

valid_uid = np.asarray(valid.user_id)
valid_y = np.asarray(valid.y)

rank_valid = ranker.predict(
    X_valid, num_iteration=ranker.current_iteration()
).astype(np.float32)
din_valid = predict_din(valid, X_valid)
mmoe_valid = predict_mmoe(valid, X_valid)

families = {
    "lambda_rank": rank_valid,
    "din_sequence": din_valid,
    "mmoe_long_click": mmoe_valid,
}
candidate_scores = {}
best_name = None
best_family = None
best_weight = 1.0
best_scores = None
best_raw = None
best_primary = -np.inf

for name, score in families.items():
    met = evaluate(valid_uid, valid_y, score)
    candidate_scores[name] = float(met["primary"])
    if float(met["primary"]) > best_primary:
        best_primary = float(met["primary"])
        best_name = name
        best_family = name
        best_weight = 1.0
        best_scores = score.copy()
        best_raw = score.copy()

inc_valid_path = (
    os.path.join(SHARED, "incumbent_valid_scores.npy")
    if SHARED else ""
)
inc_test_path = (
    os.path.join(SHARED, "incumbent_test_scores.npy")
    if SHARED else ""
)
has_inc = bool(
    inc_valid_path
    and inc_test_path
    and os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
)

if has_inc:
    inc_valid = np.load(inc_valid_path, mmap_mode="r")
    inc_metric = evaluate(valid_uid, valid_y, inc_valid)
    candidate_scores["trusted_incumbent"] = float(inc_metric["primary"])
    inc_rank = within_user_rank(valid_uid, inc_valid)

    if float(inc_metric["primary"]) > best_primary:
        best_primary = float(inc_metric["primary"])
        best_name = "trusted_incumbent"
        best_family = "trusted_incumbent"
        best_weight = 0.0
        best_scores = np.asarray(inc_valid, dtype=np.float32).copy()
        best_raw = rank_valid.copy()

    for name, score in families.items():
        own_rank = within_user_rank(valid_uid, score)
        family_best = -np.inf
        family_w = None
        for w in (0.20, 0.35, 0.50, 0.65, 0.80):
            blended = w * own_rank + (1.0 - w) * inc_rank
            met = evaluate(valid_uid, valid_y, blended)
            p = float(met["primary"])
            if p > family_best:
                family_best = p
                family_w = w
            if p > best_primary:
                best_primary = p
                best_name = "%s_inc_rankblend_%.2f" % (name, w)
                best_family = name
                best_weight = float(w)
                best_scores = blended.astype(np.float32)
                best_raw = score.copy()
        candidate_scores[name + "_best_blend"] = family_best
        print(
            "FINDINGS %s best_blend_weight=%.2f primary=%.6f"
            % (name, family_w, family_best),
            flush=True,
        )

final_metrics = evaluate(valid_uid, valid_y, best_scores)
print(
    "FINDINGS winner=%s features=%d history_video=%s history_author=%s"
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
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_weight < 1.0:
        np.save(
            os.path.join(OUT, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

del X_valid, best_scores, best_raw, valid_y
gc.collect()

test = load("test")
if best_family == "trusted_incumbent":
    test_scores = np.asarray(
        np.load(inc_test_path, mmap_mode="r"), dtype=np.float32
    ).copy()
else:
    hv_test = historical_features("test", key="video_id")
    ha_test = historical_features("test", key="author_id")
    X_test = build_matrix(test, hv_test, ha_test, is_train=False)
    del hv_test, ha_test
    gc.collect()

    if best_family == "lambda_rank":
        own_test = ranker.predict(
            X_test, num_iteration=ranker.current_iteration()
        ).astype(np.float32)
    elif best_family == "din_sequence":
        own_test = predict_din(test, X_test)
    elif best_family == "mmoe_long_click":
        own_test = predict_mmoe(test, X_test)
    else:
        raise RuntimeError("Unknown winning family: " + str(best_family))

    if best_weight < 1.0 and has_inc:
        inc_test = np.load(inc_test_path, mmap_mode="r")
        test_scores = (
            best_weight * within_user_rank(test.user_id, own_test)
            + (1.0 - best_weight)
            * within_user_rank(test.user_id, inc_test)
        ).astype(np.float32)
    else:
        test_scores = own_test
    del X_test
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