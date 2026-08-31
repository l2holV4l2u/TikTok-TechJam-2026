import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 73129
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "duration_bucket", "upload_type", "music_type", "video_type",
    "onehot_feat3", "onehot_feat7", "onehot_feat8",
    "user_active_degree", "fans_user_num_range",
    "follow_user_num_range", "friend_user_num_range",
    "register_days_bucket", "is_video_author", "hour",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def sequential_features(s):
    user = np.asarray(s.user_id, dtype=np.int64)
    tm = np.asarray(s.time_ms, dtype=np.int64)
    n = user.size
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, tm, user))
    us = user[order]
    ts = tm[order]

    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = us[1:] != us[:-1]
    starts = np.flatnonzero(first)
    counts = np.diff(np.r_[starts, n])
    start_rep = np.repeat(starts, counts)

    pos_s = np.arange(n, dtype=np.int64) - start_rep
    count_s = np.repeat(counts, counts)
    rev_s = count_s - 1 - pos_s

    batch_first = np.empty(n, dtype=bool)
    batch_first[0] = True
    batch_first[1:] = (us[1:] != us[:-1]) | (ts[1:] != ts[:-1])
    batch_starts = np.flatnonzero(batch_first)
    batch_counts = np.diff(np.r_[batch_starts, n])
    batch_start_rep = np.repeat(batch_starts, batch_counts)
    batch_pos_s = np.arange(n, dtype=np.int64) - batch_start_rep
    batch_count_s = np.repeat(batch_counts, batch_counts)

    prev_gap_s = np.zeros(n, dtype=np.float32)
    next_gap_s = np.zeros(n, dtype=np.float32)
    if n > 1:
        same = us[1:] == us[:-1]
        gap = np.maximum(ts[1:] - ts[:-1], 0).astype(np.float64)
        prev_gap_s[1:] = np.where(same, np.log1p(gap), 0.0)
        next_gap_s[:-1] = np.where(same, np.log1p(gap), 0.0)

    def unsort(a):
        out = np.empty_like(a)
        out[order] = a
        return out

    pos = unsort(pos_s).astype(np.float32)
    rev = unsort(rev_s).astype(np.float32)
    count = unsort(count_s).astype(np.float32)
    batch_pos = unsort(batch_pos_s).astype(np.float32)
    batch_count = unsort(batch_count_s).astype(np.float32)
    prev_gap = unsort(prev_gap_s).astype(np.float32)
    next_gap = unsort(next_gap_s).astype(np.float32)

    frac = pos / np.maximum(count - 1.0, 1.0)
    rev_frac = rev / np.maximum(count - 1.0, 1.0)
    batch_frac = batch_pos / np.maximum(batch_count - 1.0, 1.0)

    hour_id = np.asarray(s.X["hour"], dtype=np.float32)
    hour = np.mod(np.maximum(hour_id - 1.0, 0.0), 24.0)
    angle = 2.0 * np.pi * hour / 24.0

    dates = np.asarray(s.date, dtype=np.int64)
    unique_dates = np.unique(dates)
    date_index = np.searchsorted(unique_dates, dates).astype(np.float32)
    date_frac = date_index / max(float(len(unique_dates) - 1), 1.0)

    return np.column_stack([
        frac,
        rev_frac,
        frac * frac,
        np.log1p(pos),
        np.log1p(rev),
        np.log1p(count),
        batch_frac,
        np.log1p(batch_count),
        prev_gap,
        next_gap,
        np.sin(angle),
        np.cos(angle),
        date_frac,
        (pos == 0).astype(np.float32),
        (rev == 0).astype(np.float32),
        (batch_count > 1).astype(np.float32),
        (count <= 4).astype(np.float32),
        (count >= 8).astype(np.float32),
    ]).astype(np.float32)


def make_features(s):
    cats = np.column_stack([
        np.asarray(s.X[f], dtype=np.int32) for f in CAT_FIELDS
    ])

    raw_num = []
    for f in NUM_FIELDS:
        x = np.asarray(s.num[f], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        raw_num.append(np.log1p(np.maximum(x, 0.0)))
    nums = np.column_stack(raw_num).astype(np.float32)
    seq = sequential_features(s)
    return cats, np.column_stack([nums, seq]).astype(np.float32)


def fit_scaler(x):
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    return mean, std


def scale_num(x, mean, std):
    return np.clip((x - mean) / std, -8.0, 8.0).astype(np.float32)


def group_sort(user):
    user = np.asarray(user, dtype=np.int64)
    row = np.arange(user.size, dtype=np.int64)
    order = np.lexsort((row, user))
    sorted_user = user[order]
    first = np.empty(user.size, dtype=bool)
    first[0] = True
    first[1:] = sorted_user[1:] != sorted_user[:-1]
    starts = np.flatnonzero(first)
    counts = np.diff(np.r_[starts, user.size]).astype(np.int32)
    return order, counts


def make_lgb_matrix(cats, nums):
    return np.column_stack([cats.astype(np.float32), nums]).astype(np.float32)


def train_xendcg(cats, nums, y, users, valid_pack=None, rounds=240):
    order, groups = group_sort(users)
    x = make_lgb_matrix(cats, nums)
    ds = lgb.Dataset(
        x[order],
        label=np.asarray(y, dtype=np.float32)[order],
        group=groups,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )

    params = {
        "objective": "rank_xendcg",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "label_gain": [0, 1],
        "learning_rate": 0.055,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 180,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 0.15,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "cat_smooth": 25.0,
        "cat_l2": 12.0,
        "num_threads": min(8, os.cpu_count() or 1),
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "verbose": -1,
    }

    if valid_pack is None:
        model = lgb.train(params, ds, num_boost_round=int(rounds))
        return model, int(rounds)

    vc, vn, vy, vu = valid_pack
    vorder, vgroups = group_sort(vu)
    vx = make_lgb_matrix(vc, vn)
    vds = lgb.Dataset(
        vx[vorder],
        label=np.asarray(vy, dtype=np.float32)[vorder],
        group=vgroups,
        categorical_feature=list(range(len(CAT_FIELDS))),
        reference=ds,
        free_raw_data=True,
    )
    model = lgb.train(
        params,
        ds,
        num_boost_round=int(rounds),
        valid_sets=[vds],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model, int(model.best_iteration)


def predict_lgb(model, cats, nums, iteration):
    x = make_lgb_matrix(cats, nums)
    return model.predict(x, num_iteration=iteration).astype(np.float32)


class ListNetAdditive(nn.Module):
    def __init__(self, cards, num_dim, base_rate):
        super().__init__()
        self.tables = nn.ModuleList([nn.Embedding(c, 1) for c in cards])
        self.numeric = nn.Sequential(
            nn.Linear(num_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        self.bias = nn.Parameter(torch.tensor(
            np.log(base_rate / (1.0 - base_rate)), dtype=torch.float32
        ))
        for table in self.tables:
            nn.init.zeros_(table.weight)
        nn.init.xavier_uniform_(self.numeric[0].weight)
        nn.init.zeros_(self.numeric[0].bias)
        nn.init.zeros_(self.numeric[2].weight)
        nn.init.zeros_(self.numeric[2].bias)

    def forward(self, cats, nums):
        score = self.numeric(nums).squeeze(1) + self.bias
        for j, table in enumerate(self.tables):
            score = score + table(cats[:, j]).squeeze(1)
        return score


def train_listnet(cats, nums, y, users, epochs=2):
    torch.manual_seed(SEED + 100)
    cards = [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS]
    rate = float(np.clip(np.mean(y), 1e-5, 1.0 - 1e-5))
    model = ListNetAdditive(cards, nums.shape[1], rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=2e-6)

    order, counts = group_sort(users)
    cats_s = torch.from_numpy(cats[order].astype(np.int64, copy=False))
    nums_s = torch.from_numpy(nums[order])
    y_s = torch.from_numpy(np.asarray(y, dtype=np.float32)[order])

    starts = np.r_[0, np.cumsum(counts)].astype(np.int64)
    n_groups = len(counts)
    groups_per_block = 128
    blocks = [
        (g, min(g + groups_per_block, n_groups))
        for g in range(0, n_groups, groups_per_block)
    ]
    rng = np.random.default_rng(SEED + 200)

    for _ in range(epochs):
        rng.shuffle(blocks)
        model.train()
        for g0, g1 in blocks:
            st = int(starts[g0])
            en = int(starts[g1])
            local_counts = counts[g0:g1].astype(np.int64)

            c = cats_s[st:en]
            x = nums_s[st:en]
            target = y_s[st:en]
            logits = model(c, x)

            group_index = torch.repeat_interleave(
                torch.arange(g1 - g0, dtype=torch.int64),
                torch.from_numpy(local_counts),
            )
            maxima = torch.full((g1 - g0,), -1e30, dtype=torch.float32)
            maxima.scatter_reduce_(
                0, group_index, logits.detach(), reduce="amax", include_self=True
            )
            exp_score = torch.exp(logits - maxima[group_index])
            denominators = torch.zeros(g1 - g0, dtype=torch.float32)
            denominators.scatter_add_(0, group_index, exp_score)
            log_prob = logits - maxima[group_index] - torch.log(
                denominators[group_index] + 1e-12
            )

            positives = torch.zeros(g1 - g0, dtype=torch.float32)
            positives.scatter_add_(0, group_index, target)
            valid_group = positives > 0
            row_weight = target / torch.clamp(positives[group_index], min=1.0)
            if bool(valid_group.any()):
                list_loss = -(row_weight * log_prob).sum() / valid_group.sum()
            else:
                list_loss = logits.sum() * 0.0

            point_loss = F.binary_cross_entropy_with_logits(logits, target)
            loss = list_loss + 0.10 * point_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.no_grad()
def predict_listnet(model, cats, nums):
    model.eval()
    out = np.empty(cats.shape[0], dtype=np.float32)
    batch = 65536
    for st in range(0, cats.shape[0], batch):
        en = min(st + batch, cats.shape[0])
        out[st:en] = model(
            torch.from_numpy(cats[st:en].astype(np.int64, copy=False)),
            torch.from_numpy(nums[st:en]),
        ).numpy()
    return out


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(x.std())
    if sd < 1e-12:
        sd = 1.0
    return (x - float(x.mean())) / sd


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, users))
    us = users[order]

    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = us[1:] != us[:-1]
    starts = np.flatnonzero(first)
    counts = np.diff(np.r_[starts, n])
    positions = np.arange(n, dtype=np.float64) - np.repeat(starts, counts)
    repeated_counts = np.repeat(counts, counts)
    rank = positions / np.maximum(repeated_counts - 1, 1)
    rank[repeated_counts == 1] = 0.5

    out = np.empty(n, dtype=np.float64)
    out[order] = rank
    return out


train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
train_users = np.asarray(train.user_id, dtype=np.int64)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

train_cat, train_num_raw = make_features(train)
valid_cat, valid_num_raw = make_features(valid)
mean, std = fit_scaler(train_num_raw)
train_num = scale_num(train_num_raw, mean, std)
valid_num = scale_num(valid_num_raw, mean, std)

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.load(os.path.join(shared, "incumbent_valid_scores.npy"))
inc_test = np.load(os.path.join(shared, "incumbent_test_scores.npy"))

candidate_values = {}
candidate_specs = {}
raw_valid = {}

inc_metrics = evaluate(valid_users, valid_y, inc_valid)
candidate_values["incumbent"] = float(inc_metrics["primary"])
candidate_specs["incumbent"] = ("incumbent", 0.0, "raw")

best_name = "incumbent"
best_scores = np.asarray(inc_valid, dtype=np.float64)
best_metrics = inc_metrics
best_primary = float(inc_metrics["primary"])

xendcg, best_round = train_xendcg(
    train_cat,
    train_num,
    train_y,
    train_users,
    valid_pack=(valid_cat, valid_num, valid_y, valid_users),
    rounds=260,
)
pred_xendcg = predict_lgb(xendcg, valid_cat, valid_num, best_round)
raw_valid["rank_xendcg"] = pred_xendcg

listnet = train_listnet(
    train_cat, train_num, train_y, train_users, epochs=2
)
pred_listnet = predict_listnet(listnet, valid_cat, valid_num)
raw_valid["listnet_additive"] = pred_listnet

inc_z = zscore(inc_valid)
inc_rank = within_user_rank(valid_users, inc_valid)

for family, pred in raw_valid.items():
    met = evaluate(valid_users, valid_y, pred)
    candidate_values[family] = float(met["primary"])
    candidate_specs[family] = (family, 1.0, "raw")
    if float(met["primary"]) > best_primary:
        best_primary = float(met["primary"])
        best_name = family
        best_scores = pred.astype(np.float64)
        best_metrics = met

    pred_z = zscore(pred)
    pred_rank = within_user_rank(valid_users, pred)

    for w in (0.10, 0.18, 0.26, 0.34, 0.42, 0.50):
        name = f"{family}_zblend_{w:.2f}"
        scores = (1.0 - w) * inc_z + w * pred_z
        met = evaluate(valid_users, valid_y, scores)
        candidate_values[name] = float(met["primary"])
        candidate_specs[name] = (family, float(w), "z")
        if float(met["primary"]) > best_primary:
            best_primary = float(met["primary"])
            best_name = name
            best_scores = scores.copy()
            best_metrics = met

        name = f"{family}_rankblend_{w:.2f}"
        scores = (1.0 - w) * inc_rank + w * pred_rank
        met = evaluate(valid_users, valid_y, scores)
        candidate_values[name] = float(met["primary"])
        candidate_specs[name] = (family, float(w), "rank")
        if float(met["primary"]) > best_primary:
            best_primary = float(met["primary"])
            best_name = name
            best_scores = scores.copy()
            best_metrics = met

print("CANDIDATES " + json.dumps(
    {k: round(v, 6) for k, v in candidate_values.items()},
    sort_keys=True
))
print("FINDINGS selected=%s xendcg_rounds=%d" % (best_name, best_round))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    selected_family, selected_weight, selected_mode = candidate_specs[best_name]
    if selected_family != "incumbent":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_valid[selected_family], dtype=np.float64),
        )

# Refit the selected recipe on train + validation, then score test.
te = load("test")
test_users = np.asarray(te.user_id, dtype=np.int64)
test_cat, test_num_raw = make_features(te)

selected_family, selected_weight, selected_mode = candidate_specs[best_name]

if selected_family == "incumbent":
    test_scores = np.asarray(inc_test, dtype=np.float64)
else:
    combined_cat = np.concatenate([train_cat, valid_cat], axis=0)
    combined_num_raw = np.concatenate([train_num_raw, valid_num_raw], axis=0)
    combined_y = np.concatenate([
        train_y, valid_y.astype(np.float32)
    ], axis=0)
    combined_users = np.concatenate([train_users, valid_users], axis=0)

    comb_mean, comb_std = fit_scaler(combined_num_raw)
    combined_num = scale_num(combined_num_raw, comb_mean, comb_std)
    test_num = scale_num(test_num_raw, comb_mean, comb_std)

    if selected_family == "rank_xendcg":
        final_model, _ = train_xendcg(
            combined_cat,
            combined_num,
            combined_y,
            combined_users,
            valid_pack=None,
            rounds=best_round,
        )
        test_raw = predict_lgb(
            final_model, test_cat, test_num, best_round
        )
    else:
        final_model = train_listnet(
            combined_cat,
            combined_num,
            combined_y,
            combined_users,
            epochs=2,
        )
        test_raw = predict_listnet(final_model, test_cat, test_num)

    if selected_mode == "raw":
        test_scores = np.asarray(test_raw, dtype=np.float64)
    elif selected_mode == "z":
        test_scores = (
            (1.0 - selected_weight) * zscore(inc_test)
            + selected_weight * zscore(test_raw)
        )
    else:
        test_scores = (
            (1.0 - selected_weight)
            * within_user_rank(test_users, inc_test)
            + selected_weight
            * within_user_rank(test_users, test_raw)
        )

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))