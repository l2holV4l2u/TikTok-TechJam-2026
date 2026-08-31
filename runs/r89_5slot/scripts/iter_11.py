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
SEED = 19427
BATCH = 8192
PRED_BATCH = 65536
EPOCHS = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

CAT_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
    "onehot_feat3",
]


def make_features(s):
    user = np.asarray(s.user_id, dtype=np.int64)
    tm = np.asarray(s.time_ms, dtype=np.int64)
    n = user.size
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, tm, user))
    us = user[order]
    ts = tm[order]

    new_user = np.empty(n, dtype=bool)
    new_user[0] = True
    new_user[1:] = us[1:] != us[:-1]
    starts = np.flatnonzero(new_user)
    counts = np.diff(np.r_[starts, n])
    start_sorted = np.repeat(starts, counts)

    pos_s = np.arange(n, dtype=np.int64) - start_sorted
    cnt_s = np.repeat(counts, counts)
    rev_s = cnt_s - 1 - pos_s

    new_batch = np.empty(n, dtype=bool)
    new_batch[0] = True
    new_batch[1:] = (us[1:] != us[:-1]) | (ts[1:] != ts[:-1])
    batch_starts = np.flatnonzero(new_batch)
    batch_counts = np.diff(np.r_[batch_starts, n])
    batch_start_s = np.repeat(batch_starts, batch_counts)
    batch_pos_s = np.arange(n, dtype=np.int64) - batch_start_s
    batch_cnt_s = np.repeat(batch_counts, batch_counts)

    prev_gap_s = np.zeros(n, dtype=np.float64)
    next_gap_s = np.zeros(n, dtype=np.float64)
    if n > 1:
        same = us[1:] == us[:-1]
        delta = np.maximum(ts[1:] - ts[:-1], 0)
        prev_gap_s[1:] = np.where(same, delta, 0)
        next_gap_s[:-1] = np.where(same, delta, 0)

    def unsort(x):
        z = np.empty_like(x)
        z[order] = x
        return z

    pos = unsort(pos_s).astype(np.float32)
    rev = unsort(rev_s).astype(np.float32)
    cnt = unsort(cnt_s).astype(np.float32)
    batch_pos = unsort(batch_pos_s).astype(np.float32)
    batch_cnt = unsort(batch_cnt_s).astype(np.float32)
    prev_gap = unsort(prev_gap_s).astype(np.float32)
    next_gap = unsort(next_gap_s).astype(np.float32)

    denom = np.maximum(cnt - 1.0, 1.0)
    frac = pos / denom
    rev_frac = rev / denom
    batch_frac = batch_pos / np.maximum(batch_cnt - 1.0, 1.0)

    hour_id = np.asarray(s.X["hour"], dtype=np.float32)
    hour_value = np.mod(np.maximum(hour_id - 1.0, 0.0), 24.0)
    angle = 2.0 * np.pi * hour_value / 24.0

    duration = np.nan_to_num(
        np.asarray(s.num["duration_ms"], dtype=np.float32),
        nan=0.0, posinf=0.0, neginf=0.0
    )
    duration = np.log1p(np.maximum(duration, 0.0))

    date = np.asarray(s.date, dtype=np.int64)
    unique_dates = np.unique(date)
    date_idx = np.searchsorted(unique_dates, date).astype(np.float32)
    date_frac = date_idx / max(float(len(unique_dates) - 1), 1.0)

    x = np.column_stack([
        frac,
        rev_frac,
        frac * frac,
        frac * frac * frac,
        np.sqrt(np.maximum(frac, 0.0)),
        np.log1p(pos),
        np.log1p(rev),
        np.log1p(cnt),
        batch_frac,
        np.log1p(batch_cnt),
        np.log1p(np.maximum(prev_gap, 0.0)),
        np.log1p(np.maximum(next_gap, 0.0)),
        np.sin(angle),
        np.cos(angle),
        duration,
        date_frac,
        (pos == 0).astype(np.float32),
        (rev == 0).astype(np.float32),
        (batch_cnt > 1).astype(np.float32),
        (cnt <= 4).astype(np.float32),
        (cnt >= 8).astype(np.float32),
    ]).astype(np.float32)

    cats = np.column_stack([
        np.asarray(s.X[f], dtype=np.int64) for f in CAT_FIELDS
    ]).astype(np.int64)

    # Information needed to draw same-user partners without row loops.
    sorted_position = np.empty(n, dtype=np.int64)
    sorted_position[order] = np.arange(n, dtype=np.int64)
    row_start = start_sorted[sorted_position]
    row_count = cnt_s[sorted_position]

    group_info = {
        "order": order,
        "start": row_start.astype(np.int64),
        "count": row_count.astype(np.int64),
    }
    return x, cats, group_info


def fit_scaler(x):
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    return mean, std


def scale_features(x, mean, std):
    return np.clip((x - mean) / std, -8.0, 8.0).astype(np.float32)


class WidePairRanker(nn.Module):
    def __init__(self, cards, d, base_rate):
        super().__init__()
        self.tables = nn.ModuleList([nn.Embedding(c, 1) for c in cards])
        self.linear = nn.Linear(d, 1)
        self.bias = nn.Parameter(torch.tensor(
            np.log(base_rate / (1.0 - base_rate)), dtype=torch.float32
        ))
        for table in self.tables:
            nn.init.zeros_(table.weight)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cats, x):
        out = self.linear(x).squeeze(1) + self.bias
        for j, table in enumerate(self.tables):
            out = out + table(cats[:, j]).squeeze(1)
        return out


class FMPairRanker(nn.Module):
    def __init__(self, cards, d, base_rate, k=12):
        super().__init__()
        self.first = nn.ModuleList([nn.Embedding(c, 1) for c in cards])
        self.emb = nn.ModuleList([nn.Embedding(c, k) for c in cards])
        self.context_linear = nn.Linear(d, 1)
        self.context_emb = nn.Linear(d, k)
        self.bias = nn.Parameter(torch.tensor(
            np.log(base_rate / (1.0 - base_rate)), dtype=torch.float32
        ))
        for table in self.first:
            nn.init.zeros_(table.weight)
        for table in self.emb:
            nn.init.normal_(table.weight, std=0.02)
        nn.init.zeros_(self.context_linear.weight)
        nn.init.zeros_(self.context_linear.bias)
        nn.init.normal_(self.context_emb.weight, std=0.02)
        nn.init.zeros_(self.context_emb.bias)

    def forward(self, cats, x):
        first = self.context_linear(x).squeeze(1) + self.bias
        vectors = []
        for j in range(cats.shape[1]):
            first = first + self.first[j](cats[:, j]).squeeze(1)
            vectors.append(self.emb[j](cats[:, j]))
        vectors.append(self.context_emb(x).unsqueeze(1))
        v = torch.cat(
            [z if z.ndim == 3 else z.unsqueeze(1) for z in vectors], dim=1
        )
        summed = v.sum(dim=1)
        fm = 0.5 * (
            summed.square().sum(dim=1) - v.square().sum(dim=(1, 2))
        )
        return first + fm


class DeepPairRanker(nn.Module):
    def __init__(self, cards, d, base_rate, k=6):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c, k) for c in cards])
        inp = len(cards) * k + d
        self.net = nn.Sequential(
            nn.Linear(inp, 96),
            nn.SiLU(),
            nn.LayerNorm(96),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )
        for table in self.emb:
            nn.init.normal_(table.weight, std=0.025)
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.constant_(
            self.net[-1].bias,
            np.log(base_rate / (1.0 - base_rate))
        )

    def forward(self, cats, x):
        e = torch.cat(
            [self.emb[j](cats[:, j]) for j in range(cats.shape[1])], dim=1
        )
        return self.net(torch.cat([e, x], dim=1)).squeeze(1)


def build_model(family, cards, d, base_rate):
    base_rate = float(np.clip(base_rate, 1e-5, 1.0 - 1e-5))
    if family == "pairwise_wide":
        return WidePairRanker(cards, d, base_rate), 0.006, 2e-6
    if family == "pairwise_fm":
        return FMPairRanker(cards, d, base_rate), 0.0022, 1e-6
    return DeepPairRanker(cards, d, base_rate), 0.0018, 2e-5


def train_model(family, x, cats, y, group):
    torch.manual_seed(SEED + {
        "pairwise_wide": 11,
        "pairwise_fm": 37,
        "pairwise_deep": 71,
    }[family])

    cards = [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS]
    model, lr, wd = build_model(family, cards, x.shape[1], float(y.mean()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    xt = torch.from_numpy(x)
    ct = torch.from_numpy(cats)
    yt = torch.from_numpy(y.astype(np.float32, copy=False))
    n = len(y)
    rng = np.random.default_rng(SEED + 503)
    generator = torch.Generator()
    generator.manual_seed(SEED + 907)

    for epoch in range(EPOCHS):
        offsets = (
            rng.random(n) * group["count"].astype(np.float64)
        ).astype(np.int64)
        partner_sorted = group["start"] + offsets
        partner = group["order"][partner_sorted]
        partner_t = torch.from_numpy(partner.astype(np.int64, copy=False))
        perm = torch.randperm(n, generator=generator)

        model.train()
        for st in range(0, n, BATCH):
            idx = perm[st:min(st + BATCH, n)]
            jdx = partner_t[idx]

            li = model(ct[idx], xt[idx])
            lj = model(ct[jdx], xt[jdx])
            yi = yt[idx]
            yj = yt[jdx]

            different = yi != yj
            if bool(different.any()):
                sign = 2.0 * yi[different] - 1.0
                pair_loss = F.softplus(
                    -sign * (li[different] - lj[different])
                ).mean()
            else:
                pair_loss = li.sum() * 0.0

            # Pointwise anchoring uses singleton users and stabilizes sparse IDs.
            point_loss = F.binary_cross_entropy_with_logits(li, yi)
            loss = pair_loss + 0.18 * point_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.no_grad()
def predict_model(model, x, cats):
    model.eval()
    out = np.empty(x.shape[0], dtype=np.float32)
    for st in range(0, x.shape[0], PRED_BATCH):
        en = min(st + PRED_BATCH, x.shape[0])
        out[st:en] = model(
            torch.from_numpy(cats[st:en]),
            torch.from_numpy(x[st:en])
        ).cpu().numpy()
    return out


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(x.std())
    if sd < 1e-12:
        sd = 1.0
    return (x - float(x.mean())) / sd


def within_user_rank(user, score):
    user = np.asarray(user, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    n = len(score)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, score, user))
    us = user[order]

    new = np.empty(n, dtype=bool)
    new[0] = True
    new[1:] = us[1:] != us[:-1]
    starts = np.flatnonzero(new)
    counts = np.diff(np.r_[starts, n])
    positions = np.arange(n, dtype=np.float64) - np.repeat(starts, counts)
    denom = np.maximum(np.repeat(counts, counts) - 1, 1)
    ranked = positions / denom
    ranked[np.repeat(counts, counts) == 1] = 0.5

    out = np.empty(n, dtype=np.float64)
    out[order] = ranked
    return out


train = load("train")
valid = load("valid")
train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

train_raw, train_cat, train_group = make_features(train)
valid_raw, valid_cat, valid_group = make_features(valid)
mean, std = fit_scaler(train_raw)
train_x = scale_features(train_raw, mean, std)
valid_x = scale_features(valid_raw, mean, std)

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.load(os.path.join(shared, "incumbent_valid_scores.npy"))
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_z = zscore(inc_valid)
inc_rank = within_user_rank(valid_users, inc_valid)

families = ["pairwise_wide", "pairwise_fm", "pairwise_deep"]
own_valid = {}
candidate_scores = {}
candidate_specs = {}

inc_metrics = evaluate(valid_users, valid_y, inc_valid)
candidate_scores["incumbent"] = float(inc_metrics["primary"])
candidate_specs["incumbent"] = ("incumbent", 0.0, "raw")

best_name = "incumbent"
best_scores = np.asarray(inc_valid, dtype=np.float64)
best_metrics = inc_metrics
best_primary = float(inc_metrics["primary"])

for family in families:
    model = train_model(
        family, train_x, train_cat, train_y, train_group
    )
    pred = predict_model(model, valid_x, valid_cat)
    own_valid[family] = pred

    met = evaluate(valid_users, valid_y, pred)
    candidate_scores[family] = float(met["primary"])
    candidate_specs[family] = (family, 0.0, "raw")
    if float(met["primary"]) > best_primary:
        best_primary = float(met["primary"])
        best_name = family
        best_scores = pred.copy()
        best_metrics = met

    pred_z = zscore(pred)
    pred_rank = within_user_rank(valid_users, pred)

    for w in (0.08, 0.14, 0.20, 0.28, 0.36, 0.45):
        blended_z = (1.0 - w) * inc_z + w * pred_z
        name_z = "%s_zblend_%.2f" % (family, w)
        met_z = evaluate(valid_users, valid_y, blended_z)
        candidate_scores[name_z] = float(met_z["primary"])
        candidate_specs[name_z] = (family, float(w), "z")
        if float(met_z["primary"]) > best_primary:
            best_primary = float(met_z["primary"])
            best_name = name_z
            best_scores = blended_z.copy()
            best_metrics = met_z

        blended_rank = (1.0 - w) * inc_rank + w * pred_rank
        name_r = "%s_rankblend_%.2f" % (family, w)
        met_r = evaluate(valid_users, valid_y, blended_rank)
        candidate_scores[name_r] = float(met_r["primary"])
        candidate_specs[name_r] = (family, float(w), "rank")
        if float(met_r["primary"]) > best_primary:
            best_primary = float(met_r["primary"])
            best_name = name_r
            best_scores = blended_rank.copy()
            best_metrics = met_r

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True), flush=True)

selected_family, selected_weight, selected_mode = candidate_specs[best_name]
print(
    "FINDINGS selected=%s family=%s mode=%s own_weight=%.2f"
    % (best_name, selected_family, selected_mode, selected_weight),
    flush=True,
)

# Refit the selected recipe on train + validation, then score test.
test = load("test")
test_raw, test_cat, test_group = make_features(test)
test_users = np.asarray(test.user_id, dtype=np.int64)

if selected_family == "incumbent":
    test_scores = np.load(inc_test_path)
    selected_raw_valid = None
else:
    combined_raw = np.concatenate([train_raw, valid_raw], axis=0)
    combined_cat = np.concatenate([train_cat, valid_cat], axis=0)
    combined_y = np.concatenate([
        train_y, valid_y.astype(np.float32, copy=False)
    ])

    # Reconstruct groups over the concatenated training recipe. Pairing remains
    # same-user, while temporal context itself was computed within each split.
    combined_user = np.concatenate([
        np.asarray(train.user_id, dtype=np.int64),
        np.asarray(valid.user_id, dtype=np.int64),
    ])
    combined_tm = np.concatenate([
        np.asarray(train.time_ms, dtype=np.int64),
        np.asarray(valid.time_ms, dtype=np.int64),
    ])
    nn_rows = combined_user.size
    rows = np.arange(nn_rows, dtype=np.int64)
    order = np.lexsort((rows, combined_tm, combined_user))
    us = combined_user[order]
    new = np.empty(nn_rows, dtype=bool)
    new[0] = True
    new[1:] = us[1:] != us[:-1]
    starts = np.flatnonzero(new)
    counts = np.diff(np.r_[starts, nn_rows])
    start_s = np.repeat(starts, counts)
    count_s = np.repeat(counts, counts)
    sorted_pos = np.empty(nn_rows, dtype=np.int64)
    sorted_pos[order] = np.arange(nn_rows, dtype=np.int64)
    combined_group = {
        "order": order,
        "start": start_s[sorted_pos],
        "count": count_s[sorted_pos],
    }

    cmean, cstd = fit_scaler(combined_raw)
    combined_x = scale_features(combined_raw, cmean, cstd)
    test_x = scale_features(test_raw, cmean, cstd)

    final_model = train_model(
        selected_family,
        combined_x,
        combined_cat,
        combined_y,
        combined_group,
    )
    own_test = predict_model(final_model, test_x, test_cat)
    selected_raw_valid = own_valid[selected_family]

    if selected_weight == 0.0:
        test_scores = own_test
    else:
        inc_test = np.load(inc_test_path)
        if selected_mode == "z":
            test_scores = (
                (1.0 - selected_weight) * zscore(inc_test)
                + selected_weight * zscore(own_test)
            )
        else:
            test_scores = (
                (1.0 - selected_weight)
                * within_user_rank(test_users, inc_test)
                + selected_weight
                * within_user_rank(test_users, own_test)
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
    if selected_family != "incumbent" and selected_weight > 0.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(selected_raw_valid, dtype=np.float64),
        )

result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(time.time() - START),
}
print("METRICS " + json.dumps(result, separators=(", ", ": ")))