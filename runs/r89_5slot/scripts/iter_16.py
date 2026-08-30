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
SEED = 28183
BATCH = 8192
PRED_BATCH = 65536
EPOCHS = 1
HALF_LIFE_DAYS = 10.0

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
    "music_type",
    "hour",
    "onehot_feat3",
]


def date_to_ordinal(date):
    date = np.asarray(date, dtype=np.int64)
    year = date // 10000
    month = (date // 100) % 100
    day = date % 100

    # All benchmark dates are in one spring month range, but this expression
    # also remains monotone if a month boundary is crossed.
    return (year * 372 + month * 31 + day).astype(np.float32)


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
    repeated_starts = np.repeat(starts, counts)

    pos_s = np.arange(n, dtype=np.int64) - repeated_starts
    cnt_s = np.repeat(counts, counts)
    rev_s = cnt_s - 1 - pos_s

    new_batch = np.empty(n, dtype=bool)
    new_batch[0] = True
    new_batch[1:] = (us[1:] != us[:-1]) | (ts[1:] != ts[:-1])
    batch_starts = np.flatnonzero(new_batch)
    batch_counts = np.diff(np.r_[batch_starts, n])
    repeated_batch_starts = np.repeat(batch_starts, batch_counts)
    batch_pos_s = np.arange(n, dtype=np.int64) - repeated_batch_starts
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

    frac = pos / np.maximum(cnt - 1.0, 1.0)
    rev_frac = rev / np.maximum(cnt - 1.0, 1.0)
    batch_frac = batch_pos / np.maximum(batch_cnt - 1.0, 1.0)

    hour_id = np.asarray(s.X["hour"], dtype=np.float32)
    hour_value = np.mod(np.maximum(hour_id - 1.0, 0.0), 24.0)
    angle = 2.0 * np.pi * hour_value / 24.0

    numeric = []
    for name in [
        "duration_ms",
        "user_fans_user_num",
        "user_follow_user_num",
        "user_friend_user_num",
        "user_register_days",
    ]:
        a = np.asarray(s.num[name], dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        numeric.append(np.log1p(np.maximum(a, 0.0)))

    x = np.column_stack([
        frac,
        rev_frac,
        frac * frac,
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
        (pos == 0).astype(np.float32),
        (rev == 0).astype(np.float32),
        (batch_cnt > 1).astype(np.float32),
        (cnt <= 4).astype(np.float32),
        (cnt >= 8).astype(np.float32),
        *numeric,
    ]).astype(np.float32)

    cats = np.column_stack([
        np.asarray(s.X[f], dtype=np.int64) for f in CAT_FIELDS
    ]).astype(np.int64)

    ord_date = date_to_ordinal(s.date)
    return x, cats, ord_date


def fit_scaler(x):
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    return mean, std


def apply_scaler(x, mean, std):
    return np.clip((x - mean) / std, -8.0, 8.0).astype(np.float32)


class FirstOrderMixin:
    def first_order(self, cats, x):
        out = self.numeric_linear(x).squeeze(1) + self.bias
        for j, table in enumerate(self.first):
            out = out + table(cats[:, j]).squeeze(1)
        return out


class FieldAwareFM(nn.Module, FirstOrderMixin):
    def __init__(self, cards, d, base_rate, k=6):
        super().__init__()
        self.n_fields = len(cards)
        self.k = k
        self.first = nn.ModuleList([nn.Embedding(c, 1) for c in cards])
        self.ffm = nn.ModuleList([
            nn.Embedding(c, self.n_fields * k) for c in cards
        ])
        self.numeric_linear = nn.Linear(d, 1)
        self.numeric_interaction = nn.Linear(d, self.n_fields * k)
        self.bias = nn.Parameter(torch.tensor(
            np.log(base_rate / (1.0 - base_rate)), dtype=torch.float32
        ))

        for table in self.first:
            nn.init.zeros_(table.weight)
        for table in self.ffm:
            nn.init.normal_(table.weight, std=0.02)
        nn.init.zeros_(self.numeric_linear.weight)
        nn.init.zeros_(self.numeric_linear.bias)
        nn.init.normal_(self.numeric_interaction.weight, std=0.015)
        nn.init.zeros_(self.numeric_interaction.bias)

    def forward(self, cats, x):
        out = self.first_order(cats, x)
        vectors = [
            self.ffm[j](cats[:, j]).view(-1, self.n_fields, self.k)
            for j in range(self.n_fields)
        ]

        interaction = torch.zeros_like(out)
        for i in range(self.n_fields):
            vi = vectors[i]
            for j in range(i + 1, self.n_fields):
                interaction = interaction + (
                    vi[:, j, :] * vectors[j][:, i, :]
                ).sum(dim=1)

        context = self.numeric_interaction(x).view(
            -1, self.n_fields, self.k
        )
        for i in range(self.n_fields):
            interaction = interaction + (
                vectors[i][:, i, :] * context[:, i, :]
            ).sum(dim=1)

        return out + interaction


class AutoIntRanker(nn.Module, FirstOrderMixin):
    def __init__(self, cards, d, base_rate, k=12):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c, k) for c in cards])
        self.context_token = nn.Linear(d, k)
        self.attn1 = nn.MultiheadAttention(
            k, num_heads=3, dropout=0.0, batch_first=True
        )
        self.attn2 = nn.MultiheadAttention(
            k, num_heads=3, dropout=0.0, batch_first=True
        )
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.attn_output = nn.Linear((len(cards) + 1) * k, 1)

        self.first = nn.ModuleList([nn.Embedding(c, 1) for c in cards])
        self.numeric_linear = nn.Linear(d, 1)
        self.bias = nn.Parameter(torch.tensor(
            np.log(base_rate / (1.0 - base_rate)), dtype=torch.float32
        ))

        for table in self.emb:
            nn.init.normal_(table.weight, std=0.025)
        for table in self.first:
            nn.init.zeros_(table.weight)
        nn.init.xavier_uniform_(self.context_token.weight)
        nn.init.zeros_(self.context_token.bias)
        nn.init.zeros_(self.numeric_linear.weight)
        nn.init.zeros_(self.numeric_linear.bias)
        nn.init.xavier_uniform_(self.attn_output.weight)
        nn.init.zeros_(self.attn_output.bias)

    def forward(self, cats, x):
        tokens = torch.stack([
            self.emb[j](cats[:, j]) for j in range(cats.shape[1])
        ], dim=1)
        tokens = torch.cat(
            [tokens, self.context_token(x).unsqueeze(1)], dim=1
        )

        a, _ = self.attn1(tokens, tokens, tokens, need_weights=False)
        h = self.norm1(tokens + a)
        a, _ = self.attn2(h, h, h, need_weights=False)
        h = self.norm2(h + a)

        return self.first_order(cats, x) + self.attn_output(
            h.flatten(start_dim=1)
        ).squeeze(1)


class CINRanker(nn.Module, FirstOrderMixin):
    def __init__(self, cards, d, base_rate, k=8, widths=(12, 10)):
        super().__init__()
        self.n_fields = len(cards)
        self.emb = nn.ModuleList([nn.Embedding(c, k) for c in cards])
        self.context_token = nn.Linear(d, k)

        total_fields = self.n_fields + 1
        self.cin_layers = nn.ModuleList()
        previous = total_fields
        for width in widths:
            self.cin_layers.append(
                nn.Linear(total_fields * previous, width)
            )
            previous = width

        self.cin_output = nn.Linear(sum(widths), 1)
        self.first = nn.ModuleList([nn.Embedding(c, 1) for c in cards])
        self.numeric_linear = nn.Linear(d, 1)
        self.bias = nn.Parameter(torch.tensor(
            np.log(base_rate / (1.0 - base_rate)), dtype=torch.float32
        ))

        for table in self.emb:
            nn.init.normal_(table.weight, std=0.025)
        for table in self.first:
            nn.init.zeros_(table.weight)
        nn.init.xavier_uniform_(self.context_token.weight)
        nn.init.zeros_(self.context_token.bias)
        for layer in self.cin_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.cin_output.weight)
        nn.init.zeros_(self.cin_output.bias)
        nn.init.zeros_(self.numeric_linear.weight)
        nn.init.zeros_(self.numeric_linear.bias)

    def forward(self, cats, x):
        x0 = torch.stack([
            self.emb[j](cats[:, j]) for j in range(cats.shape[1])
        ], dim=1)
        x0 = torch.cat([x0, self.context_token(x).unsqueeze(1)], dim=1)

        h = x0
        summaries = []
        for layer in self.cin_layers:
            # Explicit field-wise outer products, independently for each
            # embedding coordinate.
            z = torch.einsum("bfk,bhk->bfhk", x0, h)
            z = z.reshape(z.shape[0], z.shape[1] * z.shape[2], z.shape[3])
            z = z.transpose(1, 2)
            h = F.silu(layer(z)).transpose(1, 2)
            summaries.append(h.sum(dim=2))

        cin = torch.cat(summaries, dim=1)
        return self.first_order(cats, x) + self.cin_output(cin).squeeze(1)


def build_model(family, cards, d, base_rate):
    base_rate = float(np.clip(base_rate, 1e-5, 1.0 - 1e-5))
    if family == "pointwise_ffm":
        return FieldAwareFM(cards, d, base_rate), 0.0018, 1e-6
    if family == "pointwise_autoint":
        return AutoIntRanker(cards, d, base_rate), 0.0015, 1e-5
    if family == "pointwise_cin":
        return CINRanker(cards, d, base_rate), 0.0016, 1e-5
    raise ValueError(family)


def recency_weights(ord_date):
    age = float(np.max(ord_date)) - ord_date.astype(np.float32)
    w = np.exp2(-age / HALF_LIFE_DAYS).astype(np.float32)
    w /= max(float(w.mean()), 1e-6)
    return w


def train_model(family, x, cats, y, ord_date, seed_offset=0):
    torch.manual_seed(SEED + seed_offset)
    cards = [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS]
    model, lr, wd = build_model(
        family, cards, x.shape[1], float(np.mean(y))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd
    )

    xt = torch.from_numpy(x)
    ct = torch.from_numpy(cats)
    yt = torch.from_numpy(y.astype(np.float32, copy=False))
    wt = torch.from_numpy(recency_weights(ord_date))

    n = len(y)
    generator = torch.Generator()
    generator.manual_seed(SEED + 7001 + seed_offset)

    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=generator)
        model.train()

        for st in range(0, n, BATCH):
            idx = perm[st:min(st + BATCH, n)]
            logits = model(ct[idx], xt[idx])
            losses = F.binary_cross_entropy_with_logits(
                logits, yt[idx], reduction="none"
            )
            loss = (losses * wt[idx]).sum() / wt[idx].sum().clamp_min(1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.no_grad()
def predict_model(model, x, cats):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float32)
    for st in range(0, x.shape[0], PRED_BATCH):
        en = min(st + PRED_BATCH, x.shape[0])
        result[st:en] = model(
            torch.from_numpy(cats[st:en]),
            torch.from_numpy(x[st:en])
        ).cpu().numpy()
    return result


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(x.std())
    if sd < 1e-12:
        sd = 1.0
    return (x - float(x.mean())) / sd


def within_user_rank(user, score):
    user = np.asarray(user, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    n = score.size
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, score, user))
    sorted_user = user[order]

    new = np.empty(n, dtype=bool)
    new[0] = True
    new[1:] = sorted_user[1:] != sorted_user[:-1]
    starts = np.flatnonzero(new)
    counts = np.diff(np.r_[starts, n])
    repeated_starts = np.repeat(starts, counts)
    repeated_counts = np.repeat(counts, counts)

    position = np.arange(n, dtype=np.float64) - repeated_starts
    ranked = position / np.maximum(repeated_counts - 1, 1)
    ranked[repeated_counts == 1] = 0.5

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

train_raw, train_cat, train_date = make_features(train)
valid_raw, valid_cat, valid_date = make_features(valid)

mean, std = fit_scaler(train_raw)
train_x = apply_scaler(train_raw, mean, std)
valid_x = apply_scaler(valid_raw, mean, std)

shared = os.environ["SHARED_ARTIFACTS"]
inc_valid = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float64
)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

inc_z = zscore(inc_valid)
inc_rank = within_user_rank(valid_users, inc_valid)

families = [
    "pointwise_ffm",
    "pointwise_autoint",
    "pointwise_cin",
]

candidate_scores = {}
candidate_specs = {}
raw_predictions = {}

inc_metrics = evaluate(valid_users, valid_y, inc_valid)
candidate_scores["incumbent"] = float(inc_metrics["primary"])
candidate_specs["incumbent"] = ("incumbent", 0.0, "raw")

best_name = "incumbent"
best_scores = inc_valid.copy()
best_metrics = inc_metrics
best_primary = float(inc_metrics["primary"])

for family_index, family in enumerate(families):
    model = train_model(
        family,
        train_x,
        train_cat,
        train_y,
        train_date,
        seed_offset=101 * (family_index + 1),
    )
    pred = predict_model(model, valid_x, valid_cat)
    raw_predictions[family] = pred

    met = evaluate(valid_users, valid_y, pred)
    candidate_scores[family] = float(met["primary"])
    candidate_specs[family] = (family, 0.0, "raw")

    if float(met["primary"]) > best_primary:
        best_primary = float(met["primary"])
        best_name = family
        best_scores = np.asarray(pred, dtype=np.float64)
        best_metrics = met

    pred_z = zscore(pred)
    pred_rank = within_user_rank(valid_users, pred)

    for weight in (0.08, 0.14, 0.20, 0.28, 0.36, 0.45, 0.55):
        name = f"{family}_zblend_{weight:.2f}"
        scores = (1.0 - weight) * inc_z + weight * pred_z
        met_blend = evaluate(valid_users, valid_y, scores)
        candidate_scores[name] = float(met_blend["primary"])
        candidate_specs[name] = (family, float(weight), "z")
        if float(met_blend["primary"]) > best_primary:
            best_primary = float(met_blend["primary"])
            best_name = name
            best_scores = scores.copy()
            best_metrics = met_blend

        name = f"{family}_rankblend_{weight:.2f}"
        scores = (1.0 - weight) * inc_rank + weight * pred_rank
        met_blend = evaluate(valid_users, valid_y, scores)
        candidate_scores[name] = float(met_blend["primary"])
        candidate_specs[name] = (family, float(weight), "rank")
        if float(met_blend["primary"]) > best_primary:
            best_primary = float(met_blend["primary"])
            best_name = name
            best_scores = scores.copy()
            best_metrics = met_blend

print("CANDIDATES " + json.dumps(
    {k: round(v, 6) for k, v in candidate_scores.items()},
    sort_keys=True
))

if len(families) >= 2:
    corr = {}
    for i in range(len(families)):
        for j in range(i + 1, len(families)):
            a = within_user_rank(valid_users, raw_predictions[families[i]])
            b = within_user_rank(valid_users, raw_predictions[families[j]])
            corr[f"{families[i]}__{families[j]}"] = float(
                np.corrcoef(a, b)[0, 1]
            )
    print("FINDINGS family_rank_correlations=" + json.dumps(corr, sort_keys=True))

chosen_family, chosen_weight, chosen_mode = candidate_specs[best_name]
print(
    "FINDINGS selected="
    + json.dumps({
        "name": best_name,
        "family": chosen_family,
        "weight": chosen_weight,
        "mode": chosen_mode,
    }, sort_keys=True)
)

# If the incumbent itself wins, retain its already-produced test scores.
if chosen_family == "incumbent":
    test_scores = np.asarray(np.load(inc_test_path), dtype=np.float64)
    chosen_raw_valid = inc_valid
else:
    # Refit the identical selected recipe on train + validation, with recency
    # measured relative to the new fitting endpoint.
    combined_x_raw = np.concatenate([train_raw, valid_raw], axis=0)
    combined_cat = np.concatenate([train_cat, valid_cat], axis=0)
    combined_y = np.concatenate([
        train_y,
        valid_y.astype(np.float32)
    ], axis=0)
    combined_date = np.concatenate([train_date, valid_date], axis=0)

    combined_mean, combined_std = fit_scaler(combined_x_raw)
    combined_x = apply_scaler(
        combined_x_raw, combined_mean, combined_std
    )

    chosen_index = families.index(chosen_family)
    final_model = train_model(
        chosen_family,
        combined_x,
        combined_cat,
        combined_y,
        combined_date,
        seed_offset=101 * (chosen_index + 1),
    )

    test = load("test")
    test_raw, test_cat, test_date = make_features(test)
    test_x = apply_scaler(test_raw, combined_mean, combined_std)
    raw_test_pred = predict_model(final_model, test_x, test_cat)

    chosen_raw_valid = raw_predictions[chosen_family]

    if chosen_mode == "raw":
        test_scores = np.asarray(raw_test_pred, dtype=np.float64)
    elif chosen_mode == "z":
        inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
        test_scores = (
            (1.0 - chosen_weight) * zscore(inc_test)
            + chosen_weight * zscore(raw_test_pred)
        )
    elif chosen_mode == "rank":
        inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
        test_users = np.asarray(test.user_id, dtype=np.int64)
        test_scores = (
            (1.0 - chosen_weight)
            * within_user_rank(test_users, inc_test)
            + chosen_weight
            * within_user_rank(test_users, raw_test_pred)
        )
    else:
        raise ValueError(chosen_mode)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64)
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )
    if chosen_family != "incumbent" and chosen_mode != "raw":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(chosen_raw_valid, dtype=np.float64)
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))