import os
import time
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 73129
BATCH = 16384
PRED_BATCH = 65536
EPOCHS = 2

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

BASE_FIELDS = [
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

BASE_CARDS = [int(FEATURE_CARDINALITIES[f]) for f in BASE_FIELDS]
ORIGIN = datetime(2022, 4, 9)


def absolute_day(date_array):
    date_array = np.asarray(date_array, dtype=np.int64)
    out = np.empty(date_array.size, dtype=np.float32)
    for value in np.unique(date_array):
        dt = datetime.strptime(str(int(value)), "%Y%m%d")
        out[date_array == value] = float((dt - ORIGIN).days)
    return out


def base_cats(split):
    return np.column_stack([
        np.asarray(split.X[f], dtype=np.int64) for f in BASE_FIELDS
    ]).astype(np.int64)


def crossed_cats(base):
    columns = [base[:, j] for j in range(base.shape[1])]
    cards = list(BASE_CARDS)

    # Explicit context interactions whose effects can differ within a user.
    specs = [
        (0, 2),  # video x tab
        (1, 2),  # author x tab
        (4, 2),  # tag x tab
        (3, 2),  # duration x tab
        (0, 7),  # video x hour
        (1, 4),  # author x tag
        (4, 3),  # tag x duration
        (8, 2),  # onehot3 x tab
    ]
    for a, b in specs:
        columns.append(base[:, a] * BASE_CARDS[b] + base[:, b])
        cards.append(BASE_CARDS[a] * BASE_CARDS[b])

    return np.column_stack(columns).astype(np.int64), cards


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

    position = np.arange(n, dtype=np.float64) - np.repeat(starts, counts)
    repeated_counts = np.repeat(counts, counts)
    rank = position / np.maximum(repeated_counts - 1, 1)
    rank[repeated_counts == 1] = 0.5

    out = np.empty(n, dtype=np.float64)
    out[order] = rank
    return out


def empirical_bayes_fit(cats, y, cards, alpha=24.0):
    y = np.asarray(y, dtype=np.float64)
    prior = float(np.clip(y.mean(), 1e-5, 1.0 - 1e-5))
    prior_logit = np.log(prior / (1.0 - prior))
    tables = []

    for j, card in enumerate(cards):
        count = np.bincount(cats[:, j], minlength=card).astype(np.float64)
        positive = np.bincount(
            cats[:, j], weights=y, minlength=card
        ).astype(np.float64)
        rate = (positive + alpha * prior) / (count + alpha)
        rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
        tables.append((np.log(rate / (1.0 - rate)) - prior_logit).astype(
            np.float32
        ))

    return prior_logit, tables


def empirical_bayes_predict(cats, prior_logit, tables):
    # Downweight redundant author/context fields relative to video identity.
    weights = np.asarray(
        [1.00, 0.55, 0.35, 0.30, 0.45, 0.25, 0.20, 0.18, 0.40],
        dtype=np.float64,
    )
    score = np.full(cats.shape[0], prior_logit, dtype=np.float64)
    for j, table in enumerate(tables):
        score += weights[j] * table[cats[:, j]]
    return score


class CrossWide(nn.Module):
    def __init__(self, cards, base_rate):
        super().__init__()
        self.tables = nn.ModuleList([nn.Embedding(c, 1) for c in cards])
        self.bias = nn.Parameter(torch.tensor(
            np.log(base_rate / (1.0 - base_rate)), dtype=torch.float32
        ))
        for table in self.tables:
            nn.init.zeros_(table.weight)

    def forward(self, cats, day_value=None):
        out = self.bias.expand(cats.shape[0])
        for j, table in enumerate(self.tables):
            out = out + table(cats[:, j]).squeeze(1)
        return out


class TemporalSlopeGAM(nn.Module):
    def __init__(self, cards, base_rate):
        super().__init__()
        self.intercept = nn.ModuleList([nn.Embedding(c, 1) for c in cards])
        self.slope = nn.ModuleList([nn.Embedding(c, 1) for c in cards])
        self.bias = nn.Parameter(torch.tensor(
            np.log(base_rate / (1.0 - base_rate)), dtype=torch.float32
        ))
        for table in self.intercept:
            nn.init.zeros_(table.weight)
        for table in self.slope:
            nn.init.zeros_(table.weight)

    def forward(self, cats, day_value):
        out = self.bias.expand(cats.shape[0])
        for j in range(cats.shape[1]):
            intercept = self.intercept[j](cats[:, j]).squeeze(1)
            slope = self.slope[j](cats[:, j]).squeeze(1)
            out = out + intercept + day_value * slope
        return out


def fit_torch_model(family, cats, day_value, y, cards):
    torch.manual_seed(SEED + (17 if family == "cross_wide" else 43))
    y = np.asarray(y, dtype=np.float32)
    base_rate = float(np.clip(y.mean(), 1e-5, 1.0 - 1e-5))

    if family == "cross_wide":
        model = CrossWide(cards, base_rate)
        lr = 0.012
        weight_decay = 2e-5
    else:
        model = TemporalSlopeGAM(cards, base_rate)
        lr = 0.008
        weight_decay = 8e-5

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    cats_t = torch.from_numpy(cats)
    day_t = torch.from_numpy(day_value.astype(np.float32, copy=False))
    y_t = torch.from_numpy(y)
    n = y.size
    generator = torch.Generator()
    generator.manual_seed(SEED + 101)

    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        model.train()
        for st in range(0, n, BATCH):
            idx = permutation[st:min(st + BATCH, n)]
            logits = model(cats_t[idx], day_t[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y_t[idx])

            if family == "temporal_slope":
                # Strongly shrink extrapolated category trends unless supported
                # repeatedly across the training dates.
                slope_penalty = torch.zeros((), dtype=torch.float32)
                for table in model.slope:
                    slope_penalty = slope_penalty + table.weight.square().mean()
                loss = loss + 0.02 * slope_penalty

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.no_grad()
def predict_torch(model, cats, day_value):
    model.eval()
    result = np.empty(cats.shape[0], dtype=np.float32)
    for st in range(0, cats.shape[0], PRED_BATCH):
        en = min(st + PRED_BATCH, cats.shape[0])
        result[st:en] = model(
            torch.from_numpy(cats[st:en]),
            torch.from_numpy(day_value[st:en].astype(np.float32, copy=False)),
        ).cpu().numpy()
    return result


train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

train_base = base_cats(train)
valid_base = base_cats(valid)

train_day_abs = absolute_day(train.date)
valid_day_abs = absolute_day(valid.date)

# Fixed absolute centering makes validation genuinely extrapolative and permits
# the identical transformation after refitting on train + validation.
train_day_scaled = ((train_day_abs - 6.0) / 6.0).astype(np.float32)
valid_day_scaled = ((valid_day_abs - 6.0) / 6.0).astype(np.float32)

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.load(os.path.join(shared, "incumbent_valid_scores.npy"))
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

candidate_scores = {}
own_valid = {}
models = {}

inc_metrics = evaluate(valid_users, valid_y, inc_valid)
candidate_scores["incumbent"] = float(inc_metrics["primary"])

# Family 1: analytical empirical-Bayes additive likelihood ratios.
eb_prior, eb_tables = empirical_bayes_fit(
    train_base, train_y, BASE_CARDS, alpha=24.0
)
eb_valid = empirical_bayes_predict(valid_base, eb_prior, eb_tables)
own_valid["empirical_bayes"] = eb_valid

# Family 2: explicit memorized categorical crosses.
train_cross, CROSS_CARDS = crossed_cats(train_base)
valid_cross, _ = crossed_cats(valid_base)
cross_model = fit_torch_model(
    "cross_wide",
    train_cross,
    train_day_scaled,
    train_y,
    CROSS_CARDS,
)
models["cross_wide"] = cross_model
cross_valid = predict_torch(cross_model, valid_cross, valid_day_scaled)
own_valid["cross_wide"] = cross_valid

# Family 3: category-specific temporal coefficient extrapolation.
trend_model = fit_torch_model(
    "temporal_slope",
    train_base,
    train_day_scaled,
    train_y,
    BASE_CARDS,
)
models["temporal_slope"] = trend_model
trend_valid = predict_torch(trend_model, valid_base, valid_day_scaled)
own_valid["temporal_slope"] = trend_valid

best_name = "incumbent"
best_family = None
best_mode = "incumbent"
best_weight = 0.0
best_scores = np.asarray(inc_valid, dtype=np.float64)
best_metrics = inc_metrics
best_primary = float(inc_metrics["primary"])

inc_z = zscore(inc_valid)
inc_rank = within_user_rank(valid_users, inc_valid)

for family, prediction in own_valid.items():
    raw_metrics = evaluate(valid_users, valid_y, prediction)
    candidate_scores[family] = float(raw_metrics["primary"])
    if float(raw_metrics["primary"]) > best_primary:
        best_name = family
        best_family = family
        best_mode = "raw"
        best_weight = 1.0
        best_scores = np.asarray(prediction, dtype=np.float64)
        best_metrics = raw_metrics
        best_primary = float(raw_metrics["primary"])

    prediction_z = zscore(prediction)
    prediction_rank = within_user_rank(valid_users, prediction)

    for weight in (0.05, 0.10, 0.15, 0.22, 0.30, 0.40, 0.52):
        z_blend = (1.0 - weight) * inc_z + weight * prediction_z
        name = "%s_zblend_%.2f" % (family, weight)
        metrics = evaluate(valid_users, valid_y, z_blend)
        candidate_scores[name] = float(metrics["primary"])
        if float(metrics["primary"]) > best_primary:
            best_name = name
            best_family = family
            best_mode = "z"
            best_weight = float(weight)
            best_scores = z_blend.copy()
            best_metrics = metrics
            best_primary = float(metrics["primary"])

        rank_blend = (
            (1.0 - weight) * inc_rank + weight * prediction_rank
        )
        name = "%s_rankblend_%.2f" % (family, weight)
        metrics = evaluate(valid_users, valid_y, rank_blend)
        candidate_scores[name] = float(metrics["primary"])
        if float(metrics["primary"]) > best_primary:
            best_name = name
            best_family = family
            best_mode = "rank"
            best_weight = float(weight)
            best_scores = rank_blend.copy()
            best_metrics = metrics
            best_primary = float(metrics["primary"])

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS selected=%s family=%s mode=%s weight=%.2f"
    % (best_name, str(best_family), best_mode, best_weight)
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_mode not in ("raw", "incumbent") and best_family is not None:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(own_valid[best_family], dtype=np.float64),
        )

# Refit the selected recipe on train + validation and score test.
test = load("test")
test_users = np.asarray(test.user_id, dtype=np.int64)
test_base = base_cats(test)
test_day_abs = absolute_day(test.date)
test_day_scaled = ((test_day_abs - 6.0) / 6.0).astype(np.float32)

if best_mode == "incumbent":
    test_scores = np.load(inc_test_path).astype(np.float64)
else:
    combined_base = np.concatenate([train_base, valid_base], axis=0)
    combined_y = np.concatenate([
        train_y,
        valid_y.astype(np.float32),
    ])
    combined_day = np.concatenate([
        train_day_scaled,
        valid_day_scaled,
    ])

    if best_family == "empirical_bayes":
        final_prior, final_tables = empirical_bayes_fit(
            combined_base, combined_y, BASE_CARDS, alpha=24.0
        )
        own_test = empirical_bayes_predict(
            test_base, final_prior, final_tables
        )
    elif best_family == "cross_wide":
        combined_cross, final_cross_cards = crossed_cats(combined_base)
        test_cross, _ = crossed_cats(test_base)
        final_model = fit_torch_model(
            "cross_wide",
            combined_cross,
            combined_day,
            combined_y,
            final_cross_cards,
        )
        own_test = predict_torch(
            final_model, test_cross, test_day_scaled
        )
    else:
        final_model = fit_torch_model(
            "temporal_slope",
            combined_base,
            combined_day,
            combined_y,
            BASE_CARDS,
        )
        own_test = predict_torch(
            final_model, test_base, test_day_scaled
        )

    if best_mode == "raw":
        test_scores = np.asarray(own_test, dtype=np.float64)
    else:
        inc_test = np.load(inc_test_path).astype(np.float64)
        if best_mode == "z":
            test_scores = (
                (1.0 - best_weight) * zscore(inc_test)
                + best_weight * zscore(own_test)
            )
        else:
            test_scores = (
                (1.0 - best_weight)
                * within_user_rank(test_users, inc_test)
                + best_weight
                * within_user_rank(test_users, own_test)
            )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))