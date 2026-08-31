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
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2025
THREADS = min(8, os.cpu_count() or 1)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

CAT_FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]
BATCH_SIZE = 4096
EPOCHS = 3

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
n_train = len(train.user_id)


def recency_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.int64)
    age = int(dates.max()) - dates
    return np.power(0.5, age.astype(np.float32) / float(half_life)).astype(np.float32)


w_train = recency_weights(train.date, 4.0)
w_train /= np.mean(w_train)


# ----------------------------------------------------------------------
# Shared categorical and historical/numeric inputs
# ----------------------------------------------------------------------
cards = [int(FEATURE_CARDINALITIES[f]) for f in CAT_FIELDS]
offsets = np.zeros(len(cards), dtype=np.int64)
offsets[1:] = np.cumsum(cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))


def categorical_matrix(split):
    z = np.empty((len(split.user_id), len(CAT_FIELDS)), dtype=np.int64)
    for j, f in enumerate(CAT_FIELDS):
        z[:, j] = np.asarray(split.X[f], dtype=np.int64) + offsets[j]
    return z


xcat_train = categorical_matrix(train)
xcat_valid = categorical_matrix(valid)
xcat_test = categorical_matrix(test)


def collect_raw_numeric(split_name, split):
    columns = []
    names = []

    for name in NUM_FIELDS:
        a = np.asarray(split.num[name], dtype=np.float32)
        if name != "user_register_days":
            a = np.log1p(np.maximum(a, 0.0))
        columns.append(a)
        names.append(name)

    for key in ["video_id", "author_id"]:
        h = historical_features(split_name, key=key)
        for name in sorted(h):
            columns.append(np.asarray(h[name], dtype=np.float32))
            names.append(name)

    return np.column_stack(columns).astype(np.float32), names


raw_num_train, numeric_names = collect_raw_numeric("train", train)
raw_num_valid, _ = collect_raw_numeric("valid", valid)
raw_num_test, _ = collect_raw_numeric("test", test)

finite_train = np.where(np.isfinite(raw_num_train), raw_num_train, np.nan)
num_mean = np.nanmean(finite_train, axis=0).astype(np.float32)
num_mean = np.where(np.isfinite(num_mean), num_mean, 0.0).astype(np.float32)

num_std = np.nanstd(finite_train, axis=0).astype(np.float32)
num_std = np.where((np.isfinite(num_std)) & (num_std > 1e-5), num_std, 1.0).astype(
    np.float32
)


def normalize_numeric(a):
    a = np.asarray(a, dtype=np.float32)
    a = np.where(np.isfinite(a), a, num_mean[None, :])
    a = (a - num_mean[None, :]) / num_std[None, :]
    return np.clip(a, -8.0, 8.0).astype(np.float32)


xnum_train = normalize_numeric(raw_num_train)
xnum_valid = normalize_numeric(raw_num_valid)
xnum_test = normalize_numeric(raw_num_test)
n_num = xnum_train.shape[1]


# ----------------------------------------------------------------------
# Deep CTR families
# ----------------------------------------------------------------------
class DeepFM(nn.Module):
    def __init__(self, cardinality, n_fields, n_numeric, rank, base_rate):
        super().__init__()
        self.embedding = nn.Embedding(cardinality, rank)
        self.linear = nn.Embedding(cardinality, 1)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)

        deep_input = n_fields * rank + n_numeric
        self.deep = nn.Sequential(
            nn.Linear(deep_input, 128),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.numeric_linear = nn.Linear(n_numeric, 1, bias=False)
        p = float(np.clip(base_rate, 1e-6, 1 - 1e-6))
        self.bias = nn.Parameter(torch.tensor(np.log(p / (1.0 - p)), dtype=torch.float32))

    def forward(self, xcat, xnum):
        e = self.embedding(xcat)
        linear = self.linear(xcat).sum(dim=1).squeeze(1)
        summed = e.sum(dim=1)
        fm = 0.5 * (summed.square().sum(dim=1) - e.square().sum(dim=(1, 2)))
        deep_input = torch.cat([e.flatten(1), xnum], dim=1)
        deep = self.deep(deep_input).squeeze(1)
        return self.bias + linear + fm + deep + self.numeric_linear(xnum).squeeze(1)


class NFM(nn.Module):
    def __init__(self, cardinality, n_numeric, rank, base_rate):
        super().__init__()
        self.embedding = nn.Embedding(cardinality, rank)
        self.linear = nn.Embedding(cardinality, 1)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)

        self.interaction_net = nn.Sequential(
            nn.Linear(rank + n_numeric, 96),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.numeric_linear = nn.Linear(n_numeric, 1, bias=False)
        p = float(np.clip(base_rate, 1e-6, 1 - 1e-6))
        self.bias = nn.Parameter(torch.tensor(np.log(p / (1.0 - p)), dtype=torch.float32))

    def forward(self, xcat, xnum):
        e = self.embedding(xcat)
        linear = self.linear(xcat).sum(dim=1).squeeze(1)
        summed = e.sum(dim=1)
        bi = 0.5 * (summed.square() - e.square().sum(dim=1))
        hidden = self.interaction_net(torch.cat([bi, xnum], dim=1)).squeeze(1)
        return self.bias + linear + hidden + self.numeric_linear(xnum).squeeze(1)


def fit_neural(model, seed):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-6)
    model.train()

    for epoch in range(EPOCHS):
        order = rng.permutation(n_train)
        total_loss = 0.0
        total_weight = 0.0

        for start in range(0, n_train, BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            cat = torch.from_numpy(xcat_train[idx])
            num = torch.from_numpy(xnum_train[idx])
            target = torch.from_numpy(y_train[idx])
            weight = torch.from_numpy(w_train[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(cat, num)
            row_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            loss = (row_loss * weight).sum() / weight.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float((row_loss.detach() * weight).sum())
            total_weight += float(weight.sum())

        print(
            "neural_epoch model=%s epoch=%d loss=%.6f"
            % (model.__class__.__name__, epoch + 1, total_loss / total_weight),
            flush=True,
        )
    return model


def predict_neural(model, xcat, xnum):
    out = np.empty(xcat.shape[0], dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, xcat.shape[0], 16384):
            end = min(start + 16384, xcat.shape[0])
            logits = model(
                torch.from_numpy(xcat[start:end]),
                torch.from_numpy(xnum[start:end]),
            )
            out[start:end] = torch.sigmoid(logits).cpu().numpy()
    return out


deepfm = fit_neural(
    DeepFM(total_cardinality, len(CAT_FIELDS), n_num, rank=12, base_rate=y_train.mean()),
    SEED,
)
deepfm_valid = predict_neural(deepfm, xcat_valid, xnum_valid)
deepfm_test = predict_neural(deepfm, xcat_test, xnum_test)
del deepfm

nfm = fit_neural(
    NFM(total_cardinality, n_num, rank=16, base_rate=y_train.mean()),
    SEED + 1,
)
nfm_valid = predict_neural(nfm, xcat_valid, xnum_valid)
nfm_test = predict_neural(nfm, xcat_test, xnum_test)
del nfm


# ----------------------------------------------------------------------
# Gradient-boosted binary and listwise ranking families
# ----------------------------------------------------------------------
def lgb_matrix(split, normalized_numeric):
    cats = np.column_stack(
        [np.asarray(split.X[f], dtype=np.float32) for f in CAT_FIELDS]
    )
    return np.ascontiguousarray(
        np.column_stack([cats, normalized_numeric]).astype(np.float32)
    )


xlgb_train = lgb_matrix(train, xnum_train)
xlgb_valid = lgb_matrix(valid, xnum_valid)
xlgb_test = lgb_matrix(test, xnum_test)
categorical_indices = list(range(len(CAT_FIELDS)))

binary_params = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.055,
    "num_leaves": 48,
    "max_depth": -1,
    "min_data_in_leaf": 250,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 1.5,
    "max_bin": 127,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "num_threads": THREADS,
    "verbose": -1,
}

dtrain_binary = lgb.Dataset(
    xlgb_train,
    label=y_train,
    weight=w_train,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
binary_model = lgb.train(binary_params, dtrain_binary, num_boost_round=180)
lgb_binary_valid = binary_model.predict(xlgb_valid)
lgb_binary_test = binary_model.predict(xlgb_test)


# LambdaRank requires rows grouped by user. Ties in time are kept in row order.
rank_order = np.lexsort(
    (
        np.arange(n_train, dtype=np.int64),
        np.asarray(train.time_ms, dtype=np.int64),
        np.asarray(train.user_id, dtype=np.int64),
    )
)
sorted_users = np.asarray(train.user_id, dtype=np.int64)[rank_order]
group_starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
group_sizes = np.diff(np.r_[group_starts, n_train]).astype(np.int32)

rank_params = {
    "objective": "lambdarank",
    "metric": "None",
    "learning_rate": 0.05,
    "num_leaves": 48,
    "min_data_in_leaf": 250,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.88,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "lambdarank_truncation_level": 10,
    "label_gain": [0, 1],
    "seed": SEED + 7,
    "num_threads": THREADS,
    "verbose": -1,
}

dtrain_rank = lgb.Dataset(
    xlgb_train[rank_order],
    label=y_train[rank_order],
    weight=w_train[rank_order],
    group=group_sizes,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
rank_model = lgb.train(rank_params, dtrain_rank, num_boost_round=160)
lgb_rank_valid = rank_model.predict(xlgb_valid)
lgb_rank_test = rank_model.predict(xlgb_test)


# ----------------------------------------------------------------------
# Non-parametric empirical Bayes family
# ----------------------------------------------------------------------
def smoothed_field_effect(field, smoothing):
    ids = np.asarray(train.X[field], dtype=np.int64)
    card = int(FEATURE_CARDINALITIES[field])
    count = np.bincount(ids, weights=w_train, minlength=card).astype(np.float64)
    positive = np.bincount(
        ids, weights=w_train * y_train, minlength=card
    ).astype(np.float64)

    global_rate = float(np.sum(w_train * y_train) / np.sum(w_train))
    rate = (positive + smoothing * global_rate) / (count + smoothing)
    rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
    global_logit = np.log(global_rate / (1.0 - global_rate))
    return np.log(rate / (1.0 - rate)) - global_logit, global_logit


eb_specs = [
    ("user_id", 35.0, 0.50),
    ("video_id", 45.0, 0.75),
    ("author_id", 60.0, 0.55),
    ("tab", 300.0, 0.25),
    ("duration_bucket", 180.0, 0.30),
]
eb_effects = {}
eb_intercept = 0.0
for i, (field, smoothing, coefficient) in enumerate(eb_specs):
    effect, base_logit = smoothed_field_effect(field, smoothing)
    eb_effects[field] = coefficient * effect
    if i == 0:
        eb_intercept = base_logit


def predict_eb(split):
    score = np.full(len(split.user_id), eb_intercept, dtype=np.float64)
    for field, _, _ in eb_specs:
        ids = np.asarray(split.X[field], dtype=np.int64)
        effect = eb_effects[field]
        safe = np.minimum(ids, len(effect) - 1)
        score += effect[safe]
    return 1.0 / (1.0 + np.exp(-np.clip(score, -25.0, 25.0)))


eb_valid = predict_eb(valid)
eb_test = predict_eb(test)


# ----------------------------------------------------------------------
# Tie-aware within-user rank aggregation with the trusted incumbent.
# This removes incompatible score calibration while preserving each model's
# ordering. Equal tree leaves retain equal ranks rather than acquiring row-order
# signal.
# ----------------------------------------------------------------------
def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((scores, user_ids))
    su = user_ids[order]
    ss = scores[order]

    group_start_mask = np.r_[True, su[1:] != su[:-1]]
    user_starts = np.flatnonzero(group_start_mask)
    user_ends = np.r_[user_starts[1:], n]
    user_lengths = user_ends - user_starts

    tie_start_mask = np.r_[
        True,
        (su[1:] != su[:-1]) | (ss[1:] != ss[:-1]),
    ]
    tie_starts = np.flatnonzero(tie_start_mask)
    tie_ends = np.r_[tie_starts[1:], n]
    tie_lengths = tie_ends - tie_starts
    tie_mid = (tie_starts + tie_ends - 1).astype(np.float64) * 0.5
    absolute_mid = np.repeat(tie_mid, tie_lengths)

    repeated_user_start = np.repeat(user_starts, user_lengths)
    repeated_denominator = np.repeat(np.maximum(user_lengths - 1, 1), user_lengths)
    ranked_sorted = (absolute_mid - repeated_user_start) / repeated_denominator
    singleton = np.repeat(user_lengths == 1, user_lengths)
    ranked_sorted[singleton] = 0.5

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


own_valid = {
    "deepfm": deepfm_valid,
    "nfm": nfm_valid,
    "lgb_binary_recency": lgb_binary_valid,
    "lgb_lambdarank_recency": lgb_rank_valid,
    "empirical_bayes": eb_valid,
}
own_test = {
    "deepfm": deepfm_test,
    "nfm": nfm_test,
    "lgb_binary_recency": lgb_binary_test,
    "lgb_lambdarank_recency": lgb_rank_test,
    "empirical_bayes": eb_test,
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.load(inc_valid_path).astype(np.float64)
inc_test = np.load(inc_test_path).astype(np.float64)

inc_rank_valid = within_user_rank(valid.user_id, inc_valid)
inc_rank_test = within_user_rank(test.user_id, inc_test)

candidate_scores = {}
best_primary = -np.inf
winner_name = None
winner_valid = None
winner_test = None
winner_raw_valid = None

blend_grid = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]

for family in own_valid:
    raw_v = np.asarray(own_valid[family], dtype=np.float64)
    raw_t = np.asarray(own_test[family], dtype=np.float64)
    standalone = evaluate(valid.user_id, valid.y, raw_v)
    candidate_scores[family] = float(standalone["primary"])

    own_rank_v = within_user_rank(valid.user_id, raw_v)
    own_rank_t = within_user_rank(test.user_id, raw_t)

    family_best = None
    family_best_alpha = None
    family_best_valid = None
    family_best_test = None

    for alpha in blend_grid:
        blended_v = alpha * own_rank_v + (1.0 - alpha) * inc_rank_valid
        m = evaluate(valid.user_id, valid.y, blended_v)
        if family_best is None or m["primary"] > family_best["primary"]:
            family_best = m
            family_best_alpha = alpha
            family_best_valid = blended_v.copy()
            family_best_test = alpha * own_rank_t + (1.0 - alpha) * inc_rank_test

    blend_name = "%s_blend_a%.2f" % (family, family_best_alpha)
    candidate_scores[blend_name] = float(family_best["primary"])

    if family_best["primary"] > best_primary:
        best_primary = float(family_best["primary"])
        winner_name = blend_name
        winner_valid = family_best_valid
        winner_test = family_best_test
        winner_raw_valid = raw_v

metrics = evaluate(valid.user_id, valid.y, winner_valid)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(winner_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(winner_raw_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(winner_test, dtype=np.float64),
    )

print("FINDINGS winner=%s" % winner_name, flush=True)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True), flush=True)

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)