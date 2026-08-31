import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 20260831
THREADS = min(8, os.cpu_count() or 8)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.float32)
yva = np.asarray(valid.y, dtype=np.int8)
uva = np.asarray(valid.user_id, dtype=np.int64)
ute = np.asarray(test.user_id, dtype=np.int64)
ntr = len(ytr)

FEATURES = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "user_active_degree",
    "register_days_bucket",
    "register_days_range",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "music_type",
    "video_type",
    "is_live_streamer",
    "is_video_author",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

NUM_FEATURES = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

CONTEXT_FIELDS = [
    "user_id",
    "tab",
    "hour",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "is_live_streamer",
]

ITEM_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "video_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

PROP_FIELDS = CONTEXT_FIELDS + ITEM_FIELDS

# ----------------------------------------------------------------------
# Train-only temporal weights.
# ----------------------------------------------------------------------
dates = np.asarray(train.date, dtype=np.int64)
unique_dates = np.unique(dates)
date_to_index = {int(d): i for i, d in enumerate(unique_dates)}
day_index = np.fromiter(
    (date_to_index[int(d)] for d in dates),
    dtype=np.int16,
    count=ntr,
)
age = (len(unique_dates) - 1 - day_index).astype(np.float32)
recency = np.exp2(-age / 4.0).astype(np.float32)
recency /= recency.mean()

# ----------------------------------------------------------------------
# Matrices and train-only numeric normalization.
# ----------------------------------------------------------------------
num_center = {}
num_scale = {}

for field in NUM_FEATURES:
    raw = np.asarray(train.num[field], dtype=np.float64)
    z = np.log1p(np.maximum(np.nan_to_num(raw, nan=0.0), 0.0))
    q25, med, q75 = np.quantile(z, [0.25, 0.50, 0.75])
    num_center[field] = float(med)
    num_scale[field] = max(float(q75 - q25), 1.0e-3)


def model_matrix(sample):
    cols = [
        np.asarray(sample.X[field], dtype=np.float32)
        for field in FEATURES
    ]
    for field in NUM_FEATURES:
        raw = np.asarray(sample.num[field], dtype=np.float64)
        z = np.log1p(np.maximum(np.nan_to_num(raw, nan=0.0), 0.0))
        z = (z - num_center[field]) / num_scale[field]
        cols.append(np.clip(z, -8.0, 8.0).astype(np.float32))
    return np.column_stack(cols).astype(np.float32, copy=False)


def categorical_matrix(sample, fields):
    return np.column_stack([
        np.asarray(sample.X[field], dtype=np.float32)
        for field in fields
    ]).astype(np.float32, copy=False)


xtr = model_matrix(train)
xva = model_matrix(valid)
xte = model_matrix(test)
cat_indices = list(range(len(FEATURES)))

# ----------------------------------------------------------------------
# Exposure-support nuisance: logged pairing versus shuffled item block.
# This uses no outcomes from validation/test and no auxiliary row outcomes.
# ----------------------------------------------------------------------
rng = np.random.default_rng(SEED)
prop_all = categorical_matrix(train, PROP_FIELDS)

prop_sample_size = min(ntr, 650000)
real_rows = rng.choice(ntr, size=prop_sample_size, replace=False)
shuffle_rows = rng.permutation(real_rows)

prop_real_fit = prop_all[real_rows]
prop_fake_fit = prop_real_fit.copy()
item_start = len(CONTEXT_FIELDS)
prop_fake_fit[:, item_start:] = prop_all[shuffle_rows, item_start:]

prop_fit_x = np.concatenate([prop_real_fit, prop_fake_fit], axis=0)
prop_fit_y = np.concatenate([
    np.ones(prop_sample_size, dtype=np.float32),
    np.zeros(prop_sample_size, dtype=np.float32),
])

prop_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.08,
    "num_leaves": 31,
    "max_depth": 7,
    "min_data_in_leaf": 1000,
    "lambda_l2": 5.0,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "max_bin": 127,
    "max_cat_threshold": 32,
    "cat_smooth": 30.0,
    "verbose": -1,
    "num_threads": THREADS,
    "seed": SEED + 1,
    "feature_fraction_seed": SEED + 2,
    "bagging_seed": SEED + 3,
}

prop_dset = lgb.Dataset(
    prop_fit_x,
    label=prop_fit_y,
    categorical_feature=list(range(len(PROP_FIELDS))),
    free_raw_data=True,
)
prop_model = lgb.train(prop_params, prop_dset, num_boost_round=85)

prop_tr = np.clip(prop_model.predict(prop_all), 0.15, 0.95).astype(np.float32)
prop_va_x = categorical_matrix(valid, PROP_FIELDS)
prop_te_x = categorical_matrix(test, PROP_FIELDS)
prop_va = np.clip(prop_model.predict(prop_va_x), 0.15, 0.95).astype(np.float32)
prop_te = np.clip(prop_model.predict(prop_te_x), 0.15, 0.95).astype(np.float32)

print(
    "FINDINGS propensity_mean=%.6f propensity_q05=%.6f "
    "propensity_q50=%.6f propensity_q95=%.6f"
    % (
        float(prop_tr.mean()),
        float(np.quantile(prop_tr, 0.05)),
        float(np.quantile(prop_tr, 0.50)),
        float(np.quantile(prop_tr, 0.95)),
    )
)

del (
    prop_all,
    prop_real_fit,
    prop_fake_fit,
    prop_fit_x,
    prop_fit_y,
    prop_dset,
    prop_model,
    prop_va_x,
    prop_te_x,
)
gc.collect()

# Add propensity as an explicit numeric support feature for outcome models.
xtr_p = np.column_stack([xtr, prop_tr]).astype(np.float32, copy=False)
xva_p = np.column_stack([xva, prop_va]).astype(np.float32, copy=False)
xte_p = np.column_stack([xte, prop_te]).astype(np.float32, copy=False)

base_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 8,
    "min_data_in_leaf": 850,
    "lambda_l2": 5.0,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.88,
    "bagging_freq": 1,
    "max_bin": 127,
    "max_cat_threshold": 32,
    "cat_smooth": 25.0,
    "verbose": -1,
    "num_threads": THREADS,
    "seed": SEED + 10,
    "feature_fraction_seed": SEED + 11,
    "bagging_seed": SEED + 12,
}

# ----------------------------------------------------------------------
# Two-fold cross-fitted outcome nuisance. The folds are deterministic and
# independent of labels. Out-of-fold residuals prevent a flexible nuisance
# model from erasing the residual correction on its own training rows.
# ----------------------------------------------------------------------
row_ids = np.arange(ntr, dtype=np.uint64)
fold = (
    (row_ids * np.uint64(11400714819323198485) + np.uint64(SEED)) >> np.uint64(63)
).astype(np.int8)

oof = np.empty(ntr, dtype=np.float32)

for k in (0, 1):
    fit_idx = np.flatnonzero(fold != k)
    pred_idx = np.flatnonzero(fold == k)

    dfit = lgb.Dataset(
        xtr_p[fit_idx],
        label=ytr[fit_idx],
        weight=recency[fit_idx],
        categorical_feature=cat_indices,
        free_raw_data=True,
    )
    nuisance = lgb.train(base_params, dfit, num_boost_round=125)
    oof[pred_idx] = nuisance.predict(xtr_p[pred_idx]).astype(np.float32)

    del fit_idx, pred_idx, dfit, nuisance
    gc.collect()

oof = np.clip(oof, 0.01, 0.99)

# Clipped doubly robust / augmented-IPW pseudo-outcome.
raw_residual = ytr - oof
dr_target = oof + raw_residual / prop_tr
dr_target = np.clip(dr_target, -1.5, 2.5).astype(np.float32)

dr_weight = recency * np.sqrt(prop_tr)
dr_weight /= dr_weight.mean()

print(
    "FINDINGS oof_logloss=%.6f residual_mean=%.6f "
    "dr_mean=%.6f dr_q01=%.6f dr_q99=%.6f"
    % (
        float(
            -np.mean(
                ytr * np.log(oof) + (1.0 - ytr) * np.log(1.0 - oof)
            )
        ),
        float(raw_residual.mean()),
        float(dr_target.mean()),
        float(np.quantile(dr_target, 0.01)),
        float(np.quantile(dr_target, 0.99)),
    )
)

# ----------------------------------------------------------------------
# Family 1: direct recency-weighted binary GBDT.
# ----------------------------------------------------------------------
dfull = lgb.Dataset(
    xtr_p,
    label=ytr,
    weight=recency,
    categorical_feature=cat_indices,
    free_raw_data=False,
)
direct_model = lgb.train(base_params, dfull, num_boost_round=190)
direct_valid = direct_model.predict(xva_p).astype(np.float64)
direct_test = direct_model.predict(xte_p).astype(np.float64)

# This full-data nuisance is also the base term for residual EB correction.
nuisance_valid = direct_valid.copy()
nuisance_test = direct_test.copy()

del direct_model, dfull
gc.collect()

# ----------------------------------------------------------------------
# Family 2: GBDT regression fitted to the DR pseudo-outcome.
# ----------------------------------------------------------------------
dr_params = dict(base_params)
dr_params.update({
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.045,
    "seed": SEED + 20,
    "feature_fraction_seed": SEED + 21,
    "bagging_seed": SEED + 22,
})

ddr = lgb.Dataset(
    xtr_p,
    label=dr_target,
    weight=dr_weight,
    categorical_feature=cat_indices,
    free_raw_data=True,
)
dr_model = lgb.train(dr_params, ddr, num_boost_round=180)
dr_valid = dr_model.predict(xva_p).astype(np.float64)
dr_test = dr_model.predict(xte_p).astype(np.float64)

del dr_model, ddr
gc.collect()

# ----------------------------------------------------------------------
# Family 3: empirical-Bayes correction of the full nuisance. Tables estimate
# cross-fitted inverse-propensity residuals, with shrinkage toward zero.
# ----------------------------------------------------------------------
EB_TERMS = [
    ("video_id",),
    ("author_id",),
    ("tag",),
    ("duration_bucket",),
    ("upload_type",),
    ("onehot_feat3",),
    ("onehot_feat8",),
    ("tab", "tag"),
    ("tab", "author_id"),
    ("duration_bucket", "tag"),
]


def term_values(sample, fields):
    if len(fields) == 1:
        f = fields[0]
        return (
            np.asarray(sample.X[f], dtype=np.int64),
            int(FEATURE_CARDINALITIES[f]),
        )
    left, right = fields
    rc = int(FEATURE_CARDINALITIES[right])
    values = (
        np.asarray(sample.X[left], dtype=np.int64) * rc
        + np.asarray(sample.X[right], dtype=np.int64)
    )
    card = int(FEATURE_CARDINALITIES[left]) * rc
    return values, card


ipw_residual = np.clip(raw_residual / prop_tr, -3.0, 3.0)
eb_valid_corr = np.zeros(len(yva), dtype=np.float64)
eb_test_corr = np.zeros(len(ute), dtype=np.float64)

for term in EB_TERMS:
    tr_values, card = term_values(train, term)
    va_values, _ = term_values(valid, term)
    te_values, _ = term_values(test, term)

    counts = np.bincount(
        tr_values,
        weights=recency,
        minlength=card,
    ).astype(np.float64)
    sums = np.bincount(
        tr_values,
        weights=recency * ipw_residual,
        minlength=card,
    ).astype(np.float64)

    shrink = 80.0 if len(term) == 1 else 140.0
    table = sums / (counts + shrink)

    eb_valid_corr += table[va_values]
    eb_test_corr += table[te_values]

    del tr_values, va_values, te_values, counts, sums, table

eb_valid = nuisance_valid + eb_valid_corr / len(EB_TERMS)
eb_test = nuisance_test + eb_test_corr / len(EB_TERMS)

# ----------------------------------------------------------------------
# Family 4: factorization model regressing the same DR pseudo-outcome.
# Its prediction is formed through global latent pairwise interactions rather
# than tree partitions or independent entity tables.
# ----------------------------------------------------------------------
FM_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "user_active_degree",
    "register_days_bucket",
    "music_type",
    "video_type",
    "onehot_feat3",
    "onehot_feat8",
]

offsets = []
running = 0
for field in FM_FIELDS:
    offsets.append(running)
    running += int(FEATURE_CARDINALITIES[field])
offsets = np.asarray(offsets, dtype=np.int64)


def fm_batch(sample, rows):
    arr = np.column_stack([
        np.asarray(sample.X[field], dtype=np.int64)[rows]
        for field in FM_FIELDS
    ])
    arr += offsets[None, :]
    return arr


class DRFactorizationMachine(nn.Module):
    def __init__(self, cardinality, dim=12):
        super().__init__()
        self.linear = nn.Embedding(cardinality, 1)
        self.embedding = nn.Embedding(cardinality, dim)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.015)

    def forward(self, indices):
        linear = self.linear(indices).sum(dim=1).squeeze(-1)
        emb = self.embedding(indices)
        summed = emb.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1)
            - emb.square().sum(dim=(1, 2))
        )
        return self.bias + linear + interaction


fm = DRFactorizationMachine(running, dim=12)
optimizer = torch.optim.AdamW(fm.parameters(), lr=0.0025, weight_decay=3.0e-6)
order = np.arange(ntr, dtype=np.int64)
batch_size = 16384

for epoch in range(2):
    rng.shuffle(order)
    total_loss = 0.0
    total_weight = 0.0
    fm.train()

    for begin in range(0, ntr, batch_size):
        rows = order[begin:begin + batch_size]
        xb = torch.from_numpy(fm_batch(train, rows))
        target = torch.from_numpy(dr_target[rows])
        weight = torch.from_numpy(dr_weight[rows])

        optimizer.zero_grad(set_to_none=True)
        prediction = fm(xb)
        losses = nn.functional.smooth_l1_loss(
            prediction,
            target,
            reduction="none",
            beta=0.5,
        )
        loss = (losses * weight).sum() / weight.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fm.parameters(), 5.0)
        optimizer.step()

        total_loss += float((losses.detach() * weight).sum())
        total_weight += float(weight.sum())

    print(
        "FINDINGS dr_fm_epoch=%d weighted_smooth_l1=%.6f"
        % (epoch + 1, total_loss / max(total_weight, 1.0))
    )


def fm_predict(sample):
    result = np.empty(len(sample.user_id), dtype=np.float64)
    fm.eval()
    all_rows = np.arange(len(result), dtype=np.int64)
    with torch.no_grad():
        for begin in range(0, len(result), 32768):
            rows = all_rows[begin:begin + 32768]
            xb = torch.from_numpy(fm_batch(sample, rows))
            result[rows] = fm(xb).cpu().numpy().astype(np.float64)
    return result


fm_valid = fm_predict(valid)
fm_test = fm_predict(test)

del fm, optimizer
gc.collect()

# ----------------------------------------------------------------------
# Rank-normalized incumbent blends. The metric depends only on within-user
# ordering, so percentile ranks put heterogeneous family scales on a common
# basis. The same selected family and alpha are applied unchanged to test.
# ----------------------------------------------------------------------
def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    rows = np.arange(len(scores), dtype=np.int64)

    order_idx = np.lexsort((rows, scores, users))
    sorted_users = users[order_idx]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], len(order_idx)]
    lengths = ends - starts

    positions = (
        np.arange(len(order_idx), dtype=np.float64)
        - np.repeat(starts, lengths)
    )
    denom = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    ranked = positions / denom

    result = np.empty(len(scores), dtype=np.float64)
    result[order_idx] = ranked
    return result


shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

inc_valid = np.load(inc_valid_path).astype(np.float64)
inc_test = np.load(inc_test_path).astype(np.float64)
inc_valid_rank = within_user_rank(uva, inc_valid)
inc_test_rank = within_user_rank(ute, inc_test)

families = {
    "direct_gbdt": (direct_valid, direct_test),
    "dr_gbdt": (dr_valid, dr_test),
    "dr_empirical_bayes": (eb_valid, eb_test),
    "dr_factorization": (fm_valid, fm_test),
}

candidate_scores = {}
best_primary = -np.inf
best_metrics = None
best_valid = None
best_test = None
best_raw_valid = None
best_name = None
best_alpha = None

alphas = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0]

for name, (va_raw, te_raw) in families.items():
    raw_metrics = evaluate(uva, yva, va_raw)
    candidate_scores[name + "_raw"] = float(raw_metrics["primary"])

    va_rank = within_user_rank(uva, va_raw)
    te_rank = within_user_rank(ute, te_raw)

    family_best = -np.inf
    family_best_alpha = None

    for alpha in alphas:
        va_blend = (1.0 - alpha) * inc_valid_rank + alpha * va_rank
        metrics = evaluate(uva, yva, va_blend)
        key = "%s_blend_a%.2f" % (name, alpha)
        candidate_scores[key] = float(metrics["primary"])

        if metrics["primary"] > family_best:
            family_best = float(metrics["primary"])
            family_best_alpha = alpha

        if metrics["primary"] > best_primary:
            best_primary = float(metrics["primary"])
            best_metrics = metrics
            best_valid = va_blend.copy()
            best_test = (
                (1.0 - alpha) * inc_test_rank + alpha * te_rank
            )
            best_raw_valid = va_raw.copy()
            best_name = name
            best_alpha = alpha

    print(
        "FINDINGS family=%s raw_primary=%.6f best_blend_primary=%.6f "
        "best_alpha=%.2f"
        % (
            name,
            float(raw_metrics["primary"]),
            family_best,
            family_best_alpha,
        )
    )

print(
    "FINDINGS selected_family=%s selected_alpha=%.2f"
    % (best_name, best_alpha)
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)