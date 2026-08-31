import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 84631
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

# Use identical safe inputs for both structurally different rankers.
CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag", "upload_type",
    "music_type", "duration_bucket", "video_type", "hour",
    "user_active_degree", "fans_user_num_range", "follow_user_num_range",
    "friend_user_num_range", "register_days_range",
    "onehot_feat0", "onehot_feat1", "onehot_feat2", "onehot_feat3",
    "onehot_feat7", "onehot_feat8",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]


def raw_numeric(s):
    cols = []
    for name in NUM_FIELDS:
        z = np.asarray(s.num[name], dtype=np.float32)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        z = np.sign(z) * np.log1p(np.abs(z))
        cols.append(z)
    return np.column_stack(cols).astype(np.float32)


def fit_num_scaler(x):
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    return mean, std


def scale_numeric(x, mean, std):
    return np.clip((x - mean) / std, -8.0, 8.0).astype(np.float32)


def categorical_arrays(s):
    return [
        np.asarray(s.X[name], dtype=np.int64)
        for name in CAT_FIELDS
    ]


def dense_matrix(cat_arrays, numeric):
    # LightGBM accepts integer-valued categorical columns represented in a
    # float32 dense matrix when their indices are declared categorical.
    n = numeric.shape[0]
    out = np.empty((n, len(CAT_FIELDS) + numeric.shape[1]), dtype=np.float32)
    for j, a in enumerate(cat_arrays):
        out[:, j] = a
    out[:, len(CAT_FIELDS):] = numeric
    return out


def grouped_order(users):
    users = np.asarray(users, dtype=np.int64)
    row = np.arange(users.size, dtype=np.int64)
    order = np.lexsort((row, users))
    us = users[order]
    starts = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1]
    groups = np.diff(np.r_[starts, len(us)]).astype(np.int32)
    return order, groups


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int64)
    uniq = np.unique(dates)
    # Date ordinal rather than YYYYMMDD arithmetic.
    ordinals = np.searchsorted(uniq, dates)
    age = ordinals.max() - ordinals
    w = np.exp2(-age.astype(np.float32) / 9.0)
    return w.astype(np.float32)


def fit_lambdamart(x, y, users, dates, rounds=230):
    order, groups = grouped_order(users)
    xs = x[order]
    ys = np.asarray(y, dtype=np.int8)[order]
    rw = recency_weights(dates)[order]

    # Give approximately equal total influence to each query while retaining
    # modest positive-count emphasis compatible with GAUC.
    inv_group = np.repeat(
        1.0 / np.sqrt(np.maximum(groups.astype(np.float32), 1.0)), groups
    )
    weights = rw * inv_group

    ds = lgb.Dataset(
        xs,
        label=ys,
        group=groups,
        weight=weights,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )
    params = {
        "objective": "lambdarank",
        "metric": "None",
        "label_gain": [0, 1],
        "lambdarank_truncation_level": 5,
        "learning_rate": 0.055,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 180,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "cat_smooth": 30.0,
        "cat_l2": 15.0,
        "num_threads": min(8, os.cpu_count() or 1),
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "deterministic": True,
        "force_col_wise": True,
        "verbose": -1,
    }
    return lgb.train(params, ds, num_boost_round=rounds)


class ListwiseAdditive(nn.Module):
    """Scalar contribution per category plus a linear numeric contribution."""

    def __init__(self, num_numeric):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(int(FEATURE_CARDINALITIES[name]), 1)
            for name in CAT_FIELDS
        ])
        self.numeric = nn.Linear(num_numeric, 1, bias=True)
        for emb in self.embeddings:
            nn.init.zeros_(emb.weight)
        nn.init.zeros_(self.numeric.weight)
        nn.init.zeros_(self.numeric.bias)

    def forward(self, cats, nums):
        out = self.numeric(nums).squeeze(1)
        for emb, c in zip(self.embeddings, cats):
            out = out + emb(c).squeeze(1)
        return out


def group_batch_boundaries(groups, target_rows=110000):
    cumulative = np.r_[0, np.cumsum(groups, dtype=np.int64)]
    boundaries = [0]
    g = 0
    ng = len(groups)
    while g < ng:
        target = cumulative[g] + target_rows
        nxt = int(np.searchsorted(cumulative, target, side="right") - 1)
        nxt = max(g + 1, min(nxt, ng))
        boundaries.append(nxt)
        g = nxt
    return cumulative, boundaries


def fit_listwise(cat_arrays, numeric, y, users, dates, epochs=4):
    order, groups = grouped_order(users)
    cats = [
        torch.from_numpy(np.asarray(a[order], dtype=np.int64))
        for a in cat_arrays
    ]
    nums = torch.from_numpy(np.asarray(numeric[order], dtype=np.float32))
    labels = torch.from_numpy(np.asarray(y[order], dtype=np.float32))
    row_recency = torch.from_numpy(recency_weights(dates)[order])

    model = ListwiseAdditive(numeric.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.018, weight_decay=2e-5
    )
    cumulative, boundaries = group_batch_boundaries(groups)
    rng = np.random.default_rng(SEED + 77)

    for epoch in range(epochs):
        batch_ids = np.arange(len(boundaries) - 1)
        rng.shuffle(batch_ids)
        model.train()

        for bi in batch_ids:
            g0, g1 = boundaries[bi], boundaries[bi + 1]
            r0, r1 = int(cumulative[g0]), int(cumulative[g1])
            local_groups = groups[g0:g1].astype(np.int64)
            qid_np = np.repeat(
                np.arange(len(local_groups), dtype=np.int64), local_groups
            )
            qid = torch.from_numpy(qid_np)
            local_y = labels[r0:r1]

            logits = model([c[r0:r1] for c in cats], nums[r0:r1])
            nq = len(local_groups)

            maxima = torch.full((nq,), -torch.inf, dtype=logits.dtype)
            maxima.scatter_reduce_(0, qid, logits, reduce="amax", include_self=True)
            shifted = logits - maxima[qid]
            exp_sum = torch.zeros(nq, dtype=logits.dtype)
            exp_sum.index_add_(0, qid, torch.exp(shifted))
            log_denom = maxima + torch.log(exp_sum.clamp_min(1e-12))

            pos_count = torch.zeros(nq, dtype=logits.dtype)
            pos_logit_sum = torch.zeros(nq, dtype=logits.dtype)
            pos_count.index_add_(0, qid, local_y)
            pos_logit_sum.index_add_(0, qid, logits * local_y)
            usable = pos_count > 0

            if not bool(usable.any()):
                continue

            query_loss = (
                log_denom[usable]
                - pos_logit_sum[usable] / pos_count[usable]
            )

            # Constant query-level temporal weight, obtained from the most
            # recent impression in each complete user query.
            local_rw = row_recency[r0:r1]
            qweight = torch.zeros(nq, dtype=logits.dtype)
            qweight.scatter_reduce_(
                0, qid, local_rw, reduce="amax", include_self=True
            )
            qweight = torch.sqrt(qweight[usable].clamp_min(1e-3))
            loss = (query_loss * qweight).sum() / qweight.sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


@torch.no_grad()
def predict_listwise(model, cat_arrays, numeric, batch=65536):
    model.eval()
    n = numeric.shape[0]
    out = np.empty(n, dtype=np.float32)
    for st in range(0, n, batch):
        en = min(st + batch, n)
        cats = [
            torch.from_numpy(np.asarray(a[st:en], dtype=np.int64))
            for a in cat_arrays
        ]
        nums = torch.from_numpy(
            np.asarray(numeric[st:en], dtype=np.float32)
        )
        out[st:en] = model(cats, nums).cpu().numpy()
    return out


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(x.std())
    if sd < 1e-12:
        sd = 1.0
    return (x - float(x.mean())) / sd


def within_user_rank(scores, users):
    """Fractional rank in [0,1], deterministic under score ties."""
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users, dtype=np.int64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, users))
    us = users[order]
    starts = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1]
    counts = np.diff(np.r_[starts, n])
    positions = np.arange(n, dtype=np.int64) - np.repeat(starts, counts)
    denom = np.maximum(np.repeat(counts, counts) - 1, 1)
    ranked = positions.astype(np.float64) / denom
    out = np.empty(n, dtype=np.float64)
    out[order] = ranked
    return out


train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.int8)
valid_y = np.asarray(valid.y, dtype=np.int8)
train_users = np.asarray(train.user_id, dtype=np.int64)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

tr_cats = categorical_arrays(train)
va_cats = categorical_arrays(valid)
tr_num_raw = raw_numeric(train)
va_num_raw = raw_numeric(valid)
num_mean, num_std = fit_num_scaler(tr_num_raw)
tr_num = scale_numeric(tr_num_raw, num_mean, num_std)
va_num = scale_numeric(va_num_raw, num_mean, num_std)

tr_dense = dense_matrix(tr_cats, tr_num)
va_dense = dense_matrix(va_cats, va_num)

lambda_model = fit_lambdamart(
    tr_dense, train_y, train_users, train.date, rounds=230
)
lambda_valid = lambda_model.predict(
    va_dense, num_iteration=lambda_model.current_iteration()
).astype(np.float32)

list_model = fit_listwise(
    tr_cats, tr_num, train_y, train_users, train.date, epochs=4
)
list_valid = predict_listwise(list_model, va_cats, va_num)

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

families = {
    "lambda_top5": lambda_valid,
    "listwise_additive": list_valid,
}

candidate_scores = {}
best_primary = -np.inf
best_metrics = None
best_scores = None
best_family = None
best_mode = None
best_weight = 0.0

inc_z = standardize(inc_valid)
inc_rank = within_user_rank(inc_valid, valid_users)

for family, own in families.items():
    met = evaluate(valid_users, valid_y, own)
    candidate_scores[family] = float(met["primary"])
    if float(met["primary"]) > best_primary:
        best_primary = float(met["primary"])
        best_metrics = met
        best_scores = np.asarray(own, dtype=np.float64)
        best_family = family
        best_mode = "standalone"
        best_weight = 1.0

    own_z = standardize(own)
    own_rank = within_user_rank(own, valid_users)

    for w in (0.10, 0.18, 0.26, 0.34, 0.44, 0.56):
        blended = (1.0 - w) * inc_z + w * own_z
        met_b = evaluate(valid_users, valid_y, blended)
        name = "%s_scoreblend_%.2f" % (family, w)
        candidate_scores[name] = float(met_b["primary"])
        if float(met_b["primary"]) > best_primary:
            best_primary = float(met_b["primary"])
            best_metrics = met_b
            best_scores = blended.copy()
            best_family = family
            best_mode = "scoreblend"
            best_weight = float(w)

        rank_blended = (1.0 - w) * inc_rank + w * own_rank
        met_r = evaluate(valid_users, valid_y, rank_blended)
        name = "%s_rankblend_%.2f" % (family, w)
        candidate_scores[name] = float(met_r["primary"])
        if float(met_r["primary"]) > best_primary:
            best_primary = float(met_r["primary"])
            best_metrics = met_r
            best_scores = rank_blended.copy()
            best_family = family
            best_mode = "rankblend"
            best_weight = float(w)

corr_lambda = float(np.corrcoef(
    within_user_rank(lambda_valid, valid_users),
    inc_rank
)[0, 1])
corr_list = float(np.corrcoef(
    within_user_rank(list_valid, valid_users),
    inc_rank
)[0, 1])
corr_new = float(np.corrcoef(
    within_user_rank(lambda_valid, valid_users),
    within_user_rank(list_valid, valid_users)
)[0, 1])

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True), flush=True)
print(
    "FINDINGS rank_corr_inc_lambda=%.4f rank_corr_inc_listwise=%.4f "
    "rank_corr_lambda_listwise=%.4f selected=%s/%s/%.2f"
    % (
        corr_lambda, corr_list, corr_new,
        best_family, best_mode, best_weight,
    ),
    flush=True,
)

# Refit the selected recipe on train + validation, without reading test labels.
combined_y = np.concatenate([train_y, valid_y])
combined_users = np.concatenate([train_users, valid_users])
combined_dates = np.concatenate([
    np.asarray(train.date), np.asarray(valid.date)
])
combined_cats = [
    np.concatenate([a, b]) for a, b in zip(tr_cats, va_cats)
]
combined_num_raw = np.concatenate([tr_num_raw, va_num_raw], axis=0)
cmean, cstd = fit_num_scaler(combined_num_raw)
combined_num = scale_numeric(combined_num_raw, cmean, cstd)

test = load("test")
test_users = np.asarray(test.user_id, dtype=np.int64)
test_cats = categorical_arrays(test)
test_num_raw = raw_numeric(test)
test_num = scale_numeric(test_num_raw, cmean, cstd)

if best_family == "lambda_top5":
    combined_dense = dense_matrix(combined_cats, combined_num)
    test_dense = dense_matrix(test_cats, test_num)
    final_model = fit_lambdamart(
        combined_dense, combined_y, combined_users, combined_dates,
        rounds=230
    )
    own_test = final_model.predict(
        test_dense, num_iteration=final_model.current_iteration()
    ).astype(np.float64)
    own_valid_selected = lambda_valid
else:
    final_model = fit_listwise(
        combined_cats, combined_num, combined_y,
        combined_users, combined_dates, epochs=4
    )
    own_test = predict_listwise(
        final_model, test_cats, test_num
    ).astype(np.float64)
    own_valid_selected = list_valid

if best_mode == "standalone":
    test_scores = own_test
elif best_mode == "scoreblend":
    inc_test = np.load(inc_test_path).astype(np.float64)
    test_scores = (
        (1.0 - best_weight) * standardize(inc_test)
        + best_weight * standardize(own_test)
    )
else:
    inc_test = np.load(inc_test_path).astype(np.float64)
    test_scores = (
        (1.0 - best_weight) * within_user_rank(inc_test, test_users)
        + best_weight * within_user_rank(own_test, test_users)
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
    if best_mode != "standalone":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(own_valid_selected, dtype=np.float64),
        )

elapsed = float(time.time() - START)
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": elapsed,
}
print("METRICS " + json.dumps(result, separators=(", ", ": ")))