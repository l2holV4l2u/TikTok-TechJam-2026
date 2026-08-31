import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

START = time.time()

train = load("train")
valid = load("valid")
test = load("test")

y = np.asarray(train.y, dtype=np.float64)
yv = np.asarray(valid.y, dtype=np.int8)

ntr = len(y)
nva = len(valid.user_id)
nte = len(test.user_id)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)


def per_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    su = user_ids[order]
    ss = scores[order]

    group_start = np.empty(n, dtype=bool)
    group_start[0] = True
    group_start[1:] = su[1:] != su[:-1]

    # Average ranks for exact score ties, computed without Python user loops.
    tie_start = np.empty(n, dtype=bool)
    tie_start[0] = True
    tie_start[1:] = group_start[1:] | (ss[1:] != ss[:-1])

    tie_starts = np.flatnonzero(tie_start)
    tie_ends = np.r_[tie_starts[1:], n]
    tie_sizes = tie_ends - tie_starts
    tie_average_global = (tie_starts + tie_ends - 1.0) / 2.0
    average_global_rank = np.repeat(tie_average_global, tie_sizes)

    starts = np.where(group_start, np.arange(n), 0)
    starts = np.maximum.accumulate(starts)
    within = average_global_rank - starts

    user_starts = np.flatnonzero(group_start)
    user_ends = np.r_[user_starts[1:], n]
    user_sizes = user_ends - user_starts
    sizes = np.repeat(user_sizes, user_sizes).astype(np.float64)

    normalized = np.where(sizes > 1, within / (sizes - 1.0), 0.5)
    result = np.empty(n, dtype=np.float64)
    result[order] = normalized
    return result


def group_context(user_ids, x):
    """Within-user z-score and distance from the user's strongest candidate."""
    users = np.asarray(user_ids, dtype=np.int64)
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))

    _, inverse = np.unique(users, return_inverse=True)
    ng = int(inverse.max()) + 1

    count = np.bincount(inverse, minlength=ng).astype(np.float64)
    total = np.bincount(inverse, weights=x, minlength=ng)
    total2 = np.bincount(inverse, weights=x * x, minlength=ng)

    mean = total / np.maximum(count, 1.0)
    variance = np.maximum(
        total2 / np.maximum(count, 1.0) - mean * mean, 1e-8
    )
    std = np.sqrt(variance)

    maximum = np.full(ng, -np.inf, dtype=np.float64)
    minimum = np.full(ng, np.inf, dtype=np.float64)
    np.maximum.at(maximum, inverse, x)
    np.minimum.at(minimum, inverse, x)

    z = (x - mean[inverse]) / std[inverse]
    span = np.maximum(maximum - minimum, 1e-6)
    max_gap = (maximum[inverse] - x) / span[inverse]

    # Singleton groups contain no ranking information.
    singleton = count[inverse] <= 1
    z[singleton] = 0.0
    max_gap[singleton] = 0.0

    return (
        np.clip(z, -6.0, 6.0).astype(np.float32),
        np.clip(max_gap, 0.0, 1.0).astype(np.float32),
    )


# Training-only recency weighting. The half-life retains support from the
# complete train window but emphasizes the dates nearest evaluation.
dates = np.asarray(train.date, dtype=np.int32)
age = dates.max() - dates
weights = np.exp(-np.log(2.0) * age.astype(np.float64) / 7.0)
weights /= weights.mean()
global_rate = float(np.sum(weights * y) / np.sum(weights))

categorical_fields = [
    "video_id",
    "author_id",
    "user_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "video_type",
    "hour",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat11",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_bucket",
]

numeric_fields = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

tr_columns = []
va_columns = []
te_columns = []
column_names = []

# Recency-weighted leave-one-out target evidence. Validation and test mappings
# are always formed solely from train.
for field in categorical_fields:
    ids_tr = np.asarray(train.X[field], dtype=np.int64)
    ids_va = np.asarray(valid.X[field], dtype=np.int64)
    ids_te = np.asarray(test.X[field], dtype=np.int64)
    card = int(FEATURE_CARDINALITIES[field])

    count = np.bincount(
        ids_tr, weights=weights, minlength=card
    ).astype(np.float64)
    positive = np.bincount(
        ids_tr, weights=weights * y, minlength=card
    ).astype(np.float64)

    if card < 100:
        prior = 18.0
    elif card < 2000:
        prior = 32.0
    else:
        prior = 48.0

    loo_count = np.maximum(count[ids_tr] - weights, 0.0)
    loo_positive = positive[ids_tr] - weights * y

    train_rate = (
        loo_positive + prior * global_rate
    ) / np.maximum(loo_count + prior, 1e-12)

    full_rate = (
        positive + prior * global_rate
    ) / np.maximum(count + prior, 1e-12)

    tr_columns.append(train_rate.astype(np.float32))
    va_columns.append(full_rate[ids_va].astype(np.float32))
    te_columns.append(full_rate[ids_te].astype(np.float32))
    column_names.append("te_" + field)

    # Support is useful for deciding whether identity evidence should dominate.
    if field in ("video_id", "author_id", "user_id"):
        tr_columns.append(
            np.log1p(np.maximum(loo_count, 0.0)).astype(np.float32)
        )
        va_columns.append(
            np.log1p(count[ids_va]).astype(np.float32)
        )
        te_columns.append(
            np.log1p(count[ids_te]).astype(np.float32)
        )
        column_names.append("support_" + field)


# Continuous quantities are heavy-tailed.
for field in numeric_fields:
    a = np.asarray(train.num[field], dtype=np.float64)
    b = np.asarray(valid.num[field], dtype=np.float64)
    c = np.asarray(test.num[field], dtype=np.float64)

    a = np.log1p(np.maximum(np.nan_to_num(a), 0.0)).astype(np.float32)
    b = np.log1p(np.maximum(np.nan_to_num(b), 0.0)).astype(np.float32)
    c = np.log1p(np.maximum(np.nan_to_num(c), 0.0)).astype(np.float32)

    tr_columns.append(a)
    va_columns.append(b)
    te_columns.append(c)
    column_names.append("num_" + field)


# Supplied histories are train-only, with leave-one-out values on train.
for entity in ("video_id", "author_id"):
    htr = historical_features("train", key=entity)
    hva = historical_features("valid", key=entity)
    hte = historical_features("test", key=entity)

    names = sorted(set(htr) & set(hva) & set(hte))
    for name in names:
        a = np.nan_to_num(np.asarray(htr[name], dtype=np.float32))
        b = np.nan_to_num(np.asarray(hva[name], dtype=np.float32))
        c = np.nan_to_num(np.asarray(hte[name], dtype=np.float32))

        tr_columns.append(a)
        va_columns.append(b)
        te_columns.append(c)
        column_names.append(name)


# Candidate-set context is attached to the strongest stable evidence columns.
# This uses only the feature rows in each prediction set, never labels.
context_priority = [
    "te_video_id",
    "te_author_id",
    "te_tab",
    "te_tag",
    "te_duration_bucket",
    "te_upload_type",
    "te_onehot_feat3",
    "te_onehot_feat8",
    "num_duration_ms",
]

for i, name in enumerate(list(column_names)):
    if (
        "long_view_rate" in name
        or name in context_priority
    ):
        if sum(n.startswith("context_z_") for n in column_names) >= 14:
            break

        trz, trg = group_context(train.user_id, tr_columns[i])
        vaz, vag = group_context(valid.user_id, va_columns[i])
        tez, teg = group_context(test.user_id, te_columns[i])

        tr_columns.extend([trz, trg])
        va_columns.extend([vaz, vag])
        te_columns.extend([tez, teg])
        column_names.extend([
            "context_z_" + name,
            "context_maxgap_" + name,
        ])


Xtr = np.column_stack(tr_columns).astype(np.float32)
Xva = np.column_stack(va_columns).astype(np.float32)
Xte = np.column_stack(te_columns).astype(np.float32)

del tr_columns, va_columns, te_columns
gc.collect()

# Training-only robust scaling. It is particularly important for linear leaves.
sample_rng = np.random.default_rng(73129)
scale_sample_size = min(300000, ntr)
scale_idx = sample_rng.choice(
    ntr, size=scale_sample_size, replace=False,
    p=weights / weights.sum()
)

center = np.median(Xtr[scale_idx], axis=0).astype(np.float64)
q25 = np.percentile(Xtr[scale_idx], 25, axis=0)
q75 = np.percentile(Xtr[scale_idx], 75, axis=0)
scale = np.maximum(q75 - q25, 1e-3)

Xtr = np.clip((Xtr - center) / scale, -8.0, 8.0).astype(np.float32)
Xva = np.clip((Xva - center) / scale, -8.0, 8.0).astype(np.float32)
Xte = np.clip((Xte - center) / scale, -8.0, 8.0).astype(np.float32)


# ----------------------------------------------------------------------
# Family 1: contextual additive scorecard.
#
# Every continuous evidence feature is quantile-binned. Each bin receives a
# smoothed recency-weighted long-view estimate, and ridge regression combines
# those univariate nonlinear response curves. There are no cross-feature
# interactions, making this deliberately stationary and low variance.
# ----------------------------------------------------------------------

rng = np.random.default_rng(2026)
edge_idx = rng.choice(
    ntr, size=min(220000, ntr), replace=False,
    p=weights / weights.sum()
)

dim = Xtr.shape[1]
Gtr = np.empty((ntr, dim), dtype=np.float32)
Gva = np.empty((nva, dim), dtype=np.float32)
Gte = np.empty((nte, dim), dtype=np.float32)

for j in range(dim):
    probe = Xtr[edge_idx, j].astype(np.float64)
    edges = np.unique(np.quantile(probe, np.linspace(0.0, 1.0, 25)[1:-1]))

    bt = np.searchsorted(edges, Xtr[:, j], side="right")
    bv = np.searchsorted(edges, Xva[:, j], side="right")
    be = np.searchsorted(edges, Xte[:, j], side="right")
    nb = len(edges) + 1

    counts = np.bincount(
        bt, weights=weights, minlength=nb
    ).astype(np.float64)
    positives = np.bincount(
        bt, weights=weights * y, minlength=nb
    ).astype(np.float64)

    prior = 120.0
    loo_count = np.maximum(counts[bt] - weights, 0.0)
    loo_pos = positives[bt] - weights * y

    rtr = (
        loo_pos + prior * global_rate
    ) / np.maximum(loo_count + prior, 1e-12)
    full = (
        positives + prior * global_rate
    ) / np.maximum(counts + prior, 1e-12)

    Gtr[:, j] = rtr.astype(np.float32)
    Gva[:, j] = full[bv].astype(np.float32)
    Gte[:, j] = full[be].astype(np.float32)

gmean = np.sum(Gtr.astype(np.float64) * weights[:, None], axis=0)
gmean /= np.sum(weights)

Gtr -= gmean.astype(np.float32)
Gva -= gmean.astype(np.float32)
Gte -= gmean.astype(np.float32)

# Fit on a large recency-weighted sample to keep the dense normal equations
# inexpensive while retaining all dates.
gam_fit_size = min(520000, ntr)
gam_idx = rng.choice(
    ntr, size=gam_fit_size, replace=False,
    p=weights / weights.sum()
)
Gs = Gtr[gam_idx].astype(np.float64)
ys = y[gam_idx]
ws = weights[gam_idx].astype(np.float64)
ws /= ws.mean()

target_mean = float(np.average(ys, weights=ws))
target = ys - target_mean

A = (Gs * ws[:, None]).T @ Gs
b = (Gs * ws[:, None]).T @ target
gam_coef = np.linalg.solve(
    A + 12.0 * np.eye(dim, dtype=np.float64), b
)

gam_valid = Gva.astype(np.float64) @ gam_coef + target_mean
gam_test = Gte.astype(np.float64) @ gam_coef + target_mean

del Gtr, Gva, Gte, Gs, A, b
gc.collect()


# ----------------------------------------------------------------------
# Family 2: model-tree boosting.
#
# Unlike ordinary constant-leaf GBDT, each leaf fits a regularized linear
# response. Splits discover regimes such as high-support versus cold items,
# while the leaf regressions smoothly rank candidates inside each regime.
# ----------------------------------------------------------------------

dtrain = lgb.Dataset(
    Xtr,
    label=y.astype(np.float32),
    weight=weights.astype(np.float32),
    free_raw_data=False,
)

linear_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.045,
    "num_leaves": 23,
    "max_depth": 7,
    "min_data_in_leaf": 1800,
    "min_sum_hessian_in_leaf": 25.0,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.82,
    "bagging_freq": 1,
    "lambda_l1": 0.15,
    "lambda_l2": 5.0,
    "linear_tree": True,
    "linear_lambda": 8.0,
    "max_bin": 127,
    "min_data_in_bin": 30,
    "verbosity": -1,
    "verbose": -1,
    "num_threads": max(1, min(12, os.cpu_count() or 1)),
    "seed": 91027,
    "feature_fraction_seed": 91028,
    "bagging_seed": 91029,
    "data_random_seed": 91030,
    "deterministic": True,
    "force_col_wise": True,
}

try:
    model_tree = lgb.train(
        linear_params,
        dtrain,
        num_boost_round=230,
    )
    tree_valid = model_tree.predict(Xva).astype(np.float64)
    tree_test = model_tree.predict(Xte).astype(np.float64)
    tree_kind = "piecewise_linear_boosting"
except Exception as exc:
    # Some CPU LightGBM builds can reject linear leaves. The fallback remains
    # a distinct rule-ensemble family and prevents losing the iteration.
    print("FINDINGS linear_tree_fallback=" + repr(str(exc)[:180]))
    fallback_params = dict(linear_params)
    fallback_params.pop("linear_tree", None)
    fallback_params.pop("linear_lambda", None)
    fallback_params.update({
        "boosting_type": "rf",
        "learning_rate": 1.0,
        "num_leaves": 31,
        "bagging_fraction": 0.70,
        "bagging_freq": 1,
        "feature_fraction": 0.65,
    })
    model_tree = lgb.train(
        fallback_params,
        dtrain,
        num_boost_round=140,
    )
    tree_valid = model_tree.predict(Xva).astype(np.float64)
    tree_test = model_tree.predict(Xte).astype(np.float64)
    tree_kind = "contextual_random_forest"

del Xtr, Xva, Xte, dtrain
gc.collect()


# Put every family on the same within-user scale before aggregation.
inc_vr = per_user_rank(valid.user_id, inc_valid)
inc_tr = per_user_rank(test.user_id, inc_test)

model_valid = {
    "contextual_additive_scorecard": per_user_rank(
        valid.user_id, gam_valid
    ),
    tree_kind: per_user_rank(valid.user_id, tree_valid),
}

model_test = {
    "contextual_additive_scorecard": per_user_rank(
        test.user_id, gam_test
    ),
    tree_kind: per_user_rank(test.user_id, tree_test),
}

# A structurally heterogeneous consensus is also tested.
model_valid["contextual_family_consensus"] = (
    model_valid["contextual_additive_scorecard"]
    + model_valid[tree_kind]
) / 2.0
model_test["contextual_family_consensus"] = (
    model_test["contextual_additive_scorecard"]
    + model_test[tree_kind]
) / 2.0

candidate_scores = {}
best_primary = -np.inf
best_valid = None
best_test = None
best_raw = None
best_name = None
best_metrics = None

inc_metrics = evaluate(valid.user_id, yv, inc_vr)
candidate_scores["incumbent"] = float(inc_metrics["primary"])

best_primary = float(inc_metrics["primary"])
best_valid = inc_vr.copy()
best_test = inc_tr.copy()
best_name = "incumbent"
best_metrics = inc_metrics

blend_weights = [0.08, 0.15, 0.25, 0.40, 0.60, 1.0]

for family_name in model_valid:
    raw_v = model_valid[family_name]
    raw_t = model_test[family_name]

    raw_metrics = evaluate(valid.user_id, yv, raw_v)
    candidate_scores[family_name] = float(raw_metrics["primary"])

    print(
        "FINDINGS "
        + family_name
        + "_raw="
        + json.dumps({
            "primary": float(raw_metrics["primary"]),
            "gauc": float(raw_metrics["gauc"]),
            "ndcg@5": float(raw_metrics["ndcg@5"]),
        }, sort_keys=True)
    )

    for alpha in blend_weights:
        if alpha == 1.0:
            blended_v = raw_v
            blended_t = raw_t
        else:
            blended_v = (1.0 - alpha) * inc_vr + alpha * raw_v
            blended_t = (1.0 - alpha) * inc_tr + alpha * raw_t

        metrics = evaluate(valid.user_id, yv, blended_v)
        name = family_name + "_blend_" + format(alpha, ".2f")
        candidate_scores[name] = float(metrics["primary"])

        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_valid = blended_v.copy()
            best_test = blended_t.copy()
            best_raw = raw_v.copy()
            best_name = name
            best_metrics = metrics

print("FINDINGS selected=" + best_name)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    if best_raw is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
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