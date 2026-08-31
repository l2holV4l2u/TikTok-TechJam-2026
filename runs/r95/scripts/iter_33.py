import os
import time
import json
import gc
import warnings

import numpy as np
import lightgbm as lgb
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")

START = time.time()
SEED = 20260831
THREADS = min(8, os.cpu_count() or 8)

np.random.seed(SEED)

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.int8)
yva = np.asarray(valid.y, dtype=np.int8)
uva = np.asarray(valid.user_id, dtype=np.int64)

ntr = len(ytr)
nva = len(yva)
nte = len(test.user_id)

CAT_FIELDS = [
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
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "music_type",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

LINEAR_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "user_active_degree",
    "onehot_feat3",
]

CROSS_PAIRS = [
    ("user_id", "tag"),
    ("user_id", "author_id"),
    ("user_id", "duration_bucket"),
    ("tab", "tag"),
    ("author_id", "tag"),
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def recency_weights(half_life):
    dates = np.asarray(train.date, dtype=np.int64)
    unique_dates = np.unique(dates)
    day_index = np.searchsorted(unique_dates, dates)
    age = (len(unique_dates) - 1 - day_index).astype(np.float32)
    weight = np.exp2(-age / float(half_life)).astype(np.float32)
    return weight / np.mean(weight)


weight_2 = recency_weights(2.0)
weight_4 = recency_weights(4.0)

print(
    "FINDINGS recency_weight_ranges="
    "hl2:%.4f/%.4f hl4:%.4f/%.4f"
    % (
        float(weight_2.min()),
        float(weight_2.max()),
        float(weight_4.min()),
        float(weight_4.max()),
    )
)


def history_matrix(split_name, expected_length):
    columns = []
    names = []

    for key in ("video_id", "author_id"):
        hist = historical_features(split_name, key=key)
        for name in sorted(hist):
            value = np.asarray(hist[name], dtype=np.float32)
            value = np.nan_to_num(
                value, nan=0.0, posinf=0.0, neginf=0.0
            )
            lower = name.lower()
            if (
                "count" in lower
                or "num" in lower
                or (len(value) and float(np.max(np.abs(value))) > 50.0)
            ):
                value = np.sign(value) * np.log1p(np.abs(value))
            columns.append(value)
            names.append(key + ":" + name)

    if not columns:
        return np.zeros((expected_length, 0), dtype=np.float32), names

    return np.column_stack(columns).astype(np.float32, copy=False), names


def numeric_matrix(sample, split_name, expected_length):
    columns = []
    for field in NUM_FIELDS:
        value = np.asarray(sample.num[field], dtype=np.float32)
        value = np.nan_to_num(
            value, nan=0.0, posinf=0.0, neginf=0.0
        )
        columns.append(np.log1p(np.maximum(value, 0.0)))

    raw = np.column_stack(columns).astype(np.float32, copy=False)
    hist, names = history_matrix(split_name, expected_length)
    return np.column_stack([raw, hist]).astype(np.float32, copy=False), names


num_tr, history_names = numeric_matrix(train, "train", ntr)
num_va, _ = numeric_matrix(valid, "valid", nva)
num_te, _ = numeric_matrix(test, "test", nte)

center = np.median(num_tr, axis=0).astype(np.float32)
q25 = np.quantile(num_tr, 0.25, axis=0).astype(np.float32)
q75 = np.quantile(num_tr, 0.75, axis=0).astype(np.float32)
scale = np.maximum(q75 - q25, 1.0e-3).astype(np.float32)

num_tr = np.clip((num_tr - center) / scale, -8.0, 8.0).astype(
    np.float32, copy=False
)
num_va = np.clip((num_va - center) / scale, -8.0, 8.0).astype(
    np.float32, copy=False
)
num_te = np.clip((num_te - center) / scale, -8.0, 8.0).astype(
    np.float32, copy=False
)

print("FINDINGS history_feature_count=%d" % len(history_names))

candidate_valid = {}
candidate_test = {}

# ----------------------------------------------------------------------
# Family 1: categorical gradient boosting, with sample weighting applied
# directly to the main predictor. Two half-lives test drift sensitivity.
# ----------------------------------------------------------------------
def lgb_matrix(sample, numeric):
    cats = np.column_stack([
        np.asarray(sample.X[field], dtype=np.float32)
        for field in CAT_FIELDS
    ])
    return np.column_stack([cats, numeric]).astype(np.float32, copy=False)


lgb_tr = lgb_matrix(train, num_tr)
lgb_va = lgb_matrix(valid, num_va)
lgb_te = lgb_matrix(test, num_te)

categorical_indices = list(range(len(CAT_FIELDS)))

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 180,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.15,
    "lambda_l2": 2.0,
    "max_cat_threshold": 32,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "max_bin": 127,
    "verbosity": -1,
    "verbose": -1,
    "num_threads": THREADS,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
}

for name, weight in (
    ("boost_recency_hl2", weight_2),
    ("boost_recency_hl4", weight_4),
):
    dtrain = lgb.Dataset(
        lgb_tr,
        label=ytr,
        weight=weight,
        categorical_feature=categorical_indices,
        free_raw_data=False,
    )
    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=260,
    )
    candidate_valid[name] = model.predict(
        lgb_va, num_iteration=model.best_iteration
    ).astype(np.float64)
    candidate_test[name] = model.predict(
        lgb_te, num_iteration=model.best_iteration
    ).astype(np.float64)
    del model, dtrain
    gc.collect()

del lgb_tr, lgb_va, lgb_te
gc.collect()

# ----------------------------------------------------------------------
# Family 2: randomized forest over leave-one-out categorical target
# statistics. This predicts through randomized high-order partitions of
# evidence strengths rather than additive logits or boosted residuals.
# ----------------------------------------------------------------------
global_rate = float(np.mean(ytr))
SMOOTH = 35.0


def target_stat_matrices():
    train_columns = []
    valid_columns = []
    test_columns = []

    for field in CAT_FIELDS:
        cardinality = int(FEATURE_CARDINALITIES[field])
        tr_id = np.asarray(train.X[field], dtype=np.int64)
        va_id = np.asarray(valid.X[field], dtype=np.int64)
        te_id = np.asarray(test.X[field], dtype=np.int64)

        count = np.bincount(tr_id, minlength=cardinality).astype(np.float32)
        positive = np.bincount(
            tr_id, weights=ytr.astype(np.float32), minlength=cardinality
        ).astype(np.float32)

        loo_count = np.maximum(count[tr_id] - 1.0, 0.0)
        loo_positive = positive[tr_id] - ytr.astype(np.float32)
        tr_rate = (
            loo_positive + SMOOTH * global_rate
        ) / (loo_count + SMOOTH)

        full_rate = (
            positive + SMOOTH * global_rate
        ) / (count + SMOOTH)

        va_safe = np.minimum(va_id, cardinality - 1)
        te_safe = np.minimum(te_id, cardinality - 1)

        train_columns.extend([
            tr_rate.astype(np.float32),
            np.log1p(loo_count).astype(np.float32),
        ])
        valid_columns.extend([
            full_rate[va_safe].astype(np.float32),
            np.log1p(count[va_safe]).astype(np.float32),
        ])
        test_columns.extend([
            full_rate[te_safe].astype(np.float32),
            np.log1p(count[te_safe]).astype(np.float32),
        ])

    return (
        np.column_stack(train_columns).astype(np.float32, copy=False),
        np.column_stack(valid_columns).astype(np.float32, copy=False),
        np.column_stack(test_columns).astype(np.float32, copy=False),
    )


te_tr, te_va, te_te = target_stat_matrices()
forest_tr = np.column_stack([te_tr, num_tr]).astype(np.float32, copy=False)
forest_va = np.column_stack([te_va, num_va]).astype(np.float32, copy=False)
forest_te = np.column_stack([te_te, num_te]).astype(np.float32, copy=False)

forest = ExtraTreesClassifier(
    n_estimators=160,
    criterion="entropy",
    max_depth=18,
    min_samples_leaf=45,
    max_features=0.70,
    bootstrap=False,
    class_weight=None,
    n_jobs=THREADS,
    random_state=SEED + 100,
)
forest.fit(forest_tr, ytr, sample_weight=weight_4)

candidate_valid["targetstat_extra_trees"] = forest.predict_proba(
    forest_va
)[:, 1].astype(np.float64)
candidate_test["targetstat_extra_trees"] = forest.predict_proba(
    forest_te
)[:, 1].astype(np.float64)

del forest, forest_tr, forest_va, forest_te
del te_tr, te_va, te_te
gc.collect()

# ----------------------------------------------------------------------
# Family 3: sparse linear logit over one-hot identities and explicitly
# hashed conjunctions. Crosses give memorized preference offsets, but the
# linear prediction form extrapolates differently from either tree family.
# ----------------------------------------------------------------------
HASH_DIM = 1 << 19

base_offsets = {}
base_total = 0
for field in LINEAR_FIELDS:
    base_offsets[field] = base_total
    base_total += int(FEATURE_CARDINALITIES[field])

total_sparse_dim = base_total + HASH_DIM


def hashed_sparse_matrix(sample, numeric):
    n = len(sample.user_id)
    feature_columns = []

    for field in LINEAR_FIELDS:
        ids = np.asarray(sample.X[field], dtype=np.int64)
        feature_columns.append(ids + base_offsets[field])

    for pair_index, (left, right) in enumerate(CROSS_PAIRS):
        a = np.asarray(sample.X[left], dtype=np.uint64)
        b = np.asarray(sample.X[right], dtype=np.uint64)
        mixed = (
            a * np.uint64(11995408973635179863)
            + b * np.uint64(10150724397891781847)
            + np.uint64((pair_index + 1) * 2654435761)
        )
        hashed = np.asarray(mixed % np.uint64(HASH_DIM), dtype=np.int64)
        feature_columns.append(base_total + hashed)

    cols = np.column_stack(feature_columns).astype(np.int32, copy=False)
    width = cols.shape[1]

    rows = np.repeat(np.arange(n, dtype=np.int32), width)
    cols_flat = cols.reshape(-1)
    data = np.ones(len(cols_flat), dtype=np.float32)

    categorical = sparse.csr_matrix(
        (data, (rows, cols_flat)),
        shape=(n, total_sparse_dim),
        dtype=np.float32,
    )
    numerical = sparse.csr_matrix(numeric, dtype=np.float32)
    return sparse.hstack(
        [categorical, numerical], format="csr", dtype=np.float32
    )


sp_tr = hashed_sparse_matrix(train, num_tr)
sp_va = hashed_sparse_matrix(valid, num_va)
sp_te = hashed_sparse_matrix(test, num_te)

linear = SGDClassifier(
    loss="log_loss",
    penalty="elasticnet",
    alpha=2.0e-6,
    l1_ratio=0.04,
    fit_intercept=True,
    max_iter=9,
    tol=None,
    shuffle=True,
    average=True,
    n_jobs=THREADS,
    random_state=SEED + 200,
)
linear.fit(sp_tr, ytr, sample_weight=weight_4)

candidate_valid["hashed_cross_linear"] = linear.decision_function(
    sp_va
).astype(np.float64)
candidate_test["hashed_cross_linear"] = linear.decision_function(
    sp_te
).astype(np.float64)

del linear, sp_tr, sp_va, sp_te
del num_tr, num_va, num_te
gc.collect()

# ----------------------------------------------------------------------
# Vectorized within-user rank normalization makes scales commensurate for
# blending without changing any standalone within-user ordering.
# ----------------------------------------------------------------------
def user_rank_score(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, users))
    sorted_users = users[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    within = np.arange(n, dtype=np.float64) - np.repeat(starts, lengths)
    denom = np.repeat(np.maximum(lengths - 1, 1), lengths).astype(np.float64)
    ranked = within / denom

    singleton = np.repeat(lengths == 1, lengths)
    ranked[singleton] = 0.5

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

if os.path.exists(inc_valid_path) and os.path.exists(inc_test_path):
    incumbent_valid = np.asarray(
        np.load(inc_valid_path), dtype=np.float64
    )
    incumbent_test = np.asarray(
        np.load(inc_test_path), dtype=np.float64
    )
else:
    incumbent_valid = None
    incumbent_test = None

valid_rank_cache = {
    name: user_rank_score(valid.user_id, score)
    for name, score in candidate_valid.items()
}
test_rank_cache = {
    name: user_rank_score(test.user_id, score)
    for name, score in candidate_test.items()
}

candidate_report = {}
best_primary = -np.inf
best_metrics = None
best_valid_scores = None
best_test_scores = None
best_raw_valid = None
best_name = None
best_alpha = 1.0

for name in candidate_valid:
    metrics = evaluate(uva, yva, candidate_valid[name])
    candidate_report[name] = float(metrics["primary"])

    if metrics["primary"] > best_primary:
        best_primary = float(metrics["primary"])
        best_metrics = metrics
        best_valid_scores = candidate_valid[name]
        best_test_scores = candidate_test[name]
        best_raw_valid = candidate_valid[name]
        best_name = name
        best_alpha = 1.0

if incumbent_valid is not None:
    incumbent_valid_rank = user_rank_score(valid.user_id, incumbent_valid)
    incumbent_test_rank = user_rank_score(test.user_id, incumbent_test)

    # Alpha is the contribution of the new family.
    blend_alphas = [0.08, 0.15, 0.25, 0.40, 0.60, 0.80]

    for name in candidate_valid:
        for alpha in blend_alphas:
            blended_valid = (
                alpha * valid_rank_cache[name]
                + (1.0 - alpha) * incumbent_valid_rank
            )
            metrics = evaluate(uva, yva, blended_valid)
            key = "%s_blend_%.2f" % (name, alpha)
            candidate_report[key] = float(metrics["primary"])

            if metrics["primary"] > best_primary:
                best_primary = float(metrics["primary"])
                best_metrics = metrics
                best_valid_scores = blended_valid
                best_test_scores = (
                    alpha * test_rank_cache[name]
                    + (1.0 - alpha) * incumbent_test_rank
                )
                best_raw_valid = candidate_valid[name]
                best_name = name
                best_alpha = float(alpha)

print(
    "FINDINGS selected_family=%s new_model_weight=%.2f"
    % (best_name, best_alpha)
)
print("CANDIDATES " + json.dumps(candidate_report, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)