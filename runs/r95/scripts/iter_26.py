import os
import time
import json
import gc
import numpy as np
import scipy.sparse as sp
import lightgbm as lgb

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

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

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

# Stable side information is emphasized over user identity. Video and author
# identity remain because they are fully covered across the date boundary and
# provide strong within-user candidate discrimination.
FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "music_type",
    "video_type",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

# Recency weighting is fixed from the train calendar only. A four-day half-life
# targets the evaluation regime rather than the much larger early-train regime.
dates = np.asarray(train.date, dtype=np.int64)
unique_dates = np.unique(dates)
day_index = np.searchsorted(unique_dates, dates)
age = (len(unique_dates) - 1 - day_index).astype(np.float32)
weights = np.exp2(-age / 4.0).astype(np.float32)
weights /= weights.mean()

# Train-only quantile discretization makes heavy-tailed numeric variables usable
# by both sparse families without imposing a linear raw-scale relationship.
bin_edges = {}
for name in NUM_FIELDS:
    x = np.asarray(train.num[name], dtype=np.float64)
    finite = np.isfinite(x)
    transformed = np.log1p(np.maximum(x[finite], 0.0))
    edges = np.unique(
        np.quantile(transformed, np.linspace(0.0, 1.0, 33)[1:-1])
    )
    bin_edges[name] = edges


def numeric_bins(sample, name):
    x = np.asarray(sample.num[name], dtype=np.float64)
    finite = np.isfinite(x)
    z = np.zeros(len(x), dtype=np.float64)
    z[finite] = np.log1p(np.maximum(x[finite], 0.0))
    # Bin zero is reserved for missing values.
    result = np.zeros(len(x), dtype=np.int32)
    result[finite] = (
        np.searchsorted(bin_edges[name], z[finite], side="right") + 1
    )
    return result


# Construct a shared one-hot representation for maximum-entropy and
# class-conditional generative models.
offsets = {}
running = 0
for field in FIELDS:
    offsets[("cat", field)] = running
    running += int(FEATURE_CARDINALITIES[field])

for name in NUM_FIELDS:
    offsets[("num", name)] = running
    running += len(bin_edges[name]) + 2

# A few bounded crosses expose stationary context interactions to the linear
# and generative models without introducing high-dimensional user crosses.
CROSSES = [
    ("tab", "tag"),
    ("duration_bucket", "tag"),
    ("upload_type", "duration_bucket"),
    ("music_type", "video_type"),
]
for left, right in CROSSES:
    offsets[("cross", left, right)] = running
    running += (
        int(FEATURE_CARDINALITIES[left])
        * int(FEATURE_CARDINALITIES[right])
    )

TOTAL_DIM = running


def sparse_matrix(sample):
    n = len(sample.user_id)
    blocks = []
    row = np.arange(n, dtype=np.int32)

    for field in FIELDS:
        ids = np.asarray(sample.X[field], dtype=np.int64)
        cols = offsets[("cat", field)] + ids
        blocks.append(
            sp.csr_matrix(
                (
                    np.ones(n, dtype=np.float32),
                    (row, cols),
                ),
                shape=(n, TOTAL_DIM),
                dtype=np.float32,
            )
        )

    for name in NUM_FIELDS:
        ids = numeric_bins(sample, name).astype(np.int64)
        cols = offsets[("num", name)] + ids
        blocks.append(
            sp.csr_matrix(
                (
                    np.ones(n, dtype=np.float32),
                    (row, cols),
                ),
                shape=(n, TOTAL_DIM),
                dtype=np.float32,
            )
        )

    for left, right in CROSSES:
        rc = int(FEATURE_CARDINALITIES[right])
        ids = (
            np.asarray(sample.X[left], dtype=np.int64) * rc
            + np.asarray(sample.X[right], dtype=np.int64)
        )
        cols = offsets[("cross", left, right)] + ids
        blocks.append(
            sp.csr_matrix(
                (
                    np.ones(n, dtype=np.float32),
                    (row, cols),
                ),
                shape=(n, TOTAL_DIM),
                dtype=np.float32,
            )
        )

    result = sum(blocks[1:], blocks[0]).tocsr()
    result.sum_duplicates()
    return result


xtr_sparse = sparse_matrix(train)
xva_sparse = sparse_matrix(valid)
xte_sparse = sparse_matrix(test)

print(
    "FINDINGS sparse_shape={} nnz={} dimension={}".format(
        xtr_sparse.shape, xtr_sparse.nnz, TOTAL_DIM
    )
)

# -------------------------------------------------------------------------
# Family 1: maximum-entropy wide model.
#
# This estimates one globally regularized discriminative utility over sparse
# categories and bounded crosses. Unlike FM/deep models it does not rely on
# latent interaction geometry, and unlike trees it does not partition rows.
# -------------------------------------------------------------------------
logistic = SGDClassifier(
    loss="log_loss",
    penalty="elasticnet",
    alpha=2.5e-6,
    l1_ratio=0.03,
    fit_intercept=True,
    max_iter=7,
    tol=None,
    shuffle=True,
    random_state=SEED,
    learning_rate="optimal",
    average=True,
)
logistic.fit(xtr_sparse, ytr, sample_weight=weights)

logit_valid = logistic.decision_function(xva_sparse).astype(np.float64)
logit_test = logistic.decision_function(xte_sparse).astype(np.float64)

print("FINDINGS completed_family=maximum_entropy_wide")

# -------------------------------------------------------------------------
# Family 2: complement Naive Bayes.
#
# Formation is class-conditional evidence accumulation rather than direct
# conditional-risk fitting. Complement statistics are less dominated by
# frequent negative-side categories and can remain stable under prevalence
# drift because ranking depends on relative evidence.
# -------------------------------------------------------------------------
nb = ComplementNB(alpha=18.0, norm=True)
nb.fit(xtr_sparse, ytr, sample_weight=weights)

nb_valid = nb.predict_log_proba(xva_sparse)[:, 1].astype(np.float64)
nb_test = nb.predict_log_proba(xte_sparse)[:, 1].astype(np.float64)

print("FINDINGS completed_family=complement_naive_bayes")

# Sparse matrices are no longer needed.
del xtr_sparse, xva_sparse, xte_sparse
gc.collect()

# -------------------------------------------------------------------------
# Family 3: randomized bagged categorical trees.
#
# Each prediction is an average across independently subsampled trees. This
# differs from sequential residual boosting: averaging reduces sensitivity to
# unstable temporal/category partitions and supplies nonlinear interactions.
# -------------------------------------------------------------------------
def tree_matrix(sample):
    cats = [
        np.asarray(sample.X[f], dtype=np.float32) for f in FIELDS
    ]
    nums = []
    for name in NUM_FIELDS:
        x = np.asarray(sample.num[name], dtype=np.float64)
        finite = np.isfinite(x)
        z = np.zeros(len(x), dtype=np.float32)
        z[finite] = np.log1p(
            np.maximum(x[finite], 0.0)
        ).astype(np.float32)
        nums.append(z)
    return np.column_stack(cats + nums).astype(np.float32, copy=False)


xtr_tree = tree_matrix(train)
xva_tree = tree_matrix(valid)
xte_tree = tree_matrix(test)

rf_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "rf",
    "learning_rate": 1.0,
    "num_leaves": 63,
    "max_depth": 10,
    "min_data_in_leaf": 700,
    "lambda_l2": 8.0,
    "feature_fraction": 0.72,
    "bagging_fraction": 0.68,
    "bagging_freq": 1,
    "max_bin": 127,
    "max_cat_threshold": 32,
    "cat_smooth": 35.0,
    "verbose": -1,
    "num_threads": THREADS,
    "seed": SEED + 100,
    "feature_fraction_seed": SEED + 101,
    "bagging_seed": SEED + 102,
}

rf_data = lgb.Dataset(
    xtr_tree,
    label=ytr,
    weight=weights,
    categorical_feature=list(range(len(FIELDS))),
    free_raw_data=True,
)
rf_model = lgb.train(rf_params, rf_data, num_boost_round=180)

rf_valid = rf_model.predict(xva_tree).astype(np.float64)
rf_test = rf_model.predict(xte_tree).astype(np.float64)

print("FINDINGS completed_family=random_forest_bagging")

del xtr_tree, xva_tree, xte_tree, rf_data
gc.collect()


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    rows = np.arange(len(scores), dtype=np.int64)

    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], len(order)]
    lengths = ends - starts

    positions = (
        np.arange(len(order), dtype=np.float64)
        - np.repeat(starts, lengths)
    )
    denominators = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    ranked = positions / denominators

    result = np.empty(len(scores), dtype=np.float64)
    result[order] = ranked
    return result


# Rank aggregation places structurally different raw scales on the metric's
# natural within-user scale. Validation selects blend weights as explicitly
# permitted for the reusable trusted incumbent; the same weight is applied to
# test without reading test labels.
inc_va_rank = within_user_rank(valid.user_id, inc_valid)
inc_te_rank = within_user_rank(test.user_id, inc_test)

families = {
    "maximum_entropy": (logit_valid, logit_test),
    "complement_nb": (nb_valid, nb_test),
    "bagged_forest": (rf_valid, rf_test),
}

alphas = [0.10, 0.20, 0.35, 0.50, 0.70]
candidate_scores = {}
best_name = "trusted_incumbent"
best_metric = evaluate(uva, yva, inc_va_rank)
best_primary = float(best_metric["primary"])
best_valid = inc_va_rank
best_test = inc_te_rank
best_raw_valid = logit_valid
best_alpha = 0.0

candidate_scores["trusted_incumbent"] = best_primary

for family_name, (raw_valid, raw_test) in families.items():
    raw_metric = evaluate(uva, yva, raw_valid)
    candidate_scores[family_name + "_raw"] = float(raw_metric["primary"])

    va_rank = within_user_rank(valid.user_id, raw_valid)
    te_rank = within_user_rank(test.user_id, raw_test)

    for alpha in alphas:
        blended_valid = (
            (1.0 - alpha) * inc_va_rank + alpha * va_rank
        )
        metric = evaluate(uva, yva, blended_valid)
        primary = float(metric["primary"])
        name = "{}_blend_{:.2f}".format(family_name, alpha)
        candidate_scores[name] = primary

        if primary > best_primary:
            best_primary = primary
            best_name = name
            best_metric = metric
            best_valid = blended_valid
            best_test = (
                (1.0 - alpha) * inc_te_rank + alpha * te_rank
            )
            best_raw_valid = raw_valid
            best_alpha = alpha

# Also test equal-weight aggregation of all three new inductive biases before
# blending with the incumbent.
ensemble_va = np.mean(
    [
        within_user_rank(valid.user_id, values[0])
        for values in families.values()
    ],
    axis=0,
)
ensemble_te = np.mean(
    [
        within_user_rank(test.user_id, values[1])
        for values in families.values()
    ],
    axis=0,
)

raw_ensemble_metric = evaluate(uva, yva, ensemble_va)
candidate_scores["three_family_rank_ensemble_raw"] = float(
    raw_ensemble_metric["primary"]
)

for alpha in alphas:
    blended_valid = (
        (1.0 - alpha) * inc_va_rank + alpha * ensemble_va
    )
    metric = evaluate(uva, yva, blended_valid)
    primary = float(metric["primary"])
    name = "three_family_blend_{:.2f}".format(alpha)
    candidate_scores[name] = primary

    if primary > best_primary:
        best_primary = primary
        best_name = name
        best_metric = metric
        best_valid = blended_valid
        best_test = (
            (1.0 - alpha) * inc_te_rank + alpha * ensemble_te
        )
        best_raw_valid = ensemble_va
        best_alpha = alpha

print(
    "FINDINGS selected={} alpha={:.2f}".format(
        best_name, best_alpha
    )
)
print(
    "CANDIDATES " + json.dumps(
        {k: round(v, 7) for k, v in candidate_scores.items()},
        sort_keys=True,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    # The selected score is either an incumbent blend or the incumbent itself;
    # retain a representative own-family score for attribution.
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.3f}'
    % (
        float(best_metric["primary"]),
        float(best_metric["gauc"]),
        float(best_metric["ndcg@5"]),
        elapsed,
    )
)