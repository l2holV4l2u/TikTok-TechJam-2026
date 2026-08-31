import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
np.random.seed(20260831)

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.float32)
yva = np.asarray(valid.y, dtype=np.int8)
uva = np.asarray(valid.user_id, dtype=np.int64)
ute = np.asarray(test.user_id, dtype=np.int64)

# These fields retain useful item/context information while avoiding a direct
# user identity parameter whose distribution is particularly unstable.
FIELDS = [
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
    "video_type",
    "is_live_streamer",
    "is_video_author",
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

dates = np.asarray(train.date, dtype=np.int64)
unique_dates = np.unique(dates)
date_to_index = {int(d): i for i, d in enumerate(unique_dates)}
day_index = np.fromiter(
    (date_to_index[int(d)] for d in dates),
    dtype=np.int16,
    count=len(dates),
)
age = (len(unique_dates) - 1 - day_index).astype(np.float32)

recent_weight = np.exp2(-age / 4.0).astype(np.float32)
recent_weight /= recent_weight.mean()

late_weight = (age <= 4).astype(np.float32)
late_weight /= max(float(late_weight.mean()), 1.0e-6)

print(
    "FINDINGS train_days=%d recent_effective_weight_last5=%.4f"
    % (
        len(unique_dates),
        float(recent_weight[age <= 4].sum() / recent_weight.sum()),
    )
)


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
    sorted_ranks = positions / denominators

    result = np.empty(len(scores), dtype=np.float64)
    result[order] = sorted_ranks
    return result


# -------------------------------------------------------------------------
# Family 1: generative categorical likelihood ratios.
#
# Each field contributes log P(value | positive) / P(value | negative).
# Unlike a discriminative embedding model, this explicitly pools evidence
# through smoothed class-conditional counts. A late-window version and a
# train-wide version permit train-only temporal extrapolation.
# -------------------------------------------------------------------------
NB_BASE_FIELDS = [
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
    "music_type",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

NB_INTERACTIONS = [
    ("tab", "tag"),
    ("duration_bucket", "tag"),
    ("author_id", "tab"),
    ("user_active_degree", "duration_bucket"),
    ("upload_type", "duration_bucket"),
]


def nb_arrays(sample):
    arrays = []
    cardinalities = []

    for field in NB_BASE_FIELDS:
        arrays.append(np.asarray(sample.X[field], dtype=np.int64))
        cardinalities.append(int(FEATURE_CARDINALITIES[field]))

    for left, right in NB_INTERACTIONS:
        a = np.asarray(sample.X[left], dtype=np.int64)
        b = np.asarray(sample.X[right], dtype=np.int64)
        kb = int(FEATURE_CARDINALITIES[right])
        arrays.append(a * kb + b)
        cardinalities.append(
            int(FEATURE_CARDINALITIES[left]) * kb
        )

    return arrays, cardinalities


tr_nb_arrays, nb_cards = nb_arrays(train)
va_nb_arrays, _ = nb_arrays(valid)
te_nb_arrays, _ = nb_arrays(test)


def fit_nb_tables(row_weight, smoothing=25.0):
    positive_weight = row_weight * ytr
    negative_weight = row_weight * (1.0 - ytr)
    total_positive = float(positive_weight.sum())
    total_negative = float(negative_weight.sum())

    tables = []
    for values, cardinality in zip(tr_nb_arrays, nb_cards):
        c1 = np.bincount(
            values,
            weights=positive_weight,
            minlength=cardinality,
        ).astype(np.float64)
        c0 = np.bincount(
            values,
            weights=negative_weight,
            minlength=cardinality,
        ).astype(np.float64)

        # Symmetric Dirichlet shrinkage. The denominator terms are retained
        # because cardinalities differ between ordinary and crossed fields.
        p1 = (c1 + smoothing) / (
            total_positive + smoothing * cardinality
        )
        p0 = (c0 + smoothing) / (
            total_negative + smoothing * cardinality
        )
        tables.append(np.log(p1) - np.log(p0))
    return tables


def predict_nb(arrays, tables):
    scores = np.zeros(len(arrays[0]), dtype=np.float64)
    for values, table in zip(arrays, tables):
        scores += table[values]
    return scores


nb_uniform_tables = fit_nb_tables(
    np.ones(len(ytr), dtype=np.float32),
    smoothing=30.0,
)
nb_recent_tables = fit_nb_tables(recent_weight, smoothing=30.0)
nb_late_tables = fit_nb_tables(late_weight, smoothing=40.0)

nb_uniform_valid = predict_nb(va_nb_arrays, nb_uniform_tables)
nb_uniform_test = predict_nb(te_nb_arrays, nb_uniform_tables)
nb_recent_valid = predict_nb(va_nb_arrays, nb_recent_tables)
nb_recent_test = predict_nb(te_nb_arrays, nb_recent_tables)
nb_late_valid = predict_nb(va_nb_arrays, nb_late_tables)
nb_late_test = predict_nb(te_nb_arrays, nb_late_tables)

# Conservative train-only trend projection: move from the all-date estimate
# halfway toward the estimate formed from the final five train days.
nb_trend_valid = nb_recent_valid + 0.5 * (
    nb_late_valid - nb_uniform_valid
)
nb_trend_test = nb_recent_test + 0.5 * (
    nb_late_test - nb_uniform_test
)

del nb_uniform_tables, nb_recent_tables, nb_late_tables
del tr_nb_arrays, va_nb_arrays, te_nb_arrays
gc.collect()


# -------------------------------------------------------------------------
# Shared matrix for two structurally different tree families.
# Numeric transformations and their normalization are computed on train only.
# -------------------------------------------------------------------------
num_location = {}
num_scale = {}

for field in NUM_FIELDS:
    raw = np.asarray(train.num[field], dtype=np.float64)
    transformed = np.log1p(np.maximum(np.nan_to_num(raw, nan=0.0), 0.0))
    median = float(np.median(transformed))
    q25, q75 = np.quantile(transformed, [0.25, 0.75])
    scale = max(float(q75 - q25), 1.0e-3)
    num_location[field] = median
    num_scale[field] = scale


def make_matrix(sample):
    columns = [
        np.asarray(sample.X[field], dtype=np.float32)
        for field in FIELDS
    ]
    for field in NUM_FIELDS:
        raw = np.asarray(sample.num[field], dtype=np.float64)
        z = np.log1p(np.maximum(np.nan_to_num(raw, nan=0.0), 0.0))
        z = (z - num_location[field]) / num_scale[field]
        z = np.clip(z, -8.0, 8.0).astype(np.float32)
        columns.append(z)
    return np.column_stack(columns).astype(np.float32, copy=False)


xtr = make_matrix(train)
xva = make_matrix(valid)
xte = make_matrix(test)
categorical_indices = list(range(len(FIELDS)))

dtrain_recent = lgb.Dataset(
    xtr,
    label=ytr,
    weight=recent_weight,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)

common = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "min_data_in_leaf": 1000,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "max_cat_threshold": 32,
    "cat_smooth": 20.0,
    "verbosity": -1,
    "num_threads": min(8, os.cpu_count() or 8),
    "seed": 2026,
    "feature_fraction_seed": 2027,
    "bagging_seed": 2028,
}


# -------------------------------------------------------------------------
# Family 2: boosted GAM. Depth-one trees form an additive nonlinear model,
# deliberately excluding high-order partitions that can overfit temporal
# identity correlations.
# -------------------------------------------------------------------------
gam_params = dict(common)
gam_params.update({
    "boosting_type": "gbdt",
    "num_leaves": 2,
    "max_depth": 1,
    "learning_rate": 0.055,
    "min_gain_to_split": 0.0,
    "feature_fraction": 1.0,
})

gam = lgb.train(
    gam_params,
    dtrain_recent,
    num_boost_round=280,
)
gam_valid = gam.predict(xva).astype(np.float64)
gam_test = gam.predict(xte).astype(np.float64)
del gam
gc.collect()


# -------------------------------------------------------------------------
# Family 3: random forest. Independent bootstrapped categorical partitions
# reduce the variance and temporal brittleness of sequential boosting.
# -------------------------------------------------------------------------
rf_params = dict(common)
rf_params.update({
    "boosting_type": "rf",
    "num_leaves": 31,
    "max_depth": 8,
    "learning_rate": 1.0,
    "bagging_fraction": 0.70,
    "bagging_freq": 1,
    "feature_fraction": 0.72,
    "min_data_in_leaf": 700,
})

rf = lgb.train(
    rf_params,
    dtrain_recent,
    num_boost_round=180,
)
rf_valid = rf.predict(xva).astype(np.float64)
rf_test = rf.predict(xte).astype(np.float64)
del rf, dtrain_recent, xtr, xva, xte
gc.collect()


# Ranking-normalize every family because only within-user order matters and
# raw score scales differ substantially across generative, GAM, and RF models.
families_valid = {
    "generative_nb_uniform": within_user_rank(uva, nb_uniform_valid),
    "generative_nb_recent": within_user_rank(uva, nb_recent_valid),
    "generative_nb_temporal_projection": within_user_rank(
        uva, nb_trend_valid
    ),
    "boosted_gam": within_user_rank(uva, gam_valid),
    "bagged_random_forest": within_user_rank(uva, rf_valid),
}
families_test = {
    "generative_nb_uniform": within_user_rank(ute, nb_uniform_test),
    "generative_nb_recent": within_user_rank(ute, nb_recent_test),
    "generative_nb_temporal_projection": within_user_rank(
        ute, nb_trend_test
    ),
    "boosted_gam": within_user_rank(ute, gam_test),
    "bagged_random_forest": within_user_rank(ute, rf_test),
}

# A cross-family ensemble tests whether the three prediction mechanisms cover
# distinct errors before they are blended with the incumbent.
families_valid["nb_gam_rf_ensemble"] = (
    families_valid["generative_nb_temporal_projection"]
    + families_valid["boosted_gam"]
    + families_valid["bagged_random_forest"]
) / 3.0
families_test["nb_gam_rf_ensemble"] = (
    families_test["generative_nb_temporal_projection"]
    + families_test["boosted_gam"]
    + families_test["bagged_random_forest"]
) / 3.0

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.asarray(
    np.load(os.path.join(shared_dir, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(shared_dir, "incumbent_test_scores.npy")),
    dtype=np.float64,
)
inc_rank_valid = within_user_rank(uva, inc_valid)
inc_rank_test = within_user_rank(ute, inc_test)

# Validation choice is restricted to the explicitly permitted trusted-
# incumbent blend. The identical selected family and alpha are used on test.
alphas = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]

candidate_scores = {}
best_primary = -np.inf
best_metrics = None
best_valid = None
best_test = None
best_raw = None
best_name = None

for name, own_valid in families_valid.items():
    own_test = families_test[name]

    standalone = evaluate(uva, yva, own_valid)
    candidate_scores[name + "_standalone"] = float(
        standalone["primary"]
    )

    correlation = float(np.corrcoef(inc_rank_valid, own_valid)[0, 1])
    print(
        "FINDINGS family=%s incumbent_rank_correlation=%.6f"
        % (name, correlation)
    )

    for alpha in alphas:
        blend_valid = (
            (1.0 - alpha) * inc_rank_valid + alpha * own_valid
        )
        blend_test = (
            (1.0 - alpha) * inc_rank_test + alpha * own_test
        )
        metrics = evaluate(uva, yva, blend_valid)
        primary = float(metrics["primary"])
        candidate_name = "%s_blend_%.2f" % (name, alpha)
        candidate_scores[candidate_name] = primary

        if primary > best_primary:
            best_primary = primary
            best_metrics = metrics
            best_valid = blend_valid.copy()
            best_test = blend_test.copy()
            best_raw = own_valid.copy()
            best_name = candidate_name

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS selected=%s primary=%.6f"
    % (best_name, best_primary)
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw, dtype=np.float64),
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