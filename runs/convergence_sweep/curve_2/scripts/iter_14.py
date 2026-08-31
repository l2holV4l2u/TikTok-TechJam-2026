import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2026
np.random.seed(SEED)

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

# ---------------------------------------------------------------------
# Train-only recency weighting
# ---------------------------------------------------------------------

train_dates = np.asarray(train.date, dtype=np.int32)
last_train_date = int(np.max(train_dates))
train_day = (train_dates % 100).astype(np.float32)
last_day = float(last_train_date % 100)
train_age = last_day - train_day
recency_weight = np.exp(
    -np.log(2.0) * train_age / 5.0
).astype(np.float32)
recency_weight /= max(float(np.mean(recency_weight)), 1e-6)

# ---------------------------------------------------------------------
# Family 1: DART interaction ensemble over relatively stationary fields,
# robust numeric transforms, and train-only entity histories.
# ---------------------------------------------------------------------

DART_CATS = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "music_type",
    "upload_type",
    "video_type",
    "hour",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat11",
    "onehot_feat12",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def fit_numeric_transform(split):
    matrix = []
    centers = {}
    scales = {}
    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        finite = np.isfinite(x)
        clean = np.where(finite, np.maximum(x, 0.0), 0.0)
        z = np.log1p(clean).astype(np.float32)
        vals = z[finite]
        if len(vals):
            center = float(np.median(vals))
            q25, q75 = np.percentile(vals, [25.0, 75.0])
            scale = max(float(q75 - q25), 0.25)
        else:
            center, scale = 0.0, 1.0
        centers[name] = center
        scales[name] = scale
        matrix.append(
            np.clip((z - center) / scale, -8.0, 8.0).astype(np.float32)
        )
        matrix.append((~finite).astype(np.float32))
    return np.column_stack(matrix).astype(np.float32), centers, scales


def apply_numeric_transform(split, centers, scales):
    matrix = []
    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        finite = np.isfinite(x)
        clean = np.where(finite, np.maximum(x, 0.0), 0.0)
        z = np.log1p(clean).astype(np.float32)
        matrix.append(
            np.clip(
                (z - centers[name]) / scales[name], -8.0, 8.0
            ).astype(np.float32)
        )
        matrix.append((~finite).astype(np.float32))
    return np.column_stack(matrix).astype(np.float32)


tr_num, num_centers, num_scales = fit_numeric_transform(train)
va_num = apply_numeric_transform(valid, num_centers, num_scales)
te_num = apply_numeric_transform(test, num_centers, num_scales)


def history_matrix(split_name):
    columns = []
    names = []
    for key in ("video_id", "author_id"):
        histories = historical_features(split_name, key=key)
        for name in sorted(histories):
            x = np.asarray(histories[name], dtype=np.float32)
            x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
            columns.append(x)
            names.append(name)
    return np.column_stack(columns).astype(np.float32), names


tr_hist, history_names = history_matrix("train")
va_hist, _ = history_matrix("valid")
te_hist, _ = history_matrix("test")


def dart_matrix(split, numeric, histories):
    cats = np.column_stack([
        np.asarray(split.X[name], dtype=np.float32)
        for name in DART_CATS
    ]).astype(np.float32)
    return np.column_stack([cats, numeric, histories]).astype(np.float32)


Xtr = dart_matrix(train, tr_num, tr_hist)
Xva = dart_matrix(valid, va_num, va_hist)
Xte = dart_matrix(test, te_num, te_hist)

dart_data = lgb.Dataset(
    Xtr,
    label=y_train,
    weight=recency_weight,
    categorical_feature=list(range(len(DART_CATS))),
    free_raw_data=False,
)

dart_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "dart",
    "learning_rate": 0.045,
    "num_leaves": 63,
    "max_depth": 10,
    "min_data_in_leaf": 180,
    "feature_fraction": 0.78,
    "bagging_fraction": 0.80,
    "bagging_freq": 1,
    "drop_rate": 0.08,
    "skip_drop": 0.55,
    "max_drop": 30,
    "lambda_l1": 0.2,
    "lambda_l2": 8.0,
    "max_bin": 127,
    "max_cat_threshold": 32,
    "cat_smooth": 30.0,
    "cat_l2": 12.0,
    "num_threads": max(1, min(12, os.cpu_count() or 1)),
    "seed": SEED,
    "drop_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "feature_fraction_seed": SEED + 3,
    "verbose": -1,
}

dart_model = lgb.train(
    dart_params,
    dart_data,
    num_boost_round=190,
)

dart_valid = dart_model.predict(Xva).astype(np.float32)
dart_test = dart_model.predict(Xte).astype(np.float32)

del dart_model, dart_data, Xtr, Xva, Xte
gc.collect()

# ---------------------------------------------------------------------
# Family 2: stationary additive categorical GAM.
#
# Every field contributes a recency-weighted, smoothed marginal log-odds.
# This deliberately excludes user identity and therefore cannot overfit
# user response propensity, which is irrelevant within a user.
# ---------------------------------------------------------------------

GAM_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "music_type",
    "upload_type",
    "video_type",
    "hour",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "is_live_streamer",
    "is_video_author",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat4",
    "onehot_feat6",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat9",
    "onehot_feat10",
    "onehot_feat11",
    "onehot_feat12",
]

global_rate = float(
    np.sum(recency_weight * y_train) / np.sum(recency_weight)
)
global_rate = np.clip(global_rate, 1e-5, 1.0 - 1e-5)
global_logit = float(np.log(global_rate / (1.0 - global_rate)))


def field_smoothing(name):
    if name == "video_id":
        return 35.0
    if name == "author_id":
        return 45.0
    if name in ("onehot_feat3", "onehot_feat8"):
        return 60.0
    return 100.0


gam_tables = {}
for name in GAM_FIELDS:
    ids = np.asarray(train.X[name], dtype=np.int64)
    size = int(max(
        int(np.max(ids)) + 1,
        int(np.max(np.asarray(valid.X[name], dtype=np.int64))) + 1,
        int(np.max(np.asarray(test.X[name], dtype=np.int64))) + 1,
    ))
    count = np.bincount(
        ids, weights=recency_weight, minlength=size
    ).astype(np.float64)
    positive = np.bincount(
        ids, weights=recency_weight * y_train, minlength=size
    ).astype(np.float64)
    smoothing = field_smoothing(name)
    rate = (
        positive + smoothing * global_rate
    ) / np.maximum(count + smoothing, 1e-12)
    rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
    contribution = np.log(rate / (1.0 - rate)) - global_logit

    # Reliability shrinkage prevents rare identities from dominating the sum.
    reliability = count / (count + smoothing)
    gam_tables[name] = (
        contribution * np.sqrt(reliability)
    ).astype(np.float32)


def predict_gam(split):
    score = np.full(len(split.user_id), global_logit, dtype=np.float32)
    for name in GAM_FIELDS:
        ids = np.asarray(split.X[name], dtype=np.int64)
        table = gam_tables[name]
        safe_ids = np.minimum(ids, len(table) - 1)
        score += table[safe_ids] / np.sqrt(float(len(GAM_FIELDS)))
    return score


gam_valid = predict_gam(valid)
gam_test = predict_gam(test)

# ---------------------------------------------------------------------
# Family 3: non-parametric personalized user-content memory.
#
# For each user and content field, store a recency-weighted posterior for
# values encountered in train. Prediction is the sum of deviations from
# each value's global posterior. All construction uses train only.
# ---------------------------------------------------------------------

MEMORY_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "music_type",
    "upload_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

train_users = np.asarray(train.user_id, dtype=np.int64)


def build_pair_table(field):
    values = np.asarray(train.X[field], dtype=np.int64)
    base = int(np.max(values)) + 1
    keys = train_users.astype(np.int64) * np.int64(base) + values

    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.r_[0, 1 + np.flatnonzero(
        sorted_keys[1:] != sorted_keys[:-1]
    )]

    unique_keys = sorted_keys[starts]
    counts = np.add.reduceat(recency_weight[order], starts).astype(np.float64)
    positives = np.add.reduceat(
        (recency_weight * y_train)[order], starts
    ).astype(np.float64)

    value_count = np.bincount(
        values, weights=recency_weight, minlength=base
    ).astype(np.float64)
    value_positive = np.bincount(
        values,
        weights=recency_weight * y_train,
        minlength=base,
    ).astype(np.float64)

    value_smooth = 40.0
    value_rate = (
        value_positive + value_smooth * global_rate
    ) / np.maximum(value_count + value_smooth, 1e-12)

    pair_values = (unique_keys % np.int64(base)).astype(np.int64)
    prior = value_rate[pair_values]

    pair_smooth = 5.0 if field in ("video_id", "author_id") else 8.0
    pair_rate = (
        positives + pair_smooth * prior
    ) / np.maximum(counts + pair_smooth, 1e-12)
    pair_rate = np.clip(pair_rate, 1e-5, 1.0 - 1e-5)
    prior = np.clip(prior, 1e-5, 1.0 - 1e-5)

    residual = (
        np.log(pair_rate / (1.0 - pair_rate))
        - np.log(prior / (1.0 - prior))
    )
    reliability = counts / (counts + pair_smooth)
    residual *= np.sqrt(reliability)

    return (
        base,
        unique_keys.astype(np.int64),
        residual.astype(np.float32),
    )


memory_tables = {
    field: build_pair_table(field)
    for field in MEMORY_FIELDS
}


def lookup_pairs(users, values, base, keys, residual):
    query = users.astype(np.int64) * np.int64(base) + values.astype(np.int64)
    positions = np.searchsorted(keys, query)
    in_range = positions < len(keys)
    safe = np.minimum(positions, len(keys) - 1)
    found = in_range & (keys[safe] == query)
    result = np.zeros(len(query), dtype=np.float32)
    result[found] = residual[safe[found]]
    return result


def predict_memory(split):
    users = np.asarray(split.user_id, dtype=np.int64)
    score = np.zeros(len(users), dtype=np.float32)
    for field in MEMORY_FIELDS:
        base, keys, residual = memory_tables[field]
        values = np.asarray(split.X[field], dtype=np.int64)
        score += lookup_pairs(users, values, base, keys, residual)
    score /= np.sqrt(float(len(MEMORY_FIELDS)))
    return score


memory_valid = predict_memory(valid)
memory_test = predict_memory(test)

# ---------------------------------------------------------------------
# Compare standalone families and every family's incumbent blend.
# ---------------------------------------------------------------------

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float32,
)
inc_test = np.asarray(
    np.load(os.path.join(shared, "incumbent_test_scores.npy")),
    dtype=np.float32,
)


def standardize_pair(valid_score, test_score):
    center = float(np.mean(valid_score))
    scale = max(float(np.std(valid_score)), 1e-6)
    return (
        ((valid_score - center) / scale).astype(np.float32),
        ((test_score - center) / scale).astype(np.float32),
    )


inc_valid_z, inc_test_z = standardize_pair(inc_valid, inc_test)

families = {
    "dart_interaction": (dart_valid, dart_test),
    "stationary_additive_gam": (gam_valid, gam_test),
    "user_content_memory": (memory_valid, memory_test),
}

candidate_log = {}

inc_metric = evaluate(valid.user_id, y_valid, inc_valid)
candidate_log["incumbent"] = float(inc_metric["primary"])

best_metric = inc_metric
best_valid = inc_valid.copy()
best_test = inc_test.copy()
best_name = "incumbent"

best_raw_metric_value = -np.inf
best_raw_valid = dart_valid.copy()
best_raw_name = "dart_interaction"

blend_alphas = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70]

for family_name, (raw_valid, raw_test) in families.items():
    raw_metric = evaluate(valid.user_id, y_valid, raw_valid)
    raw_primary = float(raw_metric["primary"])
    candidate_log[family_name] = raw_primary

    if raw_primary > best_raw_metric_value:
        best_raw_metric_value = raw_primary
        best_raw_valid = raw_valid.copy()
        best_raw_name = family_name

    if raw_primary > float(best_metric["primary"]):
        best_metric = raw_metric
        best_valid = raw_valid.copy()
        best_test = raw_test.copy()
        best_name = family_name

    raw_valid_z, raw_test_z = standardize_pair(raw_valid, raw_test)

    for alpha in blend_alphas:
        blended_valid = (
            (1.0 - alpha) * inc_valid_z + alpha * raw_valid_z
        ).astype(np.float32)
        blended_test = (
            (1.0 - alpha) * inc_test_z + alpha * raw_test_z
        ).astype(np.float32)

        metric = evaluate(valid.user_id, y_valid, blended_valid)
        name = "%s_blend_%.2f" % (family_name, alpha)
        candidate_log[name] = float(metric["primary"])

        if float(metric["primary"]) > float(best_metric["primary"]):
            best_metric = metric
            best_valid = blended_valid.copy()
            best_test = blended_test.copy()
            best_name = name

print(
    "FINDINGS best_raw_family=%s best_raw_primary=%.6f selected=%s"
    % (best_raw_name, best_raw_metric_value, best_name)
)
print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.6f}'
    % (
        float(best_metric["primary"]),
        float(best_metric["gauc"]),
        float(best_metric["ndcg@5"]),
        elapsed,
    )
)