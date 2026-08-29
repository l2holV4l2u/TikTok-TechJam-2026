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
THREADS = max(1, min(8, os.cpu_count() or 1))

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
global_rate = float(y_train.mean())

train_features = []
valid_features = []
test_features = []
feature_names = []
categorical_indices = []


def add_feature(name, tr, va, te, categorical=False):
    tr = np.asarray(tr, dtype=np.float32)
    va = np.asarray(va, dtype=np.float32)
    te = np.asarray(te, dtype=np.float32)

    train_features.append(tr)
    valid_features.append(va)
    test_features.append(te)
    feature_names.append(name)

    if categorical:
        categorical_indices.append(len(feature_names) - 1)


def sanitize_numeric(tr, va, te):
    tr = np.asarray(tr, dtype=np.float32)
    va = np.asarray(va, dtype=np.float32)
    te = np.asarray(te, dtype=np.float32)

    finite = np.isfinite(tr)
    median = float(np.median(tr[finite])) if np.any(finite) else 0.0

    tr = np.nan_to_num(tr, nan=median, posinf=median, neginf=median)
    va = np.nan_to_num(va, nan=median, posinf=median, neginf=median)
    te = np.nan_to_num(te, nan=median, posinf=median, neginf=median)
    return tr, va, te


# Low- and medium-cardinality raw categorical context. User, video and
# author identities are represented through their target statistics below.
raw_categorical_fields = [
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "video_type",
    "hour",
    "user_active_degree",
    "register_days_range",
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
    "onehot_feat5",
    "onehot_feat6",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat9",
    "onehot_feat10",
    "onehot_feat11",
    "onehot_feat12",
    "onehot_feat13",
    "onehot_feat14",
    "onehot_feat15",
    "onehot_feat16",
    "onehot_feat17",
]

for field in raw_categorical_fields:
    add_feature(
        field,
        train.X[field],
        valid.X[field],
        test.X[field],
        categorical=True,
    )


# Continuous quantities are heavy-tailed, so expose log1p values and
# missingness flags rather than their raw scales alone.
for field in [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]:
    tr_raw = np.asarray(train.num[field], dtype=np.float32)
    va_raw = np.asarray(valid.num[field], dtype=np.float32)
    te_raw = np.asarray(test.num[field], dtype=np.float32)

    add_feature(
        field + "_missing",
        ~np.isfinite(tr_raw),
        ~np.isfinite(va_raw),
        ~np.isfinite(te_raw),
    )

    tr_num, va_num, te_num = sanitize_numeric(tr_raw, va_raw, te_raw)
    tr_num = np.maximum(tr_num, 0.0)
    va_num = np.maximum(va_num, 0.0)
    te_num = np.maximum(te_num, 0.0)

    add_feature(
        field + "_log1p",
        np.log1p(tr_num),
        np.log1p(va_num),
        np.log1p(te_num),
    )


def combined_key(split, fields):
    key = np.asarray(split.X[fields[0]], dtype=np.int64).copy()
    for field in fields[1:]:
        cardinality = int(
            max(
                np.asarray(train.X[field]).max(),
                np.asarray(valid.X[field]).max(),
                np.asarray(test.X[field]).max(),
            )
            + 1
        )
        key = key * cardinality + np.asarray(split.X[field], dtype=np.int64)
    return key


def add_bayesian_encoding(name, tr_key, va_key, te_key, strength):
    """
    Train values are exact leave-one-out statistics. Validation and test
    values use all train labels. Unknown entities receive the global prior.
    """
    tr_key = np.asarray(tr_key, dtype=np.int64)
    va_key = np.asarray(va_key, dtype=np.int64)
    te_key = np.asarray(te_key, dtype=np.int64)

    unique_keys, inverse, counts = np.unique(
        tr_key, return_inverse=True, return_counts=True
    )
    positive_sums = np.bincount(
        inverse, weights=y_train, minlength=len(unique_keys)
    ).astype(np.float64)

    loo_count = counts[inverse].astype(np.float64) - 1.0
    loo_sum = positive_sums[inverse] - y_train.astype(np.float64)
    tr_rate = (
        loo_sum + strength * global_rate
    ) / (loo_count + strength)

    def map_eval(keys):
        positions = np.searchsorted(unique_keys, keys)
        matched = positions < len(unique_keys)
        safe_positions = np.minimum(positions, len(unique_keys) - 1)
        matched &= unique_keys[safe_positions] == keys

        rate = np.full(len(keys), global_rate, dtype=np.float64)
        count = np.zeros(len(keys), dtype=np.float64)

        if np.any(matched):
            p = positions[matched]
            c = counts[p].astype(np.float64)
            s = positive_sums[p]
            rate[matched] = (
                s + strength * global_rate
            ) / (c + strength)
            count[matched] = c
        return rate.astype(np.float32), np.log1p(count).astype(np.float32)

    va_rate, va_count = map_eval(va_key)
    te_rate, te_count = map_eval(te_key)

    add_feature(
        name + "_rate",
        tr_rate.astype(np.float32),
        va_rate,
        te_rate,
    )
    add_feature(
        name + "_log_count",
        np.log1p(loo_count).astype(np.float32),
        va_count,
        te_count,
    )

    return {
        "unique_train_keys": int(len(unique_keys)),
        "valid_known_fraction": float(
            np.mean(
                (np.searchsorted(unique_keys, va_key) < len(unique_keys))
            )
        ),
    }


encoding_findings = {}

# Stable entity propensities.
for name, fields, strength in [
    ("user", ["user_id"], 25.0),
    ("video", ["video_id"], 40.0),
    ("author", ["author_id"], 40.0),
    ("tag", ["tag"], 80.0),
    ("duration_bucket", ["duration_bucket"], 100.0),
    ("upload_type", ["upload_type"], 100.0),
]:
    encoding_findings[name] = add_bayesian_encoding(
        name,
        combined_key(train, fields),
        combined_key(valid, fields),
        combined_key(test, fields),
        strength,
    )


# Personalized affinities vary within a user and therefore can directly
# change impression ranking for that user.
for name, fields, strength in [
    ("user_tag", ["user_id", "tag"], 12.0),
    ("user_duration", ["user_id", "duration_bucket"], 12.0),
    ("user_tab", ["user_id", "tab"], 12.0),
    ("user_upload", ["user_id", "upload_type"], 15.0),
    ("user_author", ["user_id", "author_id"], 10.0),
    ("user_video", ["user_id", "video_id"], 8.0),
]:
    encoding_findings[name] = add_bayesian_encoding(
        name,
        combined_key(train, fields),
        combined_key(valid, fields),
        combined_key(test, fields),
        strength,
    )


# Add the organizer-provided train-only historical statistics. Train rows
# are leave-one-out by API contract; validation/test use the full train set.
for entity in ["video_id", "author_id"]:
    hist_train = historical_features("train", key=entity)
    hist_valid = historical_features("valid", key=entity)
    hist_test = historical_features("test", key=entity)

    common_names = sorted(
        set(hist_train.keys())
        & set(hist_valid.keys())
        & set(hist_test.keys())
    )
    for hist_name in common_names:
        tr_hist, va_hist, te_hist = sanitize_numeric(
            hist_train[hist_name],
            hist_valid[hist_name],
            hist_test[hist_name],
        )
        add_feature(
            "history_" + entity + "_" + hist_name,
            tr_hist,
            va_hist,
            te_hist,
        )


X_train = np.ascontiguousarray(
    np.column_stack(train_features), dtype=np.float32
)
X_valid = np.ascontiguousarray(
    np.column_stack(valid_features), dtype=np.float32
)
X_test = np.ascontiguousarray(
    np.column_stack(test_features), dtype=np.float32
)

del train_features, valid_features, test_features
gc.collect()

dtrain = lgb.Dataset(
    X_train,
    label=y_train,
    feature_name=feature_names,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
dvalid = lgb.Dataset(
    X_valid,
    label=y_valid,
    feature_name=feature_names,
    categorical_feature=categorical_indices,
    reference=dtrain,
    free_raw_data=False,
)

params = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "learning_rate": 0.04,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 250,
    "min_sum_hessian_in_leaf": 1e-3,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 3.0,
    "max_bin": 127,
    "min_data_per_group": 150,
    "cat_smooth": 20.0,
    "cat_l2": 15.0,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "data_random_seed": SEED + 3,
    "num_threads": THREADS,
    "verbose": -1,
}

model = lgb.train(
    params,
    dtrain,
    num_boost_round=700,
    valid_sets=[dvalid],
    valid_names=["valid"],
    callbacks=[
        lgb.early_stopping(60, first_metric_only=True, verbose=False),
    ],
)

lgb_valid = model.predict(
    X_valid,
    num_iteration=model.best_iteration,
    raw_score=True,
).astype(np.float64)

lgb_test = model.predict(
    X_test,
    num_iteration=model.best_iteration,
    raw_score=True,
).astype(np.float64)

lgb_metrics = evaluate(valid.user_id, y_valid, lgb_valid)


def standardize(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0
    return (values - mean) / std


def within_user_rank(user_ids, scores):
    """
    Convert scores to [0,1] ranks independently within each user's logged
    impressions. This is fully vectorized and preserves the evaluated task.
    """
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]

    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = sorted_users[1:] != sorted_users[:-1]

    positions = np.arange(n, dtype=np.int64)
    starts = np.maximum.accumulate(
        np.where(new_group, positions, 0)
    )
    ordinal = positions - starts

    group_starts = positions[new_group]
    group_ends = np.r_[group_starts[1:], n]
    group_sizes = group_ends - group_starts
    sizes_per_row = np.repeat(group_sizes, group_sizes)

    normalized = np.full(n, 0.5, dtype=np.float64)
    nonsingle = sizes_per_row > 1
    normalized[nonsingle] = (
        ordinal[nonsingle] / (sizes_per_row[nonsingle] - 1.0)
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = normalized
    return result


artifacts = os.environ.get("RUN_ARTIFACTS", "")
inc_valid_path = os.path.join(
    artifacts, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    artifacts, "incumbent_test_scores.npy"
)

candidate_report = {
    "lgb_target_statistics": float(lgb_metrics["primary"])
}
valid_scores = lgb_valid
test_scores = lgb_test
final_metrics = lgb_metrics
selected = {
    "mode": "lgb_only",
    "alpha_lgb": 1.0,
}

if os.path.exists(inc_valid_path) and os.path.exists(inc_test_path):
    incumbent_valid = np.asarray(
        np.load(inc_valid_path), dtype=np.float64
    )
    incumbent_test = np.asarray(
        np.load(inc_test_path), dtype=np.float64
    )

    if (
        incumbent_valid.shape == lgb_valid.shape
        and incumbent_test.shape == lgb_test.shape
    ):
        incumbent_metrics = evaluate(
            valid.user_id, y_valid, incumbent_valid
        )
        candidate_report["incumbent"] = float(
            incumbent_metrics["primary"]
        )

        valid_representations = {
            "raw": (incumbent_valid, lgb_valid),
            "zscore": (
                standardize(incumbent_valid),
                standardize(lgb_valid),
            ),
            "user_rank": (
                within_user_rank(valid.user_id, incumbent_valid),
                within_user_rank(valid.user_id, lgb_valid),
            ),
        }

        test_representations = {
            "raw": (incumbent_test, lgb_test),
            "zscore": (
                standardize(incumbent_test),
                standardize(lgb_test),
            ),
            "user_rank": (
                within_user_rank(test.user_id, incumbent_test),
                within_user_rank(test.user_id, lgb_test),
            ),
        }

        # Include finer weights near the likely useful residual-blend range.
        blend_weights = np.unique(
            np.r_[
                np.linspace(0.0, 1.0, 11),
                np.array([0.05, 0.15, 0.25, 0.35, 0.45]),
            ]
        )

        for mode, (inc_rep, lgb_rep) in valid_representations.items():
            for alpha in blend_weights:
                blended = (
                    (1.0 - float(alpha)) * inc_rep
                    + float(alpha) * lgb_rep
                )
                metrics = evaluate(
                    valid.user_id, y_valid, blended
                )
                name = "{}_lgb_{:.2f}".format(mode, float(alpha))
                candidate_report[name] = float(metrics["primary"])

                if float(metrics["primary"]) > float(
                    final_metrics["primary"]
                ):
                    valid_scores = blended.copy()
                    final_metrics = metrics
                    selected = {
                        "mode": mode,
                        "alpha_lgb": float(alpha),
                    }

        if selected["mode"] != "lgb_only":
            inc_te_rep, lgb_te_rep = test_representations[
                selected["mode"]
            ]
            alpha = selected["alpha_lgb"]
            test_scores = (
                (1.0 - alpha) * inc_te_rep + alpha * lgb_te_rep
            )

print(
    "CANDIDATES "
    + json.dumps(candidate_report, sort_keys=True)
)
print(
    "FINDINGS "
    + json.dumps(
        {
            "best_iteration": int(model.best_iteration),
            "n_features": int(X_train.shape[1]),
            "selected_blend": selected,
            "lgb_primary": float(lgb_metrics["primary"]),
            "encodings": encoding_findings,
        },
        sort_keys=True,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
report = {
    "primary": float(final_metrics["primary"]),
    "gauc": float(final_metrics["gauc"]),
    "ndcg@5": float(final_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(report))