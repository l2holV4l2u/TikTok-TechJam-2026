import os
import time
import json
import math
import random
import gc

import numpy as np
import lightgbm as lgb
from sklearn.linear_model import SGDClassifier

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 91837

random.seed(SEED)
np.random.seed(SEED)

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

ALL_FIELDS = sorted(train.X.keys())

STABLE_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "video_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "hour",
    "user_active_degree",
    "register_days_bucket",
]

TARGET_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "video_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "hour",
    "user_active_degree",
    "register_days_bucket",
]

NUMERIC_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

max_train_date = int(np.max(np.asarray(train.date, dtype=np.int32)))
train_age = (
    max_train_date - np.asarray(train.date, dtype=np.int32)
).astype(np.float32)

# Six-day half-life emphasizes the part of train closest to evaluation.
recency_weight = np.exp(
    -math.log(2.0) * train_age / 6.0
).astype(np.float32)
recency_weight /= np.mean(recency_weight)

print(
    "FINDINGS recency_weight="
    + json.dumps(
        {
            "min": float(recency_weight.min()),
            "max": float(recency_weight.max()),
            "last_day_mean": float(
                recency_weight[
                    np.asarray(train.date, dtype=np.int32) == max_train_date
                ].mean()
            ),
        }
    )
)


def load_histories():
    histories = {}
    for entity in ("video_id", "author_id"):
        histories[(entity, "train")] = historical_features(
            "train", key=entity
        )
        histories[(entity, "valid")] = historical_features(
            "valid", key=entity
        )
        histories[(entity, "test")] = historical_features(
            "test", key=entity
        )

    common_names = []
    for entity in ("video_id", "author_id"):
        tr_names = set(histories[(entity, "train")].keys())
        va_names = set(histories[(entity, "valid")].keys())
        te_names = set(histories[(entity, "test")].keys())
        for name in sorted(tr_names & va_names & te_names):
            common_names.append((entity, name))

    return histories, common_names


histories, HISTORY_NAMES = load_histories()

print(
    "FINDINGS historical_features="
    + json.dumps([name for _, name in HISTORY_NAMES])
)


def transformed_numeric(split, name):
    values = np.asarray(split.num[name], dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.log1p(np.maximum(values, 0.0)).astype(np.float32)


def make_gbdt_matrix(split, split_name, categorical_fields):
    columns = []

    for name in categorical_fields:
        columns.append(np.asarray(split.X[name], dtype=np.float32))

    for name in NUMERIC_FIELDS:
        columns.append(transformed_numeric(split, name))

    for entity, feature_name in HISTORY_NAMES:
        values = np.asarray(
            histories[(entity, split_name)][feature_name],
            dtype=np.float32,
        )
        values = np.nan_to_num(
            values, nan=0.0, posinf=0.0, neginf=0.0
        )
        columns.append(values)

    return np.ascontiguousarray(
        np.column_stack(columns), dtype=np.float32
    )


def train_gbdt_family(name, categorical_fields, num_rounds):
    x_tr = make_gbdt_matrix(train, "train", categorical_fields)
    x_va = make_gbdt_matrix(valid, "valid", categorical_fields)
    x_te = make_gbdt_matrix(test, "test", categorical_fields)

    dataset = lgb.Dataset(
        x_tr,
        label=y_train,
        weight=recency_weight,
        categorical_feature=list(range(len(categorical_fields))),
        free_raw_data=True,
    )

    params = {
        "objective": "binary",
        "metric": "None",
        "learning_rate": 0.045,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 500,
        "min_sum_hessian_in_leaf": 5.0,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.86,
        "bagging_freq": 1,
        "lambda_l1": 0.15,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "min_data_per_group": 200,
        "cat_smooth": 20.0,
        "cat_l2": 15.0,
        "max_cat_to_onehot": 16,
        "num_threads": min(8, os.cpu_count() or 1),
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "data_random_seed": SEED + 3,
        "deterministic": True,
        "force_col_wise": True,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        dataset,
        num_boost_round=num_rounds,
    )

    pred_va = model.predict(
        x_va, num_iteration=model.current_iteration()
    ).astype(np.float64)
    pred_te = model.predict(
        x_te, num_iteration=model.current_iteration()
    ).astype(np.float64)

    importance = model.feature_importance(
        importance_type="gain"
    ).astype(np.float64)
    top = np.argsort(-importance)[:8]
    feature_names = (
        list(categorical_fields)
        + NUMERIC_FIELDS
        + [name for _, name in HISTORY_NAMES]
    )
    finding = [
        [feature_names[int(i)], float(importance[int(i)])]
        for i in top
    ]
    print(
        "FINDINGS "
        + name
        + "_top_gain="
        + json.dumps(finding)
    )

    del model, dataset, x_tr, x_va, x_te
    gc.collect()
    return pred_va, pred_te


# Family 1a: a stationarity-oriented GBDT using compact categorical context
# plus continuous and train-only historical item features.
gbdt_stable_valid, gbdt_stable_test = train_gbdt_family(
    "gbdt_stable_history",
    STABLE_FIELDS,
    340,
)

# Family 1b: broad GBDT tests whether additional categorical context adds
# useful interactions despite its greater susceptibility to drift.
gbdt_broad_valid, gbdt_broad_test = train_gbdt_family(
    "gbdt_broad_history",
    ALL_FIELDS,
    300,
)


def weighted_target_features(field_names):
    n_tr = len(y_train)
    n_va = len(valid.user_id)
    n_te = len(test.user_id)

    tr_features = np.empty(
        (n_tr, 2 * len(field_names)), dtype=np.float32
    )
    va_features = np.empty(
        (n_va, 2 * len(field_names)), dtype=np.float32
    )
    te_features = np.empty(
        (n_te, 2 * len(field_names)), dtype=np.float32
    )

    weighted_global = float(
        np.sum(recency_weight * y_train)
        / np.sum(recency_weight)
    )
    smoothing = 30.0

    bayes_valid_logits = []
    bayes_test_logits = []

    for j, field in enumerate(field_names):
        tr_ids = np.asarray(train.X[field], dtype=np.int64)
        va_ids = np.asarray(valid.X[field], dtype=np.int64)
        te_ids = np.asarray(test.X[field], dtype=np.int64)

        cardinality = int(
            max(
                tr_ids.max(initial=0),
                va_ids.max(initial=0),
                te_ids.max(initial=0),
            )
            + 1
        )

        weighted_count = np.bincount(
            tr_ids,
            weights=recency_weight,
            minlength=cardinality,
        ).astype(np.float64)
        weighted_sum = np.bincount(
            tr_ids,
            weights=recency_weight * y_train,
            minlength=cardinality,
        ).astype(np.float64)

        # Leave-one-out encodings for training prevent same-row target leakage.
        loo_count = weighted_count[tr_ids] - recency_weight
        loo_sum = (
            weighted_sum[tr_ids] - recency_weight * y_train
        )
        loo_rate = (
            loo_sum + smoothing * weighted_global
        ) / (loo_count + smoothing)

        full_rate = (
            weighted_sum + smoothing * weighted_global
        ) / (weighted_count + smoothing)

        va_rate = full_rate[va_ids]
        te_rate = full_rate[te_ids]

        tr_features[:, 2 * j] = loo_rate.astype(np.float32)
        tr_features[:, 2 * j + 1] = np.log1p(
            np.maximum(loo_count, 0.0)
        ).astype(np.float32)

        va_features[:, 2 * j] = va_rate.astype(np.float32)
        va_features[:, 2 * j + 1] = np.log1p(
            weighted_count[va_ids]
        ).astype(np.float32)

        te_features[:, 2 * j] = te_rate.astype(np.float32)
        te_features[:, 2 * j + 1] = np.log1p(
            weighted_count[te_ids]
        ).astype(np.float32)

        if field in {
            "video_id",
            "author_id",
            "tag",
            "tab",
            "duration_bucket",
        }:
            eps = 1e-5
            bayes_valid_logits.append(
                np.log(np.clip(va_rate, eps, 1.0 - eps))
                - np.log1p(-np.clip(va_rate, eps, 1.0 - eps))
            )
            bayes_test_logits.append(
                np.log(np.clip(te_rate, eps, 1.0 - eps))
                - np.log1p(-np.clip(te_rate, eps, 1.0 - eps))
            )

    bayes_va = np.mean(
        np.column_stack(bayes_valid_logits), axis=1
    ).astype(np.float64)
    bayes_te = np.mean(
        np.column_stack(bayes_test_logits), axis=1
    ).astype(np.float64)

    return tr_features, va_features, te_features, bayes_va, bayes_te


(
    te_train,
    te_valid,
    te_test,
    bayes_valid,
    bayes_test,
) = weighted_target_features(TARGET_FIELDS)

# Standardization uses train only.
te_mean = te_train.mean(axis=0, dtype=np.float64)
te_std = te_train.std(axis=0, dtype=np.float64)
te_std = np.where(te_std > 1e-6, te_std, 1.0)

te_train = ((te_train - te_mean) / te_std).astype(np.float32)
te_valid = ((te_valid - te_mean) / te_std).astype(np.float32)
te_test = ((te_test - te_mean) / te_std).astype(np.float32)

# Family 2: an additive generalized linear model over leave-one-out,
# recency-weighted target encodings. It cannot form tree interactions and is
# therefore structurally distinct from the GBDT.
target_glm = SGDClassifier(
    loss="log_loss",
    penalty="elasticnet",
    alpha=2e-5,
    l1_ratio=0.05,
    fit_intercept=True,
    max_iter=18,
    tol=1e-4,
    shuffle=True,
    random_state=SEED,
    average=True,
)
target_glm.fit(
    te_train,
    y_train.astype(np.int8),
    sample_weight=recency_weight,
)
target_glm_valid = target_glm.decision_function(
    te_valid
).astype(np.float64)
target_glm_test = target_glm.decision_function(
    te_test
).astype(np.float64)

del target_glm, te_train, te_valid, te_test
gc.collect()

# Family 3 is the direct hierarchical-Bayes score constructed above. It
# averages independently smoothed entity logits without fitting a combiner.

families = {
    "gbdt_stable_history": (
        gbdt_stable_valid,
        gbdt_stable_test,
    ),
    "gbdt_broad_history": (
        gbdt_broad_valid,
        gbdt_broad_test,
    ),
    "target_encoding_glm": (
        target_glm_valid,
        target_glm_test,
    ),
    "hierarchical_bayes": (
        bayes_valid,
        bayes_test,
    ),
}


def normalization_parameters(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(std) or std < 1e-10:
        std = 1.0
    return mean, std


inc_mean, inc_std = normalization_parameters(inc_valid)
inc_valid_z = (inc_valid - inc_mean) / inc_std
inc_test_z = (inc_test - inc_mean) / inc_std

candidate_scores = {}
candidate_payloads = {}

for family_name, (pred_valid, pred_test) in families.items():
    own_metrics = evaluate(
        valid.user_id, y_valid, pred_valid
    )
    candidate_scores[family_name] = float(
        own_metrics["primary"]
    )
    candidate_payloads[family_name] = (
        pred_valid,
        pred_test,
        pred_valid,
    )

    model_mean, model_std = normalization_parameters(pred_valid)
    model_valid_z = (pred_valid - model_mean) / model_std
    model_test_z = (pred_test - model_mean) / model_std

    # Alpha is the new family's weight; all choices use validation only and
    # the identical fixed transformation is then applied to test.
    for alpha in (0.25, 0.50, 0.75):
        blend_valid = (
            alpha * model_valid_z
            + (1.0 - alpha) * inc_valid_z
        )
        blend_test = (
            alpha * model_test_z
            + (1.0 - alpha) * inc_test_z
        )
        blend_name = family_name + "_blend_" + str(alpha)
        blend_metrics = evaluate(
            valid.user_id, y_valid, blend_valid
        )
        candidate_scores[blend_name] = float(
            blend_metrics["primary"]
        )
        candidate_payloads[blend_name] = (
            blend_valid,
            blend_test,
            pred_valid,
        )

winner_name = max(
    candidate_scores, key=lambda name: candidate_scores[name]
)
valid_scores, test_scores, raw_valid_scores = candidate_payloads[
    winner_name
]
metrics = evaluate(valid.user_id, y_valid, valid_scores)

print(
    "FINDINGS winner="
    + json.dumps(
        {
            "name": winner_name,
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
        }
    )
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if "_blend_" in winner_name:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_valid_scores, dtype=np.float64),
        )

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