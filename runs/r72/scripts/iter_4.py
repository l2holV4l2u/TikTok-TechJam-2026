import gc
import json
import os
import time

import lightgbm as lgb
import numpy as np

from pipeline.data import load
from pipeline.evaluate import evaluate


START = time.time()
ARTIFACTS = os.environ["RUN_ARTIFACTS"]
OUT_DIR = os.environ.get("ITER_OUT")

BASE_VALID_PATH = os.path.join(
    ARTIFACTS, "incumbent_valid_scores.npy"
)
BASE_TEST_PATH = os.path.join(
    ARTIFACTS, "incumbent_test_scores.npy"
)

TE_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "onehot_feat3",
    "onehot_feat8",
    "upload_type",
    "tab",
    "duration_bucket",
]
TE_SMOOTHING = 20.0


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    mean = float(np.mean(x))
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x)
    return (x - mean) / sd


def logits(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log1p(-p)


def weekday_from_dates(dates):
    dates = np.asarray(dates)
    unique, inverse = np.unique(dates, return_inverse=True)
    weekday = np.empty(len(unique), dtype=np.float32)
    for i, value in enumerate(unique):
        text = str(int(value))
        iso = text[:4] + "-" + text[4:6] + "-" + text[6:8]
        day_number = np.datetime64(iso, "D").astype(np.int64)
        # 1970-01-01 was Thursday; Monday is zero.
        weekday[i] = float((day_number + 3) % 7)
    return weekday[inverse]


def concatenate_split_arrays(splits, kind, name):
    arrays = []
    for split in splits:
        if kind == "X":
            arrays.append(np.asarray(split.X[name]))
        elif kind == "num":
            arrays.append(np.asarray(split.num[name]))
        elif kind == "date":
            arrays.append(np.asarray(split.date))
        else:
            raise ValueError(kind)
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays)


def aggregate_statistics(entities, labels):
    entities = np.asarray(entities, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    cardinality = int(np.max(entities)) + 1

    counts = np.bincount(
        entities, minlength=cardinality
    ).astype(np.float64)
    positives = np.bincount(
        entities, weights=labels, minlength=cardinality
    ).astype(np.float64)
    return counts, positives


def response_columns_self(entities, labels, global_rate):
    """Exactly leave the current row out of its entity statistic."""
    entities = np.asarray(entities, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    counts, positives = aggregate_statistics(entities, labels)

    loo_count = counts[entities] - 1.0
    loo_positive = positives[entities] - labels

    rate = (
        loo_positive + TE_SMOOTHING * global_rate
    ) / (loo_count + TE_SMOOTHING)

    return (
        np.log1p(np.maximum(loo_count, 0.0)).astype(np.float32),
        rate.astype(np.float32),
    )


def response_columns_target(
    source_entities,
    source_labels,
    target_entities,
    global_rate,
):
    source_entities = np.asarray(source_entities, dtype=np.int64)
    target_entities = np.asarray(target_entities, dtype=np.int64)
    counts, positives = aggregate_statistics(
        source_entities, source_labels
    )

    target_count = np.zeros(len(target_entities), dtype=np.float64)
    target_positive = np.zeros(len(target_entities), dtype=np.float64)

    known = target_entities < len(counts)
    target_count[known] = counts[target_entities[known]]
    target_positive[known] = positives[target_entities[known]]

    rate = (
        target_positive + TE_SMOOTHING * global_rate
    ) / (target_count + TE_SMOOTHING)

    return (
        np.log1p(target_count).astype(np.float32),
        rate.astype(np.float32),
    )


def transformed_numeric(values):
    values = np.asarray(values, dtype=np.float32)
    missing = ~np.isfinite(values)
    clean = np.where(missing, 0.0, values)
    clean = np.maximum(clean, 0.0)
    logged = np.log1p(clean).astype(np.float32)
    return logged, missing.astype(np.float32)


def build_self_matrix(splits, labels, categorical_fields, numeric_fields):
    labels = np.asarray(labels, dtype=np.float64)
    global_rate = float(np.mean(labels))
    columns = []

    for field in categorical_fields:
        values = concatenate_split_arrays(splits, "X", field)
        columns.append(values.astype(np.float32, copy=False))

    dates = concatenate_split_arrays(splits, "date", "")
    columns.append(weekday_from_dates(dates))

    for field in numeric_fields:
        values = concatenate_split_arrays(splits, "num", field)
        logged, missing = transformed_numeric(values)
        columns.extend([logged, missing])

    for field in TE_FIELDS:
        entities = concatenate_split_arrays(splits, "X", field)
        log_count, rate = response_columns_self(
            entities, labels, global_rate
        )
        columns.extend([log_count, rate])

    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    return matrix


def build_target_matrix(
    target,
    source_splits,
    source_labels,
    categorical_fields,
    numeric_fields,
):
    source_labels = np.asarray(source_labels, dtype=np.float64)
    global_rate = float(np.mean(source_labels))
    columns = []

    for field in categorical_fields:
        columns.append(
            np.asarray(target.X[field], dtype=np.float32)
        )

    columns.append(weekday_from_dates(target.date))

    for field in numeric_fields:
        logged, missing = transformed_numeric(target.num[field])
        columns.extend([logged, missing])

    for field in TE_FIELDS:
        source_entities = concatenate_split_arrays(
            source_splits, "X", field
        )
        target_entities = np.asarray(target.X[field], dtype=np.int64)
        log_count, rate = response_columns_target(
            source_entities,
            source_labels,
            target_entities,
            global_rate,
        )
        columns.extend([log_count, rate])

    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    return matrix


if not os.path.exists(BASE_VALID_PATH):
    raise FileNotFoundError(BASE_VALID_PATH)
if not os.path.exists(BASE_TEST_PATH):
    raise FileNotFoundError(BASE_TEST_PATH)

train = load("train")
valid = load("valid")

categorical_fields = sorted(train.X.keys())
numeric_fields = sorted(train.num.keys())
categorical_indices = list(range(len(categorical_fields) + 1))

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

base_valid = standardize(np.load(BASE_VALID_PATH))
base_metrics = evaluate(valid_users, y_valid, base_valid)

X_train = build_self_matrix(
    [train], y_train, categorical_fields, numeric_fields
)
X_valid = build_target_matrix(
    valid,
    [train],
    y_train,
    categorical_fields,
    numeric_fields,
)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 150,
    "min_sum_hessian_in_leaf": 1e-3,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 2.0,
    "max_cat_to_onehot": 16,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "max_bin": 127,
    "force_col_wise": True,
    "num_threads": max(1, os.cpu_count() or 1),
    "seed": 2026,
    "feature_fraction_seed": 2027,
    "bagging_seed": 2028,
    "data_random_seed": 2029,
    "verbose": -1,
}

dtrain = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
dvalid = lgb.Dataset(
    X_valid,
    label=y_valid,
    categorical_feature=categorical_indices,
    reference=dtrain,
    free_raw_data=False,
)

model = lgb.train(
    params,
    dtrain,
    num_boost_round=600,
    valid_sets=[dvalid],
    valid_names=["valid"],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

best_iteration = int(model.best_iteration)
if best_iteration <= 0:
    best_iteration = 600

valid_probability = model.predict(
    X_valid, num_iteration=best_iteration
)
tree_valid = standardize(logits(valid_probability))

candidate_metrics = {}
candidate_predictions = {}

tree_metrics = evaluate(valid_users, y_valid, tree_valid)
candidate_metrics["tree_only"] = float(tree_metrics["primary"])
candidate_predictions["tree_only"] = (tree_valid, tree_metrics)

candidate_metrics["incumbent"] = float(base_metrics["primary"])
candidate_predictions["incumbent"] = (base_valid, base_metrics)

# Alpha is the tree-model weight. The same selected alpha is used on test.
blend_alphas = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]
for alpha in blend_alphas:
    scores = (
        (1.0 - float(alpha)) * base_valid
        + float(alpha) * tree_valid
    )
    metrics = evaluate(valid_users, y_valid, scores)
    name = f"blend_tree_{alpha:.2f}"
    candidate_metrics[name] = float(metrics["primary"])
    candidate_predictions[name] = (scores, metrics)

best_name = max(
    candidate_metrics, key=lambda name: candidate_metrics[name]
)
valid_scores, best_metrics = candidate_predictions[best_name]

if best_name.startswith("blend_tree_"):
    selected_alpha = float(best_name.rsplit("_", 1)[1])
elif best_name == "tree_only":
    selected_alpha = 1.0
else:
    selected_alpha = 0.0

valid_scores = np.asarray(valid_scores, dtype=np.float64)

if OUT_DIR:
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(
        os.path.join(OUT_DIR, "scores_valid.npy"),
        valid_scores,
    )

print(
    "FINDINGS "
    + json.dumps(
        {
            "best_iteration": best_iteration,
            "n_features": int(X_train.shape[1]),
            "selected": best_name,
            "selected_tree_alpha": selected_alpha,
            "incumbent_primary": float(base_metrics["primary"]),
            "tree_primary": float(tree_metrics["primary"]),
        },
        sort_keys=True,
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        {
            name: round(float(score), 6)
            for name, score in sorted(
                candidate_metrics.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        },
        sort_keys=True,
    )
)

# Release the validation-fit model and matrices before the required refit.
del model, dtrain, dvalid, X_train, X_valid
gc.collect()

base_test = standardize(np.load(BASE_TEST_PATH))

if selected_alpha <= 0.0:
    test_scores = base_test
else:
    # The hidden split is used only as a feature-only prediction target.
    test = load("test")

    combined_labels = np.concatenate(
        [
            y_train.astype(np.float32, copy=False),
            y_valid.astype(np.float32, copy=False),
        ]
    )

    X_combined = build_self_matrix(
        [train, valid],
        combined_labels,
        categorical_fields,
        numeric_fields,
    )
    X_test = build_target_matrix(
        test,
        [train, valid],
        combined_labels,
        categorical_fields,
        numeric_fields,
    )

    dcombined = lgb.Dataset(
        X_combined,
        label=combined_labels,
        categorical_feature=categorical_indices,
        free_raw_data=False,
    )

    final_model = lgb.train(
        params,
        dcombined,
        num_boost_round=best_iteration,
    )

    test_probability = final_model.predict(
        X_test, num_iteration=best_iteration
    )
    tree_test = standardize(logits(test_probability))

    test_scores = (
        (1.0 - selected_alpha) * base_test
        + selected_alpha * tree_test
    )

if OUT_DIR:
    np.save(
        os.path.join(OUT_DIR, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result))