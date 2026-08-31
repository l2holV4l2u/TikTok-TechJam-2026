import os
import time
import json
import gc
import numpy as np
import scipy.sparse as sp
import lightgbm as lgb

from sklearn.linear_model import SGDClassifier

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 93751
THREADS = min(16, os.cpu_count() or 1)
HALF_LIFE = 4.0
NUM_BOOST_ROUND = 260
LEAF_STRIDE = 3
BATCH_SIZE = 24000

CAT_FIELDS = [
    "author_id",
    "duration_bucket",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "hour",
    "is_live_streamer",
    "is_lowactive_period",
    "is_video_author",
    "music_type",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat10",
    "onehot_feat11",
    "onehot_feat12",
    "onehot_feat13",
    "onehot_feat14",
    "onehot_feat15",
    "onehot_feat16",
    "onehot_feat17",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat4",
    "onehot_feat5",
    "onehot_feat6",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat9",
    "register_days_bucket",
    "register_days_range",
    "tab",
    "tag",
    "upload_type",
    "user_active_degree",
    "user_id",
    "video_id",
    "video_type",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int32)
    age = dates.max() - dates
    weights = np.power(
        0.5,
        age.astype(np.float32) / HALF_LIFE,
    )
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    start_positions = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(np.concatenate((
        np.asarray([-1], dtype=np.int64),
        end_positions,
    )))
    row_sizes = np.repeat(sizes, sizes)
    positions = np.arange(n, dtype=np.int64) - start_positions

    ranked_sorted = (
        positions.astype(np.float64) + 0.5
    ) / np.maximum(row_sizes, 1).astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


def build_matrix(split, split_name):
    columns = []

    for field in CAT_FIELDS:
        columns.append(
            np.asarray(split.X[field], dtype=np.float32)
        )

    for field in NUM_FIELDS:
        values = np.asarray(split.num[field], dtype=np.float32)
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=1.0e7,
            neginf=0.0,
        )
        values = np.sign(values) * np.log1p(np.abs(values))
        columns.append(values.astype(np.float32))

    for entity in ("video_id", "author_id"):
        histories = historical_features(split_name, key=entity)
        for name in sorted(histories):
            values = np.asarray(histories[name], dtype=np.float32)
            values = np.nan_to_num(
                values,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )
            columns.append(values)

    matrix = np.ascontiguousarray(
        np.column_stack(columns),
        dtype=np.float32,
    )
    matrix = np.nan_to_num(
        matrix,
        nan=0.0,
        posinf=20.0,
        neginf=-20.0,
    )
    return matrix


def selected_leaves(model, matrix):
    leaves = model.predict(
        matrix,
        pred_leaf=True,
        num_iteration=model.best_iteration,
    )
    leaves = np.asarray(leaves)
    if leaves.ndim == 1:
        leaves = leaves[:, None]
    leaves = leaves[:, ::LEAF_STRIDE]
    return np.ascontiguousarray(leaves, dtype=np.int32)


def leaf_sparse(leaves, leaf_cardinality):
    n_rows, n_trees = leaves.shape
    rows = np.repeat(
        np.arange(n_rows, dtype=np.int32),
        n_trees,
    )
    tree_offsets = (
        np.arange(n_trees, dtype=np.int32) * leaf_cardinality
    )
    cols = (
        leaves + tree_offsets[None, :]
    ).reshape(-1).astype(np.int32)
    data = np.ones(len(cols), dtype=np.float32)

    return sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n_rows, n_trees * leaf_cardinality),
        dtype=np.float32,
    )


def fit_leaf_logistic(model, x_train, labels, weights):
    probe = selected_leaves(model, x_train[:1000])
    n_selected_trees = probe.shape[1]
    max_leaf = int(np.max(probe)) + 2

    # LightGBM uses at most num_leaves leaf identifiers, but inspect all
    # training batches and reserve a safe cardinality.
    leaf_cardinality = max(
        int(model.params.get("num_leaves", 63)) + 2,
        max_leaf,
    )

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=1.2e-5,
        l1_ratio=0.02,
        learning_rate="optimal",
        average=True,
        random_state=SEED + 17,
    )

    rng = np.random.default_rng(SEED + 31)
    order = np.arange(len(labels), dtype=np.int64)
    first = True

    for epoch in range(3):
        rng.shuffle(order)
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            leaves = selected_leaves(model, x_train[indices])
            sparse = leaf_sparse(leaves, leaf_cardinality)

            if first:
                classifier.partial_fit(
                    sparse,
                    labels[indices],
                    classes=np.asarray([0, 1], dtype=np.int8),
                    sample_weight=weights[indices],
                )
                first = False
            else:
                classifier.partial_fit(
                    sparse,
                    labels[indices],
                    sample_weight=weights[indices],
                )

        print("FINDINGS " + json.dumps({
            "leaf_logistic_epoch": epoch + 1,
            "selected_trees": int(n_selected_trees),
            "coefficient_nonzero": int(
                np.count_nonzero(classifier.coef_)
            ),
        }, sort_keys=True))

    return classifier, leaf_cardinality


def predict_leaf_logistic(
    classifier, leaf_cardinality, model, matrix
):
    output = np.empty(len(matrix), dtype=np.float32)
    for start in range(0, len(matrix), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(matrix))
        leaves = selected_leaves(model, matrix[start:end])
        sparse = leaf_sparse(leaves, leaf_cardinality)
        output[start:end] = classifier.decision_function(
            sparse
        ).astype(np.float32)
    return output


def fit_leaf_bayes(model, x_train, labels, weights):
    probe = selected_leaves(model, x_train[:1000])
    n_trees = probe.shape[1]
    leaf_cardinality = int(
        model.params.get("num_leaves", 63)
    ) + 2

    weighted_total = float(np.sum(weights))
    weighted_positive = float(np.sum(weights * labels))
    global_rate = weighted_positive / max(weighted_total, 1e-12)
    global_logit = np.log(
        np.clip(global_rate, 1e-6, 1.0 - 1e-6)
        / np.clip(1.0 - global_rate, 1e-6, 1.0)
    )

    counts = np.zeros(
        (n_trees, leaf_cardinality), dtype=np.float64
    )
    positives = np.zeros_like(counts)

    for start in range(0, len(labels), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(labels))
        leaves = selected_leaves(model, x_train[start:end])
        local_weights = weights[start:end].astype(np.float64)
        local_positive = (
            local_weights * labels[start:end]
        )

        for tree_index in range(n_trees):
            leaf = leaves[:, tree_index]
            counts[tree_index] += np.bincount(
                leaf,
                weights=local_weights,
                minlength=leaf_cardinality,
            )[:leaf_cardinality]
            positives[tree_index] += np.bincount(
                leaf,
                weights=local_positive,
                minlength=leaf_cardinality,
            )[:leaf_cardinality]

    # Later boosting leaves are typically narrower and more correlated.
    # Strong smoothing and 1/sqrt(tree age) temper that redundancy.
    smoothing = 90.0
    rates = (
        positives + smoothing * global_rate
    ) / (counts + smoothing)
    rates = np.clip(rates, 1e-5, 1.0 - 1e-5)
    leaf_effects = np.log(rates / (1.0 - rates)) - global_logit

    tree_weights = 1.0 / np.sqrt(
        1.0 + np.arange(n_trees, dtype=np.float64)
    )
    leaf_effects *= tree_weights[:, None]

    return leaf_effects.astype(np.float32), leaf_cardinality


def predict_leaf_bayes(
    model, matrix, leaf_effects, leaf_cardinality
):
    output = np.empty(len(matrix), dtype=np.float32)
    n_trees = leaf_effects.shape[0]

    for start in range(0, len(matrix), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(matrix))
        leaves = selected_leaves(model, matrix[start:end])
        leaves = np.minimum(leaves, leaf_cardinality - 1)

        scores = np.zeros(end - start, dtype=np.float32)
        for tree_index in range(n_trees):
            scores += leaf_effects[
                tree_index, leaves[:, tree_index]
            ]
        output[start:end] = scores

    return output


train = load("train")
valid = load("valid")
test = load("test")

labels = np.asarray(train.y, dtype=np.int8)
weights = recency_weights(train.date)

x_train = build_matrix(train, "train")
x_valid = build_matrix(valid, "valid")
x_test = build_matrix(test, "test")

categorical_indices = list(range(len(CAT_FIELDS)))

print("FINDINGS " + json.dumps({
    "train_rows": int(len(train)),
    "matrix_dimension": int(x_train.shape[1]),
    "categorical_fields": int(len(CAT_FIELDS)),
    "half_life_days": HALF_LIFE,
    "weight_min": float(weights.min()),
    "weight_max": float(weights.max()),
}, sort_keys=True))

train_set = lgb.Dataset(
    x_train,
    label=labels,
    weight=weights,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.045,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 180,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.86,
    "bagging_freq": 1,
    "lambda_l1": 0.15,
    "lambda_l2": 2.5,
    "max_bin": 127,
    "cat_smooth": 30.0,
    "cat_l2": 12.0,
    "min_gain_to_split": 0.01,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "data_random_seed": SEED + 3,
    "num_threads": THREADS,
    "verbose": -1,
}

booster = lgb.train(
    params,
    train_set,
    num_boost_round=NUM_BOOST_ROUND,
)
booster.best_iteration = NUM_BOOST_ROUND

gbdt_valid = booster.predict(
    x_valid,
    num_iteration=NUM_BOOST_ROUND,
).astype(np.float32)
gbdt_test = booster.predict(
    x_test,
    num_iteration=NUM_BOOST_ROUND,
).astype(np.float32)

leaf_classifier, leaf_cardinality = fit_leaf_logistic(
    booster,
    x_train,
    labels,
    weights,
)
leaf_logistic_valid = predict_leaf_logistic(
    leaf_classifier,
    leaf_cardinality,
    booster,
    x_valid,
)
leaf_logistic_test = predict_leaf_logistic(
    leaf_classifier,
    leaf_cardinality,
    booster,
    x_test,
)

leaf_effects, bayes_cardinality = fit_leaf_bayes(
    booster,
    x_train,
    labels.astype(np.float32),
    weights,
)
leaf_bayes_valid = predict_leaf_bayes(
    booster,
    x_valid,
    leaf_effects,
    bayes_cardinality,
)
leaf_bayes_test = predict_leaf_bayes(
    booster,
    x_test,
    leaf_effects,
    bayes_cardinality,
)

del x_train, train_set, leaf_classifier, leaf_effects
gc.collect()

own_valid = {
    "recency_categorical_gbdt": gbdt_valid,
    "leaf_rule_logistic": leaf_logistic_valid,
    "leaf_empirical_bayes": leaf_bayes_valid,
}
own_test = {
    "recency_categorical_gbdt": gbdt_test,
    "leaf_rule_logistic": leaf_logistic_test,
    "leaf_empirical_bayes": leaf_bayes_test,
}

own_valid_ranks = {
    name: within_user_rank(valid.user_id, scores)
    for name, scores in own_valid.items()
}
own_test_ranks = {
    name: within_user_rank(test.user_id, scores)
    for name, scores in own_test.items()
}

own_valid["three_rule_family_ensemble"] = (
    own_valid_ranks["recency_categorical_gbdt"]
    + own_valid_ranks["leaf_rule_logistic"]
    + own_valid_ranks["leaf_empirical_bayes"]
) / 3.0
own_test["three_rule_family_ensemble"] = (
    own_test_ranks["recency_categorical_gbdt"]
    + own_test_ranks["leaf_rule_logistic"]
    + own_test_ranks["leaf_empirical_bayes"]
) / 3.0

own_valid["reestimated_leaf_ensemble"] = (
    own_valid_ranks["leaf_rule_logistic"]
    + own_valid_ranks["leaf_empirical_bayes"]
) / 2.0
own_test["reestimated_leaf_ensemble"] = (
    own_test_ranks["leaf_rule_logistic"]
    + own_test_ranks["leaf_empirical_bayes"]
) / 2.0

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

candidate_valid = {
    "trusted_incumbent": inc_valid,
}
candidate_test = {
    "trusted_incumbent": inc_test,
}
candidate_source = {
    "trusted_incumbent": None,
}
candidate_metrics = {
    "trusted_incumbent": evaluate(
        valid.user_id, valid.y, inc_valid
    )
}

for family_name in own_valid:
    raw_valid = np.asarray(own_valid[family_name], dtype=np.float64)
    raw_test = np.asarray(own_test[family_name], dtype=np.float64)

    candidate_valid[family_name] = raw_valid
    candidate_test[family_name] = raw_test
    candidate_source[family_name] = family_name
    candidate_metrics[family_name] = evaluate(
        valid.user_id, valid.y, raw_valid
    )

    family_valid_rank = within_user_rank(
        valid.user_id, raw_valid
    )
    family_test_rank = within_user_rank(
        test.user_id, raw_test
    )

    for alpha in (0.10, 0.20, 0.30, 0.40, 0.50):
        name = f"{family_name}_incumbent_{alpha:.2f}"
        blended_valid = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * family_valid_rank
        )
        blended_test = (
            (1.0 - alpha) * inc_test_rank
            + alpha * family_test_rank
        )

        candidate_valid[name] = blended_valid
        candidate_test[name] = blended_test
        candidate_source[name] = family_name
        candidate_metrics[name] = evaluate(
            valid.user_id,
            valid.y,
            blended_valid,
        )

best_name = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"]),
)
best_metrics = candidate_metrics[best_name]
best_valid = np.asarray(candidate_valid[best_name], dtype=np.float64)
best_test = np.asarray(candidate_test[best_name], dtype=np.float64)

standalone_names = list(own_valid.keys())
best_own_name = max(
    standalone_names,
    key=lambda name: float(candidate_metrics[name]["primary"]),
)

audit_name = candidate_source[best_name]
if audit_name is None:
    audit_name = best_own_name
audit_valid = np.asarray(own_valid[audit_name], dtype=np.float64)

base_family_names = [
    "recency_categorical_gbdt",
    "leaf_rule_logistic",
    "leaf_empirical_bayes",
]
rank_correlations = {}
for i in range(len(base_family_names)):
    for j in range(i + 1, len(base_family_names)):
        left = base_family_names[i]
        right = base_family_names[j]
        rank_correlations[f"{left}__{right}"] = float(
            np.corrcoef(
                own_valid_ranks[left],
                own_valid_ranks[right],
            )[0, 1]
        )

print("CANDIDATES " + json.dumps({
    name: float(metrics["primary"])
    for name, metrics in candidate_metrics.items()
}, sort_keys=True))

print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "best_own_family": best_own_name,
    "rank_correlations": rank_correlations,
    "boost_rounds": NUM_BOOST_ROUND,
    "leaf_stride": LEAF_STRIDE,
    "selected_leaf_rules": int(
        (NUM_BOOST_ROUND + LEAF_STRIDE - 1) // LEAF_STRIDE
    ),
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        best_valid,
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        best_test,
    )
    if best_name == "trusted_incumbent" or "_incumbent_" in best_name:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            audit_valid,
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))