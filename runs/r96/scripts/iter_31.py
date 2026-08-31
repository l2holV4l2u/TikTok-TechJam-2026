import os
import time
import json
import gc
import numpy as np

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import SGDClassifier

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 83147
THREADS = min(16, os.cpu_count() or 1)
rng = np.random.default_rng(SEED)

TE_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "hour",
    "upload_type",
    "music_type",
    "user_active_degree",
    "fans_user_num_range",
    "register_days_range",
    "onehot_feat3",
    "onehot_feat8",
]

PAIR_FIELDS = [
    ("user_id", "tag"),
    ("user_id", "duration_bucket"),
    ("video_id", "tab"),
    ("video_id", "tag"),
    ("author_id", "tag"),
    ("author_id", "tab"),
    ("onehot_feat3", "tag"),
    ("onehot_feat8", "duration_bucket"),
]

RAW_FIELDS = [
    "tab",
    "duration_bucket",
    "tag",
    "hour",
    "upload_type",
    "music_type",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "is_video_author",
    "is_live_streamer",
    "video_type",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

N_FOLDS = 5
SMOOTH = 24.0
HALF_LIFE = 4.0
TREE_SAMPLE = 600000
RFF_DIM = 80
RFF_BATCH = 32768


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    ordered_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = ordered_users[1:] != ordered_users[:-1]
    start_pos = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = ordered_users[:-1] != ordered_users[1:]
    end_pos = np.flatnonzero(ends)
    sizes = np.diff(np.concatenate((
        np.asarray([-1], dtype=np.int64),
        end_pos,
    )))
    row_sizes = np.repeat(sizes, sizes)
    positions = np.arange(n, dtype=np.int64) - start_pos

    ranked_ordered = (
        positions.astype(np.float64) + 0.5
    ) / np.maximum(row_sizes, 1).astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_ordered
    return result


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int32)
    age = dates.max() - dates
    weights = np.power(
        0.5,
        age.astype(np.float32) / HALF_LIFE,
    )
    weights /= max(float(np.mean(weights)), 1e-8)
    return weights.astype(np.float32)


def key_array(split, specification):
    if isinstance(specification, str):
        return (
            np.asarray(split.X[specification], dtype=np.int64),
            int(FEATURE_CARDINALITIES[specification]),
        )

    left, right = specification
    left_values = np.asarray(split.X[left], dtype=np.int64)
    right_values = np.asarray(split.X[right], dtype=np.int64)
    right_cardinality = int(FEATURE_CARDINALITIES[right])
    cardinality = (
        int(FEATURE_CARDINALITIES[left]) * right_cardinality
    )
    key = left_values * right_cardinality + right_values
    return key, cardinality


def evidence_features(train, valid, test, labels, sample_weights):
    specifications = TE_FIELDS + PAIR_FIELDS
    n_train = len(train)
    global_rate = float(
        np.sum(sample_weights * labels)
        / np.maximum(np.sum(sample_weights), 1e-12)
    )

    # Row-hash folds are independent of outcomes and produce leakage-safe
    # out-of-fold evidence for every training impression.
    row_ids = np.arange(n_train, dtype=np.uint64)
    hashes = (
        row_ids * np.uint64(11400714819323198485)
        + np.uint64(SEED)
    )
    folds = np.asarray(hashes % np.uint64(N_FOLDS), dtype=np.int8)

    train_columns = []
    valid_columns = []
    test_columns = []

    for spec in specifications:
        key_train, cardinality = key_array(train, spec)
        key_valid, _ = key_array(valid, spec)
        key_test, _ = key_array(test, spec)

        full_count = np.bincount(
            key_train,
            weights=sample_weights,
            minlength=cardinality,
        ).astype(np.float64)
        full_positive = np.bincount(
            key_train,
            weights=sample_weights * labels,
            minlength=cardinality,
        ).astype(np.float64)

        train_rate = np.empty(n_train, dtype=np.float32)
        train_count = np.empty(n_train, dtype=np.float32)

        for fold in range(N_FOLDS):
            mask = folds == fold
            fold_keys = key_train[mask]
            held_count = np.bincount(
                fold_keys,
                weights=sample_weights[mask],
                minlength=cardinality,
            )
            held_positive = np.bincount(
                fold_keys,
                weights=sample_weights[mask] * labels[mask],
                minlength=cardinality,
            )

            counts = full_count[fold_keys] - held_count[fold_keys]
            positives = (
                full_positive[fold_keys] - held_positive[fold_keys]
            )
            train_rate[mask] = (
                (positives + SMOOTH * global_rate)
                / (counts + SMOOTH)
            ).astype(np.float32)
            train_count[mask] = np.log1p(
                np.maximum(counts, 0.0)
            ).astype(np.float32)

        def apply_full(keys):
            counts = full_count[keys]
            positives = full_positive[keys]
            rates = (
                (positives + SMOOTH * global_rate)
                / (counts + SMOOTH)
            )
            return (
                rates.astype(np.float32),
                np.log1p(counts).astype(np.float32),
            )

        valid_rate, valid_count = apply_full(key_valid)
        test_rate, test_count = apply_full(key_test)

        # Centering rates makes random-feature distances focus on evidence
        # deviations rather than the moving global label prevalence.
        train_columns.extend([
            train_rate - global_rate,
            train_count,
        ])
        valid_columns.extend([
            valid_rate - global_rate,
            valid_count,
        ])
        test_columns.extend([
            test_rate - global_rate,
            test_count,
        ])

    return train_columns, valid_columns, test_columns


def append_history(columns, split_name, entity):
    history = historical_features(split_name, key=entity)
    for name in sorted(history):
        values = np.asarray(history[name], dtype=np.float32)
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )
        columns.append(values)


def build_matrices(train, valid, test):
    labels = np.asarray(train.y, dtype=np.float32)
    weights = recency_weights(train.date)

    tr_cols, va_cols, te_cols = evidence_features(
        train, valid, test, labels, weights
    )

    for entity in ("video_id", "author_id"):
        append_history(tr_cols, "train", entity)
        append_history(va_cols, "valid", entity)
        append_history(te_cols, "test", entity)

    for field in NUM_FIELDS:
        for split, columns in (
            (train, tr_cols),
            (valid, va_cols),
            (test, te_cols),
        ):
            values = np.asarray(split.num[field], dtype=np.float32)
            values = np.nan_to_num(
                values,
                nan=0.0,
                posinf=1e7,
                neginf=0.0,
            )
            columns.append(
                np.sign(values) * np.log1p(np.abs(values))
            )

    for field in RAW_FIELDS:
        scale = max(float(FEATURE_CARDINALITIES[field] - 1), 1.0)
        tr_cols.append(
            np.asarray(train.X[field], dtype=np.float32) / scale
        )
        va_cols.append(
            np.asarray(valid.X[field], dtype=np.float32) / scale
        )
        te_cols.append(
            np.asarray(test.X[field], dtype=np.float32) / scale
        )

    x_train = np.ascontiguousarray(
        np.column_stack(tr_cols), dtype=np.float32
    )
    x_valid = np.ascontiguousarray(
        np.column_stack(va_cols), dtype=np.float32
    )
    x_test = np.ascontiguousarray(
        np.column_stack(te_cols), dtype=np.float32
    )

    x_train = np.nan_to_num(
        x_train, nan=0.0, posinf=20.0, neginf=-20.0
    )
    x_valid = np.nan_to_num(
        x_valid, nan=0.0, posinf=20.0, neginf=-20.0
    )
    x_test = np.nan_to_num(
        x_test, nan=0.0, posinf=20.0, neginf=-20.0
    )

    print("FINDINGS " + json.dumps({
        "evidence_dimension": int(x_train.shape[1]),
        "train_rows": int(x_train.shape[0]),
        "global_positive_rate": float(labels.mean()),
        "recency_weight_min": float(weights.min()),
        "recency_weight_max": float(weights.max()),
    }, sort_keys=True))

    return x_train, x_valid, x_test, labels, weights


def weighted_sample_indices(weights, size, seed):
    local_rng = np.random.default_rng(seed)
    probability = np.asarray(weights, dtype=np.float64)
    probability /= probability.sum()
    size = min(int(size), len(weights))
    return local_rng.choice(
        len(weights),
        size=size,
        replace=False,
        p=probability,
    )


def fit_tree_families(x_train, labels, weights, x_valid, x_test):
    indices = weighted_sample_indices(
        weights, TREE_SAMPLE, SEED + 101
    )
    xt = x_train[indices]
    yt = labels[indices]

    outputs_valid = {}
    outputs_test = {}

    random_forest = RandomForestRegressor(
        n_estimators=112,
        max_depth=17,
        min_samples_leaf=70,
        max_features=0.60,
        bootstrap=True,
        max_samples=0.80,
        n_jobs=THREADS,
        random_state=SEED + 211,
        criterion="squared_error",
    )
    random_forest.fit(xt, yt)
    outputs_valid["random_forest_bagging"] = (
        random_forest.predict(x_valid).astype(np.float32)
    )
    outputs_test["random_forest_bagging"] = (
        random_forest.predict(x_test).astype(np.float32)
    )
    print("FINDINGS " + json.dumps({
        "family": "random_forest_bagging",
        "sample_rows": int(len(indices)),
        "trees": int(random_forest.n_estimators),
    }, sort_keys=True))
    del random_forest
    gc.collect()

    extra_trees = ExtraTreesRegressor(
        n_estimators=144,
        max_depth=19,
        min_samples_leaf=55,
        max_features=0.75,
        bootstrap=False,
        n_jobs=THREADS,
        random_state=SEED + 307,
        criterion="squared_error",
    )
    extra_trees.fit(xt, yt)
    outputs_valid["extremely_randomized_trees"] = (
        extra_trees.predict(x_valid).astype(np.float32)
    )
    outputs_test["extremely_randomized_trees"] = (
        extra_trees.predict(x_test).astype(np.float32)
    )
    print("FINDINGS " + json.dumps({
        "family": "extremely_randomized_trees",
        "sample_rows": int(len(indices)),
        "trees": int(extra_trees.n_estimators),
    }, sort_keys=True))
    del extra_trees, xt, yt, indices
    gc.collect()

    return outputs_valid, outputs_test


def standardization(x_train):
    mean = np.mean(x_train, axis=0, dtype=np.float64)
    std = np.std(x_train, axis=0, dtype=np.float64)
    std = np.maximum(std, 0.05)
    return mean.astype(np.float32), std.astype(np.float32)


def fit_random_feature_kernel(
    x_train, labels, weights, x_valid, x_test
):
    mean, std = standardization(x_train)
    n_features = x_train.shape[1]
    local_rng = np.random.default_rng(SEED + 401)

    # Random tanh features approximate a smooth nonlinear kernel over the
    # evidence vector without imposing ordinal tree splits on category ids.
    projection = local_rng.normal(
        0.0,
        0.58 / np.sqrt(max(n_features, 1)),
        size=(n_features, RFF_DIM),
    ).astype(np.float32)
    bias = local_rng.uniform(
        -1.0, 1.0, size=RFF_DIM
    ).astype(np.float32)

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=2.0e-5,
        l1_ratio=0.04,
        learning_rate="optimal",
        average=True,
        random_state=SEED + 503,
    )

    indices = np.arange(len(x_train), dtype=np.int64)
    first = True
    for epoch in range(3):
        local_rng.shuffle(indices)
        loss_probe = []

        for start in range(0, len(indices), RFF_BATCH):
            batch_indices = indices[start:start + RFF_BATCH]
            normalized = (
                x_train[batch_indices] - mean[None, :]
            ) / std[None, :]
            hidden = np.tanh(
                normalized @ projection + bias[None, :]
            ).astype(np.float32)

            if first:
                classifier.partial_fit(
                    hidden,
                    labels[batch_indices].astype(np.int8),
                    classes=np.asarray([0, 1], dtype=np.int8),
                    sample_weight=weights[batch_indices],
                )
                first = False
            else:
                classifier.partial_fit(
                    hidden,
                    labels[batch_indices].astype(np.int8),
                    sample_weight=weights[batch_indices],
                )

            if len(loss_probe) < 5:
                logits = classifier.decision_function(hidden)
                probe_y = labels[batch_indices]
                probe_loss = np.mean(
                    np.logaddexp(0.0, logits)
                    - probe_y * logits
                )
                loss_probe.append(float(probe_loss))

        print("FINDINGS " + json.dumps({
            "family": "random_feature_kernel",
            "epoch": epoch + 1,
            "probe_logloss": float(np.mean(loss_probe)),
        }, sort_keys=True))

    def predict(x):
        result = np.empty(len(x), dtype=np.float32)
        for start in range(0, len(x), RFF_BATCH):
            end = min(start + RFF_BATCH, len(x))
            normalized = (
                x[start:end] - mean[None, :]
            ) / std[None, :]
            hidden = np.tanh(
                normalized @ projection + bias[None, :]
            ).astype(np.float32)
            result[start:end] = classifier.decision_function(
                hidden
            ).astype(np.float32)
        return result

    return predict(x_valid), predict(x_test)


train = load("train")
valid = load("valid")
test = load("test")

x_train, x_valid, x_test, labels, weights = build_matrices(
    train, valid, test
)

family_valid, family_test = fit_tree_families(
    x_train, labels, weights, x_valid, x_test
)

kernel_valid, kernel_test = fit_random_feature_kernel(
    x_train, labels, weights, x_valid, x_test
)
family_valid["random_feature_kernel"] = kernel_valid
family_test["random_feature_kernel"] = kernel_test

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

valid_ranks = {
    name: within_user_rank(valid.user_id, scores)
    for name, scores in family_valid.items()
}
test_ranks = {
    name: within_user_rank(test.user_id, family_test[name])
    for name in family_test
}

# A cross-family aggregate tests whether smooth kernel similarity and
# randomized rule partitions make complementary ranking errors.
bagged_rank_valid = 0.5 * (
    valid_ranks["random_forest_bagging"]
    + valid_ranks["extremely_randomized_trees"]
)
bagged_rank_test = 0.5 * (
    test_ranks["random_forest_bagging"]
    + test_ranks["extremely_randomized_trees"]
)

all_family_rank_valid = (
    valid_ranks["random_forest_bagging"]
    + valid_ranks["extremely_randomized_trees"]
    + valid_ranks["random_feature_kernel"]
) / 3.0
all_family_rank_test = (
    test_ranks["random_forest_bagging"]
    + test_ranks["extremely_randomized_trees"]
    + test_ranks["random_feature_kernel"]
) / 3.0

aggregate_valid = {
    **{
        name: np.asarray(scores, dtype=np.float64)
        for name, scores in family_valid.items()
    },
    "tree_bagging_aggregate": bagged_rank_valid,
    "tree_kernel_aggregate": all_family_rank_valid,
}
aggregate_test = {
    **{
        name: np.asarray(scores, dtype=np.float64)
        for name, scores in family_test.items()
    },
    "tree_bagging_aggregate": bagged_rank_test,
    "tree_kernel_aggregate": all_family_rank_test,
}

candidate_valid = {"trusted_incumbent": inc_valid}
candidate_test = {"trusted_incumbent": inc_test}
candidate_raw_family = {"trusted_incumbent": None}
candidate_metrics = {
    "trusted_incumbent": evaluate(
        valid.user_id, valid.y, inc_valid
    )
}

for family_name in aggregate_valid:
    raw_valid = aggregate_valid[family_name]
    raw_test = aggregate_test[family_name]

    candidate_valid[family_name] = raw_valid
    candidate_test[family_name] = raw_test
    candidate_raw_family[family_name] = family_name
    candidate_metrics[family_name] = evaluate(
        valid.user_id, valid.y, raw_valid
    )

    family_rank_valid = within_user_rank(
        valid.user_id, raw_valid
    )
    family_rank_test = within_user_rank(
        test.user_id, raw_test
    )

    for alpha in (0.10, 0.20, 0.30, 0.40, 0.50):
        candidate_name = (
            f"{family_name}_incumbent_{alpha:.2f}"
        )
        blended_valid = (
            alpha * family_rank_valid
            + (1.0 - alpha) * inc_valid_rank
        )
        blended_test = (
            alpha * family_rank_test
            + (1.0 - alpha) * inc_test_rank
        )
        candidate_valid[candidate_name] = blended_valid
        candidate_test[candidate_name] = blended_test
        candidate_raw_family[candidate_name] = family_name
        candidate_metrics[candidate_name] = evaluate(
            valid.user_id, valid.y, blended_valid
        )

best_name = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"]),
)
best_metrics = candidate_metrics[best_name]
best_valid = np.asarray(
    candidate_valid[best_name], dtype=np.float64
)
best_test = np.asarray(
    candidate_test[best_name], dtype=np.float64
)

own_standalone_names = list(aggregate_valid)
best_own_name = max(
    own_standalone_names,
    key=lambda name: float(
        candidate_metrics[name]["primary"]
    ),
)
audit_family = candidate_raw_family[best_name]
if audit_family is None:
    audit_family = best_own_name
raw_valid_for_audit = np.asarray(
    aggregate_valid[audit_family], dtype=np.float64
)

rank_correlations = {}
family_names = list(family_valid)
for i in range(len(family_names)):
    for j in range(i + 1, len(family_names)):
        left = family_names[i]
        right = family_names[j]
        correlation = np.corrcoef(
            valid_ranks[left], valid_ranks[right]
        )[0, 1]
        rank_correlations[f"{left}__{right}"] = float(correlation)

print("CANDIDATES " + json.dumps({
    name: float(metrics["primary"])
    for name, metrics in candidate_metrics.items()
}, sort_keys=True))

print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "best_own_family": best_own_name,
    "rank_correlations": rank_correlations,
    "half_life_days": HALF_LIFE,
    "target_smoothing": SMOOTH,
    "target_encoding_folds": N_FOLDS,
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
            raw_valid_for_audit,
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))