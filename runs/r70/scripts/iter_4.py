import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.evaluate import evaluate


START_TIME = time.time()
EPS = 1e-7

CATEGORICAL_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "onehot_feat3",
    "onehot_feat8",
    "music_type",
    "user_active_degree",
    "register_days_bucket",
    "fans_user_num_range",
]

HISTORY_FIELDS = [
    ("user", None, 20.0),
    ("video", "video_id", 12.0),
    ("author", "author_id", 16.0),
    ("tag", "tag", 24.0),
    ("duration", "duration_bucket", 30.0),
    ("tab", "tab", 30.0),
    ("hour", "hour", 35.0),
]

NUMERIC_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

BLEND_ALPHAS = [
    0.0, 0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90, 1.0,
]


def make_keys(sample, field_name):
    users = np.asarray(sample.user_id, dtype=np.int64)
    if field_name is None:
        return users
    contexts = np.asarray(sample.X[field_name], dtype=np.int64)
    cardinality = int(contexts.max()) + 1
    return users * np.int64(cardinality) + contexts


def make_keys_with_cardinality(sample, field_name, cardinality):
    users = np.asarray(sample.user_id, dtype=np.int64)
    if field_name is None:
        return users
    contexts = np.asarray(sample.X[field_name], dtype=np.int64)
    return users * np.int64(cardinality) + contexts


def prior_group_statistics(keys, times, labels):
    """
    For each row, calculate count and positive sum from strictly earlier
    rows with the same key. Sorting by (key, time_ms, original row) follows
    the benchmark's specified order, including timestamp ties.
    """
    keys = np.asarray(keys, dtype=np.int64)
    times = np.asarray(times, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    n = len(keys)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, times, keys))
    sorted_keys = keys[order]
    sorted_y = labels[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_keys[1:] != sorted_keys[:-1]
    start_idx = np.flatnonzero(starts)
    end_idx = np.r_[start_idx[1:], n]
    lengths = end_idx - start_idx

    group_start_for_row = np.repeat(start_idx, lengths)
    prior_count_sorted = (
        np.arange(n, dtype=np.int64) - group_start_for_row
    )

    cumulative = np.cumsum(sorted_y, dtype=np.float64)
    cumulative_before = cumulative - sorted_y

    bases = np.zeros(len(start_idx), dtype=np.float64)
    has_predecessor = start_idx > 0
    bases[has_predecessor] = cumulative[
        start_idx[has_predecessor] - 1
    ]
    prior_sum_sorted = cumulative_before - np.repeat(bases, lengths)

    prior_count = np.empty(n, dtype=np.float32)
    prior_sum = np.empty(n, dtype=np.float32)
    prior_count[order] = prior_count_sorted.astype(np.float32)
    prior_sum[order] = prior_sum_sorted.astype(np.float32)
    return prior_count, prior_sum


class HistoryLookup:
    def __init__(self, keys, labels):
        keys = np.asarray(keys, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.float64)

        unique_keys, inverse = np.unique(keys, return_inverse=True)
        counts = np.bincount(inverse).astype(np.float32)
        sums = np.bincount(
            inverse, weights=labels
        ).astype(np.float32)

        self.keys = unique_keys
        self.counts = counts
        self.sums = sums

    def transform(self, query_keys):
        query_keys = np.asarray(query_keys, dtype=np.int64)
        positions = np.searchsorted(self.keys, query_keys)

        matched = positions < len(self.keys)
        safe = np.minimum(positions, len(self.keys) - 1)
        matched &= self.keys[safe] == query_keys

        counts = np.zeros(len(query_keys), dtype=np.float32)
        sums = np.zeros(len(query_keys), dtype=np.float32)
        counts[matched] = self.counts[safe[matched]]
        sums[matched] = self.sums[safe[matched]]
        return counts, sums


def build_history_features(reference, target=None):
    """
    If target is None, produce strictly-past features for every reference
    training row. Otherwise, fit full-reference lookup tables and transform
    target without reading any target labels.
    """
    labels = np.asarray(reference.y, dtype=np.float64)
    global_rate = float(np.mean(labels))
    feature_columns = []
    lookups = []

    for short_name, field_name, strength in HISTORY_FIELDS:
        if field_name is None:
            cardinality = None
            ref_keys = np.asarray(reference.user_id, dtype=np.int64)
        else:
            ref_context = np.asarray(
                reference.X[field_name], dtype=np.int64
            )
            cardinality = int(ref_context.max()) + 1
            ref_keys = (
                np.asarray(reference.user_id, dtype=np.int64)
                * np.int64(cardinality)
                + ref_context
            )

        if target is None:
            counts, sums = prior_group_statistics(
                ref_keys, reference.time_ms, labels
            )
        else:
            lookup = HistoryLookup(ref_keys, labels)
            if field_name is None:
                target_keys = np.asarray(
                    target.user_id, dtype=np.int64
                )
            else:
                target_keys = make_keys_with_cardinality(
                    target, field_name, cardinality
                )
            counts, sums = lookup.transform(target_keys)
            lookups.append(lookup)

        rate = (
            sums.astype(np.float64) + float(strength) * global_rate
        ) / (
            counts.astype(np.float64) + float(strength)
        )

        # Centering turns each history rate into a preference residual.
        log_odds = np.log(
            np.clip(rate, EPS, 1.0 - EPS)
        ) - np.log1p(
            -np.clip(rate, EPS, 1.0 - EPS)
        )
        global_log_odds = (
            np.log(global_rate) - np.log1p(-global_rate)
        )
        residual = np.clip(
            log_odds - global_log_odds, -3.0, 3.0
        )

        feature_columns.append(
            np.log1p(counts).astype(np.float32)
        )
        feature_columns.append(
            residual.astype(np.float32)
        )

    return feature_columns


def numeric_columns(sample):
    columns = []
    for name in NUMERIC_FIELDS:
        x = np.asarray(sample.num[name], dtype=np.float64)
        finite = np.isfinite(x)
        clean = np.zeros(len(x), dtype=np.float64)
        clean[finite] = np.maximum(x[finite], 0.0)
        columns.append(np.log1p(clean).astype(np.float32))
        columns.append((~finite).astype(np.float32))
    return columns


def assemble_matrix(sample, history_columns):
    columns = []

    for name in CATEGORICAL_FIELDS:
        columns.append(
            np.asarray(sample.X[name], dtype=np.float32)
        )

    columns.extend(numeric_columns(sample))
    columns.extend(history_columns)

    # Calendar recency is safe and lets the tree compensate for the observed
    # label-rate drift without using outcomes from the evaluation split.
    date = np.asarray(sample.date, dtype=np.int32)
    date_offset = (date - int(date.min())).astype(np.float32)
    columns.append(date_offset)

    return np.column_stack(columns).astype(np.float32, copy=False)


def within_user_rank(user_ids, scores):
    """
    Convert scores to normalized within-user ranks. This preserves each
    model's ranking while putting independently calibrated models on a
    common scale for rank aggregation.
    """
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(users)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, users))
    sorted_users = users[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    start_idx = np.flatnonzero(starts)
    end_idx = np.r_[start_idx[1:], n]
    lengths = end_idx - start_idx

    positions = (
        np.arange(n, dtype=np.float64)
        - np.repeat(start_idx, lengths)
    )
    denominators = np.repeat(
        np.maximum(lengths - 1, 1), lengths
    ).astype(np.float64)
    ranked_sorted = positions / denominators

    singleton = np.repeat(lengths == 1, lengths)
    ranked_sorted[singleton] = 0.5

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def train_lgbm(X, y, num_boost_round):
    dataset = lgb.Dataset(
        X,
        label=np.asarray(y, dtype=np.float32),
        categorical_feature=list(range(len(CATEGORICAL_FIELDS))),
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
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.15,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "min_gain_to_split": 1e-4,
        "num_threads": max(1, min(16, os.cpu_count() or 1)),
        "seed": 20260829,
        "feature_fraction_seed": 20260829,
        "bagging_seed": 20260829,
        "data_random_seed": 20260829,
        "deterministic": True,
        "force_col_wise": True,
        "verbose": -1,
    }
    return lgb.train(
        params,
        dataset,
        num_boost_round=num_boost_round,
    )


artifacts_dir = os.environ.get("RUN_ARTIFACTS", "")
incumbent_valid_path = os.path.join(
    artifacts_dir, "incumbent_valid_scores.npy"
)
incumbent_test_path = os.path.join(
    artifacts_dir, "incumbent_test_scores.npy"
)

if not os.path.exists(incumbent_valid_path):
    raise FileNotFoundError(
        "Trusted incumbent validation predictions are unavailable"
    )
if not os.path.exists(incumbent_test_path):
    raise FileNotFoundError(
        "Trusted incumbent test predictions are unavailable"
    )

train = load("train")
valid = load("valid")

incumbent_valid = np.asarray(
    np.load(incumbent_valid_path), dtype=np.float64
)
if len(incumbent_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation score length mismatch")

train_history = build_history_features(train, target=None)
valid_history = build_history_features(train, target=valid)

X_train = assemble_matrix(train, train_history)
X_valid = assemble_matrix(valid, valid_history)

del train_history, valid_history
gc.collect()

NUM_BOOST_ROUND = 420
model = train_lgbm(X_train, train.y, NUM_BOOST_ROUND)
lgb_valid = model.predict(
    X_valid, num_iteration=NUM_BOOST_ROUND
).astype(np.float64)

incumbent_rank = within_user_rank(
    valid.user_id, incumbent_valid
)
lgb_rank = within_user_rank(valid.user_id, lgb_valid)

candidate_metrics = {}
candidate_scores = {}
best_primary = -np.inf
best_alpha = None
best_scores = None
best_metrics = None

raw_lgb_metrics = evaluate(
    valid.user_id, valid.y, lgb_valid
)
candidate_metrics["lgb_raw"] = float(
    raw_lgb_metrics["primary"]
)

incumbent_metrics = evaluate(
    valid.user_id, valid.y, incumbent_rank
)
candidate_metrics["incumbent"] = float(
    incumbent_metrics["primary"]
)

# Alpha is the contribution of the new sequential-history model.
for alpha in BLEND_ALPHAS:
    scores = (
        (1.0 - float(alpha)) * incumbent_rank
        + float(alpha) * lgb_rank
    )
    metrics = evaluate(valid.user_id, valid.y, scores)
    name = "rank_blend_%.2f" % alpha
    candidate_metrics[name] = float(metrics["primary"])
    candidate_scores[name] = scores

    if float(metrics["primary"]) > best_primary:
        best_primary = float(metrics["primary"])
        best_alpha = float(alpha)
        best_scores = scores
        best_metrics = metrics

print(
    "CANDIDATES "
    + json.dumps(
        candidate_metrics, sort_keys=True, separators=(",", ":")
    ),
    flush=True,
)

feature_importance = model.feature_importance(
    importance_type="gain"
)
feature_names = (
    CATEGORICAL_FIELDS
    + [
        name + suffix
        for name in NUMERIC_FIELDS
        for suffix in ("_log", "_missing")
    ]
    + [
        short_name + suffix
        for short_name, _, _ in HISTORY_FIELDS
        for suffix in ("_history_count", "_history_residual")
    ]
    + ["date_offset"]
)
top_indices = np.argsort(feature_importance)[-12:][::-1]
top_features = [
    [
        feature_names[int(i)],
        float(feature_importance[int(i)]),
    ]
    for i in top_indices
]

print(
    "FINDINGS best_rank_blend_alpha="
    + json.dumps(best_alpha)
    + " top_gain_features="
    + json.dumps(top_features, separators=(",", ":")),
    flush=True,
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )

# Free the train-only matrices and model before constructing the larger
# train+validation refit.
del X_train, X_valid, model, lgb_valid
gc.collect()

test = load("test")
incumbent_test = np.asarray(
    np.load(incumbent_test_path), dtype=np.float64
)
if len(incumbent_test) != len(test.user_id):
    raise ValueError("Incumbent test score length mismatch")

# Refit the identical fixed-round recipe on train + validation. Validation
# labels are now legal training labels for the final test model.
class CombinedSample:
    pass


combined = CombinedSample()
combined.user_id = np.concatenate([
    np.asarray(train.user_id, dtype=np.int64),
    np.asarray(valid.user_id, dtype=np.int64),
])
combined.video_id = np.concatenate([
    np.asarray(train.video_id, dtype=np.int64),
    np.asarray(valid.video_id, dtype=np.int64),
])
combined.time_ms = np.concatenate([
    np.asarray(train.time_ms, dtype=np.int64),
    np.asarray(valid.time_ms, dtype=np.int64),
])
combined.date = np.concatenate([
    np.asarray(train.date, dtype=np.int32),
    np.asarray(valid.date, dtype=np.int32),
])
combined.y = np.concatenate([
    np.asarray(train.y, dtype=np.int8),
    np.asarray(valid.y, dtype=np.int8),
])
combined.X = {
    name: np.concatenate([
        np.asarray(train.X[name], dtype=np.int64),
        np.asarray(valid.X[name], dtype=np.int64),
    ])
    for name in set(
        CATEGORICAL_FIELDS
        + [
            field_name
            for _, field_name, _ in HISTORY_FIELDS
            if field_name is not None
        ]
    )
}
combined.num = {
    name: np.concatenate([
        np.asarray(train.num[name], dtype=np.float32),
        np.asarray(valid.num[name], dtype=np.float32),
    ])
    for name in NUMERIC_FIELDS
}

combined_history = build_history_features(combined, target=None)
test_history = build_history_features(combined, target=test)

X_combined = assemble_matrix(combined, combined_history)
X_test = assemble_matrix(test, test_history)

del combined_history, test_history
gc.collect()

final_model = train_lgbm(
    X_combined, combined.y, NUM_BOOST_ROUND
)
lgb_test = final_model.predict(
    X_test, num_iteration=NUM_BOOST_ROUND
).astype(np.float64)

incumbent_test_rank = within_user_rank(
    test.user_id, incumbent_test
)
lgb_test_rank = within_user_rank(test.user_id, lgb_test)
test_scores = (
    (1.0 - best_alpha) * incumbent_test_rank
    + best_alpha * lgb_test_rank
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START_TIME
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(",", ":"),
    )
)