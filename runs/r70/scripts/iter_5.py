import os
import gc
import json
import time
import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.evaluate import evaluate


START = time.time()
EPS = 1e-7

CAT_FIELDS = [
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

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

RATE_FIELDS = [
    ("video_id", 12.0),
    ("author_id", 16.0),
    ("tag", 35.0),
    ("duration_bucket", 40.0),
    ("tab", 45.0),
    ("hour", 50.0),
    ("upload_type", 40.0),
]

BLEND_ALPHAS = [
    0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
    0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0,
]


class CombinedSample:
    pass


def make_combined(a, b):
    c = CombinedSample()
    c.user_id = np.concatenate([
        np.asarray(a.user_id, dtype=np.int64),
        np.asarray(b.user_id, dtype=np.int64),
    ])
    c.video_id = np.concatenate([
        np.asarray(a.video_id, dtype=np.int64),
        np.asarray(b.video_id, dtype=np.int64),
    ])
    c.time_ms = np.concatenate([
        np.asarray(a.time_ms, dtype=np.int64),
        np.asarray(b.time_ms, dtype=np.int64),
    ])
    c.date = np.concatenate([
        np.asarray(a.date, dtype=np.int32),
        np.asarray(b.date, dtype=np.int32),
    ])
    c.y = np.concatenate([
        np.asarray(a.y, dtype=np.int8),
        np.asarray(b.y, dtype=np.int8),
    ])
    c.X = {
        name: np.concatenate([
            np.asarray(a.X[name], dtype=np.int64),
            np.asarray(b.X[name], dtype=np.int64),
        ])
        for name in CAT_FIELDS
    }
    c.num = {
        name: np.concatenate([
            np.asarray(a.num[name], dtype=np.float32),
            np.asarray(b.num[name], dtype=np.float32),
        ])
        for name in NUM_FIELDS
    }
    return c


def group_mean_std(users, values):
    users = np.asarray(users, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)

    unique_users, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.bincount(inverse, weights=values).astype(np.float64)
    sums2 = np.bincount(
        inverse, weights=values * values
    ).astype(np.float64)

    means = sums / np.maximum(counts, 1.0)
    variances = np.maximum(
        sums2 / np.maximum(counts, 1.0) - means * means,
        0.0,
    )
    stds = np.sqrt(variances)

    centered = values - means[inverse]
    zscore = centered / np.maximum(stds[inverse], 1e-4)
    return (
        centered.astype(np.float32),
        np.clip(zscore, -6.0, 6.0).astype(np.float32),
    )


def within_user_average_percentile(users, values):
    """
    Average percentile rank within each user. Equal values receive exactly
    the same rank, avoiding row-order information in categorical/rate ties.
    """
    users = np.asarray(users, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    n = len(users)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, values, users))
    su = users[order]
    sv = values[order]

    user_start = np.empty(n, dtype=bool)
    user_start[0] = True
    user_start[1:] = su[1:] != su[:-1]
    user_starts = np.flatnonzero(user_start)
    user_ends = np.r_[user_starts[1:], n]
    user_lengths = user_ends - user_starts
    repeated_user_start = np.repeat(user_starts, user_lengths)
    repeated_user_length = np.repeat(user_lengths, user_lengths)

    tie_start = np.empty(n, dtype=bool)
    tie_start[0] = True
    tie_start[1:] = (
        (su[1:] != su[:-1])
        | (sv[1:] != sv[:-1])
    )
    tie_starts = np.flatnonzero(tie_start)
    tie_ends = np.r_[tie_starts[1:], n]
    tie_lengths = tie_ends - tie_starts

    average_sorted_position = np.repeat(
        (tie_starts + tie_ends - 1) * 0.5,
        tie_lengths,
    )
    local_average_position = (
        average_sorted_position - repeated_user_start
    )
    denominator = np.maximum(repeated_user_length - 1, 1)
    percentile_sorted = local_average_position / denominator
    percentile_sorted[repeated_user_length == 1] = 0.5

    result = np.empty(n, dtype=np.float32)
    result[order] = percentile_sorted.astype(np.float32)
    return result


def within_user_rank(users, scores):
    return within_user_average_percentile(users, scores).astype(
        np.float64
    )


def pair_frequency(users, values):
    users = np.asarray(users, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64)

    max_value = int(values.max()) + 1
    keys = users * np.int64(max_value) + values
    _, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True
    )
    return counts[inverse].astype(np.float32)


class RateLookup:
    def __init__(self, keys, labels):
        keys = np.asarray(keys, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.float64)
        self.keys, inverse = np.unique(keys, return_inverse=True)
        self.counts = np.bincount(inverse).astype(np.float64)
        self.sums = np.bincount(
            inverse, weights=labels
        ).astype(np.float64)

    def transform(self, keys):
        keys = np.asarray(keys, dtype=np.int64)
        pos = np.searchsorted(self.keys, keys)
        safe = np.minimum(pos, len(self.keys) - 1)
        matched = (
            (pos < len(self.keys))
            & (self.keys[safe] == keys)
        )

        counts = np.zeros(len(keys), dtype=np.float64)
        sums = np.zeros(len(keys), dtype=np.float64)
        counts[matched] = self.counts[safe[matched]]
        sums[matched] = self.sums[safe[matched]]
        return counts, sums


def target_rate_columns(reference, target=None):
    labels = np.asarray(reference.y, dtype=np.float64)
    global_rate = float(labels.mean())
    global_logit = np.log(global_rate) - np.log1p(-global_rate)

    columns = []
    names = []
    rate_signals = []

    for field, strength in RATE_FIELDS:
        ref_keys = np.asarray(reference.X[field], dtype=np.int64)

        if target is None:
            unique_keys, inverse = np.unique(
                ref_keys, return_inverse=True
            )
            full_counts = np.bincount(inverse).astype(np.float64)
            full_sums = np.bincount(
                inverse, weights=labels
            ).astype(np.float64)

            counts = np.maximum(full_counts[inverse] - 1.0, 0.0)
            sums = full_sums[inverse] - labels
        else:
            lookup = RateLookup(ref_keys, labels)
            target_keys = np.asarray(
                target.X[field], dtype=np.int64
            )
            counts, sums = lookup.transform(target_keys)

        rate = (
            sums + float(strength) * global_rate
        ) / (
            counts + float(strength)
        )
        clipped = np.clip(rate, EPS, 1.0 - EPS)
        residual = (
            np.log(clipped) - np.log1p(-clipped) - global_logit
        )
        residual = np.clip(residual, -3.5, 3.5)

        columns.append(np.log1p(counts).astype(np.float32))
        columns.append(residual.astype(np.float32))
        names.extend([
            field + "_entity_log_count",
            field + "_entity_rate_residual",
        ])
        rate_signals.append(
            (field + "_entity_rate", residual.astype(np.float32))
        )

    return columns, names, rate_signals


def clean_numeric(sample, name):
    x = np.asarray(sample.num[name], dtype=np.float64)
    finite = np.isfinite(x)
    clean = np.zeros(len(x), dtype=np.float64)
    clean[finite] = np.maximum(x[finite], 0.0)
    return (
        np.log1p(clean).astype(np.float32),
        (~finite).astype(np.float32),
    )


def build_matrix(reference, target=None):
    sample = reference if target is None else target
    users = np.asarray(sample.user_id, dtype=np.int64)

    columns = []
    names = []

    for field in CAT_FIELDS:
        columns.append(
            np.asarray(sample.X[field], dtype=np.float32)
        )
        names.append(field)

    numeric_signals = []
    for field in NUM_FIELDS:
        log_value, missing = clean_numeric(sample, field)
        columns.extend([log_value, missing])
        names.extend([field + "_log", field + "_missing"])
        numeric_signals.append((field + "_log", log_value))

    rate_columns, rate_names, rate_signals = target_rate_columns(
        reference, target
    )
    columns.extend(rate_columns)
    names.extend(rate_names)

    # Candidate-set relative features use only the feature rows in the split
    # being ranked. They are therefore available unchanged at inference.
    relative_signals = [
        numeric_signals[0],  # raw duration is the main item numeric signal
    ] + rate_signals

    for signal_name, signal in relative_signals:
        percentile = within_user_average_percentile(users, signal)
        centered, zscore = group_mean_std(users, signal)
        columns.extend([percentile, centered, zscore])
        names.extend([
            signal_name + "_user_set_percentile",
            signal_name + "_user_set_centered",
            signal_name + "_user_set_zscore",
        ])

    # Repeated exposure is a property of the logged candidate set, not an
    # outcome. It can identify familiar videos/authors and repeated themes.
    for field in ["video_id", "author_id", "tag", "duration_bucket"]:
        frequency = pair_frequency(users, sample.X[field])
        log_frequency = np.log1p(frequency).astype(np.float32)
        percentile = within_user_average_percentile(
            users, frequency
        )
        columns.extend([log_frequency, percentile])
        names.extend([
            "user_set_" + field + "_frequency",
            "user_set_" + field + "_frequency_percentile",
        ])

    # Number of impressions controls interpretation of percentiles and
    # repeated-exposure counts.
    _, inverse, counts = np.unique(
        users, return_inverse=True, return_counts=True
    )
    columns.append(
        np.log1p(counts[inverse]).astype(np.float32)
    )
    names.append("user_set_log_size")

    X = np.column_stack(columns).astype(np.float32, copy=False)
    return X, names


def train_model(X, y, rounds):
    dataset = lgb.Dataset(
        X,
        label=np.asarray(y, dtype=np.float32),
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.04,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 180,
        "feature_fraction": 0.86,
        "bagging_fraction": 0.88,
        "bagging_freq": 1,
        "lambda_l1": 0.15,
        "lambda_l2": 2.5,
        "max_bin": 127,
        "min_gain_to_split": 1e-4,
        "num_threads": max(
            1, min(16, os.cpu_count() or 1)
        ),
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
        num_boost_round=rounds,
    )


artifacts = os.environ.get("RUN_ARTIFACTS", "")
inc_valid_path = os.path.join(
    artifacts, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    artifacts, "incumbent_test_scores.npy"
)

if not os.path.exists(inc_valid_path):
    raise FileNotFoundError(
        "Missing trusted incumbent validation predictions"
    )
if not os.path.exists(inc_test_path):
    raise FileNotFoundError(
        "Missing trusted incumbent test predictions"
    )

train = load("train")
valid = load("valid")

inc_valid = np.asarray(
    np.load(inc_valid_path), dtype=np.float64
)
if len(inc_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation length mismatch")

X_train, feature_names = build_matrix(train, target=None)
X_valid, valid_feature_names = build_matrix(train, target=valid)

if feature_names != valid_feature_names:
    raise RuntimeError("Training and validation feature mismatch")

ROUNDS = 460
model = train_model(X_train, train.y, ROUNDS)
candidate_valid = model.predict(
    X_valid, num_iteration=ROUNDS
).astype(np.float64)

inc_rank = within_user_rank(valid.user_id, inc_valid)
candidate_rank = within_user_rank(
    valid.user_id, candidate_valid
)

candidate_results = {}
raw_metrics = evaluate(
    valid.user_id, valid.y, candidate_valid
)
candidate_results["setwise_lgb_raw"] = float(
    raw_metrics["primary"]
)

inc_metrics = evaluate(
    valid.user_id, valid.y, inc_rank
)
candidate_results["incumbent"] = float(
    inc_metrics["primary"]
)

best_primary = -np.inf
best_alpha = 0.0
best_scores = None
best_metrics = None

for alpha in BLEND_ALPHAS:
    scores = (
        (1.0 - float(alpha)) * inc_rank
        + float(alpha) * candidate_rank
    )
    metrics = evaluate(valid.user_id, valid.y, scores)
    name = "rank_blend_%.2f" % alpha
    candidate_results[name] = float(metrics["primary"])

    if float(metrics["primary"]) > best_primary:
        best_primary = float(metrics["primary"])
        best_alpha = float(alpha)
        best_scores = scores.copy()
        best_metrics = metrics

importance = model.feature_importance(
    importance_type="gain"
)
top_indices = np.argsort(importance)[-15:][::-1]
top_features = [
    [feature_names[int(i)], float(importance[int(i)])]
    for i in top_indices
]

print(
    "CANDIDATES "
    + json.dumps(
        candidate_results,
        sort_keys=True,
        separators=(",", ":"),
    ),
    flush=True,
)
print(
    "FINDINGS "
    + "best_setwise_blend_alpha="
    + json.dumps(best_alpha)
    + " raw_setwise_primary="
    + json.dumps(float(raw_metrics["primary"]))
    + " top_gain_features="
    + json.dumps(top_features, separators=(",", ":")),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )

del X_train, X_valid, model, candidate_valid
gc.collect()

# Refit the identical recipe on train + validation and apply the selected
# validation blend weight to test predictions.
test = load("test")
inc_test = np.asarray(
    np.load(inc_test_path), dtype=np.float64
)
if len(inc_test) != len(test.user_id):
    raise ValueError("Incumbent test length mismatch")

combined = make_combined(train, valid)

X_combined, combined_feature_names = build_matrix(
    combined, target=None
)
X_test, test_feature_names = build_matrix(
    combined, target=test
)

if combined_feature_names != test_feature_names:
    raise RuntimeError("Combined and test feature mismatch")
if combined_feature_names != feature_names:
    raise RuntimeError("Refit recipe feature mismatch")

test_model = train_model(
    X_combined, combined.y, ROUNDS
)
candidate_test = test_model.predict(
    X_test, num_iteration=ROUNDS
).astype(np.float64)

inc_test_rank = within_user_rank(
    test.user_id, inc_test
)
candidate_test_rank = within_user_rank(
    test.user_id, candidate_test
)
test_scores = (
    (1.0 - best_alpha) * inc_test_rank
    + best_alpha * candidate_test_rank
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
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
        },
        separators=(",", ":"),
    )
)