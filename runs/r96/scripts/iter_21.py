import os
import time
import json
import random
import gc
import warnings
import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier

from pipeline.data import load
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")
START = time.time()
SEED = 240521
THREADS = max(1, min(12, os.cpu_count() or 1))

random.seed(SEED)
np.random.seed(SEED)


def rank_percentile(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, users))
    sorted_users = users[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    start_idx = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_idx = np.flatnonzero(ends)
    sizes = np.diff(np.r_[np.int64(-1), end_idx])
    row_sizes = np.repeat(sizes, sizes)

    positions = np.arange(n, dtype=np.int64) - start_idx
    ranked = (positions.astype(np.float64) + 0.5) / row_sizes

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def row_group_sizes(user_ids):
    _, inverse, counts = np.unique(
        np.asarray(user_ids, dtype=np.int64),
        return_inverse=True,
        return_counts=True,
    )
    return counts[inverse].astype(np.float64)


def logit(x):
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def fit_evidence_tables(split, labels, weights, fields):
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    global_rate = float(np.sum(weights * labels) / np.sum(weights))
    global_logit = float(logit(global_rate))

    tables = {}
    for field in fields:
        values = np.asarray(split.X[field], dtype=np.int64)
        size = int(values.max()) + 1
        counts = np.bincount(values, weights=weights, minlength=size)
        positives = np.bincount(
            values, weights=weights * labels, minlength=size
        )

        if field in ("video_id", "author_id"):
            prior = 70.0
        elif field in ("onehot_feat3", "onehot_feat8"):
            prior = 120.0
        else:
            prior = 250.0

        rate = (positives + prior * global_rate) / (counts + prior)
        effects = logit(rate) - global_logit
        reliability = counts / (counts + prior)
        effects *= reliability

        tables[field] = {
            "count": counts.astype(np.float64),
            "positive": positives.astype(np.float64),
            "effect": effects.astype(np.float64),
            "prior": float(prior),
        }

    return {
        "tables": tables,
        "global_rate": global_rate,
        "global_logit": global_logit,
    }


def evidence_matrix(state, split, train_labels=None, train_weights=None):
    fields = list(state["tables"].keys())
    n = len(split)
    result = np.empty((n, len(fields)), dtype=np.float32)
    global_rate = state["global_rate"]
    global_logit = state["global_logit"]

    is_training = train_labels is not None
    if is_training:
        y = np.asarray(train_labels, dtype=np.float64)
        w = np.asarray(train_weights, dtype=np.float64)

    for j, field in enumerate(fields):
        values = np.asarray(split.X[field], dtype=np.int64)
        table = state["tables"][field]
        effects = table["effect"]
        prior = table["prior"]

        if not is_training:
            column = np.zeros(n, dtype=np.float64)
            known = values < len(effects)
            column[known] = effects[values[known]]
        else:
            safe = np.minimum(values, len(table["count"]) - 1)
            count = table["count"][safe] - w
            positive = table["positive"][safe] - w * y
            count = np.maximum(count, 0.0)
            positive = np.clip(positive, 0.0, count)
            rate = (positive + prior * global_rate) / (count + prior)
            column = (logit(rate) - global_logit) * (
                count / (count + prior)
            )

        result[:, j] = np.clip(column, -5.0, 5.0).astype(np.float32)

    return result


def predict_generative(state, split):
    matrix = evidence_matrix(state, split)
    # Averaging avoids allowing the number of correlated side fields to
    # arbitrarily dominate the likelihood ratio.
    return (
        state["global_logit"]
        + np.mean(matrix.astype(np.float64), axis=1)
    )


HASH_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "hour",
    "user_active_degree",
    "upload_type",
    "onehot_feat3",
]

HASH_PAIRS = [
    ("user_id", "video_id"),
    ("user_id", "author_id"),
    ("user_id", "duration_bucket"),
    ("user_id", "tag"),
    ("video_id", "tab"),
    ("video_id", "hour"),
    ("author_id", "tab"),
    ("author_id", "duration_bucket"),
    ("tag", "duration_bucket"),
    ("user_active_degree", "tab"),
]


def hashed_cross_matrix(split, n_hash=1 << 19):
    n = len(split)
    n_features_per_row = len(HASH_FIELDS) + len(HASH_PAIRS)
    columns = np.empty((n, n_features_per_row), dtype=np.int32)
    values = np.empty((n, n_features_per_row), dtype=np.float32)

    mask = np.uint64(n_hash - 1)
    p1 = np.uint64(11400714819323198485)
    p2 = np.uint64(14029467366897019727)

    j = 0
    for field_index, field in enumerate(HASH_FIELDS):
        x = np.asarray(split.X[field], dtype=np.uint64)
        h = (
            x * p1
            + np.uint64((field_index + 1) * 2654435761)
        )
        columns[:, j] = np.asarray(h & mask, dtype=np.int32)
        values[:, j] = np.where(
            ((h >> np.uint64(33)) & np.uint64(1)) == 0,
            1.0,
            -1.0,
        )
        j += 1

    for pair_index, (left, right) in enumerate(HASH_PAIRS):
        a = np.asarray(split.X[left], dtype=np.uint64)
        b = np.asarray(split.X[right], dtype=np.uint64)
        h = (
            a * p1
            + b * p2
            + np.uint64((pair_index + 31) * 2246822519)
        )
        columns[:, j] = np.asarray(h & mask, dtype=np.int32)
        values[:, j] = np.where(
            ((h >> np.uint64(31)) & np.uint64(1)) == 0,
            1.0,
            -1.0,
        )
        j += 1

    indptr = (
        np.arange(n + 1, dtype=np.int64) * n_features_per_row
    )
    matrix = sparse.csr_matrix(
        (values.ravel(), columns.ravel(), indptr),
        shape=(n, n_hash),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    return matrix


def tree_side_features(split, evidence):
    numeric_columns = []
    for name in (
        "duration_ms",
        "user_fans_user_num",
        "user_follow_user_num",
        "user_friend_user_num",
        "user_register_days",
    ):
        x = np.asarray(split.num[name], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        numeric_columns.append(
            np.log1p(np.maximum(x, 0.0)).astype(np.float32)
        )

    categorical_columns = []
    for name in (
        "hour",
        "tab",
        "duration_bucket",
        "tag",
        "upload_type",
        "user_active_degree",
        "is_video_author",
        "is_live_streamer",
        "onehot_feat1",
        "onehot_feat2",
        "onehot_feat4",
        "music_type",
    ):
        categorical_columns.append(
            np.asarray(split.X[name], dtype=np.float32)
        )

    return np.column_stack(
        [evidence] + numeric_columns + categorical_columns
    ).astype(np.float32)


train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.int8)
train_dates = np.asarray(train.date, dtype=np.int64)
max_train_date = int(train_dates.max())

# Recency is applied to every supervised family, not merely to a side model.
# A 3-day half-life strongly emphasizes the final training week while retaining
# enough support for sparse entities.
age_days = np.maximum(max_train_date - train_dates, 0).astype(np.float64)
sample_weight = np.power(0.5, age_days / 3.0)
sample_weight /= sample_weight.mean()

evidence_fields = [
    "video_id",
    "author_id",
    "duration_bucket",
    "tag",
    "tab",
    "hour",
    "upload_type",
    "user_active_degree",
    "is_video_author",
    "is_live_streamer",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat4",
    "onehot_feat8",
    "music_type",
]

evidence_state = fit_evidence_tables(
    train, y_train, sample_weight, evidence_fields
)

# Family 1: additive generative categorical likelihood ratios.
generative_valid = predict_generative(evidence_state, valid)
generative_test = predict_generative(evidence_state, test)

# Family 2: a sparse linear classifier over signed hashed conjunctions.
x_hash_train = hashed_cross_matrix(train)
hash_model = SGDClassifier(
    loss="log_loss",
    penalty="elasticnet",
    alpha=2.0e-7,
    l1_ratio=0.02,
    fit_intercept=True,
    max_iter=6,
    tol=None,
    shuffle=True,
    average=True,
    random_state=SEED + 1,
    n_jobs=THREADS,
)
hash_model.fit(x_hash_train, y_train, sample_weight=sample_weight)
del x_hash_train
gc.collect()

x_hash_valid = hashed_cross_matrix(valid)
hashed_valid = hash_model.decision_function(x_hash_valid).astype(np.float64)
del x_hash_valid
gc.collect()

x_hash_test = hashed_cross_matrix(test)
hashed_test = hash_model.decision_function(x_hash_test).astype(np.float64)
del x_hash_test, hash_model
gc.collect()

# Family 3: bagged randomized trees over leave-one-out entity evidence and
# stationary side information. LOO construction prevents target-encoding
# self leakage in the tree training rows.
ev_train_loo = evidence_matrix(
    evidence_state,
    train,
    train_labels=y_train,
    train_weights=sample_weight,
)
ev_valid = evidence_matrix(evidence_state, valid)
ev_test = evidence_matrix(evidence_state, test)

tree_train = tree_side_features(train, ev_train_loo)
tree_valid = tree_side_features(valid, ev_valid)
tree_test = tree_side_features(test, ev_test)
del ev_train_loo, ev_valid, ev_test
gc.collect()

rng = np.random.default_rng(SEED + 2)
sampling_probability = np.minimum(
    1.0, 430000.0 * sample_weight / np.sum(sample_weight)
)
sample_mask = rng.random(len(train)) < sampling_probability
sample_index = np.flatnonzero(sample_mask)

# Ensure a sufficiently large but recency-skewed training subset.
if len(sample_index) < 300000:
    probability = sample_weight / sample_weight.sum()
    sample_index = rng.choice(
        len(train), size=430000, replace=False, p=probability
    )

tree_model = ExtraTreesClassifier(
    n_estimators=128,
    criterion="entropy",
    max_depth=20,
    min_samples_leaf=80,
    max_features=0.75,
    bootstrap=True,
    max_samples=0.85,
    class_weight=None,
    n_jobs=THREADS,
    random_state=SEED + 3,
)
tree_model.fit(
    tree_train[sample_index],
    y_train[sample_index],
    sample_weight=sample_weight[sample_index],
)
tree_valid_scores = tree_model.predict_proba(tree_valid)[:, 1]
tree_test_scores = tree_model.predict_proba(tree_test)[:, 1]

del tree_model, tree_train, tree_valid, tree_test
gc.collect()

raw_valid = {
    "generative_nb": generative_valid,
    "hashed_cross_linear": hashed_valid,
    "extra_trees_evidence": tree_valid_scores,
}
raw_test = {
    "generative_nb": generative_test,
    "hashed_cross_linear": hashed_test,
    "extra_trees_evidence": tree_test_scores,
}

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = rank_percentile(valid.user_id, inc_valid)
inc_test_rank = rank_percentile(test.user_id, inc_test)

family_valid_rank = {
    name: rank_percentile(valid.user_id, score)
    for name, score in raw_valid.items()
}
family_test_rank = {
    name: rank_percentile(test.user_id, score)
    for name, score in raw_test.items()
}

# A fourth candidate is rank aggregation across all newly fitted families.
family_valid_rank["new_family_borda"] = np.mean(
    np.stack(list(family_valid_rank.values()), axis=0), axis=0
)
family_test_rank["new_family_borda"] = np.mean(
    np.stack(list(family_test_rank.values()), axis=0), axis=0
)
raw_valid["new_family_borda"] = family_valid_rank["new_family_borda"]
raw_test["new_family_borda"] = family_test_rank["new_family_borda"]

valid_sizes = row_group_sizes(valid.user_id)
test_sizes = row_group_sizes(test.user_id)

candidate_valid = {"incumbent": inc_valid}
candidate_test = {"incumbent": inc_test}
candidate_raw = {"incumbent": inc_valid}

for name in family_valid_rank:
    candidate_valid[name + "_standalone"] = raw_valid[name]
    candidate_test[name + "_standalone"] = raw_test[name]
    candidate_raw[name + "_standalone"] = raw_valid[name]

    for alpha in (0.03, 0.06, 0.10, 0.15, 0.22, 0.30, 0.40):
        key = f"{name}_global_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * family_valid_rank[name]
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank
            + alpha * family_test_rank[name]
        )
        candidate_raw[key] = raw_valid[name]

    # Within any user this is a fixed mixing coefficient, so it changes the
    # ordering only through the relative incumbent/family evidence. It gives
    # longer slates more complementary-model weight, where the diagnosed
    # ranking gap and number of useful pair comparisons are largest.
    valid_gate = np.clip(
        np.log2(np.maximum(valid_sizes, 2.0)) / 3.0, 0.35, 1.65
    )
    test_gate = np.clip(
        np.log2(np.maximum(test_sizes, 2.0)) / 3.0, 0.35, 1.65
    )

    for base_alpha in (0.04, 0.08, 0.12, 0.18, 0.25, 0.34):
        av = np.clip(base_alpha * valid_gate, 0.0, 0.60)
        at = np.clip(base_alpha * test_gate, 0.0, 0.60)
        key = f"{name}_sizegate_{base_alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - av) * inc_valid_rank
            + av * family_valid_rank[name]
        )
        candidate_test[key] = (
            (1.0 - at) * inc_test_rank
            + at * family_test_rank[name]
        )
        candidate_raw[key] = raw_valid[name]

candidate_metrics = {
    name: evaluate(valid.user_id, valid.y, scores)
    for name, scores in candidate_valid.items()
}
best_name = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"]),
)
best_metrics = candidate_metrics[best_name]
best_valid = candidate_valid[best_name]
best_test = candidate_test[best_name]

correlations = {
    name: float(np.corrcoef(inc_valid_rank, rank_score)[0, 1])
    for name, rank_score in family_valid_rank.items()
}

print("CANDIDATES " + json.dumps({
    name: float(metric["primary"])
    for name, metric in candidate_metrics.items()
}, sort_keys=True))

print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "family_rank_correlations_with_incumbent": correlations,
    "tree_training_rows": int(len(sample_index)),
    "effective_weight_final_3_days": float(
        sample_weight[age_days <= 2].sum() / sample_weight.sum()
    ),
    "valid_mean_slate_size": float(valid_sizes.mean()),
    "test_mean_slate_size": float(test_sizes.mean()),
    "standalone_metrics": {
        name: {
            "primary": float(candidate_metrics[name + "_standalone"]["primary"]),
            "gauc": float(candidate_metrics[name + "_standalone"]["gauc"]),
            "ndcg@5": float(candidate_metrics[name + "_standalone"]["ndcg@5"]),
        }
        for name in family_valid_rank
    },
}, sort_keys=True))

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
    if best_name != "incumbent":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[best_name], dtype=np.float64),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))