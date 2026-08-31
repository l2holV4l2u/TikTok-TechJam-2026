import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb
import scipy.sparse as sp

from sklearn.linear_model import SGDClassifier

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "upload_type",
    "onehot_feat3",
    "hour",
]

LINEAR_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "upload_type",
    "onehot_feat3",
    "hour",
]

PAIR_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    rows = np.arange(len(scores), dtype=np.int64)
    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], len(scores)]
    sizes = ends - starts

    rank_sorted = (
        np.arange(len(scores), dtype=np.float64)
        - np.repeat(starts, sizes)
    ) / np.repeat(np.maximum(sizes - 1, 1), sizes)

    result = np.empty(len(scores), dtype=np.float64)
    result[order] = rank_sorted
    return result


def make_tree_matrix(split_name, split):
    columns = [
        np.asarray(split.X[field], dtype=np.float32)
        for field in CAT_FIELDS
    ]

    for field in NUM_FIELDS:
        value = np.asarray(split.num[field], dtype=np.float64)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(value, 0.0)).astype(np.float32))

    for key in ("video_id", "author_id"):
        histories = historical_features(split_name, key=key)
        for name in sorted(histories):
            value = np.asarray(histories[name], dtype=np.float32)
            value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            columns.append(value)

    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


HASH_DIM = 1 << 20
RAW_OFFSET = HASH_DIM
RAW_CARDINALITY = 50000
LINEAR_DIM = RAW_OFFSET + len(LINEAR_FIELDS) * RAW_CARDINALITY


def make_hashed_linear_matrix(split):
    n = len(split.user_id)
    user = np.asarray(split.user_id, dtype=np.uint64)
    n_raw = len(LINEAR_FIELDS)
    n_pair = len(PAIR_FIELDS)
    nnz_per_row = n_raw + n_pair

    indices = np.empty((n, nnz_per_row), dtype=np.int32)

    for j, field in enumerate(LINEAR_FIELDS):
        value = np.asarray(split.X[field], dtype=np.int64)
        value = np.minimum(value, RAW_CARDINALITY - 1)
        indices[:, j] = (
            RAW_OFFSET + j * RAW_CARDINALITY + value
        ).astype(np.int32)

    mask = np.uint64(HASH_DIM - 1)
    constants = [
        np.uint64(0x9E3779B185EBCA87),
        np.uint64(0xC2B2AE3D27D4EB4F),
        np.uint64(0x165667B19E3779F9),
        np.uint64(0x85EBCA77C2B2AE63),
        np.uint64(0x27D4EB2F165667C5),
    ]

    for j, (field, constant) in enumerate(zip(PAIR_FIELDS, constants)):
        value = np.asarray(split.X[field], dtype=np.uint64)
        mixed = (
            user * constant
            + value * np.uint64(0xD6E8FEB86659FD93)
            + np.uint64(j + 1) * np.uint64(0x94D049BB133111EB)
        )
        mixed ^= mixed >> np.uint64(31)
        indices[:, n_raw + j] = (mixed & mask).astype(np.int32)

    indices = indices.reshape(-1)
    indptr = np.arange(
        0, (n + 1) * nnz_per_row, nnz_per_row, dtype=np.int64
    )
    data = np.ones(len(indices), dtype=np.float32)

    return sp.csr_matrix(
        (data, indices, indptr),
        shape=(n, LINEAR_DIM),
        dtype=np.float32,
    )


class CategoricalLikelihoodRatio:
    def __init__(self, train, fields):
        self.fields = list(fields)
        y = np.asarray(train.y, dtype=np.float64)
        self.global_rate = float(np.mean(y))
        global_logit = float(safe_logit(self.global_rate))
        self.tables = []

        for field in self.fields:
            value = np.asarray(train.X[field], dtype=np.int64)
            card = int(value.max()) + 1
            count = np.bincount(value, minlength=card).astype(np.float64)
            positive = np.bincount(
                value, weights=y, minlength=card
            ).astype(np.float64)

            # Strong field-specific shrinkage prevents rare identity values
            # from dominating the additive generative evidence.
            if field in ("video_id", "author_id", "onehot_feat3"):
                alpha = 80.0
            else:
                alpha = 250.0

            rate = (
                positive + alpha * self.global_rate
            ) / (count + alpha)
            component = safe_logit(rate) - global_logit
            self.tables.append(np.asarray(component, dtype=np.float64))

    def predict(self, split):
        result = np.zeros(len(split.user_id), dtype=np.float64)
        scale = 1.0 / np.sqrt(max(len(self.fields), 1))

        for field, table in zip(self.fields, self.tables):
            value = np.asarray(split.X[field], dtype=np.int64)
            known = value < len(table)
            component = np.zeros(len(value), dtype=np.float64)
            component[known] = table[value[known]]
            result += scale * component

        return result


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)

# ----------------------------------------------------------------------
# Family 1: random-forest-style bagging of categorical decision trees.
# Unlike boosting, trees are independently randomized and averaged.
# ----------------------------------------------------------------------
X_train_tree = make_tree_matrix("train", train)
X_valid_tree = make_tree_matrix("valid", valid)

rf_params = {
    "objective": "binary",
    "metric": "None",
    "boosting_type": "rf",
    "learning_rate": 1.0,
    "num_leaves": 63,
    "max_depth": 12,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.62,
    "bagging_fraction": 0.62,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 5.0,
    "max_bin": 127,
    "num_threads": -1,
    "seed": 1709,
    "verbose": -1,
}

rf_dataset = lgb.Dataset(
    X_train_tree,
    label=y_train,
    categorical_feature=list(range(len(CAT_FIELDS))),
    free_raw_data=True,
)
rf_model = lgb.train(
    rf_params,
    rf_dataset,
    num_boost_round=220,
)
rf_valid = rf_model.predict(X_valid_tree)
del X_train_tree, X_valid_tree, rf_dataset
gc.collect()

# ----------------------------------------------------------------------
# Family 2: averaged online logistic regression over raw categorical
# indicators and hashed user-by-content crosses.
# ----------------------------------------------------------------------
X_train_linear = make_hashed_linear_matrix(train)
linear_model = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=2.0e-7,
    fit_intercept=True,
    max_iter=6,
    tol=None,
    shuffle=True,
    average=True,
    random_state=2718,
    n_jobs=-1,
)
linear_model.fit(X_train_linear, y_train.astype(np.int8))
del X_train_linear
gc.collect()

X_valid_linear = make_hashed_linear_matrix(valid)
linear_valid = linear_model.decision_function(X_valid_linear)
del X_valid_linear
gc.collect()

# ----------------------------------------------------------------------
# Family 3: generative additive categorical likelihood-ratio estimator.
# ----------------------------------------------------------------------
nb_fields = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "tab",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat7",
    "music_type",
]
nb_model = CategoricalLikelihoodRatio(train, nb_fields)
nb_valid = nb_model.predict(valid)

own_valid = {
    "bagged_categorical_forest": within_user_rank(
        valid.user_id, rf_valid
    ),
    "hashed_crossed_linear": within_user_rank(
        valid.user_id, linear_valid
    ),
    "categorical_likelihood_ratio": within_user_rank(
        valid.user_id, nb_valid
    ),
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path):
    raise FileNotFoundError("Trusted incumbent validation predictions missing")
if not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent test predictions missing")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if len(inc_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation prediction length mismatch")
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

candidate_scores = {"incumbent": inc_valid_rank}
candidate_primary = {
    "incumbent": float(
        evaluate(valid.user_id, valid.y, inc_valid_rank)["primary"]
    )
}
candidate_recipe = {
    "incumbent": ("incumbent", "", 0.0)
}

for family, own_rank in own_valid.items():
    standalone_name = family + "_standalone"
    candidate_scores[standalone_name] = own_rank
    candidate_primary[standalone_name] = float(
        evaluate(valid.user_id, valid.y, own_rank)["primary"]
    )
    candidate_recipe[standalone_name] = (
        "standalone", family, 1.0
    )

    for weight in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        name = f"{family}_blend_{weight:.2f}"
        score = (
            weight * own_rank
            + (1.0 - weight) * inc_valid_rank
        )
        candidate_scores[name] = score
        candidate_primary[name] = float(
            evaluate(valid.user_id, valid.y, score)["primary"]
        )
        candidate_recipe[name] = ("blend", family, weight)

winner = max(candidate_primary, key=candidate_primary.get)
valid_scores = candidate_scores[winner]
recipe_type, winner_family, winner_weight = candidate_recipe[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

standalone_scores = {
    family: float(
        evaluate(valid.user_id, valid.y, score)["primary"]
    )
    for family, score in own_valid.items()
}
best_own_family = max(standalone_scores, key=standalone_scores.get)

print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner,
            "standalone": standalone_scores,
            "incumbent": candidate_primary["incumbent"],
            "winner_weight": float(winner_weight),
            "best_own_family": best_own_family,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_primary.items()},
        sort_keys=True,
        separators=(",", ":"),
    )
)

# Build each family's test prediction once, then apply exactly the
# validation-selected family and blend weight.
test = load("test")

X_test_tree = make_tree_matrix("test", test)
rf_test = rf_model.predict(X_test_tree)
del X_test_tree
gc.collect()

X_test_linear = make_hashed_linear_matrix(test)
linear_test = linear_model.decision_function(X_test_linear)
del X_test_linear
gc.collect()

nb_test = nb_model.predict(test)

own_test = {
    "bagged_categorical_forest": within_user_rank(
        test.user_id, rf_test
    ),
    "hashed_crossed_linear": within_user_rank(
        test.user_id, linear_test
    ),
    "categorical_likelihood_ratio": within_user_rank(
        test.user_id, nb_test
    ),
}

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test) != len(test.user_id):
    raise ValueError("Incumbent test prediction length mismatch")
inc_test_rank = within_user_rank(test.user_id, inc_test)

if recipe_type == "incumbent":
    test_scores = inc_test_rank
    raw_valid_scores = own_valid[best_own_family]
elif recipe_type == "standalone":
    test_scores = own_test[winner_family]
    raw_valid_scores = own_valid[winner_family]
else:
    test_scores = (
        winner_weight * own_test[winner_family]
        + (1.0 - winner_weight) * inc_test_rank
    )
    raw_valid_scores = own_valid[winner_family]

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if recipe_type in ("incumbent", "blend"):
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
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
        },
        separators=(",", ":"),
    )
)