import os
import time
import json
import gc
import warnings

import numpy as np
import lightgbm as lgb
from scipy import sparse
from sklearn.linear_model import SGDClassifier

from pipeline.data import load
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")
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
    "user_active_degree",
    "register_days_bucket",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, scores, user_ids))
    su = user_ids[order]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    ends = np.r_[starts[1:], n]
    sizes = ends - starts
    ranked = (
        np.arange(n, dtype=np.float64) - np.repeat(starts, sizes)
    ) / np.repeat(np.maximum(sizes - 1, 1), sizes)
    out = np.empty(n, dtype=np.float64)
    out[order] = ranked
    return out


def temporal_features(split):
    user = np.asarray(split.user_id, dtype=np.int64)
    tm = np.asarray(split.time_ms, dtype=np.int64)
    dates = np.asarray(split.date, dtype=np.int64)
    n = len(user)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, tm, user))
    su = user[order]
    st = tm[order]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    pos_sorted = np.arange(n, dtype=np.int64) - np.repeat(starts, sizes)
    pos = np.empty(n, dtype=np.int64)
    total = np.empty(n, dtype=np.int64)
    pos[order] = pos_sorted
    total[order] = np.repeat(sizes, sizes)

    previous_gap_sorted = np.zeros(n, dtype=np.float64)
    next_gap_sorted = np.zeros(n, dtype=np.float64)
    previous_gap_sorted[1:] = np.maximum(st[1:] - st[:-1], 0) / 1000.0
    next_gap_sorted[:-1] = np.maximum(st[1:] - st[:-1], 0) / 1000.0
    previous_gap_sorted[starts] = 0.0
    next_gap_sorted[ends - 1] = 0.0

    previous_gap = np.empty(n, dtype=np.float64)
    next_gap = np.empty(n, dtype=np.float64)
    previous_gap[order] = previous_gap_sorted
    next_gap[order] = next_gap_sorted

    _, day_code = np.unique(dates, return_inverse=True)
    day_scale = int(day_code.max()) + 1
    user_day = user * day_scale + day_code

    day_order = np.lexsort((rows, tm, user_day))
    sd = user_day[day_order]
    day_starts = np.flatnonzero(np.r_[True, sd[1:] != sd[:-1]])
    day_ends = np.r_[day_starts[1:], n]
    day_sizes = day_ends - day_starts
    day_pos_sorted = (
        np.arange(n, dtype=np.int64) - np.repeat(day_starts, day_sizes)
    )
    day_pos = np.empty(n, dtype=np.int64)
    day_total = np.empty(n, dtype=np.int64)
    day_pos[day_order] = day_pos_sorted
    day_total[day_order] = np.repeat(day_sizes, day_sizes)

    denom = np.maximum(total - 1, 1)
    day_denom = np.maximum(day_total - 1, 1)

    return np.column_stack([
        np.log1p(pos),
        pos / denom,
        (total - 1 - pos) / denom,
        np.log1p(total),
        np.log1p(day_pos),
        day_pos / day_denom,
        np.log1p(day_total),
        np.log1p(np.clip(previous_gap, 0, 86400)),
        np.log1p(np.clip(next_gap, 0, 86400)),
    ]).astype(np.float32)


def fit_numeric_scaler(train):
    center = {}
    scale = {}
    for name in NUM_FIELDS:
        x = np.asarray(train.num[name], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.sign(x) * np.log1p(np.abs(x))
        center[name] = float(np.median(x))
        s = float(np.std(x))
        scale[name] = max(s, 1e-3)
    return center, scale


def numeric_matrix(split, center, scale):
    columns = []
    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float64)
        missing = ~np.isfinite(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.sign(x) * np.log1p(np.abs(x))
        z = np.clip((x - center[name]) / scale[name], -8.0, 8.0)
        columns.append(z.astype(np.float32))
        columns.append(missing.astype(np.float32))
    return np.column_stack(columns).astype(np.float32)


def dense_matrix(split, temporal, center, scale):
    cats = np.column_stack([
        np.asarray(split.X[name], dtype=np.float32) for name in CAT_FIELDS
    ])
    nums = numeric_matrix(split, center, scale)
    return np.ascontiguousarray(
        np.column_stack([cats, nums, temporal]), dtype=np.float32
    )


def sparse_linear_matrix(split, temporal, center, scale, offsets, total_card):
    n = len(split.user_id)
    row = np.tile(np.arange(n, dtype=np.int32), len(CAT_FIELDS))
    col_parts = []
    for name, offset in zip(CAT_FIELDS, offsets):
        ids = np.asarray(split.X[name], dtype=np.int64)
        col_parts.append((ids + offset).astype(np.int32))
    col = np.concatenate(col_parts)
    data = np.ones(len(col), dtype=np.float32)
    categorical = sparse.csr_matrix(
        (data, (row, col)), shape=(n, total_card), dtype=np.float32
    )
    continuous = sparse.csr_matrix(
        np.column_stack([
            numeric_matrix(split, center, scale),
            temporal,
        ]).astype(np.float32)
    )
    return sparse.hstack([categorical, continuous], format="csr")


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.01, 0.99)
    return np.log(p) - np.log1p(-p)


class WeightedAdditiveBayes:
    def __init__(self, train, y, weights, numeric_train):
        self.global_rate = float(np.dot(weights, y) / weights.sum())
        self.global_logit = float(safe_logit(self.global_rate))
        self.cat_tables = []
        self.cat_strength = []

        for name in CAT_FIELDS:
            ids = np.asarray(train.X[name], dtype=np.int64)
            card = int(ids.max()) + 1
            count = np.bincount(
                ids, weights=weights, minlength=card
            ).astype(np.float64)
            positive = np.bincount(
                ids, weights=weights * y, minlength=card
            ).astype(np.float64)

            if name == "user_id":
                alpha = 35.0
                strength = 0.10
            elif name in ("video_id", "author_id"):
                alpha = 45.0
                strength = 0.85
            elif name in ("tag", "onehot_feat3"):
                alpha = 70.0
                strength = 0.55
            else:
                alpha = 100.0
                strength = 0.30

            rate = (
                positive + alpha * self.global_rate
            ) / (count + alpha)
            self.cat_tables.append(rate)
            self.cat_strength.append(strength)

        self.numeric_edges = []
        self.numeric_tables = []
        for j in range(numeric_train.shape[1]):
            x = np.asarray(numeric_train[:, j], dtype=np.float64)
            edges = np.unique(np.quantile(x, np.linspace(0, 1, 33)))
            if len(edges) <= 2:
                edges = np.array([-np.inf, np.inf])
            else:
                edges[0] = -np.inf
                edges[-1] = np.inf
            bins = np.searchsorted(edges[1:-1], x, side="right")
            card = len(edges) - 1
            count = np.bincount(
                bins, weights=weights, minlength=card
            ).astype(np.float64)
            positive = np.bincount(
                bins, weights=weights * y, minlength=card
            ).astype(np.float64)
            alpha = 150.0
            rate = (
                positive + alpha * self.global_rate
            ) / (count + alpha)
            self.numeric_edges.append(edges)
            self.numeric_tables.append(rate)

    def predict(self, split, numeric_values):
        result = np.zeros(len(split.user_id), dtype=np.float64)
        for name, table, strength in zip(
            CAT_FIELDS, self.cat_tables, self.cat_strength
        ):
            ids = np.asarray(split.X[name], dtype=np.int64)
            valid = ids < len(table)
            values = np.full(len(ids), self.global_rate, dtype=np.float64)
            values[valid] = table[ids[valid]]
            result += strength * (safe_logit(values) - self.global_logit)

        for j, (edges, table) in enumerate(
            zip(self.numeric_edges, self.numeric_tables)
        ):
            bins = np.searchsorted(
                edges[1:-1], numeric_values[:, j], side="right"
            )
            result += 0.22 * (
                safe_logit(table[np.minimum(bins, len(table) - 1)])
                - self.global_logit
            )
        return result


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)

center, scale = fit_numeric_scaler(train)
temporal_train = temporal_features(train)
temporal_valid = temporal_features(valid)
numeric_train = numeric_matrix(train, center, scale)
numeric_valid = numeric_matrix(valid, center, scale)

train_users = np.asarray(train.user_id, dtype=np.int64)
user_counts = np.bincount(train_users)
row_user_counts = user_counts[train_users].astype(np.float64)

age = (
    int(np.max(train.date)) - np.asarray(train.date, dtype=np.int64)
).astype(np.float64)
recency = np.power(0.5, age / 5.0)
user_balance = np.power(np.maximum(row_user_counts, 1.0), -0.45)
weights = recency * user_balance
weights *= len(weights) / weights.sum()
weights = weights.astype(np.float32)

# Family 1: weighted additive empirical-Bayes GAM.
bayes_model = WeightedAdditiveBayes(
    train, y_train.astype(np.float64), weights.astype(np.float64),
    numeric_train
)
bayes_valid_raw = bayes_model.predict(valid, numeric_valid)

# Family 2: weighted sparse linear classifier over explicit one-hot fields.
cards = [
    max(
        int(np.max(train.X[name])),
        int(np.max(valid.X[name])),
    ) + 1
    for name in CAT_FIELDS
]
offsets = np.cumsum([0] + cards[:-1]).astype(np.int64)
total_card = int(sum(cards))

linear_train = sparse_linear_matrix(
    train, temporal_train, center, scale, offsets, total_card
)
linear_model = SGDClassifier(
    loss="log_loss",
    penalty="elasticnet",
    alpha=2.0e-6,
    l1_ratio=0.03,
    max_iter=9,
    tol=None,
    average=True,
    random_state=1731,
    learning_rate="optimal",
)
linear_model.fit(linear_train, y_train, sample_weight=weights)
del linear_train
gc.collect()

linear_valid = sparse_linear_matrix(
    valid, temporal_valid, center, scale, offsets, total_card
)
linear_valid_raw = linear_model.decision_function(linear_valid)
del linear_valid
gc.collect()

# Family 3: weighted nonlinear boosted interactions.
X_train = dense_matrix(
    train, temporal_train, center, scale
)
X_valid = dense_matrix(
    valid, temporal_valid, center, scale
)
categorical_indices = list(range(len(CAT_FIELDS)))

gbdt_data = lgb.Dataset(
    X_train,
    label=y_train,
    weight=weights,
    categorical_feature=categorical_indices,
    free_raw_data=True,
)
gbdt_params = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.05,
    "num_leaves": 55,
    "min_data_in_leaf": 320,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.88,
    "bagging_freq": 1,
    "lambda_l1": 0.15,
    "lambda_l2": 4.0,
    "max_bin": 127,
    "num_threads": -1,
    "seed": 2997,
    "verbose": -1,
}
gbdt_model = lgb.train(
    gbdt_params, gbdt_data, num_boost_round=300
)
gbdt_valid_raw = gbdt_model.predict(X_valid)

del X_train, X_valid, gbdt_data
gc.collect()

own_valid = {
    "user_balanced_bayes": within_user_rank(
        valid.user_id, bayes_valid_raw
    ),
    "user_balanced_sparse_linear": within_user_rank(
        valid.user_id, linear_valid_raw
    ),
    "user_balanced_gbdt": within_user_rank(
        valid.user_id, gbdt_valid_raw
    ),
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are missing")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if len(inc_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation length mismatch")
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

candidate_scores = {"incumbent": inc_valid_rank}
candidate_primary = {
    "incumbent": float(
        evaluate(valid.user_id, valid.y, inc_valid_rank)["primary"]
    )
}
recipes = {"incumbent": ("incumbent", 0.0)}

for family, own_rank in own_valid.items():
    candidate_scores[family] = own_rank
    candidate_primary[family] = float(
        evaluate(valid.user_id, valid.y, own_rank)["primary"]
    )
    recipes[family] = (family, 1.0)

    for alpha in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55):
        name = f"{family}_blend_{alpha:.2f}"
        score = alpha * own_rank + (1.0 - alpha) * inc_valid_rank
        candidate_scores[name] = score
        candidate_primary[name] = float(
            evaluate(valid.user_id, valid.y, score)["primary"]
        )
        recipes[name] = (family, alpha)

winner = max(candidate_primary, key=candidate_primary.get)
valid_scores = candidate_scores[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)
winner_family, winner_alpha = recipes[winner]

standalone = {
    name: candidate_primary[name] for name in own_valid
}
print("FINDINGS " + json.dumps({
    "winner": winner,
    "standalone": standalone,
    "incumbent": candidate_primary["incumbent"],
    "effective_weight_recent_oldest_ratio": float(
        np.mean(weights[age == np.min(age)])
        / max(np.mean(weights[age == np.max(age)]), 1e-12)
    ),
    "weight_user_count_correlation": float(
        np.corrcoef(weights, row_user_counts)[0, 1]
    ),
}, separators=(",", ":")))
print("CANDIDATES " + json.dumps(
    {k: float(v) for k, v in candidate_primary.items()},
    sort_keys=True,
    separators=(",", ":"),
))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if winner_family in own_valid:
        raw_to_save = own_valid[winner_family]
    else:
        raw_to_save = own_valid[
            max(own_valid, key=lambda k: candidate_primary[k])
        ]
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(raw_to_save, dtype=np.float64),
    )

# Required test scoring, using the already fitted train-only models.
test = load("test")
temporal_test = temporal_features(test)
numeric_test = numeric_matrix(test, center, scale)

bayes_test_raw = bayes_model.predict(test, numeric_test)

linear_test = sparse_linear_matrix(
    test, temporal_test, center, scale, offsets, total_card
)
linear_test_raw = linear_model.decision_function(linear_test)
del linear_test
gc.collect()

X_test = dense_matrix(test, temporal_test, center, scale)
gbdt_test_raw = gbdt_model.predict(X_test)
del X_test
gc.collect()

own_test = {
    "user_balanced_bayes": within_user_rank(
        test.user_id, bayes_test_raw
    ),
    "user_balanced_sparse_linear": within_user_rank(
        test.user_id, linear_test_raw
    ),
    "user_balanced_gbdt": within_user_rank(
        test.user_id, gbdt_test_raw
    ),
}

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test) != len(test.user_id):
    raise ValueError("Incumbent test length mismatch")
inc_test_rank = within_user_rank(test.user_id, inc_test)

if winner_family == "incumbent":
    test_scores = inc_test_rank
else:
    test_scores = (
        winner_alpha * own_test[winner_family]
        + (1.0 - winner_alpha) * inc_test_rank
    )

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}, separators=(",", ":")))