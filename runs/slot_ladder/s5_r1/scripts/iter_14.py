import os
import time
import json
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()

# Identity fields are deliberately excluded. These fields describe the item,
# presentation context, or relatively low-cardinality item attributes.
CAT_FIELDS = [
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "video_type",
    "tab",
    "hour",
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
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def date_to_ordinal(date_array):
    values = np.asarray(date_array, dtype=np.int64)
    unique = np.unique(values)
    ordinal_unique = np.array(
        [
            int(
                np.datetime64(str(int(x)), "D").astype(
                    np.int64
                )
            )
            for x in unique
        ],
        dtype=np.int64,
    )
    return ordinal_unique[np.searchsorted(unique, values)]


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort(
        (np.arange(n, dtype=np.int64), scores, user_ids)
    )
    sorted_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    ranks_sorted = (
        np.arange(n, dtype=np.float64)
        - np.repeat(starts, sizes)
    ) / np.repeat(np.maximum(sizes - 1, 1), sizes)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranks_sorted
    return result


def safe_logit(probability):
    p = np.clip(
        np.asarray(probability, dtype=np.float64),
        0.015,
        0.985,
    )
    return np.log(p) - np.log1p(-p)


def make_lgb_matrix(split):
    columns = []

    for field in CAT_FIELDS:
        columns.append(
            np.asarray(split.X[field], dtype=np.float32)
        )

    for field in NUM_FIELDS:
        x = np.asarray(split.num[field], dtype=np.float64)
        finite = np.isfinite(x)
        clean = np.zeros(len(x), dtype=np.float64)
        clean[finite] = np.log1p(np.maximum(x[finite], 0.0))
        columns.append(clean.astype(np.float32))
        columns.append((~finite).astype(np.float32))

    return np.column_stack(columns).astype(
        np.float32, copy=False
    )


class CategoryTrendModel:
    """
    For every low/mid-cardinality content field, estimate:
      * a smoothed category long-view rate
      * a category-specific linear probability trend over train dates

    The slope is ridge-shrunk and is extrapolated using only the target row's
    known calendar date. No validation statistics or labels enter the fit.
    """

    def __init__(self, train):
        y = np.asarray(train.y, dtype=np.float64)
        date_ord = date_to_ordinal(train.date).astype(np.float64)
        self.origin = float(date_ord.min())
        x = date_ord - self.origin

        max_day = float(x.max())
        age = max_day - x
        weights = np.power(0.5, age / 6.0)

        self.global_rate = float(
            np.dot(weights, y) / weights.sum()
        )
        self.tables = {}

        for field in CAT_FIELDS:
            ids = np.asarray(train.X[field], dtype=np.int64)
            cardinality = int(FEATURE_CARDINALITIES[field])

            sw = np.bincount(
                ids, weights=weights, minlength=cardinality
            ).astype(np.float64)
            sy = np.bincount(
                ids, weights=weights * y, minlength=cardinality
            ).astype(np.float64)
            sx = np.bincount(
                ids, weights=weights * x, minlength=cardinality
            ).astype(np.float64)
            sxx = np.bincount(
                ids,
                weights=weights * x * x,
                minlength=cardinality,
            ).astype(np.float64)
            sxy = np.bincount(
                ids,
                weights=weights * x * y,
                minlength=cardinality,
            ).astype(np.float64)

            mean_x = sx / np.maximum(sw, 1e-12)
            mean_y_raw = sy / np.maximum(sw, 1e-12)

            # Beta smoothing stabilizes sparse category intercepts.
            mean_y = (
                sy + 20.0 * self.global_rate
            ) / (sw + 20.0)

            centered_xx = np.maximum(
                sxx - sx * sx / np.maximum(sw, 1e-12),
                0.0,
            )
            centered_xy = (
                sxy
                - sx * sy / np.maximum(sw, 1e-12)
            )

            # Ridge is expressed in weighted day-squared units.
            slope = centered_xy / (centered_xx + 300.0)

            # Categories with almost no support should have no trend.
            slope *= sw / (sw + 30.0)
            slope[sw <= 1.0] = 0.0

            self.tables[field] = {
                "mean_x": mean_x,
                "mean_y": mean_y,
                "slope": slope,
                "support": sw,
                "raw_mean": mean_y_raw,
            }

    def predict(self, split):
        date_x = (
            date_to_ordinal(split.date).astype(np.float64)
            - self.origin
        )
        field_scores = []

        for field in CAT_FIELDS:
            ids = np.asarray(split.X[field], dtype=np.int64)
            table = self.tables[field]
            max_id = len(table["mean_y"]) - 1
            safe_ids = np.minimum(ids, max_id)

            rate = (
                table["mean_y"][safe_ids]
                + table["slope"][safe_ids]
                * (date_x - table["mean_x"][safe_ids])
            )

            unseen = table["support"][safe_ids] <= 0.0
            rate[unseen] = self.global_rate
            field_scores.append(safe_logit(rate))

        stacked = np.column_stack(field_scores)

        # Stronger content fields receive more mass, while the remaining
        # descriptors collectively provide a low-variance drift signal.
        weights = np.ones(len(CAT_FIELDS), dtype=np.float64)
        weights[CAT_FIELDS.index("tag")] = 3.0
        weights[CAT_FIELDS.index("duration_bucket")] = 2.0
        weights[CAT_FIELDS.index("upload_type")] = 1.8
        weights[CAT_FIELDS.index("onehot_feat1")] = 1.5
        weights[CAT_FIELDS.index("onehot_feat3")] = 2.0
        weights[CAT_FIELDS.index("onehot_feat7")] = 1.5
        weights[CAT_FIELDS.index("onehot_feat8")] = 1.5

        return np.average(stacked, axis=1, weights=weights)


train = load("train")
valid = load("valid")

train_ord = date_to_ordinal(train.date)
max_train_day = int(train_ord.max())
train_age = max_train_day - train_ord
recency_weight = np.power(
    0.5, train_age.astype(np.float64) / 6.0
).astype(np.float32)

# Family 1: train-only temporal category trend extrapolation.
trend_model = CategoryTrendModel(train)
valid_trend_raw = trend_model.predict(valid)
valid_trend = within_user_rank(
    valid.user_id, valid_trend_raw
)

# Family 2: recency-weighted content-only gradient boosting.
X_train = make_lgb_matrix(train)
X_valid = make_lgb_matrix(valid)

categorical_indices = list(range(len(CAT_FIELDS)))
dtrain = lgb.Dataset(
    X_train,
    label=np.asarray(train.y, dtype=np.float32),
    weight=recency_weight,
    categorical_feature=categorical_indices,
    free_raw_data=True,
)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "max_depth": 9,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 3.0,
    "max_bin": 127,
    "cat_smooth": 30.0,
    "cat_l2": 15.0,
    "verbosity": -1,
    "verbose": -1,
    "num_threads": min(8, os.cpu_count() or 1),
    "seed": 2026,
    "feature_fraction_seed": 2027,
    "bagging_seed": 2028,
}

gbdt_model = lgb.train(
    params,
    dtrain,
    num_boost_round=190,
)
valid_gbdt_raw = gbdt_model.predict(
    X_valid,
    num_iteration=gbdt_model.current_iteration(),
)
valid_gbdt = within_user_rank(
    valid.user_id, valid_gbdt_raw
)

del X_train, X_valid, dtrain

# Cross-family consensus: ranks are used because only within-user order matters
# and the trend and tree outputs have unrelated calibration.
valid_consensus = (
    0.55 * valid_gbdt + 0.45 * valid_trend
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)
if not os.path.exists(inc_valid_path):
    raise FileNotFoundError(
        "Trusted incumbent validation predictions are missing"
    )

inc_valid = np.asarray(
    np.load(inc_valid_path), dtype=np.float64
)
if len(inc_valid) != len(valid.user_id):
    raise ValueError(
        "Trusted incumbent validation length mismatch"
    )

inc_valid_rank = within_user_rank(
    valid.user_id, inc_valid
)

own_candidates = {
    "temporal_category_trend": valid_trend,
    "recency_content_gbdt": valid_gbdt,
    "trend_gbdt_consensus": valid_consensus,
}

candidate_scores = {}
candidate_metrics = {}
candidate_recipe = {}
candidate_raw = {}

for family, own_score in own_candidates.items():
    standalone_name = family + "_standalone"
    standalone_eval = evaluate(
        valid.user_id, valid.y, own_score
    )
    candidate_scores[standalone_name] = own_score
    candidate_metrics[standalone_name] = float(
        standalone_eval["primary"]
    )
    candidate_recipe[standalone_name] = (
        family,
        1.0,
        True,
    )
    candidate_raw[standalone_name] = own_score

    for own_weight in (
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
    ):
        name = f"{family}_blend_w{own_weight:.2f}"
        blended = (
            own_weight * own_score
            + (1.0 - own_weight) * inc_valid_rank
        )
        result = evaluate(
            valid.user_id, valid.y, blended
        )
        candidate_scores[name] = blended
        candidate_metrics[name] = float(
            result["primary"]
        )
        candidate_recipe[name] = (
            family,
            own_weight,
            False,
        )
        candidate_raw[name] = own_score

winner = max(
    candidate_metrics, key=candidate_metrics.get
)
valid_scores = candidate_scores[winner]
metrics = evaluate(
    valid.user_id, valid.y, valid_scores
)

standalone_results = {
    k: v
    for k, v in candidate_metrics.items()
    if k.endswith("_standalone")
}
best_standalone = max(
    standalone_results,
    key=standalone_results.get,
)

print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner,
            "best_standalone": best_standalone,
            "best_standalone_primary": float(
                standalone_results[best_standalone]
            ),
            "incumbent_primary_check": float(
                evaluate(
                    valid.user_id,
                    valid.y,
                    inc_valid,
                )["primary"]
            ),
            "train_recency_weight_min": float(
                recency_weight.min()
            ),
        },
        separators=(",", ":"),
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        {
            k: float(v)
            for k, v in candidate_metrics.items()
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

family, own_weight, is_standalone = candidate_recipe[
    winner
]
if out_dir and not is_standalone:
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(
            candidate_raw[winner], dtype=np.float64
        ),
    )

# Score test with exactly the selected family and blend recipe.
test = load("test")

test_trend_raw = trend_model.predict(test)
test_trend = within_user_rank(
    test.user_id, test_trend_raw
)

X_test = make_lgb_matrix(test)
test_gbdt_raw = gbdt_model.predict(
    X_test,
    num_iteration=gbdt_model.current_iteration(),
)
test_gbdt = within_user_rank(
    test.user_id, test_gbdt_raw
)

test_candidates = {
    "temporal_category_trend": test_trend,
    "recency_content_gbdt": test_gbdt,
    "trend_gbdt_consensus": (
        0.55 * test_gbdt + 0.45 * test_trend
    ),
}
own_test = test_candidates[family]

if is_standalone:
    test_scores = own_test
else:
    if not os.path.exists(inc_test_path):
        raise FileNotFoundError(
            "Trusted incumbent test predictions are missing"
        )
    inc_test = np.asarray(
        np.load(inc_test_path), dtype=np.float64
    )
    if len(inc_test) != len(test.user_id):
        raise ValueError(
            "Trusted incumbent test length mismatch"
        )
    inc_test_rank = within_user_rank(
        test.user_id, inc_test
    )
    test_scores = (
        own_weight * own_test
        + (1.0 - own_weight) * inc_test_rank
    )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
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