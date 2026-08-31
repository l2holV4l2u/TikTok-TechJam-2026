import os
import time
import json
import numpy as np
import lightgbm as lgb

from pipeline.data import load
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


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.01, 0.99)
    return np.log(p) - np.log1p(-p)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    ranks_sorted = (
        np.arange(n, dtype=np.float64) - np.repeat(starts, sizes)
    ) / np.repeat(np.maximum(sizes - 1, 1), sizes)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranks_sorted
    return result


def ordered_group_features(primary_key, time_ms, secondary_key=None):
    primary_key = np.asarray(primary_key, dtype=np.int64)
    time_ms = np.asarray(time_ms, dtype=np.int64)
    n = len(primary_key)
    rows = np.arange(n, dtype=np.int64)

    if secondary_key is None:
        group_key = primary_key
    else:
        secondary_key = np.asarray(secondary_key, dtype=np.int64)
        sec_scale = int(secondary_key.max()) + 1
        group_key = primary_key * sec_scale + secondary_key

    order = np.lexsort((rows, time_ms, group_key))
    sorted_key = group_key[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_key[1:] != sorted_key[:-1]]
    )
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    ordinal_sorted = np.arange(n, dtype=np.int64) - np.repeat(starts, sizes)
    total_sorted = np.repeat(sizes, sizes)

    ordinal = np.empty(n, dtype=np.int64)
    total = np.empty(n, dtype=np.int64)
    ordinal[order] = ordinal_sorted
    total[order] = total_sorted
    return ordinal, total


def make_temporal_features(split):
    user = np.asarray(split.user_id, dtype=np.int64)
    tm = np.asarray(split.time_ms, dtype=np.int64)
    dates = np.asarray(split.date, dtype=np.int64)
    rows = np.arange(len(user), dtype=np.int64)

    order = np.lexsort((rows, tm, user))
    su = user[order]
    st = tm[order]

    starts = np.r_[True, su[1:] != su[:-1]]
    ends = np.r_[su[:-1] != su[1:], True]

    gap_prev_sorted = np.zeros(len(user), dtype=np.float64)
    gap_prev_sorted[1:] = (st[1:] - st[:-1]) / 1000.0
    gap_prev_sorted[starts] = 0.0

    gap_next_sorted = np.zeros(len(user), dtype=np.float64)
    gap_next_sorted[:-1] = (st[1:] - st[:-1]) / 1000.0
    gap_next_sorted[ends] = 0.0

    gap_prev = np.empty(len(user), dtype=np.float64)
    gap_next = np.empty(len(user), dtype=np.float64)
    gap_prev[order] = gap_prev_sorted
    gap_next[order] = gap_next_sorted

    user_pos, user_total = ordered_group_features(user, tm)

    _, date_inverse = np.unique(dates, return_inverse=True)
    user_day_key = user * (int(date_inverse.max()) + 1) + date_inverse
    day_pos, day_total = ordered_group_features(user_day_key, tm)

    video_pos, video_total = ordered_group_features(
        user, tm, np.asarray(split.video_id, dtype=np.int64)
    )
    author_pos, author_total = ordered_group_features(
        user, tm, np.asarray(split.X["author_id"], dtype=np.int64)
    )

    pair_dtype = np.dtype([("u", np.int64), ("t", np.int64)])
    batch_pairs = np.empty(len(user), dtype=pair_dtype)
    batch_pairs["u"] = user
    batch_pairs["t"] = tm
    _, batch_inverse, batch_counts = np.unique(
        batch_pairs, return_inverse=True, return_counts=True
    )
    batch_size = batch_counts[batch_inverse].astype(np.float64)

    user_denom = np.maximum(user_total - 1, 1)
    day_denom = np.maximum(day_total - 1, 1)

    derived = np.column_stack(
        [
            user_pos.astype(np.float64),
            user_pos / user_denom,
            (user_total - 1 - user_pos) / user_denom,
            np.log1p(user_total),
            day_pos.astype(np.float64),
            day_pos / day_denom,
            np.log1p(day_total),
            np.log1p(np.clip(gap_prev, 0.0, 86400.0)),
            np.log1p(np.clip(gap_next, 0.0, 86400.0)),
            np.log1p(batch_size),
            np.log1p(video_pos),
            np.log1p(video_total),
            np.log1p(author_pos),
            np.log1p(author_total),
        ]
    ).astype(np.float32)

    cats = np.column_stack(
        [np.asarray(split.X[f], dtype=np.float32) for f in CAT_FIELDS]
    )

    return np.ascontiguousarray(
        np.column_stack([cats, derived]), dtype=np.float32
    ), {
        "user_pos": user_pos,
        "user_total": user_total,
        "day_pos": day_pos,
        "day_total": day_total,
        "video_pos": video_pos,
        "video_total": video_total,
        "author_pos": author_pos,
        "author_total": author_total,
        "gap_prev": gap_prev,
        "gap_next": gap_next,
        "batch_size": batch_size,
    }


class HazardGAM:
    def __init__(self, train, temporal):
        y = np.asarray(train.y, dtype=np.float64)
        age = int(np.max(train.date)) - np.asarray(train.date, dtype=np.int64)
        weights = np.power(0.5, age.astype(np.float64) / 4.0)

        self.global_rate = float(np.dot(weights, y) / weights.sum())
        self.tables = []

        definitions = [
            np.minimum(temporal["user_pos"], 63),
            np.minimum(temporal["day_pos"], 31),
            np.minimum(temporal["video_pos"], 7),
            np.minimum(temporal["author_pos"], 15),
            np.minimum(temporal["video_total"], 15),
            np.minimum(temporal["author_total"], 31),
            np.minimum(
                np.floor(np.log1p(temporal["gap_prev"]) * 3.0).astype(np.int64),
                63,
            ),
            np.minimum(
                np.floor(np.log1p(temporal["gap_next"]) * 3.0).astype(np.int64),
                63,
            ),
            np.minimum(temporal["batch_size"].astype(np.int64), 31),
        ]

        self.kinds = [
            "user_pos",
            "day_pos",
            "video_pos",
            "author_pos",
            "video_total",
            "author_total",
            "gap_prev",
            "gap_next",
            "batch_size",
        ]

        for values in definitions:
            values = np.asarray(values, dtype=np.int64)
            card = int(values.max()) + 1
            count = np.bincount(
                values, weights=weights, minlength=card
            ).astype(np.float64)
            positive = np.bincount(
                values, weights=weights * y, minlength=card
            ).astype(np.float64)
            alpha = 25.0
            rate = (
                positive + alpha * self.global_rate
            ) / (count + alpha)
            self.tables.append(rate)

    def _values(self, temporal, kind):
        if kind == "user_pos":
            return np.minimum(temporal["user_pos"], 63)
        if kind == "day_pos":
            return np.minimum(temporal["day_pos"], 31)
        if kind == "video_pos":
            return np.minimum(temporal["video_pos"], 7)
        if kind == "author_pos":
            return np.minimum(temporal["author_pos"], 15)
        if kind == "video_total":
            return np.minimum(temporal["video_total"], 15)
        if kind == "author_total":
            return np.minimum(temporal["author_total"], 31)
        if kind == "gap_prev":
            return np.minimum(
                np.floor(np.log1p(temporal["gap_prev"]) * 3.0).astype(np.int64),
                63,
            )
        if kind == "gap_next":
            return np.minimum(
                np.floor(np.log1p(temporal["gap_next"]) * 3.0).astype(np.int64),
                63,
            )
        return np.minimum(temporal["batch_size"].astype(np.int64), 31)

    def predict(self, temporal):
        components = []
        global_logit = safe_logit(self.global_rate)
        for table, kind in zip(self.tables, self.kinds):
            values = self._values(temporal, kind)
            values = np.minimum(values, len(table) - 1)
            components.append(safe_logit(table[values]) - global_logit)

        components = np.column_stack(components)
        return (
            1.00 * components[:, 0]
            + 0.80 * components[:, 1]
            + 0.65 * components[:, 2]
            + 0.55 * components[:, 3]
            + 0.35 * components[:, 4]
            + 0.30 * components[:, 5]
            + 0.35 * components[:, 6]
            + 0.20 * components[:, 7]
            + 0.25 * components[:, 8]
        )


train = load("train")
valid = load("valid")

X_train, temporal_train = make_temporal_features(train)
X_valid, temporal_valid = make_temporal_features(valid)
y_train = np.asarray(train.y, dtype=np.float32)

age_days = int(np.max(train.date)) - np.asarray(train.date, dtype=np.int64)
recent_weights = np.power(0.5, age_days.astype(np.float64) / 4.0).astype(
    np.float32
)

categorical_indices = list(range(len(CAT_FIELDS)))

binary_params = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.055,
    "num_leaves": 47,
    "min_data_in_leaf": 350,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 3.0,
    "max_bin": 127,
    "num_threads": -1,
    "seed": 731,
    "verbose": -1,
}

binary_data = lgb.Dataset(
    X_train,
    label=y_train,
    weight=recent_weights,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
binary_model = lgb.train(
    binary_params,
    binary_data,
    num_boost_round=280,
)
binary_valid = binary_model.predict(X_valid)

rank_order = np.argsort(
    np.asarray(train.user_id, dtype=np.int64), kind="stable"
)
rank_users = np.asarray(train.user_id, dtype=np.int64)[rank_order]
rank_starts = np.flatnonzero(
    np.r_[True, rank_users[1:] != rank_users[:-1]]
)
rank_groups = np.diff(np.r_[rank_starts, len(rank_users)]).astype(np.int32)

rank_params = {
    "objective": "lambdarank",
    "metric": "None",
    "learning_rate": 0.045,
    "num_leaves": 39,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 4.0,
    "max_bin": 127,
    "lambdarank_truncation_level": 5,
    "label_gain": [0, 1],
    "num_threads": -1,
    "seed": 947,
    "verbose": -1,
}

rank_data = lgb.Dataset(
    X_train[rank_order],
    label=y_train[rank_order],
    weight=recent_weights[rank_order],
    group=rank_groups,
    categorical_feature=categorical_indices,
    free_raw_data=True,
)
rank_model = lgb.train(
    rank_params,
    rank_data,
    num_boost_round=230,
)
rank_valid = rank_model.predict(X_valid)

hazard_model = HazardGAM(train, temporal_train)
hazard_valid = hazard_model.predict(temporal_valid)

own_valid = {
    "position_hazard_gam": within_user_rank(valid.user_id, hazard_valid),
    "position_binary_boost": within_user_rank(valid.user_id, binary_valid),
    "position_lambdarank": within_user_rank(valid.user_id, rank_valid),
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path):
    raise FileNotFoundError("Trusted incumbent validation predictions missing")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if len(inc_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation length mismatch")
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

candidate_scores = {}
candidate_primary = {}
candidate_recipe = {}

inc_metric = evaluate(valid.user_id, valid.y, inc_valid_rank)
candidate_scores["incumbent_rank"] = inc_valid_rank
candidate_primary["incumbent_rank"] = float(inc_metric["primary"])
candidate_recipe["incumbent_rank"] = ("incumbent", "", 0.0)

for family, own_rank in own_valid.items():
    metric = evaluate(valid.user_id, valid.y, own_rank)
    standalone = family + "_standalone"
    candidate_scores[standalone] = own_rank
    candidate_primary[standalone] = float(metric["primary"])
    candidate_recipe[standalone] = ("standalone", family, 1.0)

    for weight in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        name = f"{family}_blend_{weight:.2f}"
        score = weight * own_rank + (1.0 - weight) * inc_valid_rank
        metric = evaluate(valid.user_id, valid.y, score)
        candidate_scores[name] = score
        candidate_primary[name] = float(metric["primary"])
        candidate_recipe[name] = ("blend", family, weight)

winner = max(candidate_primary, key=candidate_primary.get)
valid_scores = candidate_scores[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)
recipe_type, winner_family, winner_weight = candidate_recipe[winner]

standalone_summary = {
    family: float(
        evaluate(valid.user_id, valid.y, score)["primary"]
    )
    for family, score in own_valid.items()
}
print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner,
            "standalone": standalone_summary,
            "incumbent_rank_primary": float(inc_metric["primary"]),
            "mean_user_slate": float(
                np.mean(temporal_valid["user_total"])
            ),
            "repeat_video_row_fraction": float(
                np.mean(temporal_valid["video_pos"] > 0)
            ),
            "repeat_author_row_fraction": float(
                np.mean(temporal_valid["author_pos"] > 0)
            ),
        },
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

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if recipe_type != "incumbent":
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(own_valid[winner_family], dtype=np.float64),
        )

test = load("test")
X_test, temporal_test = make_temporal_features(test)

if recipe_type == "incumbent":
    own_test_rank = None
elif winner_family == "position_hazard_gam":
    own_test_rank = within_user_rank(
        test.user_id, hazard_model.predict(temporal_test)
    )
elif winner_family == "position_binary_boost":
    own_test_rank = within_user_rank(
        test.user_id, binary_model.predict(X_test)
    )
else:
    own_test_rank = within_user_rank(
        test.user_id, rank_model.predict(X_test)
    )

if not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent test predictions missing")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test) != len(test.user_id):
    raise ValueError("Incumbent test length mismatch")
inc_test_rank = within_user_rank(test.user_id, inc_test)

if recipe_type == "incumbent":
    test_scores = inc_test_rank
elif recipe_type == "standalone":
    test_scores = own_test_rank
else:
    test_scores = (
        winner_weight * own_test_rank
        + (1.0 - winner_weight) * inc_test_rank
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