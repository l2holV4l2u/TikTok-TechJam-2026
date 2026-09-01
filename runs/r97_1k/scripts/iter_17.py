import os
import gc
import json
import time
import warnings
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")

START = time.time()
OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")

if OUT:
    os.makedirs(OUT, exist_ok=True)

if not SHARED:
    raise RuntimeError("SHARED_ARTIFACTS is required")

INC_VALID_PATH = os.path.join(SHARED, "incumbent_valid_scores.npy")
INC_TEST_PATH = os.path.join(SHARED, "incumbent_test_scores.npy")

if not os.path.exists(INC_VALID_PATH) or not os.path.exists(INC_TEST_PATH):
    raise RuntimeError("Trusted incumbent predictions are unavailable")


CAT_FIELDS = [
    "user_id",
    "tab",
    "tag",
    "upload_type",
    "duration_bucket",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat1",
    "onehot_feat7",
    "music_type",
    "user_active_degree",
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_bucket",
    "is_video_author",
    "hour",
]

TE_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "upload_type",
    "duration_bucket",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat1",
    "onehot_feat7",
    "music_type",
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
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    start_positions = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local_positions = (
        np.arange(n, dtype=np.float32)
        - start_positions.astype(np.float32)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(np.r_[-1, end_positions]).astype(np.float32)

    group_index = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[group_index] - 1.0, 1.0)

    result = np.empty(n, dtype=np.float32)
    result[order] = local_positions / denom
    return result


def stable_numeric(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=1e8, neginf=0.0)
    return np.log1p(np.maximum(x, 0.0)).astype(np.float32)


def load_histories(split_name):
    vh = historical_features(split_name, key="video_id")
    ah = historical_features(split_name, key="author_id")
    merged = {}
    for k, v in vh.items():
        merged["video_" + k] = np.asarray(v, dtype=np.float32)
    for k, v in ah.items():
        merged["author_" + k] = np.asarray(v, dtype=np.float32)
    return merged


train = load("train")
train_y = np.asarray(train.y, dtype=np.float32)
train_dates = np.asarray(train.date, dtype=np.int32)

train_hist = load_histories("train")
HISTORY_NAMES = sorted(train_hist.keys())

# Four-day half-life strongly emphasizes the portion of train nearest to the
# future windows while retaining nonzero support from all fourteen days.
days_old = (
    int(np.max(train_dates)) - train_dates
).astype(np.float32)
sample_weight = np.exp2(-days_old / 4.0).astype(np.float32)
sample_weight /= np.mean(sample_weight)


def build_matrix(split, histories):
    cols = []

    for field in CAT_FIELDS:
        cols.append(np.asarray(split.X[field], dtype=np.float32))

    for field in NUM_FIELDS:
        cols.append(stable_numeric(split.num[field]))

    for name in HISTORY_NAMES:
        x = np.asarray(histories[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0)
        cols.append(x.astype(np.float32, copy=False))

    return np.column_stack(cols).astype(np.float32, copy=False)


print(
    "FINDINGS feature_count=%d categorical_count=%d history_count=%d"
    % (
        len(CAT_FIELDS) + len(NUM_FIELDS) + len(HISTORY_NAMES),
        len(CAT_FIELDS),
        len(HISTORY_NAMES),
    ),
    flush=True,
)

X_train = build_matrix(train, train_hist)
del train_hist
gc.collect()

categorical_indices = list(range(len(CAT_FIELDS)))

dtrain = lgb.Dataset(
    X_train,
    label=train_y,
    weight=sample_weight,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)

boost_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "min_data_in_leaf": 1200,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.80,
    "bagging_freq": 1,
    "lambda_l1": 0.15,
    "lambda_l2": 4.0,
    "max_bin": 127,
    "max_cat_threshold": 32,
    "cat_smooth": 30.0,
    "cat_l2": 15.0,
    "num_threads": max(1, min(16, os.cpu_count() or 8)),
    "seed": 1701,
    "feature_fraction_seed": 1702,
    "bagging_seed": 1703,
    "data_random_seed": 1704,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}

boost_model = lgb.train(
    boost_params,
    dtrain,
    num_boost_round=360,
)

rf_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "rf",
    "learning_rate": 1.0,
    "num_leaves": 95,
    "min_data_in_leaf": 1600,
    "feature_fraction": 0.62,
    "bagging_fraction": 0.60,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 8.0,
    "max_bin": 127,
    "max_cat_threshold": 32,
    "cat_smooth": 40.0,
    "cat_l2": 20.0,
    "num_threads": max(1, min(16, os.cpu_count() or 8)),
    "seed": 2711,
    "feature_fraction_seed": 2712,
    "bagging_seed": 2713,
    "data_random_seed": 2714,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}

rf_model = lgb.train(
    rf_params,
    dtrain,
    num_boost_round=240,
)


# A separate additive empirical-Bayes family. Its predictions are sums of
# independently smoothed, temporally weighted categorical evidence rather
# than paths through trees.
global_rate = float(
    np.sum(sample_weight * train_y) / np.sum(sample_weight)
)
global_logit = np.log(
    np.clip(global_rate, 1e-6, 1.0 - 1e-6)
    / np.clip(1.0 - global_rate, 1e-6, 1.0)
)

TE_SMOOTHING = {
    "video_id": 45.0,
    "author_id": 70.0,
    "tag": 500.0,
    "upload_type": 500.0,
    "duration_bucket": 500.0,
    "onehot_feat3": 250.0,
    "onehot_feat8": 250.0,
    "onehot_feat1": 500.0,
    "onehot_feat7": 400.0,
    "music_type": 700.0,
}

TE_COEFFICIENT = {
    "video_id": 1.8,
    "author_id": 2.0,
    "tag": 0.9,
    "upload_type": 0.7,
    "duration_bucket": 0.6,
    "onehot_feat3": 1.1,
    "onehot_feat8": 1.0,
    "onehot_feat1": 0.5,
    "onehot_feat7": 0.5,
    "music_type": 0.4,
}

te_tables = {}
for field in TE_FIELDS:
    ids = np.asarray(train.X[field], dtype=np.int64)
    cardinality = int(FEATURE_CARDINALITIES[field])
    denominator = np.bincount(
        ids,
        weights=sample_weight,
        minlength=cardinality,
    ).astype(np.float64)
    numerator = np.bincount(
        ids,
        weights=sample_weight * train_y,
        minlength=cardinality,
    ).astype(np.float64)

    smoothing = float(TE_SMOOTHING[field])
    rate = (
        numerator + smoothing * global_rate
    ) / (
        denominator + smoothing
    )
    rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
    te_tables[field] = np.log(rate / (1.0 - rate)).astype(np.float32)


def empirical_bayes_score(split):
    score = np.full(
        len(split.user_id),
        global_logit,
        dtype=np.float32,
    )
    total_coefficient = 0.0

    for field in TE_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        table = te_tables[field]
        safe_ids = np.minimum(np.maximum(ids, 0), len(table) - 1)
        coefficient = float(TE_COEFFICIENT[field])
        score += coefficient * (table[safe_ids] - global_logit)
        total_coefficient += coefficient

    score /= max(total_coefficient, 1.0)
    return score


valid = load("valid")
valid_uid = np.asarray(valid.user_id, dtype=np.int64)
valid_y = np.asarray(valid.y)
valid_hist = load_histories("valid")
X_valid = build_matrix(valid, valid_hist)
del valid_hist
gc.collect()

boost_valid_raw = boost_model.predict(
    X_valid,
    num_iteration=boost_model.current_iteration(),
).astype(np.float32)
rf_valid_raw = rf_model.predict(
    X_valid,
    num_iteration=rf_model.current_iteration(),
).astype(np.float32)
eb_valid_raw = empirical_bayes_score(valid)

inc_valid_memmap = np.load(INC_VALID_PATH, mmap_mode="r")
inc_valid_rank = within_user_rank(valid_uid, inc_valid_memmap)
boost_valid_rank = within_user_rank(valid_uid, boost_valid_raw)
rf_valid_rank = within_user_rank(valid_uid, rf_valid_raw)
eb_valid_rank = within_user_rank(valid_uid, eb_valid_raw)

fresh_valid = {
    "recency_gbdt": boost_valid_rank,
    "recency_random_forest": rf_valid_rank,
    "recency_empirical_bayes": eb_valid_rank,
    "gbdt_rf_bagboost": (
        0.62 * boost_valid_rank + 0.38 * rf_valid_rank
    ).astype(np.float32),
    "three_family_average": (
        0.52 * boost_valid_rank
        + 0.30 * rf_valid_rank
        + 0.18 * eb_valid_rank
    ).astype(np.float32),
}

candidate_metrics = {}
candidate_specs = {}

inc_metrics = evaluate(valid_uid, valid_y, inc_valid_rank)
candidate_metrics["trusted_incumbent_control"] = float(
    inc_metrics["primary"]
)

best_name = "trusted_incumbent_control"
best_scores = inc_valid_rank.copy()
best_metrics = inc_metrics
best_spec = {
    "source": "incumbent",
    "gate": "none",
    "alpha": 0.0,
}
best_fresh_valid = boost_valid_rank

for source_name, source_scores in fresh_valid.items():
    metrics = evaluate(valid_uid, valid_y, source_scores)
    candidate_metrics[source_name] = float(metrics["primary"])

    print(
        "FINDINGS standalone=%s primary=%.6f gauc=%.6f ndcg5=%.6f"
        % (
            source_name,
            float(metrics["primary"]),
            float(metrics["gauc"]),
            float(metrics["ndcg@5"]),
        ),
        flush=True,
    )

    if float(metrics["primary"]) > float(best_metrics["primary"]):
        best_name = source_name
        best_scores = source_scores.copy()
        best_metrics = metrics
        best_spec = {
            "source": source_name,
            "gate": "standalone",
            "alpha": 1.0,
        }
        best_fresh_valid = source_scores.copy()

    disagreement = np.abs(source_scores - inc_valid_rank)
    gate_high_disagreement = (
        1.0 - np.exp(-disagreement / 0.15)
    ).astype(np.float32)
    gate_low_disagreement = np.exp(
        -disagreement / 0.15
    ).astype(np.float32)
    gate_head_union = (
        np.maximum(source_scores, inc_valid_rank) >= 0.96
    ).astype(np.float32)

    gates = {
        "global": np.ones(len(valid_uid), dtype=np.float32),
        "high_disagreement": gate_high_disagreement,
        "low_disagreement": gate_low_disagreement,
        "head_union": gate_head_union,
    }

    for gate_name, gate in gates.items():
        for alpha in (0.05, 0.10, 0.16, 0.24, 0.34):
            blended = (
                inc_valid_rank
                + float(alpha)
                * gate
                * (source_scores - inc_valid_rank)
            ).astype(np.float32)

            name = "%s__%s__a%.2f" % (
                source_name,
                gate_name,
                alpha,
            )
            metrics = evaluate(valid_uid, valid_y, blended)
            primary = float(metrics["primary"])
            candidate_metrics[name] = primary
            candidate_specs[name] = {
                "source": source_name,
                "gate": gate_name,
                "alpha": float(alpha),
            }

            if primary > float(best_metrics["primary"]):
                best_name = name
                best_scores = blended.copy()
                best_metrics = metrics
                best_spec = candidate_specs[name].copy()
                best_fresh_valid = source_scores.copy()

print(
    "FINDINGS winner=%s source=%s gate=%s alpha=%.3f "
    "control=%.6f winner=%.6f delta=%+.6f"
    % (
        best_name,
        best_spec["source"],
        best_spec["gate"],
        float(best_spec["alpha"]),
        float(inc_metrics["primary"]),
        float(best_metrics["primary"]),
        float(best_metrics["primary"]) - float(inc_metrics["primary"]),
    ),
    flush=True,
)

print(
    "CANDIDATES " + json.dumps(candidate_metrics, sort_keys=True),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(OUT, "scores_valid_raw.npy"),
        np.asarray(best_fresh_valid, dtype=np.float64),
    )

del X_valid
del boost_valid_raw
del rf_valid_raw
del eb_valid_raw
del inc_valid_memmap
del valid
gc.collect()

test = load("test")
test_uid = np.asarray(test.user_id, dtype=np.int64)
test_hist = load_histories("test")
X_test = build_matrix(test, test_hist)
del test_hist
gc.collect()

boost_test_raw = boost_model.predict(
    X_test,
    num_iteration=boost_model.current_iteration(),
).astype(np.float32)
rf_test_raw = rf_model.predict(
    X_test,
    num_iteration=rf_model.current_iteration(),
).astype(np.float32)
eb_test_raw = empirical_bayes_score(test)

boost_test_rank = within_user_rank(test_uid, boost_test_raw)
rf_test_rank = within_user_rank(test_uid, rf_test_raw)
eb_test_rank = within_user_rank(test_uid, eb_test_raw)

fresh_test = {
    "recency_gbdt": boost_test_rank,
    "recency_random_forest": rf_test_rank,
    "recency_empirical_bayes": eb_test_rank,
    "gbdt_rf_bagboost": (
        0.62 * boost_test_rank + 0.38 * rf_test_rank
    ).astype(np.float32),
    "three_family_average": (
        0.52 * boost_test_rank
        + 0.30 * rf_test_rank
        + 0.18 * eb_test_rank
    ).astype(np.float32),
}

inc_test_memmap = np.load(INC_TEST_PATH, mmap_mode="r")
inc_test_rank = within_user_rank(test_uid, inc_test_memmap)

if best_spec["source"] == "incumbent":
    test_scores = inc_test_rank
elif best_spec["gate"] == "standalone":
    test_scores = fresh_test[best_spec["source"]]
else:
    source_test = fresh_test[best_spec["source"]]
    disagreement = np.abs(source_test - inc_test_rank)

    if best_spec["gate"] == "global":
        gate = np.ones(len(test_uid), dtype=np.float32)
    elif best_spec["gate"] == "high_disagreement":
        gate = (
            1.0 - np.exp(-disagreement / 0.15)
        ).astype(np.float32)
    elif best_spec["gate"] == "low_disagreement":
        gate = np.exp(-disagreement / 0.15).astype(np.float32)
    elif best_spec["gate"] == "head_union":
        gate = (
            np.maximum(source_test, inc_test_rank) >= 0.96
        ).astype(np.float32)
    else:
        raise ValueError("Unknown gate " + best_spec["gate"])

    test_scores = (
        inc_test_rank
        + float(best_spec["alpha"])
        * gate
        * (source_test - inc_test_rank)
    ).astype(np.float32)

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print(
    "METRICS "
    + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": elapsed,
    })
)