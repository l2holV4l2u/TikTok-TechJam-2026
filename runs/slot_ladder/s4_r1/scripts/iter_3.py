import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2026
THREADS = min(12, os.cpu_count() or 1)

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "hour",
    "video_type",
    "onehot_feat3",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]
HALF_LIVES = [2.0, 4.0, 8.0, 1.0e9]
BLEND_WEIGHTS = [0.25, 0.50, 0.75]
N_ROUNDS = 100


def finite_float32(x):
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0)


def build_matrix(split_name, split):
    columns = []

    for field in CAT_FIELDS:
        columns.append(np.asarray(split.X[field], dtype=np.float32))

    for field in NUM_FIELDS:
        x = np.asarray(split.num[field], dtype=np.float32)
        x = np.maximum(x, 0.0)
        columns.append(np.log1p(x).astype(np.float32))

    vh = historical_features(split_name, key="video_id")
    ah = historical_features(split_name, key="author_id")

    for name in sorted(vh):
        columns.append(finite_float32(vh[name]))
    for name in sorted(ah):
        columns.append(finite_float32(ah[name]))

    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def temporal_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.float64)
    age = dates.max() - dates
    if half_life > 1.0e8:
        w = np.ones(len(dates), dtype=np.float32)
    else:
        w = np.exp2(-age / half_life).astype(np.float32)
        w /= max(float(w.mean()), 1e-8)
    return w


def probability_to_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log1p(-p)


def empirical_bayes_tables(train, half_life):
    y = np.asarray(train.y, dtype=np.float64)
    w = temporal_weights(train.date, half_life).astype(np.float64)
    prior = float(np.sum(w * y) / np.sum(w))

    tables = {}
    for field, strength in [
        ("video_id", 18.0),
        ("author_id", 30.0),
    ]:
        ids = np.asarray(train.X[field], dtype=np.int64)
        size = int(ids.max()) + 1
        count = np.bincount(ids, weights=w, minlength=size)
        positive = np.bincount(ids, weights=w * y, minlength=size)
        rate = (positive + strength * prior) / (count + strength)
        tables[field] = np.asarray(rate, dtype=np.float64)

    return prior, tables


def empirical_bayes_predict(split, prior, tables):
    components = []
    for field in ["video_id", "author_id"]:
        ids = np.asarray(split.X[field], dtype=np.int64)
        table = tables[field]
        rate = np.full(len(ids), prior, dtype=np.float64)
        seen = ids < len(table)
        rate[seen] = table[ids[seen]]
        rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
        components.append(np.log(rate) - np.log1p(-rate))

    return 0.65 * components[0] + 0.35 * components[1]


def metric_primary(valid, scores):
    return float(evaluate(valid.user_id, valid.y, scores)["primary"])


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
x_train = build_matrix("train", train)
x_valid = build_matrix("valid", valid)

categorical_indices = list(range(len(CAT_FIELDS)))

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

candidate_scores = {}
models = {}
raw_valid_by_name = {}
blend_weight_by_name = {}

binary_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.07,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 180,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "verbosity": -1,
    "num_threads": THREADS,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "force_col_wise": True,
}

# Family 1: pointwise boosted trees with a sweep over temporal half-life.
for half_life in HALF_LIVES:
    tag = "uniform" if half_life > 1e8 else ("%gd" % half_life)
    name = "binary_gbdt_" + tag
    weights = temporal_weights(train.date, half_life)

    dataset = lgb.Dataset(
        x_train,
        label=y_train,
        weight=weights,
        categorical_feature=categorical_indices,
        free_raw_data=False,
    )
    booster = lgb.train(binary_params, dataset, num_boost_round=N_ROUNDS)
    raw = probability_to_logit(booster.predict(x_valid))
    models[name] = booster
    raw_valid_by_name[name] = raw

    standalone = metric_primary(valid, raw)
    candidate_scores[name + "_standalone"] = standalone

    best_score = standalone
    best_weight = 1.0
    for own_weight in BLEND_WEIGHTS:
        blended = own_weight * raw + (1.0 - own_weight) * inc_valid
        score = metric_primary(valid, blended)
        candidate_scores[name + "_blend_%.2f" % own_weight] = score
        if score > best_score:
            best_score = score
            best_weight = own_weight
    blend_weight_by_name[name] = best_weight

    del dataset, weights
    gc.collect()


# Family 2: LambdaRank, where each training user is one logged ranking query.
sort_idx = np.argsort(np.asarray(train.user_id), kind="stable")
sorted_users = np.asarray(train.user_id)[sort_idx]
group_starts = np.r_[0, 1 + np.flatnonzero(sorted_users[1:] != sorted_users[:-1])]
group_ends = np.r_[group_starts[1:], len(sorted_users)]
group_sizes = (group_ends - group_starts).astype(np.int32)

rank_weights = temporal_weights(train.date, 4.0)[sort_idx]
rank_dataset = lgb.Dataset(
    x_train[sort_idx],
    label=y_train[sort_idx],
    weight=rank_weights,
    group=group_sizes,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)

rank_params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "lambdarank_truncation_level": 8,
    "learning_rate": 0.06,
    "num_leaves": 31,
    "min_data_in_leaf": 180,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "verbosity": -1,
    "num_threads": THREADS,
    "seed": SEED + 1,
    "force_col_wise": True,
}

rank_name = "lambdarank_4d"
rank_model = lgb.train(rank_params, rank_dataset, num_boost_round=N_ROUNDS)
rank_valid = np.asarray(rank_model.predict(x_valid), dtype=np.float64)
models[rank_name] = rank_model
raw_valid_by_name[rank_name] = rank_valid

standalone = metric_primary(valid, rank_valid)
candidate_scores[rank_name + "_standalone"] = standalone
best_score = standalone
best_weight = 1.0
for own_weight in BLEND_WEIGHTS:
    blended = own_weight * rank_valid + (1.0 - own_weight) * inc_valid
    score = metric_primary(valid, blended)
    candidate_scores[rank_name + "_blend_%.2f" % own_weight] = score
    if score > best_score:
        best_score = score
        best_weight = own_weight
blend_weight_by_name[rank_name] = best_weight

del rank_dataset, rank_weights, sort_idx, sorted_users
gc.collect()


# Family 3: non-parametric recency-weighted empirical Bayes entity rates.
eb_tables = {}
for half_life in [2.0, 4.0, 8.0]:
    name = "empirical_bayes_%gd" % half_life
    prior, tables = empirical_bayes_tables(train, half_life)
    raw = empirical_bayes_predict(valid, prior, tables)
    eb_tables[name] = (prior, tables)
    raw_valid_by_name[name] = raw

    standalone = metric_primary(valid, raw)
    candidate_scores[name + "_standalone"] = standalone
    best_score = standalone
    best_weight = 1.0
    for own_weight in BLEND_WEIGHTS:
        blended = own_weight * raw + (1.0 - own_weight) * inc_valid
        score = metric_primary(valid, blended)
        candidate_scores[name + "_blend_%.2f" % own_weight] = score
        if score > best_score:
            best_score = score
            best_weight = own_weight
    blend_weight_by_name[name] = best_weight


# Select the family/half-life and its best incumbent blend.
family_best = {}
for name, raw in raw_valid_by_name.items():
    own_weight = blend_weight_by_name[name]
    final = own_weight * raw + (1.0 - own_weight) * inc_valid
    family_best[name] = metric_primary(valid, final)

winner = max(family_best, key=family_best.get)
winner_weight = blend_weight_by_name[winner]
winner_raw_valid = raw_valid_by_name[winner]
valid_scores = (
    winner_weight * winner_raw_valid
    + (1.0 - winner_weight) * inc_valid
)
metrics = evaluate(valid.user_id, valid.y, valid_scores)

# Create test features only after validation-side model comparison is complete.
test = load("test")

if winner.startswith("binary_gbdt"):
    x_test = build_matrix("test", test)
    winner_raw_test = probability_to_logit(models[winner].predict(x_test))
elif winner.startswith("lambdarank"):
    x_test = build_matrix("test", test)
    winner_raw_test = np.asarray(models[winner].predict(x_test), dtype=np.float64)
else:
    prior, tables = eb_tables[winner]
    winner_raw_test = empirical_bayes_predict(test, prior, tables)

test_scores = (
    winner_weight * winner_raw_test
    + (1.0 - winner_weight) * inc_test
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if winner_weight < 1.0:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(winner_raw_valid, dtype=np.float64),
        )

binary_family = {
    k: v for k, v in family_best.items() if k.startswith("binary_gbdt")
}
best_binary = max(binary_family, key=binary_family.get)
print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner,
            "winner_own_weight": winner_weight,
            "best_binary_half_life": best_binary,
            "best_binary_primary": binary_family[best_binary],
        },
        sort_keys=True,
    )
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.3f}'
    % (
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        elapsed,
    )
)