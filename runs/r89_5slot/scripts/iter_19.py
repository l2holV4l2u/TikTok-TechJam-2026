import os
import time
import json
import random
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 41773
random.seed(SEED)
np.random.seed(SEED)

CAT_FIELDS = [
    "video_id", "author_id", "tab", "tag", "duration_bucket",
    "upload_type", "music_type", "video_type", "onehot_feat3",
    "onehot_feat7", "onehot_feat8", "user_active_degree",
    "fans_user_num_range", "follow_user_num_range",
    "friend_user_num_range", "register_days_bucket",
    "is_video_author", "hour",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

# Explicit interactions whose cardinalities remain modest and whose entities
# are essentially fully warm across the date boundary.
PAIR_FIELDS = [
    ("video_id", "tab"),
    ("video_id", "tag"),
    ("author_id", "tab"),
    ("author_id", "tag"),
    ("tag", "duration_bucket"),
    ("tag", "upload_type"),
    ("onehot_feat3", "tab"),
    ("onehot_feat8", "tab"),
    ("duration_bucket", "tab"),
]


def extract_cats(s):
    return {
        f: np.asarray(s.X[f], dtype=np.int64)
        for f in CAT_FIELDS
    }


def extract_num(s):
    cols = []
    for f in NUM_FIELDS:
        x = np.asarray(s.num[f], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(np.log1p(np.maximum(x, 0.0)))
    return np.column_stack(cols).astype(np.float32)


def recency_weights(dates, half_life=7.0):
    dates = np.asarray(dates, dtype=np.int64)
    unique = np.unique(dates)
    idx = np.searchsorted(unique, dates)
    age = (len(unique) - 1 - idx).astype(np.float64)
    w = np.exp2(-age / float(half_life))
    return w.astype(np.float64)


def weighted_logit_tables(train_cats, y, weights, smoothing=18.0):
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    global_rate = float(np.sum(weights * y) / np.sum(weights))
    global_rate = float(np.clip(global_rate, 1e-5, 1.0 - 1e-5))
    global_logit = np.log(global_rate / (1.0 - global_rate))

    tables = {}
    for f in CAT_FIELDS:
        ids = train_cats[f]
        card = int(FEATURE_CARDINALITIES[f])
        total = np.bincount(ids, weights=weights, minlength=card)
        pos = np.bincount(ids, weights=weights * y, minlength=card)
        rate = (pos + smoothing * global_rate) / (total + smoothing)
        rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
        evidence = np.log(rate / (1.0 - rate)) - global_logit

        # Reliability shrinkage suppresses transient evidence from rare IDs.
        reliability = total / (total + smoothing)
        tables[f] = (evidence * reliability).astype(np.float32)

    return tables, float(global_logit)


def predict_additive(tables, bias, cats):
    score = np.full(len(next(iter(cats.values()))), bias, dtype=np.float64)

    # Strong stable entity evidence, with smaller contributions from metadata.
    coefficients = {
        "video_id": 1.00,
        "author_id": 0.65,
        "tab": 0.75,
        "tag": 0.55,
        "duration_bucket": 0.30,
        "upload_type": 0.35,
        "music_type": 0.20,
        "video_type": 0.15,
        "onehot_feat3": 0.30,
        "onehot_feat7": 0.20,
        "onehot_feat8": 0.25,
        "user_active_degree": 0.20,
        "fans_user_num_range": 0.15,
        "follow_user_num_range": 0.10,
        "friend_user_num_range": 0.10,
        "register_days_bucket": 0.15,
        "is_video_author": 0.15,
        "hour": 0.20,
    }
    for f in CAT_FIELDS:
        score += coefficients[f] * tables[f][cats[f]]
    return score.astype(np.float32)


def fit_pair_tables(train_cats, y, weights, smoothing=28.0):
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    global_rate = float(np.sum(weights * y) / np.sum(weights))
    global_rate = float(np.clip(global_rate, 1e-5, 1.0 - 1e-5))
    global_logit = np.log(global_rate / (1.0 - global_rate))

    tables = {}
    for a, b in PAIR_FIELDS:
        card_b = int(FEATURE_CARDINALITIES[b])
        card = int(FEATURE_CARDINALITIES[a]) * card_b
        code = train_cats[a] * card_b + train_cats[b]

        total = np.bincount(code, weights=weights, minlength=card)
        pos = np.bincount(code, weights=weights * y, minlength=card)
        rate = (pos + smoothing * global_rate) / (total + smoothing)
        rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
        residual = np.log(rate / (1.0 - rate)) - global_logit
        reliability = total / (total + smoothing)
        tables[(a, b)] = (residual * reliability).astype(np.float32)

    return tables, float(global_logit)


def predict_pair_tables(tables, bias, cats):
    n = len(next(iter(cats.values())))
    score = np.full(n, bias, dtype=np.float64)
    for a, b in PAIR_FIELDS:
        card_b = int(FEATURE_CARDINALITIES[b])
        code = cats[a] * card_b + cats[b]
        score += tables[(a, b)][code]
    return score.astype(np.float32)


def make_tree_matrix(cats, nums):
    cat_matrix = np.column_stack([
        cats[f].astype(np.float32, copy=False) for f in CAT_FIELDS
    ])
    return np.column_stack([cat_matrix, nums]).astype(np.float32)


def train_random_forest(cats, nums, y, weights, rounds=150):
    x = make_tree_matrix(cats, nums)
    dset = lgb.Dataset(
        x,
        label=np.asarray(y, dtype=np.float32),
        weight=np.asarray(weights, dtype=np.float32),
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "rf",
        "learning_rate": 1.0,
        "num_leaves": 127,
        "max_depth": -1,
        "min_data_in_leaf": 220,
        "feature_fraction": 0.62,
        "bagging_fraction": 0.68,
        "bagging_freq": 1,
        "max_bin": 127,
        "cat_smooth": 30.0,
        "cat_l2": 15.0,
        "lambda_l1": 0.10,
        "lambda_l2": 3.0,
        "extra_trees": True,
        "num_threads": min(8, os.cpu_count() or 1),
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "extra_seed": SEED + 3,
        "verbose": -1,
    }
    return lgb.train(params, dset, num_boost_round=int(rounds))


def predict_tree(model, cats, nums):
    return model.predict(make_tree_matrix(cats, nums)).astype(np.float32)


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, users))
    sorted_users = users[order]

    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(first)
    counts = np.diff(np.r_[starts, n])
    position = np.arange(n, dtype=np.float64) - np.repeat(starts, counts)
    denom = np.maximum(np.repeat(counts, counts) - 1, 1)
    rank = position / denom
    rank[np.repeat(counts, counts) == 1] = 0.5

    out = np.empty(n, dtype=np.float64)
    out[order] = rank
    return out


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(x.std())
    if sd < 1e-12:
        sd = 1.0
    return (x - float(x.mean())) / sd


def concat_cats(a, b):
    return {
        f: np.concatenate([a[f], b[f]]).astype(np.int64, copy=False)
        for f in CAT_FIELDS
    }


train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
train_users = np.asarray(train.user_id, dtype=np.int64)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

train_cats = extract_cats(train)
valid_cats = extract_cats(valid)
train_num = extract_num(train)
valid_num = extract_num(valid)
train_weights = recency_weights(train.date, half_life=7.0)

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.asarray(
    np.load(os.path.join(shared, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test = np.asarray(
    np.load(os.path.join(shared, "incumbent_test_scores.npy")),
    dtype=np.float64,
)

# Family 1: additive empirical-Bayes / categorical generative evidence.
add_tables, add_bias = weighted_logit_tables(
    train_cats, train_y, train_weights, smoothing=18.0
)
pred_add = predict_additive(add_tables, add_bias, valid_cats)

# Family 2: explicit non-parametric interaction memorization.
pair_tables, pair_bias = fit_pair_tables(
    train_cats, train_y, train_weights, smoothing=28.0
)
pred_pair = predict_pair_tables(pair_tables, pair_bias, valid_cats)

# Family 3: nonlinear bagged random partitions rather than boosting.
rf_model = train_random_forest(
    train_cats, train_num, train_y, train_weights, rounds=150
)
pred_rf = predict_tree(rf_model, valid_cats, valid_num)

families = {
    "generative_additive": pred_add,
    "pair_empirical_bayes": pred_pair,
    "categorical_random_forest": pred_rf,
}

candidate_values = {}
specs = {}
inc_metrics = evaluate(valid_users, valid_y, inc_valid)
candidate_values["incumbent"] = float(inc_metrics["primary"])

best_name = "incumbent"
best_scores = inc_valid.copy()
best_metrics = inc_metrics
best_primary = float(inc_metrics["primary"])
best_spec = ("generative_additive", 0.0, "rank")

inc_rank = within_user_rank(valid_users, inc_valid)
inc_z = zscore(inc_valid)

for family, pred in families.items():
    raw_met = evaluate(valid_users, valid_y, pred)
    candidate_values[family] = float(raw_met["primary"])
    if float(raw_met["primary"]) > best_primary:
        best_name = family
        best_scores = np.asarray(pred, dtype=np.float64)
        best_metrics = raw_met
        best_primary = float(raw_met["primary"])
        best_spec = (family, 1.0, "raw")

    pred_rank = within_user_rank(valid_users, pred)
    pred_z = zscore(pred)

    local_best = (-1.0, None)
    for mode, left, right in (
        ("rank", inc_rank, pred_rank),
        ("zscore", inc_z, pred_z),
    ):
        for alpha in (0.10, 0.18, 0.26, 0.34, 0.42, 0.50, 0.62):
            scores = (1.0 - alpha) * left + alpha * right
            met = evaluate(valid_users, valid_y, scores)
            primary = float(met["primary"])
            if primary > local_best[0]:
                local_best = (primary, (mode, alpha))
            if primary > best_primary:
                best_primary = primary
                best_name = f"{family}_{mode}_blend_{alpha:.2f}"
                best_scores = scores
                best_metrics = met
                best_spec = (family, float(alpha), mode)

    candidate_values[
        f"{family}_best_blend"
    ] = float(local_best[0])

print("CANDIDATES " + json.dumps(candidate_values, sort_keys=True))

raw_family, blend_alpha, blend_mode = best_spec
raw_valid = np.asarray(families[raw_family], dtype=np.float64)

# Refit the identical selected family on train+validation for test scoring.
test = load("test")
test_users = np.asarray(test.user_id, dtype=np.int64)
test_cats = extract_cats(test)
test_num = extract_num(test)

combined_cats = concat_cats(train_cats, valid_cats)
combined_num = np.concatenate([train_num, valid_num], axis=0).astype(np.float32)
combined_y = np.concatenate([
    train_y,
    valid_y.astype(np.float32),
])
combined_dates = np.concatenate([
    np.asarray(train.date, dtype=np.int64),
    np.asarray(valid.date, dtype=np.int64),
])
combined_weights = recency_weights(combined_dates, half_life=7.0)

if raw_family == "generative_additive":
    final_tables, final_bias = weighted_logit_tables(
        combined_cats, combined_y, combined_weights, smoothing=18.0
    )
    raw_test = predict_additive(final_tables, final_bias, test_cats)
elif raw_family == "pair_empirical_bayes":
    final_tables, final_bias = fit_pair_tables(
        combined_cats, combined_y, combined_weights, smoothing=28.0
    )
    raw_test = predict_pair_tables(final_tables, final_bias, test_cats)
else:
    final_rf = train_random_forest(
        combined_cats, combined_num, combined_y, combined_weights, rounds=150
    )
    raw_test = predict_tree(final_rf, test_cats, test_num)

if blend_mode == "raw":
    test_scores = np.asarray(raw_test, dtype=np.float64)
elif blend_mode == "rank":
    test_scores = (
        (1.0 - blend_alpha) * within_user_rank(test_users, inc_test)
        + blend_alpha * within_user_rank(test_users, raw_test)
    )
else:
    test_scores = (
        (1.0 - blend_alpha) * zscore(inc_test)
        + blend_alpha * zscore(raw_test)
    )

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_name != raw_family:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            raw_valid,
        )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)