import os
import time
import json
import numpy as np
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
EPS = 1.0e-6

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "music_type",
    "video_type",
    "user_active_degree",
    "is_live_streamer",
    "is_video_author",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]

EB_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "hour",
    "music_type",
]

EB_CROSSES = [
    ("video_id", "tab"),
    ("author_id", "tab"),
    ("author_id", "tag"),
    ("tab", "duration_bucket"),
    ("tag", "duration_bucket"),
    ("upload_type", "duration_bucket"),
]


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    group_ids = np.cumsum(starts_mask) - 1
    positions = np.arange(n, dtype=np.int64) - starts[group_ids]
    sizes = np.diff(np.append(starts, n))
    denom = np.maximum(sizes[group_ids] - 1, 1)

    ranked_sorted = positions.astype(np.float64) / denom
    ranked_sorted[sizes[group_ids] == 1] = 0.5

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def recency_weights(dates, half_life=5.0):
    dates = np.asarray(dates, dtype=np.int64)
    latest = int(dates.max())
    weights = np.power(
        2.0,
        (dates.astype(np.float64) - latest) / float(half_life),
    )
    return weights / weights.mean()


def get_cat(split, name, indices=None):
    if name == "user_id":
        arr = np.asarray(split.user_id, dtype=np.int32)
    elif name == "video_id":
        arr = np.asarray(split.video_id, dtype=np.int32)
    else:
        arr = np.asarray(split.X[name], dtype=np.int32)
    if indices is not None:
        arr = arr[indices]
    return arr


def build_tree_matrix(split, indices=None):
    columns = []
    for name in CAT_FIELDS:
        columns.append(get_cat(split, name, indices).astype(np.float32))

    for name in [
        "duration_ms",
        "user_fans_user_num",
        "user_follow_user_num",
        "user_friend_user_num",
        "user_register_days",
    ]:
        arr = np.asarray(split.num[name], dtype=np.float64)
        if indices is not None:
            arr = arr[indices]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(arr, 0.0)).astype(np.float32))

    dates = np.asarray(split.date, dtype=np.int64)
    if indices is not None:
        dates = dates[indices]
    weekday = ((dates - 20220404) % 7).astype(np.float32)
    columns.append(weekday)

    return np.column_stack(columns).astype(np.float32, copy=False)


def fit_tree(X, y, weights, rounds):
    dset = lgb.Dataset(
        X,
        label=np.asarray(y, dtype=np.float32),
        weight=np.asarray(weights, dtype=np.float32),
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.055,
        "num_leaves": 47,
        "max_depth": -1,
        "min_data_in_leaf": 450,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.88,
        "bagging_freq": 1,
        "lambda_l1": 0.15,
        "lambda_l2": 3.0,
        "max_bin": 127,
        "cat_smooth": 40.0,
        "cat_l2": 12.0,
        "verbosity": -1,
        "verbose": -1,
        "num_threads": max(1, min(12, os.cpu_count() or 4)),
        "seed": 2026,
        "feature_fraction_seed": 2027,
        "bagging_seed": 2028,
    }
    return lgb.train(params, dset, num_boost_round=rounds)


def eb_code(split, name, indices=None):
    return get_cat(split, name, indices).astype(np.int64)


def eb_cardinality(name):
    if name == "user_id":
        return int(FEATURE_CARDINALITIES["user_id"])
    if name == "video_id":
        return int(FEATURE_CARDINALITIES["video_id"])
    return int(FEATURE_CARDINALITIES[name])


def fit_rate_table(code, cardinality, y, weights, prior, strength):
    count = np.bincount(
        code,
        weights=weights,
        minlength=cardinality,
    ).astype(np.float64)
    positive = np.bincount(
        code,
        weights=weights * y,
        minlength=cardinality,
    ).astype(np.float64)
    rate = (positive + strength * prior) / (count + strength)
    return rate, count


def fit_eb(split, indices, y, weights):
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    prior = float(np.sum(weights * y) / np.sum(weights))
    prior_logit = float(logit(prior))

    singles = {}
    for name in EB_FIELDS:
        code = eb_code(split, name, indices)
        rate, count = fit_rate_table(
            code,
            eb_cardinality(name),
            y,
            weights,
            prior,
            strength=90.0 if name in ("video_id", "author_id") else 180.0,
        )
        singles[name] = (rate, count)

    crosses = {}
    for left, right in EB_CROSSES:
        left_code = eb_code(split, left, indices)
        right_code = eb_code(split, right, indices)
        right_card = eb_cardinality(right)
        code = left_code * right_card + right_code
        card = eb_cardinality(left) * right_card
        rate, count = fit_rate_table(
            code,
            card,
            y,
            weights,
            prior,
            strength=260.0,
        )
        crosses[(left, right)] = (rate, count)

    return {
        "prior": prior,
        "prior_logit": prior_logit,
        "singles": singles,
        "crosses": crosses,
    }


def predict_eb(model, split, indices=None):
    if indices is None:
        n = np.asarray(split.user_id).size
    else:
        n = np.asarray(indices).size

    base = model["prior_logit"]
    single_delta = np.zeros(n, dtype=np.float64)

    for name in EB_FIELDS:
        code = eb_code(split, name, indices)
        rates, counts = model["singles"][name]
        reliability = counts[code] / (counts[code] + 120.0)
        single_delta += reliability * (logit(rates[code]) - base)

    cross_delta = np.zeros(n, dtype=np.float64)
    cross_weight = np.zeros(n, dtype=np.float64)
    for left, right in EB_CROSSES:
        lc = eb_code(split, left, indices)
        rc = eb_code(split, right, indices)
        code = lc * eb_cardinality(right) + rc
        rates, counts = model["crosses"][(left, right)]
        reliability = counts[code] / (counts[code] + 350.0)
        cross_delta += reliability * (logit(rates[code]) - base)
        cross_weight += reliability

    cross_average = cross_delta / np.maximum(cross_weight, 1.0)
    score = base + 0.34 * single_delta + 0.72 * cross_average
    return score


def sorted_lookup_counts(reference_users, query_users):
    reference_users = np.asarray(reference_users, dtype=np.int64)
    query_users = np.asarray(query_users, dtype=np.int64)

    unique, counts = np.unique(reference_users, return_counts=True)
    positions = np.searchsorted(unique, query_users)
    clipped = np.minimum(positions, max(unique.size - 1, 0))

    out = np.zeros(query_users.size, dtype=np.float64)
    if unique.size:
        matched = (positions < unique.size) & (unique[clipped] == query_users)
        out[matched] = counts[clipped[matched]]
    return out


def list_sizes(user_ids):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    unique, inverse, counts = np.unique(
        user_ids, return_inverse=True, return_counts=True
    )
    del unique
    return counts[inverse].astype(np.float64)


def build_meta_features(tree_score, eb_score, query_users,
                        reference_users, query_list_sizes):
    tree_logit = logit(tree_score)
    eb_logit = np.asarray(eb_score, dtype=np.float64)
    disagreement = tree_logit - eb_logit
    history_count = sorted_lookup_counts(reference_users, query_users)

    return np.column_stack([
        tree_logit,
        eb_logit,
        0.5 * (tree_logit + eb_logit),
        disagreement,
        np.abs(disagreement),
        np.log1p(history_count),
        (history_count == 0).astype(np.float64),
        np.log1p(query_list_sizes),
        (query_list_sizes <= 2).astype(np.float64),
        ((query_list_sizes >= 8) & (query_list_sizes <= 15)).astype(np.float64),
        (query_list_sizes >= 16).astype(np.float64),
        disagreement * np.log1p(query_list_sizes),
        disagreement * (history_count == 0).astype(np.float64),
    ]).astype(np.float32)


def fit_meta(X, y, weights):
    dset = lgb.Dataset(
        X,
        label=np.asarray(y, dtype=np.float32),
        weight=np.asarray(weights, dtype=np.float32),
        free_raw_data=True,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.045,
        "num_leaves": 15,
        "max_depth": 5,
        "min_data_in_leaf": 180,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.3,
        "lambda_l2": 8.0,
        "verbosity": -1,
        "verbose": -1,
        "num_threads": max(1, min(12, os.cpu_count() or 4)),
        "seed": 3031,
    }
    return lgb.train(params, dset, num_boost_round=90)


train = load("train")
valid = load("valid")

train_dates = np.asarray(train.date, dtype=np.int64)
train_y = np.asarray(train.y, dtype=np.float64)

# Strict temporal stacking split: base learners see only Apr 9-17;
# meta-learner sees labels only from Apr 18-21.
base_idx = np.flatnonzero(train_dates <= 20220417)
meta_idx = np.flatnonzero(train_dates >= 20220418)

base_y = train_y[base_idx]
meta_y = train_y[meta_idx]
base_weights = recency_weights(train_dates[base_idx], half_life=5.0)
meta_weights = recency_weights(train_dates[meta_idx], half_life=3.0)

X_base = build_tree_matrix(train, base_idx)
X_meta = build_tree_matrix(train, meta_idx)

tree_oof_model = fit_tree(X_base, base_y, base_weights, rounds=125)
tree_meta = tree_oof_model.predict(X_meta)

eb_oof_model = fit_eb(
    train,
    base_idx,
    base_y,
    base_weights,
)
eb_meta = predict_eb(eb_oof_model, train, meta_idx)

meta_users = np.asarray(train.user_id, dtype=np.int64)[meta_idx]
base_users = np.asarray(train.user_id, dtype=np.int64)[base_idx]
meta_list_n = list_sizes(meta_users)

X_stack_meta = build_meta_features(
    tree_meta,
    eb_meta,
    meta_users,
    base_users,
    meta_list_n,
)
stack_model = fit_meta(X_stack_meta, meta_y, meta_weights)

# Refit base families on all legal training rows.
all_idx = np.arange(train_y.size, dtype=np.int64)
full_weights = recency_weights(train_dates, half_life=5.0)

X_train_full = build_tree_matrix(train)
tree_full_model = fit_tree(
    X_train_full,
    train_y,
    full_weights,
    rounds=145,
)
eb_full_model = fit_eb(
    train,
    all_idx,
    train_y,
    full_weights,
)

X_valid = build_tree_matrix(valid)
valid_tree = tree_full_model.predict(X_valid)
valid_eb = predict_eb(eb_full_model, valid)

valid_users = np.asarray(valid.user_id, dtype=np.int64)
train_users = np.asarray(train.user_id, dtype=np.int64)
valid_list_n = list_sizes(valid_users)

X_valid_meta = build_meta_features(
    valid_tree,
    valid_eb,
    valid_users,
    train_users,
    valid_list_n,
)
valid_stack = stack_model.predict(X_valid_meta)

valid_tree_rank = within_user_rank(valid_users, valid_tree)
valid_eb_rank = within_user_rank(valid_users, valid_eb)
valid_stack_rank = within_user_rank(valid_users, valid_stack)
valid_equal_rank = 0.5 * valid_tree_rank + 0.5 * valid_eb_rank

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if inc_valid.size != valid_users.size:
    raise RuntimeError("Trusted incumbent validation length mismatch")
inc_valid_rank = within_user_rank(valid_users, inc_valid)

own_valid = {
    "boosted_tree": valid_tree_rank,
    "empirical_bayes": valid_eb_rank,
    "equal_rank_ensemble": valid_equal_rank,
    "temporal_stack": valid_stack_rank,
}

candidate_scores = {}
candidate_specs = {}
candidate_metrics = {}

candidate_scores["trusted_incumbent"] = inc_valid_rank
candidate_specs["trusted_incumbent"] = ("incumbent", 0.0)
candidate_metrics["trusted_incumbent"] = float(
    evaluate(valid_users, valid.y, inc_valid_rank)["primary"]
)

for family, score in own_valid.items():
    standalone_name = family + "_standalone"
    candidate_scores[standalone_name] = score
    candidate_specs[standalone_name] = (family, 1.0)
    candidate_metrics[standalone_name] = float(
        evaluate(valid_users, valid.y, score)["primary"]
    )

    for alpha in (0.15, 0.30, 0.50, 0.70):
        name = family + "_incblend_" + f"{alpha:.2f}"
        blended = (1.0 - alpha) * inc_valid_rank + alpha * score
        candidate_scores[name] = blended
        candidate_specs[name] = (family, alpha)
        candidate_metrics[name] = float(
            evaluate(valid_users, valid.y, blended)["primary"]
        )

winner = max(candidate_metrics, key=candidate_metrics.get)
winner_family, winner_alpha = candidate_specs[winner]
valid_scores = candidate_scores[winner]
metrics = evaluate(valid_users, valid.y, valid_scores)

rank_corr = np.corrcoef(valid_tree_rank, valid_eb_rank)[0, 1]
cold_valid = sorted_lookup_counts(train_users, valid_users) == 0

print(
    "FINDINGS stack_diagnostics "
    + json.dumps({
        "base_rows": int(base_idx.size),
        "meta_rows": int(meta_idx.size),
        "tree_eb_rank_correlation": float(rank_corr),
        "valid_cold_row_fraction": float(cold_valid.mean()),
        "winner": winner,
        "winner_family": winner_family,
        "own_weight": float(winner_alpha),
    }, sort_keys=True)
)
print("CANDIDATES " + json.dumps(candidate_metrics, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    raw_valid = (
        valid_stack_rank
        if winner_family == "incumbent"
        else own_valid[winner_family]
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(raw_valid, dtype=np.float64),
    )

# Test is feature-only. No test labels are accessed.
test = load("test")
test_users = np.asarray(test.user_id, dtype=np.int64)

X_test = build_tree_matrix(test)
test_tree = tree_full_model.predict(X_test)
test_eb = predict_eb(eb_full_model, test)

test_list_n = list_sizes(test_users)
X_test_meta = build_meta_features(
    test_tree,
    test_eb,
    test_users,
    train_users,
    test_list_n,
)
test_stack = stack_model.predict(X_test_meta)

test_tree_rank = within_user_rank(test_users, test_tree)
test_eb_rank = within_user_rank(test_users, test_eb)
test_equal_rank = 0.5 * test_tree_rank + 0.5 * test_eb_rank
test_stack_rank = within_user_rank(test_users, test_stack)

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if inc_test.size != test_users.size:
    raise RuntimeError("Trusted incumbent test length mismatch")
inc_test_rank = within_user_rank(test_users, inc_test)

own_test = {
    "boosted_tree": test_tree_rank,
    "empirical_bayes": test_eb_rank,
    "equal_rank_ensemble": test_equal_rank,
    "temporal_stack": test_stack_rank,
}

if winner_family == "incumbent":
    test_scores = inc_test_rank
else:
    test_scores = (
        (1.0 - winner_alpha) * inc_test_rank
        + winner_alpha * own_test[winner_family]
    )

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)