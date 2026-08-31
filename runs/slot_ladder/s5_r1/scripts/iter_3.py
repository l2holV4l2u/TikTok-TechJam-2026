import os
import time
import json
import warnings
import numpy as np
import lightgbm as lgb
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")
START = time.time()
SEED = 2025
THREADS = max(1, min(8, os.cpu_count() or 1))

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "onehot_feat1",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def numeric_transform(split):
    cols = []
    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.sign(x) * np.log1p(np.abs(x))
        cols.append(x.astype(np.float32))
    return cols


def get_histories(split_name, history_names=None):
    hv = historical_features(split_name, key="video_id")
    ha = historical_features(split_name, key="author_id")
    merged = {}
    for k, v in hv.items():
        merged["video_" + k] = np.asarray(v, dtype=np.float32)
    for k, v in ha.items():
        merged["author_" + k] = np.asarray(v, dtype=np.float32)

    if history_names is None:
        history_names = sorted(merged.keys())
    arrays = []
    for name in history_names:
        x = merged[name]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        arrays.append(x.astype(np.float32))
    return history_names, arrays


def make_tree_matrix(split, split_name, history_names=None):
    cols = [
        np.asarray(split.X[name], dtype=np.float32)
        for name in CAT_FIELDS
    ]
    names, hcols = get_histories(split_name, history_names)
    cols.extend(hcols)
    cols.extend(numeric_transform(split))
    matrix = np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)
    feature_names = (
        CAT_FIELDS
        + names
        + ["num_" + name for name in NUM_FIELDS]
    )
    return matrix, names, feature_names


def group_order_and_sizes(user_ids):
    user_ids = np.asarray(user_ids)
    order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[order]
    if len(sorted_users) == 0:
        return order, np.empty(0, dtype=np.int32)
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(sorted_users)]).astype(np.int32)
    return order, sizes


def fit_binary(x, y, feature_names):
    ds = lgb.Dataset(
        x,
        label=np.asarray(y, dtype=np.float32),
        feature_name=feature_names,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "learning_rate": 0.055,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 250,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.86,
        "bagging_freq": 1,
        "lambda_l1": 0.15,
        "lambda_l2": 2.0,
        "max_cat_to_onehot": 16,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "seed": SEED,
        "num_threads": THREADS,
        "verbose": -1,
    }
    return lgb.train(params, ds, num_boost_round=240)


def fit_ranker(x, y, user_ids, feature_names):
    order, group = group_order_and_sizes(user_ids)
    xr = np.ascontiguousarray(x[order])
    yr = np.asarray(y, dtype=np.float32)[order]
    ds = lgb.Dataset(
        xr,
        label=yr,
        group=group,
        feature_name=feature_names,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "learning_rate": 0.045,
        "num_leaves": 47,
        "max_depth": -1,
        "min_data_in_leaf": 220,
        "feature_fraction": 0.84,
        "bagging_fraction": 0.88,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 3.0,
        "max_cat_to_onehot": 16,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "lambdarank_truncation_level": 12,
        "seed": SEED + 1,
        "num_threads": THREADS,
        "verbose": -1,
    }
    return lgb.train(params, ds, num_boost_round=180)


def fit_target_tables(train):
    y = np.asarray(train.y, dtype=np.float64)
    prior = float(y.mean())
    specs = [
        ("video_id", 15.0, 0.34),
        ("author_id", 25.0, 0.28),
        ("tag", 120.0, 0.16),
        ("duration_bucket", 500.0, 0.10),
        ("tab", 700.0, 0.07),
        ("upload_type", 300.0, 0.05),
    ]
    tables = {}
    for name, alpha, weight in specs:
        ids = np.asarray(train.X[name], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[name])
        cnt = np.bincount(ids, minlength=card).astype(np.float64)
        pos = np.bincount(ids, weights=y, minlength=card).astype(np.float64)
        rate = (pos + alpha * prior) / (cnt + alpha)
        tables[name] = (rate, weight)
    return prior, tables


def target_score(split, prior, tables):
    eps = 1e-5
    result = np.zeros(len(split.user_id), dtype=np.float64)
    total_weight = 0.0
    for name, (rate, weight) in tables.items():
        ids = np.asarray(split.X[name], dtype=np.int64)
        p = np.clip(rate[ids], eps, 1.0 - eps)
        result += weight * np.log(p / (1.0 - p))
        total_weight += weight
    if total_weight > 0:
        result /= total_weight
    return result


def fit_svd(train, rank=32):
    users = np.asarray(train.user_id, dtype=np.int64)
    videos = np.asarray(train.video_id, dtype=np.int64)
    y = np.asarray(train.y, dtype=np.float64)
    n_users = int(FEATURE_CARDINALITIES["user_id"])
    n_videos = int(FEATURE_CARDINALITIES["video_id"])

    sums = sparse.coo_matrix(
        (y, (users, videos)), shape=(n_users, n_videos)
    ).tocsr()
    counts = sparse.coo_matrix(
        (np.ones_like(y), (users, videos)), shape=(n_users, n_videos)
    ).tocsr()

    means = sums.copy()
    means.data = means.data / np.maximum(counts.data, 1.0)
    global_mean = float(y.mean())
    means.data = means.data - global_mean

    u, singular, vt = svds(
        means.astype(np.float32),
        k=rank,
        which="LM",
        random_state=SEED,
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order]
    u = u[:, order]
    vt = vt[order, :]
    root = np.sqrt(np.maximum(singular, 0.0))
    user_vec = (u * root[None, :]).astype(np.float32)
    video_vec = (vt.T * root[None, :]).astype(np.float32)
    return user_vec, video_vec


def svd_score(split, user_vec, video_vec):
    u = np.asarray(split.user_id, dtype=np.int64)
    v = np.asarray(split.video_id, dtype=np.int64)
    return np.einsum(
        "ij,ij->i", user_vec[u], video_vec[v], optimize=True
    ).astype(np.float64)


def within_user_standardize(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    max_id = int(user_ids.max()) + 1 if len(user_ids) else 0
    count = np.bincount(user_ids, minlength=max_id).astype(np.float64)
    sums = np.bincount(user_ids, weights=scores, minlength=max_id)
    squares = np.bincount(user_ids, weights=scores * scores, minlength=max_id)
    mean = sums / np.maximum(count, 1.0)
    var = squares / np.maximum(count, 1.0) - mean * mean
    sd = np.sqrt(np.maximum(var, 1e-8))
    return (scores - mean[user_ids]) / sd[user_ids]


def score_metrics(split, scores):
    return evaluate(split.user_id, split.y, np.asarray(scores, dtype=np.float64))


train = load("train")
valid = load("valid")

x_train, hist_names, feature_names = make_tree_matrix(train, "train")
x_valid, _, _ = make_tree_matrix(valid, "valid", hist_names)

binary_model = fit_binary(x_train, train.y, feature_names)
rank_model = fit_ranker(x_train, train.y, train.user_id, feature_names)

valid_binary = binary_model.predict(x_valid).astype(np.float64)
valid_rank = rank_model.predict(x_valid).astype(np.float64)

prior, target_tables = fit_target_tables(train)
valid_target = target_score(valid, prior, target_tables)

user_vec, video_vec = fit_svd(train, rank=32)
valid_svd = svd_score(valid, user_vec, video_vec)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

family_valid = {
    "lgb_binary": valid_binary,
    "lgb_lambdarank": valid_rank,
    "empirical_bayes": valid_target,
    "latent_svd": valid_svd,
}

candidate_scores = {}
candidate_arrays = {}
candidate_specs = {}

inc_metrics = score_metrics(valid, inc_valid)
candidate_scores["incumbent"] = float(inc_metrics["primary"])
candidate_arrays["incumbent"] = inc_valid
candidate_specs["incumbent"] = ("incumbent", 0.0)

inc_z = within_user_standardize(valid.user_id, inc_valid)
blend_weights = [0.20, 0.35, 0.50, 0.65]

for family, raw in family_valid.items():
    met = score_metrics(valid, raw)
    candidate_scores[family] = float(met["primary"])
    candidate_arrays[family] = raw
    candidate_specs[family] = (family, 1.0)

    raw_z = within_user_standardize(valid.user_id, raw)
    for weight in blend_weights:
        name = family + "_blend_" + str(weight)
        blended = (1.0 - weight) * inc_z + weight * raw_z
        bmet = score_metrics(valid, blended)
        candidate_scores[name] = float(bmet["primary"])
        candidate_arrays[name] = blended
        candidate_specs[name] = (family, weight)

best_name = max(candidate_scores, key=candidate_scores.get)
best_valid_scores = np.asarray(candidate_arrays[best_name], dtype=np.float64)
best_metrics = score_metrics(valid, best_valid_scores)
best_family, best_weight = candidate_specs[best_name]

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": best_name,
            "winner_family": best_family,
            "incumbent_primary": candidate_scores["incumbent"],
            "history_features": len(hist_names),
            "tree_features": len(feature_names),
        },
        sort_keys=True,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        best_valid_scores.astype(np.float64),
    )
    if best_family not in ("incumbent",):
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(family_valid[best_family], dtype=np.float64),
        )

test = load("test")

if best_family == "incumbent":
    test_scores = np.asarray(np.load(inc_test_path), dtype=np.float64)
else:
    if best_family in ("lgb_binary", "lgb_lambdarank"):
        x_test, _, _ = make_tree_matrix(test, "test", hist_names)
        if best_family == "lgb_binary":
            test_raw = binary_model.predict(x_test).astype(np.float64)
        else:
            test_raw = rank_model.predict(x_test).astype(np.float64)
    elif best_family == "empirical_bayes":
        test_raw = target_score(test, prior, target_tables)
    elif best_family == "latent_svd":
        test_raw = svd_score(test, user_vec, video_vec)
    else:
        raise RuntimeError("Unknown selected family: " + best_family)

    if best_weight >= 0.999:
        test_scores = np.asarray(test_raw, dtype=np.float64)
    else:
        inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
        inc_test_z = within_user_standardize(test.user_id, inc_test)
        raw_test_z = within_user_standardize(test.user_id, test_raw)
        test_scores = (
            (1.0 - best_weight) * inc_test_z
            + best_weight * raw_test_z
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
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(",", ":"),
    )
)