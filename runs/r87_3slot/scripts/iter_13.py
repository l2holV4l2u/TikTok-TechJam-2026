import os
import time
import json
import gc
import random
import numpy as np
import lightgbm as lgb
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 74031
THREADS = max(1, min(8, os.cpu_count() or 1))
np.random.seed(SEED)
random.seed(SEED)

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "duration_bucket", "upload_type", "music_type",
    "onehot_feat3", "onehot_feat8", "onehot_feat1",
    "onehot_feat7", "user_active_degree",
    "register_days_bucket", "register_days_range",
    "fans_user_num_range", "follow_user_num_range",
    "friend_user_num_range", "hour", "video_type",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]
SVD_RANK = 40


def make_tree_matrix(parts):
    columns = []

    for field in CAT_FIELDS:
        if len(parts) == 1:
            x = np.asarray(parts[0].X[field], dtype=np.float32)
        else:
            x = np.concatenate([
                np.asarray(p.X[field], dtype=np.float32) for p in parts
            ])
        columns.append(x)

    for field in NUM_FIELDS:
        if len(parts) == 1:
            x = np.asarray(parts[0].num[field], dtype=np.float32)
        else:
            x = np.concatenate([
                np.asarray(p.num[field], dtype=np.float32) for p in parts
            ])
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.log1p(np.maximum(x, 0.0)).astype(np.float32)
        columns.append(x)

    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def fit_random_forest(X, y):
    dataset = lgb.Dataset(
        X,
        label=np.asarray(y, dtype=np.float32),
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=True,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "rf",
        "num_leaves": 127,
        "max_depth": 12,
        "min_data_in_leaf": 300,
        "learning_rate": 1.0,
        "bagging_fraction": 0.70,
        "bagging_freq": 1,
        "feature_fraction": 0.75,
        "feature_fraction_bynode": 0.75,
        "extra_trees": True,
        "max_bin": 127,
        "max_cat_threshold": 32,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "lambda_l1": 0.0,
        "lambda_l2": 2.0,
        "verbosity": -1,
        "verbose": -1,
        "seed": SEED,
        "bagging_seed": SEED + 1,
        "feature_fraction_seed": SEED + 2,
        "extra_seed": SEED + 3,
        "num_threads": THREADS,
        "force_col_wise": True,
    }
    return lgb.train(params, dataset, num_boost_round=160)


def fit_puresvd(parts, labels):
    if len(parts) == 1:
        users = np.asarray(parts[0].X["user_id"], dtype=np.int32)
        videos = np.asarray(parts[0].X["video_id"], dtype=np.int32)
    else:
        users = np.concatenate([
            np.asarray(p.X["user_id"], dtype=np.int32) for p in parts
        ])
        videos = np.concatenate([
            np.asarray(p.X["video_id"], dtype=np.int32) for p in parts
        ])

    y = np.asarray(labels, dtype=np.float32)
    positive = y > 0.5
    users = users[positive]
    videos = videos[positive]

    n_users = int(FEATURE_CARDINALITIES["user_id"])
    n_videos = int(FEATURE_CARDINALITIES["video_id"])

    matrix = sparse.coo_matrix(
        (
            np.ones(len(users), dtype=np.float32),
            (users, videos),
        ),
        shape=(n_users, n_videos),
        dtype=np.float32,
    ).tocsr()

    # Repeated positives represent one established preference edge.
    matrix.sum_duplicates()
    matrix.data[:] = 1.0

    # Downweight universally popular videos, then normalize users so that
    # prolific users do not dominate the spectral factors.
    document_frequency = np.asarray(
        (matrix > 0).sum(axis=0)
    ).ravel().astype(np.float32)
    idf = np.log1p(
        float(n_users) / np.maximum(document_frequency, 1.0)
    ).astype(np.float32)
    weighted = matrix.multiply(idf).tocsr()

    row_norm = np.sqrt(
        np.asarray(weighted.multiply(weighted).sum(axis=1)).ravel()
    ).astype(np.float32)
    inv_norm = np.zeros_like(row_norm)
    good = row_norm > 0
    inv_norm[good] = 1.0 / row_norm[good]
    weighted = sparse.diags(inv_norm).dot(weighted).tocsr()

    rank = min(
        SVD_RANK,
        max(2, min(weighted.shape) - 2),
    )
    u, singular_values, vt = svds(
        weighted,
        k=rank,
        which="LM",
        return_singular_vectors=True,
        random_state=SEED,
        tol=1.0e-4,
        maxiter=1200,
    )
    order = np.argsort(singular_values)[::-1]
    singular_values = singular_values[order]
    u = u[:, order]
    vt = vt[order, :]

    user_factors = (
        u.astype(np.float32)
        * singular_values.astype(np.float32)[None, :]
    )
    item_factors = vt.T.astype(np.float32)
    return user_factors, item_factors


def predict_puresvd(factors, split):
    user_factors, item_factors = factors
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    videos = np.asarray(split.X["video_id"], dtype=np.int64)
    return np.einsum(
        "ij,ij->i",
        user_factors[users],
        item_factors[videos],
        optimize=True,
    ).astype(np.float64)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]
    positions = np.arange(n, dtype=np.int64)

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    start_positions = np.where(starts, positions, 0)
    start_positions = np.maximum.accumulate(start_positions)

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_positions = np.where(ends, positions + 1, n)
    end_positions = np.minimum.accumulate(end_positions[::-1])[::-1]

    group_size = end_positions - start_positions
    local_rank = positions - start_positions
    normalized = local_rank / np.maximum(group_size - 1, 1)

    result = np.empty(n, dtype=np.float64)
    result[order] = normalized
    return result


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

# Family 1: bagged extremely randomized categorical trees.
X_train = make_tree_matrix([train])
X_valid = make_tree_matrix([valid])
rf_model = fit_random_forest(X_train, y_train)
rf_valid = rf_model.predict(X_valid).astype(np.float64)
del rf_model, X_train, X_valid
gc.collect()

# Family 2: spectral collaborative filtering over positive preference edges.
svd_factors = fit_puresvd([train], y_train)
svd_valid = predict_puresvd(svd_factors, valid)
del svd_factors
gc.collect()

raw_predictions = {
    "extra_random_forest": rf_valid,
    "tfidf_puresvd": svd_valid,
}

candidate_results = {}
inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)
candidate_results["incumbent"] = float(inc_metrics["primary"])

best_scores = inc_valid.copy()
best_metrics = inc_metrics
best_name = "incumbent"
best_spec = {
    "family": "extra_random_forest",
    "blend_type": "score",
    "alpha": 0.0,
    "scale": 1.0,
}
best_raw = rf_valid.copy()

inc_std = max(float(np.std(inc_valid)), 1.0e-8)
inc_rank = within_user_rank(valid.user_id, inc_valid)
alphas = [0.10, 0.20, 0.30, 0.40, 0.55, 0.70]

for family, raw in raw_predictions.items():
    raw_metrics = evaluate(valid.user_id, y_valid, raw)
    candidate_results[family] = float(raw_metrics["primary"])

    if raw_metrics["primary"] > best_metrics["primary"]:
        best_scores = raw.copy()
        best_metrics = raw_metrics
        best_name = family
        best_raw = raw.copy()
        best_spec = {
            "family": family,
            "blend_type": "raw",
            "alpha": 1.0,
            "scale": 1.0,
        }

    raw_std = max(float(np.std(raw)), 1.0e-8)
    scale = inc_std / raw_std
    scaled_raw = raw * scale
    raw_rank = within_user_rank(valid.user_id, raw)

    for alpha in alphas:
        score_blend = (
            (1.0 - alpha) * inc_valid + alpha * scaled_raw
        )
        score_metrics = evaluate(
            valid.user_id, y_valid, score_blend
        )
        score_name = "%s_scoreblend_%.2f" % (family, alpha)
        candidate_results[score_name] = float(
            score_metrics["primary"]
        )

        if score_metrics["primary"] > best_metrics["primary"]:
            best_scores = score_blend.copy()
            best_metrics = score_metrics
            best_name = score_name
            best_raw = raw.copy()
            best_spec = {
                "family": family,
                "blend_type": "score",
                "alpha": float(alpha),
                "scale": float(scale),
            }

        rank_blend = (
            (1.0 - alpha) * inc_rank + alpha * raw_rank
        )
        rank_metrics = evaluate(
            valid.user_id, y_valid, rank_blend
        )
        rank_name = "%s_rankblend_%.2f" % (family, alpha)
        candidate_results[rank_name] = float(
            rank_metrics["primary"]
        )

        if rank_metrics["primary"] > best_metrics["primary"]:
            best_scores = rank_blend.copy()
            best_metrics = rank_metrics
            best_name = rank_name
            best_raw = raw.copy()
            best_spec = {
                "family": family,
                "blend_type": "rank",
                "alpha": float(alpha),
                "scale": 1.0,
            }

print(
    "CANDIDATES " + json.dumps(candidate_results, sort_keys=True),
    flush=True,
)
print(
    "FINDINGS rf_raw=%.6f svd_raw=%.6f selected=%s selected_family=%s "
    "selected_type=%s alpha=%.2f"
    % (
        candidate_results["extra_random_forest"],
        candidate_results["tfidf_puresvd"],
        best_name,
        best_spec["family"],
        best_spec["blend_type"],
        best_spec["alpha"],
    ),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw, dtype=np.float64),
    )

# Refit the selected recipe on train + validation, without reading test labels.
test = load("test")
inc_test = np.load(inc_test_path).astype(np.float64)
y_combined = np.concatenate([
    y_train,
    y_valid.astype(np.float32),
])

family = best_spec["family"]
alpha = best_spec["alpha"]
blend_type = best_spec["blend_type"]

if alpha == 0.0:
    test_scores = inc_test.copy()
elif family == "extra_random_forest":
    X_combined = make_tree_matrix([train, valid])
    X_test = make_tree_matrix([test])
    final_model = fit_random_forest(X_combined, y_combined)
    raw_test = final_model.predict(X_test).astype(np.float64)
    del final_model, X_combined, X_test
    gc.collect()

    if blend_type == "raw":
        test_scores = raw_test
    elif blend_type == "score":
        test_scores = (
            (1.0 - alpha) * inc_test
            + alpha * best_spec["scale"] * raw_test
        )
    else:
        test_scores = (
            (1.0 - alpha)
            * within_user_rank(test.user_id, inc_test)
            + alpha
            * within_user_rank(test.user_id, raw_test)
        )
else:
    final_factors = fit_puresvd([train, valid], y_combined)
    raw_test = predict_puresvd(final_factors, test)
    del final_factors
    gc.collect()

    if blend_type == "raw":
        test_scores = raw_test
    elif blend_type == "score":
        test_scores = (
            (1.0 - alpha) * inc_test
            + alpha * best_spec["scale"] * raw_test
        )
    else:
        test_scores = (
            (1.0 - alpha)
            * within_user_rank(test.user_id, inc_test)
            + alpha
            * within_user_rank(test.user_id, raw_test)
        )

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.4f}'
    % (
        best_metrics["primary"],
        best_metrics["gauc"],
        best_metrics["ndcg@5"],
        elapsed,
    ),
    flush=True,
)