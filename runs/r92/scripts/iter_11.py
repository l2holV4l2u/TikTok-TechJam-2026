import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load
from pipeline.evaluate import evaluate


START = time.time()
SEED = 18427
THREADS = max(1, min(8, os.cpu_count() or 1))
np.random.seed(SEED)

FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "hour",
    "is_video_author",
    "video_type",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates)
    unique_dates = np.unique(dates)
    day_index = np.searchsorted(unique_dates, dates)
    age = (len(unique_dates) - 1 - day_index).astype(np.float32)
    weights = np.exp2(-age / float(half_life))
    return (weights / np.mean(weights)).astype(np.float32)


def make_features(split):
    categorical = np.column_stack([
        np.asarray(split.X[name], dtype=np.float32)
        for name in FIELDS
    ])

    numeric_columns = []
    for name in NUM_FIELDS:
        value = np.asarray(split.num[name], dtype=np.float32)
        missing = ~np.isfinite(value)
        clean = np.where(missing, 0.0, np.maximum(value, 0.0))
        numeric_columns.append(np.log1p(clean).astype(np.float32))
        numeric_columns.append(missing.astype(np.float32))

    numeric = np.column_stack(numeric_columns)
    return np.concatenate([categorical, numeric], axis=1)


def user_day_order_and_groups(split):
    users = np.asarray(split.user_id, dtype=np.int64)
    dates = np.asarray(split.date, dtype=np.int64)
    rows = np.arange(len(users), dtype=np.int64)

    order = np.lexsort((rows, dates, users))
    su = users[order]
    sd = dates[order]

    first = np.empty(len(order), dtype=bool)
    first[0] = True
    first[1:] = (su[1:] != su[:-1]) | (sd[1:] != sd[:-1])
    starts = np.flatnonzero(first)
    groups = np.diff(np.r_[starts, len(order)]).astype(np.int32)
    return order, groups


def combined_user_day_order(parts):
    users = np.concatenate([
        np.asarray(part.user_id, dtype=np.int64) for part in parts
    ])
    dates = np.concatenate([
        np.asarray(part.date, dtype=np.int64) for part in parts
    ])
    rows = np.arange(len(users), dtype=np.int64)

    order = np.lexsort((rows, dates, users))
    su = users[order]
    sd = dates[order]

    first = np.empty(len(order), dtype=bool)
    first[0] = True
    first[1:] = (su[1:] != su[:-1]) | (sd[1:] != sd[:-1])
    starts = np.flatnonzero(first)
    groups = np.diff(np.r_[starts, len(order)]).astype(np.int32)
    return order, groups


def fit_lambdamart(x, y, dates, order, groups, rounds=210):
    weights = recency_weights(dates, half_life=4.0)

    dataset = lgb.Dataset(
        x[order],
        label=np.asarray(y, dtype=np.int8)[order],
        weight=weights[order],
        group=groups,
        categorical_feature=list(range(len(FIELDS))),
        free_raw_data=False,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "lambdarank_truncation_level": 10,
        "label_gain": [0, 1],
        "learning_rate": 0.045,
        "num_leaves": 47,
        "max_depth": -1,
        "min_data_in_leaf": 180,
        "max_bin": 127,
        "feature_fraction": 0.88,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "lambda_l1": 0.08,
        "lambda_l2": 3.0,
        "min_gain_to_split": 0.003,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "num_threads": THREADS,
        "force_col_wise": True,
        "verbose": -1,
    }
    return lgb.train(params, dataset, num_boost_round=rounds)


def fit_residual_svd(parts, labels, rank=20):
    users = np.concatenate([
        np.asarray(part.user_id, dtype=np.int64) for part in parts
    ])
    videos = np.concatenate([
        np.asarray(part.video_id, dtype=np.int64) for part in parts
    ])
    dates = np.concatenate([
        np.asarray(part.date, dtype=np.int64) for part in parts
    ])
    labels = np.asarray(labels, dtype=np.float32)

    n_users = int(max(
        30000,
        users.max(initial=0) + 1,
    ))
    n_videos = int(max(
        8000,
        videos.max(initial=0) + 1,
    ))

    weights = recency_weights(dates, half_life=4.0)
    global_rate = float(np.sum(weights * labels) / np.sum(weights))

    # Signed, recency-weighted residuals prevent the decomposition from
    # reducing to positive-only popularity. Duplicate user-video exposures
    # are summed by CSR conversion.
    values = weights * (labels - global_rate)
    matrix = sparse.coo_matrix(
        (values, (users, videos)),
        shape=(n_users, n_videos),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()

    k = min(rank, min(matrix.shape) - 1)
    u, singular, vt = svds(
        matrix,
        k=k,
        which="LM",
        random_state=SEED,
        return_singular_vectors=True,
    )

    descending = np.argsort(singular)[::-1]
    singular = singular[descending].astype(np.float32)
    u = u[:, descending].astype(np.float32)
    vt = vt[descending].astype(np.float32)

    user_factors = u * singular[None, :]
    item_factors = vt.T.copy()
    return user_factors, item_factors


def predict_svd(split, user_factors, item_factors):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    scores = np.zeros(len(users), dtype=np.float32)

    valid = (
        (users >= 0)
        & (users < len(user_factors))
        & (videos >= 0)
        & (videos < len(item_factors))
    )
    if np.any(valid):
        scores[valid] = np.einsum(
            "ij,ij->i",
            user_factors[users[valid]],
            item_factors[videos[valid]],
            optimize=True,
        )
    return scores


def within_user_ranks(user_ids, scores):
    users = np.asarray(user_ids)
    values = np.asarray(scores)
    n = len(values)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, values, users))
    sorted_users = users[order]

    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(first)
    group_id = np.cumsum(first) - 1
    local = np.arange(n, dtype=np.int64) - starts[group_id]
    sizes = np.diff(np.r_[starts, n])

    ranks_sorted = (
        local.astype(np.float64) + 0.5
    ) / sizes[group_id].astype(np.float64)

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)

x_train = make_features(train)
x_valid = make_features(valid)

train_order, train_groups = user_day_order_and_groups(train)
rank_model = fit_lambdamart(
    x_train,
    y_train,
    np.asarray(train.date),
    train_order,
    train_groups,
    rounds=210,
)
lambda_valid = rank_model.predict(
    x_valid,
    num_iteration=rank_model.current_iteration(),
).astype(np.float64)

svd_users, svd_items = fit_residual_svd(
    [train],
    y_train,
    rank=20,
)
svd_valid = predict_svd(
    valid,
    svd_users,
    svd_items,
).astype(np.float64)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)

inc_rank = within_user_ranks(valid.user_id, inc_valid)
lambda_rank = within_user_ranks(valid.user_id, lambda_valid)
svd_rank = within_user_ranks(valid.user_id, svd_valid)

lambda_metrics = evaluate(valid.user_id, y_valid, lambda_valid)
svd_metrics = evaluate(valid.user_id, y_valid, svd_valid)

candidates = {
    "lambdamart_raw": float(lambda_metrics["primary"]),
    "residual_svd_raw": float(svd_metrics["primary"]),
}
records = {}

# beta controls the composition of the two new, structurally different
# families. alpha controls how much of that own-model composition is added
# to the trusted incumbent.
for beta in (0.0, 0.25, 0.50, 0.75, 1.0):
    own_scores = beta * lambda_rank + (1.0 - beta) * svd_rank
    own_metrics = evaluate(valid.user_id, y_valid, own_scores)
    own_key = "own_mix_lambda_" + str(beta)
    candidates[own_key] = float(own_metrics["primary"])

    for alpha in (0.0, 0.15, 0.30, 0.50, 0.70, 1.0):
        scores = (1.0 - alpha) * inc_rank + alpha * own_scores
        metrics = evaluate(valid.user_id, y_valid, scores)
        key = "blend_beta_" + str(beta) + "_alpha_" + str(alpha)
        candidates[key] = float(metrics["primary"])
        records[key] = {
            "beta": float(beta),
            "alpha": float(alpha),
            "scores": scores,
            "own_scores": own_scores,
            "metrics": metrics,
        }

winner_name = max(
    records,
    key=lambda name: records[name]["metrics"]["primary"],
)
winner = records[winner_name]

print("CANDIDATES " + json.dumps(candidates, sort_keys=True))
print("FINDINGS " + json.dumps({
    "winner": winner_name,
    "winner_lambda_weight": winner["beta"],
    "winner_own_weight": winner["alpha"],
    "incumbent_primary": float(
        evaluate(valid.user_id, y_valid, inc_valid)["primary"]
    ),
    "lambdamart_raw_primary": float(lambda_metrics["primary"]),
    "lambdamart_raw_gauc": float(lambda_metrics["gauc"]),
    "lambdamart_raw_ndcg5": float(lambda_metrics["ndcg@5"]),
    "residual_svd_raw_primary": float(svd_metrics["primary"]),
    "svd_rank": 20,
    "train_user_day_groups": int(len(train_groups)),
    "train_singleton_group_share": float(np.mean(train_groups == 1)),
}, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(winner["scores"], dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(winner["own_scores"], dtype=np.float64),
    )

# Refit the identical two-family recipe on train + validation.
test = load("test")
y_combined = np.concatenate([
    y_train,
    y_valid,
])
dates_combined = np.concatenate([
    np.asarray(train.date),
    np.asarray(valid.date),
])

x_valid_refit = make_features(valid)
x_combined = np.concatenate([
    x_train,
    x_valid_refit,
], axis=0)
x_test = make_features(test)

combined_order, combined_groups = combined_user_day_order([train, valid])
final_rank_model = fit_lambdamart(
    x_combined,
    y_combined,
    dates_combined,
    combined_order,
    combined_groups,
    rounds=210,
)
lambda_test = final_rank_model.predict(
    x_test,
    num_iteration=final_rank_model.current_iteration(),
).astype(np.float64)

del final_rank_model, x_combined, x_test
gc.collect()

final_svd_users, final_svd_items = fit_residual_svd(
    [train, valid],
    y_combined,
    rank=20,
)
svd_test = predict_svd(
    test,
    final_svd_users,
    final_svd_items,
).astype(np.float64)

inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_test_rank = within_user_ranks(test.user_id, inc_test)
lambda_test_rank = within_user_ranks(test.user_id, lambda_test)
svd_test_rank = within_user_ranks(test.user_id, svd_test)

beta = winner["beta"]
alpha = winner["alpha"]
own_test = beta * lambda_test_rank + (1.0 - beta) * svd_test_rank
test_scores = (1.0 - alpha) * inc_test_rank + alpha * own_test

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

metrics = winner["metrics"]
elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))