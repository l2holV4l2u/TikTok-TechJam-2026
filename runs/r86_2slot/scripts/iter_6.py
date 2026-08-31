import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 73129
np.random.seed(SEED)

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour", "user_active_degree",
    "fans_user_num_range", "follow_user_num_range",
    "friend_user_num_range", "register_days_range",
    "video_type", "onehot_feat1", "onehot_feat3",
    "onehot_feat7", "onehot_feat8", "onehot_feat11",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
USER_CARD = int(FEATURE_CARDINALITIES["user_id"])
VIDEO_CARD = int(FEATURE_CARDINALITIES["video_id"])


def ordinal_day(dates):
    d = np.asarray(dates, dtype=np.int64)
    month = (d // 100) % 100
    day = d % 100
    return day + (month == 5) * 30


def recency_weights(dates, half_life=6.0):
    od = ordinal_day(dates)
    age = od.max() - od
    w = np.exp2(-age.astype(np.float32) / float(half_life))
    return w / np.maximum(w.mean(), 1e-8)


def make_lgb_matrix(split):
    columns = [
        np.asarray(split.X[f], dtype=np.float32) for f in CAT_FIELDS
    ]
    for f in NUM_FIELDS:
        x = np.asarray(split.num[f], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))

    od = ordinal_day(split.date).astype(np.float32)
    columns.append(od)
    columns.append((od % 7).astype(np.float32))
    return np.column_stack(columns).astype(np.float32, copy=False)


def user_sort_and_groups(user_ids):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[order]
    starts = np.r_[0, 1 + np.flatnonzero(
        sorted_users[1:] != sorted_users[:-1]
    )]
    groups = np.diff(np.r_[starts, len(order)]).astype(np.int32)
    return order, groups


def fit_lambdarank(split, y, weights, rounds=150):
    x = make_lgb_matrix(split)
    order, groups = user_sort_and_groups(split.user_id)

    dtrain = lgb.Dataset(
        x[order],
        label=np.asarray(y, dtype=np.float32)[order],
        weight=np.asarray(weights, dtype=np.float32)[order],
        group=groups,
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False,
    )
    params = {
        "objective": "lambdarank",
        "metric": "None",
        "lambdarank_truncation_level": 5,
        "label_gain": [0, 1],
        "learning_rate": 0.055,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 250,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "max_cat_threshold": 32,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "verbosity": -1,
        "verbose": -1,
        "num_threads": max(1, min(16, os.cpu_count() or 1)),
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "data_random_seed": SEED + 3,
        "force_col_wise": True,
    }
    model = lgb.train(params, dtrain, num_boost_round=rounds)
    del dtrain, order, groups
    return model


def predict_lambdarank(model, split):
    x = make_lgb_matrix(split)
    pred = model.predict(x, num_iteration=model.current_iteration())
    del x
    return np.asarray(pred, dtype=np.float64)


def positive_matrix(user_ids, video_ids, y, weights):
    mask = np.asarray(y) > 0
    rows = np.asarray(user_ids, dtype=np.int64)[mask]
    cols = np.asarray(video_ids, dtype=np.int64)[mask]
    data = np.asarray(weights, dtype=np.float32)[mask]
    mat = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(USER_CARD, VIDEO_CARD),
        dtype=np.float32,
    ).tocsr()
    mat.sum_duplicates()
    return mat


def fit_positive_svd(split, y, weights, rank=40):
    r = positive_matrix(
        split.X["user_id"], split.X["video_id"], y, weights
    )
    # Damp repeated positives so prolific users and videos do not dominate.
    if r.nnz:
        r.data = np.sqrt(r.data).astype(np.float32)
    user_degree = np.asarray(r.sum(axis=1)).ravel()
    item_degree = np.asarray(r.sum(axis=0)).ravel()
    left_scale = 1.0 / np.sqrt(np.maximum(user_degree, 1.0))
    right_scale = 1.0 / np.power(np.maximum(item_degree, 1.0), 0.25)
    normalized = sparse.diags(left_scale) @ r @ sparse.diags(right_scale)

    u, s, vt = svds(
        normalized.astype(np.float64),
        k=rank,
        which="LM",
        return_singular_vectors=True,
        random_state=SEED,
    )
    idx = np.argsort(s)[::-1]
    s = s[idx]
    u = u[:, idx]
    vt = vt[idx]
    user_factors = u * s[None, :]
    item_factors = vt.T
    del r, normalized, u, s, vt
    return (
        np.asarray(user_factors, dtype=np.float32),
        np.asarray(item_factors, dtype=np.float32),
    )


def predict_positive_svd(model, split):
    uf, vf = model
    u = np.asarray(split.X["user_id"], dtype=np.int64)
    v = np.asarray(split.X["video_id"], dtype=np.int64)
    return np.einsum("ij,ij->i", uf[u], vf[v]).astype(np.float64)


def fit_transition_model(split, y, weights, rank=36):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    rows = np.arange(len(users), dtype=np.int64)
    positives = np.asarray(y) > 0

    pos_rows = rows[positives]
    order = np.lexsort((
        pos_rows,
        times[positives],
        users[positives],
    ))
    p_users = users[positives][order]
    p_videos = videos[positives][order]
    p_weights = np.asarray(weights, dtype=np.float32)[positives][order]

    adjacent = p_users[1:] == p_users[:-1]
    src = p_videos[:-1][adjacent]
    dst = p_videos[1:][adjacent]
    edge_w = np.sqrt(
        p_weights[:-1][adjacent] * p_weights[1:][adjacent]
    ).astype(np.float32)

    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    data = np.concatenate([edge_w, edge_w])
    trans = sparse.coo_matrix(
        (data, (row, col)),
        shape=(VIDEO_CARD, VIDEO_CARD),
        dtype=np.float32,
    ).tocsr()
    trans.sum_duplicates()

    degree = np.asarray(trans.sum(axis=1)).ravel()
    scale = 1.0 / np.sqrt(np.maximum(degree, 1.0))
    norm = sparse.diags(scale) @ trans @ sparse.diags(scale)

    u, s, _ = svds(
        norm.astype(np.float64),
        k=rank,
        which="LM",
        return_singular_vectors=True,
        random_state=SEED + 17,
    )
    idx = np.argsort(s)[::-1]
    item_embedding = (
        u[:, idx] * np.sqrt(np.maximum(s[idx], 0.0))[None, :]
    ).astype(np.float32)

    history = positive_matrix(
        split.X["user_id"], split.X["video_id"], y, weights
    )
    counts = np.asarray(history.sum(axis=1)).ravel().astype(np.float32)
    profile = history @ item_embedding
    profile = np.asarray(profile, dtype=np.float32)
    profile /= np.maximum(counts[:, None], 1.0)

    del trans, norm, history, u, s
    return profile, item_embedding


def predict_transition(model, split):
    profile, item_embedding = model
    u = np.asarray(split.X["user_id"], dtype=np.int64)
    v = np.asarray(split.X["video_id"], dtype=np.int64)
    return np.einsum(
        "ij,ij->i", profile[u], item_embedding[v]
    ).astype(np.float64)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores.copy()

    order = np.lexsort((np.arange(n), scores, user_ids))
    su = user_ids[order]
    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = su[1:] != su[:-1]
    starts = np.flatnonzero(starts_mask)
    group_index = np.cumsum(starts_mask) - 1
    position = np.arange(n) - starts[group_index]
    sizes = np.diff(np.r_[starts, n])
    denominator = np.maximum(sizes[group_index] - 1, 1)
    ranked_sorted = position.astype(np.float64) / denominator
    ranked_sorted[sizes[group_index] == 1] = 0.5

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def best_blend(user_ids, labels, own_scores, incumbent_scores):
    own_rank = within_user_rank(user_ids, own_scores)
    incumbent_rank = within_user_rank(user_ids, incumbent_scores)
    best = None
    for alpha in [
        0.0, 0.10, 0.20, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.80, 0.90, 1.0,
    ]:
        score = alpha * own_rank + (1.0 - alpha) * incumbent_rank
        met = evaluate(user_ids, labels, score)
        result = (
            float(met["primary"]), float(alpha), score, met
        )
        if best is None or result[0] > best[0]:
            best = result
    return best


class JoinedSplit:
    pass


def join_splits(a, b):
    out = JoinedSplit()
    out.X = {
        f: np.concatenate([
            np.asarray(a.X[f]), np.asarray(b.X[f])
        ])
        for f in CAT_FIELDS
    }
    # Ensure fields needed by the collaborative models are present.
    for f in ["user_id", "video_id"]:
        if f not in out.X:
            out.X[f] = np.concatenate([
                np.asarray(a.X[f]), np.asarray(b.X[f])
            ])
    out.num = {
        f: np.concatenate([
            np.asarray(a.num[f]), np.asarray(b.num[f])
        ])
        for f in NUM_FIELDS
    }
    out.user_id = np.concatenate([
        np.asarray(a.user_id), np.asarray(b.user_id)
    ])
    out.video_id = np.concatenate([
        np.asarray(a.video_id), np.asarray(b.video_id)
    ])
    out.date = np.concatenate([
        np.asarray(a.date), np.asarray(b.date)
    ])
    out.time_ms = np.concatenate([
        np.asarray(a.time_ms), np.asarray(b.time_ms)
    ])
    return out


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.float32)
weights_train = recency_weights(train.date, half_life=6.0)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

family_predictions = {}

# Family 1: direct metric-aware tree ranking.
lambda_model = fit_lambdarank(
    train, y_train, weights_train, rounds=150
)
family_predictions["lambdarank"] = predict_lambdarank(
    lambda_model, valid
)
del lambda_model
gc.collect()

# Family 2: global low-rank positive interaction geometry.
svd_model = fit_positive_svd(
    train, y_train, weights_train, rank=40
)
family_predictions["positive_svd_cf"] = predict_positive_svd(
    svd_model, valid
)
del svd_model
gc.collect()

# Family 3: sequential positive-item transition geometry.
transition_model = fit_transition_model(
    train, y_train, weights_train, rank=36
)
family_predictions["positive_transition"] = predict_transition(
    transition_model, valid
)
del transition_model
gc.collect()

candidate_summary = {}
raw_summary = {}
selection = None

for name, pred in family_predictions.items():
    raw_met = evaluate(valid.user_id, valid.y, pred)
    raw_summary[name] = float(raw_met["primary"])

    blend = best_blend(
        valid.user_id, valid.y, pred, inc_valid
    )
    candidate_summary[name + "_blend"] = float(blend[0])

    if selection is None or blend[0] > selection["primary"]:
        selection = {
            "name": name,
            "primary": blend[0],
            "alpha": blend[1],
            "scores": blend[2],
            "metrics": blend[3],
            "raw": pred,
        }

print("FINDINGS raw_family_primary=" + json.dumps(
    raw_summary, sort_keys=True
))
print(
    "FINDINGS selected_family=%s own_rank_weight=%.2f"
    % (selection["name"], selection["alpha"])
)
print("CANDIDATES " + json.dumps(
    candidate_summary, sort_keys=True
))

valid_scores = np.asarray(selection["scores"], dtype=np.float64)
metrics = selection["metrics"]

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        valid_scores,
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(selection["raw"], dtype=np.float64),
    )

# Permitted identical-recipe refit on train + validation.
combined = join_splits(train, valid)
combined_y = np.concatenate([y_train, y_valid])
combined_weights = recency_weights(combined.date, half_life=6.0)
test = load("test")

if selection["name"] == "lambdarank":
    final_model = fit_lambdarank(
        combined, combined_y, combined_weights, rounds=150
    )
    own_test = predict_lambdarank(final_model, test)
    del final_model

elif selection["name"] == "positive_svd_cf":
    final_model = fit_positive_svd(
        combined, combined_y, combined_weights, rank=40
    )
    own_test = predict_positive_svd(final_model, test)
    del final_model

else:
    final_model = fit_transition_model(
        combined, combined_y, combined_weights, rank=36
    )
    own_test = predict_transition(final_model, test)
    del final_model

inc_test = np.load(inc_test_path).astype(np.float64)
test_scores = (
    selection["alpha"] * within_user_rank(test.user_id, own_test)
    + (1.0 - selection["alpha"])
    * within_user_rank(test.user_id, inc_test)
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": elapsed,
}))