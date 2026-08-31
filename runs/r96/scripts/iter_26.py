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
SEED = 94217
THREADS = min(16, os.cpu_count() or 1)
np.random.seed(SEED)

HALF_LIFE = 4.0
SVD_RANK = 32

RATE_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "hour", "upload_type", "music_type", "video_type",
    "onehot_feat2", "onehot_feat3", "onehot_feat7", "onehot_feat8",
    "user_active_degree", "fans_user_num_range",
    "follow_user_num_range", "friend_user_num_range",
    "register_days_range", "is_video_author", "is_live_streamer",
    "music_type", "upload_type",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def recency_weights(dates, half_life=HALF_LIFE):
    dates = np.asarray(dates, dtype=np.int32)
    age = dates.max() - dates
    w = np.power(0.5, age.astype(np.float64) / half_life)
    w /= max(float(np.mean(w)), 1e-12)
    return w


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores.copy()

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    ordered_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = ordered_users[1:] != ordered_users[:-1]
    start_index = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = ordered_users[:-1] != ordered_users[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(np.concatenate((
        np.asarray([-1], dtype=np.int64),
        end_positions,
    )))
    row_sizes = np.repeat(sizes, sizes)
    positions = np.arange(n, dtype=np.int64) - start_index

    ranked_ordered = (
        positions.astype(np.float64) + 0.5
    ) / row_sizes.astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_ordered
    return result


def build_additive_features(train, query, train_y, train_w,
                            leave_one_out=False):
    global_rate = float(np.sum(train_w * train_y) / np.sum(train_w))
    global_logit = np.log(
        np.clip(global_rate, 1e-5, 1 - 1e-5)
        / np.clip(1 - global_rate, 1e-5, 1 - 1e-5)
    )

    columns = []
    for field in RATE_FIELDS:
        cardinality = int(FEATURE_CARDINALITIES[field])
        tr_ids = np.asarray(train.X[field], dtype=np.int64)
        qu_ids = np.asarray(query.X[field], dtype=np.int64)

        total = np.bincount(
            tr_ids, weights=train_w, minlength=cardinality
        ).astype(np.float64)
        positive = np.bincount(
            tr_ids, weights=train_w * train_y, minlength=cardinality
        ).astype(np.float64)

        if leave_one_out:
            denominator = total[qu_ids] - train_w
            numerator = positive[qu_ids] - train_w * train_y
        else:
            denominator = total[qu_ids]
            numerator = positive[qu_ids]

        # More shrinkage for identity fields, whose future stability is lower.
        if field == "user_id":
            prior = 80.0
        elif field in ("video_id", "author_id"):
            prior = 45.0
        else:
            prior = 20.0

        rate = (
            numerator + prior * global_rate
        ) / np.maximum(denominator + prior, 1e-8)
        rate = np.clip(rate, 1e-5, 1 - 1e-5)
        logit = np.log(rate / (1.0 - rate)) - global_logit
        columns.append(np.clip(logit, -5.0, 5.0).astype(np.float32))

    for field in NUM_FIELDS:
        values = np.asarray(query.num[field], dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = np.log1p(np.maximum(values, 0.0))
        columns.append(values.astype(np.float32))

    return np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.float32)


def fit_additive_model(train, valid, test, y, weights):
    x_train = build_additive_features(
        train, train, y, weights, leave_one_out=True
    )
    x_valid = build_additive_features(
        train, valid, y, weights, leave_one_out=False
    )
    x_test = build_additive_features(
        train, test, y, weights, leave_one_out=False
    )

    center = np.median(x_train, axis=0).astype(np.float64)
    q25 = np.percentile(x_train, 25, axis=0)
    q75 = np.percentile(x_train, 75, axis=0)
    scale = np.maximum(q75 - q25, 0.05)

    def transform(x):
        z = (x.astype(np.float64) - center[None, :]) / scale[None, :]
        return np.clip(z, -8.0, 8.0)

    xt = transform(x_train)
    xv = transform(x_valid)
    xe = transform(x_test)

    # Weighted ridge logistic regression via a few IRLS steps.
    design = np.column_stack([
        np.ones(len(xt), dtype=np.float64), xt
    ])
    beta = np.zeros(design.shape[1], dtype=np.float64)
    beta[0] = np.log(np.mean(y) / (1.0 - np.mean(y)))
    penalty = np.full(design.shape[1], 2.0, dtype=np.float64)
    penalty[0] = 0.0

    for step in range(7):
        eta = np.clip(design @ beta, -15.0, 15.0)
        prob = 1.0 / (1.0 + np.exp(-eta))
        curvature = np.maximum(prob * (1.0 - prob), 1e-4)
        working = eta + (y - prob) / curvature
        effective_w = weights * curvature

        lhs = design.T @ (design * effective_w[:, None])
        lhs.flat[::lhs.shape[0] + 1] += penalty
        rhs = design.T @ (effective_w * working)
        beta = np.linalg.solve(lhs, rhs)

    valid_score = beta[0] + xv @ beta[1:]
    test_score = beta[0] + xe @ beta[1:]

    del x_train, x_valid, x_test, xt, xv, xe, design
    gc.collect()
    return valid_score, test_score


def fit_spectral_model(train, valid, test, y, weights):
    users = np.asarray(train.user_id, dtype=np.int64)
    videos = np.asarray(train.video_id, dtype=np.int64)
    n_users = int(FEATURE_CARDINALITIES["user_id"])
    n_videos = int(FEATURE_CARDINALITIES["video_id"])

    signed = (2.0 * y - 1.0) * weights
    user_degree = np.bincount(
        users, weights=weights, minlength=n_users
    ).astype(np.float64)
    video_degree = np.bincount(
        videos, weights=weights, minlength=n_videos
    ).astype(np.float64)

    normalization = np.sqrt(
        np.maximum(user_degree[users], 1.0)
        * np.maximum(video_degree[videos], 1.0)
    )
    values = signed / normalization

    matrix = sparse.coo_matrix(
        (values, (users, videos)),
        shape=(n_users, n_videos),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()

    u, singular, vt = svds(
        matrix,
        k=SVD_RANK,
        which="LM",
        random_state=SEED,
        tol=1e-3,
        maxiter=500,
    )
    order = np.argsort(singular)[::-1]
    singular = singular[order]
    u = u[:, order]
    vt = vt[order, :]

    user_factor = u * np.sqrt(singular)[None, :]
    video_factor = vt.T * np.sqrt(singular)[None, :]

    def predict(split):
        uid = np.asarray(split.user_id, dtype=np.int64)
        vid = np.asarray(split.video_id, dtype=np.int64)
        return np.sum(
            user_factor[uid] * video_factor[vid], axis=1
        ).astype(np.float64)

    valid_score = predict(valid)
    test_score = predict(test)

    del matrix, u, vt, user_factor, video_factor
    gc.collect()
    return valid_score, test_score


def exposure_features(split):
    n = len(split.user_id)
    uid = np.asarray(split.user_id, dtype=np.int64)
    time_ms = np.asarray(split.time_ms, dtype=np.int64)
    dates = np.asarray(split.date, dtype=np.int32)
    hours = np.asarray(split.X["hour"], dtype=np.float64)

    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, time_ms, dates, uid))
    sorted_uid = uid[order]
    sorted_date = dates[order]

    new_sequence = np.empty(n, dtype=bool)
    new_sequence[0] = True
    new_sequence[1:] = (
        (sorted_uid[1:] != sorted_uid[:-1])
        | (sorted_date[1:] != sorted_date[:-1])
    )
    sequence_start = np.maximum.accumulate(
        np.where(new_sequence, np.arange(n, dtype=np.int64), 0)
    )
    position_sorted = np.arange(n, dtype=np.int64) - sequence_start

    sequence_id = np.cumsum(new_sequence) - 1
    sequence_sizes = np.bincount(sequence_id)
    size_sorted = sequence_sizes[sequence_id]

    position = np.empty(n, dtype=np.float64)
    sequence_size = np.empty(n, dtype=np.float64)
    position[order] = position_sorted
    sequence_size[order] = size_sorted

    # Same-timestamp batch sizes are computed without labels.
    batch_key_change = np.empty(n, dtype=bool)
    sorted_time = time_ms[order]
    batch_key_change[0] = True
    batch_key_change[1:] = (
        (sorted_uid[1:] != sorted_uid[:-1])
        | (sorted_date[1:] != sorted_date[:-1])
        | (sorted_time[1:] != sorted_time[:-1])
    )
    batch_id = np.cumsum(batch_key_change) - 1
    batch_sizes = np.bincount(batch_id)
    batch_size_sorted = batch_sizes[batch_id]
    batch_size = np.empty(n, dtype=np.float64)
    batch_size[order] = batch_size_sorted

    frac = (position + 0.5) / np.maximum(sequence_size, 1.0)
    hour_angle = 2.0 * np.pi * hours / 24.0

    features = np.column_stack([
        np.log1p(position),
        np.log1p(sequence_size),
        frac,
        frac * frac,
        np.log1p(batch_size),
        np.sin(hour_angle),
        np.cos(hour_angle),
        np.asarray(split.X["tab"], dtype=np.float64),
        np.asarray(split.X["user_active_degree"], dtype=np.float64),
        np.log1p(np.maximum(
            np.asarray(split.num["duration_ms"], dtype=np.float64), 0.0
        )),
    ])
    return np.nan_to_num(
        features, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)


def fit_exposure_hazard(train, valid, test, y, weights):
    x_train = exposure_features(train)
    x_valid = exposure_features(valid)
    x_test = exposure_features(test)

    dataset = lgb.Dataset(
        x_train,
        label=y,
        weight=weights,
        free_raw_data=True,
    )
    params = {
        "objective": "binary",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_data_in_leaf": 1000,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 3.0,
        "max_bin": 127,
        "num_threads": THREADS,
        "seed": SEED,
        "verbose": -1,
    }
    model = lgb.train(params, dataset, num_boost_round=180)
    valid_score = model.predict(x_valid)
    test_score = model.predict(x_test)

    del x_train, x_valid, x_test, dataset, model
    gc.collect()
    return valid_score, test_score


train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float64)
train_weights = recency_weights(train.date)

add_valid, add_test = fit_additive_model(
    train, valid, test, y_train, train_weights
)
print("FINDINGS " + json.dumps({
    "family": "recency_additive_target_statistics",
    "status": "fit_complete",
}))

svd_valid, svd_test = fit_spectral_model(
    train, valid, test, y_train, train_weights
)
print("FINDINGS " + json.dumps({
    "family": "normalized_signed_spectral",
    "rank": SVD_RANK,
    "status": "fit_complete",
}))

hazard_valid, hazard_test = fit_exposure_hazard(
    train, valid, test, y_train, train_weights
)
print("FINDINGS " + json.dumps({
    "family": "exposure_position_hazard",
    "status": "fit_complete",
}))

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

valid_family_raw = {
    "additive": add_valid,
    "spectral": svd_valid,
    "hazard": hazard_valid,
}
test_family_raw = {
    "additive": add_test,
    "spectral": svd_test,
    "hazard": hazard_test,
}

valid_ranks = {
    name: rank_percentile(valid.user_id, score)
    for name, score in valid_family_raw.items()
}
test_ranks = {
    name: rank_percentile(test.user_id, score)
    for name, score in test_family_raw.items()
}
inc_valid_rank = rank_percentile(valid.user_id, inc_valid)
inc_test_rank = rank_percentile(test.user_id, inc_test)

candidate_valid = {"trusted_incumbent": inc_valid}
candidate_test = {"trusted_incumbent": inc_test}
candidate_raw = {"trusted_incumbent": valid_ranks["additive"]}

for name in valid_ranks:
    candidate_valid[name + "_standalone"] = valid_family_raw[name]
    candidate_test[name + "_standalone"] = test_family_raw[name]
    candidate_raw[name + "_standalone"] = valid_family_raw[name]

    for alpha in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
        key = f"{name}_incumbent_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * valid_ranks[name]
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank
            + alpha * test_ranks[name]
        )
        candidate_raw[key] = valid_ranks[name]

ensemble_specs = {
    "additive_spectral": ("additive", "spectral"),
    "additive_hazard": ("additive", "hazard"),
    "spectral_hazard": ("spectral", "hazard"),
    "three_family": ("additive", "spectral", "hazard"),
}

for ensemble_name, members in ensemble_specs.items():
    ev = np.mean(
        np.stack([valid_ranks[m] for m in members], axis=1), axis=1
    )
    et = np.mean(
        np.stack([test_ranks[m] for m in members], axis=1), axis=1
    )

    candidate_valid[ensemble_name + "_standalone"] = ev
    candidate_test[ensemble_name + "_standalone"] = et
    candidate_raw[ensemble_name + "_standalone"] = ev

    for alpha in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
        key = f"{ensemble_name}_incumbent_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank + alpha * ev
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank + alpha * et
        )
        candidate_raw[key] = ev

candidate_metrics = {
    name: evaluate(valid.user_id, valid.y, scores)
    for name, scores in candidate_valid.items()
}
best_name = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"])
)
best_metrics = candidate_metrics[best_name]
best_valid = np.asarray(candidate_valid[best_name], dtype=np.float64)
best_test = np.asarray(candidate_test[best_name], dtype=np.float64)

print("CANDIDATES " + json.dumps({
    name: float(metrics["primary"])
    for name, metrics in candidate_metrics.items()
}, sort_keys=True))

print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "half_life_days": HALF_LIFE,
    "families": list(valid_family_raw.keys()),
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        best_valid,
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        best_test,
    )
    if best_name == "trusted_incumbent" or "incumbent" in best_name:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[best_name], dtype=np.float64),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))