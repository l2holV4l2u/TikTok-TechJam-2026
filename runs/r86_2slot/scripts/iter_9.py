import os
import time
import json
import gc
import warnings
import numpy as np
import lightgbm as lgb
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")
START = time.time()
SEED = 73129
rng = np.random.default_rng(SEED)

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "duration_bucket", "upload_type", "music_type", "hour",
    "user_active_degree", "fans_user_num_range",
    "follow_user_num_range", "friend_user_num_range",
    "register_days_range", "video_type",
    "onehot_feat0", "onehot_feat1", "onehot_feat2",
    "onehot_feat3", "onehot_feat7", "onehot_feat8",
    "onehot_feat11", "onehot_feat12",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]


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


def ordinal_day(dates):
    d = np.asarray(dates, dtype=np.int64)
    month = (d // 100) % 100
    day = d % 100
    return day + (month == 5) * 30


def recency_weights(dates, half_life=6.0):
    day = ordinal_day(dates)
    age = day.max() - day
    w = np.exp2(-age.astype(np.float64) / float(half_life))
    return (w / np.maximum(w.mean(), 1e-12)).astype(np.float64)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, scores, users))
    su = users[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = su[1:] != su[:-1]
    starts = np.flatnonzero(starts_mask)
    group = np.cumsum(starts_mask) - 1
    pos = np.arange(n) - starts[group]
    sizes = np.diff(np.r_[starts, n])
    den = np.maximum(sizes[group] - 1, 1)

    sr = pos.astype(np.float64) / den
    sr[sizes[group] == 1] = 0.5
    out = np.empty(n, dtype=np.float64)
    out[order] = sr
    return out


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


def pair_key(a, b, b_base):
    return (
        np.asarray(a, dtype=np.uint64) * np.uint64(b_base)
        + np.asarray(b, dtype=np.uint64)
    )


def aggregate_rate(train_key, query_key, y, weights, smoothing,
                   prior=None, leave_one_out=False):
    train_key = np.asarray(train_key, dtype=np.uint64)
    query_key = np.asarray(query_key, dtype=np.uint64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if prior is None:
        prior = float(np.sum(weights * y) / np.sum(weights))

    unique, inverse = np.unique(train_key, return_inverse=True)
    count = np.bincount(
        inverse, weights=weights, minlength=len(unique)
    ).astype(np.float64)
    positive = np.bincount(
        inverse, weights=weights * y, minlength=len(unique)
    ).astype(np.float64)

    if leave_one_out:
        c = count[inverse] - weights
        s = positive[inverse] - weights * y
        return (
            (s + smoothing * prior) /
            (c + smoothing)
        ).astype(np.float32)

    loc = np.searchsorted(unique, query_key)
    found = loc < len(unique)
    safe = np.minimum(loc, max(len(unique) - 1, 0))
    if len(unique):
        found &= unique[safe] == query_key

    out = np.full(len(query_key), prior, dtype=np.float64)
    if len(unique):
        idx = safe[found]
        out[found] = (
            positive[idx] + smoothing * prior
        ) / (
            count[idx] + smoothing
        )
    return out.astype(np.float32)


def make_keys(split):
    user = np.asarray(split.user_id, dtype=np.uint64)
    video = np.asarray(split.video_id, dtype=np.uint64)
    author = np.asarray(split.X["author_id"], dtype=np.uint64)
    tag = np.asarray(split.X["tag"], dtype=np.uint64)
    duration = np.asarray(
        split.X["duration_bucket"], dtype=np.uint64
    )

    return {
        "video": video,
        "author": author,
        "tag": tag,
        "user_video": pair_key(user, video, 8192),
        "user_author": pair_key(user, author, 8192),
        "user_tag": pair_key(user, tag, 128),
        "user_duration": pair_key(user, duration, 32),
    }


SMOOTHING = {
    "video": 45.0,
    "author": 55.0,
    "tag": 120.0,
    "user_video": 10.0,
    "user_author": 14.0,
    "user_tag": 22.0,
    "user_duration": 28.0,
}


def target_statistics(fit_split, fit_y, query_split=None,
                      half_life=6.0):
    fit_y = np.asarray(fit_y, dtype=np.float64)
    weights = recency_weights(fit_split.date, half_life)
    prior = float(np.sum(weights * fit_y) / np.sum(weights))
    fit_keys = make_keys(fit_split)

    if query_split is None:
        columns = []
        for name in SMOOTHING:
            columns.append(aggregate_rate(
                fit_keys[name], fit_keys[name], fit_y, weights,
                SMOOTHING[name], prior=prior, leave_one_out=True
            ))
    else:
        query_keys = make_keys(query_split)
        columns = []
        for name in SMOOTHING:
            columns.append(aggregate_rate(
                fit_keys[name], query_keys[name], fit_y, weights,
                SMOOTHING[name], prior=prior, leave_one_out=False
            ))

    return np.column_stack(columns).astype(np.float32), prior


def empirical_scores(stats, variant):
    z = np.column_stack([
        safe_logit(stats[:, i]) for i in range(stats.shape[1])
    ])
    if variant == "entity":
        return (
            0.58 * z[:, 0] +
            0.30 * z[:, 1] +
            0.12 * z[:, 2]
        )
    if variant == "personal":
        return (
            0.42 * z[:, 0] +
            0.20 * z[:, 1] +
            0.08 * z[:, 2] +
            0.48 * z[:, 3] +
            0.40 * z[:, 4] +
            0.22 * z[:, 5] +
            0.15 * z[:, 6]
        )
    return (
        0.50 * z[:, 0] +
        0.25 * z[:, 1] +
        0.10 * z[:, 2] +
        0.28 * z[:, 3] +
        0.30 * z[:, 4] +
        0.14 * z[:, 5] +
        0.10 * z[:, 6]
    )


def raw_numeric(split):
    cols = []
    for f in NUM_FIELDS:
        x = np.asarray(split.num[f], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(np.log1p(np.maximum(x, 0.0)))
    return np.column_stack(cols).astype(np.float32)


def numeric_stats(split):
    x = raw_numeric(split)
    mean = x.mean(axis=0).astype(np.float32)
    std = np.maximum(x.std(axis=0), 0.1).astype(np.float32)
    return mean, std


def model_matrix(split, te, stats):
    cats = np.column_stack([
        np.asarray(split.X[f], dtype=np.float32)
        for f in CAT_FIELDS
    ])
    nums = raw_numeric(split)
    mean, std = stats
    nums = (nums - mean) / std

    hour = np.asarray(split.X["hour"], dtype=np.float32)
    hour_sin = np.sin(hour * (2.0 * np.pi / 24.0))
    hour_cos = np.cos(hour * (2.0 * np.pi / 24.0))
    context = np.column_stack([hour_sin, hour_cos]).astype(np.float32)

    return np.column_stack([
        cats, nums, te, context
    ]).astype(np.float32)


def fit_lgb(train_x, train_y, weights, valid_x=None,
            valid_y=None, rounds=320):
    categorical = list(range(len(CAT_FIELDS)))
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.055,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "max_cat_threshold": 64,
        "cat_l2": 12.0,
        "cat_smooth": 20.0,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "num_threads": max(1, min(16, os.cpu_count() or 1)),
        "verbose": -1,
    }
    dtrain = lgb.Dataset(
        train_x, label=train_y, weight=weights,
        categorical_feature=categorical,
        free_raw_data=False
    )

    if valid_x is not None:
        dvalid = lgb.Dataset(
            valid_x, label=valid_y,
            categorical_feature=categorical,
            reference=dtrain,
            free_raw_data=False
        )
        model = lgb.train(
            params, dtrain, num_boost_round=rounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(35, verbose=False)]
        )
    else:
        model = lgb.train(
            params, dtrain, num_boost_round=rounds
        )
    return model


def fit_svd_model(split, labels, rank=20):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    y = np.asarray(labels, dtype=np.float64)
    n_users = max(30000, int(users.max(initial=0)) + 1)
    n_videos = max(8000, int(videos.max(initial=0)) + 1)

    ones = sparse.coo_matrix(
        (np.ones(len(y), dtype=np.float64), (users, videos)),
        shape=(n_users, n_videos)
    ).tocsr()
    sums = sparse.coo_matrix(
        (y, (users, videos)), shape=(n_users, n_videos)
    ).tocsr()

    avg = sums.copy()
    avg.data = avg.data / np.maximum(ones.data, 1.0)
    prior = float(y.mean())
    avg.data = avg.data - prior

    k = min(rank, min(avg.shape) - 1)
    u, s, vt = svds(
        avg.astype(np.float32), k=k, which="LM",
        tol=1e-2, maxiter=120, random_state=SEED
    )
    order = np.argsort(s)[::-1]
    s = s[order]
    u = u[:, order]
    vt = vt[order]
    user_factors = (u * s[None, :]).astype(np.float32)
    item_factors = vt.T.astype(np.float32)
    return user_factors, item_factors


def predict_svd(model, split):
    uf, vf = model
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    out = np.zeros(len(users), dtype=np.float64)
    known = (
        (users >= 0) & (users < len(uf)) &
        (videos >= 0) & (videos < len(vf))
    )
    out[known] = np.sum(
        uf[users[known]] * vf[videos[known]], axis=1
    )
    return out


def best_incumbent_blend(users, labels, raw_scores, incumbent):
    raw_rank = within_user_rank(users, raw_scores)
    inc_rank = within_user_rank(users, incumbent)
    best = None
    for alpha in np.linspace(0.0, 1.0, 11):
        scores = alpha * raw_rank + (1.0 - alpha) * inc_rank
        metrics = evaluate(users, labels, scores)
        candidate = {
            "primary": float(metrics["primary"]),
            "alpha": float(alpha),
            "scores": scores,
            "metrics": metrics,
        }
        if best is None or candidate["primary"] > best["primary"]:
            best = candidate
    return best


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

# Leakage-free recency-weighted target statistics.
te_train, prior_train = target_statistics(
    train, y_train, query_split=None, half_life=6.0
)
te_valid, _ = target_statistics(
    train, y_train, query_split=valid, half_life=6.0
)

print(
    "FINDINGS target_prior=%.5f pair_rate_std_uv=%.5f pair_rate_std_ua=%.5f"
    % (
        prior_train,
        float(te_valid[:, 3].std()),
        float(te_valid[:, 4].std()),
    )
)

raw_candidates = {}
candidate_recipes = {}

# Non-parametric family, with three meaningfully different pooling rules.
for variant in ["entity", "balanced", "personal"]:
    p = empirical_scores(te_valid, variant)
    raw_candidates["empirical_" + variant] = p
    candidate_recipes["empirical_" + variant] = ("empirical", variant)

# Gradient-boosting family.
norm_stats = numeric_stats(train)
x_train = model_matrix(train, te_train, norm_stats)
x_valid = model_matrix(valid, te_valid, norm_stats)
train_weights = recency_weights(train.date, half_life=6.0).astype(np.float32)

gbm = fit_lgb(
    x_train, y_train, train_weights,
    valid_x=x_valid, valid_y=y_valid, rounds=340
)
gbm_valid = gbm.predict(
    x_valid, num_iteration=gbm.best_iteration
).astype(np.float64)
raw_candidates["lightgbm_target_stats"] = gbm_valid
candidate_recipes["lightgbm_target_stats"] = (
    "lightgbm", int(gbm.best_iteration)
)
print(
    "FINDINGS lightgbm_best_iteration=%d"
    % int(gbm.best_iteration)
)

del x_train, x_valid, te_train
gc.collect()

# Sparse latent collaborative family.
try:
    svd_model = fit_svd_model(train, y_train, rank=20)
    svd_valid = predict_svd(svd_model, valid)
    raw_candidates["sparse_svd"] = svd_valid
    candidate_recipes["sparse_svd"] = ("svd", 20)
except Exception as exc:
    print("FINDINGS svd_failed=%s" % repr(exc)[:180])
    svd_model = None

candidate_scores = {}
blend_records = {}
winner = None

for name, raw in raw_candidates.items():
    raw_metrics = evaluate(valid.user_id, y_valid, raw)
    candidate_scores[name + "_raw"] = float(raw_metrics["primary"])

    blended = best_incumbent_blend(
        valid.user_id, y_valid, raw, inc_valid
    )
    candidate_scores[name + "_blend"] = blended["primary"]
    blend_records[name] = blended

    if winner is None or blended["primary"] > winner["primary"]:
        winner = {
            "name": name,
            "primary": blended["primary"],
            "alpha": blended["alpha"],
            "scores": blended["scores"],
            "metrics": blended["metrics"],
            "raw": raw,
        }

print("CANDIDATES " + json.dumps(
    candidate_scores, sort_keys=True
))
print(
    "FINDINGS winner=%s alpha=%.2f raw_primary=%.6f"
    % (
        winner["name"],
        winner["alpha"],
        float(evaluate(
            valid.user_id, y_valid, winner["raw"]
        )["primary"]),
    )
)

# Refit the selected recipe on train + validation, then score test.
test = load("test")
joined = join_splits(train, valid)
y_joined = np.concatenate([y_train, y_valid]).astype(np.int8)
recipe = candidate_recipes[winner["name"]]
family = recipe[0]

if family == "empirical":
    variant = recipe[1]
    te_test, _ = target_statistics(
        joined, y_joined, query_split=test, half_life=6.0
    )
    raw_test = empirical_scores(te_test, variant)

elif family == "lightgbm":
    fixed_rounds = max(1, int(recipe[1]))
    te_joined, _ = target_statistics(
        joined, y_joined, query_split=None, half_life=6.0
    )
    te_test, _ = target_statistics(
        joined, y_joined, query_split=test, half_life=6.0
    )
    joined_stats = numeric_stats(joined)
    x_joined = model_matrix(joined, te_joined, joined_stats)
    x_test = model_matrix(test, te_test, joined_stats)
    joined_weights = recency_weights(
        joined.date, half_life=6.0
    ).astype(np.float32)

    final_gbm = fit_lgb(
        x_joined, y_joined, joined_weights,
        valid_x=None, valid_y=None, rounds=fixed_rounds
    )
    raw_test = final_gbm.predict(
        x_test, num_iteration=fixed_rounds
    ).astype(np.float64)

elif family == "svd":
    final_svd = fit_svd_model(
        joined, y_joined, rank=int(recipe[1])
    )
    raw_test = predict_svd(final_svd, test)

else:
    raise RuntimeError("Unknown selected family: " + family)

test_scores = (
    winner["alpha"] * within_user_rank(test.user_id, raw_test)
    + (1.0 - winner["alpha"])
    * within_user_rank(test.user_id, inc_test)
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(winner["scores"], dtype=np.float64)
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(winner["raw"], dtype=np.float64)
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

metrics = winner["metrics"]
elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))