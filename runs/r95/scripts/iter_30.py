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
SEED = 73129
THREADS = max(1, min(8, os.cpu_count() or 1))

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "user_active_degree",
    "hour",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "register_days_bucket",
    "music_type",
    "video_type",
    "is_video_author",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.float32)
train_dates = np.asarray(train.date, dtype=np.int64)


def transformed_numeric(a):
    a = np.asarray(a, dtype=np.float32)
    finite = np.isfinite(a)
    out = np.full(len(a), -1.0, dtype=np.float32)
    out[finite] = np.log1p(np.maximum(a[finite], 0.0))
    return out


hist_tr_video = historical_features("train", key="video_id")
hist_va_video = historical_features("valid", key="video_id")
hist_te_video = historical_features("test", key="video_id")
hist_tr_author = historical_features("train", key="author_id")
hist_va_author = historical_features("valid", key="author_id")
hist_te_author = historical_features("test", key="author_id")

video_hist_keys = sorted(
    set(hist_tr_video).intersection(hist_va_video).intersection(hist_te_video)
)
author_hist_keys = sorted(
    set(hist_tr_author).intersection(hist_va_author).intersection(hist_te_author)
)

print(
    "FINDINGS distributional_history_keys=" +
    json.dumps({
        "video": video_hist_keys,
        "author": author_hist_keys,
    }, sort_keys=True),
    flush=True,
)


def make_features(split, vh, ah):
    columns = []
    names = []

    for name in CAT_FIELDS:
        columns.append(np.asarray(split.X[name], dtype=np.float32))
        names.append(name)

    for name in NUM_FIELDS:
        columns.append(transformed_numeric(split.num[name]))
        names.append("num_" + name)

    for name in video_hist_keys:
        a = np.asarray(vh[name], dtype=np.float32)
        a = np.nan_to_num(a, nan=-1.0, posinf=1e6, neginf=-1.0)
        columns.append(a)
        names.append("video_hist_" + name)

    for name in author_hist_keys:
        a = np.asarray(ah[name], dtype=np.float32)
        a = np.nan_to_num(a, nan=-1.0, posinf=1e6, neginf=-1.0)
        columns.append(a)
        names.append("author_hist_" + name)

    X = np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)
    return X, names


Xtr, feature_names = make_features(train, hist_tr_video, hist_tr_author)
Xva, _ = make_features(valid, hist_va_video, hist_va_author)
Xte, _ = make_features(test, hist_te_video, hist_te_author)

del hist_tr_video, hist_va_video, hist_te_video
del hist_tr_author, hist_va_author, hist_te_author
gc.collect()

categorical_indices = list(range(len(CAT_FIELDS)))

print(
    "FINDINGS feature_shape=%s positive_rate=%.6f dates=%s" %
    (
        str(tuple(Xtr.shape)),
        float(ytr.mean()),
        json.dumps(sorted(np.unique(train_dates).tolist())),
    ),
    flush=True,
)

base_params = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.82,
    "lambda_l1": 0.3,
    "lambda_l2": 3.0,
    "max_bin": 127,
    "max_cat_threshold": 64,
    "cat_smooth": 20.0,
    "cat_l2": 15.0,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "data_random_seed": SEED + 3,
    "num_threads": THREADS,
    "verbose": -1,
    "force_col_wise": True,
}


def train_booster(X, y, weights, params, rounds):
    ds = lgb.Dataset(
        X,
        label=y,
        weight=weights,
        feature_name=feature_names,
        categorical_feature=categorical_indices,
        free_raw_data=True,
    )
    model = lgb.train(params, ds, num_boost_round=rounds)
    return model


# A recency-weighted conditional mean is the central member of the
# distributional family. The half-life is fixed a priori rather than selected
# on validation.
last_date = int(train_dates.max())
age = (last_date - train_dates).astype(np.float32)
recency_weight = np.exp2(-age / 4.0).astype(np.float32)
recency_weight /= recency_weight.mean()

global_model = train_booster(
    Xtr, ytr, recency_weight, base_params, rounds=180
)
global_va = global_model.predict(Xva).astype(np.float32)
global_te = global_model.predict(Xte).astype(np.float32)

print("TRAIN global_recency complete", flush=True)

# LightGBM's random-forest mode averages independently bagged trees rather
# than constructing an additive boosting trajectory. It is therefore a
# structurally different estimator of the conditional response distribution.
rf_params = dict(base_params)
rf_params.update({
    "boosting_type": "rf",
    "learning_rate": 0.08,
    "num_leaves": 127,
    "min_data_in_leaf": 350,
    "bagging_fraction": 0.68,
    "bagging_freq": 1,
    "feature_fraction": 0.72,
    "lambda_l1": 0.0,
    "lambda_l2": 5.0,
    "seed": SEED + 100,
})
rf_model = train_booster(
    Xtr, ytr, recency_weight, rf_params, rounds=180
)
rf_va = rf_model.predict(Xva).astype(np.float32)
rf_te = rf_model.predict(Xte).astype(np.float32)

print("TRAIN bagged_random_forest complete", flush=True)

# Experts see disjoint temporal regimes. Quantiles across their probabilities
# estimate tails of the prediction distribution induced by temporal drift.
unique_dates = np.sort(np.unique(train_dates))
date_blocks = np.array_split(unique_dates, 4)

expert_va = []
expert_te = []
expert_sizes = []

expert_params = dict(base_params)
expert_params.update({
    "learning_rate": 0.07,
    "num_leaves": 47,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.88,
    "bagging_fraction": 0.88,
    "bagging_freq": 1,
})

for block_index, block_dates in enumerate(date_blocks):
    mask = np.isin(train_dates, block_dates)
    expert_sizes.append(int(mask.sum()))

    p = dict(expert_params)
    p["seed"] = SEED + 200 + block_index
    p["feature_fraction_seed"] = SEED + 300 + block_index
    p["bagging_seed"] = SEED + 400 + block_index

    model = train_booster(
        Xtr[mask],
        ytr[mask],
        np.ones(int(mask.sum()), dtype=np.float32),
        p,
        rounds=125,
    )
    expert_va.append(model.predict(Xva).astype(np.float32))
    expert_te.append(model.predict(Xte).astype(np.float32))

    print(
        "TRAIN temporal_expert=%d dates=%s rows=%d" %
        (block_index, json.dumps(block_dates.tolist()), int(mask.sum())),
        flush=True,
    )
    del model
    gc.collect()

expert_va = np.stack(expert_va, axis=0)
expert_te = np.stack(expert_te, axis=0)

print(
    "FINDINGS temporal_expert_sizes=" + json.dumps(expert_sizes),
    flush=True,
)

eps = np.float32(1e-5)


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float32), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    z = np.clip(np.asarray(z, dtype=np.float32), -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))


expert_logits_va = logit(expert_va)
expert_logits_te = logit(expert_te)

mean_logit_va = expert_logits_va.mean(axis=0)
mean_logit_te = expert_logits_te.mean(axis=0)
std_logit_va = expert_logits_va.std(axis=0)
std_logit_te = expert_logits_te.std(axis=0)

own_valid = {
    "boosted_conditional_mean": global_va,
    "bagged_random_forest": rf_va,
    "temporal_expert_mean": sigmoid(mean_logit_va),
    "temporal_lower_q25": sigmoid(np.quantile(
        expert_logits_va, 0.25, axis=0
    ).astype(np.float32)),
    "temporal_upper_q75": sigmoid(np.quantile(
        expert_logits_va, 0.75, axis=0
    ).astype(np.float32)),
    "temporal_robust_lcb": sigmoid(mean_logit_va - 0.75 * std_logit_va),
    "temporal_optimistic_ucb": sigmoid(mean_logit_va + 0.75 * std_logit_va),
    "recent_regime_expert": expert_va[-1],
    "mean_rf_mixture": 0.65 * global_va + 0.35 * rf_va,
    "robust_global_mixture": (
        0.65 * global_va +
        0.35 * sigmoid(mean_logit_va - 0.75 * std_logit_va)
    ),
}

own_test = {
    "boosted_conditional_mean": global_te,
    "bagged_random_forest": rf_te,
    "temporal_expert_mean": sigmoid(mean_logit_te),
    "temporal_lower_q25": sigmoid(np.quantile(
        expert_logits_te, 0.25, axis=0
    ).astype(np.float32)),
    "temporal_upper_q75": sigmoid(np.quantile(
        expert_logits_te, 0.75, axis=0
    ).astype(np.float32)),
    "temporal_robust_lcb": sigmoid(mean_logit_te - 0.75 * std_logit_te),
    "temporal_optimistic_ucb": sigmoid(mean_logit_te + 0.75 * std_logit_te),
    "recent_regime_expert": expert_te[-1],
    "mean_rf_mixture": 0.65 * global_te + 0.35 * rf_te,
    "robust_global_mixture": (
        0.65 * global_te +
        0.35 * sigmoid(mean_logit_te - 0.75 * std_logit_te)
    ),
}

valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)
valid_labels = np.asarray(valid.y, dtype=np.int8)


def within_user_rank(users, scores):
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    # Lexicographic sorting creates deterministic ordinal ranks. Predictions
    # are effectively continuous, so tie effects are negligible.
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]

    starts = np.r_[0, np.flatnonzero(
        sorted_users[1:] != sorted_users[:-1]
    ) + 1]
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    positions = np.arange(n, dtype=np.int64) - np.repeat(starts, sizes)
    denom = np.maximum(np.repeat(sizes - 1, sizes), 1)
    ranked_sorted = positions.astype(np.float64) / denom.astype(np.float64)

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_va_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_te_path = os.path.join(shared, "incumbent_test_scores.npy")

has_incumbent = os.path.exists(inc_va_path) and os.path.exists(inc_te_path)
if has_incumbent:
    incumbent_va = np.asarray(np.load(inc_va_path), dtype=np.float64)
    incumbent_te = np.asarray(np.load(inc_te_path), dtype=np.float64)
    incumbent_va_rank = within_user_rank(valid_users, incumbent_va)
    incumbent_te_rank = within_user_rank(test_users, incumbent_te)
else:
    incumbent_va_rank = None
    incumbent_te_rank = None

candidate_scores = {}
best_primary = -np.inf
best_name = None
best_valid_scores = None
best_test_scores = None
best_raw_valid = None

blend_alphas = [0.25, 0.50, 0.75]

for name in own_valid:
    raw_va = np.asarray(own_valid[name], dtype=np.float64)
    raw_te = np.asarray(own_test[name], dtype=np.float64)

    metric = evaluate(valid_users, valid_labels, raw_va)
    candidate_scores[name] = float(metric["primary"])

    if metric["primary"] > best_primary:
        best_primary = float(metric["primary"])
        best_name = name
        best_valid_scores = raw_va
        best_test_scores = raw_te
        best_raw_valid = raw_va

    if has_incumbent:
        own_va_rank = within_user_rank(valid_users, raw_va)
        own_te_rank = within_user_rank(test_users, raw_te)

        for alpha in blend_alphas:
            blend_name = "%s_blend_%.2f" % (name, alpha)
            blend_va = (
                (1.0 - alpha) * incumbent_va_rank +
                alpha * own_va_rank
            )
            metric_blend = evaluate(valid_users, valid_labels, blend_va)
            candidate_scores[blend_name] = float(metric_blend["primary"])

            if metric_blend["primary"] > best_primary:
                best_primary = float(metric_blend["primary"])
                best_name = blend_name
                best_valid_scores = blend_va
                best_test_scores = (
                    (1.0 - alpha) * incumbent_te_rank +
                    alpha * own_te_rank
                )
                best_raw_valid = own_va_rank

if has_incumbent:
    incumbent_metric = evaluate(
        valid_users, valid_labels, incumbent_va_rank
    )
    candidate_scores["trusted_incumbent_rank"] = float(
        incumbent_metric["primary"]
    )
    if incumbent_metric["primary"] > best_primary:
        best_primary = float(incumbent_metric["primary"])
        best_name = "trusted_incumbent_rank"
        best_valid_scores = incumbent_va_rank
        best_test_scores = incumbent_te_rank
        # A raw score is still supplied from the strongest standalone new
        # family, even if the incumbent remains the selected result.
        standalone_name = max(
            own_valid,
            key=lambda n: candidate_scores[n],
        )
        best_raw_valid = np.asarray(
            own_valid[standalone_name], dtype=np.float64
        )

final_metric = evaluate(
    valid_users, valid_labels, np.asarray(best_valid_scores)
)

print(
    "FINDINGS selected_candidate=%s temporal_logit_std_mean=%.6f "
    "temporal_logit_std_p90=%.6f" %
    (
        best_name,
        float(std_logit_va.mean()),
        float(np.quantile(std_logit_va, 0.90)),
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_scores, sort_keys=True),
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test_scores, dtype=np.float64),
    )
    if has_incumbent:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS " + json.dumps({
        "primary": float(final_metric["primary"]),
        "gauc": float(final_metric["gauc"]),
        "ndcg@5": float(final_metric["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    }),
    flush=True,
)