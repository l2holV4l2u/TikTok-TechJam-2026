import os
import time
import json
import math
import random
import gc

import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 94217

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

train_dates = np.asarray(train.date, dtype=np.int32)
max_train_day = int(np.max(train_dates) % 100)
train_days = train_dates % 100
train_age = (max_train_day - train_days).astype(np.float32)

# Emphasize observations near the train/evaluation boundary without using
# validation statistics. The thirteen training dates are all in April.
HALF_LIFE = 5.0
recency_weight = np.exp(
    -math.log(2.0) * train_age / HALF_LIFE
).astype(np.float32)
recency_weight /= float(recency_weight.mean())

global_weighted_rate = float(
    np.sum(recency_weight * y_train) / np.sum(recency_weight)
)
EPS = 1e-5


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float32), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


TE_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "music_type",
]

TE_STRENGTH = {
    "video_id": 22.0,
    "author_id": 35.0,
    "tab": 80.0,
    "tag": 55.0,
    "duration_bucket": 100.0,
    "upload_type": 90.0,
    "onehot_feat3": 45.0,
    "onehot_feat8": 55.0,
    "music_type": 100.0,
}

TE_SCORE_WEIGHT = {
    "video_id": 2.0,
    "author_id": 1.4,
    "tab": 1.15,
    "tag": 0.9,
    "duration_bucket": 0.55,
    "upload_type": 0.45,
    "onehot_feat3": 0.6,
    "onehot_feat8": 0.4,
    "music_type": 0.2,
}


def build_recency_target_encodings():
    train_columns = []
    valid_columns = []
    test_columns = []

    for name in TE_FIELDS:
        tr_id = np.asarray(train.X[name], dtype=np.int64)
        va_id = np.asarray(valid.X[name], dtype=np.int64)
        te_id = np.asarray(test.X[name], dtype=np.int64)
        cardinality = int(FEATURE_CARDINALITIES[name])

        weighted_count = np.bincount(
            tr_id,
            weights=recency_weight,
            minlength=cardinality,
        ).astype(np.float32)
        weighted_positive = np.bincount(
            tr_id,
            weights=recency_weight * y_train,
            minlength=cardinality,
        ).astype(np.float32)

        strength = float(TE_STRENGTH[name])

        # Leave-one-out values for any model fitted on train.
        loo_count = weighted_count[tr_id] - recency_weight
        loo_positive = (
            weighted_positive[tr_id] - recency_weight * y_train
        )
        tr_rate = (
            loo_positive + strength * global_weighted_rate
        ) / np.maximum(loo_count + strength, 1e-6)

        def map_target(ids):
            known = (ids >= 0) & (ids < cardinality)
            result = np.full(
                len(ids), global_weighted_rate, dtype=np.float32
            )
            safe_ids = ids[known]
            result[known] = (
                weighted_positive[safe_ids]
                + strength * global_weighted_rate
            ) / (weighted_count[safe_ids] + strength)
            return result

        train_columns.append(tr_rate.astype(np.float32))
        valid_columns.append(map_target(va_id))
        test_columns.append(map_target(te_id))

    return (
        np.column_stack(train_columns).astype(np.float32),
        np.column_stack(valid_columns).astype(np.float32),
        np.column_stack(test_columns).astype(np.float32),
    )


te_train, te_valid, te_test = build_recency_target_encodings()

te_score_weights = np.asarray(
    [TE_SCORE_WEIGHT[name] for name in TE_FIELDS],
    dtype=np.float32,
)
global_logit = float(safe_logit(global_weighted_rate))

additive_train_scores = (
    (safe_logit(te_train) - global_logit) @ te_score_weights
).astype(np.float32)
additive_valid_scores = (
    (safe_logit(te_valid) - global_logit) @ te_score_weights
).astype(np.float32)
additive_test_scores = (
    (safe_logit(te_test) - global_logit) @ te_score_weights
).astype(np.float32)


def weighted_entity_stats(name):
    ids = np.asarray(train.X[name], dtype=np.int64)
    cardinality = int(FEATURE_CARDINALITIES[name])
    count = np.bincount(
        ids, weights=recency_weight, minlength=cardinality
    ).astype(np.float32)
    positive = np.bincount(
        ids,
        weights=recency_weight * y_train,
        minlength=cardinality,
    ).astype(np.float32)
    return count, positive


author_count, author_positive = weighted_entity_stats("author_id")
video_count, video_positive = weighted_entity_stats("video_id")
tab_count, tab_positive = weighted_entity_stats("tab")


def hierarchical_scores(split):
    author = np.asarray(split.X["author_id"], dtype=np.int64)
    video = np.asarray(split.X["video_id"], dtype=np.int64)
    tab = np.asarray(split.X["tab"], dtype=np.int64)

    author_prior_strength = 45.0
    video_prior_strength = 25.0
    tab_strength = 100.0

    author_rate = (
        author_positive[author]
        + author_prior_strength * global_weighted_rate
    ) / (author_count[author] + author_prior_strength)

    # A video's posterior shrinks toward its row's author rather than
    # directly toward the global mean.
    video_rate = (
        video_positive[video]
        + video_prior_strength * author_rate
    ) / (video_count[video] + video_prior_strength)

    tab_rate = (
        tab_positive[tab] + tab_strength * global_weighted_rate
    ) / (tab_count[tab] + tab_strength)

    return (
        safe_logit(video_rate)
        + 0.55 * safe_logit(author_rate)
        + 0.65 * (safe_logit(tab_rate) - global_logit)
    ).astype(np.float32)


hier_valid_scores = hierarchical_scores(valid)
hier_test_scores = hierarchical_scores(test)

# Train-only historical entity features. Train arrays are leave-one-out;
# validation and test arrays use the full train split, as guaranteed by API.
hist_train_video = historical_features("train", key="video_id")
hist_valid_video = historical_features("valid", key="video_id")
hist_test_video = historical_features("test", key="video_id")

hist_train_author = historical_features("train", key="author_id")
hist_valid_author = historical_features("valid", key="author_id")
hist_test_author = historical_features("test", key="author_id")


def aligned_history_matrix(train_dict, valid_dict, test_dict):
    keys = sorted(
        set(train_dict.keys())
        & set(valid_dict.keys())
        & set(test_dict.keys())
    )
    tr = np.column_stack([
        np.asarray(train_dict[k], dtype=np.float32) for k in keys
    ])
    va = np.column_stack([
        np.asarray(valid_dict[k], dtype=np.float32) for k in keys
    ])
    te = np.column_stack([
        np.asarray(test_dict[k], dtype=np.float32) for k in keys
    ])
    return keys, tr, va, te


video_hist_names, vh_train, vh_valid, vh_test = aligned_history_matrix(
    hist_train_video, hist_valid_video, hist_test_video
)
author_hist_names, ah_train, ah_valid, ah_test = aligned_history_matrix(
    hist_train_author, hist_valid_author, hist_test_author
)

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def numeric_matrices():
    tr_cols = []
    va_cols = []
    te_cols = []

    for name in NUM_FIELDS:
        tr = np.asarray(train.num[name], dtype=np.float32)
        va = np.asarray(valid.num[name], dtype=np.float32)
        tt = np.asarray(test.num[name], dtype=np.float32)

        finite = np.isfinite(tr)
        median = float(np.median(tr[finite])) if np.any(finite) else 0.0

        tr = np.nan_to_num(tr, nan=median, posinf=median, neginf=median)
        va = np.nan_to_num(va, nan=median, posinf=median, neginf=median)
        tt = np.nan_to_num(tt, nan=median, posinf=median, neginf=median)

        tr_cols.append(np.log1p(np.maximum(tr, 0.0)))
        va_cols.append(np.log1p(np.maximum(va, 0.0)))
        te_cols.append(np.log1p(np.maximum(tt, 0.0)))

    return (
        np.column_stack(tr_cols).astype(np.float32),
        np.column_stack(va_cols).astype(np.float32),
        np.column_stack(te_cols).astype(np.float32),
    )


num_train, num_valid, num_test = numeric_matrices()


def sanitize(x):
    return np.nan_to_num(
        np.asarray(x, dtype=np.float32),
        nan=0.0,
        posinf=20.0,
        neginf=-20.0,
    )


vh_train = sanitize(vh_train)
vh_valid = sanitize(vh_valid)
vh_test = sanitize(vh_test)
ah_train = sanitize(ah_train)
ah_valid = sanitize(ah_valid)
ah_test = sanitize(ah_test)

# A structurally simple linear stacker tests whether stable historical rates
# are sufficient without tree interactions.
linear_train = np.column_stack([
    vh_train,
    ah_train,
    te_train,
    num_train,
]).astype(np.float32)
linear_valid = np.column_stack([
    vh_valid,
    ah_valid,
    te_valid,
    num_valid,
]).astype(np.float32)
linear_test = np.column_stack([
    vh_test,
    ah_test,
    te_test,
    num_test,
]).astype(np.float32)

linear_mean = linear_train.mean(axis=0, dtype=np.float64).astype(np.float32)
linear_std = linear_train.std(axis=0, dtype=np.float64).astype(np.float32)
linear_std = np.maximum(linear_std, 1e-4)

linear_train = np.clip(
    (linear_train - linear_mean) / linear_std, -10.0, 10.0
).astype(np.float32)
linear_valid = np.clip(
    (linear_valid - linear_mean) / linear_std, -10.0, 10.0
).astype(np.float32)
linear_test = np.clip(
    (linear_test - linear_mean) / linear_std, -10.0, 10.0
).astype(np.float32)


class LinearHistoryStacker(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


linear_model = LinearHistoryStacker(linear_train.shape[1])
optimizer = torch.optim.AdamW(
    linear_model.parameters(), lr=0.018, weight_decay=2e-4
)
generator = torch.Generator()
generator.manual_seed(SEED)

linear_train_tensor = torch.from_numpy(linear_train)
label_tensor = torch.from_numpy(y_train)
weight_tensor = torch.from_numpy(recency_weight)

for epoch in range(6):
    permutation = torch.randperm(len(y_train), generator=generator)
    linear_model.train()

    for begin in range(0, len(y_train), 32768):
        idx = permutation[begin:begin + 32768]
        logits = linear_model(linear_train_tensor[idx])
        losses = nn.functional.binary_cross_entropy_with_logits(
            logits, label_tensor[idx], reduction="none"
        )
        weights = weight_tensor[idx]
        loss = (losses * weights).sum() / weights.sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def predict_linear(x, batch_size=65536):
    linear_model.eval()
    result = np.empty(len(x), dtype=np.float32)
    for begin in range(0, len(x), batch_size):
        end = min(begin + batch_size, len(x))
        result[begin:end] = (
            linear_model(torch.from_numpy(x[begin:end]))
            .cpu()
            .numpy()
        )
    return result


linear_valid_scores = predict_linear(linear_valid)
linear_test_scores = predict_linear(linear_test)

del linear_train_tensor, label_tensor, weight_tensor
gc.collect()

# LightGBM receives a deliberately stationarity-oriented categorical set:
# item/content/context fields but no user identity. Historical rates supply
# train-only outcome information and trees can learn nonlinear reliability
# corrections based on entity support.
CAT_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "user_active_degree",
    "onehot_feat3",
    "onehot_feat8",
    "hour",
    "register_days_bucket",
    "music_type",
    "video_type",
]


def categorical_matrix(split):
    return np.column_stack([
        np.asarray(split.X[name], dtype=np.float32)
        for name in CAT_FIELDS
    ]).astype(np.float32)


cat_train = categorical_matrix(train)
cat_valid = categorical_matrix(valid)
cat_test = categorical_matrix(test)

gbdt_train = np.column_stack([
    cat_train,
    vh_train,
    ah_train,
    te_train,
    num_train,
]).astype(np.float32)
gbdt_valid = np.column_stack([
    cat_valid,
    vh_valid,
    ah_valid,
    te_valid,
    num_valid,
]).astype(np.float32)
gbdt_test = np.column_stack([
    cat_test,
    vh_test,
    ah_test,
    te_test,
    num_test,
]).astype(np.float32)

feature_names = (
    CAT_FIELDS
    + ["video_hist_" + k for k in video_hist_names]
    + ["author_hist_" + k for k in author_hist_names]
    + ["te_" + k for k in TE_FIELDS]
    + ["num_" + k for k in NUM_FIELDS]
)

lgb_train = lgb.Dataset(
    gbdt_train,
    label=y_train,
    weight=recency_weight,
    feature_name=feature_names,
    categorical_feature=list(range(len(CAT_FIELDS))),
    free_raw_data=False,
)

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.15,
    "lambda_l2": 2.0,
    "max_bin": 63,
    "cat_smooth": 30.0,
    "cat_l2": 8.0,
    "verbosity": -1,
    "verbose": -1,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "data_random_seed": SEED + 3,
    "num_threads": min(8, os.cpu_count() or 1),
    "force_col_wise": True,
}

gbdt_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=240,
)

gbdt_valid_scores = gbdt_model.predict(
    gbdt_valid, num_iteration=gbdt_model.current_iteration()
).astype(np.float32)
gbdt_test_scores = gbdt_model.predict(
    gbdt_test, num_iteration=gbdt_model.current_iteration()
).astype(np.float32)

importance = gbdt_model.feature_importance(importance_type="gain")
top_indices = np.argsort(-importance)[:10]
top_features = [
    [feature_names[int(i)], float(importance[int(i)])]
    for i in top_indices
]
print("FINDINGS top_gbdt_gain=" + json.dumps(top_features))

shared = os.environ.get("SHARED_ARTIFACTS", "")
incumbent_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
incumbent_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    # Row index gives deterministic tie handling.
    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        users,
    ))
    sorted_users = users[order]
    _, counts = np.unique(sorted_users, return_counts=True)
    starts = np.repeat(
        np.cumsum(counts, dtype=np.int64) - counts,
        counts,
    )
    ordinal = np.arange(n, dtype=np.float64) - starts
    denom = np.maximum(np.repeat(counts, counts) - 1, 1)
    ranked_sorted = ordinal / denom

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


inc_rank_valid = within_user_rank(valid.user_id, incumbent_valid)
inc_rank_test = within_user_rank(test.user_id, incumbent_test)

families = {
    "recency_additive_te": (
        additive_valid_scores,
        additive_test_scores,
    ),
    "hierarchical_empirical_bayes": (
        hier_valid_scores,
        hier_test_scores,
    ),
    "linear_history_stacker": (
        linear_valid_scores,
        linear_test_scores,
    ),
    "historical_lightgbm": (
        gbdt_valid_scores,
        gbdt_test_scores,
    ),
}

candidate_scores = {}
candidate_payload = {}

for family_name, (va_scores, te_scores) in families.items():
    va_scores = np.asarray(va_scores, dtype=np.float64)
    te_scores = np.asarray(te_scores, dtype=np.float64)

    standalone_metric = evaluate(
        valid.user_id, y_valid, va_scores
    )
    candidate_scores[family_name] = float(
        standalone_metric["primary"]
    )
    candidate_payload[family_name] = {
        "valid": va_scores,
        "test": te_scores,
        "raw_valid": va_scores,
        "is_blend": False,
    }

    own_rank_valid = within_user_rank(valid.user_id, va_scores)
    own_rank_test = within_user_rank(test.user_id, te_scores)

    best_blend_metric = None
    best_blend_alpha = None
    best_blend_valid = None
    best_blend_test = None

    # Alpha is the contribution of the newly fitted family.
    for alpha in (0.25, 0.50, 0.75):
        blend_valid = (
            alpha * own_rank_valid
            + (1.0 - alpha) * inc_rank_valid
        )
        metric = evaluate(valid.user_id, y_valid, blend_valid)

        if (
            best_blend_metric is None
            or metric["primary"] > best_blend_metric["primary"]
        ):
            best_blend_metric = metric
            best_blend_alpha = alpha
            best_blend_valid = blend_valid
            best_blend_test = (
                alpha * own_rank_test
                + (1.0 - alpha) * inc_rank_test
            )

    blend_name = (
        family_name + "_incumbent_blend_a"
        + str(best_blend_alpha).replace(".", "")
    )
    candidate_scores[blend_name] = float(
        best_blend_metric["primary"]
    )
    candidate_payload[blend_name] = {
        "valid": best_blend_valid,
        "test": best_blend_test,
        "raw_valid": va_scores,
        "is_blend": True,
    }

winner_name = max(candidate_scores, key=candidate_scores.get)
winner = candidate_payload[winner_name]
valid_scores = np.asarray(winner["valid"], dtype=np.float64)
test_scores = np.asarray(winner["test"], dtype=np.float64)
metrics = evaluate(valid.user_id, y_valid, valid_scores)

print("FINDINGS selected=" + json.dumps({
    "winner": winner_name,
    "half_life": HALF_LIFE,
    "weighted_train_rate": global_weighted_rate,
    "history_features": int(vh_train.shape[1] + ah_train.shape[1]),
}))

print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_scores.items()},
        sort_keys=True,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        valid_scores,
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        test_scores,
    )
    if winner["is_blend"]:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(winner["raw_valid"], dtype=np.float64),
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