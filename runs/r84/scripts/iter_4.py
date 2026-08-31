import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour", "user_active_degree",
    "onehot_feat3", "onehot_feat8",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
RATE_FIELDS = ["video_id", "author_id"]
HALF_LIFE = 5.0
WIDE_EPOCHS = 3
WIDE_BATCH = 8192

train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")


def temporal_weights(dates):
    dates = np.asarray(dates, dtype=np.int32)
    age = int(dates.max()) - dates
    return np.exp2(-age.astype(np.float32) / HALF_LIFE).astype(np.float32)


def entity_stats_fit(ids, y, cardinality):
    ids = np.asarray(ids, dtype=np.int64)
    yy = np.asarray(y, dtype=np.float32)
    count = np.bincount(ids, minlength=cardinality).astype(np.float32)
    positive = np.bincount(ids, weights=yy, minlength=cardinality).astype(np.float32)
    global_rate = float(yy.mean())
    alpha = 20.0

    loo_count = np.maximum(count[ids] - 1.0, 0.0)
    loo_positive = positive[ids] - yy
    loo_rate = (loo_positive + alpha * global_rate) / (loo_count + alpha)
    loo_logcount = np.log1p(loo_count)

    state = (count, positive, global_rate, alpha)
    return loo_rate.astype(np.float32), loo_logcount.astype(np.float32), state


def entity_stats_apply(ids, state):
    count, positive, global_rate, alpha = state
    ids = np.asarray(ids, dtype=np.int64)
    safe = np.minimum(ids, len(count) - 1)
    c = count[safe]
    p = positive[safe]
    unseen = (ids >= len(count))
    if np.any(unseen):
        c = c.copy()
        p = p.copy()
        c[unseen] = 0.0
        p[unseen] = 0.0
    rate = (p + alpha * global_rate) / (c + alpha)
    return rate.astype(np.float32), np.log1p(c).astype(np.float32)


def make_tree_fit(split, y):
    columns = []
    states = {}
    for f in FIELDS:
        columns.append(np.asarray(split.X[f], dtype=np.float32))
    for f in NUM_FIELDS:
        a = np.asarray(split.num[f], dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(a, 0.0)).astype(np.float32))
    for f in RATE_FIELDS:
        card = int(FEATURE_CARDINALITIES[f])
        r, c, state = entity_stats_fit(split.X[f], y, card)
        columns.extend([r, c])
        states[f] = state
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32), states


def make_tree_apply(split, states):
    columns = []
    for f in FIELDS:
        columns.append(np.asarray(split.X[f], dtype=np.float32))
    for f in NUM_FIELDS:
        a = np.asarray(split.num[f], dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(a, 0.0)).astype(np.float32))
    for f in RATE_FIELDS:
        r, c = entity_stats_apply(split.X[f], states[f])
        columns.extend([r, c])
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


X_train, train_states = make_tree_fit(train, y_train)
X_valid = make_tree_apply(valid, train_states)
cat_indices = list(range(len(FIELDS)))
weights_train = temporal_weights(train.date)

common_params = {
    "learning_rate": 0.055,
    "num_leaves": 63,
    "min_data_in_leaf": 180,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.15,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "max_cat_to_onehot": 16,
    "cat_smooth": 20.0,
    "num_threads": min(8, os.cpu_count() or 1),
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "verbose": -1,
}

dtrain_binary = lgb.Dataset(
    X_train,
    label=y_train,
    weight=weights_train,
    categorical_feature=cat_indices,
    free_raw_data=False,
)
dvalid_binary = lgb.Dataset(
    X_valid,
    label=y_valid,
    categorical_feature=cat_indices,
    reference=dtrain_binary,
    free_raw_data=False,
)

binary_params = dict(common_params)
binary_params.update({
    "objective": "binary",
    "metric": "binary_logloss",
})
binary_model = lgb.train(
    binary_params,
    dtrain_binary,
    num_boost_round=320,
    valid_sets=[dvalid_binary],
    callbacks=[lgb.early_stopping(35, verbose=False)],
)
binary_rounds = int(binary_model.best_iteration or 320)
pred_binary = binary_model.predict(
    X_valid, num_iteration=binary_rounds
).astype(np.float64)


def sorted_groups(user_ids):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[order]
    _, counts = np.unique(sorted_users, return_counts=True)
    return order, counts.astype(np.int32)


tr_order, tr_groups = sorted_groups(train.user_id)
va_order, va_groups = sorted_groups(valid.user_id)

dtrain_rank = lgb.Dataset(
    X_train[tr_order],
    label=y_train[tr_order],
    weight=weights_train[tr_order],
    group=tr_groups,
    categorical_feature=cat_indices,
    free_raw_data=True,
)
dvalid_rank = lgb.Dataset(
    X_valid[va_order],
    label=y_valid[va_order],
    group=va_groups,
    categorical_feature=cat_indices,
    reference=dtrain_rank,
    free_raw_data=True,
)

rank_params = dict(common_params)
rank_params.update({
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "lambdarank_truncation_level": 10,
    "label_gain": [0, 1],
})
rank_model = lgb.train(
    rank_params,
    dtrain_rank,
    num_boost_round=240,
    valid_sets=[dvalid_rank],
    callbacks=[lgb.early_stopping(30, verbose=False)],
)
rank_rounds = int(rank_model.best_iteration or 240)
pred_rank_sorted = rank_model.predict(
    X_valid[va_order], num_iteration=rank_rounds
).astype(np.float64)
pred_rank = np.empty(len(valid.user_id), dtype=np.float64)
pred_rank[va_order] = pred_rank_sorted


offsets = np.cumsum(
    [0] + [int(FEATURE_CARDINALITIES[f]) for f in FIELDS[:-1]],
    dtype=np.int64,
)
wide_cardinality = int(sum(int(FEATURE_CARDINALITIES[f]) for f in FIELDS))


def make_wide_x(split):
    x = np.column_stack([split.X[f] for f in FIELDS]).astype(
        np.int64, copy=False
    )
    x = x + offsets[None, :]
    return np.ascontiguousarray(x)


class WideAdditive(nn.Module):
    def __init__(self, cardinality):
        super().__init__()
        self.weight = nn.Embedding(cardinality, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.weight.weight)

    def forward(self, x):
        return self.bias + self.weight(x).sum(dim=1).squeeze(1)


def fit_wide(x, y, weights, epochs):
    model = WideAdditive(wide_cardinality)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.012, weight_decay=1e-6)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    n = len(y)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    wt = torch.from_numpy(np.asarray(weights, dtype=np.float32))
    xt = torch.from_numpy(x)

    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, WIDE_BATCH):
            idx = order[start:start + WIDE_BATCH]
            logits = model(xt[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, yt[idx], reduction="none"
            )
            loss = (losses * wt[idx]).sum() / wt[idx].sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict_wide(model, x):
    xt = torch.from_numpy(x)
    out = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), WIDE_BATCH * 2):
        end = min(start + WIDE_BATCH * 2, len(x))
        out[start:end] = model(xt[start:end]).numpy().astype(np.float64)
    return out


wide_model = fit_wide(
    make_wide_x(train), y_train, weights_train, WIDE_EPOCHS
)
pred_wide = predict_wide(wide_model, make_wide_x(valid))


def metric_primary(scores):
    return float(evaluate(valid.user_id, y_valid, scores)["primary"])


def zscore(a):
    a = np.asarray(a, dtype=np.float64)
    sd = float(a.std())
    if sd < 1e-12:
        return np.zeros_like(a)
    return (a - float(a.mean())) / sd


family_predictions = {
    "wide_additive": pred_wide,
    "lgbm_binary_recency": pred_binary,
    "lgbm_lambdarank": pred_rank,
}
candidate_scores = {}
best_name = None
best_scores = None
best_primary = -np.inf
best_family = None
best_weight = None

for family, prediction in family_predictions.items():
    standalone_primary = metric_primary(prediction)
    candidate_scores[family] = standalone_primary
    if standalone_primary > best_primary:
        best_primary = standalone_primary
        best_name = family
        best_scores = prediction.copy()
        best_family = family
        best_weight = 1.0

    zi = zscore(inc_valid)
    zm = zscore(prediction)
    family_best_blend = -np.inf
    family_best_weight = 0.0
    family_best_scores = None

    for weight in np.linspace(0.10, 0.90, 9):
        blended = (1.0 - weight) * zi + weight * zm
        value = metric_primary(blended)
        if value > family_best_blend:
            family_best_blend = value
            family_best_weight = float(weight)
            family_best_scores = blended.copy()

    blend_name = family + "_blend"
    candidate_scores[blend_name] = family_best_blend
    candidate_scores[family + "_blend_weight"] = family_best_weight

    if family_best_blend > best_primary:
        best_primary = family_best_blend
        best_name = blend_name
        best_scores = family_best_scores
        best_family = family
        best_weight = family_best_weight

inc_primary = metric_primary(inc_valid)
candidate_scores["trusted_incumbent"] = inc_primary
if inc_primary >= best_primary:
    best_primary = inc_primary
    best_name = "trusted_incumbent"
    best_scores = inc_valid.copy()
    best_family = "incumbent"
    best_weight = 0.0

metrics = evaluate(valid.user_id, y_valid, best_scores)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_family != "incumbent" and best_weight < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(family_predictions[best_family], dtype=np.float64),
        )

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS selected=%s binary_rounds=%d rank_rounds=%d blend_weight=%.2f"
    % (best_name, binary_rounds, rank_rounds, float(best_weight))
)

# Test scoring: refit the selected recipe on train + validation.
if out:
    if best_family == "incumbent":
        test_scores = np.load(inc_test_path).astype(np.float64)
    else:
        test = load("test")
        y_combined = np.concatenate(
            [y_train, y_valid.astype(np.float32)]
        ).astype(np.float32)
        dates_combined = np.concatenate(
            [np.asarray(train.date), np.asarray(valid.date)]
        )
        weights_combined = temporal_weights(dates_combined)

        class CombinedSplit:
            pass

        combined = CombinedSplit()
        combined.X = {
            f: np.concatenate([train.X[f], valid.X[f]])
            for f in FIELDS
        }
        combined.num = {
            f: np.concatenate([train.num[f], valid.num[f]])
            for f in NUM_FIELDS
        }
        combined.user_id = np.concatenate(
            [np.asarray(train.user_id), np.asarray(valid.user_id)]
        )
        combined.date = dates_combined

        if best_family == "wide_additive":
            x_combined_wide = make_wide_x(combined)
            final_model = fit_wide(
                x_combined_wide, y_combined, weights_combined, WIDE_EPOCHS
            )
            raw_test = predict_wide(final_model, make_wide_x(test))
        else:
            X_combined, combined_states = make_tree_fit(combined, y_combined)
            X_test = make_tree_apply(test, combined_states)

            if best_family == "lgbm_binary_recency":
                dfinal = lgb.Dataset(
                    X_combined,
                    label=y_combined,
                    weight=weights_combined,
                    categorical_feature=cat_indices,
                    free_raw_data=True,
                )
                final_model = lgb.train(
                    binary_params,
                    dfinal,
                    num_boost_round=binary_rounds,
                )
                raw_test = final_model.predict(
                    X_test, num_iteration=binary_rounds
                ).astype(np.float64)
            else:
                co_order, co_groups = sorted_groups(combined.user_id)
                dfinal = lgb.Dataset(
                    X_combined[co_order],
                    label=y_combined[co_order],
                    weight=weights_combined[co_order],
                    group=co_groups,
                    categorical_feature=cat_indices,
                    free_raw_data=True,
                )
                final_model = lgb.train(
                    rank_params,
                    dfinal,
                    num_boost_round=rank_rounds,
                )
                raw_test = final_model.predict(
                    X_test, num_iteration=rank_rounds
                ).astype(np.float64)

        if best_weight < 1.0:
            incumbent_test = np.load(inc_test_path).astype(np.float64)
            test_scores = (
                (1.0 - best_weight) * zscore(incumbent_test)
                + best_weight * zscore(raw_test)
            )
        else:
            test_scores = raw_test

    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.6f}'
    % (
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        float(elapsed),
    )
)