import os
import time
import json
import gc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 314159
THREADS = max(1, min(8, os.cpu_count() or 1))
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(THREADS)

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tag", "tab",
    "duration_bucket", "upload_type", "hour",
    "user_active_degree", "onehot_feat3", "onehot_feat8"
]
RATE_FIELDS = [
    "video_id", "author_id", "tag", "duration_bucket", "upload_type"
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days"
]
NN_CONTENT_FIELDS = [
    "video_id", "author_id", "tag", "duration_bucket",
    "upload_type", "tab", "hour"
]
HALF_LIFE = 4.0
LGB_ROUNDS = 240
NN_EPOCHS = 5
BATCH = 8192
PRED_BATCH = 65536


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int64)
    age = int(dates.max()) - dates
    # YYYYMMDD subtraction is valid within April here.
    w = np.power(0.5, age.astype(np.float32) / HALF_LIFE)
    return (w / np.mean(w)).astype(np.float32)


def weighted_rate_tables(split, y, weights):
    y = np.asarray(y, dtype=np.float32)
    tables = {}
    total_pos = float(np.sum(weights * y))
    total_weight = float(np.sum(weights))
    prior = total_pos / max(total_weight, 1e-9)

    for field in RATE_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[field])
        count = np.bincount(ids, weights=weights, minlength=card).astype(np.float32)
        pos = np.bincount(
            ids, weights=weights * y, minlength=card
        ).astype(np.float32)
        tables[field] = (count, pos)
    return tables, prior


def rate_features_fit(split, y, weights, alpha=20.0):
    """Leave-one-row-out target statistics for fitting rows."""
    y = np.asarray(y, dtype=np.float32)
    tables, prior = weighted_rate_tables(split, y, weights)
    result = np.empty((len(y), len(RATE_FIELDS)), dtype=np.float32)

    for j, field in enumerate(RATE_FIELDS):
        ids = np.asarray(split.X[field], dtype=np.int64)
        count, pos = tables[field]
        loo_count = count[ids] - weights
        loo_pos = pos[ids] - weights * y
        result[:, j] = (
            loo_pos + alpha * prior
        ) / np.maximum(loo_count + alpha, 1e-6)
    return result, tables, prior


def rate_features_apply(split, tables, prior, alpha=20.0):
    n = len(split.user_id)
    result = np.empty((n, len(RATE_FIELDS)), dtype=np.float32)
    for j, field in enumerate(RATE_FIELDS):
        ids = np.asarray(split.X[field], dtype=np.int64)
        count, pos = tables[field]
        result[:, j] = (pos[ids] + alpha * prior) / (count[ids] + alpha)
    return result


def make_lgb_matrix(split, rate_features):
    n = len(split.user_id)
    p = len(CAT_FIELDS) + len(NUM_FIELDS) + rate_features.shape[1]
    x = np.empty((n, p), dtype=np.float32)

    col = 0
    for field in CAT_FIELDS:
        x[:, col] = np.asarray(split.X[field], dtype=np.float32)
        col += 1

    for field in NUM_FIELDS:
        v = np.asarray(split.num[field], dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        x[:, col] = np.log1p(np.maximum(v, 0.0))
        col += 1

    x[:, col:] = rate_features
    return x


def train_lgb(x, y, weights):
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
        "lambda_l1": 0.15,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "num_threads": THREADS,
        "verbose": -1,
    }
    ds = lgb.Dataset(
        x,
        label=np.asarray(y, dtype=np.float32),
        weight=np.asarray(weights, dtype=np.float32),
        categorical_feature=list(range(len(CAT_FIELDS))),
        free_raw_data=False
    )
    return lgb.train(params, ds, num_boost_round=LGB_ROUNDS)


class TwoTower(nn.Module):
    def __init__(self, positive_rate, dim=24):
        super().__init__()
        self.dim = dim
        self.user = nn.Embedding(
            int(FEATURE_CARDINALITIES["user_id"]), dim
        )
        self.content = nn.ModuleList([
            nn.Embedding(int(FEATURE_CARDINALITIES[f]), dim)
            for f in NN_CONTENT_FIELDS
        ])
        self.linear = nn.ModuleList([
            nn.Embedding(int(FEATURE_CARDINALITIES[f]), 1)
            for f in ["user_id"] + NN_CONTENT_FIELDS
        ])
        self.bias = nn.Parameter(torch.tensor(
            math.log(positive_rate / (1.0 - positive_rate)),
            dtype=torch.float32
        ))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user.weight, std=0.04)
        for emb in self.content:
            nn.init.normal_(emb.weight, std=0.04)
        for emb in self.linear:
            nn.init.zeros_(emb.weight)

    def forward(self, user, content):
        u = self.user(user)
        item = 0.0
        for j, emb in enumerate(self.content):
            item = item + emb(content[:, j])
        item = item / math.sqrt(len(self.content))
        interaction = (u * item).sum(dim=1) / math.sqrt(self.dim)

        linear = self.linear[0](user).squeeze(1)
        for j in range(len(self.content)):
            linear = linear + self.linear[j + 1](
                content[:, j]
            ).squeeze(1)
        return self.bias + linear + interaction


def nn_arrays(split):
    user = np.asarray(split.X["user_id"], dtype=np.int64)
    content = np.column_stack([
        np.asarray(split.X[f], dtype=np.int64)
        for f in NN_CONTENT_FIELDS
    ])
    return user, np.ascontiguousarray(content, dtype=np.int64)


def fit_two_tower(split, y, weights):
    torch.manual_seed(SEED)
    user_np, content_np = nn_arrays(split)
    y_np = np.asarray(y, dtype=np.float32)
    w_np = np.asarray(weights, dtype=np.float32)

    model = TwoTower(float(np.average(y_np, weights=w_np)))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.004, weight_decay=2e-6
    )

    user = torch.from_numpy(user_np)
    content = torch.from_numpy(content_np)
    labels = torch.from_numpy(y_np)
    row_weights = torch.from_numpy(w_np)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    n = len(y_np)
    for _ in range(NN_EPOCHS):
        model.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH):
            idx = order[start:min(start + BATCH, n)]
            optimizer.zero_grad(set_to_none=True)
            logits = model(user[idx], content[idx])
            losses = F.binary_cross_entropy_with_logits(
                logits, labels[idx], reduction="none"
            )
            loss = torch.mean(losses * row_weights[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def predict_two_tower(model, split):
    user_np, content_np = nn_arrays(split)
    user = torch.from_numpy(user_np)
    content = torch.from_numpy(content_np)
    out = np.empty(len(user_np), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(user_np), PRED_BATCH):
            end = min(start + PRED_BATCH, len(user_np))
            out[start:end] = model(
                user[start:end], content[start:end]
            ).cpu().numpy()
    return out


def eb_score(rate_features):
    # More specific entities receive larger coefficients, while the broad
    # duration/upload statistics stabilize unseen or sparse content.
    coef = np.asarray([1.35, 1.20, 0.75, 0.40, 0.35], dtype=np.float32)
    return np.asarray(rate_features @ coef, dtype=np.float32)


def standardized(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x))
    if sd < 1e-10:
        return x - np.mean(x)
    return (x - np.mean(x)) / sd


def best_blend(name, own, incumbent, users, labels):
    own_z = standardized(own)
    inc_z = standardized(incumbent)
    best = None
    records = {}
    for alpha in [0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00]:
        score = alpha * own_z + (1.0 - alpha) * inc_z
        met = evaluate(users, labels, score)
        key = name if alpha == 1.0 else name + "_blend_" + str(alpha)
        records[key] = float(met["primary"])
        if best is None or met["primary"] > best[0]:
            best = (
                float(met["primary"]), alpha, score.copy(), met
            )
    return best, records


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
w_train = recency_weights(train.date)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)

# Common leakage-free recency-weighted entity statistics.
rate_train, tables_train, prior_train = rate_features_fit(
    train, y_train, w_train
)
rate_valid = rate_features_apply(
    valid, tables_train, prior_train
)

# Family 1: recency-weighted categorical gradient boosting.
x_train = make_lgb_matrix(train, rate_train)
x_valid = make_lgb_matrix(valid, rate_valid)
lgb_model = train_lgb(x_train, y_train, w_train)
lgb_valid = lgb_model.predict(
    x_valid, num_iteration=LGB_ROUNDS
).astype(np.float32)

# Family 2: recency-weighted latent user/content two-tower.
nn_model = fit_two_tower(train, y_train, w_train)
nn_valid = predict_two_tower(nn_model, valid)

# Family 3: non-parametric empirical Bayes.
eb_valid = eb_score(rate_valid)

family_predictions = {
    "recency_lgb": lgb_valid,
    "latent_two_tower": nn_valid,
    "recency_empirical_bayes": eb_valid,
}

all_records = {}
best_overall = None
for family, prediction in family_predictions.items():
    best, records = best_blend(
        family, prediction, inc_valid, valid.user_id, y_valid
    )
    all_records.update(records)
    if best_overall is None or best[0] > best_overall["primary"]:
        best_overall = {
            "primary": best[0],
            "family": family,
            "alpha": best[1],
            "scores": best[2],
            "metrics": best[3],
            "raw": prediction.copy(),
        }

raw_findings = {
    name: float(evaluate(
        valid.user_id, y_valid, pred
    )["primary"])
    for name, pred in family_predictions.items()
}
print("FINDINGS " + json.dumps({
    "raw_family_primary": raw_findings,
    "selected_family": best_overall["family"],
    "selected_incumbent_blend_weight_on_new_model": best_overall["alpha"],
    "train_recency_weight_min": float(w_train.min()),
    "train_recency_weight_max": float(w_train.max())
}, sort_keys=True))
print("CANDIDATES " + json.dumps(all_records, sort_keys=True))

valid_scores = np.asarray(best_overall["scores"], dtype=np.float64)
metrics = best_overall["metrics"]

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        valid_scores
    )
    if best_overall["alpha"] < 0.999999:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(best_overall["raw"], dtype=np.float64)
        )

# Refit only the selected new family on train+validation, preserving all
# hyperparameters and the validation-selected blend weight.
selected_family = best_overall["family"]
alpha = float(best_overall["alpha"])

combined_y = np.concatenate([
    y_train, y_valid.astype(np.float32, copy=False)
])
combined_dates = np.concatenate([
    np.asarray(train.date), np.asarray(valid.date)
])
w_combined = recency_weights(combined_dates)

class CombinedSplit:
    pass


combined = CombinedSplit()
combined.user_id = np.concatenate([train.user_id, valid.user_id])
combined.date = combined_dates
combined.X = {
    field: np.concatenate([train.X[field], valid.X[field]])
    for field in set(CAT_FIELDS + RATE_FIELDS + NN_CONTENT_FIELDS + ["user_id"])
}
combined.num = {
    field: np.concatenate([train.num[field], valid.num[field]])
    for field in NUM_FIELDS
}

test = load("test")
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

if selected_family == "recency_lgb":
    rate_combined, tables_combined, prior_combined = rate_features_fit(
        combined, combined_y, w_combined
    )
    rate_test = rate_features_apply(
        test, tables_combined, prior_combined
    )
    x_combined = make_lgb_matrix(combined, rate_combined)
    x_test = make_lgb_matrix(test, rate_test)
    final_model = train_lgb(x_combined, combined_y, w_combined)
    own_test = final_model.predict(
        x_test, num_iteration=LGB_ROUNDS
    ).astype(np.float32)

elif selected_family == "latent_two_tower":
    del nn_model
    gc.collect()
    final_model = fit_two_tower(combined, combined_y, w_combined)
    own_test = predict_two_tower(final_model, test)

else:
    _, tables_combined, prior_combined = weighted_rate_tables(
        combined, combined_y, w_combined
    )
    rate_test = rate_features_apply(
        test, tables_combined, prior_combined
    )
    own_test = eb_score(rate_test)

if alpha < 0.999999:
    test_scores = (
        alpha * standardized(own_test)
        + (1.0 - alpha) * standardized(inc_test)
    )
else:
    test_scores = standardized(own_test)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed)
}))