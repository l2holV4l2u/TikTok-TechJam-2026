import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
CARDS = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
OFFSETS = np.cumsum([0] + CARDS[:-1], dtype=np.int64)
TOTAL_CARD = int(sum(CARDS))

BATCH_SIZE = 8192
EPOCHS = 6
HALF_LIFE = 4.0


def categorical_matrix(split, offset=True, dtype=np.int64):
    cols = []
    for field, off in zip(FIELDS, OFFSETS):
        x = np.asarray(split.X[field], dtype=dtype)
        if offset:
            x = x + off
        cols.append(x)
    return np.ascontiguousarray(np.column_stack(cols), dtype=dtype)


def date_weights(dates):
    dates = np.asarray(dates, dtype=np.int64)
    ages = 20220421 - dates
    weights = np.exp2(-ages.astype(np.float32) / HALF_LIFE)
    return weights / np.mean(weights)


class RecencyFM(nn.Module):
    def __init__(self, n_features, dim=16):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1)
        self.latent = nn.Embedding(n_features, dim)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.latent(x)
        interactions = 0.5 * (
            v.sum(dim=1).square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interactions


class RecencyDeepFM(nn.Module):
    def __init__(self, n_features, n_fields, dim=12):
        super().__init__()
        self.linear = nn.Embedding(n_features, 1)
        self.embedding = nn.Embedding(n_features, dim)
        self.bias = nn.Parameter(torch.zeros(1))
        self.deep = nn.Sequential(
            nn.Linear(n_fields * dim, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)
        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        fm = 0.5 * (
            v.sum(dim=1).square() - v.square().sum(dim=1)
        ).sum(dim=1)
        deep = self.deep(v.flatten(1)).squeeze(-1)
        return self.bias + linear + fm + deep


def train_torch_model(model, x, y, weights, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    n = len(y)

    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        model.train()
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE].numpy()
            xb = torch.from_numpy(x[idx])
            yb = torch.from_numpy(y[idx])
            wb = torch.from_numpy(weights[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = (losses * wb).mean()
            loss.backward()
            optimizer.step()
    return model


def torch_predict(model, x, batch_size=32768):
    result = np.empty(len(x), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            result[start:end] = model(
                torch.from_numpy(x[start:end])
            ).cpu().numpy()
    return result


def fit_recency_rates(train, fields, weights, smoothing=30.0):
    y = np.asarray(train.y, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    global_rate = float(np.sum(w * y) / np.sum(w))
    tables = {}

    for field in fields:
        ids = np.asarray(train.X[field], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[field])
        counts = np.bincount(ids, weights=w, minlength=card)
        positives = np.bincount(ids, weights=w * y, minlength=card)
        rates = (
            positives + smoothing * global_rate
        ) / (counts + smoothing)
        tables[field] = rates.astype(np.float32)

    return tables, global_rate


def empirical_bayes_predict(split, tables, global_rate):
    field_weights = {
        "video_id": 1.00,
        "author_id": 0.55,
        "tab": 0.20,
        "duration_bucket": 0.25,
    }
    result = np.zeros(len(split.user_id), dtype=np.float64)
    total_weight = 0.0

    for field, weight in field_weights.items():
        ids = np.asarray(split.X[field], dtype=np.int64)
        rates = tables[field][ids]
        rates = np.clip(rates, 1e-5, 1.0 - 1e-5)
        result += weight * np.log(rates / (1.0 - rates))
        total_weight += weight

    base_logit = np.log(global_rate / (1.0 - global_rate))
    return (result / total_weight - base_logit).astype(np.float32)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    boundaries = np.r_[
        0,
        np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1,
        n,
    ]
    counts = np.diff(boundaries)
    starts = np.repeat(boundaries[:-1], counts)
    denominators = np.repeat(np.maximum(counts - 1, 1), counts)

    sorted_ranks = (
        np.arange(n, dtype=np.float64) - starts
    ) / denominators
    sorted_ranks[counts.repeat(counts) == 1] = 0.5

    result = np.empty(n, dtype=np.float64)
    result[order] = sorted_ranks
    return result


train = load("train")
y_train = np.asarray(train.y, dtype=np.float32)
weights_train = date_weights(train.date)
x_train_torch = categorical_matrix(train, offset=True, dtype=np.int64)
x_train_lgb = categorical_matrix(train, offset=False, dtype=np.int32)

# Family 1: recency-weighted factorization machine.
fm_model = train_torch_model(
    RecencyFM(TOTAL_CARD, dim=16),
    x_train_torch,
    y_train,
    weights_train,
    lr=0.001,
)

# Family 2: recency-weighted DeepFM with a nonlinear interaction tower.
deepfm_model = train_torch_model(
    RecencyDeepFM(TOTAL_CARD, len(FIELDS), dim=12),
    x_train_torch,
    y_train,
    weights_train,
    lr=0.001,
)

# Family 3: recency-weighted categorical gradient-boosted trees.
lgb_train = lgb.Dataset(
    x_train_lgb,
    label=y_train,
    weight=weights_train,
    categorical_feature=list(range(len(FIELDS))),
    free_raw_data=False,
)
lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.06,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 250,
    "feature_fraction": 0.90,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 2.0,
    "max_cat_threshold": 64,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "max_bin": 127,
    "num_threads": min(8, max(1, os.cpu_count() or 1)),
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "verbose": -1,
}
gbdt_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=220,
)

# Family 4: non-parametric recency-weighted empirical Bayes.
rate_fields = ["video_id", "author_id", "tab", "duration_bucket"]
rate_tables, global_rate = fit_recency_rates(
    train, rate_fields, weights_train, smoothing=30.0
)

del x_train_torch, x_train_lgb, lgb_train, train

valid = load("valid")
test = load("test")

x_valid_torch = categorical_matrix(valid, offset=True, dtype=np.int64)
x_test_torch = categorical_matrix(test, offset=True, dtype=np.int64)
x_valid_lgb = categorical_matrix(valid, offset=False, dtype=np.int32)
x_test_lgb = categorical_matrix(test, offset=False, dtype=np.int32)

valid_predictions = {
    "recency_fm": torch_predict(fm_model, x_valid_torch),
    "recency_deepfm": torch_predict(deepfm_model, x_valid_torch),
    "recency_gbdt": gbdt_model.predict(
        x_valid_lgb, num_iteration=gbdt_model.best_iteration
    ).astype(np.float32),
    "recency_empirical_bayes": empirical_bayes_predict(
        valid, rate_tables, global_rate
    ),
}
test_predictions = {
    "recency_fm": torch_predict(fm_model, x_test_torch),
    "recency_deepfm": torch_predict(deepfm_model, x_test_torch),
    "recency_gbdt": gbdt_model.predict(
        x_test_lgb, num_iteration=gbdt_model.best_iteration
    ).astype(np.float32),
    "recency_empirical_bayes": empirical_bayes_predict(
        test, rate_tables, global_rate
    ),
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
have_incumbent = (
    os.path.isfile(inc_valid_path) and os.path.isfile(inc_test_path)
)

candidate_scores = {}
best_name = None
best_metrics = None
best_valid = None
best_test = None
best_raw_valid = None
best_is_blend = False

for name, valid_score in valid_predictions.items():
    m = evaluate(valid.user_id, valid.y, valid_score)
    candidate_scores[name] = float(m["primary"])
    if best_metrics is None or m["primary"] > best_metrics["primary"]:
        best_name = name
        best_metrics = m
        best_valid = np.asarray(valid_score, dtype=np.float64)
        best_test = np.asarray(test_predictions[name], dtype=np.float64)
        best_raw_valid = None
        best_is_blend = False

if have_incumbent:
    incumbent_valid = np.load(inc_valid_path)
    incumbent_test = np.load(inc_test_path)
    incumbent_valid_rank = within_user_rank(valid.user_id, incumbent_valid)
    incumbent_test_rank = within_user_rank(test.user_id, incumbent_test)

    blend_weights = (0.20, 0.35, 0.50, 0.65)
    for family_name, own_valid in valid_predictions.items():
        own_valid_rank = within_user_rank(valid.user_id, own_valid)
        own_test_rank = within_user_rank(
            test.user_id, test_predictions[family_name]
        )

        family_best = None
        for own_weight in blend_weights:
            blended_valid = (
                (1.0 - own_weight) * incumbent_valid_rank
                + own_weight * own_valid_rank
            )
            m = evaluate(valid.user_id, valid.y, blended_valid)
            blend_name = (
                family_name + "_incumbent_blend_"
                + str(int(round(100 * own_weight)))
            )
            candidate_scores[blend_name] = float(m["primary"])

            if family_best is None or m["primary"] > family_best[0]:
                family_best = (float(m["primary"]), own_weight, m, blended_valid)

        _, own_weight, m, blended_valid = family_best
        blended_test = (
            (1.0 - own_weight) * incumbent_test_rank
            + own_weight * own_test_rank
        )
        selected_name = (
            family_name + "_incumbent_blend_"
            + str(int(round(100 * own_weight)))
        )

        if m["primary"] > best_metrics["primary"]:
            best_name = selected_name
            best_metrics = m
            best_valid = np.asarray(blended_valid, dtype=np.float64)
            best_test = np.asarray(blended_test, dtype=np.float64)
            best_raw_valid = np.asarray(own_valid, dtype=np.float64)
            best_is_blend = True

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    if best_is_blend:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print("FINDINGS " + json.dumps({
    "selected": best_name,
    "recency_half_life_days": HALF_LIFE,
    "families_compared": list(valid_predictions.keys()),
}, sort_keys=True))

elapsed = time.time() - START
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result))