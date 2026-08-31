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
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 271828
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 8))

CAT_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "music_type",
    "upload_type",
    "video_type",
    "hour",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat4",
    "onehot_feat6",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat9",
    "onehot_feat10",
    "onehot_feat11",
    "onehot_feat12",
    "onehot_feat16",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]
BATCH_SIZE = 32768


def date_ordinals(dates):
    dates = np.asarray(dates, dtype=np.int32)
    result = np.empty(dates.shape[0], dtype=np.int64)
    for value in np.unique(dates):
        text = str(int(value))
        ordinal = np.datetime64(
            "{}-{}-{}".format(text[:4], text[4:6], text[6:8]), "D"
        ).astype(np.int64)
        result[dates == value] = int(ordinal)
    return result


def recency_weights(dates, half_life=4.0):
    days = date_ordinals(dates)
    age = days.max() - days
    weights = np.exp2(-age.astype(np.float32) / np.float32(half_life))
    return (weights / weights.mean()).astype(np.float32)


def signed_log1p(values):
    values = np.asarray(values, dtype=np.float32)
    return np.sign(values) * np.log1p(np.abs(values))


def load_histories(split_name):
    result = {}
    for key in ("video_id", "author_id"):
        histories = historical_features(split_name, key=key)
        for name, values in histories.items():
            result[key + "__" + name] = np.asarray(values, dtype=np.float32)
    return result


def discover_history_keys(train_hist):
    names = sorted(train_hist.keys())

    # Keep a bounded, deterministic set. Prefer rates/counts and avoid using
    # every feedback statistic if the API exposes a very large collection.
    preferred = []
    for name in names:
        lower = name.lower()
        if any(token in lower for token in (
            "long", "count", "rate", "mean", "smooth", "positive", "impression"
        )):
            preferred.append(name)

    selected = preferred[:20]
    if len(selected) < min(12, len(names)):
        for name in names:
            if name not in selected:
                selected.append(name)
            if len(selected) >= min(20, len(names)):
                break
    return selected


def raw_numeric_matrix(split, histories, history_keys):
    columns = []

    for name in NUM_FIELDS:
        values = np.asarray(split.num[name], dtype=np.float32)
        columns.append(values)

    for name in history_keys:
        if name in histories:
            columns.append(np.asarray(histories[name], dtype=np.float32))
        else:
            columns.append(np.full(len(split.user_id), np.nan, dtype=np.float32))

    # Training-relative time is useful to GBDT only as a coarse context. Trees
    # naturally stop extrapolating beyond the final observed date.
    columns.append(date_ordinals(split.date).astype(np.float32))

    return np.column_stack(columns).astype(np.float32, copy=False)


class NumericTransformer:
    def fit(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        self.median = np.zeros(raw.shape[1], dtype=np.float32)

        for column in range(raw.shape[1]):
            finite = np.isfinite(raw[:, column])
            self.median[column] = (
                np.median(raw[finite, column]).astype(np.float32)
                if finite.any()
                else np.float32(0.0)
            )

        filled = np.where(np.isfinite(raw), raw, self.median[None, :])
        transformed = signed_log1p(filled)

        self.mean = transformed.mean(axis=0).astype(np.float32)
        self.std = transformed.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-4] = 1.0
        return self

    def transform(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        missing = (~np.isfinite(raw)).astype(np.float32)
        filled = np.where(np.isfinite(raw), raw, self.median[None, :])
        transformed = signed_log1p(filled)
        standardized = (transformed - self.mean[None, :]) / self.std[None, :]
        standardized = np.clip(standardized, -8.0, 8.0)
        return np.column_stack([standardized, missing]).astype(
            np.float32, copy=False
        )


def categorical_matrix(split):
    return np.column_stack([
        np.asarray(split.X[name], dtype=np.int64)
        for name in CAT_FIELDS
    ])


def lgb_matrix(cat, numeric):
    return np.column_stack([
        cat.astype(np.float32, copy=False),
        numeric.astype(np.float32, copy=False),
    ])


def fit_bagged_gbdt(x, y, weights):
    categorical_indices = list(range(len(CAT_FIELDS)))
    models = []

    for member, seed in enumerate((SEED + 101, SEED + 202)):
        dataset = lgb.Dataset(
            x,
            label=y,
            weight=weights,
            categorical_feature=categorical_indices,
            free_raw_data=False,
        )
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.055,
            "num_leaves": 47,
            "max_depth": 9,
            "min_data_in_leaf": 350,
            "min_sum_hessian_in_leaf": 20.0,
            "lambda_l1": 0.15,
            "lambda_l2": 2.0,
            "feature_fraction": 0.82,
            "feature_fraction_seed": seed,
            "bagging_fraction": 0.78,
            "bagging_freq": 1,
            "bagging_seed": seed,
            "max_bin": 127,
            "cat_smooth": 30.0,
            "cat_l2": 15.0,
            "max_cat_threshold": 64,
            "num_threads": min(8, os.cpu_count() or 8),
            "seed": seed,
            "verbose": -1,
        }
        model = lgb.train(params, dataset, num_boost_round=190)
        models.append(model)
        print(
            "FINDINGS family=bagged_gbdt member={} trees={}".format(
                member + 1, model.current_iteration()
            ),
            flush=True,
        )
    return models


def predict_bagged_gbdt(models, x):
    predictions = np.zeros(x.shape[0], dtype=np.float64)
    for model in models:
        predictions += model.predict(x).astype(np.float64)
    predictions /= len(models)
    return predictions


class NeuralAdditiveModel(nn.Module):
    """
    Each categorical and continuous feature contributes independently.
    Excluding cross-feature interactions is deliberate regularization against
    date-specific identity combinations.
    """

    def __init__(self, n_numeric):
        super().__init__()
        self.cat_terms = nn.ModuleList([
            nn.Embedding(FEATURE_CARDINALITIES[name], 1)
            for name in CAT_FIELDS
        ])
        self.numeric_terms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 8),
                nn.Tanh(),
                nn.Linear(8, 5),
                nn.Tanh(),
                nn.Linear(5, 1),
            )
            for _ in range(n_numeric)
        ])
        self.bias = nn.Parameter(torch.zeros(()))

        for term in self.cat_terms:
            nn.init.zeros_(term.weight)
        for term in self.numeric_terms:
            for module in term:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.25)
                    nn.init.zeros_(module.bias)

    def forward(self, categorical, numeric):
        score = self.bias.expand(categorical.shape[0])
        for index, term in enumerate(self.cat_terms):
            score = score + term(categorical[:, index]).squeeze(1)
        for index, term in enumerate(self.numeric_terms):
            score = score + term(numeric[:, index:index + 1]).squeeze(1)
        return score


def fit_nam(cat, numeric, labels, weights):
    model = NeuralAdditiveModel(numeric.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0022, weight_decay=2e-5
    )

    cat_tensor = torch.from_numpy(cat)
    num_tensor = torch.from_numpy(numeric)
    y_tensor = torch.from_numpy(labels.astype(np.float32, copy=False))
    w_tensor = torch.from_numpy(weights)
    rng = np.random.RandomState(SEED + 303)

    for epoch in range(3):
        permutation = rng.permutation(cat.shape[0])
        model.train()
        total_loss = 0.0
        total_weight = 0.0

        for start in range(0, cat.shape[0], BATCH_SIZE):
            indices = torch.from_numpy(
                permutation[start:start + BATCH_SIZE]
            )
            cb = cat_tensor.index_select(0, indices)
            nb = num_tensor.index_select(0, indices)
            yb = y_tensor.index_select(0, indices)
            wb = w_tensor.index_select(0, indices)

            optimizer.zero_grad(set_to_none=True)
            logits = model(cb, nb)
            row_loss = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = torch.sum(row_loss * wb) / torch.sum(wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(torch.sum(row_loss * wb).detach())
            total_weight += float(torch.sum(wb))

        print(
            "FINDINGS family=neural_additive epoch={} weighted_loss={:.6f}".format(
                epoch + 1, total_loss / max(total_weight, 1e-9)
            ),
            flush=True,
        )
    return model


@torch.no_grad()
def predict_nam(model, cat, numeric):
    model.eval()
    cat_tensor = torch.from_numpy(cat)
    num_tensor = torch.from_numpy(numeric)
    result = np.empty(cat.shape[0], dtype=np.float64)

    for start in range(0, cat.shape[0], BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, cat.shape[0])
        result[start:end] = model(
            cat_tensor[start:end], num_tensor[start:end]
        ).numpy().astype(np.float64)
    return result


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    n = len(values)
    if n == 0:
        return values.copy()

    order = np.lexsort((np.arange(n, dtype=np.int64), values, users))
    ordered_users = users[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = ordered_users[1:] != ordered_users[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    ranked_order = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ranked_order[multi] = positions[multi] / (
        repeated_lengths[multi] - 1.0
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_order
    return result


def metric_primary(users, labels, scores):
    return float(evaluate(users, labels, scores)["primary"])


# Load only training and validation labels.
train = load("train")
valid = load("valid")

train_hist = load_histories("train")
valid_hist = load_histories("valid")
history_keys = discover_history_keys(train_hist)
print(
    "FINDINGS history_features={}".format(json.dumps(history_keys)),
    flush=True,
)

train_cat = categorical_matrix(train)
valid_cat = categorical_matrix(valid)

train_raw_num = raw_numeric_matrix(train, train_hist, history_keys)
valid_raw_num = raw_numeric_matrix(valid, valid_hist, history_keys)

numeric_transformer = NumericTransformer().fit(train_raw_num)
train_num = numeric_transformer.transform(train_raw_num)
valid_num = numeric_transformer.transform(valid_raw_num)

train_y = np.asarray(train.y, dtype=np.float32)
weights = recency_weights(train.date, half_life=4.0)

# Family 1: bagged nonlinear GBDT.
train_lgb = lgb_matrix(train_cat, train_num)
valid_lgb = lgb_matrix(valid_cat, valid_num)
gbdt_models = fit_bagged_gbdt(train_lgb, train_y, weights)
gbdt_valid = predict_bagged_gbdt(gbdt_models, valid_lgb)

# Release the largest training design matrix before fitting the torch model.
del train_lgb

# Family 2: interaction-free neural additive model.
nam_model = fit_nam(train_cat, train_num, train_y, weights)
nam_valid = predict_nam(nam_model, valid_cat, valid_num)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are unavailable")

inc_valid = np.load(inc_valid_path).astype(np.float64)
if len(inc_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation prediction length mismatch")

rank_inc = within_user_rank(valid.user_id, inc_valid)
rank_gbdt = within_user_rank(valid.user_id, gbdt_valid)
rank_nam = within_user_rank(valid.user_id, nam_valid)

candidate_scores = {
    "trusted_incumbent": rank_inc,
    "bagged_gbdt": rank_gbdt,
    "neural_additive": rank_nam,
}
candidate_recipes = {
    "trusted_incumbent": ("inc", 0.0),
    "bagged_gbdt": ("gbdt", 1.0),
    "neural_additive": ("nam", 1.0),
}

for family, family_rank in (("gbdt", rank_gbdt), ("nam", rank_nam)):
    for alpha in (0.10, 0.20, 0.30, 0.40, 0.50):
        name = "{}_inc_borda_{:.2f}".format(family, alpha)
        candidate_scores[name] = (
            (1.0 - alpha) * rank_inc + alpha * family_rank
        )
        candidate_recipes[name] = (family, alpha)

# Rank aggregation across all three structurally distinct sources.
for own_weight in (0.10, 0.20, 0.30):
    name = "three_way_borda_{:.2f}".format(own_weight)
    candidate_scores[name] = (
        (1.0 - own_weight) * rank_inc
        + 0.65 * own_weight * rank_gbdt
        + 0.35 * own_weight * rank_nam
    )
    candidate_recipes[name] = ("three", own_weight)

candidate_metrics = {}
best_name = None
best_primary = -np.inf
best_metrics = None

for name, scores in candidate_scores.items():
    metrics = evaluate(valid.user_id, valid.y, scores)
    candidate_metrics[name] = float(metrics["primary"])
    if float(metrics["primary"]) > best_primary:
        best_primary = float(metrics["primary"])
        best_name = name
        best_metrics = metrics

valid_scores = candidate_scores[best_name]
recipe_family, recipe_weight = candidate_recipes[best_name]

print(
    "CANDIDATES " + json.dumps(
        candidate_metrics, sort_keys=True
    ),
    flush=True,
)
print(
    "FINDINGS selected={} recipe_family={} weight={:.2f}".format(
        best_name, recipe_family, recipe_weight
    ),
    flush=True,
)

# Generate test features and predictions without touching test labels.
test = load("test")
test_hist = load_histories("test")
test_cat = categorical_matrix(test)
test_raw_num = raw_numeric_matrix(test, test_hist, history_keys)
test_num = numeric_transformer.transform(test_raw_num)
test_lgb = lgb_matrix(test_cat, test_num)

gbdt_test = predict_bagged_gbdt(gbdt_models, test_lgb)
nam_test = predict_nam(nam_model, test_cat, test_num)
inc_test = np.load(inc_test_path).astype(np.float64)

if len(inc_test) != len(test.user_id):
    raise ValueError("Incumbent test prediction length mismatch")

test_rank_inc = within_user_rank(test.user_id, inc_test)
test_rank_gbdt = within_user_rank(test.user_id, gbdt_test)
test_rank_nam = within_user_rank(test.user_id, nam_test)

if recipe_family == "inc":
    test_scores = test_rank_inc
    raw_valid = None
elif recipe_family == "gbdt":
    if recipe_weight >= 0.999:
        test_scores = test_rank_gbdt
        raw_valid = None
    else:
        test_scores = (
            (1.0 - recipe_weight) * test_rank_inc
            + recipe_weight * test_rank_gbdt
        )
        raw_valid = gbdt_valid
elif recipe_family == "nam":
    if recipe_weight >= 0.999:
        test_scores = test_rank_nam
        raw_valid = None
    else:
        test_scores = (
            (1.0 - recipe_weight) * test_rank_inc
            + recipe_weight * test_rank_nam
        )
        raw_valid = nam_valid
else:
    test_scores = (
        (1.0 - recipe_weight) * test_rank_inc
        + 0.65 * recipe_weight * test_rank_gbdt
        + 0.35 * recipe_weight * test_rank_nam
    )
    # The complete newly fitted component before incumbent aggregation.
    raw_valid = 0.65 * rank_gbdt + 0.35 * rank_nam

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if raw_valid is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {{"primary": {:.10f}, "gauc": {:.10f}, '
    '"ndcg@5": {:.10f}, "gpu_seconds": {:.4f}}}'.format(
        float(best_metrics["primary"]),
        float(best_metrics["gauc"]),
        float(best_metrics["ndcg@5"]),
        elapsed,
    )
)