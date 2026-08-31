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

RATE_FIELDS = [
    "video_id", "author_id", "tag", "duration_bucket", "music_type",
    "upload_type", "tab", "onehot_feat3", "onehot_feat7", "onehot_feat8",
    "onehot_feat1", "video_type",
]

SET_FIELDS = [
    "video_id", "author_id", "tag", "duration_bucket", "music_type",
    "upload_type", "tab", "onehot_feat0", "onehot_feat1", "onehot_feat2",
    "onehot_feat3", "onehot_feat4", "onehot_feat6", "onehot_feat7",
    "onehot_feat8", "onehot_feat12",
]

NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]

SET_DIM = 24
BATCH_SIZE = 8192


def recency_weights(dates, half_life=3.0):
    dates = np.asarray(dates, dtype=np.int32)
    age = dates.max() - dates
    w = np.exp2(-age.astype(np.float32) / np.float32(half_life))
    return (w / np.mean(w)).astype(np.float32)


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, users))
    ordered_users = users[order]

    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = ordered_users[1:] != ordered_users[:-1]
    starts = np.flatnonzero(starts_flag)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.float64) - repeated_starts

    ranked_order = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ranked_order[multi] = (
        positions[multi] / (repeated_lengths[multi] - 1.0)
    )

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_order
    return ranked


class RateEncoder:
    def __init__(self, fields, smoothing=(8.0, 30.0)):
        self.fields = list(fields)
        self.smoothing = tuple(float(x) for x in smoothing)
        self.tables = {}
        self.prior = 0.0

    def fit(self, split, labels, weights):
        labels = np.asarray(labels, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        self.prior = float(np.sum(weights * labels) / np.sum(weights))

        for field in self.fields:
            ids = np.asarray(split.X[field], dtype=np.int64)
            cardinality = FEATURE_CARDINALITIES[field]
            count = np.bincount(
                ids, weights=weights, minlength=cardinality
            ).astype(np.float64)
            positive = np.bincount(
                ids, weights=weights * labels, minlength=cardinality
            ).astype(np.float64)
            self.tables[field] = (count, positive)
        return self

    def transform(self, split, labels=None, weights=None):
        output = []
        is_training = labels is not None
        if is_training:
            labels = np.asarray(labels, dtype=np.float64)
            weights = np.asarray(weights, dtype=np.float64)

        for field in self.fields:
            ids = np.asarray(split.X[field], dtype=np.int64)
            count_table, positive_table = self.tables[field]
            count = count_table[ids]
            positive = positive_table[ids]

            if is_training:
                count = count - weights
                positive = positive - weights * labels

            for smoothing in self.smoothing:
                rate = (
                    positive + smoothing * self.prior
                ) / np.maximum(count + smoothing, 1e-8)
                output.append(rate.astype(np.float32))

            output.append(np.log1p(np.maximum(count, 0.0)).astype(np.float32))

        return np.column_stack(output).astype(np.float32, copy=False)


def fixed_set_embeddings(split):
    n = len(split.user_id)
    result = np.zeros((n, SET_DIM), dtype=np.float32)

    for namespace, field in enumerate(SET_FIELDS):
        values = np.asarray(split.X[field], dtype=np.uint64)
        mixed = (
            values * np.uint64(0x9E3779B185EBCA87)
            + np.uint64((namespace + 1) * 0xC2B2AE3D)
        )
        bucket = (mixed % np.uint64(SET_DIM)).astype(np.int64)
        sign = np.where(
            ((mixed >> np.uint64(21)) & np.uint64(1)) == 0,
            np.float32(1.0),
            np.float32(-1.0),
        )
        result[np.arange(n), bucket] += sign

    result /= np.float32(np.sqrt(len(SET_FIELDS)))
    return result


def exposure_set_features(split):
    """
    The centroid is a model input computed from the candidate exposure set,
    not a fitted target encoding or normalizer. It uses no outcomes.
    """
    row_embedding = fixed_set_embeddings(split)
    users = np.asarray(split.user_id, dtype=np.int64)
    _, inverse = np.unique(users, return_inverse=True)
    n_groups = int(inverse.max()) + 1

    sums = np.zeros((n_groups, SET_DIM), dtype=np.float32)
    np.add.at(sums, inverse, row_embedding)
    counts = np.bincount(inverse, minlength=n_groups).astype(np.float32)

    # Shrink sparse evaluation profiles toward zero so the much longer
    # training histories do not dominate merely because of group size.
    centroid = sums[inverse] / np.maximum(counts[inverse, None], 1.0)
    shrinkage = counts[inverse] / (counts[inverse] + np.float32(4.0))
    centroid *= shrinkage[:, None]

    interaction = row_embedding * centroid
    difference = np.abs(row_embedding - centroid)
    group_size = np.log1p(counts[inverse])[:, None].astype(np.float32)

    return np.concatenate(
        [row_embedding, centroid, interaction, difference, group_size],
        axis=1,
    ).astype(np.float32, copy=False)


class NumericTransformer:
    def __init__(self):
        self.medians = {}

    def fit(self, split):
        for name in NUM_FIELDS:
            x = np.asarray(split.num[name], dtype=np.float32)
            finite = np.isfinite(x)
            self.medians[name] = (
                float(np.median(x[finite])) if np.any(finite) else 0.0
            )
        return self

    def transform(self, split):
        cols = []
        for name in NUM_FIELDS:
            raw = np.asarray(split.num[name], dtype=np.float32)
            missing = ~np.isfinite(raw)
            clean = np.where(missing, self.medians[name], raw)
            clean = np.sign(clean) * np.log1p(np.abs(clean))
            cols.append(clean.astype(np.float32))
            cols.append(missing.astype(np.float32))

        hour = np.asarray(split.X["hour"], dtype=np.float32)
        cols.append(np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32))
        cols.append(np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32))
        return np.column_stack(cols).astype(np.float32, copy=False)


def history_matrix(split_name):
    cols = []
    names = []
    for entity in ("video_id", "author_id"):
        histories = historical_features(split_name, key=entity)
        for key in sorted(histories):
            value = np.asarray(histories[key], dtype=np.float32)
            value = np.nan_to_num(value, nan=0.0, posinf=20.0, neginf=-20.0)
            value = np.sign(value) * np.log1p(np.abs(value))
            cols.append(value.astype(np.float32))
            names.append(entity + ":" + key)

    if not cols:
        return np.empty((len(load(split_name).user_id), 0), dtype=np.float32), names
    return np.column_stack(cols).astype(np.float32, copy=False), names


def build_features(split, split_name, encoder, numeric_transformer,
                   labels=None, weights=None):
    rate = encoder.transform(split, labels=labels, weights=weights)
    setwise = exposure_set_features(split)
    numeric = numeric_transformer.transform(split)
    history, history_names = history_matrix(split_name)

    matrix = np.concatenate(
        [rate, setwise, numeric, history], axis=1
    ).astype(np.float32, copy=False)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=20.0, neginf=-20.0)
    return matrix, history_names


class DeepSetMLP(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def fit_mlp(x, labels, weights):
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(1e-3))

    model = DeepSetMLP(x.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.8e-3, weight_decay=2e-5
    )
    rng = np.random.RandomState(SEED + 19)
    labels = np.asarray(labels, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    model.train()
    for epoch in range(3):
        order = rng.permutation(len(labels))
        total = 0.0
        seen = 0

        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb = torch.from_numpy(
                ((x[idx] - mean) / std).astype(np.float32, copy=False)
            )
            yb = torch.from_numpy(labels[idx])
            wb = torch.from_numpy(weights[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = torch.sum(losses * wb) / torch.sum(wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()

            total += float(loss.detach()) * len(idx)
            seen += len(idx)

        print(
            "FINDINGS family=exposure_deepsets epoch={} loss={:.6f}".format(
                epoch + 1, total / max(seen, 1)
            ),
            flush=True,
        )

    return model, mean, std


@torch.no_grad()
def predict_mlp(model, x, mean, std):
    model.eval()
    result = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), 32768):
        end = min(start + 32768, len(x))
        xb = torch.from_numpy(
            ((x[start:end] - mean) / std).astype(np.float32, copy=False)
        )
        result[start:end] = model(xb).numpy().astype(np.float64)
    return result


class DiagonalGaussianClassifier:
    def fit(self, x, labels, weights):
        labels = np.asarray(labels, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        self.mean = []
        self.var = []
        self.log_prior = []

        for cls in (0, 1):
            w = weights * (labels == cls)
            total = max(float(w.sum()), 1e-8)
            mean = np.sum(x * w[:, None], axis=0, dtype=np.float64) / total
            centered = x.astype(np.float64) - mean
            var = np.sum(
                centered * centered * w[:, None], axis=0, dtype=np.float64
            ) / total
            var = np.maximum(var, 2e-3)
            self.mean.append(mean)
            self.var.append(var)
            self.log_prior.append(np.log(total / weights.sum()))

        return self

    def predict(self, x):
        x64 = x.astype(np.float64, copy=False)
        scores = []
        for cls in (0, 1):
            z = x64 - self.mean[cls]
            score = (
                self.log_prior[cls]
                - 0.5 * np.sum(
                    np.log(self.var[cls]) + z * z / self.var[cls], axis=1
                )
            )
            scores.append(score)
        return scores[1] - scores[0]


train = load("train")
valid = load("valid")
train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
weights = recency_weights(train.date, half_life=3.0)

encoder = RateEncoder(RATE_FIELDS, smoothing=(8.0, 30.0)).fit(
    train, train_y, weights
)
numeric_transformer = NumericTransformer().fit(train)

x_train, history_names = build_features(
    train, "train", encoder, numeric_transformer,
    labels=train_y, weights=weights,
)
x_valid, valid_history_names = build_features(
    valid, "valid", encoder, numeric_transformer
)

if history_names != valid_history_names:
    raise ValueError("Historical feature schemas differ between train and valid")

print(
    "FINDINGS dense_features={} history_features={}".format(
        x_train.shape[1], len(history_names)
    ),
    flush=True,
)

# Family 1: nonlinear boosted decision trees over target/history/set features.
lgb_train = lgb.Dataset(
    x_train,
    label=train_y,
    weight=weights,
    free_raw_data=False,
)
lgb_model = lgb.train(
    {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.045,
        "num_leaves": 47,
        "max_depth": 9,
        "min_data_in_leaf": 450,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 0.15,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "num_threads": min(8, os.cpu_count() or 8),
        "seed": SEED,
        "verbose": -1,
    },
    lgb_train,
    num_boost_round=220,
)
lgb_valid = lgb_model.predict(x_valid).astype(np.float64)

# Family 2: nonlinear permutation-invariant exposure-set representation.
mlp_model, mlp_mean, mlp_std = fit_mlp(x_train, train_y, weights)
mlp_valid = predict_mlp(mlp_model, x_valid, mlp_mean, mlp_std)

# Family 3: generative class-density comparison with diagonal covariance.
gaussian_model = DiagonalGaussianClassifier().fit(
    x_train, train_y, weights
)
gaussian_valid = gaussian_model.predict(x_valid)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are unavailable")

inc_valid = np.load(inc_valid_path).astype(np.float64)
if len(inc_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation length mismatch")

valid_users = np.asarray(valid.user_id, dtype=np.int64)
component_valid = {
    "incumbent": within_user_rank(valid_users, inc_valid),
    "setwise_lgbm": within_user_rank(valid_users, lgb_valid),
    "exposure_deepsets": within_user_rank(valid_users, mlp_valid),
    "diagonal_gaussian": within_user_rank(valid_users, gaussian_valid),
}

recipes = {}
candidate_values = {}

def add_candidate(name, coefficients):
    score = np.zeros(len(valid_users), dtype=np.float64)
    for component, coefficient in coefficients.items():
        score += coefficient * component_valid[component]
    recipes[name] = coefficients
    candidate_values[name] = score


add_candidate("trusted_incumbent", {"incumbent": 1.0})
for family in ("setwise_lgbm", "exposure_deepsets", "diagonal_gaussian"):
    add_candidate(family, {family: 1.0})
    for alpha in (0.10, 0.20, 0.30, 0.40, 0.55, 0.70):
        add_candidate(
            "{}_inc_blend_{:.2f}".format(family, alpha),
            {"incumbent": 1.0 - alpha, family: alpha},
        )

# A heterogeneous ensemble can exploit tree, neural, and density errors.
new_average = (
    component_valid["setwise_lgbm"]
    + component_valid["exposure_deepsets"]
    + component_valid["diagonal_gaussian"]
) / 3.0
component_valid["new_family_average"] = new_average
for alpha in (0.10, 0.20, 0.30, 0.40, 0.55):
    add_candidate(
        "three_family_inc_blend_{:.2f}".format(alpha),
        {"incumbent": 1.0 - alpha, "new_family_average": alpha},
    )

candidate_metrics = {}
for name, score in candidate_values.items():
    candidate_metrics[name] = evaluate(
        valid_users, valid_y, score
    )["primary"]

winner = max(candidate_metrics, key=candidate_metrics.get)
valid_scores = candidate_values[winner]
winner_metrics = evaluate(valid_users, valid_y, valid_scores)

print(
    "CANDIDATES " + json.dumps(
        {k: float(v) for k, v in candidate_metrics.items()},
        sort_keys=True,
    ),
    flush=True,
)
print(
    "FINDINGS winner={} coefficients={}".format(
        winner, json.dumps(recipes[winner], sort_keys=True)
    ),
    flush=True,
)

# Test is loaded only after all model and blend selection is complete.
test = load("test")
x_test, test_history_names = build_features(
    test, "test", encoder, numeric_transformer
)
if history_names != test_history_names:
    raise ValueError("Historical feature schemas differ between train and test")

lgb_test = lgb_model.predict(x_test).astype(np.float64)
mlp_test = predict_mlp(mlp_model, x_test, mlp_mean, mlp_std)
gaussian_test = gaussian_model.predict(x_test)

inc_test = np.load(inc_test_path).astype(np.float64)
if len(inc_test) != len(test.user_id):
    raise ValueError("Incumbent test length mismatch")

test_users = np.asarray(test.user_id, dtype=np.int64)
component_test = {
    "incumbent": within_user_rank(test_users, inc_test),
    "setwise_lgbm": within_user_rank(test_users, lgb_test),
    "exposure_deepsets": within_user_rank(test_users, mlp_test),
    "diagonal_gaussian": within_user_rank(test_users, gaussian_test),
}
component_test["new_family_average"] = (
    component_test["setwise_lgbm"]
    + component_test["exposure_deepsets"]
    + component_test["diagonal_gaussian"]
) / 3.0

test_scores = np.zeros(len(test_users), dtype=np.float64)
for component, coefficient in recipes[winner].items():
    test_scores += coefficient * component_test[component]

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

    if recipes[winner] != {winner: 1.0}:
        new_components = {
            key: value for key, value in recipes[winner].items()
            if key != "incumbent"
        }
        if new_components:
            total_new = sum(new_components.values())
            raw = np.zeros(len(valid_users), dtype=np.float64)
            for component, coefficient in new_components.items():
                raw += (coefficient / total_new) * component_valid[component]
        else:
            raw = component_valid["setwise_lgbm"]
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.6f}' % (
        winner_metrics["primary"],
        winner_metrics["gauc"],
        winner_metrics["ndcg@5"],
        elapsed,
    )
)