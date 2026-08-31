import os
import time
import json
import gc

import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 73129
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

CAT_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag", "upload_type",
    "onehot_feat3", "onehot_feat8", "duration_bucket", "music_type",
    "user_active_degree", "register_days_bucket", "register_days_range",
    "onehot_feat1", "onehot_feat7", "onehot_feat0",
]
NUM_FIELDS = [
    "duration_ms", "user_fans_user_num", "user_follow_user_num",
    "user_friend_user_num", "user_register_days",
]
HISTORY_SUFFIXES = [
    "train_count_log1p", "long_view_rate", "is_click_rate",
    "play_time_ms_logmean",
]


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates, dtype=np.int64)
    age = dates.max() - dates
    weights = np.power(0.5, age.astype(np.float32) / half_life)
    weights /= max(float(weights.mean()), 1e-8)
    return np.asarray(weights, dtype=np.float32)


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


def make_encoding_tables(train, weights):
    y = np.asarray(train.y, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    prior = float(np.sum(w * y) / np.sum(w))
    tables = {}

    for field in CAT_FIELDS:
        ids = np.asarray(train.X[field], dtype=np.int64)
        cardinality = int(FEATURE_CARDINALITIES[field])
        counts = np.bincount(ids, weights=w, minlength=cardinality)
        positives = np.bincount(ids, weights=w * y, minlength=cardinality)
        tables[field] = (
            np.asarray(counts, dtype=np.float64),
            np.asarray(positives, dtype=np.float64),
        )
    return prior, tables


def encoded_categorical_columns(split, split_name, train, weights,
                                prior, tables, strength=18.0):
    is_train = split_name == "train"
    columns = []
    y = np.asarray(train.y, dtype=np.float64) if is_train else None
    w = np.asarray(weights, dtype=np.float64) if is_train else None

    for field in CAT_FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        counts, positives = tables[field]
        row_counts = counts[ids].copy()
        row_positives = positives[ids].copy()

        if is_train:
            row_counts -= w
            row_positives -= w * y
            row_counts = np.maximum(row_counts, 0.0)
            row_positives = np.maximum(row_positives, 0.0)

        posterior = (
            row_positives + strength * prior
        ) / (row_counts + strength)

        columns.append(
            (safe_logit(posterior) - safe_logit(prior)).astype(np.float32)
        )
        columns.append(np.log1p(row_counts).astype(np.float32))

    return columns


def make_dense_matrix(split, split_name, train, weights, prior, tables):
    columns = encoded_categorical_columns(
        split, split_name, train, weights, prior, tables
    )

    for field in NUM_FIELDS:
        values = np.asarray(split.num[field], dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(values, 0.0)).astype(np.float32))

    for entity in ("video_id", "author_id"):
        histories = historical_features(split_name, key=entity)
        for suffix in HISTORY_SUFFIXES:
            key = entity + "_" + suffix
            values = np.asarray(histories[key], dtype=np.float32)
            values = np.nan_to_num(
                values, nan=0.0, posinf=0.0, neginf=0.0
            )
            if suffix.endswith("_rate"):
                values = np.clip(values, 1e-4, 1.0 - 1e-4)
                values = (
                    np.log(values) - np.log1p(-values)
                ).astype(np.float32)
            columns.append(values.astype(np.float32))
        del histories

    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def standardize_from_train(x_train, x_valid, x_test):
    mean = np.mean(x_train, axis=0, dtype=np.float64)
    std = np.std(x_train, axis=0, dtype=np.float64)
    std[~np.isfinite(std) | (std < 1e-5)] = 1.0

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)
    x_train = np.asarray((x_train - mean) / std, dtype=np.float32)
    x_valid = np.asarray((x_valid - mean) / std, dtype=np.float32)
    x_test = np.asarray((x_test - mean) / std, dtype=np.float32)

    np.clip(x_train, -8.0, 8.0, out=x_train)
    np.clip(x_valid, -8.0, 8.0, out=x_valid)
    np.clip(x_test, -8.0, 8.0, out=x_test)
    return x_train, x_valid, x_test


class ResidualDenseNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.hidden1 = nn.Linear(input_dim, 96)
        self.hidden2 = nn.Linear(96, 48)
        self.hidden3 = nn.Linear(48, 1)

    def forward(self, x):
        linear = self.linear(x).squeeze(1)
        h = F.silu(self.hidden1(x))
        h = F.silu(self.hidden2(h))
        return linear + self.hidden3(h).squeeze(1)


class RandomFourierModel(nn.Module):
    def __init__(self, input_dim, n_frequencies=128):
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(SEED + 91)
        projection = torch.randn(
            input_dim, n_frequencies, generator=generator
        ) / np.sqrt(max(input_dim, 1))
        phase = (
            2.0 * np.pi *
            torch.rand(n_frequencies, generator=generator)
        )
        self.register_buffer("projection", projection)
        self.register_buffer("phase", phase)
        self.linear = nn.Linear(2 * n_frequencies + input_dim, 1)

    def forward(self, x):
        z = x @ self.projection + self.phase
        features = torch.cat(
            [x, torch.sin(z), torch.cos(z)], dim=1
        )
        return self.linear(features).squeeze(1)


def train_torch_model(model, x_train, y_train, weights,
                      epochs, learning_rate, seed_offset):
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=2e-5
    )
    n = len(y_train)
    batch_size = 32768
    rng = np.random.default_rng(SEED + seed_offset)

    y_tensor = torch.from_numpy(
        np.asarray(y_train, dtype=np.float32)
    )
    w_tensor = torch.from_numpy(
        np.asarray(weights, dtype=np.float32)
    )

    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx_np = order[start:start + batch_size]
            idx = torch.from_numpy(idx_np)
            xb = torch.from_numpy(x_train[idx_np])
            yb = y_tensor[idx]
            wb = w_tensor[idx]

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = torch.sum(losses * wb) / torch.sum(wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    return model


def predict_torch(model, matrix):
    model.eval()
    result = np.empty(len(matrix), dtype=np.float32)
    batch_size = 65536
    with torch.no_grad():
        for start in range(0, len(matrix), batch_size):
            end = min(start + batch_size, len(matrix))
            xb = torch.from_numpy(matrix[start:end])
            result[start:end] = (
                model(xb).cpu().numpy().astype(np.float32)
            )
    return result


def train_local_linear_boosting(x_train, y_train, weights,
                                x_valid, x_test):
    dataset = lgb.Dataset(
        x_train,
        label=np.asarray(y_train, dtype=np.float32),
        weight=np.asarray(weights, dtype=np.float32),
        free_raw_data=True,
    )
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "linear_tree": True,
        "learning_rate": 0.055,
        "num_leaves": 24,
        "max_depth": 7,
        "min_data_in_leaf": 450,
        "max_bin": 127,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 0.08,
        "lambda_l2": 8.0,
        "linear_lambda": 6.0,
        "seed": SEED,
        "bagging_seed": SEED + 1,
        "feature_fraction_seed": SEED + 2,
        "num_threads": min(8, max(1, os.cpu_count() or 1)),
        "verbose": -1,
    }
    model = lgb.train(params, dataset, num_boost_round=125)
    valid_probability = model.predict(x_valid).astype(np.float64)
    test_probability = model.predict(x_test).astype(np.float64)
    valid_scores = safe_logit(valid_probability).astype(np.float32)
    test_scores = safe_logit(test_probability).astype(np.float32)
    del dataset, model
    gc.collect()
    return valid_scores, test_scores


def user_center_scale(scores, users):
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users, dtype=np.int64)
    _, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse)
    means = (
        np.bincount(inverse, weights=scores) /
        np.maximum(counts, 1)
    )
    centered = scores - means[inverse]
    scale = float(np.std(centered))
    if not np.isfinite(scale) or scale < 1e-10:
        scale = 1.0
    return centered / scale


train = load("train")
valid = load("valid")
test = load("test")

train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)

weights = recency_weights(train.date, half_life=4.0)
prior, encoding_tables = make_encoding_tables(train, weights)

x_train = make_dense_matrix(
    train, "train", train, weights, prior, encoding_tables
)
x_valid = make_dense_matrix(
    valid, "valid", train, weights, prior, encoding_tables
)
x_test = make_dense_matrix(
    test, "test", train, weights, prior, encoding_tables
)
x_train, x_valid, x_test = standardize_from_train(
    x_train, x_valid, x_test
)

families = {}

# Family 1: piecewise local linear prediction formed by linear models in
# boosted-tree regions.
try:
    ll_valid, ll_test = train_local_linear_boosting(
        x_train, train_y, weights, x_valid, x_test
    )
    families["local_linear_boosting"] = (ll_valid, ll_test)
except Exception as exc:
    print("FINDINGS local_linear_boosting_failed=" + repr(exc))

# Family 2: learned global nonlinear interactions with a linear residual.
dense_model = ResidualDenseNet(x_train.shape[1])
dense_model = train_torch_model(
    dense_model, x_train, train_y, weights,
    epochs=4, learning_rate=1.8e-3, seed_offset=200
)
dense_valid = predict_torch(dense_model, x_valid)
dense_test = predict_torch(dense_model, x_test)
families["residual_dense"] = (dense_valid, dense_test)
del dense_model
gc.collect()

# Family 3: stationary nonlinear similarity through a fixed random Fourier
# basis, with only the final ranking surface learned.
rff_model = RandomFourierModel(
    x_train.shape[1], n_frequencies=128
)
rff_model = train_torch_model(
    rff_model, x_train, train_y, weights,
    epochs=3, learning_rate=2.2e-3, seed_offset=400
)
rff_valid = predict_torch(rff_model, x_valid)
rff_test = predict_torch(rff_model, x_test)
families["random_fourier_kernel"] = (rff_valid, rff_test)
del rff_model, x_train, x_valid, x_test
gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not (
    os.path.exists(inc_valid_path) and
    os.path.exists(inc_test_path)
):
    raise FileNotFoundError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
inc_valid_norm = user_center_scale(inc_valid, valid_users)
inc_test_norm = user_center_scale(inc_test, test_users)

candidate_metrics = {}
candidate_payloads = {}

inc_metric = evaluate(valid_users, valid_y, inc_valid)
candidate_metrics["trusted_incumbent"] = float(
    inc_metric["primary"]
)
candidate_payloads["trusted_incumbent"] = (
    inc_valid, inc_test, None, False
)

for family_name, (own_valid, own_test) in families.items():
    standalone_metric = evaluate(
        valid_users, valid_y, own_valid
    )
    candidate_metrics[family_name] = float(
        standalone_metric["primary"]
    )
    candidate_payloads[family_name] = (
        own_valid, own_test, own_valid, False
    )

    own_valid_norm = user_center_scale(own_valid, valid_users)
    own_test_norm = user_center_scale(own_test, test_users)

    for alpha in (0.15, 0.30, 0.50, 0.70):
        blend_valid = (
            alpha * own_valid_norm +
            (1.0 - alpha) * inc_valid_norm
        )
        blend_test = (
            alpha * own_test_norm +
            (1.0 - alpha) * inc_test_norm
        )
        name = family_name + "_blend_" + str(alpha)
        blend_metric = evaluate(
            valid_users, valid_y, blend_valid
        )
        candidate_metrics[name] = float(
            blend_metric["primary"]
        )
        candidate_payloads[name] = (
            blend_valid, blend_test, own_valid, True
        )

best_name = max(candidate_metrics, key=candidate_metrics.get)
valid_scores, test_scores, raw_valid_scores, is_blend = (
    candidate_payloads[best_name]
)
final_metric = evaluate(valid_users, valid_y, valid_scores)

standalone_values = {
    name: candidate_metrics[name]
    for name in families
}
if standalone_values:
    best_standalone_name = max(
        standalone_values, key=standalone_values.get
    )
    print(
        "FINDINGS best_standalone={} primary={:.6f} selected={}".format(
            best_standalone_name,
            standalone_values[best_standalone_name],
            best_name,
        )
    )

print(
    "CANDIDATES " +
    json.dumps(
        {k: float(v) for k, v in candidate_metrics.items()},
        sort_keys=True
    )
)

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
    if is_blend and raw_valid_scores is not None:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_valid_scores, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS " +
    json.dumps({
        "primary": float(final_metric["primary"]),
        "gauc": float(final_metric["gauc"]),
        "ndcg@5": float(final_metric["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)