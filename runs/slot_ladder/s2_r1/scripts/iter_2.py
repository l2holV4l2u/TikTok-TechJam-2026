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
SEED = 2025
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

if OUT:
    os.makedirs(OUT, exist_ok=True)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    z = np.empty_like(x)
    positive = x >= 0
    z[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    ex = np.exp(x[~positive])
    z[~positive] = ex / (1.0 + ex)
    return z


def categorical_matrix(split):
    return np.column_stack([
        np.asarray(split.X[name], dtype=np.int32) for name in FIELDS
    ])


def temporal_weights(dates, half_life):
    dates = np.asarray(dates)
    unique_dates = np.unique(dates)
    day_index = np.searchsorted(unique_dates, dates)
    age = (len(unique_dates) - 1 - day_index).astype(np.float32)
    if half_life is None:
        return np.ones(len(dates), dtype=np.float32)
    w = np.power(0.5, age / float(half_life)).astype(np.float32)
    return w / np.mean(w)


train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

x_train_lgb = categorical_matrix(train)
x_valid_lgb = categorical_matrix(valid)
x_test_lgb = categorical_matrix(test)

inc_valid_path = os.path.join(SHARED, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(SHARED, "incumbent_test_scores.npy")
inc_valid_raw = np.load(inc_valid_path).astype(np.float64)
inc_test_raw = np.load(inc_test_path).astype(np.float64)
inc_valid = sigmoid(inc_valid_raw)
inc_test = sigmoid(inc_test_raw)

own_valid = {}
own_test = {}
models = {}

# ----------------------------------------------------------------------
# Family 1: empirical-Bayes target statistics.
# ----------------------------------------------------------------------
eb_fields = ["video_id", "author_id", "tab", "duration_bucket"]
eb_mix = {
    "video_id": 0.50,
    "author_id": 0.20,
    "tab": 0.20,
    "duration_bucket": 0.10,
}
eb_smoothing = {
    "video_id": 18.0,
    "author_id": 25.0,
    "tab": 80.0,
    "duration_bucket": 80.0,
}
eb_weight = temporal_weights(train.date, 4.0)
global_rate = float(np.sum(eb_weight * y_train) / np.sum(eb_weight))
global_rate = np.clip(global_rate, 1e-5, 1.0 - 1e-5)
global_logit = np.log(global_rate / (1.0 - global_rate))

eb_tables = {}
for field in eb_fields:
    ids = np.asarray(train.X[field], dtype=np.int64)
    card = int(FEATURE_CARDINALITIES[field])
    count = np.bincount(ids, weights=eb_weight, minlength=card)
    positive = np.bincount(ids, weights=eb_weight * y_train, minlength=card)
    smooth = eb_smoothing[field]
    rate = (positive + smooth * global_rate) / (count + smooth)
    rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
    eb_tables[field] = np.log(rate / (1.0 - rate))


def predict_eb(split):
    score = np.full(len(split.user_id), global_logit, dtype=np.float64)
    adjustment = np.zeros_like(score)
    for field in eb_fields:
        ids = np.asarray(split.X[field], dtype=np.int64)
        adjustment += eb_mix[field] * (eb_tables[field][ids] - global_logit)
    return sigmoid(score + adjustment)


own_valid["empirical_bayes"] = predict_eb(valid)
own_test["empirical_bayes"] = predict_eb(test)

# ----------------------------------------------------------------------
# Family 2: binary LightGBM, including a recency half-life sweep.
# ----------------------------------------------------------------------
binary_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.06,
    "num_leaves": 63,
    "min_data_in_leaf": 150,
    "feature_fraction": 0.90,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "max_cat_threshold": 64,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "max_bin": 255,
    "num_threads": min(16, os.cpu_count() or 1),
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "data_random_seed": SEED,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}

for half_life, name in [
    (None, "lgb_binary_uniform"),
    (8.0, "lgb_binary_hl8"),
    (4.0, "lgb_binary_hl4"),
]:
    weights = temporal_weights(train.date, half_life)
    dtrain = lgb.Dataset(
        x_train_lgb,
        label=y_train,
        weight=weights,
        categorical_feature=list(range(len(FIELDS))),
        feature_name=FIELDS,
        free_raw_data=False,
    )
    model = lgb.train(binary_params, dtrain, num_boost_round=180)
    own_valid[name] = model.predict(x_valid_lgb).astype(np.float64)
    own_test[name] = model.predict(x_test_lgb).astype(np.float64)
    models[name] = model
    del dtrain

# ----------------------------------------------------------------------
# Family 3: LambdaRank trained on each user's logged impression groups.
# ----------------------------------------------------------------------
rank_order = np.argsort(
    np.asarray(train.user_id, dtype=np.int64), kind="stable"
)
rank_users = np.asarray(train.user_id, dtype=np.int64)[rank_order]
_, rank_groups = np.unique(rank_users, return_counts=True)

rank_weights = temporal_weights(train.date, 4.0)[rank_order]
rank_params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "lambdarank_truncation_level": 10,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 120,
    "feature_fraction": 0.90,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "max_cat_threshold": 64,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "num_threads": min(16, os.cpu_count() or 1),
    "seed": SEED + 7,
    "feature_fraction_seed": SEED + 7,
    "bagging_seed": SEED + 7,
    "data_random_seed": SEED + 7,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}
drank = lgb.Dataset(
    x_train_lgb[rank_order],
    label=y_train[rank_order],
    weight=rank_weights,
    group=rank_groups,
    categorical_feature=list(range(len(FIELDS))),
    feature_name=FIELDS,
    free_raw_data=False,
)
rank_model = lgb.train(rank_params, drank, num_boost_round=180)
own_valid["lgb_lambdarank"] = sigmoid(
    rank_model.predict(x_valid_lgb).astype(np.float64)
)
own_test["lgb_lambdarank"] = sigmoid(
    rank_model.predict(x_test_lgb).astype(np.float64)
)
models["lgb_lambdarank"] = rank_model
del drank, rank_order, rank_users, rank_groups, rank_weights

# ----------------------------------------------------------------------
# Family 4: DeepFM, with a nonlinear tower beside the FM interaction.
# ----------------------------------------------------------------------
offsets = []
running = 0
for field in FIELDS:
    offsets.append(running)
    running += int(FEATURE_CARDINALITIES[field])
offsets = np.asarray(offsets, dtype=np.int64)
total_cardinality = running


def encode_deep(split):
    x = np.empty((len(split.user_id), len(FIELDS)), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:, j] = np.asarray(split.X[field], dtype=np.int64) + offsets[j]
    return x


class DeepFM(nn.Module):
    def __init__(self, cardinality, n_fields, rank=16):
        super().__init__()
        self.linear = nn.Embedding(cardinality, 1)
        self.embedding = nn.Embedding(cardinality, rank)
        self.bias = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(n_fields * rank, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.015)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        emb = self.embedding(x)
        summed = emb.sum(dim=1)
        fm = 0.5 * (
            summed.square() - emb.square().sum(dim=1)
        ).sum(dim=1)
        deep = self.mlp(emb.reshape(emb.shape[0], -1)).squeeze(-1)
        return self.bias + linear + fm + deep


x_train_deep = torch.from_numpy(encode_deep(train))
x_valid_deep = encode_deep(valid)
x_test_deep = encode_deep(test)
y_train_t = torch.from_numpy(y_train)
deep_weights_t = torch.from_numpy(temporal_weights(train.date, 4.0))

deep_model = DeepFM(total_cardinality, len(FIELDS), rank=16)
deep_optimizer = torch.optim.AdamW(
    deep_model.parameters(), lr=8e-4, weight_decay=1e-6
)

generator = torch.Generator()
generator.manual_seed(SEED + 11)
batch_size = 4096
n_train = len(y_train)

deep_model.train()
for epoch in range(3):
    order = torch.randperm(n_train, generator=generator)
    for start in range(0, n_train, batch_size):
        idx = order[start:start + batch_size]
        xb = x_train_deep[idx]
        yb = y_train_t[idx]
        wb = deep_weights_t[idx]

        deep_optimizer.zero_grad(set_to_none=True)
        logits = deep_model(xb)
        per_row = nn.functional.binary_cross_entropy_with_logits(
            logits, yb, reduction="none"
        )
        loss = (per_row * wb).sum() / wb.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(deep_model.parameters(), 5.0)
        deep_optimizer.step()


def predict_deep(encoded):
    result = np.empty(encoded.shape[0], dtype=np.float64)
    deep_model.eval()
    with torch.no_grad():
        for start in range(0, encoded.shape[0], 32768):
            end = min(start + 32768, encoded.shape[0])
            xb = torch.from_numpy(encoded[start:end])
            result[start:end] = sigmoid(
                deep_model(xb).cpu().numpy().astype(np.float64)
            )
    return result


own_valid["deepfm_hl4"] = predict_deep(x_valid_deep)
own_test["deepfm_hl4"] = predict_deep(x_test_deep)
models["deepfm_hl4"] = deep_model

# ----------------------------------------------------------------------
# Evaluate each standalone and each incumbent blend.
# ----------------------------------------------------------------------
candidate_scores = {}
candidate_arrays = {}
candidate_tests = {}
candidate_raw_own = {}
blend_weights = [0.25, 0.50, 0.75]

inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)
candidate_scores["trusted_incumbent"] = float(inc_metrics["primary"])
candidate_arrays["trusted_incumbent"] = inc_valid
candidate_tests["trusted_incumbent"] = inc_test
candidate_raw_own["trusted_incumbent"] = None

for name in own_valid:
    raw_v = own_valid[name]
    raw_t = own_test[name]

    m = evaluate(valid.user_id, y_valid, raw_v)
    candidate_scores[name] = float(m["primary"])
    candidate_arrays[name] = raw_v
    candidate_tests[name] = raw_t
    candidate_raw_own[name] = raw_v

    for alpha in blend_weights:
        blend_name = "%s_blend_%.2f" % (name, alpha)
        blend_v = alpha * raw_v + (1.0 - alpha) * inc_valid
        blend_t = alpha * raw_t + (1.0 - alpha) * inc_test
        bm = evaluate(valid.user_id, y_valid, blend_v)
        candidate_scores[blend_name] = float(bm["primary"])
        candidate_arrays[blend_name] = blend_v
        candidate_tests[blend_name] = blend_t
        candidate_raw_own[blend_name] = raw_v

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = candidate_arrays[winner]
test_scores = candidate_tests[winner]
metrics = evaluate(valid.user_id, y_valid, valid_scores)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if "_blend_" in winner:
        np.save(
            os.path.join(OUT, "scores_valid_raw.npy"),
            np.asarray(candidate_raw_own[winner], dtype=np.float64),
        )

binary_values = {
    name: candidate_scores[name]
    for name in ["lgb_binary_uniform", "lgb_binary_hl8", "lgb_binary_hl4"]
}
best_binary = max(binary_values, key=binary_values.get)
print("FINDINGS " + json.dumps({
    "best_binary_weighting": best_binary,
    "binary_primaries": binary_values,
    "winner": winner,
}, sort_keys=True))
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))