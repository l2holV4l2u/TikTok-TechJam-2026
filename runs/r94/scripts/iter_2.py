import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
N_FIELDS = len(FIELDS)
K = 12
BATCH_SIZE = 4096
EPOCHS = 3
LR = 0.001
HALF_LIFE_DAYS = 4.0

OFFSETS = []
running = 0
for field in FIELDS:
    OFFSETS.append(running)
    running += int(FEATURE_CARDINALITIES[field])
OFFSETS = np.asarray(OFFSETS, dtype=np.int64)
TOTAL_CARDINALITY = running


def build_features(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[name], dtype=np.int64) + OFFSETS[j]
            for j, name in enumerate(FIELDS)
        ]),
        dtype=np.int64,
    )


def build_local_features(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[name], dtype=np.int32)
            for name in FIELDS
        ]),
        dtype=np.int32,
    )


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int32)
    latest = int(dates.max())
    weights = np.power(
        2.0,
        (dates.astype(np.float64) - latest) / HALF_LIFE_DAYS,
    )
    weights /= weights.mean()
    return weights.astype(np.float32)


def sparse_linear(embedding, x):
    return embedding(x).squeeze(-1).sum(dim=1)


class DeepFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, K, sparse=True)
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1, sparse=True)
        self.bias = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(N_FIELDS * K, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        e = self.embedding(x)
        summed = e.sum(dim=1)
        fm = 0.5 * (
            summed.square().sum(dim=1)
            - e.square().sum(dim=(1, 2))
        )
        deep = self.mlp(e.flatten(1)).squeeze(1)
        return self.bias + sparse_linear(self.linear, x) + fm + deep

    def sparse_parameters(self):
        return [self.embedding.weight, self.linear.weight]

    def dense_parameters(self):
        return [self.bias] + list(self.mlp.parameters())


class NeuralFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, K, sparse=True)
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1, sparse=True)
        self.bias = nn.Parameter(torch.zeros(1))
        self.interaction_mlp = nn.Sequential(
            nn.Linear(K, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        e = self.embedding(x)
        summed = e.sum(dim=1)
        bi = 0.5 * (summed.square() - e.square().sum(dim=1))
        nonlinear = self.interaction_mlp(bi).squeeze(1)
        return self.bias + sparse_linear(self.linear, x) + nonlinear

    def sparse_parameters(self):
        return [self.embedding.weight, self.linear.weight]

    def dense_parameters(self):
        return [self.bias] + list(self.interaction_mlp.parameters())


class CrossNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        dim = N_FIELDS * K
        self.embedding = nn.Embedding(TOTAL_CARDINALITY, K, sparse=True)
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1, sparse=True)
        self.bias = nn.Parameter(torch.zeros(1))
        self.cross_w = nn.ParameterList([
            nn.Parameter(torch.empty(dim)) for _ in range(3)
        ])
        self.cross_b = nn.ParameterList([
            nn.Parameter(torch.zeros(dim)) for _ in range(3)
        ])
        self.output = nn.Linear(dim, 1)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)
        for w in self.cross_w:
            nn.init.normal_(w, std=0.02)

    def forward(self, x):
        x0 = self.embedding(x).flatten(1)
        xl = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scalar = (xl * w).sum(dim=1, keepdim=True)
            xl = xl + x0 * scalar + b
        crossed = self.output(xl).squeeze(1)
        return self.bias + sparse_linear(self.linear, x) + crossed

    def sparse_parameters(self):
        return [self.embedding.weight, self.linear.weight]

    def dense_parameters(self):
        return (
            [self.bias]
            + list(self.cross_w)
            + list(self.cross_b)
            + list(self.output.parameters())
        )


class FieldAwareFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            TOTAL_CARDINALITY * N_FIELDS, K, sparse=True
        )
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1, sparse=True)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        score = self.bias + sparse_linear(self.linear, x)
        for i in range(N_FIELDS):
            xi = x[:, i]
            for j in range(i + 1, N_FIELDS):
                xj = x[:, j]
                vi_for_j = self.embedding(xi * N_FIELDS + j)
                vj_for_i = self.embedding(xj * N_FIELDS + i)
                score = score + (vi_for_j * vj_for_i).sum(dim=1)
        return score

    def sparse_parameters(self):
        return [self.embedding.weight, self.linear.weight]

    def dense_parameters(self):
        return [self.bias]


def train_torch_model(model, x_train, y_train, w_train, seed):
    sparse_opt = torch.optim.SparseAdam(
        model.sparse_parameters(), lr=LR
    )
    dense_opt = torch.optim.Adam(
        model.dense_parameters(), lr=LR
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    n = x_train.shape[0]

    model.train()
    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            xb = x_train[idx]
            yb = y_train[idx]
            wb = w_train[idx]

            sparse_opt.zero_grad(set_to_none=True)
            dense_opt.zero_grad(set_to_none=True)

            logits = model(xb)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = (losses * wb).sum() / wb.sum()
            loss.backward()
            sparse_opt.step()
            dense_opt.step()
    return model


def predict_torch(model, x_np, batch_size=32768):
    model.eval()
    scores = np.empty(x_np.shape[0], dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, x_np.shape[0], batch_size):
            end = min(start + batch_size, x_np.shape[0])
            xb = torch.from_numpy(x_np[start:end])
            scores[start:end] = model(xb).cpu().numpy().astype(np.float64)
    return scores


def fit_empirical_bayes(train, weights):
    y = np.asarray(train.y, dtype=np.float64)
    global_rate = float(np.sum(weights * y) / np.sum(weights))
    models = {}
    smoothing = {
        "user_id": 40.0,
        "video_id": 25.0,
        "author_id": 35.0,
        "tab": 150.0,
        "duration_bucket": 100.0,
    }

    for field in FIELDS:
        ids = np.asarray(train.X[field], dtype=np.int64)
        cardinality = int(FEATURE_CARDINALITIES[field])
        count = np.bincount(
            ids, weights=weights, minlength=cardinality
        ).astype(np.float64)
        positive = np.bincount(
            ids, weights=weights * y, minlength=cardinality
        ).astype(np.float64)
        prior = smoothing[field]
        rate = (positive + prior * global_rate) / (count + prior)
        rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
        models[field] = np.log(rate / (1.0 - rate))
    return models


def predict_empirical_bayes(split, models):
    # User effects are constant within a user but provide calibration; video and
    # author effects dominate the actual within-user ordering.
    coefficients = {
        "user_id": 0.15,
        "video_id": 1.00,
        "author_id": 0.70,
        "tab": 0.55,
        "duration_bucket": 0.35,
    }
    result = np.zeros(len(split.user_id), dtype=np.float64)
    for field in FIELDS:
        ids = np.asarray(split.X[field], dtype=np.int64)
        result += coefficients[field] * models[field][ids]
    return result


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    row = np.arange(n, dtype=np.int64)

    # Ascending ordinal percentile within each user. The row key makes the
    # transformation deterministic for tied empirical-Bayes scores.
    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    start_positions = np.flatnonzero(starts)
    group_ids = np.cumsum(starts) - 1
    position = np.arange(n, dtype=np.int64) - start_positions[group_ids]
    sizes = np.diff(np.append(start_positions, n))
    denom = np.maximum(sizes[group_ids] - 1, 1)

    ranked_sorted = position.astype(np.float64) / denom
    ranked_sorted[sizes[group_ids] == 1] = 0.5
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


train = load("train")
valid = load("valid")

x_train_np = build_features(train)
x_valid_np = build_features(valid)
x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(np.asarray(train.y, dtype=np.float32))
sample_weights_np = recency_weights(train.date)
sample_weights = torch.from_numpy(sample_weights_np)

family_models = {}
valid_raw = {}

torch_families = [
    ("deepfm_recent", DeepFM),
    ("nfm_recent", NeuralFM),
    ("dcn_recent", CrossNetwork),
    ("ffm_recent", FieldAwareFM),
]

for family_index, (name, constructor) in enumerate(torch_families):
    torch.manual_seed(SEED + family_index)
    model = constructor()
    model = train_torch_model(
        model,
        x_train,
        y_train,
        sample_weights,
        SEED + 100 * family_index,
    )
    valid_raw[name] = predict_torch(model, x_valid_np)
    family_models[name] = model

# A structurally different boosted-tree model on the identical five fields.
x_train_lgb = build_local_features(train)
x_valid_lgb = build_local_features(valid)
lgb_train = lgb.Dataset(
    x_train_lgb,
    label=np.asarray(train.y, dtype=np.float32),
    weight=sample_weights_np,
    categorical_feature=list(range(N_FIELDS)),
    free_raw_data=False,
)
lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.07,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 2.0,
    "max_bin": 255,
    "num_threads": max(1, min(16, os.cpu_count() or 1)),
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "verbose": -1,
}
lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=180,
)
valid_raw["lightgbm_recent"] = lgb_model.predict(
    x_valid_lgb, num_iteration=180
)
family_models["lightgbm_recent"] = lgb_model

# Non-parametric shrinkage estimates constitute a sixth prediction family.
eb_model = fit_empirical_bayes(train, sample_weights_np.astype(np.float64))
valid_raw["empirical_bayes_recent"] = predict_empirical_bayes(valid, eb_model)
family_models["empirical_bayes_recent"] = eb_model

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared_dir, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared_dir, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

candidate_scores = {}
candidate_arrays = {}
candidate_raw_name = {}

for name, scores in valid_raw.items():
    standalone_metrics = evaluate(valid.user_id, valid.y, scores)
    candidate_scores[name] = float(standalone_metrics["primary"])
    candidate_arrays[name] = scores
    candidate_raw_name[name] = name

    own_rank = within_user_rank(valid.user_id, scores)
    for own_weight in (0.20, 0.35, 0.50, 0.65, 0.80):
        blend = own_weight * own_rank + (1.0 - own_weight) * inc_valid_rank
        blend_name = f"{name}_blend_{own_weight:.2f}"
        blend_metrics = evaluate(valid.user_id, valid.y, blend)
        candidate_scores[blend_name] = float(blend_metrics["primary"])
        candidate_arrays[blend_name] = blend
        candidate_raw_name[blend_name] = name

winner_name = max(candidate_scores, key=candidate_scores.get)
valid_scores = candidate_arrays[winner_name]
winner_family = candidate_raw_name[winner_name]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

if "_blend_" in winner_name:
    own_weight = float(winner_name.rsplit("_", 1)[1])
else:
    own_weight = None

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if own_weight is not None:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(valid_raw[winner_family], dtype=np.float64),
        )

# Generate test scores only after all choices have been made on validation.
test = load("test")

if winner_family in dict(torch_families):
    x_test = build_features(test)
    own_test_scores = predict_torch(
        family_models[winner_family], x_test
    )
elif winner_family == "lightgbm_recent":
    x_test_lgb = build_local_features(test)
    own_test_scores = family_models[winner_family].predict(
        x_test_lgb, num_iteration=180
    )
else:
    own_test_scores = predict_empirical_bayes(
        test, family_models[winner_family]
    )

if own_weight is None:
    test_scores = own_test_scores
else:
    inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
    own_test_rank = within_user_rank(test.user_id, own_test_scores)
    inc_test_rank = within_user_rank(test.user_id, inc_test)
    test_scores = (
        own_weight * own_test_rank
        + (1.0 - own_weight) * inc_test_rank
    )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START_TIME
result = {
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result, separators=(", ", ": ")))