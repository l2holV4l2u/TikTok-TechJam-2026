import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 271828
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
]
CARDINALITIES = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
OFFSETS = np.cumsum([0] + CARDINALITIES[:-1], dtype=np.int64)
NUM_FEATURES = int(sum(CARDINALITIES))
NUM_FIELDS = len(FIELDS)

EMBED_DIM = 12
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768
EPOCHS = 3
LR = 0.002
HALF_LIFE_DAYS = 4.0


def make_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[field], dtype=np.int64) + offset
            for field, offset in zip(FIELDS, OFFSETS)
        ]),
        dtype=np.int64,
    )


def sigmoid_np(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def as_probability_scale(x):
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
        return np.clip(x, 1e-7, 1.0 - 1e-7)
    return sigmoid_np(x)


class FieldAwareFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(NUM_FEATURES, 1)
        self.ffm = nn.Embedding(NUM_FEATURES * NUM_FIELDS, EMBED_DIM)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.ffm.weight, mean=0.0, std=0.02)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(1) + self.bias
        interaction = torch.zeros_like(linear)
        for i in range(NUM_FIELDS):
            xi = x[:, i]
            for j in range(i + 1, NUM_FIELDS):
                xj = x[:, j]
                vi = self.ffm(xi * NUM_FIELDS + j)
                vj = self.ffm(xj * NUM_FIELDS + i)
                interaction = interaction + torch.sum(vi * vj, dim=1)
        return linear + interaction


class ProductNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(NUM_FEATURES, EMBED_DIM)
        self.linear = nn.Embedding(NUM_FEATURES, 1)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear.weight)

        self.pairs = [
            (i, j)
            for i in range(NUM_FIELDS)
            for j in range(i + 1, NUM_FIELDS)
        ]
        input_dim = NUM_FIELDS * EMBED_DIM + len(self.pairs)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 160),
            nn.ReLU(),
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        e = self.embedding(x)
        products = torch.stack(
            [torch.sum(e[:, i] * e[:, j], dim=1) for i, j in self.pairs],
            dim=1,
        )
        features = torch.cat([e.reshape(e.shape[0], -1), products], dim=1)
        return (
            self.linear(x).sum(dim=1).squeeze(1)
            + self.network(features).squeeze(1)
        )


class BPRLatentModel(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 24
        self.user_embedding = nn.Embedding(CARDINALITIES[0], dim)
        self.video_embedding = nn.Embedding(CARDINALITIES[1], dim)
        self.author_embedding = nn.Embedding(CARDINALITIES[2], dim)
        self.video_bias = nn.Embedding(CARDINALITIES[1], 1)
        self.author_bias = nn.Embedding(CARDINALITIES[2], 1)
        for embedding in (
            self.user_embedding,
            self.video_embedding,
            self.author_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.03)
        nn.init.zeros_(self.video_bias.weight)
        nn.init.zeros_(self.author_bias.weight)

    def forward(self, x):
        user = x[:, 0] - OFFSETS[0]
        video = x[:, 1] - OFFSETS[1]
        author = x[:, 2] - OFFSETS[2]
        u = self.user_embedding(user)
        score = torch.sum(u * self.video_embedding(video), dim=1)
        score = score + 0.55 * torch.sum(
            u * self.author_embedding(author), dim=1
        )
        score = score + self.video_bias(video).squeeze(1)
        score = score + 0.5 * self.author_bias(author).squeeze(1)
        return score


def train_pointwise(model, x_tensor, y_tensor, weight_tensor, seed):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator()
    generator.manual_seed(seed)
    n = x_tensor.shape[0]

    model.train()
    for _ in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            xb = x_tensor.index_select(0, idx)
            yb = y_tensor.index_select(0, idx)
            wb = weight_tensor.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = torch.sum(losses * wb) / torch.sum(wb)
            loss.backward()
            optimizer.step()


def train_bpr(model, x_np, y_np, user_np, recency_weight, seed):
    rng = np.random.default_rng(seed)
    user_cardinality = CARDINALITIES[0]

    neg_rows = np.flatnonzero(y_np == 0)
    neg_order = np.argsort(user_np[neg_rows], kind="stable")
    neg_rows = neg_rows[neg_order]
    neg_users = user_np[neg_rows]

    neg_counts = np.bincount(
        neg_users, minlength=user_cardinality
    ).astype(np.int64)
    neg_starts = np.zeros(user_cardinality, dtype=np.int64)
    if user_cardinality > 1:
        neg_starts[1:] = np.cumsum(neg_counts[:-1])

    pos_rows = np.flatnonzero(y_np == 1)
    pos_users = user_np[pos_rows]
    usable = neg_counts[pos_users] > 0
    pos_rows = pos_rows[usable]
    pos_users = pos_users[usable]

    x_tensor = torch.from_numpy(x_np)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)
    n_pos = pos_rows.size

    model.train()
    for _ in range(4):
        offsets = (
            rng.random(n_pos) * neg_counts[pos_users]
        ).astype(np.int64)
        sampled_neg_rows = neg_rows[neg_starts[pos_users] + offsets]
        permutation = rng.permutation(n_pos)

        for start in range(0, n_pos, BATCH_SIZE):
            p = permutation[start:start + BATCH_SIZE]
            pos_idx_np = pos_rows[p]
            neg_idx_np = sampled_neg_rows[p]

            pos_idx = torch.from_numpy(pos_idx_np)
            neg_idx = torch.from_numpy(neg_idx_np)
            xp = x_tensor.index_select(0, pos_idx)
            xn = x_tensor.index_select(0, neg_idx)
            weights = torch.from_numpy(
                recency_weight[pos_idx_np].astype(np.float32, copy=False)
            )

            optimizer.zero_grad(set_to_none=True)
            margin = model(xp) - model(xn)
            losses = F.softplus(-margin)
            loss = torch.sum(losses * weights) / torch.sum(weights)
            loss.backward()
            optimizer.step()


def predict_model(model, matrix):
    result = np.empty(matrix.shape[0], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, matrix.shape[0], PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, matrix.shape[0])
            xb = torch.from_numpy(matrix[start:end])
            result[start:end] = model(xb).cpu().numpy()
    return result


def build_target_tables(train):
    global_rate = float(np.mean(train.y))
    tables = {}
    smoothing = {
        "video_id": 30.0,
        "author_id": 50.0,
        "tab": 300.0,
        "duration_bucket": 300.0,
        "tag": 150.0,
        "upload_type": 250.0,
        "music_type": 250.0,
    }
    for field, prior_strength in smoothing.items():
        ids = np.asarray(train.X[field], dtype=np.int64)
        cardinality = int(FEATURE_CARDINALITIES[field])
        counts = np.bincount(ids, minlength=cardinality).astype(np.float64)
        positives = np.bincount(
            ids,
            weights=np.asarray(train.y, dtype=np.float64),
            minlength=cardinality,
        )
        rates = (
            positives + prior_strength * global_rate
        ) / (counts + prior_strength)
        rates[counts == 0] = global_rate
        tables[field] = rates
    return global_rate, tables


def empirical_scores(split_name, split, global_rate, tables):
    video_hist = historical_features(split_name, key="video_id")
    author_hist = historical_features(split_name, key="author_id")

    video_rate = np.asarray(
        video_hist["video_id_long_view_rate"], dtype=np.float64
    )
    author_rate = np.asarray(
        author_hist["author_id_long_view_rate"], dtype=np.float64
    )

    eps = 1e-5
    base_logit = np.log(global_rate / (1.0 - global_rate))
    video_logit = np.log(
        np.clip(video_rate, eps, 1.0 - eps)
        / (1.0 - np.clip(video_rate, eps, 1.0 - eps))
    )
    author_logit = np.log(
        np.clip(author_rate, eps, 1.0 - eps)
        / (1.0 - np.clip(author_rate, eps, 1.0 - eps))
    )

    score = 0.70 * video_logit + 0.30 * author_logit
    field_weights = {
        "tab": 0.24,
        "duration_bucket": 0.18,
        "tag": 0.16,
        "upload_type": 0.10,
        "music_type": 0.08,
    }
    for field, weight in field_weights.items():
        rate = tables[field][np.asarray(split.X[field], dtype=np.int64)]
        rate = np.clip(rate, eps, 1.0 - eps)
        field_logit = np.log(rate / (1.0 - rate))
        score += weight * (field_logit - base_logit)
    return score


train = load("train")
x_train_np = make_matrix(train)
y_train_np = np.asarray(train.y, dtype=np.float32)
user_train_np = np.asarray(train.X["user_id"], dtype=np.int64)

max_date = int(np.max(train.date))
date_values = np.asarray(train.date, dtype=np.int64)
unique_dates = np.sort(np.unique(date_values))
date_to_age = {
    int(date): int(len(unique_dates) - 1 - index)
    for index, date in enumerate(unique_dates)
}
age_days = np.fromiter(
    (date_to_age[int(d)] for d in date_values),
    dtype=np.float32,
    count=date_values.size,
)
recency_weight_np = np.exp2(-age_days / HALF_LIFE_DAYS).astype(np.float32)
recency_weight_np /= np.mean(recency_weight_np)

x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)
weight_train = torch.from_numpy(recency_weight_np)

models = {
    "field_aware_fm": FieldAwareFM(),
    "product_network": ProductNetwork(),
    "bpr_latent": BPRLatentModel(),
}

train_pointwise(
    models["field_aware_fm"],
    x_train,
    y_train,
    weight_train,
    SEED + 10,
)
train_pointwise(
    models["product_network"],
    x_train,
    y_train,
    weight_train,
    SEED + 20,
)
train_bpr(
    models["bpr_latent"],
    x_train_np,
    y_train_np,
    user_train_np,
    recency_weight_np,
    SEED + 30,
)

global_rate, target_tables = build_target_tables(train)

del x_train, y_train, weight_train

valid = load("valid")
x_valid = make_matrix(valid)

raw_valid = {}
raw_valid["field_aware_fm"] = sigmoid_np(
    predict_model(models["field_aware_fm"], x_valid)
)
raw_valid["product_network"] = sigmoid_np(
    predict_model(models["product_network"], x_valid)
)
raw_valid["bpr_latent"] = sigmoid_np(
    predict_model(models["bpr_latent"], x_valid)
)
raw_valid["empirical_bayes"] = sigmoid_np(
    empirical_scores("valid", valid, global_rate, target_tables)
)

candidate_scores = {}
candidate_specs = {}

for name, scores in raw_valid.items():
    metric = evaluate(valid.user_id, valid.y, scores)
    candidate_scores[name] = float(metric["primary"])
    candidate_specs[name] = {
        "family": name,
        "alpha": 1.0,
        "blended": False,
        "scores": scores,
    }

shared_dir = os.environ.get("SHARED_ARTIFACTS")
inc_valid = None
inc_test_path = None
if shared_dir:
    valid_path = os.path.join(shared_dir, "incumbent_valid_scores.npy")
    test_path = os.path.join(shared_dir, "incumbent_test_scores.npy")
    if os.path.exists(valid_path) and os.path.exists(test_path):
        inc_valid = as_probability_scale(np.load(valid_path))
        inc_test_path = test_path

blend_alphas = [0.10, 0.20, 0.35, 0.50, 0.65]
if inc_valid is not None:
    for name, scores in raw_valid.items():
        for alpha in blend_alphas:
            blended = alpha * scores + (1.0 - alpha) * inc_valid
            candidate_name = f"{name}_blend_{alpha:.2f}"
            metric = evaluate(valid.user_id, valid.y, blended)
            candidate_scores[candidate_name] = float(metric["primary"])
            candidate_specs[candidate_name] = {
                "family": name,
                "alpha": alpha,
                "blended": True,
                "scores": blended,
            }

winner_name = max(candidate_scores, key=candidate_scores.get)
winner = candidate_specs[winner_name]
valid_scores = winner["scores"]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print(
    "FINDINGS recency_weight_range="
    + json.dumps({
        "half_life_days": HALF_LIFE_DAYS,
        "min": float(recency_weight_np.min()),
        "max": float(recency_weight_np.max()),
    })
)
print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if winner["blended"]:
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(raw_valid[winner["family"]], dtype=np.float64),
        )

del x_valid, valid

test = load("test")
family = winner["family"]

if family == "empirical_bayes":
    test_raw = sigmoid_np(
        empirical_scores("test", test, global_rate, target_tables)
    )
else:
    x_test = make_matrix(test)
    test_raw = sigmoid_np(predict_model(models[family], x_test))

if winner["blended"]:
    incumbent_test = as_probability_scale(np.load(inc_test_path))
    alpha = float(winner["alpha"])
    test_scores = alpha * test_raw + (1.0 - alpha) * incumbent_test
else:
    test_scores = test_raw

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
result = {
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result))