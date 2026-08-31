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
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

CAT_FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
EMBED_DIM = 12
BATCH_SIZE = 8192
EPOCHS = 5
LR = 1.0e-3
HALF_LIFE = 4.0

train = load("train")
valid = load("valid")
y_train_np = np.asarray(train.y, dtype=np.float32)
y_valid_np = np.asarray(valid.y, dtype=np.int8)


offsets = []
total_cardinality = 0
for field in CAT_FIELDS:
    offsets.append(total_cardinality)
    total_cardinality += int(FEATURE_CARDINALITIES[field])
offsets = np.asarray(offsets, dtype=np.int64)


def build_cat_matrix(split):
    columns = []
    for j, field in enumerate(CAT_FIELDS):
        columns.append(np.asarray(split.X[field], dtype=np.int64) + offsets[j])
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.int64)


x_train_np = build_cat_matrix(train)
x_valid_np = build_cat_matrix(valid)
x_train = torch.from_numpy(x_train_np)
y_train = torch.from_numpy(y_train_np)

max_train_date = int(np.max(np.asarray(train.date)))
ages = max_train_date - np.asarray(train.date, dtype=np.int32)
recency_weights_np = np.exp(-np.log(2.0) * ages / HALF_LIFE).astype(np.float32)
recency_weights_np /= recency_weights_np.mean()
recency_weights = torch.from_numpy(recency_weights_np)


class BaseCTR(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.015)

    def common(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        emb = self.embedding(x)
        return linear, emb


class DeepFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = BaseCTR()
        dim = len(CAT_FIELDS) * EMBED_DIM
        self.deep = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        linear, emb = self.base.common(x)
        summed = emb.sum(dim=1)
        fm = 0.5 * (summed.square() - emb.square().sum(dim=1)).sum(dim=1)
        deep = self.deep(emb.flatten(1)).squeeze(1)
        return self.base.bias + linear + fm + deep


class NFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = BaseCTR()
        self.interaction_net = nn.Sequential(
            nn.BatchNorm1d(EMBED_DIM),
            nn.Linear(EMBED_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        linear, emb = self.base.common(x)
        summed = emb.sum(dim=1)
        bi = 0.5 * (summed.square() - emb.square().sum(dim=1))
        interaction = self.interaction_net(bi).squeeze(1)
        return self.base.bias + linear + interaction


class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x0, x):
        scale = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * scale + self.bias + x


class DCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = BaseCTR()
        dim = len(CAT_FIELDS) * EMBED_DIM
        self.crosses = nn.ModuleList([CrossLayer(dim) for _ in range(3)])
        self.deep = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(dim + 64, 1)

    def forward(self, x):
        linear, emb = self.base.common(x)
        x0 = emb.flatten(1)
        cross = x0
        for layer in self.crosses:
            cross = layer(x0, cross)
        deep = self.deep(x0)
        nonlinear = self.output(torch.cat([cross, deep], dim=1)).squeeze(1)
        return self.base.bias + linear + nonlinear


def train_torch_model(model, seed_offset):
    generator = torch.Generator()
    generator.manual_seed(SEED + seed_offset)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(y_train)
    model.train()

    for epoch in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator)
        loss_sum = 0.0
        weight_sum = 0.0

        for start in range(0, n, BATCH_SIZE):
            idx = permutation[start:min(start + BATCH_SIZE, n)]
            xb = x_train.index_select(0, idx)
            yb = y_train.index_select(0, idx)
            wb = recency_weights.index_select(0, idx)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = torch.sum(losses * wb) / torch.sum(wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loss_sum += float(torch.sum(losses.detach() * wb))
            weight_sum += float(torch.sum(wb))

        print(
            "TRAIN model=%s epoch=%d weighted_loss=%.6f"
            % (model.__class__.__name__, epoch + 1, loss_sum / weight_sum),
            flush=True,
        )
    return model


def torch_predict(model, matrix, batch_size=32768):
    model.eval()
    result = np.empty(len(matrix), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(matrix), batch_size):
            end = min(start + batch_size, len(matrix))
            xb = torch.from_numpy(matrix[start:end])
            result[start:end] = model(xb).cpu().numpy()
    return result


models = {}
valid_predictions = {}

for name, constructor, seed_offset in [
    ("deepfm_recent", DeepFM, 11),
    ("nfm_recent", NFM, 23),
    ("dcn_recent", DCN, 37),
]:
    torch.manual_seed(SEED + seed_offset)
    model = train_torch_model(constructor(), seed_offset)
    models[name] = model
    valid_predictions[name] = torch_predict(model, x_valid_np)


def smoothed_logit_table(ids, labels, cardinality, smoothing):
    counts = np.bincount(ids, minlength=cardinality).astype(np.float64)
    positives = np.bincount(
        ids, weights=labels.astype(np.float64), minlength=cardinality
    )
    prior = float(labels.mean())
    rates = (positives + smoothing * prior) / (counts + smoothing)
    rates = np.clip(rates, 1.0e-5, 1.0 - 1.0e-5)
    return np.log(rates / (1.0 - rates)).astype(np.float32)


empirical_tables = {}
empirical_specs = [
    ("video_id", 40.0, 1.00),
    ("author_id", 60.0, 0.75),
    ("tab", 400.0, 0.35),
    ("duration_bucket", 500.0, 0.40),
    ("tag", 150.0, 0.35),
    ("upload_type", 200.0, 0.25),
]

for field, smoothing, coefficient in empirical_specs:
    empirical_tables[field] = (
        smoothed_logit_table(
            np.asarray(train.X[field], dtype=np.int64),
            y_train_np,
            int(FEATURE_CARDINALITIES[field]),
            smoothing,
        ),
        coefficient,
    )


def empirical_predict(split):
    score = np.zeros(len(split.user_id), dtype=np.float32)
    for field, (table, coefficient) in empirical_tables.items():
        ids = np.asarray(split.X[field], dtype=np.int64)
        score += coefficient * table[ids]
    return score


valid_predictions["empirical_bayes"] = empirical_predict(valid)


LGB_CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "hour",
]
LGB_NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def build_lgb_matrix(split):
    cols = [
        np.asarray(split.X[field], dtype=np.float32) for field in LGB_CAT_FIELDS
    ]
    for field in LGB_NUM_FIELDS:
        x = np.asarray(split.num[field], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float32)


x_lgb_train = build_lgb_matrix(train)
x_lgb_valid = build_lgb_matrix(valid)
lgb_dataset = lgb.Dataset(
    x_lgb_train,
    label=y_train_np,
    weight=recency_weights_np,
    categorical_feature=list(range(len(LGB_CAT_FIELDS))),
    free_raw_data=True,
)

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.06,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "cat_smooth": 40.0,
    "cat_l2": 15.0,
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "num_threads": max(1, min(8, os.cpu_count() or 1)),
    "verbose": -1,
}

lgb_model = lgb.train(lgb_params, lgb_dataset, num_boost_round=140)
valid_predictions["lightgbm_recent"] = lgb_model.predict(
    x_lgb_valid
).astype(np.float32)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)


def within_user_percentile(user_ids, scores):
    users = np.asarray(user_ids)
    values = np.asarray(scores, dtype=np.float64)
    n = len(values)
    order = np.lexsort((np.arange(n, dtype=np.int64), values, users))
    sorted_users = users[order]

    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    new_group[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(new_group)
    ends = np.r_[starts[1:], n]
    counts = ends - starts

    repeated_starts = np.repeat(starts, counts)
    repeated_denoms = np.repeat(np.maximum(counts - 1, 1), counts)
    sorted_ranks = (np.arange(n) - repeated_starts) / repeated_denoms

    result = np.empty(n, dtype=np.float64)
    result[order] = sorted_ranks
    return result


inc_valid_rank = within_user_percentile(valid.user_id, inc_valid)
candidate_scores = {}
candidate_metrics = {}
candidate_log = {}

for name, prediction in valid_predictions.items():
    raw_metric = evaluate(valid.user_id, y_valid_np, prediction)
    candidate_log[name + "_raw"] = float(raw_metric["primary"])

    new_rank = within_user_percentile(valid.user_id, prediction)
    best_score = None
    best_metric = None
    best_weight = None

    for weight in [0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]:
        blended = (1.0 - weight) * inc_valid_rank + weight * new_rank
        metric = evaluate(valid.user_id, y_valid_np, blended)
        if best_metric is None or metric["primary"] > best_metric["primary"]:
            best_metric = metric
            best_score = blended
            best_weight = weight

    key = name + "_rank_blend_w%.2f" % best_weight
    candidate_scores[key] = best_score
    candidate_metrics[key] = best_metric
    candidate_log[key] = float(best_metric["primary"])

winner_key = max(
    candidate_metrics.keys(),
    key=lambda key: candidate_metrics[key]["primary"],
)
winner_metric = candidate_metrics[winner_key]
valid_scores = candidate_scores[winner_key]

winner_family = winner_key.split("_rank_blend_w")[0]
winner_weight = float(winner_key.split("_rank_blend_w")[1])
winner_raw_valid = valid_predictions[winner_family]

print(
    "FINDINGS "
    + json.dumps(
        {
            "winner_family": winner_family,
            "winner_new_model_weight": winner_weight,
            "recency_half_life_days": HALF_LIFE,
        },
        sort_keys=True,
    ),
    flush=True,
)
print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True), flush=True)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(winner_raw_valid, dtype=np.float64),
    )

test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if winner_family in models:
    x_test_np = build_cat_matrix(test)
    winner_raw_test = torch_predict(models[winner_family], x_test_np)
elif winner_family == "empirical_bayes":
    winner_raw_test = empirical_predict(test)
elif winner_family == "lightgbm_recent":
    x_lgb_test = build_lgb_matrix(test)
    winner_raw_test = lgb_model.predict(x_lgb_test).astype(np.float32)
else:
    raise RuntimeError("Unknown winning family: " + winner_family)

inc_test_rank = within_user_percentile(test.user_id, inc_test)
winner_test_rank = within_user_percentile(test.user_id, winner_raw_test)
test_scores = (
    (1.0 - winner_weight) * inc_test_rank
    + winner_weight * winner_test_rank
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(winner_metric["primary"]),
            "gauc": float(winner_metric["gauc"]),
            "ndcg@5": float(winner_metric["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)