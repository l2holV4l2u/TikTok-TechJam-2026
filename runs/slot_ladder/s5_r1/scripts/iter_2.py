import os
import time
import json
import random
import numpy as np
import torch
from torch import nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2025
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
K = 16
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 65536
EPOCHS = 4
LR = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

OFFSETS = []
_total = 0
for field in FIELDS:
    OFFSETS.append(_total)
    _total += int(FEATURE_CARDINALITIES[field])
OFFSETS = np.asarray(OFFSETS, dtype=np.int64)
TOTAL_CARDINALITY = _total


def build_cat_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[name], dtype=np.int64) + OFFSETS[j]
            for j, name in enumerate(FIELDS)
        ]),
        dtype=np.int64,
    )


def recency_weights(dates, half_life):
    age = np.max(dates).astype(np.int64) - np.asarray(dates, dtype=np.int64)
    w = np.exp2(-age.astype(np.float32) / float(half_life))
    return w / np.mean(w)


class FM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.factor = nn.Embedding(TOTAL_CARDINALITY, K)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factor.weight, 0.0, 0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(1)
        v = self.factor(x)
        sv = v.sum(dim=1)
        interaction = 0.5 * (
            sv.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


class WideLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.bias + self.linear(x).sum(dim=1).squeeze(1)


class DeepFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.factor = nn.Embedding(TOTAL_CARDINALITY, K)
        self.bias = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(len(FIELDS) * K, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factor.weight, 0.0, 0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(1)
        v = self.factor(x)
        sv = v.sum(dim=1)
        fm = 0.5 * (sv.square() - v.square().sum(dim=1)).sum(dim=1)
        deep = self.mlp(v.reshape(v.shape[0], -1)).squeeze(1)
        return self.bias + linear + fm + deep


class NFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(TOTAL_CARDINALITY, 1)
        self.factor = nn.Embedding(TOTAL_CARDINALITY, K)
        self.bias = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(K, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factor.weight, 0.0, 0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(1)
        v = self.factor(x)
        sv = v.sum(dim=1)
        bi = 0.5 * (sv.square() - v.square().sum(dim=1))
        nonlinear = self.mlp(bi).squeeze(1)
        return self.bias + linear + nonlinear


def train_torch_model(model_cls, x_np, y_np, sample_weight, seed_offset):
    torch.manual_seed(SEED + seed_offset)
    model = model_cls()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    weight = torch.from_numpy(np.asarray(sample_weight, dtype=np.float32))
    n = x.shape[0]
    generator = torch.Generator()
    generator.manual_seed(SEED + seed_offset)

    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(x[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, y[idx], reduction="none"
            )
            loss = (losses * weight[idx]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


@torch.inference_mode()
def predict_torch(model, x_np):
    model.eval()
    x = torch.from_numpy(x_np)
    out = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, x.shape[0])
        out[start:end] = (
            model(x[start:end]).cpu().numpy().astype(np.float64)
        )
    return out


def find_hist_columns(split_name):
    vh = historical_features(split_name, key="video_id")
    ah = historical_features(split_name, key="author_id")

    def choose(hist, suffix):
        exact = [k for k in hist if k.endswith(suffix)]
        if not exact:
            exact = [k for k in hist if suffix in k]
        if not exact:
            raise RuntimeError("Missing historical feature: " + suffix)
        return np.asarray(hist[exact[0]], dtype=np.float32)

    return {
        "video_count": choose(vh, "train_count_log1p"),
        "video_rate": choose(vh, "long_view_rate"),
        "author_count": choose(ah, "train_count_log1p"),
        "author_rate": choose(ah, "long_view_rate"),
    }


def build_tree_matrix(split, hist):
    columns = []
    for name in FIELDS:
        columns.append(np.asarray(split.X[name], dtype=np.float32))

    for name in [
        "duration_ms",
        "user_fans_user_num",
        "user_follow_user_num",
        "user_friend_user_num",
        "user_register_days",
    ]:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))

    columns.extend([
        hist["video_count"],
        hist["video_rate"],
        hist["author_count"],
        hist["author_rate"],
    ])
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def recent_entity_tables(train):
    y = np.asarray(train.y, dtype=np.float64)
    dates = np.asarray(train.date, dtype=np.int64)
    mask = dates >= (dates.max() - 3)
    global_rate = float(y[mask].mean())

    tables = {}
    for name, prior in [("video_id", 20.0), ("author_id", 30.0)]:
        ids = np.asarray(train.X[name], dtype=np.int64)
        cardinality = int(FEATURE_CARDINALITIES[name])
        count = np.bincount(ids[mask], minlength=cardinality).astype(np.float64)
        positive = np.bincount(
            ids[mask], weights=y[mask], minlength=cardinality
        ).astype(np.float64)
        rate = (positive + prior * global_rate) / (count + prior)
        tables[name] = rate
    return global_rate, tables


def empirical_predict(split, global_rate, tables):
    vr = tables["video_id"][np.asarray(split.X["video_id"], dtype=np.int64)]
    ar = tables["author_id"][np.asarray(split.X["author_id"], dtype=np.int64)]
    p = np.clip(0.65 * vr + 0.35 * ar, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float64)


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    std = float(np.std(x))
    if std < 1e-12:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


train = load("train")
valid = load("valid")
x_train = build_cat_matrix(train)
x_valid = build_cat_matrix(valid)

weights_by_half_life = {
    h: recency_weights(train.date, h) for h in (2.0, 4.0, 8.0)
}

models = {}
valid_predictions = {}

for j, half_life in enumerate((2.0, 4.0, 8.0)):
    name = "recency_fm_h" + str(int(half_life))
    model = train_torch_model(
        FM, x_train, train.y, weights_by_half_life[half_life], 10 + j
    )
    models[name] = ("torch", model)
    valid_predictions[name] = predict_torch(model, x_valid)

wide = train_torch_model(
    WideLinear, x_train, train.y, weights_by_half_life[4.0], 20
)
models["recency_wide_linear"] = ("torch", wide)
valid_predictions["recency_wide_linear"] = predict_torch(wide, x_valid)

deepfm = train_torch_model(
    DeepFM, x_train, train.y, weights_by_half_life[4.0], 30
)
models["recency_deepfm"] = ("torch", deepfm)
valid_predictions["recency_deepfm"] = predict_torch(deepfm, x_valid)

nfm = train_torch_model(
    NFM, x_train, train.y, weights_by_half_life[4.0], 40
)
models["recency_nfm"] = ("torch", nfm)
valid_predictions["recency_nfm"] = predict_torch(nfm, x_valid)

hist_train = find_hist_columns("train")
hist_valid = find_hist_columns("valid")
tree_train = build_tree_matrix(train, hist_train)
tree_valid = build_tree_matrix(valid, hist_valid)

tree_dataset = lgb.Dataset(
    tree_train,
    label=np.asarray(train.y, dtype=np.float32),
    weight=weights_by_half_life[4.0],
    categorical_feature=list(range(len(FIELDS))),
    free_raw_data=True,
)
tree_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.06,
    "num_leaves": 63,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "num_threads": max(1, min(8, os.cpu_count() or 1)),
    "verbose": -1,
}
tree_model = lgb.train(tree_params, tree_dataset, num_boost_round=260)
models["recency_lightgbm"] = ("tree", tree_model)
valid_predictions["recency_lightgbm"] = tree_model.predict(
    tree_valid, num_iteration=tree_model.current_iteration()
).astype(np.float64)

global_rate, entity_tables = recent_entity_tables(train)
models["recent_empirical_bayes"] = ("empirical", None)
valid_predictions["recent_empirical_bayes"] = empirical_predict(
    valid, global_rate, entity_tables
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_valid_z = zscore(inc_valid)

candidate_log = {}
standalone_metrics = {}
best = None
blend_weights = (0.15, 0.25, 0.35, 0.50, 0.70, 1.00)

for name, scores in valid_predictions.items():
    own_metrics = evaluate(valid.user_id, valid.y, scores)
    standalone_metrics[name] = float(own_metrics["primary"])
    candidate_log[name + "_raw"] = float(own_metrics["primary"])

    own_z = zscore(scores)
    for weight in blend_weights:
        if weight == 1.0:
            blended = own_z
        else:
            blended = (1.0 - weight) * inc_valid_z + weight * own_z
        result = evaluate(valid.user_id, valid.y, blended)
        key = name + "_blend_" + ("%g" % weight)
        candidate_log[key] = float(result["primary"])
        record = (
            float(result["primary"]),
            name,
            float(weight),
            blended.copy(),
            result,
        )
        if best is None or record[0] > best[0]:
            best = record

best_primary, best_name, best_weight, valid_scores, metrics = best

print(
    "FINDINGS " + json.dumps(
        {
            "standalone_primary": standalone_metrics,
            "winner": best_name,
            "winner_blend_weight": best_weight,
            "winner_primary": best_primary,
        },
        separators=(",", ":"),
    )
)
print("CANDIDATES " + json.dumps(candidate_log, separators=(",", ":")))

test = load("test")
x_test = None
best_kind, best_model = models[best_name]

if best_kind == "torch":
    x_test = build_cat_matrix(test)
    own_test_scores = predict_torch(best_model, x_test)
elif best_kind == "tree":
    hist_test = find_hist_columns("test")
    tree_test = build_tree_matrix(test, hist_test)
    own_test_scores = best_model.predict(
        tree_test, num_iteration=best_model.current_iteration()
    ).astype(np.float64)
else:
    own_test_scores = empirical_predict(test, global_rate, entity_tables)

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if best_weight == 1.0:
    test_scores = zscore(own_test_scores)
else:
    test_scores = (
        (1.0 - best_weight) * zscore(inc_test)
        + best_weight * zscore(own_test_scores)
    )

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(valid_predictions[best_name], dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS " + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(",", ":"),
    )
)