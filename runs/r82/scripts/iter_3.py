import os
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 42
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "hour",
]
DIM = 16
BATCH_SIZE = 2048
EPOCHS = 5
LR = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
DEVICE = torch.device("cpu")

cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))


def make_numpy_features(split):
    return np.stack(
        [
            np.asarray(split.X[f], dtype=np.int64) + offsets[j]
            for j, f in enumerate(FIELDS)
        ],
        axis=1,
    )


def recency_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.int64)
    age = dates.max() - dates
    w = np.power(0.5, age.astype(np.float64) / float(half_life))
    w /= max(w.mean(), 1e-12)
    return w.astype(np.float32)


class FM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, DIM)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, 0.0, 0.01)

    def fm_parts(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear, interaction, v

    def forward(self, x):
        linear, interaction, _ = self.fm_parts(x)
        return linear + interaction


class DeepFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, DIM)
        self.bias = nn.Parameter(torch.zeros(1))
        d = len(FIELDS) * DIM
        self.deep = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, 0.0, 0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        fm = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        deep = self.deep(v.reshape(v.shape[0], -1)).squeeze(-1)
        return self.bias + linear + fm + deep


class NFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, DIM)
        self.bias = nn.Parameter(torch.zeros(1))
        self.interaction_net = nn.Sequential(
            nn.BatchNorm1d(DIM),
            nn.Linear(DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, 0.0, 0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        bi = 0.5 * (summed.square() - v.square().sum(dim=1))
        nonlinear = self.interaction_net(bi).squeeze(-1)
        return self.bias + linear + nonlinear


class CrossLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d))
        self.bias = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.weight, 0.0, 0.01)

    def forward(self, x0, x):
        scalar = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * scalar + self.bias + x


class DCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(total_cardinality, DIM)
        d = len(FIELDS) * DIM
        self.cross1 = CrossLayer(d)
        self.cross2 = CrossLayer(d)
        self.deep = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(d + 64, 1)
        nn.init.normal_(self.embedding.weight, 0.0, 0.01)

    def forward(self, x):
        x0 = self.embedding(x).reshape(x.shape[0], -1)
        cross = self.cross1(x0, x0)
        cross = self.cross2(x0, cross)
        deep = self.deep(x0)
        return self.output(torch.cat([cross, deep], dim=1)).squeeze(-1)


MODEL_CLASSES = {
    "fm_uniform": FM,
    "fm_recency4": FM,
    "deepfm": DeepFM,
    "nfm": NFM,
    "dcn": DCN,
}


def fit_torch(kind, x_np, y_np, weights=None, seed=SEED):
    torch.manual_seed(seed)
    model = MODEL_CLASSES[kind]().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    w = None if weights is None else torch.from_numpy(
        np.asarray(weights, dtype=np.float32)
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1000)
    n = len(y)

    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            xb = x[idx].to(DEVICE)
            yb = y[idx].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            if w is not None:
                wb = w[idx].to(DEVICE)
                loss = torch.sum(losses * wb) / torch.clamp(
                    wb.sum(), min=1e-6
                )
            else:
                loss = losses.mean()
            loss.backward()
            optimizer.step()

    return model


@torch.inference_mode()
def predict_torch(model, x_np, batch_size=16384):
    x = torch.from_numpy(x_np)
    out = np.empty(x_np.shape[0], dtype=np.float64)
    model.eval()
    for start in range(0, x_np.shape[0], batch_size):
        end = min(start + batch_size, x_np.shape[0])
        logits = model(x[start:end].to(DEVICE))
        out[start:end] = torch.sigmoid(logits).cpu().numpy()
    return out


def fit_lgbm(x, y, weights=None):
    ds = lgb.Dataset(
        x.astype(np.int32, copy=False),
        label=np.asarray(y, dtype=np.float32),
        weight=weights,
        categorical_feature=list(range(x.shape[1])),
        free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.06,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 150,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "num_threads": max(1, min(8, os.cpu_count() or 1)),
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "verbose": -1,
    }
    return lgb.train(params, ds, num_boost_round=260)


def fit_eb(x, y):
    y = np.asarray(y, dtype=np.float64)
    global_rate = np.clip(y.mean(), 1e-5, 1.0 - 1e-5)
    global_logit = np.log(global_rate / (1.0 - global_rate))
    tables = []

    # Exclude user_id because it is constant within each evaluated user.
    for j in range(1, x.shape[1]):
        ids = x[:, j]
        size = total_cardinality
        cnt = np.bincount(ids, minlength=size).astype(np.float64)
        pos = np.bincount(ids, weights=y, minlength=size).astype(np.float64)
        smoothing = 30.0 if j in (1, 2) else 80.0
        rate = (pos + smoothing * global_rate) / (cnt + smoothing)
        rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
        tables.append(np.log(rate / (1.0 - rate)) - global_logit)

    return global_logit, tables


def predict_eb(model, x):
    global_logit, tables = model
    score = np.full(x.shape[0], global_logit, dtype=np.float64)
    for j, table in enumerate(tables, start=1):
        score += table[x[:, j]]
    score = np.clip(score, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-score))


train = load("train")
valid = load("valid")
x_train = make_numpy_features(train)
x_valid = make_numpy_features(valid)
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.float32)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

base_predictions = {}
candidate_scores = {}
candidate_specs = {}

torch_kinds = ["fm_uniform", "fm_recency4", "deepfm", "nfm", "dcn"]
for model_index, kind in enumerate(torch_kinds):
    weights = None
    if kind == "fm_recency4":
        weights = recency_weights(train.date, 4.0)

    model = fit_torch(
        kind,
        x_train,
        y_train,
        weights=weights,
        seed=SEED + model_index * 17,
    )
    pred = predict_torch(model, x_valid)
    base_predictions[kind] = pred
    del model
    gc.collect()

lgb_model = fit_lgbm(x_train, y_train)
base_predictions["lightgbm"] = np.asarray(
    lgb_model.predict(x_valid), dtype=np.float64
)
del lgb_model
gc.collect()

eb_model = fit_eb(x_train, y_train)
base_predictions["empirical_bayes"] = predict_eb(eb_model, x_valid)
del eb_model
gc.collect()

best_primary = -np.inf
best_valid_scores = None
best_family = None
best_alpha = None

blend_grid = np.linspace(0.0, 1.0, 11)
for family, pred in base_predictions.items():
    standalone_metrics = evaluate(valid.user_id, valid.y, pred)
    candidate_scores[family] = float(standalone_metrics["primary"])

    family_best_score = -np.inf
    family_best_alpha = None
    family_best_pred = None

    for alpha in blend_grid:
        blended = alpha * pred + (1.0 - alpha) * inc_valid
        m = evaluate(valid.user_id, valid.y, blended)
        p = float(m["primary"])
        if p > family_best_score:
            family_best_score = p
            family_best_alpha = float(alpha)
            family_best_pred = blended.copy()

    blend_name = family + "_blend"
    candidate_scores[blend_name] = family_best_score
    candidate_specs[blend_name] = (family, family_best_alpha)

    if family_best_score > best_primary:
        best_primary = family_best_score
        best_valid_scores = family_best_pred
        best_family = family
        best_alpha = family_best_alpha

metrics = evaluate(valid.user_id, valid.y, best_valid_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "selected_family": best_family,
            "new_model_weight": best_alpha,
            "incumbent_weight": 1.0 - best_alpha,
            "selected_primary": float(metrics["primary"]),
        },
        sort_keys=True,
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )

# Refit the selected recipe on train+validation and score test.
test = load("test")
x_test = make_numpy_features(test)
x_combined = np.concatenate([x_train, x_valid], axis=0)
y_combined = np.concatenate([y_train, y_valid], axis=0)

if best_family in MODEL_CLASSES:
    combined_weights = None
    if best_family == "fm_recency4":
        combined_dates = np.concatenate(
            [
                np.asarray(train.date, dtype=np.int64),
                np.asarray(valid.date, dtype=np.int64),
            ]
        )
        combined_weights = recency_weights(combined_dates, 4.0)

    selected_index = torch_kinds.index(best_family)
    final_model = fit_torch(
        best_family,
        x_combined,
        y_combined,
        weights=combined_weights,
        seed=SEED + selected_index * 17,
    )
    new_test_scores = predict_torch(final_model, x_test)
    del final_model
elif best_family == "lightgbm":
    final_model = fit_lgbm(x_combined, y_combined)
    new_test_scores = np.asarray(
        final_model.predict(x_test), dtype=np.float64
    )
    del final_model
else:
    final_model = fit_eb(x_combined, y_combined)
    new_test_scores = predict_eb(final_model, x_test)
    del final_model

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
test_scores = (
    best_alpha * new_test_scores
    + (1.0 - best_alpha) * inc_test
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.6f}'
    % (
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        float(elapsed),
    )
)