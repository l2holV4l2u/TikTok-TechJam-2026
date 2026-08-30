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


START_TIME = time.time()
SEED = 2026
FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour",
]
BATCH_SIZE = 4096

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_torch_matrix(split):
    cols = [
        np.asarray(split.X[f], dtype=np.int64) + offsets[j]
        for j, f in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.int64)


def make_lgb_matrix(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[f], dtype=np.int32) for f in FIELDS
        ]),
        dtype=np.int32,
    )


def standardized(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(x.std())
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    return (x - float(x.mean())) / sd


def recency_weights(dates, half_life=7.0):
    dates = np.asarray(dates, dtype=np.float32)
    age = float(dates.max()) - dates
    w = np.exp2(-age / float(half_life)).astype(np.float32)
    w /= max(float(w.mean()), 1e-8)
    return w


class WideModel(nn.Module):
    def __init__(self, n_categories, initial_rate):
        super().__init__()
        self.linear = nn.Embedding(n_categories, 1)
        p = float(np.clip(initial_rate, 1e-5, 1 - 1e-5))
        self.bias = nn.Parameter(
            torch.tensor(np.log(p / (1.0 - p)), dtype=torch.float32)
        )
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.bias + self.linear(x).sum(dim=1).squeeze(1)


class FactorizationMachine(nn.Module):
    def __init__(self, n_categories, embedding_dim, initial_rate):
        super().__init__()
        self.linear = nn.Embedding(n_categories, 1)
        self.embedding = nn.Embedding(n_categories, embedding_dim)
        p = float(np.clip(initial_rate, 1e-5, 1 - 1e-5))
        self.bias = nn.Parameter(
            torch.tensor(np.log(p / (1.0 - p)), dtype=torch.float32)
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(1)
        emb = self.embedding(x)
        summed = emb.sum(dim=1)
        interactions = 0.5 * (
            summed.square() - emb.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interactions


class DeepFM(nn.Module):
    def __init__(self, n_categories, n_fields, embedding_dim, initial_rate):
        super().__init__()
        self.linear = nn.Embedding(n_categories, 1)
        self.embedding = nn.Embedding(n_categories, embedding_dim)
        self.deep = nn.Sequential(
            nn.Linear(n_fields * embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        p = float(np.clip(initial_rate, 1e-5, 1 - 1e-5))
        self.bias = nn.Parameter(
            torch.tensor(np.log(p / (1.0 - p)), dtype=torch.float32)
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        for module in self.deep:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(1)
        emb = self.embedding(x)
        summed = emb.sum(dim=1)
        fm = 0.5 * (
            summed.square() - emb.square().sum(dim=1)
        ).sum(dim=1)
        deep = self.deep(emb.flatten(1)).squeeze(1)
        return self.bias + linear + fm + deep


def build_torch_model(family, initial_rate):
    if family == "wide":
        return WideModel(total_cardinality, initial_rate)
    if family in ("fm", "fm_recency"):
        return FactorizationMachine(total_cardinality, 16, initial_rate)
    if family == "deepfm":
        return DeepFM(total_cardinality, len(FIELDS), 8, initial_rate)
    raise ValueError(f"Unknown family: {family}")


@torch.inference_mode()
def torch_predict(model, X, batch_size=16384):
    model.eval()
    xt = torch.from_numpy(X)
    out = np.empty(len(X), dtype=np.float32)
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        out[start:end] = model(xt[start:end]).cpu().numpy()
    return out


def fit_torch_model(
    family,
    X,
    y,
    max_epochs,
    initial_rate,
    weights=None,
    valid_data=None,
):
    torch.manual_seed(SEED)
    model = build_torch_model(family, initial_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    xt = torch.from_numpy(X)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    wt = None if weights is None else torch.from_numpy(
        np.asarray(weights, dtype=np.float32)
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    best_primary = -np.inf
    best_epoch = max_epochs
    best_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(X), generator=generator)

        for start in range(0, len(X), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(xt[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, yt[idx], reduction="none"
            )
            if wt is not None:
                loss = (losses * wt[idx]).sum() / wt[idx].sum().clamp_min(1e-8)
            else:
                loss = losses.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        if valid_data is not None:
            Xv, yv, uv = valid_data
            pred = torch_predict(model, Xv)
            score = float(evaluate(uv, yv, pred)["primary"])
            if score > best_primary:
                best_primary = score
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_epoch


def prob_to_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)

X_train = make_torch_matrix(train)
X_valid = make_torch_matrix(valid)

family_predictions = {}
selected_epochs = {}

torch_specs = [
    ("wide", 8, None),
    ("fm", 12, None),
    ("fm_recency", 12, recency_weights(train.date, half_life=7.0)),
    ("deepfm", 6, None),
]

for family, epochs, weights in torch_specs:
    model, best_epoch = fit_torch_model(
        family=family,
        X=X_train,
        y=y_train,
        max_epochs=epochs,
        initial_rate=float(y_train.mean()),
        weights=weights,
        valid_data=(X_valid, y_valid, valid_users),
    )
    family_predictions[family] = torch_predict(model, X_valid).astype(np.float64)
    selected_epochs[family] = int(best_epoch)
    del model
    gc.collect()

# A tree family using exactly the same categorical fields.
X_train_lgb = make_lgb_matrix(train)
X_valid_lgb = make_lgb_matrix(valid)

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.06,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 250,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "num_threads": min(8, os.cpu_count() or 1),
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "verbose": -1,
}

dtrain = lgb.Dataset(
    X_train_lgb,
    label=y_train,
    categorical_feature=list(range(len(FIELDS))),
    free_raw_data=False,
)
dvalid = lgb.Dataset(
    X_valid_lgb,
    label=y_valid,
    categorical_feature=list(range(len(FIELDS))),
    reference=dtrain,
    free_raw_data=False,
)
lgb_model = lgb.train(
    lgb_params,
    dtrain,
    num_boost_round=350,
    valid_sets=[dvalid],
    callbacks=[lgb.early_stopping(35, verbose=False)],
)
selected_epochs["lightgbm"] = int(lgb_model.best_iteration)
family_predictions["lightgbm"] = prob_to_logit(
    lgb_model.predict(X_valid_lgb, num_iteration=lgb_model.best_iteration)
)
del lgb_model, dtrain, dvalid
gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_valid_z = standardized(inc_valid)

candidate_scores = {}
candidate_meta = {}

for family, pred in family_predictions.items():
    pred_z = standardized(pred)

    standalone_name = family
    standalone_metric = evaluate(valid_users, y_valid, pred_z)
    candidate_scores[standalone_name] = float(standalone_metric["primary"])
    candidate_meta[standalone_name] = (family, None, pred_z)

    best_blend_primary = -np.inf
    best_alpha = None
    best_blend = None
    for alpha in np.linspace(0.10, 0.70, 13):
        blend = alpha * inc_valid_z + (1.0 - alpha) * pred_z
        primary = float(evaluate(valid_users, y_valid, blend)["primary"])
        if primary > best_blend_primary:
            best_blend_primary = primary
            best_alpha = float(alpha)
            best_blend = blend

    blend_name = family + "_incumbent_blend"
    candidate_scores[blend_name] = best_blend_primary
    candidate_meta[blend_name] = (family, best_alpha, best_blend)

winner_name = max(candidate_scores, key=candidate_scores.get)
winner_family, winner_alpha, valid_scores = candidate_meta[winner_name]
valid_scores = np.asarray(valid_scores, dtype=np.float64)
metrics = evaluate(valid_users, y_valid, valid_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
print(
    "FINDINGS " +
    json.dumps({
        "winner": winner_name,
        "winner_family": winner_family,
        "incumbent_weight": winner_alpha,
        "selected_epochs": selected_epochs,
    }, sort_keys=True)
)

# Refit the winning recipe on train + validation.
X_combined = np.concatenate([X_train, X_valid], axis=0)
y_combined = np.concatenate([y_train, y_valid.astype(np.float32)], axis=0)

test = load("test")

if winner_family == "lightgbm":
    X_combined_lgb = np.concatenate([X_train_lgb, X_valid_lgb], axis=0)
    X_test_lgb = make_lgb_matrix(test)

    dcombined = lgb.Dataset(
        X_combined_lgb,
        label=y_combined,
        categorical_feature=list(range(len(FIELDS))),
        free_raw_data=True,
    )
    final_model = lgb.train(
        lgb_params,
        dcombined,
        num_boost_round=selected_epochs["lightgbm"],
        valid_sets=None,
    )
    family_test_scores = prob_to_logit(
        final_model.predict(
            X_test_lgb,
            num_iteration=selected_epochs["lightgbm"],
        )
    )
else:
    combined_weights = None
    if winner_family == "fm_recency":
        combined_dates = np.concatenate([
            np.asarray(train.date),
            np.asarray(valid.date),
        ])
        combined_weights = recency_weights(combined_dates, half_life=7.0)

    final_model, _ = fit_torch_model(
        family=winner_family,
        X=X_combined,
        y=y_combined,
        max_epochs=selected_epochs[winner_family],
        initial_rate=float(y_combined.mean()),
        weights=combined_weights,
        valid_data=None,
    )
    X_test = make_torch_matrix(test)
    family_test_scores = torch_predict(final_model, X_test).astype(np.float64)

family_test_scores = standardized(family_test_scores)

if winner_alpha is None:
    test_scores = family_test_scores
else:
    inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
    test_scores = (
        winner_alpha * standardized(inc_test)
        + (1.0 - winner_alpha) * family_test_scores
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

wall_time = time.time() - START_TIME
print(
    "METRICS " +
    json.dumps({
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(wall_time),
    })
)