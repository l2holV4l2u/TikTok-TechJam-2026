import os
import time
import json
import random
import copy
import gc

import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 2026
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
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 65536
MAX_EPOCHS = 7
K = 16


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_all(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))


def embedding_matrix(split):
    n = len(split.user_id)
    x = np.empty((n, len(FIELDS)), dtype=np.int64)
    for j, f in enumerate(FIELDS):
        x[:, j] = np.asarray(split.X[f], dtype=np.int64) + offsets[j]
    return x


def combined_embedding_matrix(a, b):
    na = len(a.user_id)
    nb = len(b.user_id)
    x = np.empty((na + nb, len(FIELDS)), dtype=np.int64)
    for j, f in enumerate(FIELDS):
        x[:na, j] = np.asarray(a.X[f], dtype=np.int64) + offsets[j]
        x[na:, j] = np.asarray(b.X[f], dtype=np.int64) + offsets[j]
    return x


def tree_matrix(split):
    n = len(split.user_id)
    x = np.empty((n, len(FIELDS) + len(NUM_FIELDS)), dtype=np.float32)
    for j, f in enumerate(FIELDS):
        x[:, j] = np.asarray(split.X[f], dtype=np.float32)
    for k, f in enumerate(NUM_FIELDS):
        v = np.asarray(split.num[f], dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        x[:, len(FIELDS) + k] = np.log1p(np.maximum(v, 0.0))
    return x


def combined_tree_matrix(a, b):
    xa = tree_matrix(a)
    xb = tree_matrix(b)
    return np.concatenate([xa, xb], axis=0)


class ExpandedFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, K)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - v.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


class NFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, K)
        self.mlp = nn.Sequential(
            nn.Linear(K, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        v = self.embedding(x)
        summed = v.sum(dim=1)
        bi = 0.5 * (summed.square() - v.square().sum(dim=1))
        deep = self.mlp(bi).squeeze(-1)
        return self.bias + linear + deep


@torch.no_grad()
def torch_predict(model, x):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float32)
    xt = torch.from_numpy(x)
    for start in range(0, x.shape[0], PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, x.shape[0])
        result[start:end] = model(xt[start:end]).cpu().numpy()
    return result


def fit_torch_family(model_class, x_fit, y_fit, x_eval, seed, epochs=None):
    seed_all(seed)
    model = model_class()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(x_fit)
    yt = torch.from_numpy(np.asarray(y_fit, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(seed + 91)

    if epochs is not None:
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(len(xt), generator=generator)
            for start in range(0, len(xt), BATCH_SIZE):
                idx = perm[start:start + BATCH_SIZE]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(xt[idx]), yt[idx])
                loss.backward()
                optimizer.step()
        return model, torch_predict(model, x_eval), epochs

    best_primary = -np.inf
    best_state = None
    best_scores = None
    best_epoch = 1

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(xt), generator=generator)
        for start in range(0, len(xt), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()

        scores = torch_predict(model, x_eval)
        m = evaluate(valid.user_id, y_valid, scores)
        if float(m["primary"]) > best_primary:
            best_primary = float(m["primary"])
            best_epoch = epoch
            best_scores = scores.copy()
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    return model, best_scores, best_epoch


def empirical_bayes_scores(fit_split, fit_y, score_split):
    fit_y = np.asarray(fit_y, dtype=np.float64)
    global_rate = float(np.clip(fit_y.mean(), 1e-5, 1.0 - 1e-5))
    global_logit = np.log(global_rate / (1.0 - global_rate))
    result = np.full(len(score_split.user_id), global_logit, dtype=np.float64)

    stat_fields = [
        "video_id",
        "author_id",
        "tag",
        "duration_bucket",
        "upload_type",
        "music_type",
        "hour",
        "tab",
    ]
    priors = {
        "video_id": 35.0,
        "author_id": 50.0,
        "tag": 120.0,
        "duration_bucket": 180.0,
        "upload_type": 180.0,
        "music_type": 180.0,
        "hour": 180.0,
        "tab": 180.0,
    }

    for f in stat_fields:
        ids = np.asarray(fit_split.X[f], dtype=np.int64)
        score_ids = np.asarray(score_split.X[f], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[f])
        counts = np.bincount(ids, minlength=card).astype(np.float64)
        positives = np.bincount(
            ids, weights=fit_y, minlength=card
        ).astype(np.float64)
        prior = priors[f]
        rate = (positives + prior * global_rate) / (counts + prior)
        rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
        effect = np.log(rate / (1.0 - rate)) - global_logit
        reliability = counts / (counts + prior)
        result += effect[score_ids] * reliability[score_ids]

    return result.astype(np.float32)


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return (x - mean) / std


shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

x_train = embedding_matrix(train)
x_valid = embedding_matrix(valid)

family_scores = {}
family_recipes = {}

fm_model, fm_valid, fm_epoch = fit_torch_family(
    ExpandedFM, x_train, y_train, x_valid, SEED + 1
)
family_scores["expanded_fm"] = fm_valid
family_recipes["expanded_fm"] = {"kind": "torch", "epoch": fm_epoch}

del fm_model
gc.collect()

nfm_model, nfm_valid, nfm_epoch = fit_torch_family(
    NFM, x_train, y_train, x_valid, SEED + 2
)
family_scores["nfm"] = nfm_valid
family_recipes["nfm"] = {"kind": "torch", "epoch": nfm_epoch}

del nfm_model
gc.collect()

eb_valid = empirical_bayes_scores(train, y_train, valid)
family_scores["empirical_bayes"] = eb_valid
family_recipes["empirical_bayes"] = {"kind": "eb"}

x_train_tree = tree_matrix(train)
x_valid_tree = tree_matrix(valid)
categorical_indices = list(range(len(FIELDS)))

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.07,
    "num_leaves": 63,
    "min_data_in_leaf": 150,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "max_cat_threshold": 32,
    "cat_smooth": 20.0,
    "max_bin": 127,
    "num_threads": max(1, min(8, os.cpu_count() or 1)),
    "seed": SEED + 3,
    "feature_fraction_seed": SEED + 4,
    "bagging_seed": SEED + 5,
    "verbose": -1,
}

lgb_train = lgb.Dataset(
    x_train_tree,
    label=y_train,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=220,
)
lgb_valid = lgb_model.predict(
    x_valid_tree, num_iteration=lgb_model.current_iteration()
).astype(np.float32)
family_scores["lightgbm_binary"] = lgb_valid
family_recipes["lightgbm_binary"] = {"kind": "lgb", "rounds": 220}

del lgb_model, lgb_train
gc.collect()

candidate_log = {}
best_primary = -np.inf
best_name = None
best_family = None
best_alpha = None
best_valid_scores = None
best_metrics = None

inc_z = zscore(inc_valid)
inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)
candidate_log["incumbent"] = float(inc_metrics["primary"])

alphas = np.linspace(0.0, 1.0, 11)
for family, raw_scores in family_scores.items():
    standalone_metrics = evaluate(valid.user_id, y_valid, raw_scores)
    candidate_log[family] = float(standalone_metrics["primary"])
    candidate_z = zscore(raw_scores)

    local_best = -np.inf
    local_alpha = 1.0
    local_scores = None
    local_metrics = None

    for alpha in alphas:
        blended = alpha * candidate_z + (1.0 - alpha) * inc_z
        metrics_alpha = evaluate(valid.user_id, y_valid, blended)
        primary_alpha = float(metrics_alpha["primary"])
        if primary_alpha > local_best:
            local_best = primary_alpha
            local_alpha = float(alpha)
            local_scores = blended.copy()
            local_metrics = metrics_alpha

    blend_name = family + "_blend"
    candidate_log[blend_name] = float(local_best)

    if local_best > best_primary:
        best_primary = local_best
        best_name = blend_name
        best_family = family
        best_alpha = local_alpha
        best_valid_scores = local_scores
        best_metrics = local_metrics

if float(inc_metrics["primary"]) > best_primary:
    best_primary = float(inc_metrics["primary"])
    best_name = "incumbent"
    best_family = "incumbent"
    best_alpha = 0.0
    best_valid_scores = inc_valid.copy()
    best_metrics = inc_metrics

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )

test = load("test")
inc_test = np.load(inc_test_path).astype(np.float64)

if best_family == "incumbent":
    test_scores = inc_test
else:
    y_combined = np.concatenate([
        np.asarray(train.y, dtype=np.float32),
        np.asarray(valid.y, dtype=np.float32),
    ])

    if family_recipes[best_family]["kind"] == "torch":
        x_combined = combined_embedding_matrix(train, valid)
        x_test = embedding_matrix(test)
        epoch_count = int(family_recipes[best_family]["epoch"])
        model_class = ExpandedFM if best_family == "expanded_fm" else NFM
        combined_model, raw_test, _ = fit_torch_family(
            model_class,
            x_combined,
            y_combined,
            x_test,
            SEED + (1 if best_family == "expanded_fm" else 2),
            epochs=epoch_count,
        )
        del combined_model, x_combined, x_test
        gc.collect()

    elif family_recipes[best_family]["kind"] == "eb":
        class CombinedSplit:
            pass

        combined = CombinedSplit()
        combined.X = {}
        for f in FIELDS:
            combined.X[f] = np.concatenate([
                np.asarray(train.X[f], dtype=np.int64),
                np.asarray(valid.X[f], dtype=np.int64),
            ])
        combined.user_id = np.concatenate([
            np.asarray(train.user_id),
            np.asarray(valid.user_id),
        ])
        raw_test = empirical_bayes_scores(combined, y_combined, test)

    else:
        x_combined_tree = combined_tree_matrix(train, valid)
        x_test_tree = tree_matrix(test)
        combined_dataset = lgb.Dataset(
            x_combined_tree,
            label=y_combined,
            categorical_feature=categorical_indices,
            free_raw_data=False,
        )
        combined_lgb = lgb.train(
            lgb_params,
            combined_dataset,
            num_boost_round=int(family_recipes[best_family]["rounds"]),
        )
        raw_test = combined_lgb.predict(
            x_test_tree,
            num_iteration=combined_lgb.current_iteration(),
        )
        del combined_lgb, combined_dataset, x_combined_tree, x_test_tree
        gc.collect()

    test_scores = (
        best_alpha * zscore(raw_test)
        + (1.0 - best_alpha) * zscore(inc_test)
    )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

candidate_log["selected_alpha"] = float(best_alpha)
print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print("FINDINGS " + json.dumps({
    "winner": best_name,
    "family": best_family,
    "blend_candidate_weight": float(best_alpha),
    "fm_epoch": int(fm_epoch),
    "nfm_epoch": int(nfm_epoch),
}, sort_keys=True))

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))