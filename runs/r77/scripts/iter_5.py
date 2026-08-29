import os
import gc
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 271828
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 65536
EPOCHS = 4
BLEND_WEIGHTS = [0.2, 0.4, 0.6, 0.8]

# Use the compact representation that was materially better and much faster
# than exposing all 37 fields.
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

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))
n_fields = len(FIELDS)


def make_neural_features(split):
    x = np.column_stack([
        np.asarray(split.X[f], dtype=np.int64) for f in FIELDS
    ])
    x += offsets[None, :]
    return np.ascontiguousarray(x, dtype=np.int64)


def make_lgb_features(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[f], dtype=np.int32) for f in FIELDS
        ]),
        dtype=np.int32,
    )


class PNN(nn.Module):
    """Product Neural Network: explicit pairwise embedding products."""

    def __init__(self, k=8):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, k)
        self.bias = nn.Parameter(torch.zeros(1))

        pair_i, pair_j = np.triu_indices(n_fields, k=1)
        self.register_buffer(
            "pair_i", torch.from_numpy(pair_i.astype(np.int64))
        )
        self.register_buffer(
            "pair_j", torch.from_numpy(pair_j.astype(np.int64))
        )
        n_pairs = len(pair_i)
        self.mlp = nn.Sequential(
            nn.Linear(n_fields * k + n_pairs, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, x):
        v = self.embedding(x)
        products = (
            v[:, self.pair_i, :] * v[:, self.pair_j, :]
        ).sum(dim=2)
        deep_in = torch.cat([v.flatten(start_dim=1), products], dim=1)
        wide = self.linear(x).sum(dim=1).squeeze(1)
        return self.bias + wide + self.mlp(deep_in).squeeze(1)


class NFM(nn.Module):
    """Neural FM: nonlinear transformation of the bi-interaction vector."""

    def __init__(self, k=16):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, k)
        self.bias = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(k, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, x):
        v = self.embedding(x)
        summed = v.sum(dim=1)
        bi = 0.5 * (summed.square() - v.square().sum(dim=1))
        wide = self.linear(x).sum(dim=1).squeeze(1)
        return self.bias + wide + self.mlp(bi).squeeze(1)


class AutoInt(nn.Module):
    """Self-attentive field interaction model."""

    def __init__(self, k=8, heads=2):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, k)
        self.bias = nn.Parameter(torch.zeros(1))

        self.q1 = nn.Linear(k, k)
        self.k1 = nn.Linear(k, k)
        self.v1 = nn.Linear(k, k)
        self.q2 = nn.Linear(k, k)
        self.k2 = nn.Linear(k, k)
        self.v2 = nn.Linear(k, k)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.heads = heads
        self.head_dim = k // heads

        self.output = nn.Sequential(
            nn.Linear(n_fields * k, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def attention(self, z, q_layer, k_layer, v_layer):
        b, f, k = z.shape
        h = self.heads
        d = self.head_dim

        q = q_layer(z).reshape(b, f, h, d).transpose(1, 2)
        key = k_layer(z).reshape(b, f, h, d).transpose(1, 2)
        value = v_layer(z).reshape(b, f, h, d).transpose(1, 2)

        weights = torch.softmax(
            torch.matmul(q, key.transpose(-1, -2)) / np.sqrt(d),
            dim=-1,
        )
        out = torch.matmul(weights, value)
        return out.transpose(1, 2).contiguous().reshape(b, f, k)

    def forward(self, x):
        base = self.embedding(x)
        z = self.norm1(
            base + self.attention(base, self.q1, self.k1, self.v1)
        )
        z = self.norm2(
            z + self.attention(z, self.q2, self.k2, self.v2)
        )
        wide = self.linear(x).sum(dim=1).squeeze(1)
        return self.bias + wide + self.output(
            z.flatten(start_dim=1)
        ).squeeze(1)


def make_model(name):
    torch.manual_seed(SEED)
    if name == "pnn":
        return PNN()
    if name == "nfm":
        return NFM()
    if name == "autoint":
        return AutoInt()
    raise ValueError(name)


@torch.no_grad()
def predict_neural(model, x_np):
    model.eval()
    pred = np.empty(x_np.shape[0], dtype=np.float64)
    for begin in range(0, x_np.shape[0], PRED_BATCH_SIZE):
        end = min(begin + PRED_BATCH_SIZE, x_np.shape[0])
        xb = torch.from_numpy(x_np[begin:end])
        pred[begin:end] = (
            model(xb).cpu().numpy().astype(np.float64)
        )
    return pred


def train_epoch(model, optimizer, x, y, generator):
    model.train()
    order = torch.randperm(x.shape[0], generator=generator)
    for begin in range(0, x.shape[0], BATCH_SIZE):
        idx = order[begin:begin + BATCH_SIZE]
        xb = x[idx]
        yb = y[idx]

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, yb
        )
        loss.backward()
        optimizer.step()


def fit_neural_select(name, x_train_np, y_train, valid, x_valid_np):
    model = make_model(name)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x = torch.from_numpy(x_train_np)
    y = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(SEED)

    best_primary = -np.inf
    best_epoch = 1
    best_scores = None

    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, optimizer, x, y, generator)
        scores = predict_neural(model, x_valid_np)
        primary = float(
            evaluate(valid.user_id, valid.y, scores)["primary"]
        )
        if primary > best_primary:
            best_primary = primary
            best_epoch = epoch
            best_scores = scores.copy()

    del model, optimizer, x, y
    gc.collect()
    return best_scores, best_epoch


def fit_neural_fixed(name, x_np, y_np, epochs):
    model = make_model(name)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    generator = torch.Generator()
    generator.manual_seed(SEED)

    for _ in range(epochs):
        train_epoch(model, optimizer, x, y, generator)

    del optimizer, x, y
    gc.collect()
    return model


def group_sizes(sorted_users):
    if len(sorted_users) == 0:
        return np.empty(0, dtype=np.int32)
    boundaries = np.flatnonzero(
        sorted_users[1:] != sorted_users[:-1]
    ) + 1
    return np.diff(
        np.concatenate(([0], boundaries, [len(sorted_users)]))
    ).astype(np.int32)


def fit_lgb_binary(x_train, y_train, x_valid, y_valid):
    dtrain = lgb.Dataset(
        x_train,
        label=np.asarray(y_train, dtype=np.float32),
        categorical_feature=list(range(n_fields)),
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        x_valid,
        label=np.asarray(y_valid, dtype=np.float32),
        categorical_feature=list(range(n_fields)),
        reference=dtrain,
        free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.06,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "seed": SEED,
        "num_threads": min(8, os.cpu_count() or 1),
        "verbose": -1,
    }
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=220,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(25, verbose=False)],
    )
    rounds = int(model.best_iteration or 220)
    scores = model.predict(
        x_valid, num_iteration=rounds, raw_score=True
    )
    return np.asarray(scores, dtype=np.float64), rounds


def fit_lgb_binary_fixed(x, y, rounds):
    dtrain = lgb.Dataset(
        x,
        label=np.asarray(y, dtype=np.float32),
        categorical_feature=list(range(n_fields)),
        free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "metric": "None",
        "learning_rate": 0.06,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "seed": SEED,
        "num_threads": min(8, os.cpu_count() or 1),
        "verbose": -1,
    }
    return lgb.train(params, dtrain, num_boost_round=rounds)


def fit_lgb_rank(
    x_train, y_train, users_train,
    x_valid, y_valid, users_valid,
):
    train_order = np.argsort(users_train, kind="stable")
    valid_order = np.argsort(users_valid, kind="stable")

    xtr = x_train[train_order]
    ytr = np.asarray(y_train, dtype=np.int8)[train_order]
    utr = np.asarray(users_train)[train_order]

    xva = x_valid[valid_order]
    yva = np.asarray(y_valid, dtype=np.int8)[valid_order]
    uva = np.asarray(users_valid)[valid_order]

    dtrain = lgb.Dataset(
        xtr,
        label=ytr,
        group=group_sizes(utr),
        categorical_feature=list(range(n_fields)),
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        xva,
        label=yva,
        group=group_sizes(uva),
        categorical_feature=list(range(n_fields)),
        reference=dtrain,
        free_raw_data=False,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "lambdarank_truncation_level": 10,
        "seed": SEED,
        "num_threads": min(8, os.cpu_count() or 1),
        "verbose": -1,
    }
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=180,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(25, verbose=False)],
    )
    rounds = int(model.best_iteration or 180)
    sorted_scores = model.predict(xva, num_iteration=rounds)

    scores = np.empty(len(valid_order), dtype=np.float64)
    scores[valid_order] = sorted_scores
    return scores, rounds


def fit_lgb_rank_fixed(x, y, users, rounds):
    order = np.argsort(users, kind="stable")
    xs = x[order]
    ys = np.asarray(y, dtype=np.int8)[order]
    us = np.asarray(users)[order]

    dtrain = lgb.Dataset(
        xs,
        label=ys,
        group=group_sizes(us),
        categorical_feature=list(range(n_fields)),
        free_raw_data=False,
    )
    params = {
        "objective": "lambdarank",
        "metric": "None",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "max_bin": 127,
        "lambdarank_truncation_level": 10,
        "seed": SEED,
        "num_threads": min(8, os.cpu_count() or 1),
        "verbose": -1,
    }
    return lgb.train(params, dtrain, num_boost_round=rounds)


train = load("train")
valid = load("valid")

x_train_nn = make_neural_features(train)
x_valid_nn = make_neural_features(valid)
x_train_lgb = make_lgb_features(train)
x_valid_lgb = make_lgb_features(valid)

artifacts = os.environ["RUN_ARTIFACTS"]
inc_valid = np.asarray(
    np.load(os.path.join(artifacts, "incumbent_valid_scores.npy")),
    dtype=np.float64,
)
inc_test_path = os.path.join(
    artifacts, "incumbent_test_scores.npy"
)

family_scores = {}
family_recipe = {}

# Three structurally different neural interaction families.
for family in ["pnn", "nfm", "autoint"]:
    scores, epoch = fit_neural_select(
        family, x_train_nn, train.y, valid, x_valid_nn
    )
    family_scores[family] = scores
    family_recipe[family] = int(epoch)

# Pointwise boosted trees.
binary_scores, binary_rounds = fit_lgb_binary(
    x_train_lgb, train.y, x_valid_lgb, valid.y
)
family_scores["lightgbm_binary"] = binary_scores
family_recipe["lightgbm_binary"] = int(binary_rounds)

gc.collect()

# Listwise/query-aware boosted trees.
rank_scores, rank_rounds = fit_lgb_rank(
    x_train_lgb,
    train.y,
    np.asarray(train.user_id),
    x_valid_lgb,
    valid.y,
    np.asarray(valid.user_id),
)
family_scores["lightgbm_lambdarank"] = rank_scores
family_recipe["lightgbm_lambdarank"] = int(rank_rounds)

candidate_primary = {}
inc_primary = float(
    evaluate(valid.user_id, valid.y, inc_valid)["primary"]
)
candidate_primary["trusted_incumbent"] = inc_primary

best_name = "trusted_incumbent"
best_family = None
best_alpha = 0.0
best_primary = inc_primary
best_valid_scores = inc_valid.copy()

for family, scores in family_scores.items():
    standalone = float(
        evaluate(valid.user_id, valid.y, scores)["primary"]
    )
    candidate_primary[family] = standalone

    if standalone > best_primary:
        best_primary = standalone
        best_name = family
        best_family = family
        best_alpha = 1.0
        best_valid_scores = scores.copy()

    for alpha in BLEND_WEIGHTS:
        blended = alpha * scores + (1.0 - alpha) * inc_valid
        name = f"{family}_blend_{alpha:.1f}"
        primary = float(
            evaluate(valid.user_id, valid.y, blended)["primary"]
        )
        candidate_primary[name] = primary
        if primary > best_primary:
            best_primary = primary
            best_name = name
            best_family = family
            best_alpha = float(alpha)
            best_valid_scores = blended.copy()

metrics = evaluate(valid.user_id, valid.y, best_valid_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )

# Validation selection is complete. Refit only the selected family on
# train+validation, then apply the exact selected blend to test.
test = load("test")

if best_family is None:
    test_scores = np.asarray(
        np.load(inc_test_path), dtype=np.float64
    )
else:
    y_combined = np.concatenate([
        np.asarray(train.y, dtype=np.float32),
        np.asarray(valid.y, dtype=np.float32),
    ])
    users_combined = np.concatenate([
        np.asarray(train.user_id),
        np.asarray(valid.user_id),
    ])

    if best_family in {"pnn", "nfm", "autoint"}:
        x_combined = np.concatenate(
            [x_train_nn, x_valid_nn], axis=0
        )
        x_test = make_neural_features(test)
        refit = fit_neural_fixed(
            best_family,
            x_combined,
            y_combined,
            family_recipe[best_family],
        )
        new_test_scores = predict_neural(refit, x_test)

    elif best_family == "lightgbm_binary":
        x_combined = np.concatenate(
            [x_train_lgb, x_valid_lgb], axis=0
        )
        x_test = make_lgb_features(test)
        refit = fit_lgb_binary_fixed(
            x_combined,
            y_combined,
            family_recipe[best_family],
        )
        new_test_scores = np.asarray(
            refit.predict(
                x_test,
                num_iteration=family_recipe[best_family],
                raw_score=True,
            ),
            dtype=np.float64,
        )

    elif best_family == "lightgbm_lambdarank":
        x_combined = np.concatenate(
            [x_train_lgb, x_valid_lgb], axis=0
        )
        x_test = make_lgb_features(test)
        refit = fit_lgb_rank_fixed(
            x_combined,
            y_combined,
            users_combined,
            family_recipe[best_family],
        )
        new_test_scores = np.asarray(
            refit.predict(
                x_test,
                num_iteration=family_recipe[best_family],
            ),
            dtype=np.float64,
        )
    else:
        raise ValueError(best_family)

    if best_alpha < 1.0:
        inc_test = np.asarray(
            np.load(inc_test_path), dtype=np.float64
        )
        test_scores = (
            best_alpha * new_test_scores
            + (1.0 - best_alpha) * inc_test
        )
    else:
        test_scores = new_test_scores

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

print("CANDIDATES " + json.dumps(
    {k: float(v) for k, v in candidate_primary.items()},
    sort_keys=True,
))
print("FINDINGS " + json.dumps({
    "selected": best_name,
    "selected_family": best_family,
    "selected_new_family_weight": float(best_alpha),
    "family_recipe": {
        k: int(v) for k, v in family_recipe.items()
    },
    "fields": FIELDS,
}, sort_keys=True))

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))