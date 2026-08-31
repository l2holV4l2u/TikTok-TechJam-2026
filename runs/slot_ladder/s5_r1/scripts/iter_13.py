import os
import time
import json
import random
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 20260831
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(12, os.cpu_count() or 4))

CAT_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "onehot_feat3",
    "onehot_feat8",
    "upload_type",
    "tab",
    "hour",
    "user_active_degree",
    "register_days_bucket",
    "music_type",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    ordered_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_users[1:] != ordered_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    sorted_ranks = (
        np.arange(n, dtype=np.float64) - np.repeat(starts, sizes)
    ) / np.repeat(np.maximum(sizes - 1, 1), sizes)

    result = np.empty(n, dtype=np.float64)
    result[order] = sorted_ranks
    return result


def get_history(split_name):
    result = {}
    for entity in ("video_id", "author_id"):
        block = historical_features(split_name, key=entity)
        for name, values in block.items():
            result[name] = np.asarray(values, dtype=np.float32)
    return result


def raw_dense(split, history, history_names):
    columns = []
    for name in history_names:
        x = np.asarray(history[name], dtype=np.float32)
        columns.append(x)

    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.maximum(x, 0.0)
        columns.append(np.log1p(x))

    return np.column_stack(columns).astype(np.float32)


def standardize_dense(train_dense, other_dense):
    center = np.nanmedian(train_dense, axis=0).astype(np.float32)
    tr = np.where(np.isfinite(train_dense), train_dense, center)
    ot = np.where(np.isfinite(other_dense), other_dense, center)

    q25 = np.percentile(tr, 25, axis=0).astype(np.float32)
    q75 = np.percentile(tr, 75, axis=0).astype(np.float32)
    scale = q75 - q25
    std = np.std(tr, axis=0).astype(np.float32)
    scale = np.where(scale > 1e-5, scale, std)
    scale = np.where(scale > 1e-5, scale, 1.0).astype(np.float32)

    tr = np.clip((tr - center) / scale, -8.0, 8.0).astype(np.float32)
    ot = np.clip((ot - center) / scale, -8.0, 8.0).astype(np.float32)
    return tr, ot, center, scale


def apply_standardization(raw, center, scale):
    x = np.where(np.isfinite(raw), raw, center)
    return np.clip((x - center) / scale, -8.0, 8.0).astype(np.float32)


def categorical_matrix(split):
    return np.column_stack(
        [np.asarray(split.X[f], dtype=np.int32) for f in CAT_FIELDS]
    ).astype(np.int32)


def make_lgb_matrix(dense, cats):
    return np.column_stack([dense, cats.astype(np.float32)]).astype(np.float32)


class AdditiveEmbedding(nn.Module):
    def __init__(self, dense_dim):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(FEATURE_CARDINALITIES[f], 1)
                for f in CAT_FIELDS
            ]
        )
        self.dense = nn.Linear(dense_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        for emb in self.embeddings:
            nn.init.zeros_(emb.weight)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, dense, cats):
        value = self.dense(dense).squeeze(1) + self.bias
        for j, emb in enumerate(self.embeddings):
            value = value + emb(cats[:, j]).squeeze(1)
        return value


class HistoryInteractionMLP(nn.Module):
    def __init__(self, dense_dim, embedding_dim=6):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(FEATURE_CARDINALITIES[f], embedding_dim)
                for f in CAT_FIELDS
            ]
        )
        input_dim = dense_dim + embedding_dim * len(CAT_FIELDS)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.SiLU(),
            nn.LayerNorm(96),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.015)

    def forward(self, dense, cats):
        pieces = [dense]
        pieces.extend(
            emb(cats[:, j])
            for j, emb in enumerate(self.embeddings)
        )
        return self.net(torch.cat(pieces, dim=1)).squeeze(1)


def fit_torch_model(model, dense, cats, labels, weights, lr, epochs):
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=2e-6
    )
    n = len(labels)
    batch_size = 8192
    rng = np.random.default_rng(SEED + int(lr * 100000))

    dense_tensor = torch.from_numpy(dense)
    cats_tensor = torch.from_numpy(cats.astype(np.int64, copy=False))
    labels_tensor = torch.from_numpy(labels.astype(np.float32, copy=False))
    weights_tensor = torch.from_numpy(weights.astype(np.float32, copy=False))

    for _ in range(epochs):
        permutation = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx_np = permutation[start:start + batch_size]
            idx = torch.from_numpy(idx_np)
            logits = model(dense_tensor[idx], cats_tensor[idx])
            loss_raw = nn.functional.binary_cross_entropy_with_logits(
                logits, labels_tensor[idx], reduction="none"
            )
            batch_weights = weights_tensor[idx]
            loss = torch.sum(loss_raw * batch_weights) / torch.sum(
                batch_weights
            ).clamp_min(1e-6)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


@torch.inference_mode()
def predict_torch(model, dense, cats):
    model.eval()
    batch_size = 32768
    output = np.empty(len(dense), dtype=np.float32)
    dense_tensor = torch.from_numpy(dense)
    cats_tensor = torch.from_numpy(cats.astype(np.int64, copy=False))
    for start in range(0, len(dense), batch_size):
        end = min(start + batch_size, len(dense))
        output[start:end] = (
            model(dense_tensor[start:end], cats_tensor[start:end])
            .cpu()
            .numpy()
        )
    return output


train = load("train")
valid = load("valid")
labels = np.asarray(train.y, dtype=np.float32)

train_history = get_history("train")
valid_history = get_history("valid")
history_names = sorted(set(train_history) & set(valid_history))
if not history_names:
    raise RuntimeError("No historical features were returned")

train_raw_dense = raw_dense(train, train_history, history_names)
valid_raw_dense = raw_dense(valid, valid_history, history_names)
train_dense, valid_dense, dense_center, dense_scale = standardize_dense(
    train_raw_dense, valid_raw_dense
)

train_cats = categorical_matrix(train)
valid_cats = categorical_matrix(valid)

max_train_date = int(np.max(train.date))
age_days = (
    max_train_date - np.asarray(train.date, dtype=np.int64)
).astype(np.float32)
recent_weights = np.power(0.5, age_days / 4.0).astype(np.float32)
recent_weights /= np.mean(recent_weights)

# Family 1: boosted trees form explicit thresholded interactions among
# historical feedback channels, numeric quantities, and context categories.
lgb_train_x = make_lgb_matrix(train_dense, train_cats)
lgb_valid_x = make_lgb_matrix(valid_dense, valid_cats)
categorical_indices = list(
    range(train_dense.shape[1], lgb_train_x.shape[1])
)

lgb_dataset = lgb.Dataset(
    lgb_train_x,
    label=labels,
    weight=recent_weights,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
lgb_params = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.055,
    "num_leaves": 39,
    "max_depth": 8,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "min_gain_to_split": 0.002,
    "num_threads": min(12, os.cpu_count() or 4),
    "seed": SEED,
    "feature_fraction_seed": SEED + 1,
    "bagging_seed": SEED + 2,
    "verbose": -1,
}
lgb_model = lgb.train(
    lgb_params,
    lgb_dataset,
    num_boost_round=190,
)
valid_lgb = lgb_model.predict(
    lgb_valid_x, num_iteration=lgb_model.current_iteration()
)

# Family 2: a generalized additive embedding model estimates stable marginal
# effects without introducing high-order identity interactions.
additive_model = AdditiveEmbedding(train_dense.shape[1])
fit_torch_model(
    additive_model,
    train_dense,
    train_cats,
    labels,
    recent_weights,
    lr=0.025,
    epochs=2,
)
valid_additive = predict_torch(
    additive_model, valid_dense, valid_cats
)

# Family 3: an MLP forms nonlinear interactions between history channels and
# categorical embeddings.
mlp_model = HistoryInteractionMLP(train_dense.shape[1])
fit_torch_model(
    mlp_model,
    train_dense,
    train_cats,
    labels,
    recent_weights,
    lr=0.0025,
    epochs=2,
)
valid_mlp = predict_torch(
    mlp_model, valid_dense, valid_cats
)

valid_family_scores = {
    "history_lgb": valid_lgb,
    "history_additive": valid_additive,
    "history_mlp": valid_mlp,
}
valid_family_ranks = {
    name: within_user_rank(valid.user_id, score)
    for name, score in valid_family_scores.items()
}
valid_family_ranks["history_three_family_borda"] = np.mean(
    np.column_stack(
        [
            valid_family_ranks["history_lgb"],
            valid_family_ranks["history_additive"],
            valid_family_ranks["history_mlp"],
        ]
    ),
    axis=1,
)
valid_family_ranks["history_lgb_mlp_borda"] = (
    0.60 * valid_family_ranks["history_lgb"]
    + 0.40 * valid_family_ranks["history_mlp"]
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
incumbent_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
incumbent_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)
if not os.path.exists(incumbent_valid_path):
    raise FileNotFoundError("Trusted incumbent validation scores missing")

incumbent_valid = np.asarray(
    np.load(incumbent_valid_path), dtype=np.float64
)
if len(incumbent_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation score length mismatch")
incumbent_valid_rank = within_user_rank(
    valid.user_id, incumbent_valid
)

candidate_scores = {}
candidate_primary = {}
candidate_recipe = {}
candidate_raw = {}

for family, own_rank in valid_family_ranks.items():
    name = family + "_standalone"
    candidate_scores[name] = own_rank
    candidate_recipe[name] = ("standalone", family, 1.0)
    candidate_raw[name] = own_rank
    candidate_primary[name] = float(
        evaluate(valid.user_id, valid.y, own_rank)["primary"]
    )

    for own_weight in (0.10, 0.20, 0.30, 0.40, 0.50, 0.65):
        name = f"{family}_incumbent_borda_w{own_weight:.2f}"
        score = (
            own_weight * own_rank
            + (1.0 - own_weight) * incumbent_valid_rank
        )
        candidate_scores[name] = score
        candidate_recipe[name] = (
            "incumbent_blend",
            family,
            own_weight,
        )
        candidate_raw[name] = own_rank
        candidate_primary[name] = float(
            evaluate(valid.user_id, valid.y, score)["primary"]
        )

winner = max(candidate_primary, key=candidate_primary.get)
valid_scores = candidate_scores[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

standalone_summary = {
    name: candidate_primary[name]
    for name in candidate_primary
    if name.endswith("_standalone")
}
print(
    "FINDINGS "
    + json.dumps(
        {
            "history_feature_count": len(history_names),
            "best_standalone": max(
                standalone_summary, key=standalone_summary.get
            ),
            "best_candidate": winner,
            "incumbent_check": float(
                evaluate(
                    valid.user_id, valid.y, incumbent_valid
                )["primary"]
            ),
            "recent_weight_min_max": [
                float(recent_weights.min()),
                float(recent_weights.max()),
            ],
        },
        separators=(",", ":"),
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_primary.items()},
        sort_keys=True,
        separators=(",", ":"),
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if candidate_recipe[winner][0] != "standalone":
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[winner], dtype=np.float64),
        )

# Test scoring uses exactly the already fitted models and selected recipe.
test = load("test")
test_history = get_history("test")
missing_history = [k for k in history_names if k not in test_history]
if missing_history:
    raise RuntimeError(
        "Test historical features missing: " + ",".join(missing_history)
    )

test_raw_dense = raw_dense(test, test_history, history_names)
test_dense = apply_standardization(
    test_raw_dense, dense_center, dense_scale
)
test_cats = categorical_matrix(test)
test_lgb_x = make_lgb_matrix(test_dense, test_cats)

test_family_scores = {
    "history_lgb": lgb_model.predict(
        test_lgb_x, num_iteration=lgb_model.current_iteration()
    ),
    "history_additive": predict_torch(
        additive_model, test_dense, test_cats
    ),
    "history_mlp": predict_torch(
        mlp_model, test_dense, test_cats
    ),
}
test_family_ranks = {
    name: within_user_rank(test.user_id, score)
    for name, score in test_family_scores.items()
}
test_family_ranks["history_three_family_borda"] = np.mean(
    np.column_stack(
        [
            test_family_ranks["history_lgb"],
            test_family_ranks["history_additive"],
            test_family_ranks["history_mlp"],
        ]
    ),
    axis=1,
)
test_family_ranks["history_lgb_mlp_borda"] = (
    0.60 * test_family_ranks["history_lgb"]
    + 0.40 * test_family_ranks["history_mlp"]
)

recipe_type, selected_family, own_weight = candidate_recipe[winner]
own_test_rank = test_family_ranks[selected_family]

if recipe_type == "standalone":
    test_scores = own_test_rank
else:
    if not os.path.exists(incumbent_test_path):
        raise FileNotFoundError("Trusted incumbent test scores missing")
    incumbent_test = np.asarray(
        np.load(incumbent_test_path), dtype=np.float64
    )
    if len(incumbent_test) != len(test.user_id):
        raise ValueError("Incumbent test score length mismatch")
    incumbent_test_rank = within_user_rank(
        test.user_id, incumbent_test
    )
    test_scores = (
        own_weight * own_test_rank
        + (1.0 - own_weight) * incumbent_test_rank
    )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(",", ":"),
    )
)