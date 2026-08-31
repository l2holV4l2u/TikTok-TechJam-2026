import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 314159
THREADS = max(1, min(16, os.cpu_count() or 1))
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "video_type",
    "hour",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
]
NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

AUTOIN_T_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "tag",
    "upload_type",
    "music_type",
    "video_type",
    "hour",
    "onehot_feat3",
    "onehot_feat8",
]

RECENCY_HALF_LIVES = {
    "lambdarank_h2": 2.0,
    "lambdarank_h4": 4.0,
    "lambdarank_h8": 8.0,
}


def recency_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.int32)
    latest = int(dates.max())
    w = np.power(
        2.0,
        (dates.astype(np.float64) - latest) / float(half_life),
    )
    w /= np.mean(w)
    return w.astype(np.float32)


def make_lgb_features(split_name, split):
    blocks = []
    for name in CAT_FIELDS:
        blocks.append(np.asarray(split.X[name], dtype=np.float32)[:, None])

    for name in NUM_FIELDS:
        x = np.asarray(split.num[name], dtype=np.float32)
        x = np.where(np.isfinite(x), np.maximum(x, 0.0), np.nan)
        x = np.log1p(x)
        blocks.append(x[:, None])

    for key in ("video_id", "author_id"):
        histories = historical_features(split_name, key=key)
        for name in sorted(histories):
            x = np.asarray(histories[name], dtype=np.float32)
            x = np.where(np.isfinite(x), x, np.nan)
            blocks.append(x[:, None])

    return np.ascontiguousarray(np.concatenate(blocks, axis=1), dtype=np.float32)


def user_sort_and_groups(user_ids):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    row = np.arange(user_ids.size, dtype=np.int64)
    order = np.lexsort((row, user_ids))
    sorted_users = user_ids[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1], True]
    )
    groups = np.diff(boundaries).astype(np.int32)
    return order, groups


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    start_positions = np.flatnonzero(starts)
    group_index = np.cumsum(starts) - 1
    positions = np.arange(n, dtype=np.int64) - start_positions[group_index]
    sizes = np.diff(np.r_[start_positions, n])
    denominators = np.maximum(sizes[group_index] - 1, 1)

    ranked_sorted = positions.astype(np.float64) / denominators
    ranked_sorted[sizes[group_index] == 1] = 0.5
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


OFFSETS = []
total_cardinality = 0
for field in AUTOIN_T_FIELDS:
    OFFSETS.append(total_cardinality)
    total_cardinality += int(FEATURE_CARDINALITIES[field])
OFFSETS = np.asarray(OFFSETS, dtype=np.int64)


def make_autoint_features(split):
    return np.ascontiguousarray(
        np.column_stack([
            np.asarray(split.X[field], dtype=np.int64) + OFFSETS[j]
            for j, field in enumerate(AUTOIN_T_FIELDS)
        ]),
        dtype=np.int64,
    )


class AutoInt(nn.Module):
    def __init__(self, embedding_dim=12, heads=3):
        super().__init__()
        self.embedding = nn.Embedding(
            total_cardinality, embedding_dim, sparse=True
        )
        self.linear = nn.Embedding(total_cardinality, 1, sparse=True)
        self.attention1 = nn.MultiheadAttention(
            embedding_dim, heads, batch_first=True
        )
        self.attention2 = nn.MultiheadAttention(
            embedding_dim, heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        dim = len(AUTOIN_T_FIELDS) * embedding_dim
        self.output = nn.Sequential(
            nn.Linear(dim, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        e = self.embedding(x)
        a1, _ = self.attention1(e, e, e, need_weights=False)
        z1 = self.norm1(e + a1)
        a2, _ = self.attention2(z1, z1, z1, need_weights=False)
        z2 = self.norm2(z1 + a2)
        deep = self.output(z2.flatten(1)).squeeze(1)
        wide = self.linear(x).squeeze(-1).sum(dim=1)
        return self.bias + wide + deep

    def sparse_parameters(self):
        return [self.embedding.weight, self.linear.weight]

    def dense_parameters(self):
        return (
            [self.bias]
            + list(self.attention1.parameters())
            + list(self.attention2.parameters())
            + list(self.norm1.parameters())
            + list(self.norm2.parameters())
            + list(self.output.parameters())
        )


def train_autoint(model, x_np, y_np, weights_np):
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))
    weights = torch.from_numpy(np.asarray(weights_np, dtype=np.float32))

    sparse_optimizer = torch.optim.SparseAdam(
        model.sparse_parameters(), lr=0.0012
    )
    dense_optimizer = torch.optim.Adam(
        model.dense_parameters(), lr=0.0012, weight_decay=1e-6
    )

    n = x.shape[0]
    generator = torch.Generator()
    generator.manual_seed(SEED + 77)
    batch_size = 4096

    model.train()
    for _ in range(2):
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = x[idx]
            yb = y[idx]
            wb = weights[idx]

            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = torch.sum(losses * wb) / torch.sum(wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.dense_parameters(), 5.0)
            sparse_optimizer.step()
            dense_optimizer.step()

    return model


def predict_autoint(model, x_np):
    model.eval()
    result = np.empty(x_np.shape[0], dtype=np.float64)
    batch_size = 32768
    with torch.inference_mode():
        for start in range(0, x_np.shape[0], batch_size):
            end = min(start + batch_size, x_np.shape[0])
            xb = torch.from_numpy(x_np[start:end])
            result[start:end] = (
                model(xb).cpu().numpy().astype(np.float64)
            )
    return result


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

x_train_lgb = make_lgb_features("train", train)
x_valid_lgb = make_lgb_features("valid", valid)

train_order, train_groups = user_sort_and_groups(train.user_id)
x_train_sorted = np.ascontiguousarray(x_train_lgb[train_order])
y_train_sorted = np.ascontiguousarray(y_train[train_order])

categorical_indices = list(range(len(CAT_FIELDS)))
lgb_models = {}
valid_raw = {}

lgb_params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "label_gain": [0, 1],
    "lambdarank_truncation_level": 10,
    "learning_rate": 0.055,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 350,
    "max_bin": 127,
    "feature_fraction": 0.88,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l1": 0.15,
    "lambda_l2": 2.5,
    "min_gain_to_split": 0.0,
    "num_threads": THREADS,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "verbose": -1,
}

for model_index, (name, half_life) in enumerate(RECENCY_HALF_LIVES.items()):
    weights = recency_weights(train.date, half_life)[train_order]
    dataset = lgb.Dataset(
        x_train_sorted,
        label=y_train_sorted,
        weight=weights,
        group=train_groups,
        categorical_feature=categorical_indices,
        free_raw_data=False,
    )
    params = dict(lgb_params)
    params["seed"] = SEED + model_index
    params["feature_fraction_seed"] = SEED + model_index
    params["bagging_seed"] = SEED + model_index

    booster = lgb.train(
        params,
        dataset,
        num_boost_round=190,
    )
    lgb_models[name] = booster
    valid_raw[name] = booster.predict(
        x_valid_lgb, num_iteration=190
    )
    del dataset, weights
    gc.collect()

# Structurally distinct self-attentive field-interaction model.
x_train_auto = make_autoint_features(train)
x_valid_auto = make_autoint_features(valid)
autoint_weights = recency_weights(train.date, 4.0)
autoint_model = AutoInt()
autoint_model = train_autoint(
    autoint_model, x_train_auto, y_train, autoint_weights
)
valid_raw["autoint_h4"] = predict_autoint(
    autoint_model, x_valid_auto
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

candidate_values = {}
candidate_arrays = {}
candidate_sources = {}
candidate_alphas = {}

for name, raw_scores in valid_raw.items():
    standalone = evaluate(valid.user_id, y_valid, raw_scores)
    candidate_values[name] = float(standalone["primary"])
    candidate_arrays[name] = raw_scores
    candidate_sources[name] = name
    candidate_alphas[name] = 1.0

    own_rank = within_user_rank(valid.user_id, raw_scores)
    for alpha in (0.15, 0.25, 0.35, 0.50, 0.65, 0.80):
        blend_name = f"{name}_blend_{alpha:.2f}"
        blend = alpha * own_rank + (1.0 - alpha) * inc_valid_rank
        blend_metrics = evaluate(valid.user_id, y_valid, blend)
        candidate_values[blend_name] = float(blend_metrics["primary"])
        candidate_arrays[blend_name] = blend
        candidate_sources[blend_name] = name
        candidate_alphas[blend_name] = alpha

# Also compare rank aggregation across the direct-ranking and attention families.
rank_components = {
    name: within_user_rank(valid.user_id, scores)
    for name, scores in valid_raw.items()
}
best_lambda_name = max(
    RECENCY_HALF_LIVES,
    key=lambda n: candidate_values[n]
)
family_ensemble_rank = (
    0.65 * rank_components[best_lambda_name]
    + 0.35 * rank_components["autoint_h4"]
)
for alpha in (0.25, 0.40, 0.55, 0.70):
    name = f"rank_family_ensemble_blend_{alpha:.2f}"
    blend = alpha * family_ensemble_rank + (1.0 - alpha) * inc_valid_rank
    m = evaluate(valid.user_id, y_valid, blend)
    candidate_values[name] = float(m["primary"])
    candidate_arrays[name] = blend
    candidate_sources[name] = "rank_family_ensemble"
    candidate_alphas[name] = alpha

winner = max(candidate_values, key=candidate_values.get)
valid_scores = candidate_arrays[winner]
metrics = evaluate(valid.user_id, y_valid, valid_scores)

print("CANDIDATES " + json.dumps(candidate_values, sort_keys=True))

# Build test scores using exactly the selected model(s) and blend weight.
test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
inc_test_rank = within_user_rank(test.user_id, inc_test)

source = candidate_sources[winner]
alpha = candidate_alphas[winner]

x_test_lgb = None
x_test_auto = None

def get_test_raw(model_name):
    global x_test_lgb, x_test_auto
    if model_name in lgb_models:
        if x_test_lgb is None:
            x_test_lgb = make_lgb_features("test", test)
        return lgb_models[model_name].predict(
            x_test_lgb, num_iteration=190
        )
    if model_name == "autoint_h4":
        if x_test_auto is None:
            x_test_auto = make_autoint_features(test)
        return predict_autoint(autoint_model, x_test_auto)
    raise KeyError(model_name)


if source == "rank_family_ensemble":
    lambda_test_raw = get_test_raw(best_lambda_name)
    autoint_test_raw = get_test_raw("autoint_h4")
    own_test_rank = (
        0.65 * within_user_rank(test.user_id, lambda_test_raw)
        + 0.35 * within_user_rank(test.user_id, autoint_test_raw)
    )
    test_scores = alpha * own_test_rank + (1.0 - alpha) * inc_test_rank
    own_valid_raw = family_ensemble_rank
else:
    own_test_raw = get_test_raw(source)
    own_valid_raw = valid_raw[source]
    if winner == source:
        test_scores = own_test_raw
    else:
        own_test_rank = within_user_rank(test.user_id, own_test_raw)
        test_scores = alpha * own_test_rank + (1.0 - alpha) * inc_test_rank

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
    if winner != source or source == "rank_family_ensemble":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(own_valid_raw, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)