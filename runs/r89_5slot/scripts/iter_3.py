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
SEED = 7319
FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour"
]
EMBED_DIM = 8
BATCH_SIZE = 4096
PRED_BATCH_SIZE = 32768
TORCH_EPOCHS = 7
HALF_LIFE_DAYS = 10.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

offsets = []
total_cardinality = 0
for name in FIELDS:
    offsets.append(total_cardinality)
    total_cardinality += int(FEATURE_CARDINALITIES[name])
OFFSETS = np.asarray(offsets, dtype=np.int64)


def make_matrix(split, offset=True):
    x = np.empty((len(split.user_id), len(FIELDS)), dtype=np.int64)
    for j, name in enumerate(FIELDS):
        x[:, j] = np.asarray(split.X[name], dtype=np.int64)
        if offset:
            x[:, j] += OFFSETS[j]
    return x


def recency_weights(dates):
    day = np.asarray(dates, dtype=np.int32) % 100
    age = int(day.max()) - day
    w = np.exp(-np.log(2.0) * age.astype(np.float32) / HALF_LIFE_DAYS)
    return w.astype(np.float32)


class WideModel(nn.Module):
    def __init__(self, positive_rate):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        p = float(np.clip(positive_rate, 1e-5, 1 - 1e-5))
        self.bias = nn.Parameter(
            torch.tensor(np.log(p / (1.0 - p)), dtype=torch.float32)
        )
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.bias + self.linear(x).sum(dim=1).squeeze(-1)


class ProductNeuralNetwork(nn.Module):
    def __init__(self, positive_rate):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)

        n_fields = len(FIELDS)
        n_pairs = n_fields * (n_fields - 1) // 2
        input_dim = n_fields * EMBED_DIM + n_pairs

        self.register_buffer(
            "pair_i",
            torch.tensor(
                [i for i in range(n_fields) for j in range(i + 1, n_fields)],
                dtype=torch.long
            )
        )
        self.register_buffer(
            "pair_j",
            torch.tensor(
                [j for i in range(n_fields) for j in range(i + 1, n_fields)],
                dtype=torch.long
            )
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        p = float(np.clip(positive_rate, 1e-5, 1 - 1e-5))
        self.bias = nn.Parameter(
            torch.tensor(np.log(p / (1.0 - p)), dtype=torch.float32)
        )

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        emb = self.embedding(x)
        pair_products = (
            emb[:, self.pair_i, :] * emb[:, self.pair_j, :]
        ).sum(dim=2)
        deep_input = torch.cat(
            [emb.reshape(emb.shape[0], -1), pair_products], dim=1
        )
        wide = self.linear(x).sum(dim=1).squeeze(-1)
        return self.bias + wide + self.mlp(deep_input).squeeze(-1)


def make_torch_model(kind, positive_rate):
    if kind == "wide":
        return WideModel(positive_rate)
    if kind == "pnn":
        return ProductNeuralNetwork(positive_rate)
    raise ValueError(kind)


@torch.no_grad()
def torch_predict(model, x_np):
    model.eval()
    result = np.empty(x_np.shape[0], dtype=np.float32)
    for start in range(0, x_np.shape[0], PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, x_np.shape[0])
        xb = torch.from_numpy(x_np[start:end])
        result[start:end] = model(xb).cpu().numpy()
    return result


def fit_torch_select(kind, train_x, train_y, train_w,
                     valid_x, valid_y, valid_users):
    torch.manual_seed(SEED + (0 if kind == "wide" else 101))
    model = make_torch_model(kind, float(train_y.mean()))
    lr = 0.002 if kind == "wide" else 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = torch.from_numpy(train_x)
    y = torch.from_numpy(train_y.astype(np.float32, copy=False))
    w = torch.from_numpy(train_w.astype(np.float32, copy=False))
    generator = torch.Generator()
    generator.manual_seed(SEED + (0 if kind == "wide" else 101))

    best_score = -np.inf
    best_epoch = 1
    best_predictions = None
    stale = 0

    for epoch in range(1, TORCH_EPOCHS + 1):
        model.train()
        order = torch.randperm(x.shape[0], generator=generator)
        for start in range(0, x.shape[0], BATCH_SIZE):
            idx = order[start:min(start + BATCH_SIZE, x.shape[0])]
            logits = model(x[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, y[idx], reduction="none"
            )
            loss = (losses * w[idx]).sum() / w[idx].sum().clamp_min(1e-6)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        predictions = torch_predict(model, valid_x)
        metrics = evaluate(valid_users, valid_y, predictions)
        primary = float(metrics["primary"])
        print(
            "FINDINGS family=%s epoch=%d primary=%.6f gauc=%.6f ndcg5=%.6f"
            % (
                kind, epoch, primary,
                metrics["gauc"], metrics["ndcg@5"]
            ),
            flush=True
        )

        if primary > best_score:
            if primary > best_score + 0.00015:
                stale = 0
            else:
                stale += 1
            best_score = primary
            best_epoch = epoch
            best_predictions = predictions.copy()
        else:
            stale += 1

        if epoch >= 4 and stale >= 2:
            break

    del model, x, y, w
    gc.collect()
    return best_epoch, best_predictions


def fit_torch_fixed(kind, x_np, y_np, weights, epochs):
    torch.manual_seed(SEED + (0 if kind == "wide" else 101))
    model = make_torch_model(kind, float(y_np.mean()))
    lr = 0.002 if kind == "wide" else 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np.astype(np.float32, copy=False))
    w = torch.from_numpy(weights.astype(np.float32, copy=False))
    generator = torch.Generator()
    generator.manual_seed(SEED + (0 if kind == "wide" else 101))

    for _ in range(epochs):
        model.train()
        order = torch.randperm(x.shape[0], generator=generator)
        for start in range(0, x.shape[0], BATCH_SIZE):
            idx = order[start:min(start + BATCH_SIZE, x.shape[0])]
            logits = model(x[idx])
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, y[idx], reduction="none"
            )
            loss = (losses * w[idx]).sum() / w[idx].sum().clamp_min(1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    sd = float(values.std())
    if sd < 1e-12:
        sd = 1.0
    return (values - float(values.mean())) / sd


def within_user_rank(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores)
    order = np.lexsort((scores, users))
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    lengths = ends - starts
    positions = np.arange(len(order)) - np.repeat(starts, lengths)

    ranked_sorted = (
        positions.astype(np.float64) + 0.5
    ) / np.repeat(lengths, lengths).astype(np.float64)
    ranked = np.empty(len(order), dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def transformed(mode, users, scores):
    if mode == "z":
        return zscore(scores)
    if mode == "rank":
        return within_user_rank(users, scores)
    raise ValueError(mode)


train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)
train_weights = recency_weights(train.date)

train_x_offset = make_matrix(train, offset=True)
valid_x_offset = make_matrix(valid, offset=True)

family_predictions = {}
family_recipe = {}

for family in ["wide", "pnn"]:
    epoch, prediction = fit_torch_select(
        family, train_x_offset, train_y, train_weights,
        valid_x_offset, valid_y, valid_users
    )
    family_predictions[family] = prediction
    family_recipe[family] = {"epoch": int(epoch)}
    print(
        "FINDINGS selected_%s_epoch=%d" % (family, epoch),
        flush=True
    )

# A structurally different tree family, using the same categorical fields.
train_x_tree = make_matrix(train, offset=False).astype(np.int32, copy=False)
valid_x_tree = make_matrix(valid, offset=False).astype(np.int32, copy=False)

lgb_params = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.06,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 250,
    "feature_fraction": 0.90,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "verbosity": -1,
    "num_threads": min(8, os.cpu_count() or 1),
    "seed": SEED
}
lgb_train = lgb.Dataset(
    train_x_tree,
    label=train_y,
    weight=train_weights,
    categorical_feature=list(range(len(FIELDS))),
    free_raw_data=False
)
tree_model = lgb.train(lgb_params, lgb_train, num_boost_round=300)

best_tree_primary = -np.inf
best_tree_round = 100
best_tree_prediction = None
for nround in [100, 180, 240, 300]:
    prediction = tree_model.predict(
        valid_x_tree, num_iteration=nround
    ).astype(np.float32)
    metrics = evaluate(valid_users, valid_y, prediction)
    primary = float(metrics["primary"])
    print(
        "FINDINGS family=lightgbm rounds=%d primary=%.6f gauc=%.6f ndcg5=%.6f"
        % (
            nround, primary, metrics["gauc"], metrics["ndcg@5"]
        ),
        flush=True
    )
    if primary > best_tree_primary:
        best_tree_primary = primary
        best_tree_round = nround
        best_tree_prediction = prediction.copy()

family_predictions["lightgbm"] = best_tree_prediction
family_recipe["lightgbm"] = {"rounds": int(best_tree_round)}
del tree_model, lgb_train
gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.load(inc_valid_path).astype(np.float64)

candidate_log = {}
best_primary = -np.inf
winner_family = None
winner_alpha = None
winner_mode = None
winner_scores = None
winner_raw = None

alphas = [0.25, 0.40, 0.55, 0.70, 0.85, 1.0]
for family, raw_prediction in family_predictions.items():
    standalone_metrics = evaluate(valid_users, valid_y, raw_prediction)
    candidate_log[family + "_standalone"] = float(
        standalone_metrics["primary"]
    )

    for mode in ["z", "rank"]:
        own_t = transformed(mode, valid_users, raw_prediction)
        inc_t = transformed(mode, valid_users, inc_valid)

        for alpha in alphas:
            blended = alpha * own_t + (1.0 - alpha) * inc_t
            metrics = evaluate(valid_users, valid_y, blended)
            primary = float(metrics["primary"])
            name = "%s_%s_a%.2f" % (family, mode, alpha)
            candidate_log[name] = primary

            if primary > best_primary:
                best_primary = primary
                winner_family = family
                winner_alpha = float(alpha)
                winner_mode = mode
                winner_scores = blended.copy()
                winner_raw = raw_prediction.copy()

print(
    "FINDINGS winner_family=%s mode=%s own_weight=%.2f recipe=%s"
    % (
        winner_family, winner_mode, winner_alpha,
        json.dumps(family_recipe[winner_family], sort_keys=True)
    ),
    flush=True
)
print(
    "CANDIDATES " + json.dumps(candidate_log, sort_keys=True),
    flush=True
)

valid_metrics = evaluate(valid_users, valid_y, winner_scores)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(winner_scores, dtype=np.float64)
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(winner_raw, dtype=np.float64)
    )

# Refit only the selected family using the identical recipe on train+validation.
combined_y = np.concatenate(
    [train_y, valid_y.astype(np.float32, copy=False)]
)
combined_dates = np.concatenate(
    [np.asarray(train.date), np.asarray(valid.date)]
)
combined_weights = recency_weights(combined_dates)

test = load("test")
test_users = np.asarray(test.user_id)

if winner_family in ("wide", "pnn"):
    combined_x = np.concatenate(
        [train_x_offset, valid_x_offset], axis=0
    )
    test_x = make_matrix(test, offset=True)
    final_model = fit_torch_fixed(
        winner_family,
        combined_x,
        combined_y,
        combined_weights,
        family_recipe[winner_family]["epoch"]
    )
    own_test = torch_predict(final_model, test_x)
    del final_model, combined_x, test_x
else:
    combined_x = np.concatenate(
        [train_x_tree, valid_x_tree], axis=0
    )
    test_x = make_matrix(test, offset=False).astype(np.int32, copy=False)
    combined_set = lgb.Dataset(
        combined_x,
        label=combined_y,
        weight=combined_weights,
        categorical_feature=list(range(len(FIELDS))),
        free_raw_data=False
    )
    final_model = lgb.train(
        lgb_params,
        combined_set,
        num_boost_round=family_recipe[winner_family]["rounds"]
    )
    own_test = final_model.predict(
        test_x,
        num_iteration=family_recipe[winner_family]["rounds"]
    ).astype(np.float32)
    del final_model, combined_set, combined_x, test_x

inc_test = np.load(inc_test_path).astype(np.float64)
own_test_t = transformed(winner_mode, test_users, own_test)
inc_test_t = transformed(winner_mode, test_users, inc_test)
test_scores = (
    winner_alpha * own_test_t +
    (1.0 - winner_alpha) * inc_test_t
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

elapsed = float(time.time() - START)
result = {
    "primary": float(valid_metrics["primary"]),
    "gauc": float(valid_metrics["gauc"]),
    "ndcg@5": float(valid_metrics["ndcg@5"]),
    "gpu_seconds": elapsed
}
print("METRICS " + json.dumps(result, separators=(", ", ": ")))