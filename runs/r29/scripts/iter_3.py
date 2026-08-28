import json
import math
import os
import random

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
CONTEXT_FIELDS = [
    "tab",
    "hour",
    "user_active_degree",
    "onehot_feat1",
    "onehot_feat3",
    "onehot_feat8",
    "upload_type",
    "music_type",
    "tag",
    "duration_bucket",
]
EMBED_DIM = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 2048
PRED_BATCH_SIZE = 32768
EPOCHS = 10
TE_SMOOTH = 20.0
PAIR_SMOOTH = 12.0


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

cardinalities = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cardinalities[:-1], dtype=np.int64)
total_cardinality = int(sum(cardinalities))


def make_fm_features(split):
    x = np.stack(
        [np.asarray(split.X[f], dtype=np.int64) for f in FIELDS],
        axis=1,
    )
    x += offsets[None, :]
    return torch.from_numpy(np.ascontiguousarray(x))


class FactorizationMachine(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim + 1)
        self.bias = nn.Parameter(torch.zeros(()))

        with torch.no_grad():
            self.embedding.weight[:, :embedding_dim].normal_(
                mean=0.0, std=0.01
            )
            self.embedding.weight[:, embedding_dim].zero_()

    def forward(self, x):
        parameters = self.embedding(x)
        factors = parameters[:, :, :EMBED_DIM]
        linear = parameters[:, :, EMBED_DIM].sum(dim=1)

        summed = factors.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - factors.square().sum(dim=1)
        ).sum(dim=1)

        return self.bias + linear + interaction


def predict_fm(model, x):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, x.shape[0], PRED_BATCH_SIZE):
            end = min(start + PRED_BATCH_SIZE, x.shape[0])
            result[start:end] = model(x[start:end]).cpu().numpy()
    return result


def fit_single_target_encoder(ids, labels, cardinality):
    ids = np.asarray(ids, dtype=np.int64)
    counts = np.bincount(ids, minlength=cardinality).astype(np.float64)
    sums = np.bincount(
        ids, weights=labels, minlength=cardinality
    ).astype(np.float64)
    return counts, sums


def apply_single_target_encoder(
    ids, counts, sums, global_mean, smooth, labels=None
):
    ids = np.asarray(ids, dtype=np.int64)
    if labels is None:
        rate = (
            sums[ids] + smooth * global_mean
        ) / (counts[ids] + smooth)
        count = counts[ids]
    else:
        labels = np.asarray(labels, dtype=np.float64)
        rate = (
            sums[ids] - labels + smooth * global_mean
        ) / (counts[ids] - 1.0 + smooth)
        count = np.maximum(counts[ids] - 1.0, 0.0)

    return (
        np.asarray(rate, dtype=np.float32),
        np.log1p(count).astype(np.float32),
    )


def fit_pair_target_encoder(left, right, right_cardinality, labels):
    keys = (
        np.asarray(left, dtype=np.int64) * np.int64(right_cardinality)
        + np.asarray(right, dtype=np.int64)
    )
    unique_keys, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True
    )
    sums = np.bincount(
        inverse, weights=labels, minlength=unique_keys.size
    ).astype(np.float64)
    return unique_keys, counts.astype(np.float64), sums, inverse


def apply_pair_target_encoder(
    left,
    right,
    right_cardinality,
    unique_keys,
    counts,
    sums,
    global_mean,
    smooth,
    labels=None,
    train_inverse=None,
):
    if train_inverse is not None:
        labels = np.asarray(labels, dtype=np.float64)
        rate = (
            sums[train_inverse] - labels + smooth * global_mean
        ) / (counts[train_inverse] - 1.0 + smooth)
        count = np.maximum(counts[train_inverse] - 1.0, 0.0)
        return (
            np.asarray(rate, dtype=np.float32),
            np.log1p(count).astype(np.float32),
        )

    keys = (
        np.asarray(left, dtype=np.int64) * np.int64(right_cardinality)
        + np.asarray(right, dtype=np.int64)
    )
    positions = np.searchsorted(unique_keys, keys)
    matched = positions < unique_keys.size
    safe_positions = np.minimum(positions, unique_keys.size - 1)
    matched &= unique_keys[safe_positions] == keys

    rate = np.full(keys.shape[0], global_mean, dtype=np.float64)
    count = np.zeros(keys.shape[0], dtype=np.float64)
    if np.any(matched):
        p = positions[matched]
        rate[matched] = (
            sums[p] + smooth * global_mean
        ) / (counts[p] + smooth)
        count[matched] = counts[p]

    return rate.astype(np.float32), np.log1p(count).astype(np.float32)


def make_stack_features(
    split,
    fm_scores,
    video_encoder,
    author_encoder,
    pair_encoder,
    global_mean,
    training_labels=None,
):
    columns = [np.asarray(fm_scores, dtype=np.float32)]

    for field in CONTEXT_FIELDS:
        columns.append(np.asarray(split.X[field], dtype=np.float32))

    video_counts, video_sums = video_encoder
    video_rate, video_count = apply_single_target_encoder(
        split.X["video_id"],
        video_counts,
        video_sums,
        global_mean,
        TE_SMOOTH,
        labels=training_labels,
    )
    columns.extend([video_rate, video_count])

    author_counts, author_sums = author_encoder
    author_rate, author_count = apply_single_target_encoder(
        split.X["author_id"],
        author_counts,
        author_sums,
        global_mean,
        TE_SMOOTH,
        labels=training_labels,
    )
    columns.extend([author_rate, author_count])

    pair_keys, pair_counts, pair_sums, train_inverse = pair_encoder
    if training_labels is None:
        pair_rate, pair_count = apply_pair_target_encoder(
            split.X["user_id"],
            split.X["author_id"],
            int(FEATURE_CARDINALITIES["author_id"]),
            pair_keys,
            pair_counts,
            pair_sums,
            global_mean,
            PAIR_SMOOTH,
        )
    else:
        pair_rate, pair_count = apply_pair_target_encoder(
            split.X["user_id"],
            split.X["author_id"],
            int(FEATURE_CARDINALITIES["author_id"]),
            pair_keys,
            pair_counts,
            pair_sums,
            global_mean,
            PAIR_SMOOTH,
            labels=training_labels,
            train_inverse=train_inverse,
        )
    columns.extend([pair_rate, pair_count])

    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


train = load("train")
valid = load("valid")

x_train = make_fm_features(train)
x_valid = make_fm_features(valid)
train_y_np = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y)
valid_users = np.asarray(valid.user_id)
y_train = torch.from_numpy(train_y_np)

model = FactorizationMachine(total_cardinality, EMBED_DIM)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    foreach=True,
)

generator = torch.Generator()
generator.manual_seed(SEED)

best_primary = -math.inf
best_metrics = None
best_state = None
best_valid_fm = None
n_train = x_train.shape[0]

for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(n_train, generator=generator)

    loss_sum = 0.0
    seen = 0

    for start in range(0, n_train, BATCH_SIZE):
        indices = permutation[start:start + BATCH_SIZE]
        xb = x_train[indices]
        yb = y_train[indices]

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        optimizer.step()

        batch_n = indices.numel()
        loss_sum += float(loss.detach()) * batch_n
        seen += batch_n

    valid_fm = predict_fm(model, x_valid)
    metrics = evaluate(valid_users, valid_y, valid_fm)

    print(
        f"epoch={epoch + 1} "
        f"loss={loss_sum / seen:.6f} "
        f"primary={float(metrics['primary']):.6f} "
        f"gauc={float(metrics['gauc']):.6f} "
        f"ndcg@5={float(metrics['ndcg@5']):.6f}",
        flush=True,
    )

    if float(metrics["primary"]) > best_primary:
        best_primary = float(metrics["primary"])
        best_metrics = {k: float(v) for k, v in metrics.items()}
        best_valid_fm = valid_fm.copy()
        best_state = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

model.load_state_dict(best_state)
train_fm = predict_fm(model, x_train)

global_mean = float(np.mean(train_y_np))

video_encoder = fit_single_target_encoder(
    train.X["video_id"],
    train_y_np,
    int(FEATURE_CARDINALITIES["video_id"]),
)
author_encoder = fit_single_target_encoder(
    train.X["author_id"],
    train_y_np,
    int(FEATURE_CARDINALITIES["author_id"]),
)
pair_encoder = fit_pair_target_encoder(
    train.X["user_id"],
    train.X["author_id"],
    int(FEATURE_CARDINALITIES["author_id"]),
    train_y_np,
)

stack_train = make_stack_features(
    train,
    train_fm,
    video_encoder,
    author_encoder,
    pair_encoder,
    global_mean,
    training_labels=train_y_np,
)
stack_valid = make_stack_features(
    valid,
    best_valid_fm,
    video_encoder,
    author_encoder,
    pair_encoder,
    global_mean,
)

categorical_indices = list(range(1, 1 + len(CONTEXT_FIELDS)))

dtrain = lgb.Dataset(
    stack_train,
    label=train_y_np,
    init_score=train_fm.astype(np.float64),
    categorical_feature=categorical_indices,
    free_raw_data=True,
)
dvalid = lgb.Dataset(
    stack_valid,
    label=valid_y,
    init_score=best_valid_fm.astype(np.float64),
    categorical_feature=categorical_indices,
    reference=dtrain,
    free_raw_data=True,
)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.04,
    "num_leaves": 31,
    "max_depth": 8,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 4.0,
    "max_bin": 127,
    "cat_smooth": 30.0,
    "cat_l2": 15.0,
    "boost_from_average": False,
    "num_threads": min(8, os.cpu_count() or 1),
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "verbose": -1,
}

booster = lgb.train(
    params,
    dtrain,
    num_boost_round=220,
    valid_sets=[dvalid],
    callbacks=[lgb.early_stopping(30, verbose=False)],
)

valid_residual = booster.predict(
    stack_valid,
    num_iteration=booster.best_iteration,
    raw_score=True,
)

best_alpha = 0.0
best_scores = best_valid_fm.copy()
best_metrics = evaluate(valid_users, valid_y, best_scores)

for alpha in [0.25, 0.5, 0.75, 1.0]:
    candidate_scores = best_valid_fm + alpha * valid_residual
    candidate_metrics = evaluate(valid_users, valid_y, candidate_scores)
    print(
        f"residual_alpha={alpha:.2f} "
        f"primary={float(candidate_metrics['primary']):.6f} "
        f"gauc={float(candidate_metrics['gauc']):.6f} "
        f"ndcg@5={float(candidate_metrics['ndcg@5']):.6f}",
        flush=True,
    )
    if float(candidate_metrics["primary"]) > float(best_metrics["primary"]):
        best_alpha = alpha
        best_scores = candidate_scores.copy()
        best_metrics = candidate_metrics

best_metrics = {k: float(v) for k, v in best_metrics.items()}
print(
    f"selected_alpha={best_alpha:.2f} "
    f"best_iteration={booster.best_iteration}",
    flush=True,
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    test = load("test")
    x_test = make_fm_features(test)
    test_fm = predict_fm(model, x_test)

    if best_alpha == 0.0:
        test_scores = test_fm
    else:
        stack_test = make_stack_features(
            test,
            test_fm,
            video_encoder,
            author_encoder,
            pair_encoder,
            global_mean,
        )
        test_residual = booster.predict(
            stack_test,
            num_iteration=booster.best_iteration,
            raw_score=True,
        )
        test_scores = test_fm + best_alpha * test_residual

    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

final_metrics = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": 0.0,
}
print("METRICS " + json.dumps(final_metrics))