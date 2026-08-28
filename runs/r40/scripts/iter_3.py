import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2024
FM_FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
FM_RANK = 16
FM_LR = 0.001
FM_BATCH = 4096
FM_EPOCHS = 12

LGB_ROUNDS = 350
CHECKPOINTS = [100, 150, 200, 250, 300, 350]
BLEND_WEIGHTS = [0.0, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def group_order_and_sizes(user_ids):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[order]
    if len(sorted_users) == 0:
        return order, np.empty(0, dtype=np.int32)
    boundaries = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1], True]
    )
    sizes = np.diff(boundaries).astype(np.int32)
    return order, sizes


def causal_sequence_features(split):
    n = len(split.y)
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    rows = np.arange(n, dtype=np.int64)

    chronological = np.lexsort((rows, times, users))
    sorted_users = users[chronological]
    sorted_times = times[chronological]

    new_user = np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    starts = np.where(new_user, np.arange(n, dtype=np.int64), 0)
    starts = np.maximum.accumulate(starts)

    position_sorted = np.arange(n, dtype=np.int64) - starts
    first_time = sorted_times[starts]
    elapsed_hours_sorted = (
        sorted_times.astype(np.float64) - first_time.astype(np.float64)
    ) / 3600000.0

    position = np.empty(n, dtype=np.float32)
    elapsed_hours = np.empty(n, dtype=np.float32)
    position[chronological] = position_sorted.astype(np.float32)
    elapsed_hours[chronological] = np.clip(
        elapsed_hours_sorted, 0.0, 24.0 * 30.0
    ).astype(np.float32)

    return (
        position,
        np.log1p(position).astype(np.float32),
        elapsed_hours,
        np.log1p(elapsed_hours).astype(np.float32),
    )


ALL_CAT_FIELDS = sorted(list(FEATURE_CARDINALITIES.keys()))


def make_lgb_matrix(split):
    n = len(split.y)
    columns = []

    for name in ALL_CAT_FIELDS:
        columns.append(np.asarray(split.X[name], dtype=np.float32))

    for name in sorted(split.num.keys()):
        raw = np.asarray(split.num[name], dtype=np.float32)
        columns.append(raw)

    date = np.asarray(split.date, dtype=np.int32)
    date_day = (date % 100).astype(np.float32)
    columns.append(date_day)

    pos, log_pos, elapsed, log_elapsed = causal_sequence_features(split)
    columns.extend([pos, log_pos, elapsed, log_elapsed])

    matrix = np.empty((n, len(columns)), dtype=np.float32)
    for j, col in enumerate(columns):
        matrix[:, j] = col
    return matrix


fm_cards = [int(FEATURE_CARDINALITIES[name]) for name in FM_FIELDS]
fm_offsets = np.cumsum([0] + fm_cards[:-1], dtype=np.int64)
fm_total_cardinality = int(sum(fm_cards))


def make_fm_matrix(split):
    return torch.from_numpy(
        np.stack(
            [
                np.asarray(split.X[name], dtype=np.int64) + offset
                for name, offset in zip(FM_FIELDS, fm_offsets)
            ],
            axis=1,
        )
    )


class FactorizationMachine(nn.Module):
    def __init__(self, cardinality, rank):
        super().__init__()
        self.embedding = nn.Embedding(cardinality, rank + 1, sparse=True)
        self.bias = nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            self.embedding.weight[:, 0].zero_()
            self.embedding.weight[:, 1:].normal_(0.0, 0.01)

    def forward(self, x):
        embedded = self.embedding(x)
        linear = embedded[:, :, 0].sum(dim=1)
        factors = embedded[:, :, 1:]
        factor_sum = factors.sum(dim=1)
        interaction = 0.5 * (
            factor_sum.square().sum(dim=1)
            - factors.square().sum(dim=(1, 2))
        )
        return self.bias + linear + interaction


@torch.no_grad()
def predict_fm(model, x, batch_size=65536):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        result[start:end] = (
            model(x[start:end]).cpu().numpy().astype(np.float64)
        )
    return result


def standardized(values):
    values = np.asarray(values, dtype=np.float64)
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return (values - float(np.mean(values))) / scale


train = load("train")
valid = load("valid")

# First retain the incumbent collaborative FM signal.
x_train_fm = make_fm_matrix(train)
x_valid_fm = make_fm_matrix(valid)
y_train_t = torch.from_numpy(np.asarray(train.y, dtype=np.float32))

fm = FactorizationMachine(fm_total_cardinality, FM_RANK)
fm_sparse_optimizer = torch.optim.SparseAdam(
    [fm.embedding.weight], lr=FM_LR
)
fm_dense_optimizer = torch.optim.Adam([fm.bias], lr=FM_LR)
criterion = nn.BCEWithLogitsLoss()

generator = torch.Generator()
generator.manual_seed(SEED)
n_train = len(train.y)

best_fm_primary = -np.inf
best_fm_state = None
best_fm_metrics = None
best_fm_valid_scores = None

for epoch in range(FM_EPOCHS):
    fm.train()
    permutation = torch.randperm(n_train, generator=generator)
    last_loss = 0.0

    for start in range(0, n_train, FM_BATCH):
        idx = permutation[start:min(start + FM_BATCH, n_train)]
        fm_sparse_optimizer.zero_grad(set_to_none=True)
        fm_dense_optimizer.zero_grad(set_to_none=True)

        logits = fm(x_train_fm[idx])
        loss = criterion(logits, y_train_t[idx])
        loss.backward()
        fm_sparse_optimizer.step()
        fm_dense_optimizer.step()
        last_loss = float(loss.detach())

    valid_fm_scores = predict_fm(fm, x_valid_fm)
    fm_metrics = evaluate(valid.user_id, valid.y, valid_fm_scores)
    fm_primary = float(fm_metrics["primary"])

    if fm_primary > best_fm_primary:
        best_fm_primary = fm_primary
        best_fm_metrics = {
            "primary": fm_primary,
            "gauc": float(fm_metrics["gauc"]),
            "ndcg@5": float(fm_metrics["ndcg@5"]),
        }
        best_fm_valid_scores = valid_fm_scores.copy()
        best_fm_state = {
            key: value.detach().clone()
            for key, value in fm.state_dict().items()
        }

    print(
        "fm_epoch=%d loss=%.6f primary=%.6f"
        % (epoch + 1, last_loss, fm_primary),
        flush=True,
    )

fm.load_state_dict(best_fm_state)

# Train LambdaMART with each user represented as a query.
train_matrix = make_lgb_matrix(train)
valid_matrix = make_lgb_matrix(valid)

train_order, train_groups = group_order_and_sizes(train.user_id)
valid_order, valid_groups = group_order_and_sizes(valid.user_id)

train_matrix = train_matrix[train_order]
train_labels_sorted = np.asarray(train.y, dtype=np.float32)[train_order]
valid_matrix_sorted = valid_matrix[valid_order]
valid_labels_sorted = np.asarray(valid.y, dtype=np.float32)[valid_order]

categorical_indices = list(range(len(ALL_CAT_FIELDS)))

dtrain = lgb.Dataset(
    train_matrix,
    label=train_labels_sorted,
    group=train_groups,
    categorical_feature=categorical_indices,
    free_raw_data=True,
)
dvalid = lgb.Dataset(
    valid_matrix_sorted,
    label=valid_labels_sorted,
    group=valid_groups,
    categorical_feature=categorical_indices,
    reference=dtrain,
    free_raw_data=True,
)

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_at": [5],
    "label_gain": [0, 1],
    "lambdarank_truncation_level": 10,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 150,
    "min_sum_hessian_in_leaf": 1e-3,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 1.0,
    "max_bin": 127,
    "max_cat_threshold": 64,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "min_data_per_group": 100,
    "verbosity": -1,
    "verbose": -1,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "data_random_seed": SEED,
    "num_threads": max(1, min(8, os.cpu_count() or 1)),
    "force_col_wise": True,
}

ranker = lgb.train(
    params,
    dtrain,
    num_boost_round=LGB_ROUNDS,
    valid_sets=[dvalid],
    valid_names=["valid"],
    callbacks=[lgb.log_evaluation(period=50)],
)

del dtrain, dvalid, train_matrix, train_labels_sorted
del valid_matrix_sorted, valid_labels_sorted

fm_z = standardized(best_fm_valid_scores)
candidates = {"fm": float(best_fm_primary)}

best_primary = best_fm_primary
best_metrics = best_fm_metrics
best_round = None
best_blend_weight = None
best_valid_scores = best_fm_valid_scores

for num_iteration in CHECKPOINTS:
    sorted_predictions = ranker.predict(
        valid_matrix,
        num_iteration=num_iteration,
    )
    # valid_matrix is in original row order because the sorted copy was
    # separately materialized above.
    tree_scores = np.asarray(sorted_predictions, dtype=np.float64)
    tree_z = standardized(tree_scores)

    tree_metrics = evaluate(valid.user_id, valid.y, tree_scores)
    candidates["ranker_%d" % num_iteration] = float(
        tree_metrics["primary"]
    )

    for weight in BLEND_WEIGHTS:
        blended = tree_z + weight * fm_z
        metrics = evaluate(valid.user_id, valid.y, blended)
        primary = float(metrics["primary"])
        name = "ranker_%d_fm_%.1f" % (num_iteration, weight)
        candidates[name] = primary

        if primary > best_primary:
            best_primary = primary
            best_round = num_iteration
            best_blend_weight = weight
            best_valid_scores = blended.copy()
            best_metrics = {
                "primary": primary,
                "gauc": float(metrics["gauc"]),
                "ndcg@5": float(metrics["ndcg@5"]),
            }

print("CANDIDATES " + json.dumps(candidates, sort_keys=True))
print(
    "FINDINGS selected_round=%s selected_fm_weight=%s fm_primary=%.6f"
    % (
        str(best_round),
        str(best_blend_weight),
        best_fm_primary,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    test = load("test")
    x_test_fm = make_fm_matrix(test)
    test_fm_scores = predict_fm(fm, x_test_fm)

    if best_round is None:
        test_scores = test_fm_scores
    else:
        test_matrix = make_lgb_matrix(test)
        test_tree_scores = ranker.predict(
            test_matrix,
            num_iteration=best_round,
        )
        test_scores = (
            standardized(test_tree_scores)
            + best_blend_weight * standardized(test_fm_scores)
        )

    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

final = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": 0.0,
}
print("METRICS " + json.dumps(final))