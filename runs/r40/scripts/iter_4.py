import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


SEED = 2025
FM_FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
HISTORY_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat7",
    "duration_bucket",
    "upload_type",
    "music_type",
    "video_type",
    "tab",
]
HISTORY_ALPHA = {
    "video_id": 8.0,
    "author_id": 12.0,
    "tag": 20.0,
    "onehot_feat3": 15.0,
    "onehot_feat8": 18.0,
    "onehot_feat7": 20.0,
    "duration_bucket": 30.0,
    "upload_type": 30.0,
    "music_type": 30.0,
    "video_type": 30.0,
    "tab": 25.0,
}
RAW_CAT_FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "music_type",
    "video_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "hour",
    "user_active_degree",
]

FM_RANK = 16
FM_LR = 0.001
FM_BATCH = 4096
FM_EPOCHS = 12

LGB_ROUNDS = 400
LGB_CHECKPOINTS = [100, 200, 300, 400]
BLEND_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def metric_dict(metrics):
    return {
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
    }


def fit_standardizer(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return mean, scale


def apply_standardizer(values, mean, scale):
    return (np.asarray(values, dtype=np.float64) - mean) / scale


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
        result[start:end] = model(x[start:end]).cpu().numpy()
    return result


class HistoryEncoder:
    def __init__(self, train):
        self.global_rate = float(np.mean(train.y))
        self.models = {}
        self.train_features = {}

        users = np.asarray(train.user_id, dtype=np.int64)
        labels = np.asarray(train.y, dtype=np.float64)

        for field in HISTORY_FIELDS:
            values = np.asarray(train.X[field], dtype=np.int64)
            cardinality = int(FEATURE_CARDINALITIES[field])
            alpha = float(HISTORY_ALPHA[field])

            category_count = np.bincount(
                values, minlength=cardinality
            ).astype(np.float64)
            category_sum = np.bincount(
                values, weights=labels, minlength=cardinality
            ).astype(np.float64)
            category_rate = (
                category_sum + 20.0 * self.global_rate
            ) / (category_count + 20.0)

            keys = users * np.int64(cardinality) + values
            unique_keys, inverse, counts = np.unique(
                keys, return_inverse=True, return_counts=True
            )
            sums = np.bincount(
                inverse, weights=labels, minlength=len(unique_keys)
            ).astype(np.float64)
            counts = counts.astype(np.float64)

            loo_count = counts[inverse] - 1.0
            loo_sum = sums[inverse] - labels
            priors = category_rate[values]
            loo_rate = (loo_sum + alpha * priors) / (loo_count + alpha)
            reliability = loo_count / (loo_count + alpha)
            residual = loo_rate - priors

            self.train_features[field] = (
                loo_rate.astype(np.float32),
                np.log1p(loo_count).astype(np.float32),
                residual.astype(np.float32),
                reliability.astype(np.float32),
            )
            self.models[field] = {
                "cardinality": cardinality,
                "alpha": alpha,
                "keys": unique_keys,
                "counts": counts,
                "sums": sums,
                "category_rate": category_rate,
            }

    def transform(self, split):
        users = np.asarray(split.user_id, dtype=np.int64)
        result = {}

        for field in HISTORY_FIELDS:
            model = self.models[field]
            values = np.asarray(split.X[field], dtype=np.int64)
            keys = users * np.int64(model["cardinality"]) + values

            unique_keys = model["keys"]
            positions = np.searchsorted(unique_keys, keys)
            clipped = np.minimum(positions, len(unique_keys) - 1)
            found = (
                (positions < len(unique_keys))
                & (unique_keys[clipped] == keys)
            )

            counts = np.zeros(len(keys), dtype=np.float64)
            sums = np.zeros(len(keys), dtype=np.float64)
            counts[found] = model["counts"][positions[found]]
            sums[found] = model["sums"][positions[found]]

            priors = model["category_rate"][values]
            alpha = model["alpha"]
            rates = (sums + alpha * priors) / (counts + alpha)
            reliability = counts / (counts + alpha)
            residual = rates - priors

            result[field] = (
                rates.astype(np.float32),
                np.log1p(counts).astype(np.float32),
                residual.astype(np.float32),
                reliability.astype(np.float32),
            )

        return result


def history_summary(history_features):
    residuals = []
    for field in HISTORY_FIELDS:
        _, _, residual, reliability = history_features[field]
        residuals.append(
            residual.astype(np.float64)
            * (0.25 + 0.75 * reliability.astype(np.float64))
        )
    return np.mean(np.stack(residuals, axis=1), axis=1)


NUM_FIELDS = sorted([
    "collect_cnt",
    "comment_cnt",
    "complete_play_cnt",
    "counts",
    "download_cnt",
    "duration_ms",
    "follow_cnt",
    "like_cnt",
    "long_time_play_cnt",
    "play_cnt",
    "play_duration",
    "play_progress",
    "play_user_num",
    "share_cnt",
    "short_time_play_cnt",
    "show_cnt",
    "show_user_num",
    "valid_play_cnt",
])


def make_lgb_matrix(split, history_features):
    n = len(split.user_id)
    columns = []

    for field in HISTORY_FIELDS:
        rate, log_count, residual, reliability = history_features[field]
        columns.extend([rate, log_count, residual, reliability])

    for name in NUM_FIELDS:
        raw = np.asarray(split.num[name], dtype=np.float32)
        columns.append(np.sign(raw) * np.log1p(np.abs(raw)))
        columns.append(raw)

    categorical_start = len(columns)
    for name in RAW_CAT_FIELDS:
        columns.append(np.asarray(split.X[name], dtype=np.float32))

    date = np.asarray(split.date, dtype=np.int32)
    columns.append((date % 100).astype(np.float32))

    matrix = np.empty((n, len(columns)), dtype=np.float32)
    for j, column in enumerate(columns):
        matrix[:, j] = column

    categorical_indices = list(
        range(categorical_start, categorical_start + len(RAW_CAT_FIELDS))
    )
    return matrix, categorical_indices


train = load("train")
valid = load("valid")

# Incumbent collaborative FM.
x_train_fm = make_fm_matrix(train)
x_valid_fm = make_fm_matrix(valid)
y_train_tensor = torch.from_numpy(np.asarray(train.y, dtype=np.float32))

fm = FactorizationMachine(fm_total_cardinality, FM_RANK)
sparse_optimizer = torch.optim.SparseAdam(
    [fm.embedding.weight], lr=FM_LR
)
dense_optimizer = torch.optim.Adam([fm.bias], lr=FM_LR)
criterion = nn.BCEWithLogitsLoss()

generator = torch.Generator()
generator.manual_seed(SEED)

best_fm_primary = -np.inf
best_fm_state = None
best_fm_scores = None
best_fm_metrics = None

for epoch in range(FM_EPOCHS):
    fm.train()
    permutation = torch.randperm(len(train.y), generator=generator)
    last_loss = 0.0

    for start in range(0, len(train.y), FM_BATCH):
        indices = permutation[start:min(start + FM_BATCH, len(train.y))]
        sparse_optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)

        logits = fm(x_train_fm[indices])
        loss = criterion(logits, y_train_tensor[indices])
        loss.backward()
        sparse_optimizer.step()
        dense_optimizer.step()
        last_loss = float(loss.detach())

    scores = predict_fm(fm, x_valid_fm)
    metrics = evaluate(valid.user_id, valid.y, scores)
    primary = float(metrics["primary"])
    if primary > best_fm_primary:
        best_fm_primary = primary
        best_fm_scores = scores.copy()
        best_fm_metrics = metric_dict(metrics)
        best_fm_state = {
            key: value.detach().clone()
            for key, value in fm.state_dict().items()
        }

    print(
        "fm_epoch=%d loss=%.6f primary=%.6f"
        % (epoch + 1, last_loss, primary),
        flush=True,
    )

fm.load_state_dict(best_fm_state)

# Empirical-Bayes user-content history features. Training rows use
# leave-one-out aggregates, so their own target cannot enter their features.
history_encoder = HistoryEncoder(train)
train_history = history_encoder.train_features
valid_history = history_encoder.transform(valid)

train_history_score = history_summary(train_history)
valid_history_score = history_summary(valid_history)
history_metrics = evaluate(
    valid.user_id, valid.y, valid_history_score
)

train_matrix, categorical_indices = make_lgb_matrix(
    train, train_history
)
valid_matrix, _ = make_lgb_matrix(valid, valid_history)

dtrain = lgb.Dataset(
    train_matrix,
    label=np.asarray(train.y, dtype=np.float32),
    categorical_feature=categorical_indices,
    free_raw_data=False,
)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 200,
    "min_sum_hessian_in_leaf": 1e-3,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l1": 0.05,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "max_cat_threshold": 64,
    "cat_smooth": 20.0,
    "cat_l2": 10.0,
    "verbosity": -1,
    "verbose": -1,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "data_random_seed": SEED,
    "num_threads": max(1, min(8, os.cpu_count() or 1)),
    "force_col_wise": True,
}

booster = lgb.train(
    params,
    dtrain,
    num_boost_round=LGB_ROUNDS,
    callbacks=[lgb.log_evaluation(period=100)],
)

fm_mean, fm_scale = fit_standardizer(best_fm_scores)
fm_valid_z = apply_standardizer(best_fm_scores, fm_mean, fm_scale)

hist_mean, hist_scale = fit_standardizer(valid_history_score)
hist_valid_z = apply_standardizer(
    valid_history_score, hist_mean, hist_scale
)

candidates = {
    "fm": best_fm_primary,
    "history_only": float(history_metrics["primary"]),
}
best_primary = best_fm_primary
best_metrics = best_fm_metrics
best_scores = best_fm_scores
best_round = None
best_fm_weight = None
best_tree_mean = None
best_tree_scale = None

# The history-only blend checks whether the empirical-Bayes signal adds
# information even without relying on the tree model.
for weight in [0.25, 0.5, 1.0, 2.0]:
    scores = hist_valid_z + weight * fm_valid_z
    metrics = evaluate(valid.user_id, valid.y, scores)
    name = "history_fm_%.2f" % weight
    candidates[name] = float(metrics["primary"])
    if float(metrics["primary"]) > best_primary:
        best_primary = float(metrics["primary"])
        best_metrics = metric_dict(metrics)
        best_scores = scores.copy()
        best_round = -1
        best_fm_weight = weight

for num_iteration in LGB_CHECKPOINTS:
    tree_scores = np.asarray(
        booster.predict(valid_matrix, num_iteration=num_iteration),
        dtype=np.float64,
    )
    tree_mean, tree_scale = fit_standardizer(tree_scores)
    tree_z = apply_standardizer(tree_scores, tree_mean, tree_scale)

    tree_metrics = evaluate(valid.user_id, valid.y, tree_scores)
    candidates["history_gbdt_%d" % num_iteration] = float(
        tree_metrics["primary"]
    )

    for weight in BLEND_WEIGHTS:
        blended = tree_z + weight * fm_valid_z
        metrics = evaluate(valid.user_id, valid.y, blended)
        primary = float(metrics["primary"])
        name = "history_gbdt_%d_fm_%.2f" % (num_iteration, weight)
        candidates[name] = primary

        if primary > best_primary:
            best_primary = primary
            best_metrics = metric_dict(metrics)
            best_scores = blended.copy()
            best_round = num_iteration
            best_fm_weight = weight
            best_tree_mean = tree_mean
            best_tree_scale = tree_scale

print("CANDIDATES " + json.dumps(candidates, sort_keys=True))
print(
    "FINDINGS history_only_primary=%.6f selected_round=%s selected_fm_weight=%s"
    % (
        float(history_metrics["primary"]),
        str(best_round),
        str(best_fm_weight),
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    test = load("test")

    x_test_fm = make_fm_matrix(test)
    test_fm_scores = predict_fm(fm, x_test_fm)
    test_fm_z = apply_standardizer(
        test_fm_scores, fm_mean, fm_scale
    )

    test_history = history_encoder.transform(test)
    test_history_score = history_summary(test_history)
    test_history_z = apply_standardizer(
        test_history_score, hist_mean, hist_scale
    )

    if best_round is None:
        test_scores = test_fm_scores
    elif best_round == -1:
        test_scores = test_history_z + best_fm_weight * test_fm_z
    else:
        test_matrix, _ = make_lgb_matrix(test, test_history)
        test_tree_scores = booster.predict(
            test_matrix, num_iteration=best_round
        )
        test_tree_z = apply_standardizer(
            test_tree_scores, best_tree_mean, best_tree_scale
        )
        test_scores = test_tree_z + best_fm_weight * test_fm_z

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