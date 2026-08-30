import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 24117
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

KEYS = ["video_id", "author_id", "tag", "tab", "upload_type", "hour"]
AUX_NAMES = ["is_click", "is_like", "is_follow"]
NUM_NAMES = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]
SMOOTHING = {
    "video_id": 15.0,
    "author_id": 20.0,
    "tag": 35.0,
    "tab": 35.0,
    "upload_type": 35.0,
    "hour": 50.0,
}
BATCH_SIZE = 32768
LINEAR_EPOCHS = 5
HALF_LIFE = 8.0


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    su = user_ids[order]
    starts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    counts = ends - starts
    repeated_starts = np.repeat(starts, counts)
    repeated_counts = np.repeat(counts, counts)
    pos = np.arange(n, dtype=np.float64) - repeated_starts
    ranked = np.full(n, 0.5, dtype=np.float64)
    mask = repeated_counts > 1
    ranked[mask] = pos[mask] / (repeated_counts[mask] - 1.0)
    out = np.empty(n, dtype=np.float64)
    out[order] = ranked
    return out


def date_weights(dates):
    d = np.asarray(dates)
    unique = np.unique(d)
    index = np.searchsorted(unique, d).astype(np.float32)
    age = float(len(unique) - 1) - index
    w = np.exp2(-age / HALF_LIFE).astype(np.float32)
    return w / max(float(w.mean()), 1e-6)


def safe_aux(split, name):
    if name not in split.aux:
        return np.zeros(len(split.user_id), dtype=np.float32)
    x = np.asarray(split.aux[name], dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)


def outcomes(split, include_label=True):
    cols = []
    if include_label:
        cols.append(np.asarray(split.y, dtype=np.float32))
    for name in AUX_NAMES:
        cols.append(safe_aux(split, name))
    return np.column_stack(cols).astype(np.float32, copy=False)


def aggregate_tables(fit_keys, fit_outcomes):
    tables = {}
    global_means = fit_outcomes.mean(axis=0).astype(np.float64)
    for name in KEYS:
        ids = np.asarray(fit_keys[name], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[name])
        counts = np.bincount(ids, minlength=card).astype(np.float64)
        sums = np.empty((fit_outcomes.shape[1], card), dtype=np.float64)
        for j in range(fit_outcomes.shape[1]):
            sums[j] = np.bincount(
                ids,
                weights=fit_outcomes[:, j].astype(np.float64),
                minlength=card,
            )
        tables[name] = (counts, sums)
    return tables, global_means


def construct_features(
    row_split,
    tables,
    global_means,
    row_outcomes=None,
    leave_one_out=False,
):
    n = len(row_split.user_id)
    columns = []

    for name in KEYS:
        ids = np.asarray(row_split.X[name], dtype=np.int64)
        counts, sums = tables[name]
        alpha = float(SMOOTHING[name])

        c = counts[ids]
        if leave_one_out:
            c_eff = np.maximum(c - 1.0, 0.0)
        else:
            c_eff = c

        columns.append(np.log1p(c_eff).astype(np.float32))

        for j in range(len(global_means)):
            numer = sums[j, ids]
            if leave_one_out:
                numer = numer - row_outcomes[:, j]
            rate = (
                numer + alpha * global_means[j]
            ) / np.maximum(c_eff + alpha, 1e-8)
            columns.append(rate.astype(np.float32))

    for name in NUM_NAMES:
        x = np.asarray(row_split.num[name], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        columns.append(np.log1p(np.maximum(x, 0.0)).astype(np.float32))

    return np.column_stack(columns).astype(np.float32, copy=False)


def standardize_fit(x):
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 1e-5, std, 1.0).astype(np.float32)
    return mean, std


def apply_standardize(x, mean, std):
    return ((x - mean) / std).astype(np.float32, copy=False)


class DenseLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Linear(dim, 1)

    def forward(self, x):
        return self.layer(x).squeeze(1)


def fit_linear(x, y, weights, epochs=LINEAR_EPOCHS):
    torch.manual_seed(SEED)
    model = DenseLinear(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.012, weight_decay=2e-5)

    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y.astype(np.float32, copy=False))
    wt = torch.from_numpy(weights.astype(np.float32, copy=False))
    n = len(y)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        model.train()
        for lo in range(0, n, BATCH_SIZE):
            idx = order[lo:min(lo + BATCH_SIZE, n)]
            logits = model(xt[idx])
            loss_vec = nn.functional.binary_cross_entropy_with_logits(
                logits, yt[idx], reduction="none"
            )
            loss = (loss_vec * wt[idx]).sum() / wt[idx].sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def predict_linear(model, x):
    result = np.empty(len(x), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for lo in range(0, len(x), BATCH_SIZE * 2):
            hi = min(lo + BATCH_SIZE * 2, len(x))
            result[lo:hi] = model(torch.from_numpy(x[lo:hi])).cpu().numpy()
    return result


def fit_gaussian(x, y, weights):
    y = np.asarray(y, dtype=np.int8)
    weights = np.asarray(weights, dtype=np.float64)
    params = []
    for cls in (0, 1):
        mask = y == cls
        w = weights[mask]
        xx = x[mask].astype(np.float64)
        denom = max(w.sum(), 1.0)
        mean = (xx * w[:, None]).sum(axis=0) / denom
        var = (((xx - mean) ** 2) * w[:, None]).sum(axis=0) / denom
        var = np.maximum(var, 0.08)
        prior = denom / max(weights.sum(), 1.0)
        params.append((mean, var, prior))
    return params


def predict_gaussian(params, x):
    xx = x.astype(np.float64)
    logs = []
    for mean, var, prior in params:
        logp = (
            -0.5 * np.log(var)
            -0.5 * ((xx - mean) ** 2) / var
        ).sum(axis=1)
        logs.append(logp + np.log(max(prior, 1e-8)))
    return (logs[1] - logs[0]).astype(np.float32)


def fit_rf(x, y, weights):
    dset = lgb.Dataset(
        x,
        label=y,
        weight=weights,
        free_raw_data=True,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "rf",
        "num_leaves": 127,
        "max_depth": 12,
        "min_data_in_leaf": 120,
        "feature_fraction": 0.72,
        "bagging_fraction": 0.70,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 1.0,
        "learning_rate": 0.08,
        "max_bin": 127,
        "num_threads": min(8, max(1, os.cpu_count() or 1)),
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "verbose": -1,
    }
    return lgb.train(params, dset, num_boost_round=180)


def fit_family(name, x, y, weights):
    if name == "bagged_rf":
        return fit_rf(x, y, weights)
    if name == "dense_linear":
        return fit_linear(x, y, weights)
    if name == "gaussian_generative":
        return fit_gaussian(x, y, weights)
    raise ValueError(name)


def predict_family(name, model, x):
    if name == "bagged_rf":
        return model.predict(x).astype(np.float32)
    if name == "dense_linear":
        return predict_linear(model, x)
    if name == "gaussian_generative":
        return predict_gaussian(model, x)
    raise ValueError(name)


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)
train_outcomes = outcomes(train, include_label=True)

train_keys = {name: np.asarray(train.X[name], dtype=np.int64) for name in KEYS}
tables, global_means = aggregate_tables(train_keys, train_outcomes)

x_train_raw = construct_features(
    train,
    tables,
    global_means,
    row_outcomes=train_outcomes,
    leave_one_out=True,
)
x_valid_raw = construct_features(
    valid,
    tables,
    global_means,
    row_outcomes=None,
    leave_one_out=False,
)

mean, std = standardize_fit(x_train_raw)
x_train = apply_standardize(x_train_raw, mean, std)
x_valid = apply_standardize(x_valid_raw, mean, std)
del x_train_raw, x_valid_raw

weights_train = date_weights(train.date)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_valid_rank = within_user_rank(valid_users, inc_valid)

families = ["bagged_rf", "dense_linear", "gaussian_generative"]
models = {}
raw_valid = {}
candidate_log = {}
best_primary = -np.inf
best_family = None
best_alpha = None
best_scores = None
best_raw = None
best_metrics = None

alphas = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]

for family in families:
    model = fit_family(family, x_train, y_train, weights_train)
    models[family] = model
    pred = predict_family(family, model, x_valid)
    raw_valid[family] = pred

    standalone_metrics = evaluate(valid_users, y_valid, pred)
    candidate_log[family + "_standalone"] = float(standalone_metrics["primary"])

    pred_rank = within_user_rank(valid_users, pred)
    for alpha in alphas:
        blended = (1.0 - alpha) * inc_valid_rank + alpha * pred_rank
        metrics = evaluate(valid_users, y_valid, blended)
        name = family + "_blend_" + str(alpha)
        candidate_log[name] = float(metrics["primary"])
        if float(metrics["primary"]) > best_primary:
            best_primary = float(metrics["primary"])
            best_family = family
            best_alpha = float(alpha)
            best_scores = blended.copy()
            best_raw = pred.copy()
            best_metrics = metrics

print("FINDINGS " + json.dumps({
    "selected_family": best_family,
    "selected_alpha": best_alpha,
    "feature_count": int(x_train.shape[1]),
    "train_rows": int(len(y_train)),
}, sort_keys=True))
print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    if best_alpha < 1.0:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

# Refit the selected recipe on train + validation, then score test.
valid_outcomes = outcomes(valid, include_label=True)
y_fit = np.concatenate([y_train, y_valid]).astype(np.int8, copy=False)
outcomes_fit = np.concatenate([train_outcomes, valid_outcomes], axis=0)

fit_keys = {
    name: np.concatenate(
        [
            np.asarray(train.X[name], dtype=np.int64),
            np.asarray(valid.X[name], dtype=np.int64),
        ]
    )
    for name in KEYS
}
tables_fit, global_fit = aggregate_tables(fit_keys, outcomes_fit)

class JoinedSplit:
    pass

joined = JoinedSplit()
joined.user_id = np.concatenate([
    np.asarray(train.user_id),
    np.asarray(valid.user_id),
])
joined.X = {
    name: np.concatenate([
        np.asarray(train.X[name]),
        np.asarray(valid.X[name]),
    ])
    for name in KEYS
}
joined.num = {
    name: np.concatenate([
        np.asarray(train.num[name]),
        np.asarray(valid.num[name]),
    ])
    for name in NUM_NAMES
}

x_fit_raw = construct_features(
    joined,
    tables_fit,
    global_fit,
    row_outcomes=outcomes_fit,
    leave_one_out=True,
)
mean_fit, std_fit = standardize_fit(x_fit_raw)
x_fit = apply_standardize(x_fit_raw, mean_fit, std_fit)
del x_fit_raw

dates_fit = np.concatenate([
    np.asarray(train.date),
    np.asarray(valid.date),
])
weights_fit = date_weights(dates_fit)

selected_model = fit_family(
    best_family,
    x_fit,
    y_fit,
    weights_fit,
)

test = load("test")
x_test_raw = construct_features(
    test,
    tables_fit,
    global_fit,
    row_outcomes=None,
    leave_one_out=False,
)
x_test = apply_standardize(x_test_raw, mean_fit, std_fit)
test_raw = predict_family(best_family, selected_model, x_test)
test_rank = within_user_rank(np.asarray(test.user_id), test_raw)

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
inc_test_rank = within_user_rank(np.asarray(test.user_id), inc_test)
test_scores = (1.0 - best_alpha) * inc_test_rank + best_alpha * test_rank

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))