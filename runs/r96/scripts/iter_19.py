import os
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load
from pipeline.evaluate import evaluate

START = time.time()
SEED = 91731

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))


def rank_percentile(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, users))
    su = users[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = su[1:] != su[:-1]
    start_index = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = su[:-1] != su[1:]
    end_index = np.flatnonzero(ends)
    sizes = np.diff(np.r_[np.int64(-1), end_index])
    row_sizes = np.repeat(sizes, sizes)

    position = np.arange(n, dtype=np.int64) - start_index
    ranked = (position.astype(np.float64) + 0.5) / row_sizes

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def group_position_and_size(start_flags):
    n = len(start_flags)
    indices = np.arange(n, dtype=np.int64)
    starts = np.maximum.accumulate(np.where(start_flags, indices, 0))
    position = indices - starts

    end_flags = np.empty(n, dtype=bool)
    end_flags[:-1] = start_flags[1:]
    end_flags[-1] = True
    ends = np.flatnonzero(end_flags)
    sizes = np.diff(np.r_[np.int64(-1), ends])
    row_sizes = np.repeat(sizes, sizes)
    return position, row_sizes


def exposure_features(split):
    n = len(split)
    rows = np.arange(n, dtype=np.int64)
    users = np.asarray(split.user_id, dtype=np.int64)
    times = np.asarray(split.time_ms, dtype=np.int64)
    dates = np.asarray(split.date, dtype=np.int64)

    order = np.lexsort((rows, times, users))
    su = users[order]
    st = times[order]
    sd = dates[order]

    user_start = np.empty(n, dtype=bool)
    user_start[0] = True
    user_start[1:] = su[1:] != su[:-1]

    day_start = np.empty(n, dtype=bool)
    day_start[0] = True
    day_start[1:] = (su[1:] != su[:-1]) | (sd[1:] != sd[:-1])

    batch_start = np.empty(n, dtype=bool)
    batch_start[0] = True
    batch_start[1:] = (su[1:] != su[:-1]) | (st[1:] != st[:-1])

    gap_prev_ms = np.zeros(n, dtype=np.int64)
    gap_prev_ms[1:] = np.maximum(st[1:] - st[:-1], 0)
    gap_prev_ms[user_start] = 0

    session_start = user_start | (gap_prev_ms > 30 * 60 * 1000)

    user_pos, user_size = group_position_and_size(user_start)
    day_pos, day_size = group_position_and_size(day_start)
    batch_pos, batch_size = group_position_and_size(batch_start)
    session_pos, session_size = group_position_and_size(session_start)

    gap_next_ms = np.zeros(n, dtype=np.int64)
    gap_next_ms[:-1] = np.maximum(st[1:] - st[:-1], 0)
    user_end = np.empty(n, dtype=bool)
    user_end[-1] = True
    user_end[:-1] = su[:-1] != su[1:]
    gap_next_ms[user_end] = 0

    ordered = np.column_stack([
        np.log1p(user_pos),
        np.log1p(user_size),
        (user_pos + 0.5) / np.maximum(user_size, 1),
        np.log1p(day_pos),
        np.log1p(day_size),
        (day_pos + 0.5) / np.maximum(day_size, 1),
        np.log1p(session_pos),
        np.log1p(session_size),
        (session_pos + 0.5) / np.maximum(session_size, 1),
        np.log1p(batch_pos),
        np.log1p(batch_size),
        (batch_pos + 0.5) / np.maximum(batch_size, 1),
        np.log1p(gap_prev_ms.astype(np.float64) / 1000.0),
        np.log1p(gap_next_ms.astype(np.float64) / 1000.0),
        (batch_size > 1).astype(np.float64),
        (session_pos == 0).astype(np.float64),
    ]).astype(np.float32)

    base = np.empty_like(ordered)
    base[order] = ordered

    duration = np.asarray(split.num["duration_ms"], dtype=np.float64)
    duration = np.nan_to_num(duration, nan=0.0, posinf=0.0, neginf=0.0)
    duration = np.log1p(np.maximum(duration, 0.0)).astype(np.float32)

    extras = np.column_stack([
        duration,
        np.asarray(split.X["hour"], dtype=np.float32),
        np.asarray(split.X["tab"], dtype=np.float32),
        np.asarray(split.X["duration_bucket"], dtype=np.float32),
        np.asarray(split.X["user_active_degree"], dtype=np.float32),
        np.asarray(split.X["is_video_author"], dtype=np.float32),
        np.asarray(split.X["is_live_streamer"], dtype=np.float32),
        np.asarray(split.X["tag"], dtype=np.float32),
        np.asarray(split.X["upload_type"], dtype=np.float32),
        np.asarray(split.X["video_type"], dtype=np.float32),
    ])

    matrix = np.column_stack([base, extras]).astype(np.float32)

    discrete = {
        "user_pos": np.minimum(
            np.floor(np.expm1(base[:, 0])).astype(np.int64), 63
        ),
        "day_pos": np.minimum(
            np.floor(np.expm1(base[:, 3])).astype(np.int64), 31
        ),
        "session_pos": np.minimum(
            np.floor(np.expm1(base[:, 6])).astype(np.int64), 31
        ),
        "batch_pos": np.minimum(
            np.floor(np.expm1(base[:, 9])).astype(np.int64), 15
        ),
        "batch_size": np.minimum(
            np.floor(np.expm1(base[:, 10])).astype(np.int64), 31
        ),
        "hour": np.asarray(split.X["hour"], dtype=np.int64),
        "tab": np.asarray(split.X["tab"], dtype=np.int64),
        "duration_bucket": np.asarray(
            split.X["duration_bucket"], dtype=np.int64
        ),
        "user_active_degree": np.asarray(
            split.X["user_active_degree"], dtype=np.int64
        ),
    }
    return matrix, discrete


def fit_empirical_bayes(train_discrete, labels, weights):
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    global_rate = float(np.sum(weights * labels) / np.sum(weights))
    global_logit = np.log(global_rate / max(1.0 - global_rate, 1e-8))

    state = {"global_logit": global_logit, "tables": {}}
    smoothing = {
        "user_pos": 120.0,
        "day_pos": 100.0,
        "session_pos": 80.0,
        "batch_pos": 60.0,
        "batch_size": 100.0,
        "hour": 250.0,
        "tab": 250.0,
        "duration_bucket": 180.0,
        "user_active_degree": 250.0,
    }

    for name, values in train_discrete.items():
        values = np.asarray(values, dtype=np.int64)
        size = int(values.max()) + 1
        count = np.bincount(values, weights=weights, minlength=size)
        positive = np.bincount(
            values, weights=weights * labels, minlength=size
        )
        alpha = smoothing[name]
        rate = (positive + alpha * global_rate) / (count + alpha)
        rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
        effect = np.log(rate / (1.0 - rate)) - global_logit
        reliability = count / (count + alpha)
        state["tables"][name] = effect * reliability
    return state


def predict_empirical_bayes(state, discrete):
    result = np.full(
        len(next(iter(discrete.values()))),
        state["global_logit"],
        dtype=np.float64,
    )
    effects = []
    for name, values in discrete.items():
        table = state["tables"][name]
        values = np.asarray(values, dtype=np.int64)
        clipped = np.minimum(values, len(table) - 1)
        effects.append(table[clipped])
    if effects:
        result += np.mean(np.stack(effects, axis=0), axis=0)
    return result


class PositionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def fit_position_mlp(x, y, weights, seed):
    torch.manual_seed(seed)
    model = PositionMLP(x.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0025, weight_decay=2e-5
    )
    rng = np.random.default_rng(seed)
    batch_size = 8192

    xt = torch.from_numpy(x)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    wt = torch.from_numpy(np.asarray(weights, dtype=np.float32))

    model.train()
    losses = []
    for epoch in range(4):
        permutation = rng.permutation(len(x))
        total_loss = 0.0
        total_weight = 0.0
        for start in range(0, len(x), batch_size):
            index = permutation[start:start + batch_size]
            logits = model(xt[index])
            element = nn.functional.binary_cross_entropy_with_logits(
                logits, yt[index], reduction="none"
            )
            denominator = wt[index].sum().clamp_min(1.0)
            loss = (element * wt[index]).sum() / denominator

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float((element * wt[index]).sum().detach())
            total_weight += float(denominator.detach())
        losses.append(total_loss / max(total_weight, 1.0))
    return model, losses


@torch.no_grad()
def predict_position_mlp(model, x):
    model.eval()
    output = np.empty(len(x), dtype=np.float64)
    batch_size = 16384
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        output[start:end] = model(
            torch.from_numpy(x[start:end])
        ).numpy()
    return output


train = load("train")
valid = load("valid")
test = load("test")

x_train, d_train = exposure_features(train)
x_valid, d_valid = exposure_features(valid)
x_test, d_test = exposure_features(test)

# Every normalizer is fitted on train only.
mean = x_train.mean(axis=0, dtype=np.float64)
std = x_train.std(axis=0, dtype=np.float64)
std = np.maximum(std, 1e-4)

xn_train = np.clip((x_train - mean) / std, -8.0, 8.0).astype(np.float32)
xn_valid = np.clip((x_valid - mean) / std, -8.0, 8.0).astype(np.float32)
xn_test = np.clip((x_test - mean) / std, -8.0, 8.0).astype(np.float32)

train_dates = np.asarray(train.date, dtype=np.int64)
age = np.maximum(int(train_dates.max()) - train_dates, 0).astype(np.float64)
sample_weight = np.power(0.5, age / 4.0)
sample_weight /= sample_weight.mean()

y_train = np.asarray(train.y, dtype=np.float32)

# Family 1: smoothed non-parametric positional hazard model.
eb_state = fit_empirical_bayes(d_train, y_train, sample_weight)
eb_valid = predict_empirical_bayes(eb_state, d_valid)
eb_test = predict_empirical_bayes(eb_state, d_test)

# Family 2: nonlinear neural scorer over the same exposure-context inputs.
mlp, mlp_losses = fit_position_mlp(
    xn_train, y_train, sample_weight, SEED + 1
)
mlp_valid = predict_position_mlp(mlp, xn_valid)
mlp_test = predict_position_mlp(mlp, xn_test)
del mlp
gc.collect()

# Family 3: boosted trees capture threshold effects such as batch-position
# discontinuities and interactions between fatigue, duration, hour, and tab.
binary_train = lgb.Dataset(
    x_train,
    label=y_train,
    weight=sample_weight,
    free_raw_data=False,
)
binary_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.045,
    "num_leaves": 31,
    "min_data_in_leaf": 800,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "num_threads": max(1, min(12, os.cpu_count() or 1)),
    "seed": SEED + 2,
    "verbose": -1,
}
binary_model = lgb.train(
    binary_params,
    binary_train,
    num_boost_round=350,
)
gbdt_valid = binary_model.predict(x_valid)
gbdt_test = binary_model.predict(x_test)
del binary_model, binary_train
gc.collect()

raw_valid = {
    "empirical_position": eb_valid,
    "position_mlp": mlp_valid,
    "position_gbdt": gbdt_valid,
}
raw_test = {
    "empirical_position": eb_test,
    "position_mlp": mlp_test,
    "position_gbdt": gbdt_test,
}

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = rank_percentile(valid.user_id, inc_valid)
inc_test_rank = rank_percentile(test.user_id, inc_test)

position_valid_rank = {
    name: rank_percentile(valid.user_id, scores)
    for name, scores in raw_valid.items()
}
position_test_rank = {
    name: rank_percentile(test.user_id, scores)
    for name, scores in raw_test.items()
}

candidate_valid = {"incumbent": inc_valid}
candidate_test = {"incumbent": inc_test}
candidate_raw = {"incumbent": inc_valid}

for name in raw_valid:
    candidate_valid[name + "_standalone"] = raw_valid[name]
    candidate_test[name + "_standalone"] = raw_test[name]
    candidate_raw[name + "_standalone"] = raw_valid[name]

    for alpha in (0.03, 0.06, 0.10, 0.15, 0.22, 0.30, 0.40):
        key = f"{name}_incblend_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * position_valid_rank[name]
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank
            + alpha * position_test_rank[name]
        )
        candidate_raw[key] = raw_valid[name]

# Borda aggregation across structurally different estimates of exposure bias.
position_ensemble_valid = np.mean(
    np.stack(list(position_valid_rank.values()), axis=0), axis=0
)
position_ensemble_test = np.mean(
    np.stack(list(position_test_rank.values()), axis=0), axis=0
)
candidate_valid["position_family_ensemble"] = position_ensemble_valid
candidate_test["position_family_ensemble"] = position_ensemble_test
candidate_raw["position_family_ensemble"] = position_ensemble_valid

for alpha in (0.04, 0.08, 0.12, 0.18, 0.25, 0.35):
    key = f"position_ensemble_incblend_{alpha:.2f}"
    candidate_valid[key] = (
        (1.0 - alpha) * inc_valid_rank
        + alpha * position_ensemble_valid
    )
    candidate_test[key] = (
        (1.0 - alpha) * inc_test_rank
        + alpha * position_ensemble_test
    )
    candidate_raw[key] = position_ensemble_valid

candidate_metrics = {
    name: evaluate(valid.user_id, valid.y, scores)
    for name, scores in candidate_valid.items()
}
best_name = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"]),
)
best_metrics = candidate_metrics[best_name]
best_valid = candidate_valid[best_name]
best_test = candidate_test[best_name]

correlations = {
    name: float(np.corrcoef(inc_valid_rank, rank_scores)[0, 1])
    for name, rank_scores in position_valid_rank.items()
}

print("CANDIDATES " + json.dumps({
    name: float(metrics["primary"])
    for name, metrics in candidate_metrics.items()
}, sort_keys=True))

print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "mlp_losses": mlp_losses,
    "rank_correlations_with_incumbent": correlations,
    "train_mean_batch_size": float(np.mean(np.expm1(x_train[:, 10]))),
    "valid_mean_batch_size": float(np.mean(np.expm1(x_valid[:, 10]))),
    "test_mean_batch_size": float(np.mean(np.expm1(x_test[:, 10]))),
    "train_multirow_batch_rate": float(np.mean(x_train[:, 14])),
    "valid_multirow_batch_rate": float(np.mean(x_valid[:, 14])),
    "test_multirow_batch_rate": float(np.mean(x_test[:, 14])),
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    if best_name != "incumbent":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[best_name], dtype=np.float64),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))