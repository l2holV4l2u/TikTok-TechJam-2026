import os
import gc
import json
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")

START = time.time()
OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")

if OUT:
    os.makedirs(OUT, exist_ok=True)

np.random.seed(2718)
torch.manual_seed(2718)
torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))

CROSS_FIELDS = [
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
]

RAW_CATEGORICALS = [
    "user_id",
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "hour",
    "user_active_degree",
    "music_type",
]

NUMERIC_FIELDS = [
    "duration_ms",
    "user_follow_user_num",
    "user_fans_user_num",
    "user_friend_user_num",
    "user_register_days",
]

HALF_LIFE_DAYS = 5.0
SMOOTH = 20.0


def clipped_cat(split, name):
    x = np.asarray(split.X[name], dtype=np.int64)
    card = int(FEATURE_CARDINALITIES[name])
    return np.where((x >= 0) & (x < card), x, 0).astype(np.int64)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    su = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = su[1:] != su[:-1]

    start_pos = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local = np.arange(n, dtype=np.float64) - start_pos

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = su[:-1] != su[1:]
    end_pos = np.flatnonzero(ends)
    sizes = np.diff(np.r_[-1, end_pos]).astype(np.float64)

    group = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[group] - 1.0, 1.0)

    result = np.empty(n, dtype=np.float32)
    result[order] = (local / denom).astype(np.float32)
    return result


def make_key(split, field):
    uid = clipped_cat(split, "user_id")
    value = clipped_cat(split, field)
    card = int(FEATURE_CARDINALITIES[field])
    return uid * np.int64(card) + value


def weighted_global_rate(labels, date):
    max_date = int(np.max(date))
    age = max_date - date.astype(np.int64)
    weight = np.exp(
        -np.log(2.0) * age.astype(np.float64) / HALF_LIFE_DAYS
    )
    return float(np.sum(weight * labels) / np.sum(weight))


def causal_prior_day_statistics(key, day_index, labels, global_rate):
    """
    Each training row receives statistics accumulated strictly before its
    calendar day. Thus no row uses its own label or any same-day outcome.
    Frozen totals over all train days are returned for valid/test lookup.
    """
    key = np.asarray(key, dtype=np.int64)
    day_index = np.asarray(day_index, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    n = len(key)

    packed = key * np.int64(32) + day_index
    order = np.argsort(packed, kind="stable")
    sp = packed[order]
    sy = labels[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sp[1:] != sp[:-1]
    group_starts = np.flatnonzero(starts)
    group_ends = np.r_[group_starts[1:], n]

    group_packed = sp[group_starts]
    group_key = group_packed // np.int64(32)
    group_count = (group_ends - group_starts).astype(np.float64)
    group_sum = np.add.reduceat(sy, group_starts).astype(np.float64)

    key_starts = np.empty(len(group_key), dtype=bool)
    key_starts[0] = True
    key_starts[1:] = group_key[1:] != group_key[:-1]
    key_start_pos = np.flatnonzero(key_starts)

    cumulative_count = np.cumsum(group_count)
    cumulative_sum = np.cumsum(group_sum)

    base_count = np.zeros(len(key_start_pos), dtype=np.float64)
    base_sum = np.zeros(len(key_start_pos), dtype=np.float64)
    has_previous = key_start_pos > 0
    base_count[has_previous] = cumulative_count[
        key_start_pos[has_previous] - 1
    ]
    base_sum[has_previous] = cumulative_sum[
        key_start_pos[has_previous] - 1
    ]

    key_group_index = np.cumsum(key_starts, dtype=np.int64) - 1
    prior_count_group = (
        cumulative_count - group_count - base_count[key_group_index]
    )
    prior_sum_group = (
        cumulative_sum - group_sum - base_sum[key_group_index]
    )

    posterior_group = (
        prior_sum_group + SMOOTH * global_rate
    ) / (prior_count_group + SMOOTH)

    group_id_sorted = np.cumsum(starts, dtype=np.int64) - 1
    rate_sorted = posterior_group[group_id_sorted]
    count_sorted = prior_count_group[group_id_sorted]

    rate = np.empty(n, dtype=np.float32)
    log_count = np.empty(n, dtype=np.float32)
    rate[order] = rate_sorted.astype(np.float32)
    log_count[order] = np.log1p(count_sorted[group_id_sorted]).astype(
        np.float32
    )

    last_group_indices = np.r_[
        key_start_pos[1:] - 1, len(group_key) - 1
    ]
    frozen_keys = group_key[last_group_indices].astype(np.int64)
    frozen_count = (
        cumulative_count[last_group_indices]
        - base_count[np.arange(len(key_start_pos))]
    )
    frozen_sum = (
        cumulative_sum[last_group_indices]
        - base_sum[np.arange(len(key_start_pos))]
    )
    frozen_rate = (
        frozen_sum + SMOOTH * global_rate
    ) / (frozen_count + SMOOTH)

    state = {
        "keys": frozen_keys,
        "rate": frozen_rate.astype(np.float32),
        "log_count": np.log1p(frozen_count).astype(np.float32),
    }
    return rate, log_count, state


def frozen_lookup(key, state, global_rate):
    key = np.asarray(key, dtype=np.int64)
    keys = state["keys"]
    pos = np.searchsorted(keys, key)
    valid = pos < len(keys)
    matched = np.zeros(len(key), dtype=bool)
    matched[valid] = keys[pos[valid]] == key[valid]

    rate = np.full(len(key), global_rate, dtype=np.float32)
    log_count = np.zeros(len(key), dtype=np.float32)
    rate[matched] = state["rate"][pos[matched]]
    log_count[matched] = state["log_count"][pos[matched]]
    return rate, log_count


def transform_numeric(x):
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    result = np.zeros(len(x), dtype=np.float32)
    result[finite] = np.log1p(np.maximum(x[finite], 0.0))
    return result


def history_matrix(split_name):
    pieces = []
    names = []
    for entity in ["video_id", "author_id"]:
        history = historical_features(split_name, key=entity)
        for name in sorted(history.keys()):
            x = np.asarray(history[name], dtype=np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0)
            pieces.append(x)
            names.append(entity + ":" + name)
    if not pieces:
        return np.empty((len(load(split_name)), 0), dtype=np.float32), names
    return np.column_stack(pieces).astype(np.float32), names


def build_matrix(split, split_name, memory_features, hist):
    columns = []
    names = []
    categorical_indices = []

    for field in RAW_CATEGORICALS:
        categorical_indices.append(len(columns))
        columns.append(clipped_cat(split, field).astype(np.float32))
        names.append("cat:" + field)

    for field in NUMERIC_FIELDS:
        columns.append(transform_numeric(split.num[field]))
        names.append("num:" + field)

    for name, x in memory_features:
        columns.append(np.asarray(x, dtype=np.float32))
        names.append(name)

    if hist.shape[1]:
        for j in range(hist.shape[1]):
            columns.append(hist[:, j])
            names.append("history_%d" % j)

    matrix = np.column_stack(columns).astype(np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=20.0, neginf=-20.0)
    return matrix, names, categorical_indices


class AdditiveLogistic(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear = nn.Linear(dimension, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


def train_additive(train_x, labels, sample_weight):
    mean = np.mean(train_x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(train_x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 1e-5, std, 1.0).astype(np.float32)

    model = AdditiveLogistic(train_x.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.015, weight_decay=2e-5
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(991)
    batch = 65536

    model.train()
    for epoch in range(3):
        permutation = rng.permutation(len(labels))
        loss_sum = 0.0
        weight_sum = 0.0
        for start in range(0, len(labels), batch):
            idx = permutation[start:start + batch]
            xb = (
                train_x[idx] - mean[None, :]
            ) / std[None, :]
            xb = np.clip(xb, -8.0, 8.0)
            xb = torch.from_numpy(xb.astype(np.float32, copy=False))
            yb = torch.from_numpy(labels[idx].astype(np.float32))
            wb = torch.from_numpy(sample_weight[idx].astype(np.float32))

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = criterion(logits, yb)
            loss = torch.sum(losses * wb) / torch.sum(wb).clamp_min(1.0)
            loss.backward()
            optimizer.step()

            loss_sum += float(torch.sum(losses * wb).detach())
            weight_sum += float(torch.sum(wb))

        print(
            "FINDINGS additive_epoch=%d weighted_logloss=%.6f"
            % (epoch + 1, loss_sum / max(weight_sum, 1.0)),
            flush=True,
        )
    return model, mean, std


def predict_additive(model, x, mean, std):
    result = np.empty(len(x), dtype=np.float32)
    batch = 131072
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch):
            end = min(start + batch, len(x))
            xb = (x[start:end] - mean[None, :]) / std[None, :]
            xb = np.clip(xb, -8.0, 8.0)
            tensor = torch.from_numpy(xb.astype(np.float32, copy=False))
            result[start:end] = torch.sigmoid(model(tensor)).numpy()
    return result


def empirical_bayes_score(memory_features):
    rates = []
    weights = []
    for name, x in memory_features:
        if name.endswith("_rate"):
            rates.append(np.asarray(x, dtype=np.float32))
        elif name.endswith("_log_count"):
            weights.append(np.asarray(x, dtype=np.float32))

    score_num = np.zeros(len(rates[0]), dtype=np.float64)
    score_den = np.zeros(len(rates[0]), dtype=np.float64)
    for rate, log_count in zip(rates, weights):
        confidence = np.minimum(log_count.astype(np.float64), 8.0) + 0.5
        score_num += confidence * rate
        score_den += confidence
    return (score_num / np.maximum(score_den, 1e-6)).astype(np.float32)


def evaluate_candidate(name, raw_scores, incumbent_rank, uid, labels,
                       candidates, best):
    raw_rank = within_user_rank(uid, raw_scores)
    raw_metrics = evaluate(uid, labels, raw_rank)
    candidates[name + "_standalone"] = float(raw_metrics["primary"])

    print(
        "FINDINGS family=%s standalone_primary=%.6f gauc=%.6f ndcg5=%.6f"
        % (
            name,
            float(raw_metrics["primary"]),
            float(raw_metrics["gauc"]),
            float(raw_metrics["ndcg@5"]),
        ),
        flush=True,
    )

    for alpha in [0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]:
        blended = (
            (1.0 - alpha) * incumbent_rank + alpha * raw_rank
        ).astype(np.float32)
        metrics = evaluate(uid, labels, blended)
        key = "%s_blend_%.3f" % (name, alpha)
        candidates[key] = float(metrics["primary"])

        if float(metrics["primary"]) > best["primary"]:
            best.update({
                "primary": float(metrics["primary"]),
                "scores": blended.copy(),
                "raw_rank": raw_rank.copy(),
                "family": name,
                "alpha": float(alpha),
            })
    return raw_rank


inc_valid_path = (
    os.path.join(SHARED, "incumbent_valid_scores.npy")
    if SHARED else ""
)
inc_test_path = (
    os.path.join(SHARED, "incumbent_test_scores.npy")
    if SHARED else ""
)
if not (
    inc_valid_path
    and inc_test_path
    and os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise RuntimeError("Trusted incumbent artifacts are required")

train = load("train")
valid = load("valid")

train_y = np.asarray(train.y, dtype=np.float32)
train_date = np.asarray(train.date, dtype=np.int64)
unique_dates = np.unique(train_date)
day_index = np.searchsorted(unique_dates, train_date).astype(np.int64)
global_rate = weighted_global_rate(train_y, train_date)

age_days = int(train_date.max()) - train_date
sample_weight = np.exp(
    -np.log(2.0) * age_days.astype(np.float32) / HALF_LIFE_DAYS
).astype(np.float32)
sample_weight /= np.mean(sample_weight)

train_memory = []
valid_memory = []
states = {}

# User-only state.
user_key_train = clipped_cat(train, "user_id")
user_rate, user_count, state = causal_prior_day_statistics(
    user_key_train, day_index, train_y, global_rate
)
states["__user__"] = state
train_memory.extend([
    ("user_rate", user_rate),
    ("user_log_count", user_count),
])

user_rate_v, user_count_v = frozen_lookup(
    clipped_cat(valid, "user_id"), state, global_rate
)
valid_memory.extend([
    ("user_rate", user_rate_v),
    ("user_log_count", user_count_v),
])

# User-by-context persistent preference states.
for field in CROSS_FIELDS:
    print("FINDINGS building_prior_day_state=%s" % field, flush=True)
    key_train = make_key(train, field)
    rate, count, state = causal_prior_day_statistics(
        key_train, day_index, train_y, global_rate
    )
    states[field] = state
    train_memory.extend([
        (field + "_rate", rate),
        (field + "_log_count", count),
    ])

    key_valid = make_key(valid, field)
    rate_v, count_v = frozen_lookup(key_valid, state, global_rate)
    valid_memory.extend([
        (field + "_rate", rate_v),
        (field + "_log_count", count_v),
    ])
    del key_train, key_valid, rate, count, rate_v, count_v
    gc.collect()

train_hist, history_names = history_matrix("train")
valid_hist, _ = history_matrix("valid")

train_x, feature_names, categorical_indices = build_matrix(
    train, "train", train_memory, train_hist
)
valid_x, _, _ = build_matrix(
    valid, "valid", valid_memory, valid_hist
)

print(
    "FINDINGS dense_features=%d global_rate=%.6f memory_fields=%d"
    % (train_x.shape[1], global_rate, 1 + len(CROSS_FIELDS)),
    flush=True,
)

# Non-parametric empirical-Bayes family.
eb_valid = empirical_bayes_score(valid_memory)

# Additive logistic family only sees dense state/history/numerical columns.
# Exclude raw categorical IDs because their numeric ordering is arbitrary.
dense_start = len(RAW_CATEGORICALS)
additive_train_x = train_x[:, dense_start:]
additive_valid_x = valid_x[:, dense_start:]
additive_model, additive_mean, additive_std = train_additive(
    additive_train_x, train_y, sample_weight
)
additive_valid = predict_additive(
    additive_model, additive_valid_x, additive_mean, additive_std
)

# Nonlinear boosted-tree family.
dtrain = lgb.Dataset(
    train_x,
    label=train_y,
    weight=sample_weight,
    categorical_feature=categorical_indices,
    free_raw_data=False,
)
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.055,
    "num_leaves": 63,
    "min_data_in_leaf": 1500,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.82,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 2.0,
    "max_bin": 127,
    "min_gain_to_split": 0.001,
    "num_threads": max(1, min(12, os.cpu_count() or 1)),
    "seed": 31415,
    "feature_fraction_seed": 31415,
    "bagging_seed": 31415,
    "verbose": -1,
}
booster = lgb.train(params, dtrain, num_boost_round=450)
tree_valid = booster.predict(valid_x).astype(np.float32)

valid_uid = np.asarray(valid.user_id, dtype=np.int64)
valid_labels = np.asarray(valid.y)
inc_valid = np.asarray(
    np.load(inc_valid_path, mmap_mode="r"), dtype=np.float32
)
inc_valid_rank = within_user_rank(valid_uid, inc_valid)
inc_metrics = evaluate(valid_uid, valid_labels, inc_valid_rank)

candidates = {
    "trusted_incumbent": float(inc_metrics["primary"])
}
best = {
    "primary": float(inc_metrics["primary"]),
    "scores": inc_valid_rank.copy(),
    "raw_rank": None,
    "family": "trusted_incumbent",
    "alpha": 0.0,
}

evaluate_candidate(
    "empirical_bayes_state", eb_valid, inc_valid_rank,
    valid_uid, valid_labels, candidates, best
)
evaluate_candidate(
    "additive_logistic_state", additive_valid, inc_valid_rank,
    valid_uid, valid_labels, candidates, best
)
evaluate_candidate(
    "nonlinear_lgbm_state", tree_valid, inc_valid_rank,
    valid_uid, valid_labels, candidates, best
)

final_metrics = evaluate(valid_uid, valid_labels, best["scores"])

print(
    "FINDINGS winner=%s alpha=%.3f incumbent=%.6f winner=%.6f delta=%+.6f"
    % (
        best["family"],
        best["alpha"],
        float(inc_metrics["primary"]),
        float(final_metrics["primary"]),
        float(final_metrics["primary"]) - float(inc_metrics["primary"]),
    ),
    flush=True,
)
print("CANDIDATES " + json.dumps(candidates, sort_keys=True), flush=True)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best["scores"], dtype=np.float64),
    )
    if best["raw_rank"] is not None:
        np.save(
            os.path.join(OUT, "scores_valid_raw.npy"),
            np.asarray(best["raw_rank"], dtype=np.float64),
        )

# Release split-specific training/validation arrays before constructing test.
del valid
del train_x, valid_x
del additive_train_x, additive_valid_x
del train_hist, valid_hist
del train_memory, valid_memory
del eb_valid, additive_valid, tree_valid
del dtrain
gc.collect()

test = load("test")
test_memory = []

user_rate_t, user_count_t = frozen_lookup(
    clipped_cat(test, "user_id"), states["__user__"], global_rate
)
test_memory.extend([
    ("user_rate", user_rate_t),
    ("user_log_count", user_count_t),
])

for field in CROSS_FIELDS:
    key_test = make_key(test, field)
    rate_t, count_t = frozen_lookup(
        key_test, states[field], global_rate
    )
    test_memory.extend([
        (field + "_rate", rate_t),
        (field + "_log_count", count_t),
    ])
    del key_test, rate_t, count_t

test_hist, _ = history_matrix("test")
test_x, _, _ = build_matrix(test, "test", test_memory, test_hist)

inc_test = np.asarray(
    np.load(inc_test_path, mmap_mode="r"), dtype=np.float32
)
inc_test_rank = within_user_rank(test.user_id, inc_test)

if best["family"] == "trusted_incumbent":
    test_scores = inc_test_rank
elif best["family"] == "empirical_bayes_state":
    raw_test = empirical_bayes_score(test_memory)
    raw_test_rank = within_user_rank(test.user_id, raw_test)
    test_scores = (
        (1.0 - best["alpha"]) * inc_test_rank
        + best["alpha"] * raw_test_rank
    ).astype(np.float32)
elif best["family"] == "additive_logistic_state":
    raw_test = predict_additive(
        additive_model,
        test_x[:, dense_start:],
        additive_mean,
        additive_std,
    )
    raw_test_rank = within_user_rank(test.user_id, raw_test)
    test_scores = (
        (1.0 - best["alpha"]) * inc_test_rank
        + best["alpha"] * raw_test_rank
    ).astype(np.float32)
elif best["family"] == "nonlinear_lgbm_state":
    raw_test = booster.predict(test_x).astype(np.float32)
    raw_test_rank = within_user_rank(test.user_id, raw_test)
    test_scores = (
        (1.0 - best["alpha"]) * inc_test_rank
        + best["alpha"] * raw_test_rank
    ).astype(np.float32)
else:
    raise RuntimeError("Unknown winning family")

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print(
    "METRICS "
    + json.dumps({
        "primary": float(final_metrics["primary"]),
        "gauc": float(final_metrics["gauc"]),
        "ndcg@5": float(final_metrics["ndcg@5"]),
        "gpu_seconds": elapsed,
    })
)