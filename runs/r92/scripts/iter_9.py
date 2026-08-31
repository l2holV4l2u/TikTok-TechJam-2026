import os
import time
import json
import gc
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7319
THREADS = max(1, min(8, os.cpu_count() or 1))

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)


BASE_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "user_active_degree",
    "is_live_streamer",
    "is_video_author",
]

SEQ_NAMES = [
    "user_prior",
    "day_prior",
    "batch_prior",
    "video_prior",
    "author_prior",
    "gap_prev",
    "gap_video",
    "gap_author",
]


def recency_weights(dates, half_life=4.0):
    dates = np.asarray(dates)
    unique_dates = np.unique(dates)
    day_index = np.searchsorted(unique_dates, dates)
    age = (len(unique_dates) - 1 - day_index).astype(np.float32)
    w = np.exp2(-age / float(half_life))
    return (w / np.mean(w)).astype(np.float32)


def local_rank_from_order(order, group_change):
    n = len(order)
    starts = np.flatnonzero(group_change)
    group_id = np.cumsum(group_change) - 1
    local = np.arange(n, dtype=np.int64) - starts[group_id]
    out = np.empty(n, dtype=np.float32)
    out[order] = local.astype(np.float32)
    return out


def sequence_features(parts):
    lengths = [len(p.user_id) for p in parts]
    cuts = np.cumsum([0] + lengths)
    n = cuts[-1]

    users = np.concatenate([
        np.asarray(p.user_id, dtype=np.int64) for p in parts
    ])
    videos = np.concatenate([
        np.asarray(p.video_id, dtype=np.int64) for p in parts
    ])
    authors = np.concatenate([
        np.asarray(p.X["author_id"], dtype=np.int64) for p in parts
    ])
    dates = np.concatenate([
        np.asarray(p.date, dtype=np.int64) for p in parts
    ])
    times = np.concatenate([
        np.asarray(p.time_ms, dtype=np.int64) for p in parts
    ])
    rows = np.arange(n, dtype=np.int64)

    result = np.zeros((n, len(SEQ_NAMES)), dtype=np.float32)

    # Prior impressions and time since the previous impression for this user.
    order = np.lexsort((rows, times, users))
    su = users[order]
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = su[1:] != su[:-1]
    result[:, 0] = local_rank_from_order(order, change)

    gap = np.zeros(n, dtype=np.float32)
    dt = np.diff(times[order]).astype(np.float64) / 1000.0
    valid_pair = ~change[1:]
    sorted_gap = np.zeros(n, dtype=np.float32)
    sorted_gap[1:] = np.where(
        valid_pair,
        np.clip(dt, 0.0, 86400.0 * 30.0),
        0.0,
    ).astype(np.float32)
    gap[order] = sorted_gap
    result[:, 5] = gap

    # Prior impressions during the same user-day.
    order = np.lexsort((rows, times, dates, users))
    su = users[order]
    sd = dates[order]
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = (su[1:] != su[:-1]) | (sd[1:] != sd[:-1])
    result[:, 1] = local_rank_from_order(order, change)

    # Position inside a feed batch sharing the same timestamp.
    order = np.lexsort((rows, times, users))
    su = users[order]
    st = times[order]
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = (su[1:] != su[:-1]) | (st[1:] != st[:-1])
    result[:, 2] = local_rank_from_order(order, change)

    # Candidate video repetition count and time since its previous exposure.
    order = np.lexsort((rows, times, videos, users))
    su = users[order]
    sv = videos[order]
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = (su[1:] != su[:-1]) | (sv[1:] != sv[:-1])
    result[:, 3] = local_rank_from_order(order, change)

    sorted_gap = np.zeros(n, dtype=np.float32)
    dt = np.diff(times[order]).astype(np.float64) / 1000.0
    sorted_gap[1:] = np.where(
        ~change[1:],
        np.clip(dt, 0.0, 86400.0 * 30.0),
        0.0,
    ).astype(np.float32)
    result[order, 6] = sorted_gap

    # Candidate author repetition count and time since its previous exposure.
    order = np.lexsort((rows, times, authors, users))
    su = users[order]
    sa = authors[order]
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = (su[1:] != su[:-1]) | (sa[1:] != sa[:-1])
    result[:, 4] = local_rank_from_order(order, change)

    sorted_gap = np.zeros(n, dtype=np.float32)
    dt = np.diff(times[order]).astype(np.float64) / 1000.0
    sorted_gap[1:] = np.where(
        ~change[1:],
        np.clip(dt, 0.0, 86400.0 * 30.0),
        0.0,
    ).astype(np.float32)
    result[order, 7] = sorted_gap

    # Compress heavy-tailed counts and gaps while retaining exact zero.
    result[:, :5] = np.log1p(result[:, :5])
    result[:, 5:] = np.log1p(result[:, 5:])

    outputs = [
        result[cuts[i]:cuts[i + 1]].copy()
        for i in range(len(parts))
    ]
    return outputs


def base_categorical_matrix(split):
    return np.column_stack([
        np.asarray(split.X[name], dtype=np.int32)
        for name in BASE_FIELDS
    ])


def lgb_matrix(split, seq):
    cats = base_categorical_matrix(split)
    return np.concatenate([
        cats.astype(np.float32),
        seq.astype(np.float32),
    ], axis=1)


def fit_sequence_gbdt(x, y, dates, rounds=220):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.055,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 140,
        "max_bin": 127,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.92,
        "bagging_freq": 1,
        "lambda_l1": 0.04,
        "lambda_l2": 2.0,
        "min_gain_to_split": 0.002,
        "seed": SEED,
        "feature_fraction_seed": SEED + 1,
        "bagging_seed": SEED + 2,
        "num_threads": THREADS,
        "force_col_wise": True,
        "verbose": -1,
    }
    dset = lgb.Dataset(
        x,
        label=y,
        weight=recency_weights(dates, 4.0),
        categorical_feature=list(range(len(BASE_FIELDS))),
        free_raw_data=False,
    )
    return lgb.train(params, dset, num_boost_round=rounds)


def discretize_sequence(seq):
    # Inputs are already log1p transformed. Half-log-unit bins distinguish
    # early positions and repetitions while pooling sparse long histories.
    bins = np.floor(seq * 2.0).astype(np.int64)
    return np.clip(bins, 0, 19)


def wide_matrix(split, seq):
    base = base_categorical_matrix(split).astype(np.int64)
    sb = discretize_sequence(seq)

    cardinalities = [
        int(FEATURE_CARDINALITIES[name]) for name in BASE_FIELDS
    ] + [20] * len(SEQ_NAMES)

    raw = np.concatenate([base, sb], axis=1)
    offsets = np.cumsum([0] + cardinalities[:-1]).astype(np.int64)
    raw += offsets[None, :]
    return raw, cardinalities


class WideHazard(nn.Module):
    def __init__(self, cardinalities):
        super().__init__()
        total = int(sum(cardinalities))
        self.weight = nn.Embedding(total, 1)
        self.bias = nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            self.weight.weight.zero_()

    def forward(self, x):
        return self.weight(x).squeeze(-1).sum(dim=1) + self.bias


def fit_wide(x, y, dates, cardinalities, epochs=4):
    torch.manual_seed(SEED + 20)
    rng = np.random.default_rng(SEED + 21)

    model = WideHazard(cardinalities)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.018, weight_decay=2e-6
    )

    tx = torch.from_numpy(x)
    ty = torch.from_numpy(np.asarray(y, dtype=np.float32))
    weights = recency_weights(dates, 4.0)
    batch_size = 16384

    for _ in range(epochs):
        order = rng.permutation(len(y))
        model.train()
        for begin in range(0, len(order), batch_size):
            idx_np = order[begin:begin + batch_size]
            idx = torch.from_numpy(idx_np)
            logits = model(tx[idx])
            target = ty[idx]
            row_weight = torch.from_numpy(weights[idx_np])
            loss = (
                F.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                ) * row_weight
            ).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


def predict_wide(model, x, batch_size=32768):
    tx = torch.from_numpy(x)
    result = np.empty(len(x), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(x), batch_size):
            end = min(begin + batch_size, len(x))
            result[begin:end] = model(tx[begin:end]).cpu().numpy()
    return result


def within_user_ranks(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)
    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    sorted_users = user_ids[order]

    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(first)
    group_id = np.cumsum(first) - 1
    local = np.arange(n, dtype=np.int64) - starts[group_id]
    sizes = np.diff(np.r_[starts, n])

    rank = (
        local.astype(np.float64) + 0.5
    ) / sizes[group_id].astype(np.float64)

    result = np.empty(n, dtype=np.float64)
    result[order] = rank
    return result


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

# Construct validation features with all chronologically prior train
# impressions available, while never using train or validation outcomes.
seq_train, seq_valid = sequence_features([train, valid])

x_train_lgb = lgb_matrix(train, seq_train)
x_valid_lgb = lgb_matrix(valid, seq_valid)

raw_predictions = {}
metadata = {}

gbdt = fit_sequence_gbdt(
    x_train_lgb, y_train, np.asarray(train.date), rounds=220
)
raw_predictions["exposure_state_gbdt"] = gbdt.predict(
    x_valid_lgb, num_iteration=gbdt.current_iteration()
).astype(np.float32)
metadata["exposure_state_gbdt"] = {"kind": "gbdt", "rounds": 220}
del gbdt
gc.collect()

x_train_wide, wide_cards = wide_matrix(train, seq_train)
x_valid_wide, _ = wide_matrix(valid, seq_valid)

wide = fit_wide(
    x_train_wide,
    y_train,
    np.asarray(train.date),
    wide_cards,
    epochs=4,
)
raw_predictions["wide_additive_hazard"] = predict_wide(
    wide, x_valid_wide
)
metadata["wide_additive_hazard"] = {
    "kind": "wide",
    "epochs": 4,
}
del wide
gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)

inc_rank = within_user_ranks(valid.user_id, inc_valid)
candidates = {}
records = {}

for family, raw in raw_predictions.items():
    raw = np.asarray(raw, dtype=np.float64)
    raw_metrics = evaluate(valid.user_id, y_valid, raw)
    candidates[family + "_raw"] = float(raw_metrics["primary"])
    raw_rank = within_user_ranks(valid.user_id, raw)

    for alpha in (0.15, 0.30, 0.50, 0.70, 1.00):
        scores = (1.0 - alpha) * inc_rank + alpha * raw_rank
        metrics = evaluate(valid.user_id, y_valid, scores)
        key = family + "_blend_" + str(alpha)
        candidates[key] = float(metrics["primary"])
        records[key] = {
            "family": family,
            "alpha": float(alpha),
            "scores": scores,
            "raw": raw,
            "metrics": metrics,
        }

winner_name = max(
    records, key=lambda k: records[k]["metrics"]["primary"]
)
winner = records[winner_name]

repeat_valid = np.expm1(seq_valid[:, 3]) > 0
author_repeat_valid = np.expm1(seq_valid[:, 4]) > 0

print("CANDIDATES " + json.dumps(candidates, sort_keys=True))
print("FINDINGS " + json.dumps({
    "winner": winner_name,
    "winner_family": winner["family"],
    "winner_alpha": winner["alpha"],
    "incumbent_primary": float(
        evaluate(valid.user_id, y_valid, inc_valid)["primary"]
    ),
    "video_repeat_share_valid": float(np.mean(repeat_valid)),
    "author_repeat_share_valid": float(np.mean(author_repeat_valid)),
    "wide_raw_primary": candidates["wide_additive_hazard_raw"],
    "gbdt_raw_primary": candidates["exposure_state_gbdt_raw"],
}, sort_keys=True))

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(winner["scores"], dtype=np.float64),
    )
    np.save(
        os.path.join(out_dir, "scores_valid_raw.npy"),
        np.asarray(winner["raw"], dtype=np.float64),
    )

# Refit the selected recipe on train+validation, then form test features
# using all earlier exposure logs. Test labels are never accessed.
test = load("test")
seq_train_all, seq_valid_all, seq_test = sequence_features(
    [train, valid, test]
)

y_combined = np.concatenate([
    y_train,
    y_valid.astype(np.float32, copy=False),
])
dates_combined = np.concatenate([
    np.asarray(train.date),
    np.asarray(valid.date),
])

selected_family = winner["family"]

if metadata[selected_family]["kind"] == "gbdt":
    x_train_all = lgb_matrix(train, seq_train_all)
    x_valid_all = lgb_matrix(valid, seq_valid_all)
    x_combined = np.concatenate([x_train_all, x_valid_all], axis=0)
    x_test = lgb_matrix(test, seq_test)

    final_model = fit_sequence_gbdt(
        x_combined,
        y_combined,
        dates_combined,
        rounds=metadata[selected_family]["rounds"],
    )
    raw_test = final_model.predict(
        x_test, num_iteration=final_model.current_iteration()
    ).astype(np.float64)
else:
    x_train_all, final_cards = wide_matrix(train, seq_train_all)
    x_valid_all, _ = wide_matrix(valid, seq_valid_all)
    x_combined = np.concatenate([x_train_all, x_valid_all], axis=0)
    x_test, _ = wide_matrix(test, seq_test)

    final_model = fit_wide(
        x_combined,
        y_combined,
        dates_combined,
        final_cards,
        epochs=metadata[selected_family]["epochs"],
    )
    raw_test = predict_wide(final_model, x_test).astype(np.float64)

inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

test_scores = (
    (1.0 - winner["alpha"])
    * within_user_ranks(test.user_id, inc_test)
    + winner["alpha"]
    * within_user_ranks(test.user_id, raw_test)
)

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

metrics = winner["metrics"]
elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))