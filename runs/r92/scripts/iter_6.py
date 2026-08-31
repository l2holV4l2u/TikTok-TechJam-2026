import os
import time
import json
import math
import gc
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 1847
THREADS = max(1, min(8, os.cpu_count() or 1))

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)


CAT_FIELDS = [
    "user_id",
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
    "fans_user_num_range",
    "follow_user_num_range",
    "friend_user_num_range",
    "register_days_bucket",
    "hour",
    "is_live_streamer",
    "is_video_author",
]

NUM_FIELDS = [
    "duration_ms",
    "user_fans_user_num",
    "user_follow_user_num",
    "user_friend_user_num",
    "user_register_days",
]

CAT_INDICES = list(range(len(CAT_FIELDS)))


def make_matrix(split):
    n = len(split.user_id)
    x = np.empty((n, len(CAT_FIELDS) + len(NUM_FIELDS)), dtype=np.float32)

    for j, name in enumerate(CAT_FIELDS):
        x[:, j] = split.X[name].astype(np.float32, copy=False)

    for k, name in enumerate(NUM_FIELDS):
        z = np.asarray(split.num[name], dtype=np.float32)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        z = np.log1p(np.maximum(z, 0.0))
        x[:, len(CAT_FIELDS) + k] = z

    return x


def recency_weights(dates, half_life):
    if half_life is None:
        return np.ones(len(dates), dtype=np.float32)
    age = np.max(dates).astype(np.int64) - dates.astype(np.int64)
    # YYYYMMDD subtraction is safe inside these short, single-month windows.
    w = np.exp2(-age.astype(np.float32) / float(half_life))
    return (w / np.mean(w)).astype(np.float32)


def within_user_ranks(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)
    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids
    ))
    su = user_ids[order]

    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = su[1:] != su[:-1]
    starts = np.flatnonzero(first)
    group_number = np.cumsum(first) - 1
    local = np.arange(n, dtype=np.int64) - starts[group_number]
    sizes = np.diff(np.r_[starts, n])

    ranked = (
        local.astype(np.float64) + 0.5
    ) / sizes[group_number].astype(np.float64)

    out = np.empty(n, dtype=np.float64)
    out[order] = ranked
    return out


def grouped_order_and_sizes(users):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    first = np.empty(len(users), dtype=bool)
    first[0] = True
    first[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(first)
    sizes = np.diff(np.r_[starts, len(users)]).astype(np.int32)
    return order, sizes


def fit_binary_gbdt(x, y, dates, half_life, rounds=180):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.065,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 120,
        "max_bin": 127,
        "feature_fraction": 0.88,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 1.5,
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
        weight=recency_weights(dates, half_life),
        categorical_feature=CAT_INDICES,
        free_raw_data=False,
    )
    return lgb.train(params, dset, num_boost_round=rounds)


def fit_lambdarank(x, y, users, dates, half_life=4.0, rounds=170):
    order, sizes = grouped_order_and_sizes(users)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_at": [5],
        "label_gain": [0, 1],
        "lambdarank_truncation_level": 8,
        "learning_rate": 0.055,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 100,
        "max_bin": 127,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.92,
        "bagging_freq": 1,
        "lambda_l1": 0.03,
        "lambda_l2": 1.5,
        "min_gain_to_split": 0.001,
        "seed": SEED + 10,
        "feature_fraction_seed": SEED + 11,
        "bagging_seed": SEED + 12,
        "num_threads": THREADS,
        "force_col_wise": True,
        "verbose": -1,
    }
    weights = recency_weights(dates, half_life)
    dset = lgb.Dataset(
        x[order],
        label=y[order],
        weight=weights[order],
        group=sizes,
        categorical_feature=CAT_INDICES,
        free_raw_data=False,
    )
    return lgb.train(params, dset, num_boost_round=rounds)


class BPRModel(nn.Module):
    def __init__(self, rank=24):
        super().__init__()
        nu = int(FEATURE_CARDINALITIES["user_id"])
        nv = int(FEATURE_CARDINALITIES["video_id"])
        na = int(FEATURE_CARDINALITIES["author_id"])
        nt = int(FEATURE_CARDINALITIES["tag"])
        nd = int(FEATURE_CARDINALITIES["duration_bucket"])

        self.user = nn.Embedding(nu, rank)
        self.video = nn.Embedding(nv, rank)
        self.author = nn.Embedding(na, rank)

        self.video_bias = nn.Embedding(nv, 1)
        self.author_bias = nn.Embedding(na, 1)
        self.tag_bias = nn.Embedding(nt, 1)
        self.duration_bias = nn.Embedding(nd, 1)

        with torch.no_grad():
            self.user.weight.normal_(0.0, 0.04)
            self.video.weight.normal_(0.0, 0.04)
            self.author.weight.normal_(0.0, 0.04)
            self.video_bias.weight.zero_()
            self.author_bias.weight.zero_()
            self.tag_bias.weight.zero_()
            self.duration_bias.weight.zero_()

    def forward(self, u, v, a, tag, duration):
        uv = (self.user(u) * self.video(v)).sum(dim=1)
        ua = (self.user(u) * self.author(a)).sum(dim=1)
        return (
            uv
            + 0.60 * ua
            + self.video_bias(v).squeeze(1)
            + 0.65 * self.author_bias(a).squeeze(1)
            + 0.25 * self.tag_bias(tag).squeeze(1)
            + 0.20 * self.duration_bias(duration).squeeze(1)
        )


def bpr_arrays(split):
    return (
        np.asarray(split.X["user_id"], dtype=np.int64),
        np.asarray(split.X["video_id"], dtype=np.int64),
        np.asarray(split.X["author_id"], dtype=np.int64),
        np.asarray(split.X["tag"], dtype=np.int64),
        np.asarray(split.X["duration_bucket"], dtype=np.int64),
    )


def fit_bpr(arrays, y, dates, epochs=3, half_life=4.0):
    torch.manual_seed(SEED + 30)
    rng = np.random.default_rng(SEED + 31)
    model = BPRModel(rank=24)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.010, weight_decay=2e-6
    )

    users = arrays[0]
    order = np.argsort(users, kind="mergesort")
    sorted_arrays = tuple(a[order] for a in arrays)
    sy = y[order].astype(np.int8, copy=False)
    sdates = dates[order]

    su = sorted_arrays[0]
    first = np.empty(len(su), dtype=bool)
    first[0] = True
    first[1:] = su[1:] != su[:-1]
    starts = np.flatnonzero(first)
    sizes = np.diff(np.r_[starts, len(su)])
    row_start = np.repeat(starts, sizes)
    row_size = np.repeat(sizes, sizes)

    positive_rows = np.flatnonzero(sy == 1)
    tensors = tuple(torch.from_numpy(a) for a in sorted_arrays)
    date_weights = recency_weights(sdates, half_life)
    batch_size = 8192

    for _ in range(epochs):
        sampled_offset = (
            rng.random(len(positive_rows)) * row_size[positive_rows]
        ).astype(np.int64)
        negative_rows = row_start[positive_rows] + sampled_offset
        keep = sy[negative_rows] == 0
        pos = positive_rows[keep]
        neg = negative_rows[keep]

        shuffle = rng.permutation(len(pos))
        pos = pos[shuffle]
        neg = neg[shuffle]

        model.train()
        for begin in range(0, len(pos), batch_size):
            end = min(begin + batch_size, len(pos))
            pi = torch.from_numpy(pos[begin:end])
            ni = torch.from_numpy(neg[begin:end])

            pos_score = model(*(x[pi] for x in tensors))
            neg_score = model(*(x[ni] for x in tensors))
            weight = torch.from_numpy(
                date_weights[pos[begin:end]].astype(np.float32, copy=False)
            )
            loss = (F.softplus(-(pos_score - neg_score)) * weight).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


def predict_bpr(model, arrays, batch_size=32768):
    tensors = tuple(torch.from_numpy(a) for a in arrays)
    result = np.empty(len(arrays[0]), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(result), batch_size):
            end = min(begin + batch_size, len(result))
            result[begin:end] = model(
                *(x[begin:end] for x in tensors)
            ).cpu().numpy()
    return result


def score_candidates(raw_predictions, valid, y_valid, incumbent):
    inc_rank = within_user_ranks(valid.user_id, incumbent)
    candidates = {}
    records = {}

    for name, raw in raw_predictions.items():
        raw = np.asarray(raw, dtype=np.float64)
        raw_metric = evaluate(valid.user_id, y_valid, raw)
        candidates[name + "_raw"] = float(raw_metric["primary"])

        raw_rank = within_user_ranks(valid.user_id, raw)
        for alpha in (0.15, 0.30, 0.50, 0.70):
            blend = (1.0 - alpha) * inc_rank + alpha * raw_rank
            metric = evaluate(valid.user_id, y_valid, blend)
            key = name + "_blend_" + str(alpha)
            candidates[key] = float(metric["primary"])
            records[key] = {
                "family": name,
                "alpha": float(alpha),
                "scores": blend,
                "raw": raw,
                "metrics": metric,
            }

    winner = max(records, key=lambda k: records[k]["metrics"]["primary"])
    return candidates, winner, records[winner]


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)

x_train = make_matrix(train)
x_valid = make_matrix(valid)

raw_predictions = {}
model_metadata = {}

# Family 1: pointwise nonlinear tabular prediction, with the main model
# itself trained under several temporal half-lives.
for half_life in (None, 7.0, 3.0):
    name = "binary_gbdt_uniform" if half_life is None else (
        "binary_gbdt_hl" + str(int(half_life))
    )
    booster = fit_binary_gbdt(
        x_train, y_train, train.date, half_life, rounds=180
    )
    raw_predictions[name] = booster.predict(
        x_valid, num_iteration=booster.current_iteration()
    ).astype(np.float32)
    model_metadata[name] = {
        "kind": "binary",
        "half_life": half_life,
        "rounds": 180,
    }
    del booster
    gc.collect()

# Family 2: directly user-grouped LambdaRank.
rank_booster = fit_lambdarank(
    x_train,
    y_train,
    np.asarray(train.user_id),
    train.date,
    half_life=4.0,
    rounds=170,
)
raw_predictions["lambdarank_hl4"] = rank_booster.predict(
    x_valid, num_iteration=rank_booster.current_iteration()
).astype(np.float32)
model_metadata["lambdarank_hl4"] = {
    "kind": "rank",
    "half_life": 4.0,
    "rounds": 170,
}
del rank_booster
gc.collect()

# Family 3: latent collaborative BPR, whose score is formed from learned
# user-video and user-author order preferences rather than tree partitions.
train_bpr_arrays = bpr_arrays(train)
valid_bpr_arrays = bpr_arrays(valid)
bpr_model = fit_bpr(
    train_bpr_arrays,
    y_train,
    np.asarray(train.date),
    epochs=3,
    half_life=4.0,
)
raw_predictions["latent_bpr_hl4"] = predict_bpr(
    bpr_model, valid_bpr_arrays
)
model_metadata["latent_bpr_hl4"] = {
    "kind": "bpr",
    "half_life": 4.0,
    "epochs": 3,
}
del bpr_model
gc.collect()

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)

candidates, winner_name, winner = score_candidates(
    raw_predictions, valid, y_valid, inc_valid
)

print("CANDIDATES " + json.dumps(candidates, sort_keys=True))
print("FINDINGS " + json.dumps({
    "winner": winner_name,
    "winner_family": winner["family"],
    "winner_alpha": winner["alpha"],
    "winner_raw_primary": candidates[winner["family"] + "_raw"],
    "incumbent_primary": float(
        evaluate(valid.user_id, y_valid, inc_valid)["primary"]
    ),
    "pointwise_recency_delta_hl3_vs_uniform": float(
        candidates["binary_gbdt_hl3_raw"]
        - candidates["binary_gbdt_uniform_raw"]
    ),
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(winner["scores"], dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(winner["raw"], dtype=np.float64),
    )

# Refit the selected recipe on train + validation. No test labels are read.
test = load("test")
x_test = make_matrix(test)

x_combined = np.concatenate([x_train, x_valid], axis=0)
y_combined = np.concatenate([
    y_train,
    y_valid.astype(np.float32, copy=False),
])
date_combined = np.concatenate([
    np.asarray(train.date),
    np.asarray(valid.date),
])
user_combined = np.concatenate([
    np.asarray(train.user_id),
    np.asarray(valid.user_id),
])

selected_family = winner["family"]
meta = model_metadata[selected_family]

if meta["kind"] == "binary":
    refit = fit_binary_gbdt(
        x_combined,
        y_combined,
        date_combined,
        meta["half_life"],
        rounds=meta["rounds"],
    )
    raw_test = refit.predict(
        x_test, num_iteration=refit.current_iteration()
    ).astype(np.float32)

elif meta["kind"] == "rank":
    refit = fit_lambdarank(
        x_combined,
        y_combined,
        user_combined,
        date_combined,
        half_life=meta["half_life"],
        rounds=meta["rounds"],
    )
    raw_test = refit.predict(
        x_test, num_iteration=refit.current_iteration()
    ).astype(np.float32)

else:
    combined_bpr_arrays = tuple(
        np.concatenate([a, b])
        for a, b in zip(train_bpr_arrays, valid_bpr_arrays)
    )
    refit = fit_bpr(
        combined_bpr_arrays,
        y_combined,
        date_combined,
        epochs=meta["epochs"],
        half_life=meta["half_life"],
    )
    raw_test = predict_bpr(refit, bpr_arrays(test))

inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

test_scores = (
    (1.0 - winner["alpha"]) * within_user_ranks(test.user_id, inc_test)
    + winner["alpha"] * within_user_ranks(test.user_id, raw_test)
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
metrics = winner["metrics"]
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))