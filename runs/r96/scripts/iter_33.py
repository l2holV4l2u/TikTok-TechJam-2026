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
SEED = 73421
THREADS = min(16, os.cpu_count() or 1)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(THREADS)

LATENT_FIELDS = [
    "user_id", "video_id", "author_id", "tag", "duration_bucket",
    "tab", "upload_type", "music_type",
]
PROXY_FIELDS = [
    "user_id", "video_id", "author_id", "tag", "duration_bucket", "tab",
    "hour", "upload_type", "music_type", "user_active_degree",
    "fans_user_num_range", "follow_user_num_range",
    "friend_user_num_range", "register_days_range",
    "is_live_streamer", "is_video_author", "onehot_feat1",
    "onehot_feat2", "onehot_feat3", "onehot_feat7", "onehot_feat8",
]
PROFILE_FIELDS = [
    "user_active_degree", "fans_user_num_range", "follow_user_num_range",
    "friend_user_num_range", "register_days_range", "is_live_streamer",
    "is_video_author", "onehot_feat0", "onehot_feat1", "onehot_feat2",
]


def recency_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.int32)
    age = dates.max() - dates
    w = np.power(0.5, age.astype(np.float32) / float(half_life))
    return (w / np.maximum(w.mean(), 1e-8)).astype(np.float32)


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    su = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = su[1:] != su[:-1]
    group_start = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = su[:-1] != su[1:]
    end_pos = np.flatnonzero(ends)
    sizes = np.diff(np.concatenate((np.array([-1]), end_pos)))
    row_sizes = np.repeat(sizes, sizes)

    position = np.arange(n, dtype=np.int64) - group_start
    ranked = (position.astype(np.float64) + 0.5) / row_sizes
    out = np.empty(n, dtype=np.float64)
    out[order] = ranked
    return out


class PersonalizedLatent(nn.Module):
    def __init__(self, cardinalities, dim=40):
        super().__init__()
        self.user = nn.Embedding(cardinalities["user_id"], dim)
        self.video = nn.Embedding(cardinalities["video_id"], dim)
        self.author = nn.Embedding(cardinalities["author_id"], dim)
        self.tag = nn.Embedding(cardinalities["tag"], dim)
        self.duration = nn.Embedding(cardinalities["duration_bucket"], dim)
        self.tab = nn.Embedding(cardinalities["tab"], dim)

        self.user_bias = nn.Embedding(cardinalities["user_id"], 1)
        self.video_bias = nn.Embedding(cardinalities["video_id"], 1)
        self.author_bias = nn.Embedding(cardinalities["author_id"], 1)
        self.tag_bias = nn.Embedding(cardinalities["tag"], 1)
        self.duration_bias = nn.Embedding(
            cardinalities["duration_bucket"], 1
        )
        self.upload_bias = nn.Embedding(cardinalities["upload_type"], 1)
        self.music_bias = nn.Embedding(cardinalities["music_type"], 1)
        self.global_bias = nn.Parameter(torch.zeros(()))

        for module in self.modules():
            if isinstance(module, nn.Embedding):
                if module.embedding_dim == 1:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.normal_(module.weight, std=0.035)

    def forward(self, x):
        u = self.user(x[:, 0])
        content = (
            self.video(x[:, 1])
            + 0.70 * self.author(x[:, 2])
            + 0.45 * self.tag(x[:, 3])
            + 0.30 * self.duration(x[:, 4])
            + 0.20 * self.tab(x[:, 5])
        )
        interaction = torch.sum(u * content, dim=1) / np.sqrt(
            float(u.shape[1])
        )
        bias = (
            self.user_bias(x[:, 0]).squeeze(1)
            + self.video_bias(x[:, 1]).squeeze(1)
            + self.author_bias(x[:, 2]).squeeze(1)
            + self.tag_bias(x[:, 3]).squeeze(1)
            + self.duration_bias(x[:, 4]).squeeze(1)
            + self.upload_bias(x[:, 6]).squeeze(1)
            + self.music_bias(x[:, 7]).squeeze(1)
        )
        return self.global_bias + interaction + bias


def latent_matrix(split):
    return np.ascontiguousarray(
        np.stack([
            np.asarray(split.X[f], dtype=np.int64)
            for f in LATENT_FIELDS
        ], axis=1),
        dtype=np.int64,
    )


def fit_latent(x, y, dates, indices, epochs, seed):
    torch.manual_seed(seed)
    model = PersonalizedLatent(FEATURE_CARDINALITIES, dim=40)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.006, weight_decay=2e-5
    )

    indices = np.asarray(indices, dtype=np.int64)
    local_dates = np.asarray(dates[indices], dtype=np.int32)
    weights = recency_weights(local_dates, half_life=4.0)
    batch_size = 16384
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        permutation = rng.permutation(len(indices))
        for start in range(0, len(indices), batch_size):
            local = permutation[start:start + batch_size]
            rows = indices[local]
            xb = torch.from_numpy(x[rows])
            yb = torch.from_numpy(y[rows].astype(np.float32, copy=False))
            wb = torch.from_numpy(weights[local])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = (
                F.binary_cross_entropy_with_logits(
                    logits, yb, reduction="none"
                ) * wb
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def predict_latent(model, x):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    batch_size = 32768
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.from_numpy(x[start:end])
            result[start:end] = model(xb).numpy()
    return result


def proxy_matrix(split):
    cols = [
        np.asarray(split.X[f], dtype=np.float32)
        for f in PROXY_FIELDS
    ]
    for name in (
        "duration_ms", "user_fans_user_num", "user_follow_user_num",
        "user_friend_user_num", "user_register_days",
    ):
        v = np.asarray(split.num[name], dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(np.log1p(np.maximum(v, 0.0)).astype(np.float32))
    return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.float32)


def fit_proxy(x, y, dates, indices):
    w = recency_weights(dates[indices], half_life=4.0)
    dataset = lgb.Dataset(
        x[indices],
        label=y[indices],
        weight=w,
        categorical_feature=list(range(len(PROXY_FIELDS))),
        free_raw_data=True,
    )
    params = {
        "objective": "binary",
        "metric": "None",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 180,
        "feature_fraction": 0.88,
        "bagging_fraction": 0.88,
        "bagging_freq": 1,
        "lambda_l2": 1.5,
        "max_bin": 127,
        "num_threads": THREADS,
        "seed": SEED + 11,
        "verbose": -1,
    }
    return lgb.train(params, dataset, num_boost_round=260)


def user_history_arrays(user_ids, labels, dates, indices):
    card = FEATURE_CARDINALITIES["user_id"]
    uid = np.asarray(user_ids[indices], dtype=np.int64)
    yy = np.asarray(labels[indices], dtype=np.float64)

    count = np.bincount(uid, minlength=card).astype(np.float32)
    positive = np.bincount(
        uid, weights=yy, minlength=card
    ).astype(np.float32)

    last_date = np.zeros(card, dtype=np.int32)
    np.maximum.at(last_date, uid, np.asarray(dates[indices], dtype=np.int32))

    rate = (positive + 4.0 * float(yy.mean())) / (count + 4.0)
    return count, positive, rate.astype(np.float32), last_date


def user_group_data(split, score_a, score_b, history, labels=None):
    uid = np.asarray(split.user_id, dtype=np.int64)
    score_a = np.asarray(score_a, dtype=np.float64)
    score_b = np.asarray(score_b, dtype=np.float64)
    rank_a = rank_percentile(uid, score_a)
    rank_b = rank_percentile(uid, score_b)

    order = np.argsort(uid, kind="stable")
    su = uid[order]
    unique_users, starts, counts = np.unique(
        su, return_index=True, return_counts=True
    )
    inverse = np.searchsorted(unique_users, uid)

    hist_count, hist_positive, hist_rate, last_date = history
    features = np.zeros(
        (len(unique_users), 12 + len(PROFILE_FIELDS)), dtype=np.float32
    )

    delta = None
    if labels is not None:
        delta = np.zeros(len(unique_users), dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int8)

    dates = np.asarray(split.date, dtype=np.int32)

    for j, (start, count) in enumerate(zip(starts, counts)):
        rows = order[start:start + count]
        u = unique_users[j]
        a = rank_a[rows]
        b = rank_b[rows]
        disagreement = a - b

        if count > 1 and np.std(a) > 1e-9 and np.std(b) > 1e-9:
            corr = float(np.corrcoef(a, b)[0, 1])
        else:
            corr = 0.0

        features[j, 0] = np.log1p(hist_count[u])
        features[j, 1] = hist_rate[u]
        features[j, 2] = np.log1p(hist_positive[u])
        features[j, 3] = max(0, int(dates[rows[0]]) - int(last_date[u]))
        features[j, 4] = np.log1p(count)
        features[j, 5] = float(np.mean(np.abs(disagreement)))
        features[j, 6] = float(np.max(np.abs(disagreement)))
        features[j, 7] = float(np.std(disagreement))
        features[j, 8] = float(np.std(a))
        features[j, 9] = float(np.std(b))
        features[j, 10] = float(np.mean(score_a[rows]))
        features[j, 11] = float(np.mean(score_b[rows]))

        first = rows[0]
        for k, field in enumerate(PROFILE_FIELDS):
            features[j, 12 + k] = float(split.X[field][first])

        if labels is not None:
            rel = labels[rows]
            positives = int(rel.sum())
            if positives == 0:
                delta[j] = 0.0
            else:
                ideal_k = min(5, positives)
                ideal = np.sum(
                    1.0 / np.log2(np.arange(2, ideal_k + 2))
                )

                oa = np.argsort(-score_a[rows], kind="stable")[:5]
                ob = np.argsort(-score_b[rows], kind="stable")[:5]
                dcg_a = np.sum(
                    rel[oa] / np.log2(np.arange(2, len(oa) + 2))
                )
                dcg_b = np.sum(
                    rel[ob] / np.log2(np.arange(2, len(ob) + 2))
                )
                delta[j] = float((dcg_a - dcg_b) / ideal)

    return unique_users, inverse, features, delta, rank_a, rank_b


def fit_gate(features, delta, counts):
    useful = (counts >= 2) & (np.abs(delta) > 1e-6)
    x = features[useful]
    target = (delta[useful] > 0).astype(np.float32)
    weight = (0.05 + np.abs(delta[useful])).astype(np.float32)

    dataset = lgb.Dataset(
        x, label=target, weight=weight, free_raw_data=True
    )
    params = {
        "objective": "binary",
        "metric": "None",
        "learning_rate": 0.04,
        "num_leaves": 15,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 4.0,
        "num_threads": THREADS,
        "seed": SEED + 99,
        "verbose": -1,
    }
    model = lgb.train(params, dataset, num_boost_round=140)
    return model, useful


train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float32)
dates = np.asarray(train.date, dtype=np.int32)
latent_train = latent_matrix(train)
latent_valid = latent_matrix(valid)
latent_test = latent_matrix(test)

holdout_start = 20220419
base_idx = np.flatnonzero(dates < holdout_start)
hold_idx = np.flatnonzero(dates >= holdout_start)

# Build genuinely pseudo-future predictions for learning the selector.
early_latent = fit_latent(
    latent_train, y_train, dates, base_idx, epochs=4, seed=SEED
)
hold_latent = predict_latent(early_latent, latent_train[hold_idx])

proxy_x = proxy_matrix(train)
early_proxy = fit_proxy(proxy_x, y_train, dates, base_idx)
hold_proxy = early_proxy.predict(proxy_x[hold_idx])

history_base = user_history_arrays(
    np.asarray(train.user_id), y_train, dates, base_idx
)

class HoldoutView:
    pass


hold = HoldoutView()
hold.user_id = np.asarray(train.user_id)[hold_idx]
hold.date = np.asarray(train.date)[hold_idx]
hold.X = {
    field: np.asarray(train.X[field])[hold_idx]
    for field in PROFILE_FIELDS
}

(
    hold_users,
    hold_inverse,
    hold_features,
    hold_delta,
    hold_latent_rank,
    hold_proxy_rank,
) = user_group_data(
    hold,
    hold_latent,
    hold_proxy,
    history_base,
    labels=y_train[hold_idx].astype(np.int8),
)

hold_counts = np.bincount(
    hold_inverse, minlength=len(hold_users)
).astype(np.int32)
gate, useful_gate_users = fit_gate(
    hold_features, hold_delta, hold_counts
)

gate_train_accuracy = float(np.mean(
    (
        gate.predict(hold_features[useful_gate_users]) >= 0.5
    ) == (
        hold_delta[useful_gate_users] > 0
    )
))

del early_latent, early_proxy, proxy_x
gc.collect()

# Refit the complementary latent expert on every permitted training row.
all_idx = np.arange(len(y_train), dtype=np.int64)
full_latent = fit_latent(
    latent_train, y_train, dates, all_idx, epochs=5, seed=SEED + 200
)
latent_valid_score = predict_latent(full_latent, latent_valid)
latent_test_score = predict_latent(full_latent, latent_test)

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

history_full = user_history_arrays(
    np.asarray(train.user_id), y_train, dates, all_idx
)

(
    valid_users,
    valid_inverse,
    valid_gate_features,
    _,
    valid_latent_rank,
    valid_inc_rank,
) = user_group_data(
    valid,
    latent_valid_score,
    inc_valid,
    history_full,
    labels=None,
)
(
    test_users,
    test_inverse,
    test_gate_features,
    _,
    test_latent_rank,
    test_inc_rank,
) = user_group_data(
    test,
    latent_test_score,
    inc_test,
    history_full,
    labels=None,
)

valid_gate_probability = np.clip(
    gate.predict(valid_gate_features), 0.0, 1.0
)
test_gate_probability = np.clip(
    gate.predict(test_gate_features), 0.0, 1.0
)

candidate_valid = {
    "trusted_incumbent": inc_valid,
    "latent_standalone": latent_valid_score.astype(np.float64),
}
candidate_test = {
    "trusted_incumbent": inc_test,
    "latent_standalone": latent_test_score.astype(np.float64),
}

# Global mixtures provide a direct control for whether personalization of the
# blend, rather than merely diversity, is responsible for any improvement.
for alpha in (0.10, 0.20, 0.30, 0.40, 0.50):
    name = f"global_latent_blend_{alpha:.2f}"
    candidate_valid[name] = (
        alpha * valid_latent_rank + (1.0 - alpha) * valid_inc_rank
    )
    candidate_test[name] = (
        alpha * test_latent_rank + (1.0 - alpha) * test_inc_rank
    )

for scale in (0.25, 0.50, 0.75, 1.00):
    vg = np.clip(scale * valid_gate_probability, 0.0, 1.0)
    tg = np.clip(scale * test_gate_probability, 0.0, 1.0)
    name = f"oof_user_gate_soft_{scale:.2f}"
    candidate_valid[name] = (
        vg[valid_inverse] * valid_latent_rank
        + (1.0 - vg[valid_inverse]) * valid_inc_rank
    )
    candidate_test[name] = (
        tg[test_inverse] * test_latent_rank
        + (1.0 - tg[test_inverse]) * test_inc_rank
    )

for threshold in (0.55, 0.65, 0.75):
    vg = (valid_gate_probability >= threshold).astype(np.float64)
    tg = (test_gate_probability >= threshold).astype(np.float64)
    name = f"oof_user_gate_hard_{threshold:.2f}"
    candidate_valid[name] = (
        vg[valid_inverse] * valid_latent_rank
        + (1.0 - vg[valid_inverse]) * valid_inc_rank
    )
    candidate_test[name] = (
        tg[test_inverse] * test_latent_rank
        + (1.0 - tg[test_inverse]) * test_inc_rank
    )

candidate_metrics = {
    name: evaluate(valid.user_id, valid.y, score)
    for name, score in candidate_valid.items()
}
best_name = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"])
)
best_metrics = candidate_metrics[best_name]
best_valid = candidate_valid[best_name]
best_test = candidate_test[best_name]

print("CANDIDATES " + json.dumps({
    name: float(metrics["primary"])
    for name, metrics in candidate_metrics.items()
}, sort_keys=True))
print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "holdout_start": holdout_start,
    "holdout_users": int(len(hold_users)),
    "gate_training_users": int(useful_gate_users.sum()),
    "gate_in_sample_winner_accuracy": gate_train_accuracy,
    "valid_mean_latent_gate_probability":
        float(valid_gate_probability.mean()),
    "test_mean_latent_gate_probability":
        float(test_gate_probability.mean()),
    "valid_fraction_gate_above_0.5":
        float(np.mean(valid_gate_probability >= 0.5)),
    "holdout_mean_latent_minus_proxy_ndcg":
        float(np.mean(hold_delta)),
    "mechanism":
        "train-holdout user-level expert selection from support and disagreement",
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
    if best_name != "latent_standalone":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(latent_valid_score, dtype=np.float64),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))