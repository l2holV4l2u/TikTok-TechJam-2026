import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from scipy.sparse.linalg import svds

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7319
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 1))

FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "hour",
    "tag",
    "upload_type",
    "user_active_degree",
    "is_live_streamer",
    "is_video_author",
    "fans_user_num_range",
    "follow_user_num_range",
    "register_days_range",
]
EMBED_DIM = 10
USER_BATCH = 256
PRED_BATCH = 16384
EPOCHS = 4
LR = 0.002
HALF_LIFE = 4.0


def make_matrix(split):
    cards = np.asarray(
        [FEATURE_CARDINALITIES[name] for name in FIELDS], dtype=np.int64
    )
    offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(cards[:-1])]
    )
    columns = [
        np.asarray(split.X[name], dtype=np.int64) + offsets[i]
        for i, name in enumerate(FIELDS)
    ]
    return np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.int64)


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int32)
    age = (int(dates.max()) - dates).astype(np.float32)
    weights = np.power(0.5, age / HALF_LIFE).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def rank_percentile(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    ordered_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = ordered_users[1:] != ordered_users[:-1]
    group_start = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = ordered_users[:-1] != ordered_users[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(
        np.concatenate((np.asarray([-1], dtype=np.int64), end_positions))
    )
    sizes_per_row = np.repeat(sizes, sizes)

    position = np.arange(n, dtype=np.int64) - group_start
    ranked = (position.astype(np.float64) + 0.5) / sizes_per_row

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


class FieldWeightedFM(nn.Module):
    def __init__(self, num_features, num_fields, initial_bias):
        super().__init__()
        self.linear = nn.Embedding(num_features, 1)
        self.embedding = nn.Embedding(num_features, EMBED_DIM)

        pair_i, pair_j = np.triu_indices(num_fields, k=1)
        self.register_buffer(
            "pair_i", torch.from_numpy(pair_i.astype(np.int64))
        )
        self.register_buffer(
            "pair_j", torch.from_numpy(pair_j.astype(np.int64))
        )
        # A separate diagonal bilinear transform for every field pair.
        self.pair_weights = nn.Parameter(
            torch.ones(len(pair_i), EMBED_DIM, dtype=torch.float32)
        )
        self.bias = nn.Parameter(
            torch.tensor(initial_bias, dtype=torch.float32)
        )

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.025)
        nn.init.normal_(self.pair_weights, mean=1.0, std=0.03)

    def forward(self, x):
        emb = self.embedding(x)
        left = emb[:, self.pair_i, :]
        right = emb[:, self.pair_j, :]
        interactions = (
            left * right * self.pair_weights.unsqueeze(0)
        ).sum(dim=(1, 2))
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        return self.bias + linear + interactions


def build_user_groups(user_ids):
    order = np.argsort(np.asarray(user_ids, dtype=np.int64), kind="stable")
    sorted_users = np.asarray(user_ids, dtype=np.int64)[order]
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1,
        )
    )
    ends = np.concatenate(
        (starts[1:], np.asarray([len(order)], dtype=np.int64))
    )
    return order, starts, ends


def segment_listwise_loss(logits, labels, weights, group_ids, n_groups):
    log_weights = torch.log(torch.clamp(weights, min=1e-6))
    adjusted = logits + log_weights

    maxima = torch.full(
        (n_groups,),
        -torch.inf,
        dtype=logits.dtype,
        device=logits.device,
    )
    maxima.scatter_reduce_(
        0, group_ids, adjusted, reduce="amax", include_self=True
    )

    exponentials = torch.exp(adjusted - maxima[group_ids])
    denominators = torch.zeros(
        n_groups, dtype=logits.dtype, device=logits.device
    )
    denominators.scatter_add_(0, group_ids, exponentials)
    log_norm = maxima + torch.log(torch.clamp(denominators, min=1e-12))

    positive_weight = labels * weights
    group_positive_weight = torch.zeros(
        n_groups, dtype=logits.dtype, device=logits.device
    )
    group_positive_weight.scatter_add_(0, group_ids, positive_weight)

    row_losses = positive_weight * (log_norm[group_ids] - logits)
    group_losses = torch.zeros(
        n_groups, dtype=logits.dtype, device=logits.device
    )
    group_losses.scatter_add_(0, group_ids, row_losses)

    valid_groups = group_positive_weight > 0
    if bool(valid_groups.any()):
        list_loss = (
            group_losses[valid_groups]
            / group_positive_weight[valid_groups].clamp_min(1e-8)
        ).mean()
    else:
        list_loss = logits.sum() * 0.0

    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    bce = (bce * weights).mean()
    return list_loss + 0.20 * bce


def train_listwise(model, x_train, y_train, row_weights, user_ids):
    order, starts, ends = build_user_groups(user_ids)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-6
    )
    rng = np.random.default_rng(SEED + 101)
    num_users = len(starts)

    model.train()
    for epoch in range(EPOCHS):
        user_permutation = rng.permutation(num_users)
        epoch_loss = 0.0
        epoch_batches = 0

        for batch_start in range(0, num_users, USER_BATCH):
            selected_users = user_permutation[
                batch_start:batch_start + USER_BATCH
            ]
            pieces = [
                order[starts[u]:ends[u]] for u in selected_users
            ]
            row_indices_np = np.concatenate(pieces)
            group_sizes = np.asarray(
                [ends[u] - starts[u] for u in selected_users],
                dtype=np.int64,
            )
            group_ids_np = np.repeat(
                np.arange(len(selected_users), dtype=np.int64),
                group_sizes,
            )

            row_indices = torch.from_numpy(row_indices_np)
            group_ids = torch.from_numpy(group_ids_np)
            logits = model(x_train[row_indices])
            loss = segment_listwise_loss(
                logits,
                y_train[row_indices],
                row_weights[row_indices],
                group_ids,
                len(selected_users),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += float(loss.detach())
            epoch_batches += 1

        print(
            "FINDINGS " + json.dumps({
                "listwise_epoch": epoch + 1,
                "mean_training_loss": epoch_loss / max(epoch_batches, 1),
            })
        )
    return model


def predict_torch(model, matrix):
    result = np.empty(len(matrix), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(matrix), PRED_BATCH):
            end = min(start + PRED_BATCH, len(matrix))
            xb = torch.from_numpy(matrix[start:end])
            result[start:end] = model(xb).cpu().numpy()
    return result


def history_score(split_name):
    component_scores = []
    count_scores = []
    used_keys = {}

    for entity in ("video_id", "author_id"):
        features = historical_features(split_name, key=entity)
        rate_keys = [
            key for key in features
            if key.endswith("_long_view_rate")
        ]
        count_keys = [
            key for key in features
            if key.endswith("_train_count_log1p")
        ]

        if not rate_keys:
            raise RuntimeError(
                "No long_view history rate available for " + entity
            )

        rate_key = rate_keys[0]
        rate = np.asarray(features[rate_key], dtype=np.float64)
        finite_rate = np.isfinite(rate)
        fallback = (
            float(np.nanmedian(rate[finite_rate]))
            if finite_rate.any() else 0.336
        )
        rate = np.nan_to_num(
            rate, nan=fallback, posinf=fallback, neginf=fallback
        )
        rate = np.clip(rate, 1e-4, 1.0 - 1e-4)
        component_scores.append(np.log(rate / (1.0 - rate)))

        if count_keys:
            count_key = count_keys[0]
            count = np.asarray(features[count_key], dtype=np.float64)
            count = np.nan_to_num(count, nan=0.0, posinf=0.0, neginf=0.0)
            count_scores.append(count)
        else:
            count_key = None

        used_keys[entity] = {
            "rate": rate_key,
            "count": count_key,
        }

    score = np.mean(np.stack(component_scores, axis=1), axis=1)
    if count_scores:
        score += 0.025 * np.mean(
            np.stack(count_scores, axis=1), axis=1
        )
    return score.astype(np.float64), used_keys


def fit_latent_svd(train_split, weights, rank=24):
    users = np.asarray(train_split.user_id, dtype=np.int64)
    videos = np.asarray(train_split.video_id, dtype=np.int64)
    labels = np.asarray(train_split.y, dtype=np.float64)

    n_users = FEATURE_CARDINALITIES["user_id"]
    n_videos = FEATURE_CARDINALITIES["video_id"]

    # Center outcomes by day, suppressing the changing daily intercept while
    # retaining user-video preference structure. Recency weights emphasize
    # factors supported close to the deployment boundary.
    centered = np.empty(len(labels), dtype=np.float64)
    dates = np.asarray(train_split.date, dtype=np.int32)
    for date in np.unique(dates):
        mask = dates == date
        centered[mask] = labels[mask] - float(labels[mask].mean())

    values = centered * np.asarray(weights, dtype=np.float64)
    matrix = sparse.coo_matrix(
        (values, (users, videos)),
        shape=(n_users, n_videos),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()

    u, singular_values, vt = svds(
        matrix,
        k=rank,
        which="LM",
        tol=5e-3,
        maxiter=400,
        random_state=SEED,
    )
    ordering = np.argsort(singular_values)[::-1]
    singular_values = singular_values[ordering]
    u = u[:, ordering]
    vt = vt[ordering, :]

    user_factors = u * np.sqrt(singular_values)[None, :]
    video_factors = vt.T * np.sqrt(singular_values)[None, :]
    return user_factors.astype(np.float32), video_factors.astype(np.float32)


def latent_predict(split, user_factors, video_factors):
    users = np.asarray(split.user_id, dtype=np.int64)
    videos = np.asarray(split.video_id, dtype=np.int64)
    result = (
        user_factors[users] * video_factors[videos]
    ).sum(axis=1)
    return np.asarray(result, dtype=np.float64)


train = load("train")
valid = load("valid")

x_train_np = make_matrix(train)
x_valid_np = make_matrix(valid)
x_train = torch.from_numpy(x_train_np)
y_train_np = np.asarray(train.y, dtype=np.float32)
y_train = torch.from_numpy(y_train_np)
weight_np = recency_weights(train.date)
row_weights = torch.from_numpy(weight_np)

positive_rate = float(y_train_np.mean())
initial_bias = float(
    np.log(
        np.clip(positive_rate, 1e-6, 1 - 1e-6)
        / np.clip(1.0 - positive_rate, 1e-6, 1 - 1e-6)
    )
)

num_features = int(
    sum(FEATURE_CARDINALITIES[name] for name in FIELDS)
)
list_model = FieldWeightedFM(
    num_features, len(FIELDS), initial_bias
)
list_model = train_listwise(
    list_model,
    x_train,
    y_train,
    row_weights,
    train.user_id,
)
list_valid = predict_torch(list_model, x_valid_np).astype(np.float64)

eb_valid, history_keys = history_score("valid")

user_factors, video_factors = fit_latent_svd(
    train, weight_np, rank=24
)
svd_valid = latent_predict(valid, user_factors, video_factors)

raw_valid = {
    "listwise_fwfm": list_valid,
    "empirical_bayes": eb_valid,
    "latent_svd": svd_valid,
}
raw_rank_valid = {
    name: rank_percentile(valid.user_id, values)
    for name, values in raw_valid.items()
}

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_rank_valid = rank_percentile(valid.user_id, inc_valid)

candidate_scores = {}
candidate_specs = {}

for name, raw in raw_valid.items():
    candidate_scores[name] = raw
    candidate_specs[name] = {"raw": {name: 1.0}, "inc": 0.0}

    for alpha in (0.25, 0.50, 0.75):
        key = f"{name}_incblend_{alpha:.2f}"
        candidate_scores[key] = (
            alpha * raw_rank_valid[name]
            + (1.0 - alpha) * inc_rank_valid
        )
        candidate_specs[key] = {
            "raw": {name: alpha},
            "inc": 1.0 - alpha,
        }

ensemble_rank = np.mean(
    np.stack(list(raw_rank_valid.values()), axis=1), axis=1
)
candidate_scores["three_family_rank"] = ensemble_rank
candidate_specs["three_family_rank"] = {
    "raw": {name: 1.0 / 3.0 for name in raw_valid},
    "inc": 0.0,
}

for alpha in (0.25, 0.50, 0.75):
    key = f"three_family_incblend_{alpha:.2f}"
    candidate_scores[key] = (
        alpha * ensemble_rank + (1.0 - alpha) * inc_rank_valid
    )
    candidate_specs[key] = {
        "raw": {
            name: alpha / 3.0 for name in raw_valid
        },
        "inc": 1.0 - alpha,
    }

candidate_metrics = {
    name: evaluate(valid.user_id, valid.y, score)
    for name, score in candidate_scores.items()
}
best_key = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"]),
)
best_scores = candidate_scores[best_key]
best_metrics = candidate_metrics[best_key]
best_spec = candidate_specs[best_key]

candidate_summary = {
    name: float(metrics["primary"])
    for name, metrics in candidate_metrics.items()
}
print("CANDIDATES " + json.dumps(candidate_summary, sort_keys=True))

rank_names = list(raw_rank_valid)
rank_matrix = np.stack(
    [raw_rank_valid[name] for name in rank_names], axis=1
)
rank_correlations = np.corrcoef(rank_matrix, rowvar=False)
print(
    "FINDINGS " + json.dumps({
        "best_candidate": best_key,
        "recency_half_life_days": HALF_LIFE,
        "history_keys": history_keys,
        "rank_family_order": rank_names,
        "rank_correlation_matrix": rank_correlations.tolist(),
        "raw_primary": {
            name: float(candidate_metrics[name]["primary"])
            for name in raw_valid
        },
    }, sort_keys=True)
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )
    # Save the non-incumbent portion whenever the reported result uses the
    # trusted incumbent.
    if best_spec["inc"] > 0:
        own_total = sum(best_spec["raw"].values())
        if own_total > 0:
            own_valid = np.zeros(len(valid), dtype=np.float64)
            for name, coefficient in best_spec["raw"].items():
                own_valid += (
                    coefficient / own_total
                ) * raw_rank_valid[name]
        else:
            own_valid = ensemble_rank
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            own_valid.astype(np.float64),
        )

test = load("test")
x_test_np = make_matrix(test)

raw_test = {
    "listwise_fwfm": predict_torch(
        list_model, x_test_np
    ).astype(np.float64),
    "empirical_bayes": history_score("test")[0],
    "latent_svd": latent_predict(
        test, user_factors, video_factors
    ),
}
raw_rank_test = {
    name: rank_percentile(test.user_id, values)
    for name, values in raw_test.items()
}

test_scores = np.zeros(len(test), dtype=np.float64)
for name, coefficient in best_spec["raw"].items():
    test_scores += coefficient * raw_rank_test[name]

if best_spec["inc"] > 0:
    inc_test = np.load(
        os.path.join(shared, "incumbent_test_scores.npy")
    ).astype(np.float64)
    inc_rank_test = rank_percentile(test.user_id, inc_test)
    test_scores += best_spec["inc"] * inc_rank_test

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS " + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)