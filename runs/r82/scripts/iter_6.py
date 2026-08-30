import os
import gc
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 7319
FIELDS = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket",
    "tag", "upload_type", "music_type", "hour",
]
N_FIELDS = len(FIELDS)
EMBED_DIM = 10
BATCH_SIZE = 8192
PRED_BATCH_SIZE = 32768
MAX_EPOCHS = 5
HALF_LIFE_DAYS = 10.0


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_all(SEED)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum([0] + cards[:-1], dtype=np.int64)
total_cardinality = int(sum(cards))


def make_matrix(split):
    n = len(split.user_id)
    x = np.empty((n, N_FIELDS), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:, j] = np.asarray(split.X[field], dtype=np.int64) + offsets[j]
    return x


def make_combined_matrix(a, b):
    n1 = len(a.user_id)
    n2 = len(b.user_id)
    x = np.empty((n1 + n2, N_FIELDS), dtype=np.int64)
    for j, field in enumerate(FIELDS):
        x[:n1, j] = np.asarray(a.X[field], dtype=np.int64) + offsets[j]
        x[n1:, j] = np.asarray(b.X[field], dtype=np.int64) + offsets[j]
    return x


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.maximum.accumulate(
        np.where(starts_flag, np.arange(n, dtype=np.int64), 0)
    )
    rank_sorted = np.arange(n, dtype=np.int64) - starts

    _, counts = np.unique(sorted_users, return_counts=True)
    group_sizes = np.repeat(counts, counts)
    denom = np.maximum(group_sizes - 1, 1)
    rank_sorted = rank_sorted.astype(np.float64) / denom
    rank_sorted[group_sizes == 1] = 0.5

    result = np.empty(n, dtype=np.float64)
    result[order] = rank_sorted
    return result


def temporal_weights(dates, half_life=None):
    dates = np.asarray(dates, dtype=np.int64)
    if half_life is None:
        return np.ones(len(dates), dtype=np.float64)
    day = dates % 100
    max_day = int(day.max())
    age = max_day - day
    return np.exp2(-age.astype(np.float64) / float(half_life))


def safe_logit(p):
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


def categorical_rate(train_values, query_values, y, weights, smoothing, global_rate):
    train_values = np.asarray(train_values, dtype=np.int64)
    query_values = np.asarray(query_values, dtype=np.int64)
    size = int(max(train_values.max(initial=0), query_values.max(initial=0))) + 1
    den = np.bincount(train_values, weights=weights, minlength=size)
    num = np.bincount(train_values, weights=weights * y, minlength=size)
    rate = (num + smoothing * global_rate) / (den + smoothing)
    return rate[query_values]


def sparse_pair_delta(
    train_users, train_values, query_users, query_values,
    value_cardinality, y, weights, smoothing, global_rate
):
    train_keys = (
        np.asarray(train_users, dtype=np.int64) * int(value_cardinality)
        + np.asarray(train_values, dtype=np.int64)
    )
    query_keys = (
        np.asarray(query_users, dtype=np.int64) * int(value_cardinality)
        + np.asarray(query_values, dtype=np.int64)
    )

    order = np.argsort(train_keys, kind="mergesort")
    sorted_keys = train_keys[order]
    sorted_w = weights[order]
    sorted_yw = (weights * y)[order]

    starts = np.r_[0, 1 + np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1])]
    unique_keys = sorted_keys[starts]
    den = np.add.reduceat(sorted_w, starts)
    num = np.add.reduceat(sorted_yw, starts)

    positions = np.searchsorted(unique_keys, query_keys)
    valid = positions < len(unique_keys)
    clipped = np.minimum(positions, len(unique_keys) - 1)
    valid &= unique_keys[clipped] == query_keys

    result = np.zeros(len(query_keys), dtype=np.float64)
    if np.any(valid):
        r = (
            num[clipped[valid]] + smoothing * global_rate
        ) / (
            den[clipped[valid]] + smoothing
        )
        result[valid] = safe_logit(r) - safe_logit(global_rate)

    del order, sorted_keys, sorted_w, sorted_yw, starts, unique_keys, den, num
    return result


def empirical_bayes_scores(source, query, half_life, personalized):
    y = np.asarray(source["y"], dtype=np.float64)
    weights = temporal_weights(source["date"], half_life)
    global_rate = float(np.sum(weights * y) / np.sum(weights))

    coefficients = {
        "video_id": (2.0, 30.0),
        "author_id": (1.0, 50.0),
        "tag": (0.75, 100.0),
        "duration_bucket": (0.45, 100.0),
        "tab": (0.35, 200.0),
        "upload_type": (0.25, 150.0),
        "music_type": (0.20, 150.0),
        "hour": (0.15, 200.0),
    }

    score = np.zeros(len(query["user_id"]), dtype=np.float64)
    for field, (coef, smoothing) in coefficients.items():
        rate = categorical_rate(
            source[field], query[field], y, weights, smoothing, global_rate
        )
        score += coef * safe_logit(rate)

    if personalized:
        pair_specs = [
            ("author_id", 0.85, 8.0),
            ("tag", 0.55, 12.0),
            ("duration_bucket", 0.40, 15.0),
        ]
        for field, coef, smoothing in pair_specs:
            delta = sparse_pair_delta(
                source["user_id"],
                source[field],
                query["user_id"],
                query[field],
                FEATURE_CARDINALITIES[field],
                y,
                weights,
                smoothing,
                global_rate,
            )
            score += coef * delta
            del delta
            gc.collect()

    return score.astype(np.float32)


def split_dict(split, include_y):
    d = {
        "user_id": np.asarray(split.user_id, dtype=np.int64),
        "date": np.asarray(split.date, dtype=np.int64),
    }
    for field in FIELDS:
        if field == "user_id":
            continue
        d[field] = np.asarray(split.X[field], dtype=np.int64)
    if include_y:
        d["y"] = np.asarray(split.y, dtype=np.float64)
    return d


def combined_dict(a, b):
    d = {
        "user_id": np.concatenate([
            np.asarray(a.user_id, dtype=np.int64),
            np.asarray(b.user_id, dtype=np.int64),
        ]),
        "date": np.concatenate([
            np.asarray(a.date, dtype=np.int64),
            np.asarray(b.date, dtype=np.int64),
        ]),
        "y": np.concatenate([
            np.asarray(a.y, dtype=np.float64),
            np.asarray(b.y, dtype=np.float64),
        ]),
    }
    for field in FIELDS:
        if field == "user_id":
            continue
        d[field] = np.concatenate([
            np.asarray(a.X[field], dtype=np.int64),
            np.asarray(b.X[field], dtype=np.int64),
        ])
    return d


class LatentBPR(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.user_embedding = nn.Embedding(cards[0], 24)
        self.video_embedding = nn.Embedding(cards[1], 24)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.video_embedding.weight, std=0.02)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        raw_user = x[:, 0] - offsets[0]
        raw_video = x[:, 1] - offsets[1]
        latent = (
            self.user_embedding(raw_user) * self.video_embedding(raw_video)
        ).sum(dim=1)
        return linear + latent


class DCNBPR(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Embedding(total_cardinality, 1)
        self.embedding = nn.Embedding(total_cardinality, EMBED_DIM)
        dim = N_FIELDS * EMBED_DIM
        self.cross1 = nn.Linear(dim, 1)
        self.cross2 = nn.Linear(dim, 1)
        self.deep = nn.Sequential(
            nn.Linear(dim, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.output = nn.Linear(dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.015)

    def forward(self, x):
        wide = self.linear(x).sum(dim=1).squeeze(-1)
        x0 = self.embedding(x).reshape(x.shape[0], -1)
        x1 = x0 * self.cross1(x0) + x0
        x2 = x0 * self.cross2(x1) + x1
        return (
            wide
            + self.output(x2).squeeze(-1)
            + self.deep(x0).squeeze(-1)
        )


def make_model(name):
    if name == "latent_bpr":
        return LatentBPR()
    if name == "dcn_bpr":
        return DCNBPR()
    raise ValueError(name)


def positive_negative_pairs(users, labels, rng):
    users = np.asarray(users, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    max_user = max(int(users.max(initial=0)) + 1, cards[0])

    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    neg_order = np.argsort(users[neg_idx], kind="stable")
    neg_sorted = neg_idx[neg_order]

    neg_counts = np.bincount(users[neg_sorted], minlength=max_user)
    neg_starts = np.cumsum(
        np.r_[0, neg_counts[:-1]], dtype=np.int64
    )

    eligible = neg_counts[users[pos_idx]] > 0
    pos_idx = pos_idx[eligible]
    pos_users = users[pos_idx]
    offsets_random = (
        rng.random(len(pos_idx)) * neg_counts[pos_users]
    ).astype(np.int64)
    sampled_neg = neg_sorted[neg_starts[pos_users] + offsets_random]
    return pos_idx, sampled_neg


@torch.no_grad()
def predict_model(model, x):
    model.eval()
    xt = torch.from_numpy(x)
    out = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(x))
        out[start:end] = model(xt[start:end]).cpu().numpy()
    return out


def train_bpr_epoch(model, optimizer, x_tensor, users, labels, dates, epoch_seed):
    model.train()
    rng = np.random.default_rng(epoch_seed)
    pos_idx, neg_idx = positive_negative_pairs(users, labels, rng)
    order = rng.permutation(len(pos_idx))

    day = np.asarray(dates, dtype=np.int64) % 100
    age = int(day.max()) - day
    row_weight = np.exp2(-age.astype(np.float32) / HALF_LIFE_DAYS)
    row_weight /= max(float(row_weight.mean()), 1e-6)

    total = 0.0
    seen = 0
    for start in range(0, len(order), BATCH_SIZE):
        ids = order[start:start + BATCH_SIZE]
        p_np = pos_idx[ids]
        n_np = neg_idx[ids]
        p = torch.from_numpy(p_np)
        n = torch.from_numpy(n_np)

        optimizer.zero_grad(set_to_none=True)
        positive_score = model(x_tensor[p])
        negative_score = model(x_tensor[n])
        weights = torch.from_numpy(row_weight[p_np])
        loss = (
            nn.functional.softplus(-(positive_score - negative_score)) * weights
        ).mean()
        loss.backward()
        optimizer.step()

        total += float(loss.detach()) * len(ids)
        seen += len(ids)

    return total / max(seen, 1)


def fit_neural_candidate(name, x_train, train, x_valid, valid, y_valid):
    seed_all(SEED + (101 if name == "latent_bpr" else 303))
    model = make_model(name)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.003 if name == "latent_bpr" else 0.0015,
        weight_decay=1e-6,
    )
    xt = torch.from_numpy(x_train)
    labels = np.asarray(train.y, dtype=np.int8)

    best_primary = -np.inf
    best_epoch = 1
    best_scores = None

    for epoch in range(1, MAX_EPOCHS + 1):
        train_bpr_epoch(
            model, optimizer, xt, train.user_id, labels, train.date,
            SEED + epoch * 1009
        )
        scores = predict_model(model, x_valid)
        metric = evaluate(valid.user_id, y_valid, scores)
        if float(metric["primary"]) > best_primary:
            best_primary = float(metric["primary"])
            best_epoch = epoch
            best_scores = scores.copy()

    del model, optimizer, xt
    gc.collect()
    return best_scores, best_epoch, best_primary


def refit_neural(name, epochs, x_combined, combined_users,
                 combined_labels, combined_dates, x_test):
    seed_all(SEED + (101 if name == "latent_bpr" else 303))
    model = make_model(name)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.003 if name == "latent_bpr" else 0.0015,
        weight_decay=1e-6,
    )
    xt = torch.from_numpy(x_combined)
    for epoch in range(1, epochs + 1):
        train_bpr_epoch(
            model, optimizer, xt, combined_users, combined_labels,
            combined_dates, SEED + epoch * 1009
        )
    scores = predict_model(model, x_test)
    del model, optimizer, xt
    gc.collect()
    return scores


train = load("train")
valid = load("valid")
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64, copy=False)
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

train_dict = split_dict(train, include_y=True)
valid_dict = split_dict(valid, include_y=False)

candidate_predictions = {}
candidate_epochs = {}
candidate_log = {}

eb_uniform = empirical_bayes_scores(
    train_dict, valid_dict, half_life=None, personalized=False
)
candidate_predictions["eb_uniform"] = eb_uniform
candidate_log["eb_uniform"] = float(
    evaluate(valid.user_id, y_valid, eb_uniform)["primary"]
)

eb_temporal_personal = empirical_bayes_scores(
    train_dict, valid_dict,
    half_life=HALF_LIFE_DAYS,
    personalized=True,
)
candidate_predictions["eb_temporal_personal"] = eb_temporal_personal
candidate_log["eb_temporal_personal"] = float(
    evaluate(valid.user_id, y_valid, eb_temporal_personal)["primary"]
)

x_train = make_matrix(train)
x_valid = make_matrix(valid)

for neural_name in ["latent_bpr", "dcn_bpr"]:
    scores, epoch, standalone = fit_neural_candidate(
        neural_name, x_train, train, x_valid, valid, y_valid
    )
    candidate_predictions[neural_name] = scores
    candidate_epochs[neural_name] = int(epoch)
    candidate_log[neural_name] = float(standalone)

blend_grid = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0]
winner = None
winner_primary = -np.inf

for name, scores in candidate_predictions.items():
    candidate_rank = within_user_rank(valid.user_id, scores)
    best_for_family = -np.inf
    best_alpha = 0.0
    best_scores = None

    for alpha in blend_grid:
        blended = (
            alpha * candidate_rank
            + (1.0 - alpha) * inc_valid_rank
        )
        primary = float(
            evaluate(valid.user_id, y_valid, blended)["primary"]
        )
        if primary > best_for_family:
            best_for_family = primary
            best_alpha = float(alpha)
            best_scores = blended.copy()

    candidate_log[name + "_incumbent_rank_blend"] = best_for_family
    if best_for_family > winner_primary:
        winner_primary = best_for_family
        winner = {
            "name": name,
            "alpha": best_alpha,
            "valid_scores": best_scores,
            "epoch": candidate_epochs.get(name, 0),
        }

valid_scores = np.asarray(winner["valid_scores"], dtype=np.float64)
metrics = evaluate(valid.user_id, y_valid, valid_scores)

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print("FINDINGS " + json.dumps({
    "selected_family": winner["name"],
    "candidate_rank_weight": winner["alpha"],
    "selected_epoch": winner["epoch"],
    "temporal_half_life_days": HALF_LIFE_DAYS,
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "scores_valid.npy"), valid_scores)

selected_name = winner["name"]
selected_alpha = float(winner["alpha"])

del candidate_predictions, eb_uniform, eb_temporal_personal
del x_train, x_valid
gc.collect()

test = load("test")

if selected_name.startswith("eb_"):
    source_combined = combined_dict(train, valid)
    query_test = split_dict(test, include_y=False)
    if selected_name == "eb_uniform":
        new_test_scores = empirical_bayes_scores(
            source_combined, query_test,
            half_life=None,
            personalized=False,
        )
    else:
        new_test_scores = empirical_bayes_scores(
            source_combined, query_test,
            half_life=HALF_LIFE_DAYS,
            personalized=True,
        )
else:
    x_combined = make_combined_matrix(train, valid)
    x_test = make_matrix(test)
    combined_users = np.concatenate([
        np.asarray(train.user_id, dtype=np.int64),
        np.asarray(valid.user_id, dtype=np.int64),
    ])
    combined_labels = np.concatenate([
        np.asarray(train.y, dtype=np.int8),
        np.asarray(valid.y, dtype=np.int8),
    ])
    combined_dates = np.concatenate([
        np.asarray(train.date, dtype=np.int64),
        np.asarray(valid.date, dtype=np.int64),
    ])
    new_test_scores = refit_neural(
        selected_name,
        int(winner["epoch"]),
        x_combined,
        combined_users,
        combined_labels,
        combined_dates,
        x_test,
    )

candidate_test_rank = within_user_rank(test.user_id, new_test_scores)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64, copy=False)
inc_test_rank = within_user_rank(test.user_id, inc_test)
test_scores = (
    selected_alpha * candidate_test_rank
    + (1.0 - selected_alpha) * inc_test_rank
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))