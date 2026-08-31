import os
import time
import json
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import load
from pipeline.evaluate import evaluate


START = time.time()
SEED = 84217
rng = np.random.default_rng(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

HASH_BITS = 21
HASH_SIZE = 1 << HASH_BITS
HASH_MASK = HASH_SIZE - 1
BATCH_SIZE = 32768
PRED_BATCH_SIZE = 65536
EPOCHS = 3
HALF_LIFE = 5.0

SINGLE_FIELDS = [
    "user_id", "video_id", "author_id", "tab", "tag",
    "upload_type", "duration_bucket", "onehot_feat3",
    "onehot_feat8", "onehot_feat1", "music_type",
    "user_active_degree", "register_days_bucket",
]

PAIR_FIELDS = [
    ("user_id", "tag"),
    ("user_id", "tab"),
    ("user_id", "duration_bucket"),
    ("user_id", "upload_type"),
    ("user_id", "author_id"),
    ("user_id", "onehot_feat3"),
    ("video_id", "tab"),
    ("author_id", "user_active_degree"),
]


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int64)
    age = dates.max() - dates
    w = np.power(0.5, age.astype(np.float32) / HALF_LIFE)
    w /= max(float(w.mean()), 1e-8)
    return np.ascontiguousarray(w, dtype=np.float32)


def hash_single(x, field_number):
    x = np.asarray(x, dtype=np.int64)
    z = (
        x * np.int64(1000003)
        + np.int64(15485863 * (field_number + 1))
    )
    z ^= z >> np.int64(17)
    return np.asarray(z & HASH_MASK, dtype=np.int32)


def hash_pair(a, b, field_number):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    z = (
        a * np.int64(1000003)
        + b * np.int64(9176)
        + np.int64(32452843 * (field_number + 1))
    )
    z ^= z >> np.int64(16)
    return np.asarray(z & HASH_MASK, dtype=np.int32)


def make_hash_tokens(split):
    columns = []

    # A constant token lets the sparse optimizer learn an intercept.
    columns.append(np.zeros(len(split.user_id), dtype=np.int32))

    for j, name in enumerate(SINGLE_FIELDS):
        columns.append(hash_single(split.X[name], j + 1))

    base = len(SINGLE_FIELDS) + 2
    for j, (left, right) in enumerate(PAIR_FIELDS):
        columns.append(
            hash_pair(
                split.X[left],
                split.X[right],
                base + j,
            )
        )

    return np.ascontiguousarray(
        np.column_stack(columns), dtype=np.int32
    )


class HashedCrossLogistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Embedding(
            HASH_SIZE, 1, sparse=True
        )
        nn.init.zeros_(self.weight.weight)

    def forward(self, tokens):
        return self.weight(tokens).sum(dim=1).squeeze(-1)


def fit_hashed_model(tokens, labels, weights):
    model = HashedCrossLogistic()
    optimizer = torch.optim.SparseAdam(
        model.parameters(), lr=0.035
    )
    labels = np.asarray(labels, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    local_rng = np.random.default_rng(SEED + 19)

    model.train()
    for _ in range(EPOCHS):
        order = local_rng.permutation(len(labels))
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            x = torch.from_numpy(
                tokens[idx].astype(np.int64, copy=False)
            )
            y = torch.from_numpy(labels[idx])
            w = torch.from_numpy(weights[idx])

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            row_loss = F.binary_cross_entropy_with_logits(
                logits, y, reduction="none"
            )
            loss = (row_loss * w).sum() / w.sum()
            loss.backward()
            optimizer.step()

    return model


@torch.no_grad()
def predict_hashed(model, tokens):
    model.eval()
    result = np.empty(len(tokens), dtype=np.float32)
    for start in range(0, len(tokens), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(tokens))
        x = torch.from_numpy(
            tokens[start:end].astype(np.int64, copy=False)
        )
        result[start:end] = model(x).cpu().numpy()
    return result


class SmoothedRateTable:
    def __init__(self, keys, rates, default_rate):
        self.keys = np.asarray(keys, dtype=np.int64)
        self.rates = np.asarray(rates, dtype=np.float32)
        self.default_rate = float(default_rate)

    def predict_rate(self, query_keys):
        query_keys = np.asarray(query_keys, dtype=np.int64)
        pos = np.searchsorted(self.keys, query_keys)
        clipped = np.minimum(pos, max(len(self.keys) - 1, 0))

        result = np.full(
            len(query_keys), self.default_rate, dtype=np.float32
        )
        if len(self.keys):
            found = (
                (pos < len(self.keys))
                & (self.keys[clipped] == query_keys)
            )
            result[found] = self.rates[clipped[found]]
        return result

    def predict_logit(self, query_keys):
        p = self.predict_rate(query_keys)
        p = np.clip(p, 1e-4, 1.0 - 1e-4)
        return np.log(p / (1.0 - p)).astype(np.float32)


def fit_rate_table(keys, labels, weights, smoothing, prior=None):
    keys = np.asarray(keys, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if prior is None:
        prior = float(
            np.sum(weights * labels) / max(np.sum(weights), 1e-12)
        )

    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
    sorted_w = weights[order]
    sorted_yw = sorted_w * labels[order]

    unique_keys, starts = np.unique(
        sorted_keys, return_index=True
    )
    counts = np.add.reduceat(sorted_w, starts)
    positives = np.add.reduceat(sorted_yw, starts)

    rates = (
        positives + float(smoothing) * float(prior)
    ) / (counts + float(smoothing))

    return SmoothedRateTable(unique_keys, rates, prior)


def pair_keys(split, left, right, right_cardinality):
    a = np.asarray(split.X[left], dtype=np.int64)
    b = np.asarray(split.X[right], dtype=np.int64)
    return a * np.int64(right_cardinality) + b


def hierarchical_scores(train, other, labels, weights):
    global_prior = float(
        np.sum(weights * labels) / np.sum(weights)
    )

    # Stable entity quality.
    video_table = fit_rate_table(
        train.X["video_id"], labels, weights,
        smoothing=35.0, prior=global_prior
    )
    author_table = fit_rate_table(
        train.X["author_id"], labels, weights,
        smoothing=55.0, prior=global_prior
    )

    score = (
        0.34 * video_table.predict_logit(other.X["video_id"])
        + 0.18 * author_table.predict_logit(other.X["author_id"])
    )

    pair_specs = [
        ("user_id", "tag", 64, 12.0, 0.14),
        ("user_id", "tab", 20, 10.0, 0.11),
        ("user_id", "duration_bucket", 12, 12.0, 0.09),
        ("user_id", "upload_type", 20, 14.0, 0.07),
        ("user_id", "onehot_feat3", 1600, 18.0, 0.07),
    ]

    for left, right, card, smoothing, coefficient in pair_specs:
        tr_keys = pair_keys(train, left, right, card)
        table = fit_rate_table(
            tr_keys, labels, weights,
            smoothing=smoothing, prior=global_prior
        )
        query = pair_keys(other, left, right, card)
        score += coefficient * table.predict_logit(query)
        del tr_keys, query, table

    return np.asarray(score, dtype=np.float32)


def fit_temporal_entity(train, labels, field, global_prior):
    dates = np.asarray(train.date, dtype=np.int64)
    cutoff = int(dates.max()) - 4

    recent = dates >= cutoff
    older = ~recent

    recent_table = fit_rate_table(
        train.X[field][recent],
        labels[recent],
        np.ones(np.sum(recent), dtype=np.float32),
        smoothing=40.0 if field == "video_id" else 65.0,
        prior=global_prior,
    )
    older_table = fit_rate_table(
        train.X[field][older],
        labels[older],
        np.ones(np.sum(older), dtype=np.float32),
        smoothing=55.0 if field == "video_id" else 80.0,
        prior=global_prior,
    )
    return recent_table, older_table


def temporal_trend_scores(train, other, labels):
    global_prior = float(np.mean(labels))
    result = np.zeros(len(other.user_id), dtype=np.float32)

    for field, coefficient in (
        ("video_id", 0.62),
        ("author_id", 0.28),
        ("tag", 0.10),
    ):
        recent, older = fit_temporal_entity(
            train, labels, field, global_prior
        )
        recent_logit = recent.predict_logit(other.X[field])
        older_logit = older.predict_logit(other.X[field])

        # Extrapolate only part of the observed train-window movement;
        # shrinkage in both tables suppresses unstable rare-entity trends.
        forecast = recent_logit + 0.65 * (
            recent_logit - older_logit
        )
        forecast = np.clip(forecast, -7.0, 7.0)
        result += coefficient * forecast.astype(np.float32)

    return result


def within_user_rank(scores, users):
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users, dtype=np.int64)
    n = len(scores)
    if n == 0:
        return scores.copy()

    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]

    starts_flag = np.empty(n, dtype=bool)
    starts_flag[0] = True
    starts_flag[1:] = sorted_users[1:] != sorted_users[:-1]

    positions = np.arange(n, dtype=np.int64)
    starts = np.maximum.accumulate(
        np.where(starts_flag, positions, 0)
    )
    local_position = positions - starts

    ends_flag = np.empty(n, dtype=bool)
    ends_flag[-1] = True
    ends_flag[:-1] = sorted_users[:-1] != sorted_users[1:]
    group_ends = np.minimum.accumulate(
        np.where(ends_flag, positions, n - 1)[::-1]
    )[::-1]
    counts = group_ends - starts + 1
    denominator = np.maximum(counts - 1, 1)

    ranked = local_position.astype(np.float64) / denominator
    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def primary(users, labels, scores):
    return float(evaluate(users, labels, scores)["primary"])


train = load("train")
valid = load("valid")
test = load("test")

train_y = np.asarray(train.y, dtype=np.float32)
valid_y = np.asarray(valid.y, dtype=np.int8)
train_weights = recency_weights(train.date)

# Family 1: discriminative sparse logistic regression over explicit
# high-cardinality conjunctions.
train_tokens = make_hash_tokens(train)
valid_tokens = make_hash_tokens(valid)
test_tokens = make_hash_tokens(test)

hashed_model = fit_hashed_model(
    train_tokens, train_y, train_weights
)
hashed_valid = predict_hashed(hashed_model, valid_tokens)
hashed_test = predict_hashed(hashed_model, test_tokens)

del train_tokens, valid_tokens, test_tokens, hashed_model
gc.collect()

# Family 2: hierarchical non-parametric personalization. Each table has
# independent empirical-Bayes shrinkage rather than shared logistic weights.
hier_valid = hierarchical_scores(
    train, valid, train_y, train_weights
)
hier_test = hierarchical_scores(
    train, test, train_y, train_weights
)

# Family 3: a temporal entity-state forecaster based on recent versus older
# train windows, explicitly extrapolating item/author/tag movement.
trend_valid = temporal_trend_scores(train, valid, train_y)
trend_test = temporal_trend_scores(train, test, train_y)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)
if not (
    os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise FileNotFoundError(
        "Trusted incumbent predictions are unavailable"
    )

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

valid_models = {
    "hashed_cross_logistic": np.asarray(
        hashed_valid, dtype=np.float64
    ),
    "hierarchical_target_tables": np.asarray(
        hier_valid, dtype=np.float64
    ),
    "temporal_entity_forecast": np.asarray(
        trend_valid, dtype=np.float64
    ),
}
test_models = {
    "hashed_cross_logistic": np.asarray(
        hashed_test, dtype=np.float64
    ),
    "hierarchical_target_tables": np.asarray(
        hier_test, dtype=np.float64
    ),
    "temporal_entity_forecast": np.asarray(
        trend_test, dtype=np.float64
    ),
}

valid_users = np.asarray(valid.user_id, dtype=np.int64)
test_users = np.asarray(test.user_id, dtype=np.int64)

inc_valid_rank = within_user_rank(inc_valid, valid_users)
inc_test_rank = within_user_rank(inc_test, test_users)

valid_ranks = {
    name: within_user_rank(score, valid_users)
    for name, score in valid_models.items()
}
test_ranks = {
    name: within_user_rank(score, test_users)
    for name, score in test_models.items()
}

candidate_scores = {}
candidate_payloads = {}

# Standalone scores are recorded for family-level evidence.
for name in valid_models:
    p = primary(valid_users, valid_y, valid_models[name])
    candidate_scores[name + "_standalone"] = p
    candidate_payloads[name + "_standalone"] = (
        valid_models[name],
        test_models[name],
        valid_models[name],
        False,
    )

# Compare each structurally new family with the trusted incumbent.
blend_alphas = [0.08, 0.14, 0.20, 0.28, 0.36, 0.46]
for name in valid_models:
    best_p = -np.inf
    best_alpha = None
    best_valid = None
    best_test = None

    for alpha in blend_alphas:
        va = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * valid_ranks[name]
        )
        p = primary(valid_users, valid_y, va)
        if p > best_p:
            best_p = p
            best_alpha = alpha
            best_valid = va
            best_test = (
                (1.0 - alpha) * inc_test_rank
                + alpha * test_ranks[name]
            )

    key = name + "_incumbent_blend"
    candidate_scores[key] = float(best_p)
    candidate_payloads[key] = (
        best_valid,
        best_test,
        valid_models[name],
        True,
    )
    print(
        "FINDINGS "
        + json.dumps({
            "family": name,
            "best_blend_alpha": best_alpha,
            "standalone_primary": candidate_scores[
                name + "_standalone"
            ],
            "blend_primary": best_p,
        }, sort_keys=True)
    )

# A rank ensemble tests whether the three different error structures are
# complementary even when none is individually strong.
ensemble_valid_rank = np.mean(
    np.column_stack([
        valid_ranks["hashed_cross_logistic"],
        valid_ranks["hierarchical_target_tables"],
        valid_ranks["temporal_entity_forecast"],
    ]),
    axis=1,
)
ensemble_test_rank = np.mean(
    np.column_stack([
        test_ranks["hashed_cross_logistic"],
        test_ranks["hierarchical_target_tables"],
        test_ranks["temporal_entity_forecast"],
    ]),
    axis=1,
)

ensemble_raw_primary = primary(
    valid_users, valid_y, ensemble_valid_rank
)
candidate_scores["new_family_rank_ensemble_standalone"] = (
    ensemble_raw_primary
)
candidate_payloads["new_family_rank_ensemble_standalone"] = (
    ensemble_valid_rank,
    ensemble_test_rank,
    ensemble_valid_rank,
    False,
)

best_ensemble_p = -np.inf
best_ensemble_alpha = None
best_ensemble_valid = None
best_ensemble_test = None
for alpha in blend_alphas:
    va = (
        (1.0 - alpha) * inc_valid_rank
        + alpha * ensemble_valid_rank
    )
    p = primary(valid_users, valid_y, va)
    if p > best_ensemble_p:
        best_ensemble_p = p
        best_ensemble_alpha = alpha
        best_ensemble_valid = va
        best_ensemble_test = (
            (1.0 - alpha) * inc_test_rank
            + alpha * ensemble_test_rank
        )

candidate_scores["new_family_ensemble_incumbent_blend"] = float(
    best_ensemble_p
)
candidate_payloads["new_family_ensemble_incumbent_blend"] = (
    best_ensemble_valid,
    best_ensemble_test,
    ensemble_valid_rank,
    True,
)

print(
    "FINDINGS "
    + json.dumps({
        "ensemble_standalone_primary": ensemble_raw_primary,
        "ensemble_best_alpha": best_ensemble_alpha,
        "ensemble_blend_primary": best_ensemble_p,
    }, sort_keys=True)
)

# Include the unchanged trusted incumbent as a conservative selectable
# reference, while still retaining the strongest own-model raw score.
inc_primary = primary(valid_users, valid_y, inc_valid)
candidate_scores["trusted_incumbent_reference"] = inc_primary
candidate_payloads["trusted_incumbent_reference"] = (
    inc_valid,
    inc_test,
    ensemble_valid_rank,
    True,
)

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores, test_scores, raw_valid_scores, is_combination = (
    candidate_payloads[winner]
)

metrics = evaluate(valid_users, valid_y, valid_scores)

print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_scores.items()},
        sort_keys=True,
    )
)
print(
    "FINDINGS "
    + json.dumps({
        "selected_candidate": winner,
        "selected_primary": float(metrics["primary"]),
        "incumbent_primary": inc_primary,
    }, sort_keys=True)
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    if is_combination:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_valid_scores, dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)