import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 9187
HALF_LIFE_DAYS = 7.0
BATCH_SIZE = 8192
PRED_BATCH = 65536
EPOCHS_LINEAR = 4
EPOCHS_MLP = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

ENTITY_FIELDS = [
    "video_id", "author_id", "tag", "tab",
    "duration_bucket", "hour", "upload_type", "music_type"
]
PAIR_FIELDS = [
    "tag", "author_id", "tab", "duration_bucket", "upload_type"
]

# Corresponding entity feature for each personalized feature.
PAIR_TO_ENTITY = {
    "tag": "tag",
    "author_id": "author_id",
    "tab": "tab",
    "duration_bucket": "duration_bucket",
    "upload_type": "upload_type"
}


def chronological_day_index(dates):
    dates = np.asarray(dates, dtype=np.int64)
    unique_dates = np.unique(dates)
    return np.searchsorted(unique_dates, dates).astype(np.float32)


def recency_weights(dates):
    day_index = chronological_day_index(dates)
    age = float(day_index.max()) - day_index
    weights = np.exp2(-age / HALF_LIFE_DAYS).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def clipped_logit(rate):
    p = np.clip(np.asarray(rate, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


def fit_entity_stat(ref_values, eval_values, y, weights, prior, alpha):
    ref_values = np.asarray(ref_values, dtype=np.int64)
    eval_values = np.asarray(eval_values, dtype=np.int64)
    card = int(max(
        int(ref_values.max(initial=0)),
        int(eval_values.max(initial=0))
    )) + 1

    count = np.bincount(
        ref_values, weights=weights, minlength=card
    ).astype(np.float64)
    positive = np.bincount(
        ref_values, weights=weights * y, minlength=card
    ).astype(np.float64)

    loo_count = np.maximum(count[ref_values] - weights, 0.0)
    loo_positive = np.maximum(
        positive[ref_values] - weights * y, 0.0
    )
    train_rate = (
        loo_positive + alpha * prior
    ) / (loo_count + alpha)

    safe_eval = np.clip(eval_values, 0, card - 1)
    known = eval_values < card
    eval_count = np.where(known, count[safe_eval], 0.0)
    eval_positive = np.where(known, positive[safe_eval], 0.0)
    eval_rate = (
        eval_positive + alpha * prior
    ) / (eval_count + alpha)

    return (
        train_rate.astype(np.float32),
        np.log1p(loo_count).astype(np.float32),
        eval_rate.astype(np.float32),
        np.log1p(eval_count).astype(np.float32)
    )


def fit_sparse_pair_stat(ref_codes, eval_codes, y, weights, prior, alpha):
    ref_codes = np.asarray(ref_codes, dtype=np.int64)
    eval_codes = np.asarray(eval_codes, dtype=np.int64)

    unique_codes, inverse = np.unique(ref_codes, return_inverse=True)
    count = np.bincount(
        inverse, weights=weights, minlength=len(unique_codes)
    ).astype(np.float64)
    positive = np.bincount(
        inverse, weights=weights * y, minlength=len(unique_codes)
    ).astype(np.float64)

    loo_count = np.maximum(count[inverse] - weights, 0.0)
    loo_positive = np.maximum(
        positive[inverse] - weights * y, 0.0
    )
    train_rate = (
        loo_positive + alpha * prior
    ) / (loo_count + alpha)

    locations = np.searchsorted(unique_codes, eval_codes)
    locations_safe = np.minimum(locations, len(unique_codes) - 1)
    known = (
        (locations < len(unique_codes))
        & (unique_codes[locations_safe] == eval_codes)
    )

    eval_count = np.zeros(len(eval_codes), dtype=np.float64)
    eval_positive = np.zeros(len(eval_codes), dtype=np.float64)
    eval_count[known] = count[locations_safe[known]]
    eval_positive[known] = positive[locations_safe[known]]
    eval_rate = (
        eval_positive + alpha * prior
    ) / (eval_count + alpha)

    return (
        train_rate.astype(np.float32),
        np.log1p(loo_count).astype(np.float32),
        eval_rate.astype(np.float32),
        np.log1p(eval_count).astype(np.float32)
    )


def make_history_features(reference, evaluation):
    y = np.asarray(reference.y, dtype=np.float32)
    weights = recency_weights(reference.date)
    prior = float(np.sum(weights * y) / np.sum(weights))

    train_columns = []
    eval_columns = []
    feature_index = {}
    rate_index = {}

    for field in ENTITY_FIELDS:
        result = fit_entity_stat(
            reference.X[field], evaluation.X[field],
            y, weights, prior, alpha=24.0
        )
        train_rate, train_count, eval_rate, eval_count = result

        rate_index[("entity", field)] = len(train_columns)
        feature_index["entity_" + field + "_rate"] = len(train_columns)
        train_columns.append(train_rate)
        eval_columns.append(eval_rate)

        feature_index["entity_" + field + "_logcount"] = len(train_columns)
        train_columns.append(train_count)
        eval_columns.append(eval_count)

    user_card = int(FEATURE_CARDINALITIES["user_id"])
    ref_user = np.asarray(reference.X["user_id"], dtype=np.int64)
    eval_user = np.asarray(evaluation.X["user_id"], dtype=np.int64)

    for field in PAIR_FIELDS:
        card = int(FEATURE_CARDINALITIES[field])
        ref_code = ref_user * np.int64(card) + np.asarray(
            reference.X[field], dtype=np.int64
        )
        eval_code = eval_user * np.int64(card) + np.asarray(
            evaluation.X[field], dtype=np.int64
        )

        result = fit_sparse_pair_stat(
            ref_code, eval_code, y, weights, prior, alpha=12.0
        )
        train_rate, train_count, eval_rate, eval_count = result

        rate_index[("pair", field)] = len(train_columns)
        feature_index["user_" + field + "_rate"] = len(train_columns)
        train_columns.append(train_rate)
        eval_columns.append(eval_rate)

        feature_index["user_" + field + "_logcount"] = len(train_columns)
        train_columns.append(train_count)
        eval_columns.append(eval_count)

    # Raw stationary context quantities let the nonlinear family gate history
    # strength by content type without adding high-cardinality identity inputs.
    for field in ["duration_ms"]:
        ref_num = np.asarray(reference.num[field], dtype=np.float32)
        eval_num = np.asarray(evaluation.num[field], dtype=np.float32)
        ref_num = np.log1p(np.maximum(np.nan_to_num(ref_num), 0.0))
        eval_num = np.log1p(np.maximum(np.nan_to_num(eval_num), 0.0))
        feature_index["numeric_" + field] = len(train_columns)
        train_columns.append(ref_num.astype(np.float32))
        eval_columns.append(eval_num.astype(np.float32))

    X_train = np.ascontiguousarray(
        np.column_stack(train_columns), dtype=np.float32
    )
    X_eval = np.ascontiguousarray(
        np.column_stack(eval_columns), dtype=np.float32
    )

    metadata = {
        "prior": prior,
        "rate_index": rate_index,
        "feature_index": feature_index
    }
    return X_train, X_eval, weights, metadata


def empirical_bayes_scores(X, metadata):
    ri = metadata["rate_index"]

    entity_weights = {
        "video_id": 1.00,
        "author_id": 0.85,
        "tag": 0.55,
        "tab": 0.55,
        "duration_bucket": 0.35,
        "hour": 0.15,
        "upload_type": 0.25,
        "music_type": 0.15
    }

    score = np.zeros(len(X), dtype=np.float64)
    total_weight = 0.0
    for field, weight in entity_weights.items():
        score += weight * clipped_logit(X[:, ri[("entity", field)]])
        total_weight += weight
    score /= total_weight

    # Personalized preference lifts are measured relative to the item's
    # population-level content propensity, preventing popular tags/authors
    # from being counted twice.
    pair_weights = {
        "tag": 0.75,
        "author_id": 0.90,
        "tab": 0.55,
        "duration_bucket": 0.45,
        "upload_type": 0.30
    }
    for field, weight in pair_weights.items():
        pair_rate = X[:, ri[("pair", field)]]
        entity_field = PAIR_TO_ENTITY[field]
        entity_rate = X[:, ri[("entity", entity_field)]]
        score += weight * (
            clipped_logit(pair_rate) - clipped_logit(entity_rate)
        )

    return score


class LinearHistory(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear = nn.Linear(dimension, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


class GatedHistoryMLP(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, 48),
            nn.LayerNorm(48),
            nn.SiLU(),
            nn.Linear(48, 24),
            nn.SiLU(),
            nn.Linear(24, 1)
        )
        self.skip = nn.Linear(dimension, 1)

    def forward(self, x):
        return (self.net(x) + self.skip(x)).squeeze(1)


def standardize_fit(X):
    mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-4, 1.0, std).astype(np.float32)
    return mean, std


def standardize_apply(X, mean, std):
    result = (X - mean[None, :]) / std[None, :]
    return np.ascontiguousarray(
        np.clip(result, -8.0, 8.0), dtype=np.float32
    )


def fit_dense_model(X, y, sample_weights, family):
    seed_offset = 31 if family == "linear" else 67
    torch.manual_seed(SEED + seed_offset)

    mean, std = standardize_fit(X)
    Xs = standardize_apply(X, mean, std)
    y = np.asarray(y, dtype=np.float32)
    sample_weights = np.asarray(sample_weights, dtype=np.float32)
    sample_weights = sample_weights / max(float(sample_weights.mean()), 1e-8)

    if family == "linear":
        model = LinearHistory(X.shape[1])
        epochs = EPOCHS_LINEAR
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.012, weight_decay=2e-5
        )
    elif family == "mlp":
        model = GatedHistoryMLP(X.shape[1])
        epochs = EPOCHS_MLP
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.003, weight_decay=5e-5
        )
    else:
        raise ValueError(family)

    xt = torch.from_numpy(Xs)
    yt = torch.from_numpy(y)
    wt = torch.from_numpy(sample_weights)
    rng = np.random.default_rng(SEED + seed_offset + 100)

    for _ in range(epochs):
        order = rng.permutation(len(Xs))
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            idx_np = order[start:start + BATCH_SIZE]
            idx = torch.from_numpy(idx_np)
            logits = model(xt[idx])
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yt[idx], reduction="none"
            )
            loss = (losses * wt[idx]).sum() / wt[idx].sum().clamp_min(1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model, mean, std


@torch.inference_mode()
def predict_dense(model, X, mean, std):
    Xs = standardize_apply(X, mean, std)
    xt = torch.from_numpy(Xs)
    result = np.empty(len(X), dtype=np.float32)
    model.eval()
    for start in range(0, len(X), PRED_BATCH):
        end = min(start + PRED_BATCH, len(X))
        result[start:end] = model(xt[start:end]).cpu().numpy()
    return result.astype(np.float64)


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    std = float(x.std())
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - float(x.mean())) / std


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((scores, user_ids))
    sorted_users = user_ids[order]
    starts_mask = np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    group_start = np.repeat(starts, lengths)
    positions = np.arange(n, dtype=np.int64) - group_start
    denominators = np.repeat(np.maximum(lengths - 1, 1), lengths)
    ranked_sorted = positions / denominators

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked_sorted
    return result


def add_family_candidates(container, name, scores, incumbent, users):
    scores = np.asarray(scores, dtype=np.float64)
    container.append((name + "_raw", scores, (name, "raw", 0.0)))

    new_z = zscore(scores)
    inc_z = zscore(incumbent)
    new_rank = within_user_rank(users, scores)
    inc_rank = within_user_rank(users, incumbent)

    for alpha in (0.25, 0.50, 0.75):
        container.append((
            name + "_zblend_inc%.2f" % alpha,
            alpha * inc_z + (1.0 - alpha) * new_z,
            (name, "zblend", alpha)
        ))
        container.append((
            name + "_rankblend_inc%.2f" % alpha,
            alpha * inc_rank + (1.0 - alpha) * new_rank,
            (name, "rankblend", alpha)
        ))


def combine_split_objects(train, valid):
    class Combined:
        pass

    combined = Combined()
    combined.X = {
        field: np.concatenate([
            np.asarray(train.X[field]),
            np.asarray(valid.X[field])
        ])
        for field in train.X
    }
    combined.num = {
        field: np.concatenate([
            np.asarray(train.num[field]),
            np.asarray(valid.num[field])
        ])
        for field in train.num
    }
    combined.y = np.concatenate([
        np.asarray(train.y), np.asarray(valid.y)
    ])
    combined.date = np.concatenate([
        np.asarray(train.date), np.asarray(valid.date)
    ])
    return combined


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)

X_train, X_valid, train_weights, metadata = make_history_features(
    train, valid
)

eb_valid = empirical_bayes_scores(X_valid, metadata)

linear_model, linear_mean, linear_std = fit_dense_model(
    X_train, y_train, train_weights, "linear"
)
linear_valid = predict_dense(
    linear_model, X_valid, linear_mean, linear_std
)

mlp_model, mlp_mean, mlp_std = fit_dense_model(
    X_train, y_train, train_weights, "mlp"
)
mlp_valid = predict_dense(
    mlp_model, X_valid, mlp_mean, mlp_std
)

candidates = [
    (
        "trusted_incumbent",
        inc_valid,
        ("incumbent", "raw", 1.0)
    )
]
add_family_candidates(
    candidates, "empirical_bayes", eb_valid, inc_valid, valid_users
)
add_family_candidates(
    candidates, "linear_history", linear_valid, inc_valid, valid_users
)
add_family_candidates(
    candidates, "gated_history_mlp", mlp_valid, inc_valid, valid_users
)

candidate_results = {}
best_name = None
best_scores = None
best_spec = None
best_primary = -np.inf

for name, scores, spec in candidates:
    result = evaluate(valid_users, y_valid, scores)
    primary = float(result["primary"])
    candidate_results[name] = primary
    if primary > best_primary:
        best_primary = primary
        best_name = name
        best_scores = np.asarray(scores, dtype=np.float64)
        best_spec = spec

metrics = evaluate(valid_users, y_valid, best_scores)

print("CANDIDATES " + json.dumps(candidate_results, sort_keys=True))
print("FINDINGS " + json.dumps({
    "winner": best_name,
    "trusted_incumbent": candidate_results["trusted_incumbent"],
    "empirical_bayes_raw": candidate_results["empirical_bayes_raw"],
    "linear_history_raw": candidate_results["linear_history_raw"],
    "gated_history_mlp_raw": candidate_results["gated_history_mlp_raw"],
    "history_feature_count": int(X_train.shape[1]),
    "recency_weight_min": float(train_weights.min()),
    "recency_weight_max": float(train_weights.max())
}, sort_keys=True))

test = load("test")
test_users = np.asarray(test.user_id)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

family, transform, alpha = best_spec

if family == "incumbent":
    test_scores = inc_test.copy()
else:
    combined = combine_split_objects(train, valid)

    del X_train, X_valid
    del linear_model, mlp_model

    X_combined, X_test, combined_weights, final_metadata = (
        make_history_features(combined, test)
    )
    y_combined = np.asarray(combined.y, dtype=np.int8)

    if family == "empirical_bayes":
        new_test = empirical_bayes_scores(X_test, final_metadata)
    elif family == "linear_history":
        final_model, final_mean, final_std = fit_dense_model(
            X_combined, y_combined, combined_weights, "linear"
        )
        new_test = predict_dense(
            final_model, X_test, final_mean, final_std
        )
    elif family == "gated_history_mlp":
        final_model, final_mean, final_std = fit_dense_model(
            X_combined, y_combined, combined_weights, "mlp"
        )
        new_test = predict_dense(
            final_model, X_test, final_mean, final_std
        )
    else:
        raise RuntimeError("Unknown selected family: " + family)

    if transform == "raw":
        test_scores = new_test
    elif transform == "zblend":
        test_scores = (
            float(alpha) * zscore(inc_test)
            + (1.0 - float(alpha)) * zscore(new_test)
        )
    elif transform == "rankblend":
        test_scores = (
            float(alpha) * within_user_rank(test_users, inc_test)
            + (1.0 - float(alpha)) * within_user_rank(test_users, new_test)
        )
    else:
        raise RuntimeError("Unknown transformation: " + transform)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64)
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64)
    )

wall = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(wall)
}))