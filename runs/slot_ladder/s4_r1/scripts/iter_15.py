import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SEED = 18437
BATCH_SIZE = 8192
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))

train = load("train")
valid = load("valid")
test = load("test")

y_all = np.asarray(train.y, dtype=np.float32)
train_dates = np.asarray(train.date, dtype=np.int64)

# Base models use 20220409-18. The gate is trained on 19-20 and assessed on
# 21, so neither its architecture nor the chosen expert is selected on valid.
base_mask = train_dates <= 20220418
gate_train_mask = (train_dates >= 20220419) & (train_dates <= 20220420)
gate_check_mask = train_dates == 20220421
gate_all_mask = gate_train_mask | gate_check_mask

base_idx = np.flatnonzero(base_mask)
gate_train_idx = np.flatnonzero(gate_train_mask)
gate_check_idx = np.flatnonzero(gate_check_mask)
gate_all_idx = np.flatnonzero(gate_all_mask)

cards = [int(FEATURE_CARDINALITIES[f]) for f in FIELDS]
offsets = np.cumsum(np.r_[0, np.asarray(cards[:-1], dtype=np.int64)])
total_cardinality = int(sum(cards))


def make_matrix(split, indices=None):
    columns = [np.asarray(split.X[f], dtype=np.int64) for f in FIELDS]
    matrix = np.stack(columns, axis=1)
    if indices is not None:
        matrix = matrix[indices]
    matrix = matrix + offsets[None, :]
    return np.ascontiguousarray(matrix, dtype=np.int64)


x_all = make_matrix(train)
x_valid = make_matrix(valid)
x_test = make_matrix(test)


class FactorizationMachine(nn.Module):
    def __init__(self, cardinality, n_fields, rank=16):
        super().__init__()
        self.linear = nn.Embedding(cardinality, 1)
        self.embedding = nn.Embedding(cardinality, rank)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.015)
        self.n_fields = n_fields

    def forward(self, x):
        linear = self.linear(x).squeeze(-1).sum(dim=1)
        latent = self.embedding(x)
        summed = latent.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - latent.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear + interaction


def fit_fm(indices, half_life=None, epochs=4, seed=SEED):
    torch.manual_seed(seed)
    model = FactorizationMachine(
        total_cardinality, len(FIELDS), rank=16
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0012
    )

    local_dates = train_dates[indices]
    if half_life is None:
        sample_weights = np.ones(len(indices), dtype=np.float32)
    else:
        sample_weights = np.exp2(
            (local_dates - local_dates.max()).astype(np.float32)
            / float(half_life)
        )
        sample_weights /= max(float(sample_weights.mean()), 1e-8)
        sample_weights = sample_weights.astype(np.float32)

    rng = np.random.default_rng(seed)
    model.train()

    for _ in range(epochs):
        order = rng.permutation(len(indices))
        for lo in range(0, len(order), BATCH_SIZE):
            local = order[lo:lo + BATCH_SIZE]
            rows = indices[local]

            xb = torch.from_numpy(x_all[rows])
            target = torch.from_numpy(y_all[rows])
            weight = torch.from_numpy(sample_weights[local])

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            loss = (losses * weight).sum() / weight.sum()
            loss.backward()
            optimizer.step()

    return model


@torch.inference_mode()
def predict_fm(model, matrix):
    model.eval()
    result = np.empty(len(matrix), dtype=np.float64)
    for lo in range(0, len(matrix), 32768):
        hi = min(lo + 32768, len(matrix))
        result[lo:hi] = model(
            torch.from_numpy(matrix[lo:hi])
        ).cpu().numpy().astype(np.float64)
    return result


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


class EmpiricalBayesExpert:
    def __init__(self, row_indices, half_life=4.0):
        dates = train_dates[row_indices]
        weights = np.exp2(
            (dates - dates.max()).astype(np.float64) / half_life
        )
        weights /= max(weights.mean(), 1e-12)
        labels = y_all[row_indices].astype(np.float64)

        self.global_rate = float(
            np.sum(weights * labels) / np.sum(weights)
        )
        self.tables = {}

        # Different shrinkage levels reflect entity sparsity. The resulting
        # score is a hierarchical additive posterior log-odds estimate.
        strengths = {
            "user_id": 24.0,
            "video_id": 35.0,
            "author_id": 45.0,
            "tab": 120.0,
            "tag": 90.0,
            "duration_bucket": 120.0,
        }

        for field, strength in strengths.items():
            ids = np.asarray(train.X[field], dtype=np.int64)[row_indices]
            card = int(FEATURE_CARDINALITIES[field])
            count = np.bincount(
                ids, weights=weights, minlength=card
            ).astype(np.float64)
            positive = np.bincount(
                ids, weights=weights * labels, minlength=card
            ).astype(np.float64)
            posterior = (
                positive + strength * self.global_rate
            ) / (count + strength)
            reliability = count / (count + strength)
            self.tables[field] = (
                logit(posterior) - logit(self.global_rate),
                reliability,
            )

    def predict(self, split, indices=None):
        n = len(split.user_id) if indices is None else len(indices)
        result = np.full(n, logit(self.global_rate), dtype=np.float64)

        field_weights = {
            "user_id": 0.65,
            "video_id": 1.00,
            "author_id": 0.75,
            "tab": 0.45,
            "tag": 0.35,
            "duration_bucket": 0.30,
        }

        for field, coefficient in field_weights.items():
            ids = np.asarray(split.X[field], dtype=np.int64)
            if indices is not None:
                ids = ids[indices]
            effect, reliability = self.tables[field]
            result += coefficient * effect[ids] * np.sqrt(reliability[ids])
        return result


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    ordered_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_users[1:] != ordered_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.int64) - repeated_starts

    ranked = np.full(n, 0.5, dtype=np.float64)
    multiple = repeated_lengths > 1
    ranked[multiple] = (
        positions[multiple] / (repeated_lengths[multiple] - 1.0)
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def history_statistics(row_indices):
    user_card = int(FEATURE_CARDINALITIES["user_id"])
    users = np.asarray(train.X["user_id"], dtype=np.int64)[row_indices]
    labels = y_all[row_indices].astype(np.float64)

    counts = np.bincount(users, minlength=user_card).astype(np.float64)
    positives = np.bincount(
        users, weights=labels, minlength=user_card
    ).astype(np.float64)

    global_rate = float(labels.mean())
    rates = (positives + 5.0 * global_rate) / (counts + 5.0)
    return counts, positives, rates


def build_gate_features(
    split,
    expert_scores,
    history,
    indices=None,
):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    if indices is not None:
        users = users[indices]

    n = len(users)
    ranks = [
        within_user_rank(users, score) for score in expert_scores
    ]

    max_user = int(FEATURE_CARDINALITIES["user_id"])
    list_sizes_table = np.bincount(
        users, minlength=max_user
    ).astype(np.float64)
    list_size = list_sizes_table[users]

    count, positives, rate = history
    safe_users = np.clip(users, 0, len(count) - 1)

    def field_values(name):
        values = np.asarray(split.X[name], dtype=np.float64)
        return values if indices is None else values[indices]

    duration = np.asarray(split.num["duration_ms"], dtype=np.float64)
    if indices is not None:
        duration = duration[indices]
    duration = np.log1p(np.maximum(np.nan_to_num(duration), 0.0))

    uniform, recent, bayes = [
        np.asarray(x, dtype=np.float64) for x in expert_scores
    ]

    features = np.column_stack([
        uniform,
        recent,
        bayes,
        ranks[0],
        ranks[1],
        ranks[2],
        recent - uniform,
        bayes - recent,
        np.log1p(list_size),
        np.log1p(count[safe_users]),
        np.log1p(positives[safe_users]),
        rate[safe_users],
        field_values("tab"),
        field_values("tag"),
        field_values("duration_bucket"),
        field_values("hour"),
        duration,
    ])
    return np.asarray(features, dtype=np.float32)


def sorted_ranking_data(features, users, labels=None):
    users = np.asarray(users, dtype=np.int64)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    _, group = np.unique(sorted_users, return_counts=True)
    sorted_labels = None if labels is None else np.asarray(labels)[order]
    return features[order], sorted_labels, group, order


def fit_gate(features, users, labels, rounds=170):
    sx, sy, groups, _ = sorted_ranking_data(
        features, users, labels
    )
    dataset = lgb.Dataset(
        sx,
        label=sy.astype(np.int32),
        group=groups,
        free_raw_data=False,
    )
    params = {
        "objective": "lambdarank",
        "metric": "None",
        "label_gain": [0, 1],
        "learning_rate": 0.035,
        "num_leaves": 15,
        "max_depth": 5,
        "min_data_in_leaf": 180,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 2.0,
        "max_position": 5,
        "num_threads": min(8, os.cpu_count() or 1),
        "seed": SEED,
        "verbose": -1,
    }
    return lgb.train(params, dataset, num_boost_round=rounds)


# ------------------------------------------------------------------
# Out-of-fold experts: all are fitted strictly before their gate rows.
# ------------------------------------------------------------------
uniform_early = fit_fm(
    base_idx, half_life=None, epochs=4, seed=SEED + 1
)
recent_early = fit_fm(
    base_idx, half_life=4.0, epochs=4, seed=SEED + 2
)
bayes_early = EmpiricalBayesExpert(base_idx, half_life=4.0)

x_gate_train = x_all[gate_train_idx]
x_gate_check = x_all[gate_check_idx]
x_gate_all = x_all[gate_all_idx]

early_train_scores = [
    predict_fm(uniform_early, x_gate_train),
    predict_fm(recent_early, x_gate_train),
    bayes_early.predict(train, gate_train_idx),
]
early_check_scores = [
    predict_fm(uniform_early, x_gate_check),
    predict_fm(recent_early, x_gate_check),
    bayes_early.predict(train, gate_check_idx),
]
early_all_scores = [
    predict_fm(uniform_early, x_gate_all),
    predict_fm(recent_early, x_gate_all),
    bayes_early.predict(train, gate_all_idx),
]

early_history = history_statistics(base_idx)

gate_train_features = build_gate_features(
    train, early_train_scores, early_history, gate_train_idx
)
gate_check_features = build_gate_features(
    train, early_check_scores, early_history, gate_check_idx
)

gate_probe = fit_gate(
    gate_train_features,
    np.asarray(train.user_id)[gate_train_idx],
    y_all[gate_train_idx],
    rounds=150,
)
gate_check_scores = gate_probe.predict(gate_check_features)

check_users = np.asarray(train.user_id)[gate_check_idx]
check_labels = y_all[gate_check_idx]
holdout_metrics = {
    "uniform_fm": evaluate(
        check_users, check_labels, early_check_scores[0]
    )["primary"],
    "recent_fm": evaluate(
        check_users, check_labels, early_check_scores[1]
    )["primary"],
    "empirical_bayes": evaluate(
        check_users, check_labels, early_check_scores[2]
    )["primary"],
    "conditional_gate": evaluate(
        check_users, check_labels, gate_check_scores
    )["primary"],
}
selected_family = max(holdout_metrics, key=holdout_metrics.get)

print("FINDINGS " + json.dumps({
    "train_day21_primary": {
        k: round(float(v), 6) for k, v in holdout_metrics.items()
    },
    "selected_without_validation": selected_family,
}, sort_keys=True))

# Refit the gate on every out-of-fold row after its family has been selected.
gate_all_features = build_gate_features(
    train, early_all_scores, early_history, gate_all_idx
)
gate_model = fit_gate(
    gate_all_features,
    np.asarray(train.user_id)[gate_all_idx],
    y_all[gate_all_idx],
    rounds=170,
)

del uniform_early, recent_early, bayes_early, gate_probe
del gate_train_features, gate_check_features, gate_all_features


# ------------------------------------------------------------------
# Full-TRAIN experts. Hyperparameters and gate were fixed above.
# ------------------------------------------------------------------
full_idx = np.arange(len(y_all), dtype=np.int64)

uniform_full = fit_fm(
    full_idx, half_life=None, epochs=5, seed=SEED + 11
)
recent_full = fit_fm(
    full_idx, half_life=4.0, epochs=5, seed=SEED + 12
)
bayes_full = EmpiricalBayesExpert(full_idx, half_life=4.0)

valid_experts = [
    predict_fm(uniform_full, x_valid),
    predict_fm(recent_full, x_valid),
    bayes_full.predict(valid),
]
test_experts = [
    predict_fm(uniform_full, x_test),
    predict_fm(recent_full, x_test),
    bayes_full.predict(test),
]

full_history = history_statistics(full_idx)
valid_gate_features = build_gate_features(
    valid, valid_experts, full_history
)
test_gate_features = build_gate_features(
    test, test_experts, full_history
)

valid_gate = gate_model.predict(valid_gate_features)
test_gate = gate_model.predict(test_gate_features)

own_valid_by_family = {
    "uniform_fm": valid_experts[0],
    "recent_fm": valid_experts[1],
    "empirical_bayes": valid_experts[2],
    "conditional_gate": valid_gate,
}
own_test_by_family = {
    "uniform_fm": test_experts[0],
    "recent_fm": test_experts[1],
    "empirical_bayes": test_experts[2],
    "conditional_gate": test_gate,
}

own_valid = own_valid_by_family[selected_family]
own_test = own_test_by_family[selected_family]


# A fixed blend weight is specified before validation evaluation. Rank
# normalization makes heterogeneous expert scales transferable to test.
shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

BLEND_ALPHA = 0.35

candidate_values = {}
for name, scores in own_valid_by_family.items():
    candidate_values[name + "_raw"] = float(
        evaluate(valid.user_id, valid.y, scores)["primary"]
    )
    blended = (
        (1.0 - BLEND_ALPHA) * inc_valid_rank
        + BLEND_ALPHA * within_user_rank(valid.user_id, scores)
    )
    candidate_values[name + "_fixed_blend"] = float(
        evaluate(valid.user_id, valid.y, blended)["primary"]
    )

valid_scores = (
    (1.0 - BLEND_ALPHA) * inc_valid_rank
    + BLEND_ALPHA * within_user_rank(valid.user_id, own_valid)
)
test_scores = (
    (1.0 - BLEND_ALPHA) * inc_test_rank
    + BLEND_ALPHA * within_user_rank(test.user_id, own_test)
)

metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(
    {k: round(v, 7) for k, v in candidate_values.items()},
    sort_keys=True,
))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(own_valid, dtype=np.float64),
    )
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