import os
import time
import json
import random
import gc
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 240531

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))

THREADS = max(1, min(12, os.cpu_count() or 1))


def rank_percentile(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, users))
    sorted_users = users[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]
    start_idx = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_idx = np.flatnonzero(ends)
    sizes = np.diff(np.r_[np.int64(-1), end_idx])
    row_sizes = np.repeat(sizes, sizes)

    position = np.arange(n, dtype=np.int64) - start_idx
    ranked = (position.astype(np.float64) + 0.5) / np.maximum(row_sizes, 1)

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


train = load("train")
valid = load("valid")
test = load("test")

cat_fields = [
    name for name in train.X.keys()
    if name != "is_lowactive_period"
]
num_fields = list(train.num.keys())

cat_indices = list(range(len(cat_fields)))


def make_tree_matrix(split):
    categorical = np.column_stack([
        np.asarray(split.X[name], dtype=np.float32)
        for name in cat_fields
    ])

    numeric_columns = []
    for name in num_fields:
        values = np.asarray(split.num[name], dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = np.log1p(np.maximum(values, 0.0))
        numeric_columns.append(values.astype(np.float32))

    # Time-of-feed context is known at scoring time and can interact with
    # user/item attributes without using any outcome from the row.
    time_ms = np.asarray(split.time_ms, dtype=np.int64)
    hour_sin = np.sin(
        2.0 * np.pi * np.asarray(split.X["hour"], dtype=np.float64) / 24.0
    ).astype(np.float32)
    hour_cos = np.cos(
        2.0 * np.pi * np.asarray(split.X["hour"], dtype=np.float64) / 24.0
    ).astype(np.float32)
    time_phase = ((time_ms // 60000) % 60).astype(np.float32) / 60.0

    numeric = np.column_stack(
        numeric_columns + [hour_sin, hour_cos, time_phase]
    ).astype(np.float32)

    return np.column_stack([categorical, numeric]).astype(np.float32)


x_train = make_tree_matrix(train)
x_valid = make_tree_matrix(valid)
x_test = make_tree_matrix(test)

y_train = np.asarray(train.y, dtype=np.float32)
train_dates = np.asarray(train.date, dtype=np.int64)
unique_dates = np.sort(np.unique(train_dates))

# Estimate p(context | late train) / p(context | early train) entirely inside
# train. We use separated endpoint windows so the domain classifier learns
# persistent temporal movement rather than adjacent-day noise.
early_dates = unique_dates[:4]
late_dates = unique_dates[-4:]
domain_mask = np.isin(train_dates, early_dates) | np.isin(
    train_dates, late_dates
)
domain_index = np.flatnonzero(domain_mask)
domain_y = np.isin(train_dates[domain_index], late_dates).astype(np.float32)

n_early = float(np.sum(domain_y == 0))
n_late = float(np.sum(domain_y == 1))
domain_weight = np.where(
    domain_y > 0,
    0.5 * len(domain_y) / max(n_late, 1.0),
    0.5 * len(domain_y) / max(n_early, 1.0),
).astype(np.float32)

domain_dataset = lgb.Dataset(
    x_train[domain_index],
    label=domain_y,
    weight=domain_weight,
    categorical_feature=cat_indices,
    free_raw_data=True,
)

domain_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.06,
    "num_leaves": 31,
    "min_data_in_leaf": 1200,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.5,
    "lambda_l2": 5.0,
    "max_bin": 127,
    "num_threads": THREADS,
    "seed": SEED,
    "verbose": -1,
}

domain_model = lgb.train(
    domain_params,
    domain_dataset,
    num_boost_round=180,
)

domain_train_prob = np.clip(
    domain_model.predict(x_train), 1e-4, 1.0 - 1e-4
)
domain_valid_prob = np.clip(
    domain_model.predict(x_valid), 1e-4, 1.0 - 1e-4
)
domain_test_prob = np.clip(
    domain_model.predict(x_test), 1e-4, 1.0 - 1e-4
)

# Balanced domain-class training makes posterior odds an estimate of the
# late/early density ratio. Tempering limits variance from imperfect overlap.
density_ratio = domain_train_prob / (1.0 - domain_train_prob)
density_weight = np.clip(density_ratio, 0.20, 5.0)
density_weight = np.sqrt(density_weight)
density_weight /= np.mean(density_weight)

# A mild explicit proximity prior complements the context density ratio:
# contexts can remain common while their conditional response drifts.
date_age = (
    int(unique_dates[-1]) - train_dates
).astype(np.float64)
recency_weight = np.power(0.5, np.maximum(date_age, 0.0) / 8.0)
recency_weight /= np.mean(recency_weight)

outcome_weight = density_weight * recency_weight
outcome_weight = np.clip(outcome_weight, 0.10, 6.0)
outcome_weight /= np.mean(outcome_weight)
outcome_weight = outcome_weight.astype(np.float32)

del domain_dataset, domain_model
gc.collect()


# -------------------------------------------------------------------------
# Family 1: interaction-boosted categorical trees.
# -------------------------------------------------------------------------
boost_dataset = lgb.Dataset(
    x_train,
    label=y_train,
    weight=outcome_weight,
    categorical_feature=cat_indices,
    free_raw_data=False,
)

boost_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.035,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 900,
    "feature_fraction": 0.82,
    "bagging_fraction": 0.84,
    "bagging_freq": 1,
    "lambda_l1": 0.25,
    "lambda_l2": 4.0,
    "max_bin": 127,
    "num_threads": THREADS,
    "seed": SEED + 1,
    "verbose": -1,
}

boost_model = lgb.train(
    boost_params,
    boost_dataset,
    num_boost_round=420,
)
boost_valid = boost_model.predict(x_valid)
boost_test = boost_model.predict(x_test)

del boost_model
gc.collect()


# -------------------------------------------------------------------------
# Family 2: bagged random forest. Unlike gradient boosting, predictions are
# averages of independently randomized interaction trees, reducing sensitivity
# to idiosyncratic days and high-variance density weights.
# -------------------------------------------------------------------------
rf_params = {
    "boosting_type": "rf",
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 1.0,
    "num_leaves": 63,
    "max_depth": 12,
    "min_data_in_leaf": 700,
    "feature_fraction": 0.68,
    "bagging_fraction": 0.62,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 3.0,
    "max_bin": 127,
    "num_threads": THREADS,
    "seed": SEED + 2,
    "verbose": -1,
}

rf_model = lgb.train(
    rf_params,
    boost_dataset,
    num_boost_round=240,
)
rf_valid = rf_model.predict(x_valid)
rf_test = rf_model.predict(x_test)

del rf_model, boost_dataset
gc.collect()


# -------------------------------------------------------------------------
# Family 3: additive categorical GAM with train-quantile numeric bins.
# This cannot form feature interactions: each field contributes a scalar
# effect, providing a lower-variance drift-robust contrast to both tree models.
# -------------------------------------------------------------------------
NUM_BINS = 32
numeric_edges = {}

for name in num_fields:
    values = np.asarray(train.num[name], dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.log1p(np.maximum(values, 0.0))
    edges = np.unique(
        np.quantile(values, np.linspace(0.0, 1.0, NUM_BINS + 1)[1:-1])
    )
    numeric_edges[name] = edges


def make_gam_fields(split):
    fields = [
        np.asarray(split.X[name], dtype=np.int64)
        for name in cat_fields
    ]
    cardinalities = [
        int(FEATURE_CARDINALITIES[name])
        for name in cat_fields
    ]

    for name in num_fields:
        values = np.asarray(split.num[name], dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = np.log1p(np.maximum(values, 0.0))
        bins = np.searchsorted(
            numeric_edges[name], values, side="right"
        ).astype(np.int64)
        fields.append(bins)
        cardinalities.append(len(numeric_edges[name]) + 1)

    return fields, cardinalities


gam_train_fields, gam_cardinalities = make_gam_fields(train)
gam_valid_fields, _ = make_gam_fields(valid)
gam_test_fields, _ = make_gam_fields(test)


class AdditiveGAM(nn.Module):
    def __init__(self, cardinalities):
        super().__init__()
        self.effects = nn.ModuleList([
            nn.Embedding(cardinality, 1)
            for cardinality in cardinalities
        ])
        self.bias = nn.Parameter(torch.zeros(1))
        for embedding in self.effects:
            nn.init.zeros_(embedding.weight)

    def forward(self, fields):
        result = self.bias.expand(fields[0].shape[0])
        for embedding, values in zip(self.effects, fields):
            result = result + embedding(values).squeeze(-1)
        return result


gam_model = AdditiveGAM(gam_cardinalities)
gam_optimizer = torch.optim.AdamW(
    gam_model.parameters(), lr=0.018, weight_decay=2e-5
)

gam_train_tensors = [
    torch.from_numpy(values) for values in gam_train_fields
]
gam_y_tensor = torch.from_numpy(y_train)
gam_w_tensor = torch.from_numpy(outcome_weight)

rng = np.random.default_rng(SEED + 3)
batch_size = 32768
gam_losses = []

gam_model.train()
for epoch in range(5):
    permutation = rng.permutation(len(train))
    total_loss = 0.0
    total_weight = 0.0

    for start in range(0, len(train), batch_size):
        index = permutation[start:start + batch_size]
        batch_fields = [
            values[index] for values in gam_train_tensors
        ]
        logits = gam_model(batch_fields)
        element_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, gam_y_tensor[index], reduction="none"
        )
        denominator = gam_w_tensor[index].sum().clamp_min(1.0)
        loss = (
            element_loss * gam_w_tensor[index]
        ).sum() / denominator

        gam_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gam_optimizer.step()

        total_loss += float(
            (element_loss * gam_w_tensor[index]).sum().detach()
        )
        total_weight += float(denominator.detach())

    gam_losses.append(total_loss / max(total_weight, 1.0))


@torch.no_grad()
def predict_gam(model, fields):
    model.eval()
    tensors = [torch.from_numpy(values) for values in fields]
    predictions = np.empty(len(fields[0]), dtype=np.float64)
    batch = 65536
    for start in range(0, len(predictions), batch):
        end = min(start + batch, len(predictions))
        predictions[start:end] = model([
            values[start:end] for values in tensors
        ]).numpy()
    return predictions


gam_valid = predict_gam(gam_model, gam_valid_fields)
gam_test = predict_gam(gam_model, gam_test_fields)

del gam_model, gam_train_tensors
gc.collect()


raw_valid = {
    "density_boosted_trees": boost_valid,
    "density_random_forest": rf_valid,
    "density_additive_gam": gam_valid,
}
raw_test = {
    "density_boosted_trees": boost_test,
    "density_random_forest": rf_test,
    "density_additive_gam": gam_test,
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

raw_valid_rank = {
    name: rank_percentile(valid.user_id, scores)
    for name, scores in raw_valid.items()
}
raw_test_rank = {
    name: rank_percentile(test.user_id, scores)
    for name, scores in raw_test.items()
}

candidate_valid = {"incumbent": inc_valid}
candidate_test = {"incumbent": inc_test}
candidate_own_raw = {}

for name in raw_valid:
    candidate_valid[name + "_standalone"] = raw_valid[name]
    candidate_test[name + "_standalone"] = raw_test[name]
    candidate_own_raw[name + "_standalone"] = raw_valid[name]

    for alpha in (0.03, 0.06, 0.10, 0.15, 0.22, 0.30, 0.40):
        key = f"{name}_incblend_{alpha:.2f}"
        candidate_valid[key] = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * raw_valid_rank[name]
        )
        candidate_test[key] = (
            (1.0 - alpha) * inc_test_rank
            + alpha * raw_test_rank[name]
        )
        candidate_own_raw[key] = raw_valid[name]

# Rank aggregation exploits the different variance profiles of additive,
# boosted, and independently bagged estimators.
family_ensemble_valid = np.mean(
    np.stack(list(raw_valid_rank.values()), axis=0), axis=0
)
family_ensemble_test = np.mean(
    np.stack(list(raw_test_rank.values()), axis=0), axis=0
)

candidate_valid["density_family_ensemble"] = family_ensemble_valid
candidate_test["density_family_ensemble"] = family_ensemble_test
candidate_own_raw["density_family_ensemble"] = family_ensemble_valid

for alpha in (0.04, 0.08, 0.12, 0.18, 0.25, 0.35):
    key = f"density_family_ensemble_incblend_{alpha:.2f}"
    candidate_valid[key] = (
        (1.0 - alpha) * inc_valid_rank
        + alpha * family_ensemble_valid
    )
    candidate_test[key] = (
        (1.0 - alpha) * inc_test_rank
        + alpha * family_ensemble_test
    )
    candidate_own_raw[key] = family_ensemble_valid

candidate_metrics = {
    name: evaluate(valid.user_id, valid.y, scores)
    for name, scores in candidate_valid.items()
}
best_name = max(
    candidate_metrics,
    key=lambda name: float(candidate_metrics[name]["primary"])
)
best_metrics = candidate_metrics[best_name]
best_valid = candidate_valid[best_name]
best_test = candidate_test[best_name]

new_family_best_name = max(
    (name for name in candidate_metrics if name != "incumbent"),
    key=lambda name: float(candidate_metrics[name]["primary"])
)

print("CANDIDATES " + json.dumps({
    name: float(metrics["primary"])
    for name, metrics in candidate_metrics.items()
}, sort_keys=True))

print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "best_new_candidate": new_family_best_name,
    "best_new_primary": float(
        candidate_metrics[new_family_best_name]["primary"]
    ),
    "gam_losses": gam_losses,
    "density_weight_q01_q50_q99": [
        float(v) for v in np.quantile(
            density_weight, [0.01, 0.50, 0.99]
        )
    ],
    "outcome_weight_q01_q50_q99": [
        float(v) for v in np.quantile(
            outcome_weight, [0.01, 0.50, 0.99]
        )
    ],
    "domain_probability_train_mean": float(
        np.mean(domain_train_prob)
    ),
    "domain_probability_valid_mean": float(
        np.mean(domain_valid_prob)
    ),
    "domain_probability_test_mean": float(
        np.mean(domain_test_prob)
    ),
    "rank_correlations_with_incumbent": {
        name: float(np.corrcoef(
            inc_valid_rank, raw_valid_rank[name]
        )[0, 1])
        for name in raw_valid_rank
    },
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64)
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64)
    )

    if best_name == "incumbent":
        own_raw_to_save = candidate_own_raw[new_family_best_name]
    else:
        own_raw_to_save = candidate_own_raw[best_name]

    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(own_raw_to_save, dtype=np.float64)
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))