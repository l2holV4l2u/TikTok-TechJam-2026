import os
import time
import json
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
ARTIFACTS = os.environ["RUN_ARTIFACTS"]
OUT_DIR = os.environ.get("ITER_OUT")

BASE_VALID_PATH = os.path.join(ARTIFACTS, "incumbent_valid_scores.npy")
BASE_TEST_PATH = os.path.join(ARTIFACTS, "incumbent_test_scores.npy")

FIELDS = ["author_id", "video_id", "tag"]
BETAS = [5.0, 20.0]
HALF_LIVES = [None, 7.0]
ALPHAS = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
EPS = 1e-4


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / sd


def day_ordinals(dates):
    dates = np.asarray(dates)
    unique_dates, inverse = np.unique(dates, return_inverse=True)
    ordinals = np.empty(len(unique_dates), dtype=np.float64)
    for i, value in enumerate(unique_dates):
        text = str(int(value))
        iso = text[:4] + "-" + text[4:6] + "-" + text[6:8]
        ordinals[i] = float(
            np.datetime64(iso, "D").astype(np.int64)
        )
    return ordinals[inverse]


def recency_weights(dates, half_life):
    if half_life is None:
        return np.ones(len(dates), dtype=np.float64)
    days = day_ordinals(dates)
    age = float(np.max(days)) - days
    return np.exp2(-age / float(half_life))


def prepare_pair_statistics(
    users,
    entities,
    labels,
    dates,
    entity_cardinality,
    half_life,
):
    users = np.asarray(users, dtype=np.int64)
    entities = np.asarray(entities, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float64)
    weights = recency_weights(dates, half_life)

    pair_keys = users * np.int64(entity_cardinality) + entities
    unique_keys, inverse = np.unique(pair_keys, return_inverse=True)

    pair_count = np.bincount(
        inverse, weights=weights, minlength=len(unique_keys)
    ).astype(np.float64)
    pair_positive = np.bincount(
        inverse, weights=weights * labels, minlength=len(unique_keys)
    ).astype(np.float64)

    entity_count = np.bincount(
        entities, weights=weights, minlength=entity_cardinality
    ).astype(np.float64)
    entity_positive = np.bincount(
        entities,
        weights=weights * labels,
        minlength=entity_cardinality,
    ).astype(np.float64)

    global_rate = float(np.sum(weights * labels) / np.sum(weights))
    entity_rate = (
        entity_positive + 20.0 * global_rate
    ) / (entity_count + 20.0)

    return unique_keys, pair_count, pair_positive, entity_rate


def score_pair_residual(
    target_users,
    target_entities,
    entity_cardinality,
    statistics,
    beta,
):
    unique_keys, pair_count, pair_positive, entity_rate = statistics

    target_users = np.asarray(target_users, dtype=np.int64)
    target_entities = np.asarray(target_entities, dtype=np.int64)
    target_keys = (
        target_users * np.int64(entity_cardinality) + target_entities
    )

    positions = np.searchsorted(unique_keys, target_keys)
    matched = positions < len(unique_keys)
    safe_positions = np.minimum(positions, len(unique_keys) - 1)
    matched &= unique_keys[safe_positions] == target_keys

    counts = np.zeros(len(target_keys), dtype=np.float64)
    positives = np.zeros(len(target_keys), dtype=np.float64)
    counts[matched] = pair_count[safe_positions[matched]]
    positives[matched] = pair_positive[safe_positions[matched]]

    prior = entity_rate[target_entities]
    posterior = (positives + float(beta) * prior) / (
        counts + float(beta)
    )

    prior = np.clip(prior, EPS, 1.0 - EPS)
    posterior = np.clip(posterior, EPS, 1.0 - EPS)

    prior_logit = np.log(prior) - np.log1p(-prior)
    posterior_logit = np.log(posterior) - np.log1p(-posterior)
    residual = posterior_logit - prior_logit

    # An unseen pair exactly backs off to the entity prior.
    residual[~matched] = 0.0
    return residual


if not os.path.exists(BASE_VALID_PATH):
    raise FileNotFoundError(BASE_VALID_PATH)
if not os.path.exists(BASE_TEST_PATH):
    raise FileNotFoundError(BASE_TEST_PATH)

train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float64)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

base_valid_raw = np.load(BASE_VALID_PATH)
if len(base_valid_raw) != len(valid_users):
    raise ValueError("Trusted incumbent validation prediction length mismatch")

base_valid = standardize(base_valid_raw)
base_metrics = evaluate(valid_users, y_valid, base_valid)
base_primary = float(base_metrics["primary"])

residuals = {}
coverage = {}

for half_life in HALF_LIVES:
    hl_name = "all" if half_life is None else "hl7"
    for field in FIELDS:
        cardinality = int(FEATURE_CARDINALITIES[field])
        stats = prepare_pair_statistics(
            train.user_id,
            train.X[field],
            y_train,
            train.date,
            cardinality,
            half_life,
        )
        for beta in BETAS:
            key = (hl_name, field, int(beta))
            residual = score_pair_residual(
                valid.user_id,
                valid.X[field],
                cardinality,
                stats,
                beta,
            )
            residuals[key] = standardize(residual)
            coverage[key] = float(np.mean(np.abs(residual) > 1e-12))


candidate_components = {}

for hl_name in ["all", "hl7"]:
    for beta in [5, 20]:
        for field in FIELDS:
            name = f"{hl_name}_b{beta}_{field}"
            candidate_components[name] = [
                (hl_name, field, beta)
            ]

        candidate_components[f"{hl_name}_b{beta}_author_tag"] = [
            (hl_name, "author_id", beta),
            (hl_name, "tag", beta),
        ]
        candidate_components[f"{hl_name}_b{beta}_author_video"] = [
            (hl_name, "author_id", beta),
            (hl_name, "video_id", beta),
        ]
        candidate_components[f"{hl_name}_b{beta}_all"] = [
            (hl_name, "author_id", beta),
            (hl_name, "video_id", beta),
            (hl_name, "tag", beta),
        ]

candidate_scores = {"incumbent": base_primary}
best_name = "incumbent"
best_alpha = 0.0
best_component_keys = []
best_scores = base_valid.copy()
best_metrics = base_metrics
best_observed_primary = base_primary
best_observed = ("incumbent", 0.0, [], base_valid.copy(), base_metrics)

for name, component_keys in candidate_components.items():
    combined_residual = np.zeros(len(valid_users), dtype=np.float64)
    for key in component_keys:
        combined_residual += residuals[key]
    combined_residual = standardize(combined_residual)

    local_best = -np.inf
    for alpha in ALPHAS:
        scores = base_valid + float(alpha) * combined_residual
        metrics = evaluate(valid_users, y_valid, scores)
        primary = float(metrics["primary"])
        local_best = max(local_best, primary)

        if primary > best_observed_primary:
            best_observed_primary = primary
            best_observed = (
                name,
                float(alpha),
                list(component_keys),
                scores.copy(),
                metrics,
            )

    candidate_scores[name] = local_best

# Require a modest validation margin before modifying the trusted incumbent.
# This avoids replacing it with a target-encoding blend selected from many
# closely spaced noisy candidates.
if best_observed_primary >= base_primary + 0.001:
    (
        best_name,
        best_alpha,
        best_component_keys,
        best_scores,
        best_metrics,
    ) = best_observed

valid_scores = np.asarray(best_scores, dtype=np.float64)

if OUT_DIR:
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(
        os.path.join(OUT_DIR, "scores_valid.npy"),
        valid_scores,
    )

coverage_summary = {
    f"{k[0]}:{k[1]}:b{k[2]}": round(v, 4)
    for k, v in coverage.items()
}
print(
    "FINDINGS "
    + json.dumps(
        {
            "incumbent_primary": base_primary,
            "best_observed_primary": best_observed_primary,
            "selected": best_name,
            "selected_alpha": best_alpha,
            "pair_coverage": coverage_summary,
        },
        sort_keys=True,
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        {
            key: round(float(value), 6)
            for key, value in sorted(
                candidate_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        },
        sort_keys=True,
    )
)

base_test_raw = np.load(BASE_TEST_PATH)

if best_name == "incumbent":
    test_scores = standardize(base_test_raw)
else:
    # Refit the exact selected statistic recipe on train + validation so the
    # histories reach the day immediately preceding the hidden test split.
    test = load("test")

    combined_users = np.concatenate(
        (
            np.asarray(train.user_id, dtype=np.int64),
            np.asarray(valid.user_id, dtype=np.int64),
        )
    )
    combined_labels = np.concatenate(
        (y_train, y_valid.astype(np.float64))
    )
    combined_dates = np.concatenate(
        (
            np.asarray(train.date),
            np.asarray(valid.date),
        )
    )

    test_residual = np.zeros(len(test.user_id), dtype=np.float64)

    selected_half_life_name = best_component_keys[0][0]
    selected_half_life = (
        None if selected_half_life_name == "all" else 7.0
    )

    for _, field, beta in best_component_keys:
        cardinality = int(FEATURE_CARDINALITIES[field])
        combined_entities = np.concatenate(
            (
                np.asarray(train.X[field], dtype=np.int64),
                np.asarray(valid.X[field], dtype=np.int64),
            )
        )

        stats = prepare_pair_statistics(
            combined_users,
            combined_entities,
            combined_labels,
            combined_dates,
            cardinality,
            selected_half_life,
        )
        component = score_pair_residual(
            test.user_id,
            test.X[field],
            cardinality,
            stats,
            beta,
        )
        test_residual += standardize(component)

    test_residual = standardize(test_residual)
    test_scores = (
        standardize(base_test_raw)
        + float(best_alpha) * test_residual
    )

if OUT_DIR:
    np.save(
        os.path.join(OUT_DIR, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
result = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(result))