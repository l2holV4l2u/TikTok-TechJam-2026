import os
import gc
import json
import time
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()
OUT_DIR = os.environ.get("ITER_OUT")
ARTIFACTS = os.environ.get("RUN_ARTIFACTS", "")

if OUT_DIR:
    os.makedirs(OUT_DIR, exist_ok=True)

FIELDS = ["author_id", "video_id", "tag", "duration_bucket"]
BETAS = [3.0, 10.0, 30.0]
BLEND_WEIGHTS = [0.0, 0.08, 0.15, 0.25, 0.35, 0.50, 0.70]
CONFIGS = [
    ("author_id",),
    ("video_id",),
    ("tag",),
    ("author_id", "tag"),
    ("author_id", "video_id"),
    ("author_id", "video_id", "tag"),
    ("author_id", "tag", "duration_bucket"),
]


def normalized(x):
    x = np.asarray(x, dtype=np.float64)
    mean = float(np.mean(x))
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    return (x - mean) / sd


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


class PairStatistics:
    """
    Sparse empirical-Bayes user x entity statistics.

    The entity posterior is the fallback expectation. The returned feature is
    the log-odds residual of the user-specific posterior relative to that
    entity expectation, so global popularity already represented by the
    incumbent is not added a second time.
    """

    def __init__(
        self,
        fit_users,
        fit_entities,
        fit_labels,
        entity_cardinality,
        entity_alpha=30.0,
    ):
        users = np.asarray(fit_users, dtype=np.int64)
        entities = np.asarray(fit_entities, dtype=np.int64)
        labels = np.asarray(fit_labels, dtype=np.float64)

        self.cardinality = int(entity_cardinality)
        prior = float(np.mean(labels))

        entity_count = np.bincount(
            entities, minlength=self.cardinality
        ).astype(np.float64)
        entity_pos = np.bincount(
            entities, weights=labels, minlength=self.cardinality
        ).astype(np.float64)
        self.entity_rate = (
            entity_pos + float(entity_alpha) * prior
        ) / (entity_count + float(entity_alpha))

        keys = users * np.int64(self.cardinality) + entities
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        pair_count = np.bincount(inverse).astype(np.float64)
        pair_pos = np.bincount(
            inverse, weights=labels
        ).astype(np.float64)

        self.keys = unique_keys
        self.count = pair_count
        self.positives = pair_pos

        del keys, inverse, entity_count, entity_pos
        gc.collect()

    def query(self, query_users, query_entities, beta):
        users = np.asarray(query_users, dtype=np.int64)
        entities = np.asarray(query_entities, dtype=np.int64)
        query_keys = users * np.int64(self.cardinality) + entities

        positions = np.searchsorted(self.keys, query_keys)
        safe_positions = np.minimum(positions, len(self.keys) - 1)
        seen = (
            (positions < len(self.keys))
            & (self.keys[safe_positions] == query_keys)
        )

        baseline = self.entity_rate[entities]
        result = np.zeros(len(users), dtype=np.float64)

        if np.any(seen):
            pos = safe_positions[seen]
            base_seen = baseline[seen]
            posterior = (
                self.positives[pos] + float(beta) * base_seen
            ) / (self.count[pos] + float(beta))
            result[seen] = logit(posterior) - logit(base_seen)

        return result, seen


def build_residual_bank(fit_split, labels, query_split):
    bank = {}
    coverage = {}

    fit_users = np.asarray(fit_split.user_id, dtype=np.int64)
    query_users = np.asarray(query_split.user_id, dtype=np.int64)

    for field in FIELDS:
        stats = PairStatistics(
            fit_users=fit_users,
            fit_entities=np.asarray(fit_split.X[field], dtype=np.int64),
            fit_labels=labels,
            entity_cardinality=int(FEATURE_CARDINALITIES[field]),
            entity_alpha=30.0,
        )

        for beta in BETAS:
            residual, seen = stats.query(
                query_users,
                np.asarray(query_split.X[field], dtype=np.int64),
                beta,
            )
            bank[(field, beta)] = residual
            coverage[field] = float(np.mean(seen))

        del stats
        gc.collect()

    return bank, coverage


def combine_residuals(bank, config, beta):
    parts = [normalized(bank[(field, beta)]) for field in config]
    if len(parts) == 1:
        return parts[0]
    return np.mean(np.column_stack(parts), axis=1)


def make_combined_split(train, valid):
    class Combined:
        pass

    combined = Combined()
    combined.user_id = np.concatenate([
        np.asarray(train.user_id, dtype=np.int64),
        np.asarray(valid.user_id, dtype=np.int64),
    ])
    combined.X = {}
    for field in FIELDS:
        combined.X[field] = np.concatenate([
            np.asarray(train.X[field], dtype=np.int64),
            np.asarray(valid.X[field], dtype=np.int64),
        ])
    return combined


inc_valid_path = os.path.join(
    ARTIFACTS, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    ARTIFACTS, "incumbent_test_scores.npy"
)

if not (
    ARTIFACTS
    and os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise RuntimeError(
        "Trusted incumbent predictions required but not found in RUN_ARTIFACTS"
    )

train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float64)
y_valid = np.asarray(valid.y, dtype=np.int8)

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if len(inc_valid) != len(y_valid):
    raise RuntimeError(
        "incumbent_valid_scores.npy has an incompatible row count"
    )

inc_valid_z = normalized(inc_valid)
base_metrics = evaluate(valid.user_id, y_valid, inc_valid_z)

valid_bank, valid_coverage = build_residual_bank(
    train, y_train, valid
)

best_scores = inc_valid_z.copy()
best_metrics = base_metrics
best_config = None
best_beta = None
best_weight = 0.0

candidate_summary = {
    "trusted_incumbent": round(float(base_metrics["primary"]), 6)
}

for config in CONFIGS:
    config_name = "+".join(config)
    config_best = -np.inf

    for beta in BETAS:
        residual = combine_residuals(valid_bank, config, beta)

        for weight in BLEND_WEIGHTS:
            if weight == 0.0:
                scores = inc_valid_z
            else:
                scores = inc_valid_z + float(weight) * residual

            metrics = evaluate(valid.user_id, y_valid, scores)
            primary = float(metrics["primary"])

            if primary > config_best:
                config_best = primary

            if primary > float(best_metrics["primary"]):
                best_scores = np.asarray(scores, dtype=np.float64).copy()
                best_metrics = metrics
                best_config = tuple(config)
                best_beta = float(beta)
                best_weight = float(weight)

    candidate_summary[
        "personalized_" + config_name
    ] = round(config_best, 6)

print(
    "FINDINGS "
    + json.dumps(
        {
            "pair_coverage": {
                k: round(v, 4) for k, v in valid_coverage.items()
            },
            "selected_fields": (
                list(best_config) if best_config is not None else []
            ),
            "selected_beta": best_beta,
            "selected_residual_weight": best_weight,
            "incumbent_primary": round(
                float(base_metrics["primary"]), 6
            ),
            "selected_primary": round(
                float(best_metrics["primary"]), 6
            ),
        },
        sort_keys=True,
    )
)
print("CANDIDATES " + json.dumps(candidate_summary, sort_keys=True))

if OUT_DIR:
    np.save(
        os.path.join(OUT_DIR, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )

# Apply the validation-selected recipe to test. The personalized statistics
# are refit on train + validation, while no test labels are loaded or used.
test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test) != len(test.user_id):
    raise RuntimeError(
        "incumbent_test_scores.npy has an incompatible row count"
    )

inc_test_z = normalized(inc_test)

if best_config is None or best_weight == 0.0:
    test_scores = inc_test_z
else:
    combined = make_combined_split(train, valid)
    combined_labels = np.concatenate([
        y_train,
        y_valid.astype(np.float64),
    ])

    # Build only the fields selected on validation.
    test_parts = []
    for field in best_config:
        stats = PairStatistics(
            fit_users=combined.user_id,
            fit_entities=combined.X[field],
            fit_labels=combined_labels,
            entity_cardinality=int(FEATURE_CARDINALITIES[field]),
            entity_alpha=30.0,
        )
        residual, _ = stats.query(
            np.asarray(test.user_id, dtype=np.int64),
            np.asarray(test.X[field], dtype=np.int64),
            best_beta,
        )
        test_parts.append(normalized(residual))
        del stats, residual
        gc.collect()

    if len(test_parts) == 1:
        test_residual = test_parts[0]
    else:
        test_residual = np.mean(
            np.column_stack(test_parts), axis=1
        )

    test_scores = (
        inc_test_z + best_weight * test_residual
    )

if OUT_DIR:
    np.save(
        os.path.join(OUT_DIR, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START_TIME)
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": elapsed,
        }
    )
)