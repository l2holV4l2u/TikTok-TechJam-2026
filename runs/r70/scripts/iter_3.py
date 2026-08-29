import os
import time
import json
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START_TIME = time.time()

PAIR_SPECS = {
    "duration": ("duration_bucket", 30.0, 80.0),
    "tag": ("tag", 25.0, 60.0),
    "author": ("author_id", 12.0, 30.0),
    "video": ("video_id", 8.0, 20.0),
    "hour": ("hour", 35.0, 80.0),
    "upload": ("upload_type", 30.0, 70.0),
}

ALPHAS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
MAX_GREEDY_STAGES = 4
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


class HierarchicalPairStatistic:
    def __init__(
        self,
        user_ids,
        context_ids,
        labels,
        context_cardinality,
        pair_strength,
        context_strength,
    ):
        users = np.asarray(user_ids, dtype=np.int64)
        contexts = np.asarray(context_ids, dtype=np.int64)
        y = np.asarray(labels, dtype=np.float64)
        self.cardinality = int(context_cardinality)
        self.pair_strength = float(pair_strength)

        global_rate = float(np.mean(y))
        self.global_logit = float(logit(global_rate))

        context_count = np.bincount(
            contexts, minlength=self.cardinality
        ).astype(np.float64)
        context_sum = np.bincount(
            contexts, weights=y, minlength=self.cardinality
        ).astype(np.float64)

        self.context_rate = (
            context_sum + float(context_strength) * global_rate
        ) / (context_count + float(context_strength))
        self.context_logit = logit(self.context_rate)

        keys = users * np.int64(self.cardinality) + contexts
        unique_keys, inverse = np.unique(keys, return_inverse=True)

        pair_count = np.bincount(inverse).astype(np.float64)
        pair_sum = np.bincount(inverse, weights=y).astype(np.float64)

        key_context = np.remainder(
            unique_keys, np.int64(self.cardinality)
        ).astype(np.int64)
        prior_rate = self.context_rate[key_context]
        posterior_rate = (
            pair_sum + self.pair_strength * prior_rate
        ) / (pair_count + self.pair_strength)

        self.keys = unique_keys
        self.pair_residual = np.clip(
            logit(posterior_rate) - logit(prior_rate),
            -2.5,
            2.5,
        ).astype(np.float32)

    def transform(self, user_ids, context_ids):
        users = np.asarray(user_ids, dtype=np.int64)
        contexts = np.asarray(context_ids, dtype=np.int64)
        query_keys = users * np.int64(self.cardinality) + contexts

        positions = np.searchsorted(self.keys, query_keys)
        matched = positions < len(self.keys)
        safe_positions = np.minimum(positions, len(self.keys) - 1)
        matched &= self.keys[safe_positions] == query_keys

        pair = np.zeros(len(users), dtype=np.float64)
        pair[matched] = self.pair_residual[safe_positions[matched]]

        valid_context = (
            (contexts >= 0) & (contexts < len(self.context_logit))
        )
        entity = np.zeros(len(users), dtype=np.float64)
        entity[valid_context] = (
            self.context_logit[contexts[valid_context]]
            - self.global_logit
        )
        entity = np.clip(entity, -3.0, 3.0)

        return pair, entity


def build_components(reference, target):
    labels = np.asarray(reference.y, dtype=np.float64)
    components = {}
    models = {}

    for short_name, (
        field_name,
        pair_strength,
        context_strength,
    ) in PAIR_SPECS.items():
        model = HierarchicalPairStatistic(
            user_ids=reference.user_id,
            context_ids=reference.X[field_name],
            labels=labels,
            context_cardinality=int(FEATURE_CARDINALITIES[field_name]),
            pair_strength=pair_strength,
            context_strength=context_strength,
        )
        pair, entity = model.transform(
            target.user_id, target.X[field_name]
        )
        components["pair_" + short_name] = pair
        models[short_name] = model

        if short_name in ("video", "author", "tag", "duration"):
            components["entity_" + short_name] = entity

    return components, models


def transform_components(models, target):
    components = {}
    for short_name, model in models.items():
        field_name = PAIR_SPECS[short_name][0]
        pair, entity = model.transform(
            target.user_id, target.X[field_name]
        )
        components["pair_" + short_name] = pair
        if short_name in ("video", "author", "tag", "duration"):
            components["entity_" + short_name] = entity
    return components


def metric_primary(valid, scores):
    return float(
        evaluate(valid.user_id, valid.y, scores)["primary"]
    )


def select_corrections(valid, base_scores, components):
    candidate_log = {}
    base_metrics = evaluate(valid.user_id, valid.y, base_scores)
    candidate_log["incumbent"] = float(base_metrics["primary"])

    current_scores = np.asarray(base_scores, dtype=np.float64).copy()
    current_primary = float(base_metrics["primary"])
    selected = []
    remaining = set(components.keys())

    # Record the best individually useful correction. This also makes the
    # experiment diagnostic if combinations do not help.
    best_single = None
    best_single_primary = -np.inf
    for name in sorted(remaining):
        for alpha in ALPHAS:
            p = metric_primary(
                valid, base_scores + alpha * components[name]
            )
            if p > best_single_primary:
                best_single_primary = p
                best_single = (name, float(alpha))
    candidate_log["best_single"] = float(best_single_primary)

    for stage in range(1, MAX_GREEDY_STAGES + 1):
        best_choice = None
        best_scores = None
        best_primary = current_primary

        for name in sorted(remaining):
            correction = components[name]
            for alpha in ALPHAS:
                trial_scores = current_scores + alpha * correction
                p = metric_primary(valid, trial_scores)
                if p > best_primary:
                    best_primary = p
                    best_choice = (name, float(alpha))
                    best_scores = trial_scores

        if best_choice is None:
            break

        name, alpha = best_choice
        selected.append((name, alpha))
        remaining.remove(name)
        current_scores = best_scores
        current_primary = best_primary
        candidate_log["greedy_stage_%d" % stage] = current_primary

    final_metrics = evaluate(
        valid.user_id, valid.y, current_scores
    )
    return selected, current_scores, final_metrics, candidate_log


artifacts_dir = os.environ.get("RUN_ARTIFACTS", "")
incumbent_valid_path = os.path.join(
    artifacts_dir, "incumbent_valid_scores.npy"
)
incumbent_test_path = os.path.join(
    artifacts_dir, "incumbent_test_scores.npy"
)

if not os.path.exists(incumbent_valid_path):
    raise FileNotFoundError(
        "Trusted incumbent validation predictions are unavailable"
    )
if not os.path.exists(incumbent_test_path):
    raise FileNotFoundError(
        "Trusted incumbent test predictions are unavailable"
    )

train = load("train")
valid = load("valid")

incumbent_valid = np.asarray(
    np.load(incumbent_valid_path), dtype=np.float64
)
if len(incumbent_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation prediction length mismatch")

valid_components, _ = build_components(train, valid)

selected, valid_scores, metrics, candidate_log = select_corrections(
    valid,
    incumbent_valid,
    valid_components,
)

print(
    "CANDIDATES "
    + json.dumps(candidate_log, sort_keys=True, separators=(",", ":")),
    flush=True,
)
print(
    "FINDINGS selected_corrections="
    + json.dumps(selected, separators=(",", ":")),
    flush=True,
)

component_stds = {
    name: float(np.std(values))
    for name, values in valid_components.items()
}
print(
    "FINDINGS component_std="
    + json.dumps(component_stds, sort_keys=True, separators=(",", ":")),
    flush=True,
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

# Refit the identical empirical-Bayes recipe on train + validation, then
# generate test corrections. No test labels are accessed.
test = load("test")
incumbent_test = np.asarray(
    np.load(incumbent_test_path), dtype=np.float64
)
if len(incumbent_test) != len(test.user_id):
    raise ValueError("Incumbent test prediction length mismatch")

combined_user = np.concatenate(
    [
        np.asarray(train.user_id, dtype=np.int64),
        np.asarray(valid.user_id, dtype=np.int64),
    ]
)
combined_y = np.concatenate(
    [
        np.asarray(train.y, dtype=np.float64),
        np.asarray(valid.y, dtype=np.float64),
    ]
)

test_scores = incumbent_test.copy()

# Only rebuild statistics selected on validation, reducing final-fit work.
for component_name, alpha in selected:
    kind, short_name = component_name.split("_", 1)
    field_name, pair_strength, context_strength = PAIR_SPECS[short_name]

    combined_context = np.concatenate(
        [
            np.asarray(train.X[field_name], dtype=np.int64),
            np.asarray(valid.X[field_name], dtype=np.int64),
        ]
    )

    fitted = HierarchicalPairStatistic(
        user_ids=combined_user,
        context_ids=combined_context,
        labels=combined_y,
        context_cardinality=int(FEATURE_CARDINALITIES[field_name]),
        pair_strength=pair_strength,
        context_strength=context_strength,
    )

    pair_values, entity_values = fitted.transform(
        test.user_id, test.X[field_name]
    )
    correction = (
        pair_values if kind == "pair" else entity_values
    )
    test_scores += float(alpha) * correction

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START_TIME
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        },
        separators=(",", ":"),
    )
)