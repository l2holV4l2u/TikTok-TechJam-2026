import os
import time
import json
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()

PAIR_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "onehot_feat3",
    "upload_type",
    "tab",
]
ENTITY_FIELDS = ["video_id", "author_id"]
CONTEXT_FIELDS = [
    "tag",
    "duration_bucket",
    "onehot_feat3",
    "upload_type",
    "tab",
]
SMOOTHING = {
    "video_id": 3.0,
    "author_id": 4.0,
    "tag": 8.0,
    "duration_bucket": 10.0,
    "onehot_feat3": 6.0,
    "upload_type": 10.0,
    "tab": 12.0,
}


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.015, 0.985)
    return np.log(p) - np.log1p(-p)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]

    # Row position is a deterministic label-independent tie breaker.
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    sizes = ends - starts

    repeated_starts = np.repeat(starts, sizes)
    denominators = np.repeat(np.maximum(sizes - 1, 1), sizes)
    sorted_ranks = (
        np.arange(n, dtype=np.float64) - repeated_starts
    ) / denominators

    result = np.empty(n, dtype=np.float64)
    result[order] = sorted_ranks
    return result


class PairTable:
    def __init__(
        self,
        user_ids,
        feature_ids,
        labels,
        cardinality,
        row_weights,
        alpha,
    ):
        self.cardinality = int(cardinality)
        self.alpha = float(alpha)

        users = np.asarray(user_ids, dtype=np.int64)
        features = np.asarray(feature_ids, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.float64)
        weights = np.asarray(row_weights, dtype=np.float64)

        keys = users * self.cardinality + features
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        self.keys = unique_keys

        self.count = np.bincount(
            inverse, weights=weights, minlength=len(unique_keys)
        ).astype(np.float64)
        self.positive = np.bincount(
            inverse,
            weights=weights * labels,
            minlength=len(unique_keys),
        ).astype(np.float64)

        self.feature_count = np.bincount(
            features,
            weights=weights,
            minlength=self.cardinality,
        ).astype(np.float64)
        self.feature_positive = np.bincount(
            features,
            weights=weights * labels,
            minlength=self.cardinality,
        ).astype(np.float64)

        total_weight = float(weights.sum())
        global_rate = float(np.dot(weights, labels) / max(total_weight, 1.0))
        self.global_rate = global_rate

        self.feature_rate = (
            self.feature_positive + 20.0 * global_rate
        ) / (self.feature_count + 20.0)

    def predict(self, user_ids, feature_ids):
        users = np.asarray(user_ids, dtype=np.int64)
        features = np.asarray(feature_ids, dtype=np.int64)
        keys = users * self.cardinality + features

        positions = np.searchsorted(self.keys, keys)
        found = positions < len(self.keys)
        safe_positions = np.minimum(positions, len(self.keys) - 1)
        found &= self.keys[safe_positions] == keys

        base = self.feature_rate[
            np.minimum(features, self.cardinality - 1)
        ].astype(np.float64, copy=True)

        count = np.zeros(len(keys), dtype=np.float64)
        positive = np.zeros(len(keys), dtype=np.float64)
        count[found] = self.count[safe_positions[found]]
        positive[found] = self.positive[safe_positions[found]]

        posterior = (
            positive + self.alpha * base
        ) / (count + self.alpha)

        reliability = count / (count + self.alpha)
        return base, posterior, reliability


class HierarchicalMemory:
    def __init__(self, train):
        labels = np.asarray(train.y, dtype=np.float64)
        max_date = int(np.max(train.date))
        age_days = max_date - np.asarray(train.date, dtype=np.int64)

        uniform_weights = np.ones(len(labels), dtype=np.float64)
        recent_weights = np.power(
            0.5, age_days.astype(np.float64) / 4.0
        )

        self.tables = {"uniform": {}, "recent": {}}
        for mode, weights in (
            ("uniform", uniform_weights),
            ("recent", recent_weights),
        ):
            for field in PAIR_FIELDS:
                self.tables[mode][field] = PairTable(
                    train.user_id,
                    train.X[field],
                    labels,
                    FEATURE_CARDINALITIES[field],
                    weights,
                    SMOOTHING[field],
                )

    def predict_mode(self, split, mode):
        global_logits = {}
        posterior_logits = {}
        reliabilities = {}

        for field in PAIR_FIELDS:
            base, posterior, reliability = self.tables[mode][field].predict(
                split.user_id, split.X[field]
            )
            global_logits[field] = safe_logit(base)
            posterior_logits[field] = safe_logit(posterior)
            reliabilities[field] = reliability

        item_prior = (
            0.48 * global_logits["video_id"]
            + 0.32 * global_logits["author_id"]
            + 0.20 * global_logits["tag"]
        )

        entity_memory = (
            0.58 * posterior_logits["video_id"]
            + 0.42 * posterior_logits["author_id"]
        )

        context_memory = np.mean(
            np.column_stack(
                [posterior_logits[f] for f in CONTEXT_FIELDS]
            ),
            axis=1,
        )

        # Only personalized evidence contributes to the residual. This avoids
        # counting a category prior repeatedly when a pair was never observed.
        lifts = []
        lift_weights = []
        for field in PAIR_FIELDS:
            lifts.append(
                posterior_logits[field] - global_logits[field]
            )
            lift_weights.append(reliabilities[field])

        lifts = np.column_stack(lifts)
        lift_weights = np.column_stack(lift_weights)
        preference_residual = (
            np.sum(lifts * lift_weights, axis=1)
            / np.maximum(np.sum(lift_weights, axis=1), 1.0)
        )

        hybrid_memory = (
            item_prior
            + 0.60 * preference_residual
            + 0.15 * (
                context_memory
                - np.mean(
                    np.column_stack(
                        [global_logits[f] for f in CONTEXT_FIELDS]
                    ),
                    axis=1,
                )
            )
        )

        return {
            f"{mode}_item_prior": item_prior,
            f"{mode}_entity_memory": entity_memory,
            f"{mode}_context_memory": context_memory,
            f"{mode}_hybrid_memory": hybrid_memory,
        }

    def predict(self, split):
        result = {}
        result.update(self.predict_mode(split, "uniform"))
        result.update(self.predict_mode(split, "recent"))

        # A temporal consensus reduces variance when the four-day half-life
        # overreacts for an otherwise stationary entity.
        result["temporal_hybrid_consensus"] = (
            0.45 * result["uniform_hybrid_memory"]
            + 0.55 * result["recent_hybrid_memory"]
        )
        return result


train = load("train")
valid = load("valid")

memory = HierarchicalMemory(train)
valid_raw = memory.predict(valid)

# Rank aggregation is deliberately performed inside each logged slate because
# only within-user order matters and the component score scales are unrelated.
valid_ranks = {
    name: within_user_rank(valid.user_id, score)
    for name, score in valid_raw.items()
}

valid_ranks["memory_borda_consensus"] = np.mean(
    np.column_stack(
        [
            valid_ranks["uniform_item_prior"],
            valid_ranks["recent_item_prior"],
            valid_ranks["uniform_hybrid_memory"],
            valid_ranks["recent_hybrid_memory"],
            valid_ranks["uniform_entity_memory"],
        ]
    ),
    axis=1,
)

shared = os.environ.get("SHARED_ARTIFACTS", "")
incumbent_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
incumbent_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)
if not os.path.exists(incumbent_valid_path):
    raise FileNotFoundError("Trusted incumbent validation scores are missing")

incumbent_valid = np.asarray(
    np.load(incumbent_valid_path), dtype=np.float64
)
if len(incumbent_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation score length mismatch")

incumbent_valid_rank = within_user_rank(
    valid.user_id, incumbent_valid
)

candidate_scores = {}
candidate_metrics = {}
candidate_recipes = {}
candidate_raw = {}

for family, own_rank in valid_ranks.items():
    standalone_name = family + "_standalone"
    standalone_metric = evaluate(
        valid.user_id, valid.y, own_rank
    )
    candidate_scores[standalone_name] = own_rank
    candidate_metrics[standalone_name] = float(
        standalone_metric["primary"]
    )
    candidate_recipes[standalone_name] = (
        "standalone",
        family,
        1.0,
    )
    candidate_raw[standalone_name] = own_rank

    for own_weight in (0.10, 0.20, 0.30, 0.40, 0.50):
        name = f"{family}_borda_w{own_weight:.2f}"
        blended = (
            own_weight * own_rank
            + (1.0 - own_weight) * incumbent_valid_rank
        )
        metric = evaluate(valid.user_id, valid.y, blended)
        candidate_scores[name] = blended
        candidate_metrics[name] = float(metric["primary"])
        candidate_recipes[name] = (
            "borda_blend",
            family,
            own_weight,
        )
        candidate_raw[name] = own_rank

winner = max(candidate_metrics, key=candidate_metrics.get)
valid_scores = candidate_scores[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print(
    "FINDINGS "
    + json.dumps(
        {
            "best_family": winner,
            "best_standalone": max(
                (
                    (k, v)
                    for k, v in candidate_metrics.items()
                    if k.endswith("_standalone")
                ),
                key=lambda x: x[1],
            )[0],
            "incumbent_primary_check": float(
                evaluate(
                    valid.user_id, valid.y, incumbent_valid
                )["primary"]
            ),
        },
        separators=(",", ":"),
    )
)
print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_metrics.items()},
        sort_keys=True,
        separators=(",", ":"),
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    if candidate_recipes[winner][0] != "standalone":
        np.save(
            os.path.join(out_dir, "scores_valid_raw.npy"),
            np.asarray(candidate_raw[winner], dtype=np.float64),
        )

test = load("test")
test_raw = memory.predict(test)
test_ranks = {
    name: within_user_rank(test.user_id, score)
    for name, score in test_raw.items()
}
test_ranks["memory_borda_consensus"] = np.mean(
    np.column_stack(
        [
            test_ranks["uniform_item_prior"],
            test_ranks["recent_item_prior"],
            test_ranks["uniform_hybrid_memory"],
            test_ranks["recent_hybrid_memory"],
            test_ranks["uniform_entity_memory"],
        ]
    ),
    axis=1,
)

recipe_type, family, own_weight = candidate_recipes[winner]
own_test = test_ranks[family]

if recipe_type == "standalone":
    test_scores = own_test
else:
    if not os.path.exists(incumbent_test_path):
        raise FileNotFoundError("Trusted incumbent test scores are missing")
    incumbent_test = np.asarray(
        np.load(incumbent_test_path), dtype=np.float64
    )
    if len(incumbent_test) != len(test.user_id):
        raise ValueError("Incumbent test score length mismatch")
    incumbent_test_rank = within_user_rank(
        test.user_id, incumbent_test
    )
    test_scores = (
        own_weight * own_test
        + (1.0 - own_weight) * incumbent_test_rank
    )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
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