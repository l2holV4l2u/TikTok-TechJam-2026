import os
import time
import json
import gc
import numpy as np

_start_time = time.time()

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    positions = np.arange(n, dtype=np.float64) - np.repeat(starts, lengths)
    denominators = np.maximum(np.repeat(lengths, lengths) - 1.0, 1.0)

    ranked_sorted = positions / denominators
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def category_codes(split, specification, duration_edges):
    duration_bucket = np.asarray(
        split.X["duration_bucket"], dtype=np.int64
    )
    tab = np.asarray(split.X["tab"], dtype=np.int64)
    tag = np.asarray(split.X["tag"], dtype=np.int64)
    hour = np.asarray(split.X["hour"], dtype=np.int64)

    if specification == "duration_bucket":
        return duration_bucket, int(FEATURE_CARDINALITIES["duration_bucket"])

    if specification == "duration_quantile":
        raw = np.asarray(split.num["duration_ms"], dtype=np.float64)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        codes = np.searchsorted(duration_edges, raw, side="right")
        return codes.astype(np.int64), len(duration_edges) + 1

    if specification == "duration_tab":
        tab_card = int(FEATURE_CARDINALITIES["tab"])
        codes = duration_bucket * tab_card + tab
        card = int(FEATURE_CARDINALITIES["duration_bucket"]) * tab_card
        return codes.astype(np.int64), card

    if specification == "duration_tag":
        tag_card = int(FEATURE_CARDINALITIES["tag"])
        codes = duration_bucket * tag_card + tag
        card = int(FEATURE_CARDINALITIES["duration_bucket"]) * tag_card
        return codes.astype(np.int64), card

    if specification == "duration_hour":
        hour_card = int(FEATURE_CARDINALITIES["hour"])
        codes = duration_bucket * hour_card + hour
        card = int(FEATURE_CARDINALITIES["duration_bucket"]) * hour_card
        return codes.astype(np.int64), card

    if specification == "tab_tag":
        tag_card = int(FEATURE_CARDINALITIES["tag"])
        codes = tab * tag_card + tag
        card = int(FEATURE_CARDINALITIES["tab"]) * tag_card
        return codes.astype(np.int64), card

    raise ValueError("Unknown specification: %s" % specification)


class HierarchicalPairTable:
    """
    Hierarchical empirical-Bayes estimate for P(y=1 | user, category).

    The prior for a pair combines the user's overall propensity and the
    category's overall propensity additively on the log-odds scale. Pair
    observations then update that prior. Sparse sorted pair keys avoid large
    dense user-by-category matrices.
    """

    def __init__(self, user_ids, category, category_cardinality, labels):
        users = np.asarray(user_ids, dtype=np.int64)
        cats = np.asarray(category, dtype=np.int64)
        y = np.asarray(labels, dtype=np.float64)

        self.category_cardinality = int(category_cardinality)
        self.user_cardinality = max(
            int(FEATURE_CARDINALITIES["user_id"]),
            int(users.max()) + 1,
        )
        self.global_rate = float(np.mean(y))

        user_count = np.bincount(
            users, minlength=self.user_cardinality
        ).astype(np.float64)
        user_sum = np.bincount(
            users, weights=y, minlength=self.user_cardinality
        ).astype(np.float64)

        category_count = np.bincount(
            cats, minlength=self.category_cardinality
        ).astype(np.float64)
        category_sum = np.bincount(
            cats, weights=y, minlength=self.category_cardinality
        ).astype(np.float64)

        # Stable marginal estimates. These are only priors for the sparse
        # pair estimate, not the final candidate score.
        user_strength = 35.0
        category_strength = 80.0

        self.user_rate = (
            user_sum + user_strength * self.global_rate
        ) / (user_count + user_strength)
        self.category_rate = (
            category_sum + category_strength * self.global_rate
        ) / (category_count + category_strength)

        pair_keys = users * self.category_cardinality + cats
        unique_keys, inverse = np.unique(pair_keys, return_inverse=True)
        self.pair_keys = unique_keys.astype(np.int64, copy=False)
        self.pair_count = np.bincount(inverse).astype(np.float64)
        self.pair_sum = np.bincount(
            inverse, weights=y
        ).astype(np.float64)

        del inverse, pair_keys
        gc.collect()

    def score(self, user_ids, category, strength):
        users = np.asarray(user_ids, dtype=np.int64)
        cats = np.asarray(category, dtype=np.int64)

        safe_users = np.clip(users, 0, self.user_cardinality - 1)
        safe_cats = np.clip(cats, 0, self.category_cardinality - 1)

        known_user = (
            (users >= 0) & (users < self.user_cardinality)
        )
        known_cat = (
            (cats >= 0) & (cats < self.category_cardinality)
        )

        user_rate = np.where(
            known_user,
            self.user_rate[safe_users],
            self.global_rate,
        )
        category_rate = np.where(
            known_cat,
            self.category_rate[safe_cats],
            self.global_rate,
        )

        prior_logit = (
            logit(user_rate)
            + logit(category_rate)
            - logit(self.global_rate)
        )
        prior_rate = sigmoid(prior_logit)

        query_keys = users * self.category_cardinality + cats
        positions = np.searchsorted(self.pair_keys, query_keys)
        clipped = np.minimum(positions, len(self.pair_keys) - 1)
        matched = (
            (positions < len(self.pair_keys))
            & (self.pair_keys[clipped] == query_keys)
            & known_user
            & known_cat
        )

        counts = np.zeros(len(users), dtype=np.float64)
        sums = np.zeros(len(users), dtype=np.float64)
        counts[matched] = self.pair_count[clipped[matched]]
        sums[matched] = self.pair_sum[clipped[matched]]

        posterior = (
            sums + float(strength) * prior_rate
        ) / (counts + float(strength))

        # Return log-odds so evidence of different confidence combines more
        # naturally with the incumbent than raw probabilities.
        return logit(posterior)


artifacts = os.environ.get("RUN_ARTIFACTS", "")
inc_valid_path = os.path.join(
    artifacts, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    artifacts, "incumbent_test_scores.npy"
)

if not (
    os.path.isfile(inc_valid_path)
    and os.path.isfile(inc_test_path)
):
    raise FileNotFoundError(
        "Trusted incumbent validation and test scores are required"
    )

train = load("train")
valid = load("valid")

incumbent_valid = np.asarray(
    np.load(inc_valid_path), dtype=np.float64
)
if len(incumbent_valid) != len(valid.y):
    raise ValueError("Incumbent validation length mismatch")

incumbent_metrics = evaluate(
    valid.user_id, valid.y, incumbent_valid
)
incumbent_valid_rank = within_user_rank(
    valid.user_id, incumbent_valid
)

raw_train_duration = np.asarray(
    train.num["duration_ms"], dtype=np.float64
)
raw_train_duration = np.nan_to_num(
    raw_train_duration, nan=0.0, posinf=0.0, neginf=0.0
)

# Repeated quantile edges are removed so unusually discrete duration values
# do not create empty or inconsistent bins.
duration_edges = np.unique(
    np.quantile(raw_train_duration, np.linspace(0.04, 0.96, 23))
).astype(np.float64)

specifications = [
    "duration_bucket",
    "duration_quantile",
    "duration_tab",
    "duration_tag",
    "duration_hour",
    "tab_tag",
]
strengths = [4.0, 12.0, 35.0]

tables = {}
valid_categories = {}
valid_sources = {}

train_users = np.asarray(train.user_id, dtype=np.int64)
train_labels = np.asarray(train.y, dtype=np.float64)

for specification in specifications:
    train_cat, cardinality = category_codes(
        train, specification, duration_edges
    )
    valid_cat, valid_cardinality = category_codes(
        valid, specification, duration_edges
    )
    if cardinality != valid_cardinality:
        raise ValueError("Category cardinality mismatch")

    table = HierarchicalPairTable(
        train_users, train_cat, cardinality, train_labels
    )
    tables[specification] = table
    valid_categories[specification] = valid_cat

    for strength in strengths:
        name = "%s_s%d" % (specification, int(strength))
        valid_sources[name] = table.score(
            valid.user_id, valid_cat, strength
        )

    del train_cat
    gc.collect()

# Aggregate related curves to reduce variance while retaining their distinct
# notions of duration-conditioned preference.
for strength in strengths:
    suffix = "_s%d" % int(strength)

    duration_names = [
        "duration_bucket" + suffix,
        "duration_quantile" + suffix,
        "duration_tab" + suffix,
        "duration_tag" + suffix,
        "duration_hour" + suffix,
    ]
    valid_sources["duration_ensemble" + suffix] = np.mean(
        np.column_stack(
            [
                within_user_rank(valid.user_id, valid_sources[name])
                for name in duration_names
            ]
        ),
        axis=1,
    )

    all_names = [
        specification + suffix
        for specification in specifications
    ]
    valid_sources["all_context_ensemble" + suffix] = np.mean(
        np.column_stack(
            [
                within_user_rank(valid.user_id, valid_sources[name])
                for name in all_names
            ]
        ),
        axis=1,
    )

candidate_scores = {
    "incumbent": float(incumbent_metrics["primary"])
}

best = {
    "name": "incumbent",
    "source_name": None,
    "alpha": 0.0,
    "scores": incumbent_valid.copy(),
    "metrics": incumbent_metrics,
}

alphas = [
    0.03, 0.05, 0.075, 0.10, 0.15, 0.20,
    0.25, 0.30, 0.40, 0.50, 0.65, 0.80,
]

source_standalone = {}

for source_name, source_values in valid_sources.items():
    source_rank = within_user_rank(valid.user_id, source_values)
    standalone_metrics = evaluate(
        valid.user_id, valid.y, source_rank
    )
    standalone_name = source_name + "_standalone"
    candidate_scores[standalone_name] = float(
        standalone_metrics["primary"]
    )
    source_standalone[source_name] = float(
        standalone_metrics["primary"]
    )

    for alpha in alphas:
        scores = (
            (1.0 - alpha) * incumbent_valid_rank
            + alpha * source_rank
        )
        metrics = evaluate(valid.user_id, valid.y, scores)
        name = "%s_blend_%.3f" % (source_name, alpha)
        primary = float(metrics["primary"])
        candidate_scores[name] = primary

        if primary > float(best["metrics"]["primary"]):
            best = {
                "name": name,
                "source_name": source_name,
                "alpha": float(alpha),
                "scores": scores.copy(),
                "metrics": metrics,
            }

top_candidates = dict(
    sorted(
        candidate_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:15]
)

best_standalone_name = max(
    source_standalone, key=source_standalone.get
)
print(
    "FINDINGS "
    + json.dumps(
        {
            "incumbent_primary": float(
                incumbent_metrics["primary"]
            ),
            "selected": best["name"],
            "selected_primary": float(
                best["metrics"]["primary"]
            ),
            "best_pair_curve": best_standalone_name,
            "best_pair_curve_primary": float(
                source_standalone[best_standalone_name]
            ),
            "duration_quantile_bins": int(
                len(duration_edges) + 1
            ),
        },
        sort_keys=True,
    )
)
print("CANDIDATES " + json.dumps(top_candidates, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best["scores"], dtype=np.float64),
    )

# Test is scored using exactly the validation-selected source and blend
# coefficient. No test labels are loaded or evaluated.
test = load("test")
incumbent_test = np.asarray(
    np.load(inc_test_path), dtype=np.float64
)
if len(incumbent_test) != len(test.user_id):
    raise ValueError("Incumbent test length mismatch")

if best["source_name"] is None:
    test_scores = incumbent_test.copy()
else:
    source_name = best["source_name"]

    if source_name.startswith("duration_ensemble"):
        strength = float(source_name.rsplit("_s", 1)[1])
        component_specs = [
            "duration_bucket",
            "duration_quantile",
            "duration_tab",
            "duration_tag",
            "duration_hour",
        ]
        component_ranks = []
        for specification in component_specs:
            test_cat, _ = category_codes(
                test, specification, duration_edges
            )
            values = tables[specification].score(
                test.user_id, test_cat, strength
            )
            component_ranks.append(
                within_user_rank(test.user_id, values)
            )
        test_source = np.mean(
            np.column_stack(component_ranks), axis=1
        )

    elif source_name.startswith("all_context_ensemble"):
        strength = float(source_name.rsplit("_s", 1)[1])
        component_ranks = []
        for specification in specifications:
            test_cat, _ = category_codes(
                test, specification, duration_edges
            )
            values = tables[specification].score(
                test.user_id, test_cat, strength
            )
            component_ranks.append(
                within_user_rank(test.user_id, values)
            )
        test_source = np.mean(
            np.column_stack(component_ranks), axis=1
        )

    else:
        specification, strength_text = source_name.rsplit("_s", 1)
        strength = float(strength_text)
        test_cat, _ = category_codes(
            test, specification, duration_edges
        )
        test_source = tables[specification].score(
            test.user_id, test_cat, strength
        )

    incumbent_test_rank = within_user_rank(
        test.user_id, incumbent_test
    )
    test_source_rank = within_user_rank(
        test.user_id, test_source
    )
    alpha = float(best["alpha"])
    test_scores = (
        (1.0 - alpha) * incumbent_test_rank
        + alpha * test_source_rank
    )

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - _start_time
metrics = best["metrics"]
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(metrics["primary"]),
            "gauc": float(metrics["gauc"]),
            "ndcg@5": float(metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)