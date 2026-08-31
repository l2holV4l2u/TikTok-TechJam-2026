import os
import time
import json

import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()

PROFILE_FIELDS = [
    "tag",
    "duration_bucket",
    "upload_type",
    "tab",
    "onehot_feat1",
    "onehot_feat3",
    "onehot_feat7",
    "onehot_feat8",
    "music_type",
    "video_type",
]

PROFILE_WEIGHTS = {
    "tag": 0.55,
    "duration_bucket": 0.40,
    "upload_type": 0.30,
    "tab": 0.28,
    "onehot_feat1": 0.22,
    "onehot_feat3": 0.42,
    "onehot_feat7": 0.25,
    "onehot_feat8": 0.28,
    "music_type": 0.18,
    "video_type": 0.08,
}

GLOBAL_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "tab",
]

GLOBAL_WEIGHTS = {
    "video_id": 0.50,
    "author_id": 0.43,
    "tag": 0.30,
    "duration_bucket": 0.18,
    "upload_type": 0.18,
    "onehot_feat3": 0.22,
    "tab": 0.14,
}

TRANSITION_FIELDS = [
    "tag",
    "duration_bucket",
    "tab",
    "upload_type",
    "onehot_feat3",
    "author_id",
]

TRANSITION_WEIGHTS = {
    "tag": 0.55,
    "duration_bucket": 0.40,
    "tab": 0.30,
    "upload_type": 0.28,
    "onehot_feat3": 0.32,
    "author_id": 0.35,
}


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.01, 0.99)
    return np.log(p) - np.log1p(-p)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    ordered_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_users[1:] != ordered_users[:-1]]
    )
    sizes = np.diff(np.r_[starts, n])

    rank = (
        np.arange(n, dtype=np.float64) - np.repeat(starts, sizes)
    ) / np.repeat(np.maximum(sizes - 1, 1), sizes)

    result = np.empty(n, dtype=np.float64)
    result[order] = rank
    return result


def ordered_previous_indices(split):
    n = len(split.user_id)
    rows = np.arange(n, dtype=np.int64)
    order = np.lexsort((rows, split.time_ms, split.user_id))

    prev = np.full(n, -1, dtype=np.int64)
    if n > 1:
        current = order[1:]
        previous = order[:-1]
        same_user = (
            np.asarray(split.user_id)[current]
            == np.asarray(split.user_id)[previous]
        )
        prev[current[same_user]] = previous[same_user]
    return prev


class SparseSmoothedTable:
    def __init__(self, keys, labels, prior, strength):
        keys = np.asarray(keys, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.float64)

        unique, inverse = np.unique(keys, return_inverse=True)
        count = np.bincount(inverse).astype(np.float64)
        positive = np.bincount(inverse, weights=labels).astype(np.float64)

        rate = (positive + strength * prior) / (count + strength)
        self.keys = unique
        self.values = safe_logit(rate).astype(np.float32)
        self.counts = count.astype(np.float32)
        self.default = float(safe_logit(prior))

    def lookup(self, query_keys):
        query_keys = np.asarray(query_keys, dtype=np.int64)
        pos = np.searchsorted(self.keys, query_keys)
        valid = pos < len(self.keys)
        safe_pos = np.minimum(pos, max(len(self.keys) - 1, 0))

        if len(self.keys) == 0:
            return (
                np.full(len(query_keys), self.default, dtype=np.float64),
                np.zeros(len(query_keys), dtype=np.float64),
            )

        valid &= self.keys[safe_pos] == query_keys
        values = np.full(len(query_keys), self.default, dtype=np.float64)
        counts = np.zeros(len(query_keys), dtype=np.float64)
        values[valid] = self.values[safe_pos[valid]]
        counts[valid] = self.counts[safe_pos[valid]]
        return values, counts


class ComplementaryPredictors:
    def __init__(self, train):
        self.global_rate = float(np.mean(train.y))
        self.global_logit = float(safe_logit(self.global_rate))
        y = np.asarray(train.y, dtype=np.float64)
        users = np.asarray(train.user_id, dtype=np.int64)
        self.max_user = int(max(
            FEATURE_CARDINALITIES["user_id"] - 1,
            users.max(initial=0),
        ))

        self.global_tables = {}
        self.category_logits = {}
        self.profile_tables = {}

        for field in sorted(set(GLOBAL_FIELDS + PROFILE_FIELDS)):
            ids = np.asarray(train.X[field], dtype=np.int64)
            card = int(FEATURE_CARDINALITIES[field])

            count = np.bincount(ids, minlength=card).astype(np.float64)
            positive = np.bincount(
                ids, weights=y, minlength=card
            ).astype(np.float64)

            strength = 35.0 if card > 100 else 80.0
            rate = (
                positive + strength * self.global_rate
            ) / (count + strength)
            logits = safe_logit(rate)
            self.category_logits[field] = logits.astype(np.float32)

            if field in GLOBAL_FIELDS:
                self.global_tables[field] = (
                    logits - self.global_logit
                ).astype(np.float32)

            if field in PROFILE_FIELDS:
                # Exact user-by-content-value observations form a personalized
                # likelihood model. The prior is supplied per query from the
                # content marginal, so retain sufficient statistics here.
                keys = users * np.int64(card) + ids
                unique, inverse = np.unique(keys, return_inverse=True)
                pair_count = np.bincount(inverse).astype(np.float64)
                pair_positive = np.bincount(
                    inverse, weights=y
                ).astype(np.float64)
                self.profile_tables[field] = {
                    "keys": unique,
                    "count": pair_count.astype(np.float32),
                    "positive": pair_positive.astype(np.float32),
                    "card": card,
                }

        # User-specific continuous-duration likelihood ratio. Positive and
        # negative duration distributions can differ even when the global
        # duration-label correlation is approximately zero.
        log_duration = np.log1p(
            np.maximum(
                np.nan_to_num(
                    np.asarray(
                        train.num["duration_ms"], dtype=np.float64
                    ),
                    nan=0.0,
                ),
                0.0,
            )
        )

        n_users = self.max_user + 1
        pos_count = np.bincount(
            users, weights=y, minlength=n_users
        ).astype(np.float64)
        neg_count = np.bincount(
            users, weights=1.0 - y, minlength=n_users
        ).astype(np.float64)
        pos_sum = np.bincount(
            users, weights=y * log_duration, minlength=n_users
        ).astype(np.float64)
        neg_sum = np.bincount(
            users, weights=(1.0 - y) * log_duration, minlength=n_users
        ).astype(np.float64)
        pos_sq = np.bincount(
            users, weights=y * log_duration * log_duration,
            minlength=n_users,
        ).astype(np.float64)
        neg_sq = np.bincount(
            users,
            weights=(1.0 - y) * log_duration * log_duration,
            minlength=n_users,
        ).astype(np.float64)

        global_pos = log_duration[y > 0.5]
        global_neg = log_duration[y <= 0.5]
        gp_mean = float(global_pos.mean())
        gn_mean = float(global_neg.mean())
        gp_var = float(global_pos.var() + 0.08)
        gn_var = float(global_neg.var() + 0.08)

        duration_prior = 8.0
        pos_mean = (
            pos_sum + duration_prior * gp_mean
        ) / (pos_count + duration_prior)
        neg_mean = (
            neg_sum + duration_prior * gn_mean
        ) / (neg_count + duration_prior)

        pos_second = (
            pos_sq + duration_prior * (gp_var + gp_mean * gp_mean)
        ) / (pos_count + duration_prior)
        neg_second = (
            neg_sq + duration_prior * (gn_var + gn_mean * gn_mean)
        ) / (neg_count + duration_prior)

        pos_var = np.maximum(pos_second - pos_mean * pos_mean, 0.08)
        neg_var = np.maximum(neg_second - neg_mean * neg_mean, 0.08)

        self.duration_pos_mean = pos_mean.astype(np.float32)
        self.duration_neg_mean = neg_mean.astype(np.float32)
        self.duration_pos_var = pos_var.astype(np.float32)
        self.duration_neg_var = neg_var.astype(np.float32)
        self.duration_reliability = (
            (pos_count + neg_count) / (pos_count + neg_count + 15.0)
        ).astype(np.float32)

        # First-order transition estimators use only the identity/content of
        # the preceding logged impression, never its outcome.
        previous = ordered_previous_indices(train)
        usable = previous >= 0
        current_rows = np.flatnonzero(usable)
        previous_rows = previous[usable]

        self.transition_tables = {}
        for field in TRANSITION_FIELDS:
            values = np.asarray(train.X[field], dtype=np.int64)
            card = int(FEATURE_CARDINALITIES[field])
            keys = (
                values[previous_rows] * np.int64(card)
                + values[current_rows]
            )
            strength = 45.0 if card > 100 else 75.0
            self.transition_tables[field] = (
                SparseSmoothedTable(
                    keys,
                    y[current_rows],
                    self.global_rate,
                    strength,
                ),
                card,
            )

    def _global_content(self, split):
        score = np.zeros(len(split.user_id), dtype=np.float64)
        for field in GLOBAL_FIELDS:
            ids = np.asarray(split.X[field], dtype=np.int64)
            table = self.global_tables[field]
            safe_ids = np.clip(ids, 0, len(table) - 1)
            score += GLOBAL_WEIGHTS[field] * table[safe_ids]
        return score

    def personalized_naive_bayes(self, split):
        users = np.asarray(split.user_id, dtype=np.int64)
        score = 0.35 * self._global_content(split)

        for field in PROFILE_FIELDS:
            ids = np.asarray(split.X[field], dtype=np.int64)
            info = self.profile_tables[field]
            card = info["card"]
            keys = users * np.int64(card) + ids

            pos = np.searchsorted(info["keys"], keys)
            valid = pos < len(info["keys"])
            safe_pos = np.minimum(
                pos, max(len(info["keys"]) - 1, 0)
            )
            valid &= info["keys"][safe_pos] == keys

            pair_count = np.zeros(len(users), dtype=np.float64)
            pair_positive = np.zeros(len(users), dtype=np.float64)
            pair_count[valid] = info["count"][safe_pos[valid]]
            pair_positive[valid] = info["positive"][safe_pos[valid]]

            marginal = self.category_logits[field][
                np.clip(ids, 0, len(self.category_logits[field]) - 1)
            ].astype(np.float64)
            marginal_rate = 1.0 / (1.0 + np.exp(-marginal))

            shrink = 5.0
            pair_rate = (
                pair_positive + shrink * marginal_rate
            ) / (pair_count + shrink)
            residual = safe_logit(pair_rate) - marginal

            reliability = pair_count / (pair_count + 3.0)
            score += (
                PROFILE_WEIGHTS[field]
                * reliability
                * np.clip(residual, -2.0, 2.0)
            )

        return score

    def duration_ideal_point(self, split):
        users = np.asarray(split.user_id, dtype=np.int64)
        known = (users >= 0) & (users < len(self.duration_pos_mean))
        safe_users = np.clip(
            users, 0, len(self.duration_pos_mean) - 1
        )

        x = np.log1p(
            np.maximum(
                np.nan_to_num(
                    np.asarray(
                        split.num["duration_ms"], dtype=np.float64
                    ),
                    nan=0.0,
                ),
                0.0,
            )
        )

        pm = self.duration_pos_mean[safe_users].astype(np.float64)
        nm = self.duration_neg_mean[safe_users].astype(np.float64)
        pv = self.duration_pos_var[safe_users].astype(np.float64)
        nv = self.duration_neg_var[safe_users].astype(np.float64)

        positive_log_density = (
            -0.5 * np.log(pv) - 0.5 * (x - pm) ** 2 / pv
        )
        negative_log_density = (
            -0.5 * np.log(nv) - 0.5 * (x - nm) ** 2 / nv
        )
        likelihood_ratio = positive_log_density - negative_log_density

        reliability = self.duration_reliability[
            safe_users
        ].astype(np.float64)
        likelihood_ratio *= reliability * known
        likelihood_ratio = np.clip(likelihood_ratio, -2.5, 2.5)

        # Marginal content evidence stabilizes users with short histories.
        return (
            0.55 * likelihood_ratio
            + 0.45 * self._global_content(split)
        )

    def transition_model(self, split):
        n = len(split.user_id)
        score = 0.55 * self._global_content(split)
        previous = ordered_previous_indices(split)
        usable = previous >= 0

        transition_score = np.zeros(n, dtype=np.float64)
        transition_weight = np.zeros(n, dtype=np.float64)

        current_rows = np.flatnonzero(usable)
        previous_rows = previous[usable]

        for field in TRANSITION_FIELDS:
            values = np.asarray(split.X[field], dtype=np.int64)
            table, card = self.transition_tables[field]
            keys = (
                values[previous_rows] * np.int64(card)
                + values[current_rows]
            )
            logits, counts = table.lookup(keys)
            residual = logits - self.global_logit
            reliability = counts / (counts + 20.0)

            contribution = (
                TRANSITION_WEIGHTS[field]
                * reliability
                * np.clip(residual, -1.8, 1.8)
            )
            transition_score[current_rows] += contribution
            transition_weight[current_rows] += (
                TRANSITION_WEIGHTS[field] * reliability
            )

        normalized = transition_score / np.maximum(
            transition_weight, 0.35
        )
        score += 0.75 * normalized
        return score

    def predict(self, split):
        nb = self.personalized_naive_bayes(split)
        duration = self.duration_ideal_point(split)
        transition = self.transition_model(split)

        # This fourth family is a mixture-of-evidence consensus rather than
        # another parameter setting: persistent preference governs ordinary
        # rows while transition evidence can alter rows inside sessions.
        consensus = (
            0.48 * within_user_rank(split.user_id, nb)
            + 0.20 * within_user_rank(split.user_id, duration)
            + 0.32 * within_user_rank(split.user_id, transition)
        )

        return {
            "personalized_naive_bayes": nb,
            "duration_ideal_point": duration,
            "feed_transition_markov": transition,
            "preference_session_consensus": consensus,
        }


train = load("train")
valid = load("valid")

model = ComplementaryPredictors(train)
valid_raw = model.predict(valid)
valid_rank = {
    name: within_user_rank(valid.user_id, score)
    for name, score in valid_raw.items()
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)

if not os.path.exists(inc_valid_path):
    raise FileNotFoundError(
        "Trusted incumbent validation scores are missing"
    )
if not os.path.exists(inc_test_path):
    raise FileNotFoundError(
        "Trusted incumbent test scores are missing"
    )

inc_valid = np.asarray(
    np.load(inc_valid_path), dtype=np.float64
)
if len(inc_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation length mismatch")
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

candidate_scores = {"incumbent": inc_valid_rank}
candidate_primary = {
    "incumbent": float(
        evaluate(
            valid.user_id, valid.y, inc_valid_rank
        )["primary"]
    )
}
recipes = {"incumbent": ("incumbent", "", 0.0)}

blend_weights = (
    0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50
)

for family, own_rank in valid_rank.items():
    standalone = family + "_standalone"
    candidate_scores[standalone] = own_rank
    candidate_primary[standalone] = float(
        evaluate(
            valid.user_id, valid.y, own_rank
        )["primary"]
    )
    recipes[standalone] = ("standalone", family, 1.0)

    for weight in blend_weights:
        name = f"{family}_blend_{weight:.2f}"
        blended = (
            (1.0 - weight) * inc_valid_rank
            + weight * own_rank
        )
        candidate_scores[name] = blended
        candidate_primary[name] = float(
            evaluate(
                valid.user_id, valid.y, blended
            )["primary"]
        )
        recipes[name] = ("blend", family, weight)

winner = max(candidate_primary, key=candidate_primary.get)
valid_scores = candidate_scores[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)
recipe_type, winner_family, winner_weight = recipes[winner]

best_own_family = max(
    valid_rank,
    key=lambda name: candidate_primary[
        name + "_standalone"
    ],
)
raw_for_audit = valid_rank[
    winner_family
    if winner_family in valid_rank
    else best_own_family
]

previous_valid = ordered_previous_indices(valid)
print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner,
            "best_own_family": best_own_family,
            "incumbent_primary": candidate_primary["incumbent"],
            "standalone_primary": {
                name: candidate_primary[name + "_standalone"]
                for name in valid_rank
            },
            "valid_rows_with_predecessor": float(
                np.mean(previous_valid >= 0)
            ),
            "train_global_rate": model.global_rate,
        },
        separators=(",", ":"),
    )
)

print(
    "CANDIDATES "
    + json.dumps(
        {
            key: float(value)
            for key, value in candidate_primary.items()
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)

test = load("test")
test_raw = model.predict(test)
test_rank = {
    name: within_user_rank(test.user_id, score)
    for name, score in test_raw.items()
}

inc_test = np.asarray(
    np.load(inc_test_path), dtype=np.float64
)
if len(inc_test) != len(test.user_id):
    raise ValueError("Incumbent test length mismatch")
inc_test_rank = within_user_rank(test.user_id, inc_test)

if recipe_type == "incumbent":
    test_scores = inc_test_rank
elif recipe_type == "standalone":
    test_scores = test_rank[winner_family]
else:
    test_scores = (
        (1.0 - winner_weight) * inc_test_rank
        + winner_weight * test_rank[winner_family]
    )

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(raw_for_audit, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
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