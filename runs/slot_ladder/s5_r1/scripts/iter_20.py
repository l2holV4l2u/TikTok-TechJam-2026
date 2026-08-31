import os
import time
import json
from datetime import datetime

import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()

FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "tab",
    "music_type",
    "video_type",
]

# Coefficients control double-counting of correlated video/author and low-cardinality
# content evidence. They are fixed before validation evaluation.
STATIC_COEF = {
    "video_id": 0.45,
    "author_id": 0.45,
    "tag": 0.32,
    "duration_bucket": 0.18,
    "upload_type": 0.20,
    "onehot_feat3": 0.25,
    "tab": 0.18,
    "music_type": 0.10,
    "video_type": 0.08,
}

FILTER_COEF = {
    "video_id": 0.52,
    "author_id": 0.48,
    "tag": 0.30,
    "duration_bucket": 0.16,
    "upload_type": 0.18,
    "onehot_feat3": 0.22,
    "tab": 0.16,
    "music_type": 0.08,
    "video_type": 0.06,
}

TREND_COEF = {
    "video_id": 0.48,
    "author_id": 0.46,
    "tag": 0.32,
    "duration_bucket": 0.16,
    "upload_type": 0.18,
    "onehot_feat3": 0.22,
    "tab": 0.14,
    "music_type": 0.08,
    "video_type": 0.05,
}


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.015, 0.985)
    return np.log(p) - np.log1p(-p)


def ordinal_dates(values):
    values = np.asarray(values, dtype=np.int64)
    unique, inverse = np.unique(values, return_inverse=True)
    ordinals = np.empty(len(unique), dtype=np.int64)
    for i, value in enumerate(unique):
        ordinals[i] = datetime.strptime(str(int(value)), "%Y%m%d").toordinal()
    return ordinals[inverse]


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    sizes = np.diff(np.r_[starts, n])

    ranked = (
        np.arange(n, dtype=np.float64) - np.repeat(starts, sizes)
    ) / np.repeat(np.maximum(sizes - 1, 1), sizes)

    out = np.empty(n, dtype=np.float64)
    out[order] = ranked
    return out


class TemporalContentModels:
    def __init__(self, train):
        y = np.asarray(train.y, dtype=np.float64)
        train_ord = ordinal_dates(train.date)
        self.last_train_ordinal = int(train_ord.max())
        day_index = train_ord - train_ord.min()
        self.n_days = int(day_index.max()) + 1

        daily_count = np.bincount(
            day_index, minlength=self.n_days
        ).astype(np.float64)
        daily_positive = np.bincount(
            day_index, weights=y, minlength=self.n_days
        ).astype(np.float64)

        self.global_rate = float(y.mean())
        self.global_logit = float(safe_logit(self.global_rate))
        self.daily_rate = (
            daily_positive + 80.0 * self.global_rate
        ) / (daily_count + 80.0)
        self.daily_logit = safe_logit(self.daily_rate)

        # x=0 is the final training day, making extrapolation interpretable.
        self.x_day = (
            np.arange(self.n_days, dtype=np.float64) - (self.n_days - 1)
        )
        self.tables = {}

        for field in FIELDS:
            ids = np.asarray(train.X[field], dtype=np.int64)
            card = int(FEATURE_CARDINALITIES[field])
            flat = ids * self.n_days + day_index

            count = np.bincount(
                flat, minlength=card * self.n_days
            ).reshape(card, self.n_days).astype(np.float64)
            positive = np.bincount(
                flat, weights=y, minlength=card * self.n_days
            ).reshape(card, self.n_days).astype(np.float64)

            total = count.sum(axis=1)
            total_positive = positive.sum(axis=1)

            # Family 1: stationary empirical-Bayes generative log odds.
            alpha_static = 35.0 if card > 100 else 70.0
            static_rate = (
                total_positive + alpha_static * self.global_rate
            ) / (total + alpha_static)
            static_residual = safe_logit(static_rate) - self.global_logit

            # Family 2: discounted Bayesian filtering. This is a state estimate
            # rather than a fitted trend: evidence loses half its mass every
            # three days as it gets older.
            age = (self.n_days - 1) - np.arange(self.n_days)
            discount = np.power(0.5, age.astype(np.float64) / 3.0)
            filtered_count = count @ discount
            filtered_positive = positive @ discount
            alpha_filter = 24.0 if card > 100 else 45.0
            filtered_rate = (
                filtered_positive + alpha_filter * self.global_rate
            ) / (filtered_count + alpha_filter)
            filtered_residual = (
                safe_logit(filtered_rate) - self.global_logit
            )

            # Family 3: entity-specific weighted regression of daily residual
            # log odds. Empty entity-days carry no regression weight. The
            # denominator ridge strongly shrinks noisy slopes.
            alpha_day = 12.0 if card > 100 else 25.0
            day_rate = (
                positive + alpha_day * self.daily_rate[None, :]
            ) / (count + alpha_day)
            response = safe_logit(day_rate) - self.daily_logit[None, :]

            w = count
            x = self.x_day[None, :]
            sw = w.sum(axis=1)
            sx = (w * x).sum(axis=1)
            sy = (w * response).sum(axis=1)
            sxx = (w * x * x).sum(axis=1)
            sxy = (w * x * response).sum(axis=1)

            safe_sw = np.maximum(sw, 1.0)
            centered_xx = sxx - sx * sx / safe_sw
            centered_xy = sxy - sx * sy / safe_sw

            slope_ridge = 180.0 if card > 100 else 350.0
            slope = centered_xy / (centered_xx + slope_ridge)
            slope = np.clip(slope, -0.025, 0.025)

            intercept = (sy - slope * sx) / safe_sw
            reliability = sw / (sw + (35.0 if card > 100 else 80.0))
            intercept *= reliability
            slope *= reliability
            intercept[sw == 0] = 0.0
            slope[sw == 0] = 0.0

            self.tables[field] = {
                "static": static_residual.astype(np.float32),
                "filtered": filtered_residual.astype(np.float32),
                "intercept": intercept.astype(np.float32),
                "slope": slope.astype(np.float32),
                "seen": (total > 0),
            }

    def predict(self, split):
        n = len(split.user_id)
        static_score = np.zeros(n, dtype=np.float64)
        filtered_score = np.zeros(n, dtype=np.float64)
        trend_score = np.zeros(n, dtype=np.float64)

        future_days = (
            ordinal_dates(split.date) - self.last_train_ordinal
        ).astype(np.float64)
        # Avoid unbounded extrapolation while preserving the actual distinction
        # between the validation and later hidden-test horizons.
        future_days = np.clip(future_days, 1.0, 18.0)

        unseen_rows = np.zeros(n, dtype=np.int16)

        for field in FIELDS:
            ids = np.asarray(split.X[field], dtype=np.int64)
            table = self.tables[field]
            max_id = len(table["static"]) - 1
            valid_id = (ids >= 0) & (ids <= max_id)
            safe_id = np.where(valid_id, ids, 0)

            seen = valid_id & table["seen"][safe_id]
            unseen_rows += (~seen).astype(np.int16)

            static_component = table["static"][safe_id]
            filtered_component = table["filtered"][safe_id]
            trend_component = (
                table["intercept"][safe_id]
                + table["slope"][safe_id] * future_days
            )

            static_component = np.where(seen, static_component, 0.0)
            filtered_component = np.where(seen, filtered_component, 0.0)
            trend_component = np.where(seen, trend_component, 0.0)
            trend_component = np.clip(trend_component, -1.8, 1.8)

            static_score += STATIC_COEF[field] * static_component
            filtered_score += FILTER_COEF[field] * filtered_component
            trend_score += TREND_COEF[field] * trend_component

        # A fourth structurally distinct prediction is a product-of-experts
        # consensus: only evidence agreed on by stationary and filtered
        # estimators is retained strongly.
        consensus = (
            0.55 * static_score
            + 0.55 * filtered_score
            - 0.10 * np.abs(static_score - filtered_score)
        )

        return {
            "static_generative": static_score,
            "discounted_filter": filtered_score,
            "temporal_trend": trend_score,
            "content_consensus": consensus,
        }, unseen_rows


train = load("train")
valid = load("valid")

models = TemporalContentModels(train)
valid_raw, valid_unseen = models.predict(valid)

valid_rank = {
    name: within_user_rank(valid.user_id, score)
    for name, score in valid_raw.items()
}

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

if not os.path.exists(inc_valid_path):
    raise FileNotFoundError("Trusted incumbent validation scores are missing")
if not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent test scores are missing")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if len(inc_valid) != len(valid.user_id):
    raise ValueError("Incumbent validation length mismatch")
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

candidate_scores = {"incumbent": inc_valid_rank}
candidate_primary = {
    "incumbent": float(
        evaluate(valid.user_id, valid.y, inc_valid_rank)["primary"]
    )
}
recipes = {"incumbent": ("incumbent", "", 0.0)}

blend_weights = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)

for family, own_rank in valid_rank.items():
    standalone_name = family + "_standalone"
    candidate_scores[standalone_name] = own_rank
    candidate_primary[standalone_name] = float(
        evaluate(valid.user_id, valid.y, own_rank)["primary"]
    )
    recipes[standalone_name] = ("standalone", family, 1.0)

    for weight in blend_weights:
        name = f"{family}_blend_{weight:.2f}"
        score = (1.0 - weight) * inc_valid_rank + weight * own_rank
        candidate_scores[name] = score
        candidate_primary[name] = float(
            evaluate(valid.user_id, valid.y, score)["primary"]
        )
        recipes[name] = ("blend", family, weight)

winner = max(candidate_primary, key=candidate_primary.get)
valid_scores = candidate_scores[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)
recipe_type, winner_family, winner_weight = recipes[winner]

best_own_family = max(
    valid_rank,
    key=lambda name: candidate_primary[name + "_standalone"],
)
raw_for_audit = valid_rank[
    winner_family if winner_family in valid_rank else best_own_family
]

print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner,
            "best_own_family": best_own_family,
            "standalone": {
                name: candidate_primary[name + "_standalone"]
                for name in valid_rank
            },
            "incumbent": candidate_primary["incumbent"],
            "mean_unseen_content_fields_valid": float(valid_unseen.mean()),
            "train_global_rate": models.global_rate,
            "train_last_day_rate": float(models.daily_rate[-1]),
            "mean_abs_video_trend": float(
                np.mean(np.abs(models.tables["video_id"]["slope"]))
            ),
            "mean_abs_author_trend": float(
                np.mean(np.abs(models.tables["author_id"]["slope"]))
            ),
        },
        separators=(",", ":"),
    )
)

print(
    "CANDIDATES "
    + json.dumps(
        {k: float(v) for k, v in candidate_primary.items()},
        sort_keys=True,
        separators=(",", ":"),
    )
)

test = load("test")
test_raw, _ = models.predict(test)
test_rank = {
    name: within_user_rank(test.user_id, score)
    for name, score in test_raw.items()
}

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
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