import os
import time
import json
from datetime import datetime

import numpy as np
from scipy.special import ndtri

from pipeline.data import load
from pipeline.evaluate import evaluate


START = time.time()


def ordinal_dates(values):
    values = np.asarray(values, dtype=np.int64)
    unique, inverse = np.unique(values, return_inverse=True)
    mapped = np.array(
        [
            datetime.strptime(str(int(value)), "%Y%m%d").toordinal()
            for value in unique
        ],
        dtype=np.int64,
    )
    return mapped[inverse]


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.01, 0.99)
    return np.log(p) - np.log1p(-p)


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

    rank = (
        np.arange(n, dtype=np.float64) - np.repeat(starts, sizes)
    ) / np.repeat(np.maximum(sizes - 1, 1), sizes)

    out = np.empty(n, dtype=np.float64)
    out[order] = rank
    return out


def incumbent_normal_score(user_ids, scores):
    rank = within_user_rank(user_ids, scores)
    # A bounded Gaussian quantile gives the incumbent a common within-query
    # scale while retaining substantially more top-rank separation than Borda.
    q = 0.02 + 0.96 * rank
    return ndtri(q)


class TemporalCalibrators:
    def __init__(self, train):
        y = np.asarray(train.y, dtype=np.float64)
        ords = ordinal_dates(train.date)
        self.first_ord = int(ords.min())
        self.last_ord = int(ords.max())
        day = ords - self.first_ord
        n_days = int(day.max()) + 1
        self.n_days = n_days
        self.global_rate = float(y.mean())
        self.global_logit = float(safe_logit(self.global_rate))

        hour = np.asarray(train.X["hour"], dtype=np.int64)
        hour = np.clip(hour, 0, 23)
        weekday = ords % 7

        day_count = np.bincount(day, minlength=n_days).astype(np.float64)
        day_pos = np.bincount(
            day, weights=y, minlength=n_days
        ).astype(np.float64)
        day_rate = (
            day_pos + 300.0 * self.global_rate
        ) / (day_count + 300.0)
        day_logit = safe_logit(day_rate)

        # Robust recency-weighted linear drift. Centering at the final train
        # day makes extrapolation stable and directly interpretable.
        x = np.arange(n_days, dtype=np.float64) - (n_days - 1)
        recency_weight = day_count * np.power(
            0.5, ((n_days - 1) - np.arange(n_days)) / 5.0
        )
        sw = recency_weight.sum()
        xbar = np.sum(recency_weight * x) / sw
        ybar = np.sum(recency_weight * day_logit) / sw
        numerator = np.sum(recency_weight * (x - xbar) * (day_logit - ybar))
        denominator = np.sum(recency_weight * (x - xbar) ** 2) + 2500.0
        self.trend_slope = float(
            np.clip(numerator / denominator, -0.035, 0.035)
        )
        self.trend_intercept = float(
            ybar - self.trend_slope * xbar
        )

        # Hierarchical hour effect, estimated as a residual around each
        # training day's baseline so date composition does not confound it.
        hour_count = np.bincount(hour, minlength=24).astype(np.float64)
        hour_resid_sum = np.bincount(
            hour,
            weights=y - day_rate[day],
            minlength=24,
        ).astype(np.float64)
        self.hour_effect = hour_resid_sum / (hour_count + 2500.0)
        self.hour_effect = np.clip(
            self.hour_effect / max(self.global_rate * (1-self.global_rate), 0.05),
            -0.30,
            0.30,
        )

        # Empirical-Bayes weekday-by-hour hazard. This captures recurring
        # weekly audience regimes without extrapolating entity identities.
        cell = weekday * 24 + hour
        cell_count = np.bincount(cell, minlength=7 * 24).astype(np.float64)
        cell_pos = np.bincount(
            cell, weights=y, minlength=7 * 24
        ).astype(np.float64)

        weekday_count = np.bincount(
            weekday, minlength=7
        ).astype(np.float64)
        weekday_pos = np.bincount(
            weekday, weights=y, minlength=7
        ).astype(np.float64)
        weekday_rate = (
            weekday_pos + 1800.0 * self.global_rate
        ) / (weekday_count + 1800.0)

        prior = np.repeat(weekday_rate, 24)
        cell_rate = (
            cell_pos + 700.0 * prior
        ) / (cell_count + 700.0)
        self.seasonal_logit = safe_logit(cell_rate) - self.global_logit

        # Recent-kernel family: each future weekday/hour receives a weighted
        # average of observed daily rates at nearby hours, with evidence from
        # later train days carrying more mass.
        kernel_num = np.zeros((7, 24), dtype=np.float64)
        kernel_den = np.zeros((7, 24), dtype=np.float64)
        age = (n_days - 1) - day
        row_recency = np.power(0.5, age / 3.5)

        for h_target in range(24):
            circular_distance = np.minimum(
                np.abs(hour - h_target),
                24 - np.abs(hour - h_target),
            )
            hour_kernel = np.exp(-circular_distance / 2.0)
            base_weight = row_recency * hour_kernel
            for wd_target in range(7):
                weekday_distance = np.minimum(
                    np.abs(weekday - wd_target),
                    7 - np.abs(weekday - wd_target),
                )
                weights = base_weight * np.exp(-weekday_distance / 0.75)
                kernel_num[wd_target, h_target] = np.sum(weights * y)
                kernel_den[wd_target, h_target] = np.sum(weights)

        kernel_rate = (
            kernel_num + 500.0 * self.global_rate
        ) / (kernel_den + 500.0)
        self.kernel_logit = safe_logit(kernel_rate) - self.global_logit

    def predict(self, split):
        ords = ordinal_dates(split.date)
        future = ords.astype(np.float64) - self.last_ord
        hour = np.clip(
            np.asarray(split.X["hour"], dtype=np.int64), 0, 23
        )
        weekday = ords % 7
        cell = weekday * 24 + hour

        # Continuous time separates impressions within the same day/hour
        # without using any row outcome. It only refines the fitted trend.
        time_days = (
            np.asarray(split.time_ms, dtype=np.float64) / 86400000.0
        )
        date_midnight_days = ords.astype(np.float64)
        frac_day = np.clip(time_days - np.floor(time_days), 0.0, 1.0)
        future_continuous = future + frac_day

        trend = (
            self.trend_intercept
            + self.trend_slope * future_continuous
            + self.hour_effect[hour]
        )

        seasonal = (
            self.seasonal_logit[cell]
            + 0.45 * self.trend_slope * future_continuous
        )

        kernel = (
            self.kernel_logit[weekday, hour]
            + 0.65 * self.trend_slope * future_continuous
        )

        # A hazard consensus keeps only temporal corrections supported by both
        # the parametric trend and recurring seasonal estimator.
        consensus = (
            0.45 * trend
            + 0.55 * seasonal
            - 0.12 * np.abs(trend - seasonal)
        )

        return {
            "linear_drift_hazard": trend,
            "hierarchical_seasonal_hazard": seasonal,
            "recent_kernel_hazard": kernel,
            "temporal_consensus": consensus,
        }


train = load("train")
valid = load("valid")

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path):
    raise FileNotFoundError("Trusted incumbent validation scores missing")
if not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent test scores missing")

inc_valid_raw = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if len(inc_valid_raw) != len(valid.user_id):
    raise ValueError("Validation incumbent length mismatch")

inc_valid = incumbent_normal_score(valid.user_id, inc_valid_raw)
model = TemporalCalibrators(train)
own_valid = model.predict(valid)

candidate_scores = {"incumbent": inc_valid}
candidate_primary = {
    "incumbent": float(
        evaluate(valid.user_id, valid.y, inc_valid)["primary"]
    )
}
recipes = {"incumbent": ("incumbent", "", 0.0)}

# Temporal corrections are in log-odds units while the incumbent is on a
# Gaussian rank scale. The grid therefore tests conservative calibrations.
weights = (0.03, 0.06, 0.10, 0.15, 0.22, 0.30)

for family, correction in own_valid.items():
    correction = np.asarray(correction, dtype=np.float64)
    standalone = within_user_rank(valid.user_id, correction)
    standalone_name = family + "_standalone"
    candidate_scores[standalone_name] = standalone
    candidate_primary[standalone_name] = float(
        evaluate(valid.user_id, valid.y, standalone)["primary"]
    )
    recipes[standalone_name] = ("standalone", family, 1.0)

    for weight in weights:
        name = f"{family}_cal_{weight:.2f}"
        score = inc_valid + weight * correction
        candidate_scores[name] = score
        candidate_primary[name] = float(
            evaluate(valid.user_id, valid.y, score)["primary"]
        )
        recipes[name] = ("calibration", family, weight)

winner = max(candidate_primary, key=candidate_primary.get)
valid_scores = candidate_scores[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)
recipe_type, winner_family, winner_weight = recipes[winner]

best_own_family = max(
    own_valid,
    key=lambda name: candidate_primary[name + "_standalone"],
)
raw_for_audit = within_user_rank(
    valid.user_id,
    own_valid[
        winner_family if winner_family in own_valid else best_own_family
    ],
)

print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": winner,
            "incumbent_primary": candidate_primary["incumbent"],
            "trend_slope_logodds_per_day": model.trend_slope,
            "best_own_family": best_own_family,
            "standalone": {
                name: candidate_primary[name + "_standalone"]
                for name in own_valid
            },
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
inc_test_raw = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test_raw) != len(test.user_id):
    raise ValueError("Test incumbent length mismatch")

inc_test = incumbent_normal_score(test.user_id, inc_test_raw)
own_test = model.predict(test)

if recipe_type == "incumbent":
    test_scores = inc_test
elif recipe_type == "standalone":
    test_scores = within_user_rank(
        test.user_id, own_test[winner_family]
    )
else:
    test_scores = (
        inc_test + winner_weight * own_test[winner_family]
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