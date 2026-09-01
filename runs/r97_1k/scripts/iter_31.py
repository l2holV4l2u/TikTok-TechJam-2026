import os
import gc
import json
import time
import warnings
import numpy as np
from scipy.special import ndtri

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")

START = time.time()
OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")

if OUT:
    os.makedirs(OUT, exist_ok=True)

FIELDS = [
    "user_id",
    "author_id",
    "video_id",
    "tag",
    "tab",
    "onehot_feat3",
    "onehot_feat8",
    "duration_bucket",
    "upload_type",
]

FIELD_WEIGHTS = {
    "user_id": 2.00,
    "author_id": 1.25,
    "video_id": 0.65,
    "tag": 0.90,
    "tab": 1.20,
    "onehot_feat3": 1.00,
    "onehot_feat8": 0.85,
    "duration_bucket": 0.55,
    "upload_type": 0.55,
}

PRIOR_STRENGTHS = {
    "user_id": 120.0,
    "author_id": 250.0,
    "video_id": 180.0,
    "tag": 800.0,
    "tab": 800.0,
    "onehot_feat3": 220.0,
    "onehot_feat8": 220.0,
    "duration_bucket": 700.0,
    "upload_type": 600.0,
}

HALF_LIFE = 3.5
MAX_SLOPE_PER_DAY = 0.018
ALPHAS = [0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.14, 0.20, 0.28]
GAMMAS = [0.7, 1.0, 1.5, 2.0]


def day_number(date_array):
    date_array = np.asarray(date_array, dtype=np.int64)
    unique = np.unique(date_array)
    converted = {}

    for value in unique:
        text = str(int(value))
        iso = text[:4] + "-" + text[4:6] + "-" + text[6:8]
        converted[int(value)] = int(
            np.datetime64(iso, "D").astype(np.int64)
        )

    out = np.empty(len(date_array), dtype=np.int32)
    for value, ordinal in converted.items():
        out[date_array == value] = ordinal
    return out


def safe_ids(values, card):
    values = np.asarray(values, dtype=np.int64)
    return np.where(
        (values >= 0) & (values < card), values, 0
    ).astype(np.int64, copy=False)


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float32), 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    start_positions = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local_rank = (
        np.arange(n, dtype=np.float64)
        - start_positions.astype(np.float64)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]

    sizes = np.diff(np.r_[-1, np.flatnonzero(ends)]).astype(np.float64)
    group_index = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[group_index] - 1.0, 1.0)

    result = np.empty(n, dtype=np.float32)
    result[order] = (local_rank / denom).astype(np.float32)
    return result


def copula(rank):
    rank = np.asarray(rank, dtype=np.float64)
    return ndtri(np.clip(rank, 1e-4, 1.0 - 1e-4)).astype(np.float32)


def fit_temporal_tables(train):
    labels = np.asarray(train.y, dtype=np.float32)
    dates = day_number(train.date)
    last_day = int(np.max(dates))
    relative_day = (dates - last_day).astype(np.float32)
    global_prior = float(np.mean(labels))

    recency_weight = np.exp(
        np.log(2.0) * relative_day / HALF_LIFE
    ).astype(np.float32)

    recent_mask = relative_day >= -2.0
    previous_mask = (relative_day >= -7.0) & (relative_day <= -3.0)

    tables = {}

    for field in FIELDS:
        card = int(FEATURE_CARDINALITIES[field])
        ids = safe_ids(train.X[field], card)
        strength = float(PRIOR_STRENGTHS[field])

        total_count = np.bincount(
            ids, minlength=card
        ).astype(np.float32)
        total_sum = np.bincount(
            ids, weights=labels, minlength=card
        ).astype(np.float32)

        weighted_count = np.bincount(
            ids, weights=recency_weight, minlength=card
        ).astype(np.float32)
        weighted_sum = np.bincount(
            ids,
            weights=recency_weight * labels,
            minlength=card,
        ).astype(np.float32)

        recency_rate = (
            weighted_sum + strength * global_prior
        ) / (weighted_count + strength)

        # Weighted least-squares trend in daily probability. Sufficient
        # statistics avoid materializing entity-by-day matrices.
        x = relative_day
        sw = weighted_count
        sx = np.bincount(
            ids, weights=recency_weight * x, minlength=card
        ).astype(np.float32)
        sy = weighted_sum
        sxx = np.bincount(
            ids, weights=recency_weight * x * x, minlength=card
        ).astype(np.float32)
        sxy = np.bincount(
            ids,
            weights=recency_weight * x * labels,
            minlength=card,
        ).astype(np.float32)

        denominator = sw * sxx - sx * sx
        numerator = sw * sxy - sx * sy
        slope = np.divide(
            numerator,
            denominator,
            out=np.zeros(card, dtype=np.float32),
            where=np.abs(denominator) > 1e-5,
        )

        reliability = total_count / (total_count + 4.0 * strength)
        slope *= reliability.astype(np.float32)
        slope = np.clip(
            slope, -MAX_SLOPE_PER_DAY, MAX_SLOPE_PER_DAY
        ).astype(np.float32)

        recent_count = np.bincount(
            ids[recent_mask], minlength=card
        ).astype(np.float32)
        recent_sum = np.bincount(
            ids[recent_mask],
            weights=labels[recent_mask],
            minlength=card,
        ).astype(np.float32)

        previous_count = np.bincount(
            ids[previous_mask], minlength=card
        ).astype(np.float32)
        previous_sum = np.bincount(
            ids[previous_mask],
            weights=labels[previous_mask],
            minlength=card,
        ).astype(np.float32)

        recent_rate = (
            recent_sum + strength * global_prior
        ) / (recent_count + strength)
        previous_rate = (
            previous_sum + strength * global_prior
        ) / (previous_count + strength)

        change = recent_rate - previous_rate
        change_reliability = (
            np.minimum(recent_count, previous_count)
            / (
                np.minimum(recent_count, previous_count)
                + 2.0 * strength
            )
        )
        change *= change_reliability
        change = np.clip(change, -0.10, 0.10).astype(np.float32)

        full_rate = (
            total_sum + strength * global_prior
        ) / (total_count + strength)

        tables[field] = {
            "recency_rate": recency_rate.astype(np.float32),
            "full_rate": full_rate.astype(np.float32),
            "slope": slope,
            "change": change,
            "count": total_count,
        }

        print(
            "FINDINGS field=%s mean_abs_slope=%.7f "
            "mean_abs_change=%.7f active_ids=%d"
            % (
                field,
                float(np.mean(np.abs(slope[total_count > 0]))),
                float(np.mean(np.abs(change[total_count > 0]))),
                int(np.sum(total_count > 0)),
            ),
            flush=True,
        )

        del (
            ids,
            total_count,
            total_sum,
            weighted_count,
            weighted_sum,
            sw,
            sx,
            sy,
            sxx,
            sxy,
            recent_count,
            recent_sum,
            previous_count,
            previous_sum,
        )
        gc.collect()

    return tables, last_day, global_prior


def temporal_scores(split, tables, train_last_day, global_prior):
    dates = day_number(split.date)
    horizon = np.maximum(
        dates.astype(np.float32) - float(train_last_day), 1.0
    )

    n = len(split.user_id)
    outputs = {
        "recency_level": np.zeros(n, dtype=np.float32),
        "linear_extrapolation": np.zeros(n, dtype=np.float32),
        "change_point": np.zeros(n, dtype=np.float32),
    }
    weight_total = 0.0
    prior_logit = float(logit(np.asarray([global_prior]))[0])

    for field in FIELDS:
        card = int(FEATURE_CARDINALITIES[field])
        ids = safe_ids(split.X[field], card)
        table = tables[field]
        weight = float(FIELD_WEIGHTS[field])
        weight_total += weight

        level = table["recency_rate"][ids]
        linear_probability = np.clip(
            level + table["slope"][ids] * horizon,
            0.015,
            0.985,
        )

        # A detected last-three-day shift is extrapolated conservatively and
        # saturates after one validation week rather than diverging.
        change_scale = np.minimum(horizon / 4.0, 2.0)
        change_probability = np.clip(
            level + table["change"][ids] * change_scale,
            0.015,
            0.985,
        )

        outputs["recency_level"] += weight * (
            logit(level) - prior_logit
        )
        outputs["linear_extrapolation"] += weight * (
            logit(linear_probability) - prior_logit
        )
        outputs["change_point"] += weight * (
            logit(change_probability) - prior_logit
        )

    for name in outputs:
        outputs[name] /= max(weight_total, 1e-6)

    # Consensus only rewards drift directions shared by estimators with
    # different temporal assumptions.
    outputs["temporal_consensus"] = (
        outputs["recency_level"]
        + outputs["linear_extrapolation"]
        + outputs["change_point"]
    ) / 3.0

    return outputs


def candidate_metrics(uid, y, scores):
    result = evaluate(uid, y, scores)
    return {
        "primary": float(result["primary"]),
        "gauc": float(result["gauc"]),
        "ndcg@5": float(result["ndcg@5"]),
    }


inc_valid_path = (
    os.path.join(SHARED, "incumbent_valid_scores.npy")
    if SHARED else ""
)
inc_test_path = (
    os.path.join(SHARED, "incumbent_test_scores.npy")
    if SHARED else ""
)

if not (
    inc_valid_path
    and inc_test_path
    and os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise RuntimeError("Trusted incumbent artifacts are required")

train = load("train")
valid = load("valid")

valid_uid = np.asarray(valid.user_id, dtype=np.int64)
valid_y = np.asarray(valid.y)

tables, train_last_day, global_prior = fit_temporal_tables(train)
valid_raw_families = temporal_scores(
    valid, tables, train_last_day, global_prior
)

inc_valid_raw = np.asarray(
    np.load(inc_valid_path, mmap_mode="r"),
    dtype=np.float32,
)
if len(inc_valid_raw) != len(valid_uid):
    raise RuntimeError("Incumbent validation length mismatch")

inc_valid_rank = within_user_rank(valid_uid, inc_valid_raw)
inc_valid_copula = copula(inc_valid_rank)

candidate_log = {}
inc_metrics = candidate_metrics(valid_uid, valid_y, inc_valid_rank)
candidate_log["trusted_incumbent"] = inc_metrics["primary"]

best_scores = inc_valid_rank.copy()
best_metrics = inc_metrics
best_family = None
best_alpha = 0.0
best_gamma = 1.0
best_family_rank = None

valid_family_ranks = {}

for family, raw_scores in valid_raw_families.items():
    family_rank = within_user_rank(valid_uid, raw_scores)
    valid_family_ranks[family] = family_rank

    standalone = candidate_metrics(
        valid_uid, valid_y, family_rank
    )
    candidate_log[family + "_standalone"] = standalone["primary"]

    corr = float(np.corrcoef(inc_valid_rank, family_rank)[0, 1])
    disagreement = float(
        np.mean(np.abs(inc_valid_rank - family_rank))
    )

    print(
        "FINDINGS family=%s standalone_primary=%.6f "
        "gauc=%.6f ndcg5=%.6f incumbent_corr=%.6f "
        "mean_abs_disagreement=%.6f"
        % (
            family,
            standalone["primary"],
            standalone["gauc"],
            standalone["ndcg@5"],
            corr,
            disagreement,
        ),
        flush=True,
    )

    family_best = standalone["primary"]
    family_best_name = family + "_standalone"

    for gamma in GAMMAS:
        shaped_rank = np.power(
            np.clip(family_rank, 0.0, 1.0), gamma
        ).astype(np.float32)
        family_copula = copula(shaped_rank)

        for alpha in ALPHAS:
            blended = (
                (1.0 - alpha) * inc_valid_copula
                + alpha * family_copula
            ).astype(np.float32)

            metrics = candidate_metrics(
                valid_uid, valid_y, blended
            )
            name = "%s_copula_g%.1f_a%.3f" % (
                family, gamma, alpha
            )

            if metrics["primary"] > family_best:
                family_best = metrics["primary"]
                family_best_name = name

            if metrics["primary"] > best_metrics["primary"]:
                best_scores = blended.copy()
                best_metrics = metrics
                best_family = family
                best_alpha = float(alpha)
                best_gamma = float(gamma)
                best_family_rank = family_rank.copy()

    candidate_log[family + "_best_blend"] = family_best
    print(
        "FINDINGS family=%s best_candidate=%s "
        "best_family_primary=%.6f"
        % (family, family_best_name, family_best),
        flush=True,
    )

print(
    "FINDINGS winner=%s alpha=%.3f gamma=%.1f "
    "incumbent_primary=%.6f final_primary=%.6f"
    % (
        best_family if best_family is not None else "trusted_incumbent",
        best_alpha,
        best_gamma,
        inc_metrics["primary"],
        best_metrics["primary"],
    ),
    flush=True,
)
print(
    "CANDIDATES " + json.dumps(candidate_log, sort_keys=True),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )

    raw_to_save = (
        best_family_rank
        if best_family_rank is not None
        else valid_family_ranks[
            max(
                valid_family_ranks,
                key=lambda name: candidate_log[
                    name + "_standalone"
                ],
            )
        ]
    )
    np.save(
        os.path.join(OUT, "scores_valid_raw.npy"),
        np.asarray(raw_to_save, dtype=np.float64),
    )

del valid_raw_families
del valid_family_ranks
del inc_valid_raw
gc.collect()

test = load("test")
test_uid = np.asarray(test.user_id, dtype=np.int64)

inc_test_raw = np.asarray(
    np.load(inc_test_path, mmap_mode="r"),
    dtype=np.float32,
)
if len(inc_test_raw) != len(test_uid):
    raise RuntimeError("Incumbent test length mismatch")

inc_test_rank = within_user_rank(test_uid, inc_test_raw)

if best_family is None:
    test_scores = inc_test_rank
else:
    test_raw_families = temporal_scores(
        test, tables, train_last_day, global_prior
    )
    selected_test_rank = within_user_rank(
        test_uid, test_raw_families[best_family]
    )
    selected_test_rank = np.power(
        np.clip(selected_test_rank, 0.0, 1.0),
        best_gamma,
    ).astype(np.float32)

    test_scores = (
        (1.0 - best_alpha) * copula(inc_test_rank)
        + best_alpha * copula(selected_test_rank)
    ).astype(np.float32)

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)

print(
    "METRICS "
    + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": elapsed,
    })
)