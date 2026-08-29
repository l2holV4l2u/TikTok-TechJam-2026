import os
import time
import json
import gc
import datetime
import numpy as np

from pipeline.data import load
from pipeline.evaluate import evaluate


START_TIME = time.time()

FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "duration_bucket",
    "upload_type",
    "music_type",
]

WEIGHT_PRESETS = {
    "balanced": np.asarray([1.0, 1.2, 1.0, 0.7, 0.6, 0.5], dtype=np.float64),
    "stable": np.asarray([0.4, 1.4, 1.1, 0.8, 0.7, 0.5], dtype=np.float64),
    "personal": np.asarray([0.6, 1.5, 1.2, 0.9, 0.8, 0.6], dtype=np.float64),
}

PAIR_SMOOTHINGS = [4.0, 12.0, 36.0]
BLEND_ALPHAS = [0.20, 0.40, 0.55, 0.70, 0.82, 0.92]
HALF_LIFE_DAYS = 10.0
ENTITY_SMOOTHING = 30.0


class CombinedSplit:
    def __init__(self, first, second):
        self.user_id = np.concatenate([
            np.asarray(first.user_id, dtype=np.int64),
            np.asarray(second.user_id, dtype=np.int64),
        ])
        self.date = np.concatenate([
            np.asarray(first.date, dtype=np.int32),
            np.asarray(second.date, dtype=np.int32),
        ])
        self.X = {}
        for field in FIELDS:
            self.X[field] = np.concatenate([
                np.asarray(first.X[field], dtype=np.int64),
                np.asarray(second.X[field], dtype=np.int64),
            ])


def date_ordinals(dates):
    dates = np.asarray(dates, dtype=np.int32)
    unique_dates, inverse = np.unique(dates, return_inverse=True)
    ordinals = np.empty(len(unique_dates), dtype=np.float64)
    for i, value in enumerate(unique_dates):
        value = int(value)
        year = value // 10000
        month = (value // 100) % 100
        day = value % 100
        ordinals[i] = datetime.date(year, month, day).toordinal()
    return ordinals[inverse]


def recency_weights(dates, half_life=HALF_LIFE_DAYS):
    ordinal = date_ordinals(dates)
    age = ordinal.max() - ordinal
    return np.exp2(-age / half_life).astype(np.float64)


def lookup_sorted(keys, values, query):
    positions = np.searchsorted(keys, query)
    present = positions < len(keys)
    safe = np.minimum(positions, len(keys) - 1)
    present &= keys[safe] == query

    result = np.zeros(len(query), dtype=np.float64)
    result[present] = values[safe[present]]
    return result


def sufficient_statistics(fit, y_fit, pred):
    y_fit = np.asarray(y_fit, dtype=np.float64)
    users_fit = np.asarray(fit.user_id, dtype=np.int64)
    users_pred = np.asarray(pred.user_id, dtype=np.int64)
    row_weight = recency_weights(fit.date)
    weighted_positive = row_weight * y_fit

    global_rate = float(
        weighted_positive.sum() / np.maximum(row_weight.sum(), 1e-12)
    )

    statistics = {}

    for field in FIELDS:
        fit_value = np.asarray(fit.X[field], dtype=np.int64)
        pred_value = np.asarray(pred.X[field], dtype=np.int64)

        value_size = int(max(
            fit_value.max(initial=0),
            pred_value.max(initial=0),
        )) + 1

        entity_count = np.bincount(
            fit_value,
            weights=row_weight,
            minlength=value_size,
        ).astype(np.float64)
        entity_sum = np.bincount(
            fit_value,
            weights=weighted_positive,
            minlength=value_size,
        ).astype(np.float64)

        pred_entity_count = entity_count[pred_value]
        pred_entity_sum = entity_sum[pred_value]

        pair_keys_fit = users_fit * np.int64(value_size) + fit_value
        pair_keys_pred = users_pred * np.int64(value_size) + pred_value

        unique_keys, inverse = np.unique(
            pair_keys_fit,
            return_inverse=True,
        )
        pair_count_unique = np.bincount(
            inverse,
            weights=row_weight,
            minlength=len(unique_keys),
        ).astype(np.float64)
        pair_sum_unique = np.bincount(
            inverse,
            weights=weighted_positive,
            minlength=len(unique_keys),
        ).astype(np.float64)

        pred_pair_count = lookup_sorted(
            unique_keys, pair_count_unique, pair_keys_pred
        )
        pred_pair_sum = lookup_sorted(
            unique_keys, pair_sum_unique, pair_keys_pred
        )

        statistics[field] = (
            pred_pair_count,
            pred_pair_sum,
            pred_entity_count,
            pred_entity_sum,
        )

        del pair_keys_fit, pair_keys_pred, unique_keys, inverse
        del pair_count_unique, pair_sum_unique
        gc.collect()

    return statistics, global_rate


def safe_logit(probability):
    probability = np.clip(
        np.asarray(probability, dtype=np.float64),
        1e-5,
        1.0 - 1e-5,
    )
    return np.log(probability) - np.log1p(-probability)


def construct_components(statistics, global_rate, pair_smoothing):
    n = len(next(iter(statistics.values()))[0])
    absolute = np.empty((n, len(FIELDS)), dtype=np.float64)
    residual = np.empty((n, len(FIELDS)), dtype=np.float64)

    for j, field in enumerate(FIELDS):
        pair_count, pair_sum, entity_count, entity_sum = statistics[field]

        entity_rate = (
            entity_sum + ENTITY_SMOOTHING * global_rate
        ) / (
            entity_count + ENTITY_SMOOTHING
        )

        pair_rate = (
            pair_sum + pair_smoothing * entity_rate
        ) / (
            pair_count + pair_smoothing
        )

        entity_logit = safe_logit(entity_rate)
        pair_logit = safe_logit(pair_rate)

        absolute[:, j] = pair_logit
        residual[:, j] = pair_logit - entity_logit

    return absolute, residual


def center_by_user(values, user_ids):
    values = np.asarray(values, dtype=np.float64)
    user_ids = np.asarray(user_ids, dtype=np.int64)
    size = int(user_ids.max(initial=0)) + 1

    counts = np.bincount(user_ids, minlength=size).astype(np.float64)
    sums = np.bincount(
        user_ids,
        weights=values,
        minlength=size,
    ).astype(np.float64)
    means = sums / np.maximum(counts, 1.0)

    centered = values - means[user_ids]
    scale = float(np.sqrt(np.mean(centered * centered)))
    if not np.isfinite(scale) or scale < 1e-10:
        scale = 1.0
    return centered / scale


def aggregate_components(matrix, weights, user_ids):
    score = np.asarray(matrix, dtype=np.float64).dot(
        np.asarray(weights, dtype=np.float64)
    )
    return center_by_user(score, user_ids)


def build_candidate_score(
    incumbent,
    user_ids,
    statistics,
    global_rate,
    pair_smoothing,
    preset_name,
    mode,
    alpha,
):
    absolute, residual = construct_components(
        statistics,
        global_rate,
        pair_smoothing,
    )
    matrix = residual if mode == "residual" else absolute
    history_score = aggregate_components(
        matrix,
        WEIGHT_PRESETS[preset_name],
        user_ids,
    )
    incumbent_score = center_by_user(incumbent, user_ids)
    return alpha * incumbent_score + (1.0 - alpha) * history_score


train = load("train")
valid = load("valid")

y_train = np.asarray(train.y, dtype=np.float64)
y_valid = np.asarray(valid.y, dtype=np.int8)

artifacts = os.environ["RUN_ARTIFACTS"]
inc_valid_path = os.path.join(
    artifacts, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    artifacts, "incumbent_test_scores.npy"
)

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if len(inc_valid) != len(y_valid):
    raise RuntimeError("Incumbent validation score length mismatch")

valid_statistics, valid_global_rate = sufficient_statistics(
    train,
    y_train,
    valid,
)

candidates = {}
best_metrics = evaluate(valid.user_id, y_valid, inc_valid)
best_scores = inc_valid.copy()
best_spec = {
    "kind": "incumbent",
}

candidates["incumbent"] = float(best_metrics["primary"])

for smoothing in PAIR_SMOOTHINGS:
    absolute, residual = construct_components(
        valid_statistics,
        valid_global_rate,
        smoothing,
    )

    for preset_name, field_weights in WEIGHT_PRESETS.items():
        for mode, matrix in [
            ("residual", residual),
            ("absolute", absolute),
        ]:
            history_score = aggregate_components(
                matrix,
                field_weights,
                valid.user_id,
            )

            history_name = "{}_{}_s{:g}".format(
                mode, preset_name, smoothing
            )
            history_metrics = evaluate(
                valid.user_id,
                y_valid,
                history_score,
            )
            candidates[history_name] = float(
                history_metrics["primary"]
            )

            incumbent_centered = center_by_user(
                inc_valid,
                valid.user_id,
            )

            for alpha in BLEND_ALPHAS:
                blended = (
                    alpha * incumbent_centered
                    + (1.0 - alpha) * history_score
                )
                name = "{}_a{:.2f}".format(history_name, alpha)
                metrics = evaluate(
                    valid.user_id,
                    y_valid,
                    blended,
                )
                candidates[name] = float(metrics["primary"])

                if metrics["primary"] > best_metrics["primary"]:
                    best_metrics = metrics
                    best_scores = np.asarray(
                        blended, dtype=np.float64
                    ).copy()
                    best_spec = {
                        "kind": "blend",
                        "smoothing": float(smoothing),
                        "preset": preset_name,
                        "mode": mode,
                        "alpha": float(alpha),
                    }

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(best_scores, dtype=np.float64),
    )

print("CANDIDATES " + json.dumps(candidates, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps({
        "selected": best_spec,
        "global_rate_recency_weighted": valid_global_rate,
        "half_life_days": HALF_LIFE_DAYS,
        "incumbent_primary": candidates["incumbent"],
        "best_primary": float(best_metrics["primary"]),
    }, sort_keys=True)
)

del valid_statistics
gc.collect()

test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if len(inc_test) != len(test.user_id):
    raise RuntimeError("Incumbent test score length mismatch")

if best_spec["kind"] == "incumbent":
    test_scores = inc_test
else:
    combined = CombinedSplit(train, valid)
    y_combined = np.concatenate([
        y_train,
        y_valid.astype(np.float64),
    ])

    test_statistics, test_global_rate = sufficient_statistics(
        combined,
        y_combined,
        test,
    )

    test_scores = build_candidate_score(
        incumbent=inc_test,
        user_ids=test.user_id,
        statistics=test_statistics,
        global_rate=test_global_rate,
        pair_smoothing=best_spec["smoothing"],
        preset_name=best_spec["preset"],
        mode=best_spec["mode"],
        alpha=best_spec["alpha"],
    )

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START_TIME
print(
    "METRICS "
    + json.dumps({
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)