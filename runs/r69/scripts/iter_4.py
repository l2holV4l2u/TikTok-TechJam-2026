import os
import time
import json
import gc
import numpy as np

from pipeline.data import load
from pipeline.evaluate import evaluate


START = time.time()

train = load("train")
valid = load("valid")
test = load("test")

y_train = np.asarray(train.y, dtype=np.float64)
y_valid = np.asarray(valid.y, dtype=np.int8)

artifacts = os.environ.get("RUN_ARTIFACTS", "")
inc_valid_path = os.path.join(artifacts, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(artifacts, "incumbent_test_scores.npy")

if not (os.path.exists(inc_valid_path) and os.path.exists(inc_test_path)):
    raise RuntimeError("Trusted incumbent predictions are required for this experiment")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if inc_valid.shape != (len(valid.user_id),):
    raise RuntimeError("Validation incumbent shape mismatch")
if inc_test.shape != (len(test.user_id),):
    raise RuntimeError("Test incumbent shape mismatch")


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd < 1e-10:
        sd = 1.0
    return (x - mu) / sd


def sigmoid(x):
    x = np.clip(x, -25.0, 25.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def recency_weights(dates, half_life_days=None):
    dates = np.asarray(dates, dtype=np.int32)
    if half_life_days is None:
        return np.ones(len(dates), dtype=np.float64)

    unique_dates = np.unique(dates)
    last = np.datetime64("2022-04-21")
    mapping = {}
    for d in unique_dates:
        text = str(int(d))
        dt = np.datetime64(
            text[:4] + "-" + text[4:6] + "-" + text[6:8]
        )
        age = float((last - dt) / np.timedelta64(1, "D"))
        mapping[int(d)] = max(age, 0.0)

    ages = np.zeros(len(dates), dtype=np.float64)
    for d, age in mapping.items():
        ages[dates == d] = age

    w = np.exp(-np.log(2.0) * ages / float(half_life_days))
    w /= max(float(np.mean(w)), 1e-12)
    return w


def map_sparse(unique_keys, values, query_keys):
    query_keys = np.asarray(query_keys, dtype=np.int64)
    pos = np.searchsorted(unique_keys, query_keys)
    matched = pos < len(unique_keys)
    safe = np.minimum(pos, len(unique_keys) - 1)
    matched &= unique_keys[safe] == query_keys

    result = np.zeros(len(query_keys), dtype=np.float64)
    if np.any(matched):
        result[matched] = values[pos[matched]]
    return result, matched


def interaction_residual(
    field,
    sample_weight,
    smoothing,
    user_strength=35.0,
    category_strength=60.0,
):
    """
    Estimate a shrunk user-by-category residual.

    The expected response is an additive-logit combination of the user's
    overall long-view propensity and the category's global propensity.
    The pair statistic therefore focuses on personalized preference rather
    than duplicating global popularity or user activity.
    """
    tr_user = np.asarray(train.user_id, dtype=np.int64)
    va_user = np.asarray(valid.user_id, dtype=np.int64)
    te_user = np.asarray(test.user_id, dtype=np.int64)

    tr_cat = np.asarray(train.X[field], dtype=np.int64)
    va_cat = np.asarray(valid.X[field], dtype=np.int64)
    te_cat = np.asarray(test.X[field], dtype=np.int64)

    user_card = int(
        max(tr_user.max(), va_user.max(), te_user.max()) + 1
    )
    cat_card = int(
        max(tr_cat.max(), va_cat.max(), te_cat.max()) + 1
    )

    w = np.asarray(sample_weight, dtype=np.float64)
    wy = w * y_train
    global_rate = float(wy.sum() / max(w.sum(), 1e-12))

    user_count = np.bincount(
        tr_user, weights=w, minlength=user_card
    ).astype(np.float64)
    user_sum = np.bincount(
        tr_user, weights=wy, minlength=user_card
    ).astype(np.float64)
    user_rate = (
        user_sum + user_strength * global_rate
    ) / (user_count + user_strength)

    cat_count = np.bincount(
        tr_cat, weights=w, minlength=cat_card
    ).astype(np.float64)
    cat_sum = np.bincount(
        tr_cat, weights=wy, minlength=cat_card
    ).astype(np.float64)
    cat_rate = (
        cat_sum + category_strength * global_rate
    ) / (cat_count + category_strength)

    expected = sigmoid(
        logit(user_rate[tr_user])
        + logit(cat_rate[tr_cat])
        - logit(global_rate)
    )
    row_residual = w * (y_train - expected)

    pair_key = tr_user * np.int64(cat_card) + tr_cat
    unique_pair, inverse = np.unique(pair_key, return_inverse=True)

    pair_count = np.bincount(
        inverse, weights=w, minlength=len(unique_pair)
    ).astype(np.float64)
    pair_residual_sum = np.bincount(
        inverse, weights=row_residual, minlength=len(unique_pair)
    ).astype(np.float64)

    pair_value = pair_residual_sum / (pair_count + float(smoothing))

    # Reliability tempering avoids rare pairs receiving the same influence
    # as repeatedly observed preferences.
    reliability = np.sqrt(
        pair_count / (pair_count + float(smoothing))
    )
    pair_value *= reliability

    va_key = va_user * np.int64(cat_card) + va_cat
    te_key = te_user * np.int64(cat_card) + te_cat

    va_score, va_known = map_sparse(unique_pair, pair_value, va_key)
    te_score, te_known = map_sparse(unique_pair, pair_value, te_key)

    # Normalize using the distribution among train impressions. This scale
    # is label-independent for validation and is reused unchanged on test.
    train_pair_values = pair_value[inverse]
    scale = float(np.std(train_pair_values))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0

    va_score /= scale
    te_score /= scale

    finding = {
        "pairs": int(len(unique_pair)),
        "valid_known": float(np.mean(va_known)),
        "test_known": float(np.mean(te_known)),
        "scale": scale,
    }
    return va_score, te_score, finding


fields = [
    ("video_id", 5.0),
    ("author_id", 7.0),
    ("tag", 12.0),
    ("duration_bucket", 12.0),
    ("upload_type", 12.0),
    ("music_type", 15.0),
    ("tab", 15.0),
    ("onehot_feat1", 15.0),
    ("onehot_feat3", 10.0),
    ("onehot_feat7", 12.0),
    ("onehot_feat8", 12.0),
]

weight_schemes = {
    "all_time": recency_weights(train.date, None),
    "recent_h14": recency_weights(train.date, 14.0),
    "recent_h7": recency_weights(train.date, 7.0),
}

valid_components = {}
test_components = {}
findings = {}

for scheme_name, weights in weight_schemes.items():
    for field, smoothing in fields:
        name = scheme_name + "__" + field
        va, te, info = interaction_residual(
            field=field,
            sample_weight=weights,
            smoothing=smoothing,
        )
        valid_components[name] = va
        test_components[name] = te
        findings[name] = info

gc.collect()


def average_components(names, source):
    arrays = [source[name] for name in names]
    if not arrays:
        raise ValueError("Empty component collection")
    result = np.zeros_like(arrays[0], dtype=np.float64)
    for x in arrays:
        result += x
    result /= np.sqrt(float(len(arrays)))
    return result


# Prespecified groups limit validation adaptivity while testing whether
# stable entity memory and coarse-content preference are complementary.
groups = {
    "alltime_entity": [
        "all_time__video_id",
        "all_time__author_id",
    ],
    "alltime_content": [
        "all_time__tag",
        "all_time__duration_bucket",
        "all_time__upload_type",
        "all_time__music_type",
        "all_time__tab",
    ],
    "alltime_side": [
        "all_time__onehot_feat1",
        "all_time__onehot_feat3",
        "all_time__onehot_feat7",
        "all_time__onehot_feat8",
    ],
    "alltime_all": [
        "all_time__" + field for field, _ in fields
    ],
    "h14_all": [
        "recent_h14__" + field for field, _ in fields
    ],
    "h7_all": [
        "recent_h7__" + field for field, _ in fields
    ],
    "mixed_entity_recent_content": [
        "all_time__video_id",
        "all_time__author_id",
        "recent_h7__tag",
        "recent_h7__duration_bucket",
        "recent_h7__upload_type",
        "recent_h7__music_type",
        "recent_h7__tab",
        "recent_h7__onehot_feat1",
        "recent_h7__onehot_feat3",
        "recent_h7__onehot_feat7",
        "recent_h7__onehot_feat8",
    ],
}

for group_name, names in groups.items():
    valid_components[group_name] = average_components(
        names, valid_components
    )
    test_components[group_name] = average_components(
        names, test_components
    )

inc_metrics = evaluate(valid.user_id, y_valid, inc_valid)
best_metrics = inc_metrics
best_valid = inc_valid.copy()
best_test = inc_test.copy()
best_name = "incumbent"
best_alpha = 0.0

candidate_report = {
    "incumbent": float(inc_metrics["primary"])
}

inc_valid_z = standardize(inc_valid)
inc_test_z = standardize(inc_test)

# First test prespecified aggregates and each all-time field. Recency variants
# enter through aggregate hypotheses rather than a large per-field search.
search_names = list(groups.keys()) + [
    "all_time__" + field for field, _ in fields
]

alphas = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40]

for name in search_names:
    va_component = standardize(valid_components[name])
    te_component = standardize(test_components[name])

    local_best = -np.inf
    local_alpha = 0.0

    for alpha in alphas:
        va_score = inc_valid_z + float(alpha) * va_component
        metrics = evaluate(valid.user_id, y_valid, va_score)
        primary = float(metrics["primary"])

        if primary > local_best:
            local_best = primary
            local_alpha = float(alpha)

        if primary > float(best_metrics["primary"]):
            best_metrics = metrics
            best_valid = va_score.copy()
            best_test = inc_test_z + float(alpha) * te_component
            best_name = name
            best_alpha = float(alpha)

    candidate_report[name] = float(local_best)
    candidate_report[name + "_alpha"] = float(local_alpha)

# A negative residual weight is a useful falsification check: if it wins,
# the residual definition is directionally wrong rather than merely noisy.
for name in ["alltime_all", "h7_all", "mixed_entity_recent_content"]:
    va_component = standardize(valid_components[name])
    te_component = standardize(test_components[name])
    alpha = -0.10
    va_score = inc_valid_z + alpha * va_component
    metrics = evaluate(valid.user_id, y_valid, va_score)
    candidate_report[name + "_negative"] = float(metrics["primary"])
    if float(metrics["primary"]) > float(best_metrics["primary"]):
        best_metrics = metrics
        best_valid = va_score.copy()
        best_test = inc_test_z + alpha * te_component
        best_name = name + "_negative"
        best_alpha = alpha

print(
    "FINDINGS "
    + json.dumps(
        {
            "selected": best_name,
            "selected_alpha": best_alpha,
            "incumbent_primary": float(inc_metrics["primary"]),
            "selected_primary": float(best_metrics["primary"]),
            "selected_gain": float(
                best_metrics["primary"] - inc_metrics["primary"]
            ),
            "coverage": {
                name: {
                    "valid_known": round(info["valid_known"], 4),
                    "test_known": round(info["test_known"], 4),
                }
                for name, info in findings.items()
                if name.startswith("all_time__")
            },
        },
        sort_keys=True,
    )
)

print("CANDIDATES " + json.dumps(candidate_report, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(best_metrics["primary"]),
            "gauc": float(best_metrics["gauc"]),
            "ndcg@5": float(best_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)