import os
import gc
import json
import time
import warnings
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")

START = time.time()
OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")
if OUT:
    os.makedirs(OUT, exist_ok=True)

FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
]

EXPOSURE_FIELDS = [
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
]

FIELD_WEIGHT = {
    "video_id": 0.70,
    "author_id": 1.00,
    "tag": 0.65,
    "tab": 0.90,
    "duration_bucket": 0.35,
    "upload_type": 0.45,
    "onehot_feat3": 0.75,
    "onehot_feat8": 0.65,
}

MARGINAL_ALPHA = {
    "video_id": 35.0,
    "author_id": 60.0,
    "tag": 180.0,
    "tab": 250.0,
    "duration_bucket": 250.0,
    "upload_type": 250.0,
    "onehot_feat3": 120.0,
    "onehot_feat8": 120.0,
}

PAIR_ALPHA = {
    "video_id": 8.0,
    "author_id": 16.0,
    "tag": 35.0,
    "tab": 50.0,
    "duration_bucket": 55.0,
    "upload_type": 45.0,
    "onehot_feat3": 28.0,
    "onehot_feat8": 28.0,
}

SCHEMES = ("hl2", "hl4", "hl8", "ips4")


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float32), 1e-4, 1.0 - 1e-4)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    group_start = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local = np.arange(n, dtype=np.float32) - group_start.astype(np.float32)

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(np.r_[-1, end_positions]).astype(np.float32)

    group_index = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[group_index] - 1.0, 1.0)

    ranked = np.empty(n, dtype=np.float32)
    ranked[order] = local / denom
    return ranked


train = load("train")
train_y = np.asarray(train.y, dtype=np.float32)
train_uid = np.asarray(train.user_id, dtype=np.int64)
train_date = np.asarray(train.date, dtype=np.int32)
n_train = len(train_uid)

max_date = int(np.max(train_date))
age = (max_date - train_date).astype(np.float32)

w_hl2 = np.power(0.5, age / 2.0).astype(np.float32)
w_hl4 = np.power(0.5, age / 4.0).astype(np.float32)
w_hl8 = np.power(0.5, age / 8.0).astype(np.float32)

# Estimate how frequently the logger exposes each coarse content value to
# each user. Rare user-content exposures receive moderate inverse-propensity
# weight; clipping prevents tiny cells from dominating.
user_card = int(FEATURE_CARDINALITIES["user_id"])
user_totals = np.bincount(
    train_uid, minlength=user_card
).astype(np.float32)
user_totals = np.maximum(user_totals, 1.0)

neg_log_propensity = np.zeros(n_train, dtype=np.float32)

for field in EXPOSURE_FIELDS:
    ids = np.asarray(train.X[field], dtype=np.int64)
    card = int(FEATURE_CARDINALITIES[field])
    keys = train_uid * card + ids
    counts = np.bincount(
        keys, minlength=user_card * card
    ).astype(np.float32)
    p = counts[keys] / user_totals[train_uid]
    neg_log_propensity += -np.log(np.maximum(p, 1e-5)).astype(np.float32)
    del keys, counts, ids
    gc.collect()

neg_log_propensity /= float(len(EXPOSURE_FIELDS))
ips = np.exp(0.35 * neg_log_propensity).astype(np.float32)
lo, hi = np.quantile(ips, [0.02, 0.98])
ips = np.clip(ips, lo, hi)
ips /= np.mean(ips)

w_ips4 = (w_hl4 * ips).astype(np.float32)

weights = {
    "hl2": w_hl2,
    "hl4": w_hl4,
    "hl8": w_hl8,
    "ips4": w_ips4,
}

for key in weights:
    weights[key] = (
        weights[key] / np.mean(weights[key])
    ).astype(np.float32)

priors = {
    key: float(np.sum(w * train_y) / np.sum(w))
    for key, w in weights.items()
}

print(
    "FINDINGS ips_weight_q01_q50_q99=%.4f,%.4f,%.4f priors=%s"
    % (
        float(np.quantile(ips, 0.01)),
        float(np.quantile(ips, 0.50)),
        float(np.quantile(ips, 0.99)),
        json.dumps(priors, sort_keys=True),
    ),
    flush=True,
)

# maps[field] contains weighted marginal entity statistics and sparse
# user-entity statistics for all temporal/counterfactual schemes.
maps = {}

for field in FIELDS:
    ids = np.asarray(train.X[field], dtype=np.int64)
    card = int(FEATURE_CARDINALITIES[field])

    marginal = {}
    for scheme, w in weights.items():
        marginal[scheme] = (
            np.bincount(ids, weights=w, minlength=card).astype(np.float32),
            np.bincount(
                ids, weights=w * train_y, minlength=card
            ).astype(np.float32),
        )

    pair_keys = train_uid * np.int64(card) + ids
    order = np.argsort(pair_keys, kind="mergesort")
    sorted_keys = pair_keys[order]

    starts = np.empty(n_train, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_keys[1:] != sorted_keys[:-1]
    start_index = np.flatnonzero(starts)
    unique_keys = sorted_keys[start_index].copy()

    pair_stats = {}
    y_sorted = train_y[order]

    for scheme, w in weights.items():
        ws = w[order]
        pair_stats[scheme] = (
            np.add.reduceat(ws, start_index).astype(np.float32),
            np.add.reduceat(ws * y_sorted, start_index).astype(np.float32),
        )

    maps[field] = {
        "card": card,
        "marginal": marginal,
        "keys": unique_keys,
        "pair": pair_stats,
    }

    print(
        "FINDINGS field=%s unique_user_entity_pairs=%d"
        % (field, len(unique_keys)),
        flush=True,
    )

    del ids, pair_keys, order, sorted_keys, starts
    del start_index, unique_keys, y_sorted
    gc.collect()


def marginal_rate(field, ids, scheme):
    info = maps[field]
    sw, sy = info["marginal"][scheme]
    alpha = float(MARGINAL_ALPHA[field])
    prior = priors[scheme]

    result = np.full(len(ids), prior, dtype=np.float32)
    ok = (ids >= 0) & (ids < len(sw))
    selected = ids[ok]
    result[ok] = (
        sy[selected] + alpha * prior
    ) / np.maximum(sw[selected] + alpha, 1e-6)
    return result


def pair_rate(split, field, scheme, base_rate):
    info = maps[field]
    ids = np.asarray(split.X[field], dtype=np.int64)
    query = np.asarray(split.user_id, dtype=np.int64) * np.int64(
        info["card"]
    ) + ids

    keys = info["keys"]
    sw, sy = info["pair"][scheme]
    positions = np.searchsorted(keys, query)

    found = positions < len(keys)
    clipped = np.minimum(positions, max(len(keys) - 1, 0))
    if len(keys):
        found &= keys[clipped] == query
    else:
        found[:] = False

    result = base_rate.copy()
    if np.any(found):
        pos = positions[found]
        alpha = float(PAIR_ALPHA[field])
        result[found] = (
            sy[pos] + alpha * base_rate[found]
        ) / np.maximum(sw[pos] + alpha, 1e-6)
    return result


def score_split(split, family):
    n = len(split.user_id)

    if family == "marginal_hl4":
        total = np.zeros(n, dtype=np.float32)
        scale = 0.0
        for field in FIELDS:
            ids = np.asarray(split.X[field], dtype=np.int64)
            rate = marginal_rate(field, ids, "hl4")
            fw = float(FIELD_WEIGHT[field])
            total += fw * safe_logit(rate)
            scale += fw
        return total / scale

    if family in ("personal_hl4", "personal_ips4"):
        scheme = "hl4" if family == "personal_hl4" else "ips4"
        total = np.zeros(n, dtype=np.float32)
        scale = 0.0
        for field in FIELDS:
            ids = np.asarray(split.X[field], dtype=np.int64)
            base = marginal_rate(field, ids, scheme)
            personalized = pair_rate(split, field, scheme, base)
            fw = float(FIELD_WEIGHT[field])
            total += fw * safe_logit(personalized)
            scale += fw
        return total / scale

    if family == "temporal_trend":
        recent = np.zeros(n, dtype=np.float32)
        stable = np.zeros(n, dtype=np.float32)
        scale = 0.0

        for field in FIELDS:
            ids = np.asarray(split.X[field], dtype=np.int64)

            base2 = marginal_rate(field, ids, "hl2")
            pair2 = pair_rate(split, field, "hl2", base2)

            base8 = marginal_rate(field, ids, "hl8")
            pair8 = pair_rate(split, field, "hl8", base8)

            fw = float(FIELD_WEIGHT[field])
            recent += fw * safe_logit(pair2)
            stable += fw * safe_logit(pair8)
            scale += fw

        recent /= scale
        stable /= scale
        # Conservative linear extrapolation of movement from the long-memory
        # estimate toward the recent estimate.
        return (recent + 0.35 * (recent - stable)).astype(np.float32)

    raise ValueError("unknown family: " + family)


valid = load("valid")
valid_uid = np.asarray(valid.user_id)
valid_y = np.asarray(valid.y)

FAMILIES = [
    "marginal_hl4",
    "personal_hl4",
    "personal_ips4",
    "temporal_trend",
]

valid_predictions = {}
candidate_scores = {}

for family in FAMILIES:
    pred = score_split(valid, family)
    valid_predictions[family] = pred
    met = evaluate(valid_uid, valid_y, pred)
    candidate_scores[family] = float(met["primary"])

inc_valid_path = (
    os.path.join(SHARED, "incumbent_valid_scores.npy")
    if SHARED else ""
)
inc_test_path = (
    os.path.join(SHARED, "incumbent_test_scores.npy")
    if SHARED else ""
)
has_incumbent = bool(
    inc_valid_path
    and inc_test_path
    and os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
)

best_name = max(candidate_scores, key=candidate_scores.get)
best_family = best_name
best_weight = 1.0
best_valid = valid_predictions[best_family].copy()
best_raw_valid = best_valid.copy()
best_primary = candidate_scores[best_name]

if has_incumbent:
    incumbent_valid = np.load(inc_valid_path, mmap_mode="r")
    incumbent_metrics = evaluate(valid_uid, valid_y, incumbent_valid)
    incumbent_primary = float(incumbent_metrics["primary"])
    candidate_scores["trusted_incumbent"] = incumbent_primary

    incumbent_rank = within_user_rank(valid_uid, incumbent_valid)

    if incumbent_primary > best_primary:
        best_primary = incumbent_primary
        best_name = "trusted_incumbent"
        best_family = "trusted_incumbent"
        best_weight = 0.0
        best_valid = np.asarray(incumbent_valid, dtype=np.float32).copy()
        strongest = max(
            FAMILIES, key=lambda name: candidate_scores[name]
        )
        best_raw_valid = valid_predictions[strongest].copy()

    for family in FAMILIES:
        own_rank = within_user_rank(
            valid_uid, valid_predictions[family]
        )
        local_best = -np.inf
        local_weight = 0.0

        for weight in (
            0.05, 0.10, 0.15, 0.20, 0.25,
            0.30, 0.40, 0.50, 0.65, 0.80,
        ):
            blended = (
                (1.0 - weight) * incumbent_rank
                + weight * own_rank
            ).astype(np.float32)
            met = evaluate(valid_uid, valid_y, blended)
            primary = float(met["primary"])

            if primary > local_best:
                local_best = primary
                local_weight = float(weight)

            if primary > best_primary:
                best_primary = primary
                best_name = "%s_rankblend_w%.2f" % (family, weight)
                best_family = family
                best_weight = float(weight)
                best_valid = blended.copy()
                best_raw_valid = valid_predictions[family].copy()

        candidate_scores[family + "_best_blend"] = local_best
        print(
            "FINDINGS family=%s best_blend_weight=%.2f "
            "best_blend_primary=%.6f"
            % (family, local_weight, local_best),
            flush=True,
        )

final_metrics = evaluate(valid_uid, valid_y, best_valid)

print(
    "FINDINGS winner=%s ips_vs_direct_rank_corr=%.6f"
    % (
        best_name,
        float(np.corrcoef(
            within_user_rank(
                valid_uid, valid_predictions["personal_hl4"]
            ),
            within_user_rank(
                valid_uid, valid_predictions["personal_ips4"]
            ),
        )[0, 1]),
    ),
    flush=True,
)

print(
    "CANDIDATES " + json.dumps(candidate_scores, sort_keys=True),
    flush=True,
)

if OUT:
    np.save(
        os.path.join(OUT, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    if best_weight < 1.0:
        np.save(
            os.path.join(OUT, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

del valid_predictions, best_valid, best_raw_valid
gc.collect()

test = load("test")

if best_family == "trusted_incumbent":
    test_scores = np.asarray(
        np.load(inc_test_path, mmap_mode="r"),
        dtype=np.float32,
    ).copy()
else:
    own_test = score_split(test, best_family)

    if best_weight < 1.0 and has_incumbent:
        incumbent_test = np.load(inc_test_path, mmap_mode="r")
        own_test_rank = within_user_rank(test.user_id, own_test)
        incumbent_test_rank = within_user_rank(
            test.user_id, incumbent_test
        )
        test_scores = (
            best_weight * own_test_rank
            + (1.0 - best_weight) * incumbent_test_rank
        ).astype(np.float32)
    else:
        test_scores = own_test

if OUT:
    np.save(
        os.path.join(OUT, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = float(time.time() - START)
print(
    "METRICS "
    + json.dumps({
        "primary": float(final_metrics["primary"]),
        "gauc": float(final_metrics["gauc"]),
        "ndcg@5": float(final_metrics["ndcg@5"]),
        "gpu_seconds": elapsed,
    })
)