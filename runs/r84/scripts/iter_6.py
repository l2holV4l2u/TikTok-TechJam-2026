import os
import time
import json
import gc
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
EPS = 1e-5
HALF_LIFE = 7.0

train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.float32)
y_valid = np.asarray(valid.y, dtype=np.int8)


def concat_splits(splits):
    out = {
        "user_id": np.concatenate(
            [np.asarray(s.user_id, dtype=np.int64) for s in splits]
        ),
        "time_ms": np.concatenate(
            [np.asarray(s.time_ms, dtype=np.int64) for s in splits]
        ),
        "date": np.concatenate(
            [np.asarray(s.date, dtype=np.int32) for s in splits]
        ),
    }
    needed = [
        "video_id", "author_id", "tag", "duration_bucket",
        "upload_type", "tab", "hour"
    ]
    for f in needed:
        out[f] = np.concatenate(
            [np.asarray(s.X[f], dtype=np.int64) for s in splits]
        )
    return out


def split_arrays(split):
    out = {
        "user_id": np.asarray(split.user_id, dtype=np.int64),
        "time_ms": np.asarray(split.time_ms, dtype=np.int64),
        "date": np.asarray(split.date, dtype=np.int32),
    }
    for f in [
        "video_id", "author_id", "tag", "duration_bucket",
        "upload_type", "tab", "hour"
    ]:
        out[f] = np.asarray(split.X[f], dtype=np.int64)
    return out


def temporal_weights(dates):
    dates = np.asarray(dates, dtype=np.int32)
    unique = np.unique(dates)
    idx = np.searchsorted(unique, dates)
    age = idx.max() - idx
    return np.power(0.5, age.astype(np.float32) / HALF_LIFE).astype(
        np.float32
    )


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def aggregate_lookup(fit_key, target_key, y, weights):
    fit_key = np.asarray(fit_key, dtype=np.int64)
    target_key = np.asarray(target_key, dtype=np.int64)
    order = np.argsort(fit_key, kind="mergesort")
    sk = fit_key[order]
    starts = np.r_[0, np.flatnonzero(sk[1:] != sk[:-1]) + 1]
    unique_key = sk[starts]
    count = np.add.reduceat(weights[order], starts).astype(np.float64)
    positive = np.add.reduceat(
        weights[order] * y[order], starts
    ).astype(np.float64)

    loc = np.searchsorted(unique_key, target_key)
    found = loc < len(unique_key)
    safe = np.minimum(loc, len(unique_key) - 1)
    found &= unique_key[safe] == target_key

    target_count = np.zeros(len(target_key), dtype=np.float64)
    target_positive = np.zeros(len(target_key), dtype=np.float64)
    target_count[found] = count[safe[found]]
    target_positive[found] = positive[safe[found]]
    return target_positive, target_count


def entity_posterior(fit_id, target_id, y, weights, cardinality,
                     smooth, prior):
    fit_id = np.asarray(fit_id, dtype=np.int64)
    target_id = np.asarray(target_id, dtype=np.int64)
    cnt = np.bincount(
        fit_id, weights=weights, minlength=cardinality
    ).astype(np.float64)
    pos = np.bincount(
        fit_id, weights=weights * y, minlength=cardinality
    ).astype(np.float64)
    target_cnt = cnt[target_id]
    target_pos = pos[target_id]
    return (target_pos + smooth * prior) / (target_cnt + smooth)


def pair_posterior(fit_left, fit_right, target_left, target_right,
                   right_card, y, weights, smooth, target_prior):
    fit_key = (
        np.asarray(fit_left, dtype=np.int64) * int(right_card)
        + np.asarray(fit_right, dtype=np.int64)
    )
    target_key = (
        np.asarray(target_left, dtype=np.int64) * int(right_card)
        + np.asarray(target_right, dtype=np.int64)
    )
    pos, cnt = aggregate_lookup(fit_key, target_key, y, weights)
    return (pos + smooth * target_prior) / (cnt + smooth)


def previous_indices(fit, target=None):
    if target is None:
        uid = fit["user_id"]
        tm = fit["time_ms"]
        nfit = len(uid)
        ntarget = 0
    else:
        uid = np.concatenate([fit["user_id"], target["user_id"]])
        tm = np.concatenate([fit["time_ms"], target["time_ms"]])
        nfit = len(fit["user_id"])
        ntarget = len(target["user_id"])

    row = np.arange(len(uid), dtype=np.int64)
    order = np.lexsort((row, tm, uid))
    prev = np.full(len(uid), -1, dtype=np.int64)
    if len(order) > 1:
        current = order[1:]
        preceding = order[:-1]
        same = uid[current] == uid[preceding]
        prev[current[same]] = preceding[same]

    if target is None:
        return prev

    return prev[nfit:nfit + ntarget]


def day_position_bucket(data):
    n = len(data["user_id"])
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort(
        (row, data["time_ms"], data["date"], data["user_id"])
    )
    uid_sorted = data["user_id"][order]
    date_sorted = data["date"][order]
    boundary = np.r_[
        True,
        (uid_sorted[1:] != uid_sorted[:-1])
        | (date_sorted[1:] != date_sorted[:-1])
    ]
    starts = np.maximum.accumulate(
        np.where(boundary, np.arange(n), 0)
    )
    rank_sorted = np.arange(n) - starts
    rank = np.empty(n, dtype=np.int64)
    rank[order] = rank_sorted
    return np.minimum(rank, 8)


def target_previous_values(fit, target, field):
    prev = previous_indices(fit, target)
    combined_values = np.concatenate([fit[field], target[field]])
    result = np.zeros(len(target["user_id"]), dtype=np.int64)
    found = prev >= 0
    result[found] = combined_values[prev[found]]
    return result


def fit_previous_values(fit, field):
    prev = previous_indices(fit)
    result = np.zeros(len(fit["user_id"]), dtype=np.int64)
    found = prev >= 0
    result[found] = fit[field][prev[found]]
    return result


def build_family_scores(fit, target, y):
    y = np.asarray(y, dtype=np.float32)
    weights = temporal_weights(fit["date"])
    global_rate = float(
        np.sum(weights * y) / np.maximum(np.sum(weights), 1.0)
    )

    video_rate = entity_posterior(
        fit["video_id"], target["video_id"], y, weights,
        int(FEATURE_CARDINALITIES["video_id"]), 24.0, global_rate
    )
    author_rate = entity_posterior(
        fit["author_id"], target["author_id"], y, weights,
        int(FEATURE_CARDINALITIES["author_id"]), 35.0, global_rate
    )
    tag_rate = entity_posterior(
        fit["tag"], target["tag"], y, weights,
        int(FEATURE_CARDINALITIES["tag"]), 100.0, global_rate
    )
    duration_rate = entity_posterior(
        fit["duration_bucket"], target["duration_bucket"], y, weights,
        int(FEATURE_CARDINALITIES["duration_bucket"]),
        180.0, global_rate
    )
    upload_rate = entity_posterior(
        fit["upload_type"], target["upload_type"], y, weights,
        int(FEATURE_CARDINALITIES["upload_type"]),
        180.0, global_rate
    )

    content_score = (
        1.00 * logit(video_rate)
        + 0.55 * logit(author_rate)
        + 0.30 * logit(tag_rate)
        + 0.28 * logit(duration_rate)
        + 0.12 * logit(upload_rate)
    )

    ua_rate = pair_posterior(
        fit["user_id"], fit["author_id"],
        target["user_id"], target["author_id"],
        int(FEATURE_CARDINALITIES["author_id"]),
        y, weights, 7.0, author_rate
    )
    ut_rate = pair_posterior(
        fit["user_id"], fit["tag"],
        target["user_id"], target["tag"],
        int(FEATURE_CARDINALITIES["tag"]),
        y, weights, 10.0, tag_rate
    )
    ud_rate = pair_posterior(
        fit["user_id"], fit["duration_bucket"],
        target["user_id"], target["duration_bucket"],
        int(FEATURE_CARDINALITIES["duration_bucket"]),
        y, weights, 10.0, duration_rate
    )
    uv_rate = pair_posterior(
        fit["user_id"], fit["video_id"],
        target["user_id"], target["video_id"],
        int(FEATURE_CARDINALITIES["video_id"]),
        y, weights, 5.0, video_rate
    )

    affinity_residual = (
        0.75 * (logit(ua_rate) - logit(author_rate))
        + 0.45 * (logit(ut_rate) - logit(tag_rate))
        + 0.35 * (logit(ud_rate) - logit(duration_rate))
        + 0.55 * (logit(uv_rate) - logit(video_rate))
    )
    affinity_score = content_score + affinity_residual

    fit_prev_tag = fit_previous_values(fit, "tag")
    target_prev_tag = target_previous_values(fit, target, "tag")
    fit_prev_tab = fit_previous_values(fit, "tab")
    target_prev_tab = target_previous_values(fit, target, "tab")

    tag_transition_prior = tag_rate
    tag_transition = pair_posterior(
        fit_prev_tag, fit["tag"],
        target_prev_tag, target["tag"],
        int(FEATURE_CARDINALITIES["tag"]),
        y, weights, 45.0, tag_transition_prior
    )
    tab_transition_prior = entity_posterior(
        fit["tab"], target["tab"], y, weights,
        int(FEATURE_CARDINALITIES["tab"]), 150.0, global_rate
    )
    tab_transition = pair_posterior(
        fit_prev_tab, fit["tab"],
        target_prev_tab, target["tab"],
        int(FEATURE_CARDINALITIES["tab"]),
        y, weights, 80.0, tab_transition_prior
    )

    fit_pos = day_position_bucket(fit)
    target_pos = day_position_bucket(target)
    pos_rate = entity_posterior(
        fit_pos, target_pos, y, weights, 9, 250.0, global_rate
    )

    sequence_score = (
        0.65 * (logit(tag_transition) - logit(tag_transition_prior))
        + 0.35 * (logit(tab_transition) - logit(tab_transition_prior))
        + 0.25 * logit(pos_rate)
    )
    hybrid_score = affinity_score + sequence_score

    return {
        "hierarchical_content": content_score.astype(np.float64),
        "sparse_user_affinity": affinity_score.astype(np.float64),
        "markov_sequence": sequence_score.astype(np.float64),
        "affinity_sequence_hybrid": hybrid_score.astype(np.float64),
    }


fit_train = split_arrays(train)
target_valid = split_arrays(valid)
valid_families = build_family_scores(fit_train, target_valid, y_train)

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
inc_scale = max(float(np.std(inc_valid)), 1e-8)

candidate_log = {}
best_primary = -np.inf
best_family = None
best_alpha = None
best_score = None
best_raw = None
best_metrics = None
best_own_scale = None

blend_weights = [0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80]

for family, raw in valid_families.items():
    raw = np.asarray(raw, dtype=np.float64)
    own_scale = max(float(np.std(raw)), 1e-8)

    own_metrics = evaluate(valid.user_id, y_valid, raw)
    candidate_log[family + "_standalone"] = float(
        own_metrics["primary"]
    )
    if float(own_metrics["primary"]) > best_primary:
        best_primary = float(own_metrics["primary"])
        best_family = family
        best_alpha = 1.0
        best_score = raw.copy()
        best_raw = raw.copy()
        best_metrics = own_metrics
        best_own_scale = own_scale

    local_best = None
    for alpha in blend_weights:
        blended = (
            alpha * raw / own_scale
            + (1.0 - alpha) * inc_valid / inc_scale
        )
        met = evaluate(valid.user_id, y_valid, blended)
        primary = float(met["primary"])
        if local_best is None or primary > local_best[0]:
            local_best = (primary, alpha, blended.copy(), met)

    primary, alpha, blended, met = local_best
    candidate_log[family + "_incumbent_blend"] = primary
    if primary > best_primary:
        best_primary = primary
        best_family = family
        best_alpha = float(alpha)
        best_score = blended
        best_raw = raw.copy()
        best_metrics = met
        best_own_scale = own_scale

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print(
    "FINDINGS selected=%s own_weight=%.2f own_std=%.6f"
    % (best_family, best_alpha, best_own_scale)
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_score, dtype=np.float64),
    )
    if best_alpha < 0.999999:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(best_raw, dtype=np.float64),
        )

# Refit the selected recipe on train + validation, then score test.
test = load("test")
fit_combined = concat_splits([train, valid])
target_test = split_arrays(test)
y_combined = np.concatenate(
    [y_train, y_valid.astype(np.float32)]
)

del valid_families
gc.collect()

test_families = build_family_scores(
    fit_combined, target_test, y_combined
)
raw_test = np.asarray(
    test_families[best_family], dtype=np.float64
)

if best_alpha < 0.999999:
    inc_test = np.load(inc_test_path).astype(np.float64)
    test_score = (
        best_alpha * raw_test / best_own_scale
        + (1.0 - best_alpha) * inc_test / inc_scale
    )
else:
    test_score = raw_test

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_score, dtype=np.float64),
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