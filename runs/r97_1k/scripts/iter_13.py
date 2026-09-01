import os
import gc
import json
import time
import warnings
import numpy as np

from pipeline.data import load
from pipeline.evaluate import evaluate

warnings.filterwarnings("ignore")

START = time.time()
OUT = os.environ.get("ITER_OUT")
SHARED = os.environ.get("SHARED_ARTIFACTS")

if OUT:
    os.makedirs(OUT, exist_ok=True)


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float32), 1e-4, 1.0 - 1e-4)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    su = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = su[1:] != su[:-1]

    start_pos = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local = np.arange(n, dtype=np.float32) - start_pos.astype(np.float32)

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = su[:-1] != su[1:]
    end_pos = np.flatnonzero(ends)
    sizes = np.diff(np.r_[-1, end_pos]).astype(np.float32)

    group = np.cumsum(starts, dtype=np.int64) - 1
    denom = np.maximum(sizes[group] - 1.0, 1.0)

    result = np.empty(n, dtype=np.float32)
    result[order] = local / denom
    return result


def sequence_base_features(split):
    uid = np.asarray(split.user_id, dtype=np.int64)
    tm = np.asarray(split.time_ms, dtype=np.int64)
    n = len(uid)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, tm, uid))
    su = uid[order]
    st = tm[order]

    new_user = np.empty(n, dtype=bool)
    new_user[0] = True
    new_user[1:] = su[1:] != su[:-1]

    prev_gap_ms = np.zeros(n, dtype=np.int64)
    gap_sorted = np.zeros(n, dtype=np.int64)
    valid_prev = ~new_user
    gap_sorted[valid_prev] = np.maximum(
        st[valid_prev] - st[np.flatnonzero(valid_prev) - 1], 0
    )
    prev_gap_ms[order] = gap_sorted

    session_start = new_user.copy()
    session_start |= gap_sorted > 30 * 60 * 1000

    session_start_index = np.maximum.accumulate(
        np.where(session_start, np.arange(n, dtype=np.int64), 0)
    )
    session_pos_sorted = np.arange(n, dtype=np.int64) - session_start_index

    user_start_index = np.maximum.accumulate(
        np.where(new_user, np.arange(n, dtype=np.int64), 0)
    )
    user_pos_sorted = np.arange(n, dtype=np.int64) - user_start_index

    session_pos = np.empty(n, dtype=np.int32)
    user_pos = np.empty(n, dtype=np.int32)
    session_pos[order] = session_pos_sorted.astype(np.int32)
    user_pos[order] = user_pos_sorted.astype(np.int32)

    previous_tag = np.full(n, -1, dtype=np.int32)
    previous_tab = np.full(n, -1, dtype=np.int32)
    previous_duration = np.full(n, -1, dtype=np.int32)

    for name, destination in (
        ("tag", previous_tag),
        ("tab", previous_tab),
        ("duration_bucket", previous_duration),
    ):
        values = np.asarray(split.X[name], dtype=np.int32)
        sv = values[order]
        pv = np.full(n, -1, dtype=np.int32)
        pv[1:] = sv[:-1]
        pv[new_user] = -1
        destination[order] = pv

    return {
        "gap_ms": prev_gap_ms,
        "session_pos": session_pos,
        "user_pos": user_pos,
        "previous_tag": previous_tag,
        "previous_tab": previous_tab,
        "previous_duration": previous_duration,
    }


def repeated_entity_features(split, field):
    uid = np.asarray(split.user_id, dtype=np.int64)
    tm = np.asarray(split.time_ms, dtype=np.int64)
    entity = np.asarray(split.X[field], dtype=np.int64)
    n = len(uid)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, tm, entity, uid))
    su = uid[order]
    se = entity[order]
    st = tm[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = (su[1:] != su[:-1]) | (se[1:] != se[:-1])

    start_index = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    count_sorted = np.arange(n, dtype=np.int64) - start_index

    gap_sorted = np.zeros(n, dtype=np.int64)
    has_previous = ~starts
    previous_indices = np.flatnonzero(has_previous) - 1
    gap_sorted[has_previous] = np.maximum(
        st[has_previous] - st[previous_indices], 0
    )

    # Unknown entity zero is not a meaningful repeated item/creator.
    unknown = se == 0
    count_sorted[unknown] = 0
    gap_sorted[unknown] = 0

    count = np.empty(n, dtype=np.int32)
    gap = np.empty(n, dtype=np.int64)
    count[order] = count_sorted.astype(np.int32)
    gap[order] = gap_sorted

    return count, gap


def log_bucket(x, cap):
    x = np.asarray(x)
    return np.minimum(
        np.floor(np.log2(np.maximum(x, 0) + 1.0)).astype(np.int32),
        int(cap),
    )


def gap_bucket(gap_ms):
    gap_minutes = np.asarray(gap_ms, dtype=np.float64) / 60000.0
    # 0 means no preceding exposure; remaining bins cover seconds to days.
    edges = np.asarray(
        [0.001, 0.25, 1.0, 5.0, 30.0, 120.0, 720.0, 2880.0],
        dtype=np.float64,
    )
    return np.searchsorted(edges, gap_minutes, side="right").astype(np.int32)


def construct_codes(split):
    base = sequence_base_features(split)

    video_count, video_gap = repeated_entity_features(split, "video_id")
    author_count, author_gap = repeated_entity_features(split, "author_id")

    tag = np.asarray(split.X["tag"], dtype=np.int32)
    tab = np.asarray(split.X["tab"], dtype=np.int32)
    duration = np.asarray(split.X["duration_bucket"], dtype=np.int32)
    hour = np.asarray(split.X["hour"], dtype=np.int32)

    prev_tag = base["previous_tag"]
    prev_tab = base["previous_tab"]
    prev_duration = base["previous_duration"]

    tag_card = int(max(np.max(tag), np.max(prev_tag), 0)) + 2
    tab_card = int(max(np.max(tab), np.max(prev_tab), 0)) + 2
    dur_card = int(max(np.max(duration), np.max(prev_duration), 0)) + 2

    tag_transition = (prev_tag + 1) * tag_card + (tag + 1)
    tab_transition = (prev_tab + 1) * tab_card + (tab + 1)
    duration_transition = (
        (prev_duration + 1) * dur_card + (duration + 1)
    )

    # Crosses distinguish rapid repeated exposure from repetitions separated
    # by hours or days.
    video_count_bucket = np.minimum(video_count, 7).astype(np.int32)
    author_count_bucket = np.minimum(author_count, 15).astype(np.int32)
    video_gap_bucket = gap_bucket(video_gap)
    author_gap_bucket = gap_bucket(author_gap)

    video_repeat_cross = (
        video_count_bucket * 10 + video_gap_bucket
    ).astype(np.int32)
    author_repeat_cross = (
        author_count_bucket * 10 + author_gap_bucket
    ).astype(np.int32)

    codes = {
        "hour": np.maximum(hour, 0).astype(np.int32),
        "user_position": log_bucket(base["user_pos"], 13),
        "session_position": log_bucket(base["session_pos"], 10),
        "previous_gap": gap_bucket(base["gap_ms"]),
        "video_repeat": video_repeat_cross,
        "author_repeat": author_repeat_cross,
        "video_count": video_count_bucket,
        "author_count": author_count_bucket,
        "tag_transition": tag_transition.astype(np.int32),
        "tab_transition": tab_transition.astype(np.int32),
        "duration_transition": duration_transition.astype(np.int32),
    }

    del base, video_count, video_gap, author_count, author_gap
    gc.collect()
    return codes


def fit_rate_table(code, y, alpha, weights, prior):
    code = np.asarray(code, dtype=np.int64)
    size = int(np.max(code)) + 1
    sw = np.bincount(
        code, weights=weights, minlength=size
    ).astype(np.float32)
    sy = np.bincount(
        code, weights=weights * y, minlength=size
    ).astype(np.float32)
    rate = (sy + float(alpha) * prior) / np.maximum(
        sw + float(alpha), 1e-6
    )
    return safe_logit(rate), float(safe_logit(np.asarray([prior]))[0])


def apply_table(code, table_info):
    table, fallback = table_info
    code = np.asarray(code, dtype=np.int64)
    result = np.full(len(code), fallback, dtype=np.float32)
    ok = (code >= 0) & (code < len(table))
    result[ok] = table[code[ok]]
    return result


train = load("train")
train_y = np.asarray(train.y, dtype=np.float32)
train_date = np.asarray(train.date, dtype=np.int32)

max_date = int(np.max(train_date))
age = (max_date - train_date).astype(np.float32)
train_weight = np.power(0.5, age / 5.0).astype(np.float32)
train_weight /= np.mean(train_weight)

prior = float(
    np.sum(train_weight * train_y) / np.maximum(np.sum(train_weight), 1e-6)
)

print(
    "FINDINGS temporal_prior=%.6f weight_q01_q50_q99=%.4f,%.4f,%.4f"
    % (
        prior,
        float(np.quantile(train_weight, 0.01)),
        float(np.quantile(train_weight, 0.50)),
        float(np.quantile(train_weight, 0.99)),
    ),
    flush=True,
)

train_codes = construct_codes(train)

TABLE_SPECS = {
    "hour": 500.0,
    "user_position": 800.0,
    "session_position": 500.0,
    "previous_gap": 500.0,
    "video_repeat": 250.0,
    "author_repeat": 300.0,
    "video_count": 500.0,
    "author_count": 500.0,
    "tag_transition": 180.0,
    "tab_transition": 250.0,
    "duration_transition": 250.0,
}

tables = {}
for name, alpha in TABLE_SPECS.items():
    tables[name] = fit_rate_table(
        train_codes[name],
        train_y,
        alpha,
        train_weight,
        prior,
    )

del train_codes, train_y, train_weight, train_date, train
gc.collect()


FAMILY_FEATURES = {
    # Position and elapsed-time hazard, irrespective of item identity.
    "session_hazard": [
        ("hour", 0.35),
        ("user_position", 0.65),
        ("session_position", 1.00),
        ("previous_gap", 1.00),
    ],
    # Repeated-item and repeated-creator fatigue/affinity.
    "exposure_fatigue": [
        ("video_repeat", 1.00),
        ("author_repeat", 1.00),
        ("video_count", 0.35),
        ("author_count", 0.35),
    ],
    # A first-order Markov model over coarse feed content.
    "content_markov": [
        ("tag_transition", 1.00),
        ("tab_transition", 0.90),
        ("duration_transition", 0.75),
    ],
    # A structurally mixed sequential hazard model.
    "sequence_mixture": [
        ("hour", 0.15),
        ("session_position", 0.40),
        ("previous_gap", 0.55),
        ("video_repeat", 0.80),
        ("author_repeat", 1.00),
        ("tag_transition", 0.65),
        ("tab_transition", 0.50),
        ("duration_transition", 0.35),
    ],
}


def score_families(split):
    codes = construct_codes(split)
    outputs = {}

    prior_logit = float(safe_logit(np.asarray([prior]))[0])

    for family, features in FAMILY_FEATURES.items():
        score = np.zeros(len(split.user_id), dtype=np.float32)
        scale = 0.0
        for name, weight in features:
            contribution = apply_table(codes[name], tables[name])
            score += float(weight) * (contribution - prior_logit)
            scale += abs(float(weight))
        outputs[family] = (score / max(scale, 1e-6)).astype(np.float32)

    del codes
    gc.collect()
    return outputs


valid = load("valid")
valid_uid = np.asarray(valid.user_id, dtype=np.int64)
valid_y = np.asarray(valid.y)
valid_family_scores = score_families(valid)

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

inc_valid = np.load(inc_valid_path, mmap_mode="r")
inc_valid_rank = within_user_rank(valid_uid, inc_valid)
inc_metrics = evaluate(valid_uid, valid_y, inc_valid_rank)

candidate_scores = {
    "trusted_incumbent": float(inc_metrics["primary"])
}

best_name = "trusted_incumbent"
best_family = None
best_alpha = 0.0
best_valid = inc_valid_rank.copy()
best_raw_valid = None
best_primary = float(inc_metrics["primary"])

ALPHAS = (
    -0.20, -0.12, -0.08, -0.05, -0.03,
     0.02,  0.03,  0.05,  0.08,  0.12,
     0.18,  0.25,  0.35,
)

for family, raw_score in valid_family_scores.items():
    raw_metrics = evaluate(valid_uid, valid_y, raw_score)
    candidate_scores[family + "_standalone"] = float(
        raw_metrics["primary"]
    )

    correction_rank = within_user_rank(valid_uid, raw_score)
    local_best = -np.inf
    local_alpha = 0.0

    for alpha in ALPHAS:
        blended = (
            inc_valid_rank + float(alpha) * (correction_rank - 0.5)
        ).astype(np.float32)
        met = evaluate(valid_uid, valid_y, blended)
        primary = float(met["primary"])

        if primary > local_best:
            local_best = primary
            local_alpha = float(alpha)

        if primary > best_primary:
            best_primary = primary
            best_name = "%s_adjustment_%+.2f" % (family, alpha)
            best_family = family
            best_alpha = float(alpha)
            best_valid = blended.copy()
            best_raw_valid = raw_score.copy()

    candidate_scores[family + "_best_adjusted"] = float(local_best)
    print(
        "FINDINGS family=%s standalone=%.6f best_alpha=%+.2f "
        "best_adjusted=%.6f rank_corr_incumbent=%.6f"
        % (
            family,
            float(raw_metrics["primary"]),
            local_alpha,
            local_best,
            float(np.corrcoef(inc_valid_rank, correction_rank)[0, 1]),
        ),
        flush=True,
    )

final_metrics = evaluate(valid_uid, valid_y, best_valid)

print(
    "FINDINGS winner=%s alpha=%+.2f"
    % (best_name, best_alpha),
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
    if best_family is not None:
        np.save(
            os.path.join(OUT, "scores_valid_raw.npy"),
            np.asarray(best_raw_valid, dtype=np.float64),
        )

del valid_family_scores, best_valid, best_raw_valid
del inc_valid, inc_valid_rank
gc.collect()

test = load("test")
inc_test = np.load(inc_test_path, mmap_mode="r")
inc_test_rank = within_user_rank(test.user_id, inc_test)

if best_family is None:
    test_scores = inc_test_rank
else:
    test_family_scores = score_families(test)
    raw_test = test_family_scores[best_family]
    correction_test_rank = within_user_rank(test.user_id, raw_test)
    test_scores = (
        inc_test_rank
        + best_alpha * (correction_test_rank - 0.5)
    ).astype(np.float32)
    del test_family_scores, raw_test, correction_test_rank

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