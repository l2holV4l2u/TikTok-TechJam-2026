import os
import time
import json
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
SMOOTH_ENTITY = 20.0
SMOOTH_PAIR = 6.0
K_RECENT = 4


def day_number(date):
    date = np.asarray(date, dtype=np.int32)
    month = (date // 100) % 100
    day = date % 100
    return day + np.where(month >= 5, 30, 0)


def temporal_weights(date, endpoint, half_life):
    if half_life is None:
        return np.ones(len(date), dtype=np.float64)
    age = np.maximum(float(endpoint) - day_number(date), 0.0)
    return np.exp2(-age / float(half_life))


def weighted_prior(y, w):
    return float(np.sum(w * y) / np.maximum(np.sum(w), 1e-12))


def entity_rate(source_ids, y, w, query_ids, cardinality, smooth, prior):
    source_ids = np.asarray(source_ids, dtype=np.int64)
    query_ids = np.asarray(query_ids, dtype=np.int64)
    count = np.bincount(
        source_ids, weights=w, minlength=cardinality
    ).astype(np.float64, copy=False)
    positive = np.bincount(
        source_ids, weights=w * y, minlength=cardinality
    ).astype(np.float64, copy=False)
    rate = (positive + smooth * prior) / (count + smooth)
    return rate[query_ids].astype(np.float32)


def sparse_pair_rate(source_user, source_value, y, w,
                     query_user, query_value, value_cardinality,
                     smooth, prior):
    source_user = np.asarray(source_user, dtype=np.int64)
    source_value = np.asarray(source_value, dtype=np.int64)
    query_user = np.asarray(query_user, dtype=np.int64)
    query_value = np.asarray(query_value, dtype=np.int64)

    source_key = source_user * np.int64(value_cardinality) + source_value
    order = np.argsort(source_key, kind="stable")
    sorted_key = source_key[order]
    unique_key, starts = np.unique(sorted_key, return_index=True)

    sorted_w = w[order]
    sorted_pos = sorted_w * y[order]
    counts = np.add.reduceat(sorted_w, starts)
    positives = np.add.reduceat(sorted_pos, starts)
    rates = (positives + smooth * prior) / (counts + smooth)

    query_key = query_user * np.int64(value_cardinality) + query_value
    loc = np.searchsorted(unique_key, query_key)
    safe = np.minimum(loc, len(unique_key) - 1)
    found = (loc < len(unique_key)) & (unique_key[safe] == query_key)

    result = np.full(len(query_key), prior, dtype=np.float32)
    result[found] = rates[safe[found]].astype(np.float32)
    return result


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def recent_positive_score(source, y, query):
    users = np.asarray(source.user_id, dtype=np.int64)
    times = np.asarray(source.time_ms, dtype=np.int64)
    positive_rows = np.flatnonzero(y == 1)

    if len(positive_rows) == 0:
        return np.zeros(len(query.user_id), dtype=np.float32)

    order = np.lexsort((
        positive_rows,
        times[positive_rows],
        users[positive_rows],
    ))
    rows = positive_rows[order]
    sorted_users = users[rows]

    new_group = np.empty(len(rows), dtype=bool)
    new_group[0] = True
    new_group[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.maximum.accumulate(
        np.where(new_group, np.arange(len(rows), dtype=np.int64), 0)
    )
    position = np.arange(len(rows), dtype=np.int64) - starts
    group_counts = np.bincount(sorted_users)
    reverse_position = group_counts[sorted_users] - 1 - position
    keep = reverse_position < K_RECENT

    rows = rows[keep]
    slot = reverse_position[keep]
    kept_users = users[rows]

    max_user = int(max(
        np.max(source.user_id),
        np.max(query.user_id),
    )) + 1

    recent_video = np.full((max_user, K_RECENT), -1, dtype=np.int32)
    recent_author = np.full((max_user, K_RECENT), -1, dtype=np.int32)
    recent_tag = np.full((max_user, K_RECENT), -1, dtype=np.int16)
    recent_duration = np.full((max_user, K_RECENT), -1, dtype=np.int16)

    recent_video[kept_users, slot] = np.asarray(
        source.video_id, dtype=np.int32
    )[rows]
    recent_author[kept_users, slot] = np.asarray(
        source.X["author_id"], dtype=np.int32
    )[rows]
    recent_tag[kept_users, slot] = np.asarray(
        source.X["tag"], dtype=np.int16
    )[rows]
    recent_duration[kept_users, slot] = np.asarray(
        source.X["duration_bucket"], dtype=np.int16
    )[rows]

    qu = np.asarray(query.user_id, dtype=np.int64)
    qvideo = np.asarray(query.video_id, dtype=np.int32)
    qauthor = np.asarray(query.X["author_id"], dtype=np.int32)
    qtag = np.asarray(query.X["tag"], dtype=np.int16)
    qduration = np.asarray(query.X["duration_bucket"], dtype=np.int16)

    score = np.zeros(len(qu), dtype=np.float32)
    decay = np.asarray([1.0, 0.65, 0.42, 0.27], dtype=np.float32)

    for k in range(K_RECENT):
        valid_slot = recent_video[qu, k] >= 0
        score += decay[k] * (
            2.00 * (recent_video[qu, k] == qvideo)
            + 1.20 * (recent_author[qu, k] == qauthor)
            + 0.55 * (recent_tag[qu, k] == qtag)
            + 0.20 * (recent_duration[qu, k] == qduration)
        ).astype(np.float32) * valid_slot.astype(np.float32)

    return score


def build_scores(source, query, half_life):
    y = np.asarray(source.y, dtype=np.float64)
    endpoint = int(np.max(day_number(source.date)))
    w = temporal_weights(source.date, endpoint, half_life)
    prior = weighted_prior(y, w)

    video_rate = entity_rate(
        source.video_id, y, w, query.video_id,
        int(FEATURE_CARDINALITIES["video_id"]),
        SMOOTH_ENTITY, prior,
    )
    author_rate = entity_rate(
        source.X["author_id"], y, w, query.X["author_id"],
        int(FEATURE_CARDINALITIES["author_id"]),
        SMOOTH_ENTITY, prior,
    )
    tag_rate = entity_rate(
        source.X["tag"], y, w, query.X["tag"],
        int(FEATURE_CARDINALITIES["tag"]),
        SMOOTH_ENTITY, prior,
    )
    duration_rate = entity_rate(
        source.X["duration_bucket"], y, w,
        query.X["duration_bucket"],
        int(FEATURE_CARDINALITIES["duration_bucket"]),
        SMOOTH_ENTITY, prior,
    )

    user_tag = sparse_pair_rate(
        source.user_id, source.X["tag"], y, w,
        query.user_id, query.X["tag"],
        int(FEATURE_CARDINALITIES["tag"]),
        SMOOTH_PAIR, prior,
    )
    user_author = sparse_pair_rate(
        source.user_id, source.X["author_id"], y, w,
        query.user_id, query.X["author_id"],
        int(FEATURE_CARDINALITIES["author_id"]),
        SMOOTH_PAIR, prior,
    )
    user_duration = sparse_pair_rate(
        source.user_id, source.X["duration_bucket"], y, w,
        query.user_id, query.X["duration_bucket"],
        int(FEATURE_CARDINALITIES["duration_bucket"]),
        SMOOTH_PAIR, prior,
    )

    entity = (
        0.42 * logit(video_rate)
        + 0.32 * logit(author_rate)
        + 0.18 * logit(tag_rate)
        + 0.08 * logit(duration_rate)
    ).astype(np.float32)

    personalized = (
        entity
        + 0.55 * (logit(user_tag) - logit(tag_rate))
        + 0.55 * (logit(user_author) - logit(author_rate))
        + 0.25 * (logit(user_duration) - logit(duration_rate))
    ).astype(np.float32)

    sequence = recent_positive_score(source, y, query)
    return {
        "entity": entity,
        "personalized": personalized,
        "sequence": sequence,
        "personalized_sequence": personalized + 0.55 * sequence,
    }


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    boundary = np.empty(n, dtype=bool)
    boundary[0] = True
    boundary[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.maximum.accumulate(
        np.where(boundary, np.arange(n, dtype=np.int64), 0)
    )
    position = np.arange(n, dtype=np.int64) - starts

    counts = np.bincount(sorted_users)
    denominator = np.maximum(counts[sorted_users] - 1, 1)
    ranked = np.empty(n, dtype=np.float32)
    ranked[order] = (position / denominator).astype(np.float32)
    return ranked


class CombinedSplit:
    pass


def combine_splits(a, b):
    c = CombinedSplit()
    c.user_id = np.concatenate([
        np.asarray(a.user_id), np.asarray(b.user_id)
    ])
    c.video_id = np.concatenate([
        np.asarray(a.video_id), np.asarray(b.video_id)
    ])
    c.time_ms = np.concatenate([
        np.asarray(a.time_ms), np.asarray(b.time_ms)
    ])
    c.date = np.concatenate([
        np.asarray(a.date), np.asarray(b.date)
    ])
    c.y = np.concatenate([
        np.asarray(a.y), np.asarray(b.y)
    ])
    needed = ["author_id", "tag", "duration_bucket"]
    c.X = {
        name: np.concatenate([
            np.asarray(a.X[name]), np.asarray(b.X[name])
        ])
        for name in needed
    }
    return c


train = load("train")
valid = load("valid")
y_valid = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float32, copy=False)
inc_rank = within_user_rank(valid.user_id, inc_valid)

candidate_log = {
    "trusted_incumbent": float(
        evaluate(valid.user_id, y_valid, inc_valid)["primary"]
    )
}
profiles = {}
best = {
    "primary": candidate_log["trusted_incumbent"],
    "half_life": None,
    "family": "incumbent",
    "alpha": 0.0,
    "scores": inc_valid.copy(),
}

half_lives = [None, 14.0, 7.0, 3.5]
blend_grid = np.linspace(0.0, 0.8, 17)

for half_life in half_lives:
    label = "uniform" if half_life is None else ("hl" + str(half_life))
    family_scores = build_scores(train, valid, half_life)
    profiles[label] = {}

    for family, raw_scores in family_scores.items():
        family_rank = within_user_rank(valid.user_id, raw_scores)
        standalone = evaluate(
            valid.user_id, y_valid, family_rank
        )
        candidate_log[label + "_" + family] = float(
            standalone["primary"]
        )

        local_best = -np.inf
        local_alpha = 0.0
        local_scores = None

        for alpha in blend_grid:
            blended = (
                (1.0 - float(alpha)) * inc_rank
                + float(alpha) * family_rank
            )
            primary = float(
                evaluate(valid.user_id, y_valid, blended)["primary"]
            )
            if primary > local_best:
                local_best = primary
                local_alpha = float(alpha)
                local_scores = blended.copy()

        candidate_log[label + "_" + family + "_blend"] = local_best
        profiles[label][family] = {
            "standalone": float(standalone["primary"]),
            "blend": local_best,
            "alpha": local_alpha,
        }

        if local_best > best["primary"]:
            best = {
                "primary": local_best,
                "half_life": half_life,
                "family": family,
                "alpha": local_alpha,
                "scores": local_scores,
            }

valid_scores = np.asarray(best["scores"], dtype=np.float32)
metrics = evaluate(valid.user_id, y_valid, valid_scores)

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print("FINDINGS " + json.dumps({
    "selected_half_life": best["half_life"],
    "selected_family": best["family"],
    "selected_alpha": best["alpha"],
    "profiles": profiles,
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

test = load("test")
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float32, copy=False)

if best["family"] == "incumbent" or best["alpha"] <= 0.0:
    test_scores = inc_test
else:
    combined = combine_splits(train, valid)
    test_families = build_scores(combined, test, best["half_life"])
    new_test_rank = within_user_rank(
        test.user_id, test_families[best["family"]]
    )
    inc_test_rank = within_user_rank(test.user_id, inc_test)
    test_scores = (
        (1.0 - best["alpha"]) * inc_test_rank
        + best["alpha"] * new_test_rank
    ).astype(np.float32)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(metrics["primary"]),
    "gauc": float(metrics["gauc"]),
    "ndcg@5": float(metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))