import os
import time
import json
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
EPS = 1e-5
HALF_LIVES = [None, 2.0, 4.0, 8.0]
BLEND_WEIGHTS = [0.15, 0.30, 0.45, 0.60, 0.75]


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def recency_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.float64)
    if half_life is None:
        return np.ones(dates.shape[0], dtype=np.float64)
    age = float(np.max(dates)) - dates
    w = np.exp2(-age / float(half_life))
    w /= max(float(w.mean()), EPS)
    return w


def entity_rate(train_ids, query_ids, y, weights, cardinality, smoothing, prior):
    train_ids = np.asarray(train_ids, dtype=np.int64)
    query_ids = np.asarray(query_ids, dtype=np.int64)
    count = np.bincount(
        train_ids,
        weights=weights,
        minlength=int(cardinality),
    ).astype(np.float64)
    positive = np.bincount(
        train_ids,
        weights=weights * y,
        minlength=int(cardinality),
    ).astype(np.float64)

    rate = (positive + float(smoothing) * float(prior)) / (
        count + float(smoothing)
    )
    q = np.clip(query_ids, 0, len(rate) - 1)
    return rate[q], count[q]


def sparse_pair_rate(
    train_left,
    train_right,
    query_left,
    query_right,
    right_cardinality,
    y,
    weights,
    query_prior,
    smoothing,
):
    train_left = np.asarray(train_left, dtype=np.int64)
    train_right = np.asarray(train_right, dtype=np.int64)
    query_left = np.asarray(query_left, dtype=np.int64)
    query_right = np.asarray(query_right, dtype=np.int64)

    train_key = train_left * np.int64(right_cardinality) + train_right
    query_key = query_left * np.int64(right_cardinality) + query_right

    unique_key, inverse = np.unique(train_key, return_inverse=True)
    count = np.bincount(inverse, weights=weights).astype(np.float64)
    positive = np.bincount(inverse, weights=weights * y).astype(np.float64)

    pos = np.searchsorted(unique_key, query_key)
    clipped = np.minimum(pos, max(len(unique_key) - 1, 0))
    found = (pos < len(unique_key))
    if len(unique_key):
        found &= unique_key[clipped] == query_key
    else:
        found[:] = False

    q_count = np.zeros(query_key.shape[0], dtype=np.float64)
    q_positive = np.zeros(query_key.shape[0], dtype=np.float64)
    if len(unique_key):
        q_count[found] = count[clipped[found]]
        q_positive[found] = positive[clipped[found]]

    query_prior = np.asarray(query_prior, dtype=np.float64)
    rate = (q_positive + float(smoothing) * query_prior) / (
        q_count + float(smoothing)
    )
    return rate, q_count


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((np.arange(n, dtype=np.int64), scores, user_ids))
    sorted_users = user_ids[order]

    is_start = np.empty(n, dtype=bool)
    is_start[0] = True
    is_start[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.maximum.accumulate(
        np.where(is_start, np.arange(n, dtype=np.int64), 0)
    )
    ascending_position = np.arange(n, dtype=np.float64) - starts

    start_indices = np.flatnonzero(is_start)
    end_indices = np.r_[start_indices[1:], n]
    group_sizes = end_indices - start_indices
    repeated_sizes = np.repeat(group_sizes, group_sizes).astype(np.float64)

    ranked = ascending_position / np.maximum(repeated_sizes - 1.0, 1.0)
    out = np.empty(n, dtype=np.float64)
    out[order] = ranked
    return out


def build_nonparametric_scores(fit, query, half_life):
    y = np.asarray(fit.y, dtype=np.float64)
    weights = recency_weights(fit.date, half_life)
    global_rate = float(np.sum(weights * y) / np.sum(weights))

    video_rate, _ = entity_rate(
        fit.X["video_id"],
        query.X["video_id"],
        y,
        weights,
        FEATURE_CARDINALITIES["video_id"],
        smoothing=20.0,
        prior=global_rate,
    )
    author_rate, _ = entity_rate(
        fit.X["author_id"],
        query.X["author_id"],
        y,
        weights,
        FEATURE_CARDINALITIES["author_id"],
        smoothing=30.0,
        prior=global_rate,
    )
    tag_rate, _ = entity_rate(
        fit.X["tag"],
        query.X["tag"],
        y,
        weights,
        FEATURE_CARDINALITIES["tag"],
        smoothing=80.0,
        prior=global_rate,
    )
    duration_rate, _ = entity_rate(
        fit.X["duration_bucket"],
        query.X["duration_bucket"],
        y,
        weights,
        FEATURE_CARDINALITIES["duration_bucket"],
        smoothing=100.0,
        prior=global_rate,
    )

    lv = logit(video_rate)
    la = logit(author_rate)
    lt = logit(tag_rate)
    ld = logit(duration_rate)

    entity_score = 0.65 * lv + 0.35 * la
    content_score = 0.50 * lv + 0.25 * la + 0.15 * lt + 0.10 * ld

    user_video_rate, user_video_count = sparse_pair_rate(
        fit.X["user_id"],
        fit.X["video_id"],
        query.X["user_id"],
        query.X["video_id"],
        FEATURE_CARDINALITIES["video_id"],
        y,
        weights,
        video_rate,
        smoothing=4.0,
    )
    user_author_rate, user_author_count = sparse_pair_rate(
        fit.X["user_id"],
        fit.X["author_id"],
        query.X["user_id"],
        query.X["author_id"],
        FEATURE_CARDINALITIES["author_id"],
        y,
        weights,
        author_rate,
        smoothing=7.0,
    )
    user_tag_rate, user_tag_count = sparse_pair_rate(
        fit.X["user_id"],
        fit.X["tag"],
        query.X["user_id"],
        query.X["tag"],
        FEATURE_CARDINALITIES["tag"],
        y,
        weights,
        tag_rate,
        smoothing=10.0,
    )

    luv = logit(user_video_rate)
    lua = logit(user_author_rate)
    lut = logit(user_tag_rate)

    repeat_video_score = entity_score + 0.70 * (luv - lv)
    affinity_score = entity_score + 0.55 * (lua - la) + 0.30 * (lut - lt)
    hierarchical_score = (
        content_score
        + 0.45 * (luv - lv)
        + 0.40 * (lua - la)
        + 0.25 * (lut - lt)
    )

    diagnostics = {
        "global_rate": global_rate,
        "uv_seen": float(np.mean(user_video_count > 0)),
        "ua_seen": float(np.mean(user_author_count > 0)),
        "ut_seen": float(np.mean(user_tag_count > 0)),
    }
    scores = {
        "entity_eb": entity_score,
        "content_eb": content_score,
        "repeat_video_eb": repeat_video_score,
        "affinity_eb": affinity_score,
        "hierarchical_eb": hierarchical_score,
    }
    return scores, diagnostics


train = load("train")
valid = load("valid")

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")
if not os.path.exists(inc_valid_path):
    raise RuntimeError("Trusted incumbent validation predictions are unavailable")
if not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent test predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_valid_logit = logit(inc_valid)
inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

candidate_scores = {}
candidate_predictions = {}
candidate_recipes = {}

inc_metrics = evaluate(valid.user_id, valid.y, inc_valid)
candidate_scores["incumbent"] = float(inc_metrics["primary"])
candidate_predictions["incumbent"] = inc_valid
candidate_recipes["incumbent"] = {
    "half_life": None,
    "family": None,
    "mode": "incumbent",
    "weight": 0.0,
}

diagnostic_lines = []

for half_life in HALF_LIVES:
    family_scores, diagnostics = build_nonparametric_scores(
        train, valid, half_life
    )
    hl_name = "uniform" if half_life is None else "hl%.1f" % half_life

    diagnostic_lines.append(
        "%s global=%.5f uv_seen=%.3f ua_seen=%.3f ut_seen=%.3f"
        % (
            hl_name,
            diagnostics["global_rate"],
            diagnostics["uv_seen"],
            diagnostics["ua_seen"],
            diagnostics["ut_seen"],
        )
    )

    for family, raw_score in family_scores.items():
        base_name = "%s_%s" % (family, hl_name)

        met = evaluate(valid.user_id, valid.y, raw_score)
        candidate_scores[base_name] = float(met["primary"])
        candidate_predictions[base_name] = raw_score
        candidate_recipes[base_name] = {
            "half_life": half_life,
            "family": family,
            "mode": "standalone",
            "weight": 1.0,
        }

        raw_rank = within_user_rank(valid.user_id, raw_score)

        for weight in BLEND_WEIGHTS:
            probability_name = "%s_logitblend%.2f" % (base_name, weight)
            probability_blend = (
                weight * raw_score + (1.0 - weight) * inc_valid_logit
            )
            met = evaluate(valid.user_id, valid.y, probability_blend)
            candidate_scores[probability_name] = float(met["primary"])
            candidate_predictions[probability_name] = probability_blend
            candidate_recipes[probability_name] = {
                "half_life": half_life,
                "family": family,
                "mode": "logitblend",
                "weight": weight,
            }

            rank_name = "%s_rankblend%.2f" % (base_name, weight)
            rank_blend = (
                weight * raw_rank + (1.0 - weight) * inc_valid_rank
            )
            met = evaluate(valid.user_id, valid.y, rank_blend)
            candidate_scores[rank_name] = float(met["primary"])
            candidate_predictions[rank_name] = rank_blend
            candidate_recipes[rank_name] = {
                "half_life": half_life,
                "family": family,
                "mode": "rankblend",
                "weight": weight,
            }

winner = max(candidate_scores, key=candidate_scores.get)
valid_scores = np.asarray(candidate_predictions[winner], dtype=np.float64)
winner_recipe = candidate_recipes[winner]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))
for line in diagnostic_lines:
    print("FINDINGS " + line)
print(
    "FINDINGS winner=%s mode=%s family=%s half_life=%s weight=%.2f delta_incumbent=%+.6f"
    % (
        winner,
        winner_recipe["mode"],
        str(winner_recipe["family"]),
        str(winner_recipe["half_life"]),
        float(winner_recipe["weight"]),
        float(metrics["primary"] - inc_metrics["primary"]),
    )
)

out_dir = os.environ.get("ITER_OUT")
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )

test = load("test")
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if winner_recipe["mode"] == "incumbent":
    test_scores = inc_test
else:
    class CombinedSplit:
        pass

    combined = CombinedSplit()
    combined.X = {}
    needed_fields = [
        "user_id",
        "video_id",
        "author_id",
        "tag",
        "duration_bucket",
    ]
    for field in needed_fields:
        combined.X[field] = np.concatenate(
            [
                np.asarray(train.X[field], dtype=np.int64),
                np.asarray(valid.X[field], dtype=np.int64),
            ]
        )
    combined.y = np.concatenate(
        [
            np.asarray(train.y, dtype=np.int8),
            np.asarray(valid.y, dtype=np.int8),
        ]
    )
    combined.date = np.concatenate(
        [
            np.asarray(train.date),
            np.asarray(valid.date),
        ]
    )

    final_families, _ = build_nonparametric_scores(
        combined,
        test,
        winner_recipe["half_life"],
    )
    raw_test = np.asarray(
        final_families[winner_recipe["family"]],
        dtype=np.float64,
    )
    weight = float(winner_recipe["weight"])

    if winner_recipe["mode"] == "standalone":
        test_scores = raw_test
    elif winner_recipe["mode"] == "logitblend":
        test_scores = weight * raw_test + (1.0 - weight) * logit(inc_test)
    elif winner_recipe["mode"] == "rankblend":
        raw_test_rank = within_user_rank(test.user_id, raw_test)
        inc_test_rank = within_user_rank(test.user_id, inc_test)
        test_scores = (
            weight * raw_test_rank + (1.0 - weight) * inc_test_rank
        )
    else:
        raise ValueError("Unknown winning mode: %s" % winner_recipe["mode"])

if out_dir:
    np.save(
        os.path.join(out_dir, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, "gpu_seconds": %.6f}'
    % (
        float(metrics["primary"]),
        float(metrics["gauc"]),
        float(metrics["ndcg@5"]),
        float(elapsed),
    )
)