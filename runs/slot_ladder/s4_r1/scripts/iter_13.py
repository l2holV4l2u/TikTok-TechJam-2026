import os
import time
import json
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
HALF_LIFE = 4.0
SMOOTH = 20.0

train = load("train")
valid = load("valid")
test = load("test")

y = np.asarray(train.y, dtype=np.float64)
dates = np.asarray(train.date, dtype=np.int64)
weights = np.exp2((dates - dates.max()).astype(np.float64) / HALF_LIFE)
weights /= weights.mean()
global_rate = float(np.sum(weights * y) / np.sum(weights))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def aggregate_rate(train_ids, query_ids, cardinality, strength=SMOOTH):
    train_ids = np.asarray(train_ids, dtype=np.int64)
    query_ids = np.asarray(query_ids, dtype=np.int64)

    count = np.bincount(
        train_ids, weights=weights, minlength=cardinality
    ).astype(np.float64)
    positives = np.bincount(
        train_ids, weights=weights * y, minlength=cardinality
    ).astype(np.float64)

    posterior = (positives + strength * global_rate) / (count + strength)
    return posterior[query_ids], count[query_ids]


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    # Row position gives deterministic ordering for exact score ties.
    order = np.lexsort(
        (np.arange(n, dtype=np.int64), scores, user_ids)
    )
    ordered_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_users[1:] != ordered_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(n, dtype=np.int64) - repeated_starts

    ordered_ranks = np.full(n, 0.5, dtype=np.float64)
    mask = repeated_lengths > 1
    ordered_ranks[mask] = (
        positions[mask] / (repeated_lengths[mask] - 1.0)
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ordered_ranks
    return result


# -------------------------------------------------------------------------
# Family 1: Generative categorical Naive Bayes.
#
# This forms a prediction from class-conditional feature likelihoods rather
# than discriminative embeddings. Contributions are clipped so highly rare
# categories cannot dominate the slate ordering.
# -------------------------------------------------------------------------
NB_FIELDS = [
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
    "onehot_feat1",
    "onehot_feat7",
    "register_days_bucket",
    "user_active_degree",
    "music_type",
]

sum_w_pos = float(np.sum(weights * y))
sum_w_neg = float(np.sum(weights * (1.0 - y)))
prior_log_odds = np.log(sum_w_pos / sum_w_neg)


def naive_bayes_scores(split):
    score = np.full(len(split.user_id), prior_log_odds, dtype=np.float64)

    for field in NB_FIELDS:
        card = int(FEATURE_CARDINALITIES[field])
        tr_ids = np.asarray(train.X[field], dtype=np.int64)
        q_ids = np.asarray(split.X[field], dtype=np.int64)

        pos = np.bincount(
            tr_ids, weights=weights * y, minlength=card
        ).astype(np.float64)
        neg = np.bincount(
            tr_ids, weights=weights * (1.0 - y), minlength=card
        ).astype(np.float64)

        # Symmetric Dirichlet smoothing of the class-conditional multinomials.
        alpha = 1.0
        log_ratio = (
            np.log(pos + alpha)
            - np.log(sum_w_pos + alpha * card)
            - np.log(neg + alpha)
            + np.log(sum_w_neg + alpha * card)
        )
        score += np.clip(log_ratio[q_ids], -2.5, 2.5)

    return score


nb_valid = naive_bayes_scores(valid)
nb_test = naive_bayes_scores(test)


# -------------------------------------------------------------------------
# Family 2: Supervised heterogeneous graph diffusion.
#
# Video nodes receive direct label evidence. Their values are repeatedly
# smoothed through author, tag, and duration nodes. This lets sparse videos
# borrow evidence from multiple side-information neighborhoods without
# learning an embedding geometry.
# -------------------------------------------------------------------------
video_card = int(FEATURE_CARDINALITIES["video_id"])
graph_fields = ["author_id", "tag", "duration_bucket"]

tr_video = np.asarray(train.X["video_id"], dtype=np.int64)
video_count = np.bincount(
    tr_video, weights=weights, minlength=video_card
).astype(np.float64)
video_positive = np.bincount(
    tr_video, weights=weights * y, minlength=video_card
).astype(np.float64)

video_value = (
    video_positive + SMOOTH * global_rate
) / (video_count + SMOOTH)

attribute_values = {}

for _ in range(3):
    neighbor_sum = np.zeros(video_card, dtype=np.float64)
    neighbor_weight = np.zeros(video_card, dtype=np.float64)
    attribute_values = {}

    for field in graph_fields:
        ids = np.asarray(train.X[field], dtype=np.int64)
        card = int(FEATURE_CARDINALITIES[field])

        attr_count = np.bincount(
            ids, weights=weights, minlength=card
        ).astype(np.float64)
        attr_sum = np.bincount(
            ids,
            weights=weights * video_value[tr_video],
            minlength=card,
        ).astype(np.float64)

        attr_value = (
            attr_sum + 12.0 * global_rate
        ) / (attr_count + 12.0)
        attribute_values[field] = attr_value

        neighbor_sum += np.bincount(
            tr_video,
            weights=weights * attr_value[ids],
            minlength=video_card,
        )
        neighbor_weight += np.bincount(
            tr_video, weights=weights, minlength=video_card
        )

    graph_strength = 1.5
    video_value = (
        video_positive
        + SMOOTH * global_rate
        + graph_strength * neighbor_sum
    ) / (
        video_count
        + SMOOTH
        + graph_strength * neighbor_weight
    )


def graph_scores(split):
    q_video = np.asarray(split.X["video_id"], dtype=np.int64)
    score = 1.8 * logit(video_value[q_video])

    for field in graph_fields:
        q_ids = np.asarray(split.X[field], dtype=np.int64)
        score += 0.55 * logit(attribute_values[field][q_ids])

    # Deterministically break residual graph-score ties.
    score += 1e-10 * q_video
    return score


graph_valid = graph_scores(valid)
graph_test = graph_scores(test)


# -------------------------------------------------------------------------
# Family 3: User–entity neighborhood propagation.
#
# A user's historical long-view evidence is propagated to currently logged
# videos through exact video, author, and tag edges. Unlike a global target
# statistic, these scores represent personalized neighborhoods and vary
# across candidate impressions for the same user.
# -------------------------------------------------------------------------
def pair_posterior(train_left, train_right, query_left, query_right,
                   right_cardinality, strength):
    train_left = np.asarray(train_left, dtype=np.int64)
    train_right = np.asarray(train_right, dtype=np.int64)
    query_left = np.asarray(query_left, dtype=np.int64)
    query_right = np.asarray(query_right, dtype=np.int64)

    train_key = train_left * np.int64(right_cardinality) + train_right
    query_key = query_left * np.int64(right_cardinality) + query_right

    unique_key, inverse = np.unique(train_key, return_inverse=True)
    count = np.bincount(inverse, weights=weights).astype(np.float64)
    positive = np.bincount(
        inverse, weights=weights * y
    ).astype(np.float64)

    posterior = (
        positive + strength * global_rate
    ) / (count + strength)

    position = np.searchsorted(unique_key, query_key)
    found = position < len(unique_key)
    safe_position = np.minimum(position, len(unique_key) - 1)
    found &= unique_key[safe_position] == query_key

    result = np.full(len(query_key), global_rate, dtype=np.float64)
    result[found] = posterior[safe_position[found]]

    support = np.zeros(len(query_key), dtype=np.float64)
    support[found] = count[safe_position[found]]
    return result, support


def neighborhood_scores(split):
    tr_user = np.asarray(train.X["user_id"], dtype=np.int64)
    q_user = np.asarray(split.X["user_id"], dtype=np.int64)

    components = []
    confidences = []

    definitions = [
        ("video_id", 8.0),
        ("author_id", 12.0),
        ("tag", 18.0),
    ]

    for field, strength in definitions:
        card = int(FEATURE_CARDINALITIES[field])
        rate, support = pair_posterior(
            tr_user,
            np.asarray(train.X[field], dtype=np.int64),
            q_user,
            np.asarray(split.X[field], dtype=np.int64),
            card,
            strength,
        )
        components.append(logit(rate))
        confidences.append(support / (support + strength))

    video_rate, _ = aggregate_rate(
        train.X["video_id"],
        split.X["video_id"],
        int(FEATURE_CARDINALITIES["video_id"]),
        strength=18.0,
    )
    author_rate, _ = aggregate_rate(
        train.X["author_id"],
        split.X["author_id"],
        int(FEATURE_CARDINALITIES["author_id"]),
        strength=24.0,
    )

    personalized_num = np.zeros(len(q_user), dtype=np.float64)
    personalized_den = np.zeros(len(q_user), dtype=np.float64)
    component_weights = [1.2, 0.9, 0.5]

    for component, confidence, coefficient in zip(
        components, confidences, component_weights
    ):
        effective = coefficient * confidence
        personalized_num += effective * component
        personalized_den += effective

    personalized = np.divide(
        personalized_num,
        personalized_den,
        out=np.full(len(q_user), logit(global_rate), dtype=np.float64),
        where=personalized_den > 0,
    )

    return (
        1.15 * personalized
        + 0.85 * logit(video_rate)
        + 0.45 * logit(author_rate)
        + 1e-10 * np.asarray(split.X["video_id"], dtype=np.float64)
    )


neighbor_valid = neighborhood_scores(valid)
neighbor_test = neighborhood_scores(test)


raw_valid = {
    "generative_naive_bayes": nb_valid,
    "heterogeneous_graph_diffusion": graph_valid,
    "user_entity_neighborhood": neighbor_valid,
}
raw_test = {
    "generative_naive_bayes": nb_test,
    "heterogeneous_graph_diffusion": graph_test,
    "user_entity_neighborhood": neighbor_test,
}

shared = os.environ.get("SHARED_ARTIFACTS")
if not shared:
    raise RuntimeError("SHARED_ARTIFACTS is required")

inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)
inc_test_rank = within_user_rank(test.user_id, inc_test)

candidate_metrics = {}
candidate_valid = {}
candidate_test = {}
candidate_own_valid = {}
candidate_is_blend = {}

inc_metrics = evaluate(valid.user_id, valid.y, inc_valid)
candidate_metrics["trusted_incumbent"] = float(inc_metrics["primary"])
candidate_valid["trusted_incumbent"] = inc_valid
candidate_test["trusted_incumbent"] = inc_test
candidate_own_valid["trusted_incumbent"] = inc_valid
candidate_is_blend["trusted_incumbent"] = False

blend_alphas = [0.10, 0.20, 0.30, 0.45, 0.60]

for family in raw_valid:
    rv = raw_valid[family]
    rt = raw_test[family]

    metrics = evaluate(valid.user_id, valid.y, rv)
    candidate_metrics[family] = float(metrics["primary"])
    candidate_valid[family] = rv
    candidate_test[family] = rt
    candidate_own_valid[family] = rv
    candidate_is_blend[family] = False

    rv_rank = within_user_rank(valid.user_id, rv)
    rt_rank = within_user_rank(test.user_id, rt)

    for alpha in blend_alphas:
        name = family + "_rank_blend_" + str(alpha)
        bv = (1.0 - alpha) * inc_valid_rank + alpha * rv_rank
        bt = (1.0 - alpha) * inc_test_rank + alpha * rt_rank

        metrics = evaluate(valid.user_id, valid.y, bv)
        candidate_metrics[name] = float(metrics["primary"])
        candidate_valid[name] = bv
        candidate_test[name] = bt
        candidate_own_valid[name] = rv
        candidate_is_blend[name] = True

# Also test an equal-rank graph consensus before blending it with the incumbent.
consensus_valid = np.mean(
    [
        within_user_rank(valid.user_id, raw_valid[name])
        for name in raw_valid
    ],
    axis=0,
)
consensus_test = np.mean(
    [
        within_user_rank(test.user_id, raw_test[name])
        for name in raw_test
    ],
    axis=0,
)

consensus_metrics = evaluate(
    valid.user_id, valid.y, consensus_valid
)
candidate_metrics["three_family_consensus"] = float(
    consensus_metrics["primary"]
)
candidate_valid["three_family_consensus"] = consensus_valid
candidate_test["three_family_consensus"] = consensus_test
candidate_own_valid["three_family_consensus"] = consensus_valid
candidate_is_blend["three_family_consensus"] = False

for alpha in blend_alphas:
    name = "three_family_consensus_blend_" + str(alpha)
    bv = (1.0 - alpha) * inc_valid_rank + alpha * consensus_valid
    bt = (1.0 - alpha) * inc_test_rank + alpha * consensus_test
    metrics = evaluate(valid.user_id, valid.y, bv)

    candidate_metrics[name] = float(metrics["primary"])
    candidate_valid[name] = bv
    candidate_test[name] = bt
    candidate_own_valid[name] = consensus_valid
    candidate_is_blend[name] = True

winner = max(candidate_metrics, key=candidate_metrics.get)
winner_valid = candidate_valid[winner]
winner_test = candidate_test[winner]
final_metrics = evaluate(valid.user_id, valid.y, winner_valid)

# Rank-correlation diagnostics quantify whether each mechanism is genuinely
# changing ordering rather than merely recalibrating incumbent scores.
inc_rank_centered = inc_valid_rank - inc_valid_rank.mean()
correlations = {}
for family, values in raw_valid.items():
    family_rank = within_user_rank(valid.user_id, values)
    centered = family_rank - family_rank.mean()
    denom = np.sqrt(
        np.sum(inc_rank_centered ** 2) * np.sum(centered ** 2)
    )
    correlations[family] = float(
        np.sum(inc_rank_centered * centered) / denom
    )

print("FINDINGS rank_correlations_with_incumbent " + json.dumps(
    correlations, sort_keys=True
))
print("FINDINGS selected_candidate " + winner)
print("CANDIDATES " + json.dumps(
    {k: round(v, 6) for k, v in candidate_metrics.items()},
    sort_keys=True
))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(winner_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(winner_test, dtype=np.float64),
    )
    if candidate_is_blend[winner]:
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(candidate_own_valid[winner], dtype=np.float64),
        )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps(
        {
            "primary": float(final_metrics["primary"]),
            "gauc": float(final_metrics["gauc"]),
            "ndcg@5": float(final_metrics["ndcg@5"]),
            "gpu_seconds": float(elapsed),
        }
    )
)