import os
import time
import json
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()

train = load("train")
valid = load("valid")
test = load("test")

y = np.asarray(train.y, dtype=np.float64)
dates = np.asarray(train.date, dtype=np.int64)

# Moderate temporal discounting keeps all thirteen days represented while
# emphasizing preferences closest to the future evaluation period.
row_weight = np.exp2((dates - dates.max()).astype(np.float64) / 4.0)
row_weight /= row_weight.mean()


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)

    order = np.lexsort((
        np.arange(n, dtype=np.int64),
        scores,
        user_ids,
    ))
    ordered_users = user_ids[order]
    starts = np.flatnonzero(
        np.r_[True, ordered_users[1:] != ordered_users[:-1]]
    )
    ends = np.r_[starts[1:], n]
    lengths = ends - starts

    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    position = np.arange(n, dtype=np.int64) - repeated_starts

    ranked = np.full(n, 0.5, dtype=np.float64)
    multi = repeated_lengths > 1
    ranked[multi] = (
        position[multi] / (repeated_lengths[multi] - 1.0)
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranked
    return result


def logit(p):
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def aggregate_sorted(keys, numerator, denominator):
    keys = np.asarray(keys, dtype=np.int64)
    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]
    )
    unique_keys = sorted_keys[starts]
    num_sum = np.add.reduceat(
        np.asarray(numerator, dtype=np.float64)[order], starts
    )
    den_sum = np.add.reduceat(
        np.asarray(denominator, dtype=np.float64)[order], starts
    )
    return unique_keys, num_sum, den_sum


def lookup_table(query_keys, keys, values, default=0.0):
    query_keys = np.asarray(query_keys, dtype=np.int64)
    positions = np.searchsorted(keys, query_keys)
    result = np.full(len(query_keys), default, dtype=np.float64)
    valid_pos = positions < len(keys)
    if np.any(valid_pos):
        rows = np.flatnonzero(valid_pos)
        matched = keys[positions[rows]] == query_keys[rows]
        rows = rows[matched]
        result[rows] = values[positions[rows]]
    return result


# ---------------------------------------------------------------------
# Family 1: hierarchical user-content empirical Bayes.
#
# Each user gets a separate posterior preference for author, tag,
# duration class and video. Sparse user/entity pairs shrink toward the
# entity's global recency-weighted rate instead of an unstable user mean.
# ---------------------------------------------------------------------
PROFILE_FIELDS = [
    ("video_id", 0.65, 8.0),
    ("author_id", 0.85, 10.0),
    ("tag", 0.45, 14.0),
    ("duration_bucket", 0.35, 18.0),
    ("upload_type", 0.20, 20.0),
]

profile_tables = {}
global_tables = {}

train_users = np.asarray(train.X["user_id"], dtype=np.int64)
weighted_positive = row_weight * y

for field, field_weight, prior_strength in PROFILE_FIELDS:
    card = int(FEATURE_CARDINALITIES[field])
    entity = np.asarray(train.X[field], dtype=np.int64)

    entity_pos = np.bincount(
        entity, weights=weighted_positive, minlength=card
    ).astype(np.float64)
    entity_den = np.bincount(
        entity, weights=row_weight, minlength=card
    ).astype(np.float64)
    global_mean = float(weighted_positive.sum() / row_weight.sum())
    global_rate = (
        entity_pos + 20.0 * global_mean
    ) / (entity_den + 20.0)
    global_tables[field] = global_rate

    pair_key = train_users * np.int64(card) + entity
    keys, pos_sum, den_sum = aggregate_sorted(
        pair_key, weighted_positive, row_weight
    )
    entity_for_key = keys % np.int64(card)
    posterior = (
        pos_sum + prior_strength * global_rate[entity_for_key]
    ) / (den_sum + prior_strength)

    # Center against the entity prior: this component measures specifically
    # how the user's taste differs from generic item quality.
    preference = (
        logit(posterior) - logit(global_rate[entity_for_key])
    )
    reliability = den_sum / (den_sum + prior_strength)
    values = preference * reliability

    profile_tables[field] = (
        keys,
        values,
        field_weight,
        card,
    )


def predict_profile(split):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    result = np.zeros(len(users), dtype=np.float64)

    # Generic quality remains useful when the user/entity pair is absent.
    result += 0.55 * logit(
        global_tables["video_id"][
            np.asarray(split.X["video_id"], dtype=np.int64)
        ]
    )
    result += 0.35 * logit(
        global_tables["author_id"][
            np.asarray(split.X["author_id"], dtype=np.int64)
        ]
    )

    for field, (keys, values, multiplier, card) in profile_tables.items():
        entity = np.asarray(split.X[field], dtype=np.int64)
        query = users * np.int64(card) + entity
        result += multiplier * lookup_table(query, keys, values)
    return result


profile_valid = predict_profile(valid)
profile_test = predict_profile(test)


# ---------------------------------------------------------------------
# Family 2: cohort-conditioned preference tables.
#
# This deliberately does not memorize user IDs. It estimates how stable
# user cohorts respond to content categories, allowing sparse users to
# borrow evidence from behaviorally similar accounts.
# ---------------------------------------------------------------------
COHORT_CROSSES = [
    ("user_active_degree", "author_id", 0.55, 35.0),
    ("user_active_degree", "tag", 0.45, 30.0),
    ("register_days_bucket", "tag", 0.35, 35.0),
    ("fans_user_num_range", "author_id", 0.35, 45.0),
    ("friend_user_num_range", "tag", 0.25, 45.0),
    ("tab", "author_id", 0.55, 30.0),
    ("tab", "duration_bucket", 0.35, 30.0),
]

cohort_tables = []

for cohort_field, item_field, multiplier, strength in COHORT_CROSSES:
    item_card = int(FEATURE_CARDINALITIES[item_field])
    cohort = np.asarray(train.X[cohort_field], dtype=np.int64)
    item = np.asarray(train.X[item_field], dtype=np.int64)
    key = cohort * np.int64(item_card) + item

    keys, pos_sum, den_sum = aggregate_sorted(
        key, weighted_positive, row_weight
    )
    item_for_key = keys % np.int64(item_card)
    prior = global_tables.get(item_field)

    if prior is None:
        item_pos = np.bincount(
            item, weights=weighted_positive, minlength=item_card
        ).astype(np.float64)
        item_den = np.bincount(
            item, weights=row_weight, minlength=item_card
        ).astype(np.float64)
        mean_y = float(weighted_positive.sum() / row_weight.sum())
        prior = (item_pos + 20.0 * mean_y) / (item_den + 20.0)

    posterior = (
        pos_sum + strength * prior[item_for_key]
    ) / (den_sum + strength)
    deviation = logit(posterior) - logit(prior[item_for_key])
    reliability = den_sum / (den_sum + strength)

    cohort_tables.append((
        cohort_field,
        item_field,
        item_card,
        keys,
        multiplier * deviation * reliability,
    ))


def predict_cohort(split):
    video = np.asarray(split.X["video_id"], dtype=np.int64)
    author = np.asarray(split.X["author_id"], dtype=np.int64)

    result = (
        0.65 * logit(global_tables["video_id"][video])
        + 0.45 * logit(global_tables["author_id"][author])
    )

    for cohort_field, item_field, item_card, keys, values in cohort_tables:
        cohort = np.asarray(split.X[cohort_field], dtype=np.int64)
        item = np.asarray(split.X[item_field], dtype=np.int64)
        query = cohort * np.int64(item_card) + item
        result += lookup_table(query, keys, values)

    return result


cohort_valid = predict_cohort(valid)
cohort_test = predict_cohort(test)


# ---------------------------------------------------------------------
# Family 3: ordered positive-transition recommender.
#
# Consecutive positive entities in each user's train history define a
# first-order Markov model. Predictions use transitions from the user's
# final three positive entities, with destination-popularity correction.
# This uses actual (user_id, time_ms, row position) order.
# ---------------------------------------------------------------------
def fit_transition(field):
    card = int(FEATURE_CARDINALITIES[field])
    users = np.asarray(train.user_id, dtype=np.int64)
    times = np.asarray(train.time_ms, dtype=np.int64)
    entity = np.asarray(train.X[field], dtype=np.int64)
    rows = np.arange(len(y), dtype=np.int64)

    positive_rows = np.flatnonzero(y > 0.5)
    ordering = np.lexsort((
        rows[positive_rows],
        times[positive_rows],
        users[positive_rows],
    ))
    pos_rows = positive_rows[ordering]
    pos_users = users[pos_rows]
    pos_entities = entity[pos_rows]

    adjacent = pos_users[1:] == pos_users[:-1]
    source = pos_entities[:-1][adjacent]
    destination = pos_entities[1:][adjacent]
    transition_weight = row_weight[pos_rows[1:][adjacent]]

    transition_key = source * np.int64(card) + destination
    keys, count_sum, _ = aggregate_sorted(
        transition_key,
        transition_weight,
        np.ones_like(transition_weight),
    )

    destination_pop = np.bincount(
        entity,
        weights=weighted_positive,
        minlength=card,
    ).astype(np.float64)
    destination_for_key = keys % np.int64(card)

    # Downweight transitions explained only by globally popular destinations.
    values = np.log1p(
        count_sum / np.sqrt(destination_pop[destination_for_key] + 2.0)
    )

    user_card = int(FEATURE_CARDINALITIES["user_id"])
    last_entities = np.zeros((user_card, 3), dtype=np.int64)
    has_history = np.zeros((user_card, 3), dtype=bool)

    if len(pos_users):
        group_starts = np.flatnonzero(
            np.r_[True, pos_users[1:] != pos_users[:-1]]
        )
        group_ends = np.r_[group_starts[1:], len(pos_users)]
        group_users = pos_users[group_starts]

        for lag in range(3):
            candidate_index = group_ends - 1 - lag
            available = candidate_index >= group_starts
            u = group_users[available]
            idx = candidate_index[available]
            last_entities[u, lag] = pos_entities[idx]
            has_history[u, lag] = True

    return {
        "card": card,
        "keys": keys,
        "values": values,
        "last": last_entities,
        "has": has_history,
        "global_rate": (
            destination_pop + 2.0 * float(y.mean())
        ) / (
            np.bincount(entity, weights=row_weight, minlength=card)
            + 2.0
        ),
    }


video_transition = fit_transition("video_id")
author_transition = fit_transition("author_id")


def predict_transition_one(split, field, model):
    users = np.asarray(split.X["user_id"], dtype=np.int64)
    destination = np.asarray(split.X[field], dtype=np.int64)
    result = 0.25 * logit(model["global_rate"][destination])

    lag_weights = (1.0, 0.55, 0.30)
    known_user = (users >= 0) & (users < len(model["last"]))

    for lag, lag_weight in enumerate(lag_weights):
        source = np.zeros(len(users), dtype=np.int64)
        available = np.zeros(len(users), dtype=bool)
        rows = np.flatnonzero(known_user)
        source[rows] = model["last"][users[rows], lag]
        available[rows] = model["has"][users[rows], lag]

        query = source * np.int64(model["card"]) + destination
        contribution = lookup_table(query, model["keys"], model["values"])
        result += lag_weight * contribution * available

    return result


transition_valid = (
    0.65 * predict_transition_one(valid, "video_id", video_transition)
    + 0.35 * predict_transition_one(valid, "author_id", author_transition)
)
transition_test = (
    0.65 * predict_transition_one(test, "video_id", video_transition)
    + 0.35 * predict_transition_one(test, "author_id", author_transition)
)


# A heterogeneous non-parametric ensemble is also a candidate. Rank
# aggregation prevents one family's arbitrary score scale from dominating.
profile_valid_rank = within_user_rank(valid.user_id, profile_valid)
profile_test_rank = within_user_rank(test.user_id, profile_test)
cohort_valid_rank = within_user_rank(valid.user_id, cohort_valid)
cohort_test_rank = within_user_rank(test.user_id, cohort_test)
transition_valid_rank = within_user_rank(valid.user_id, transition_valid)
transition_test_rank = within_user_rank(test.user_id, transition_test)

ensemble_valid = (
    0.45 * profile_valid_rank
    + 0.35 * cohort_valid_rank
    + 0.20 * transition_valid_rank
)
ensemble_test = (
    0.45 * profile_test_rank
    + 0.35 * cohort_test_rank
    + 0.20 * transition_test_rank
)

families = {
    "hierarchical_user_content": (profile_valid, profile_test),
    "cohort_conditioned_tables": (cohort_valid, cohort_test),
    "ordered_markov_transitions": (transition_valid, transition_test),
    "heterogeneous_nonparametric": (ensemble_valid, ensemble_test),
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

candidate_summary = {}
candidate_arrays = {}

inc_metrics = evaluate(valid.user_id, valid.y, inc_valid)
candidate_summary["trusted_incumbent"] = float(inc_metrics["primary"])
candidate_arrays["trusted_incumbent"] = (
    inc_valid,
    inc_test,
    ensemble_valid,
)

# Fixed grid, with every selected validation weight transferred unchanged
# to test. Including the incumbent prevents an uninformative new family
# from degrading the submitted iteration.
blend_alphas = (0.10, 0.20, 0.30, 0.45)

for name, (own_valid, own_test) in families.items():
    own_metrics = evaluate(valid.user_id, valid.y, own_valid)
    candidate_summary[name + "_raw"] = float(own_metrics["primary"])
    candidate_arrays[name + "_raw"] = (
        own_valid,
        own_test,
        own_valid,
    )

    own_valid_rank = within_user_rank(valid.user_id, own_valid)
    own_test_rank = within_user_rank(test.user_id, own_test)

    for alpha in blend_alphas:
        blended_valid = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * own_valid_rank
        )
        blended_test = (
            (1.0 - alpha) * inc_test_rank
            + alpha * own_test_rank
        )
        blend_name = name + "_blend_" + str(alpha)
        metrics = evaluate(valid.user_id, valid.y, blended_valid)
        candidate_summary[blend_name] = float(metrics["primary"])
        candidate_arrays[blend_name] = (
            blended_valid,
            blended_test,
            own_valid,
        )

winner = max(candidate_summary, key=candidate_summary.get)
valid_scores, test_scores, raw_valid_scores = candidate_arrays[winner]
final_metrics = evaluate(valid.user_id, valid.y, valid_scores)

print("CANDIDATES " + json.dumps(
    candidate_summary, sort_keys=True
))
print("FINDINGS " + json.dumps({
    "winner": winner,
    "profile_raw": candidate_summary["hierarchical_user_content_raw"],
    "cohort_raw": candidate_summary["cohort_conditioned_tables_raw"],
    "transition_raw": candidate_summary["ordered_markov_transitions_raw"],
    "ensemble_raw": candidate_summary["heterogeneous_nonparametric_raw"],
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(raw_valid_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(final_metrics["primary"]),
    "gauc": float(final_metrics["gauc"]),
    "ndcg@5": float(final_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))