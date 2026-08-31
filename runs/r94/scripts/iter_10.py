import os
import time
import json
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
EPS = 1.0e-5


def group_positions(order, change):
    n = order.size
    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = change
    starts = np.flatnonzero(starts_mask)
    group_ids = np.cumsum(starts_mask) - 1
    positions_sorted = np.arange(n, dtype=np.int64) - starts[group_ids]
    positions = np.empty(n, dtype=np.int64)
    positions[order] = positions_sorted
    return positions


def make_context(split):
    uid = np.asarray(split.user_id, dtype=np.int64)
    time_ms = np.asarray(split.time_ms, dtype=np.int64)
    dates = np.asarray(split.date, dtype=np.int64)
    rows = np.arange(uid.size, dtype=np.int64)

    chronological = np.lexsort((rows, time_ms, uid))
    su = uid[chronological]
    st = time_ms[chronological]
    sd = dates[chronological]

    day_change = (
        (su[1:] != su[:-1])
        | (sd[1:] != sd[:-1])
    )
    day_pos = group_positions(chronological, day_change)

    batch_change = (
        (su[1:] != su[:-1])
        | (st[1:] != st[:-1])
    )
    batch_pos = group_positions(chronological, batch_change)

    gap_sorted = np.empty(uid.size, dtype=np.float64)
    gap_sorted[0] = 1.0e12
    gap_sorted[1:] = st[1:].astype(np.float64) - st[:-1].astype(np.float64)
    same_user = np.empty(uid.size, dtype=bool)
    same_user[0] = False
    same_user[1:] = su[1:] == su[:-1]
    gap_sorted[~same_user] = 1.0e12
    gap = np.empty(uid.size, dtype=np.float64)
    gap[chronological] = gap_sorted
    gap_bin = np.digitize(
        gap,
        np.asarray([0.0, 1000.0, 10000.0, 60000.0, 600000.0,
                    3600000.0, 86400000.0], dtype=np.float64),
        right=True,
    ).astype(np.int64)

    def repeat_position(entity):
        entity = np.asarray(entity, dtype=np.int64)
        order = np.lexsort((rows, time_ms, entity, dates, uid))
        ou = uid[order]
        od = dates[order]
        oe = entity[order]
        change = (
            (ou[1:] != ou[:-1])
            | (od[1:] != od[:-1])
            | (oe[1:] != oe[:-1])
        )
        return group_positions(order, change)

    repeat_video = repeat_position(split.video_id)
    repeat_author = repeat_position(split.X["author_id"])

    weekday = ((dates - 20220404) % 7).astype(np.int64)

    values = {
        "tab": np.asarray(split.X["tab"], dtype=np.int64),
        "hour": np.asarray(split.X["hour"], dtype=np.int64),
        "duration": np.asarray(split.X["duration_bucket"], dtype=np.int64),
        "tag": np.asarray(split.X["tag"], dtype=np.int64),
        "upload": np.asarray(split.X["upload_type"], dtype=np.int64),
        "weekday": weekday,
        "day_pos": np.minimum(day_pos, 15).astype(np.int64),
        "batch_pos": np.minimum(batch_pos, 7).astype(np.int64),
        "repeat_video": np.minimum(repeat_video, 7).astype(np.int64),
        "repeat_author": np.minimum(repeat_author, 11).astype(np.int64),
        "gap": gap_bin,
    }

    cards = {
        "tab": int(FEATURE_CARDINALITIES["tab"]),
        "hour": int(FEATURE_CARDINALITIES["hour"]),
        "duration": int(FEATURE_CARDINALITIES["duration_bucket"]),
        "tag": int(FEATURE_CARDINALITIES["tag"]),
        "upload": int(FEATURE_CARDINALITIES["upload_type"]),
        "weekday": 7,
        "day_pos": 16,
        "batch_pos": 8,
        "repeat_video": 8,
        "repeat_author": 12,
        "gap": 8,
    }
    return values, cards


def recency_weights(dates, half_life):
    dates = np.asarray(dates, dtype=np.int64)
    latest = int(dates.max())
    w = np.power(2.0, (dates.astype(np.float64) - latest) / half_life)
    return w / w.mean()


def sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def fit_rate_table(code, cardinality, y, weights, prior_mean, prior_strength):
    counts = np.bincount(
        code,
        weights=weights,
        minlength=cardinality,
    ).astype(np.float64)
    positives = np.bincount(
        code,
        weights=weights * y,
        minlength=cardinality,
    ).astype(np.float64)
    rates = (
        positives + prior_strength * prior_mean
    ) / (
        counts + prior_strength
    )
    return rates, counts


SINGLES = [
    "tab", "hour", "duration", "tag", "upload",
    "weekday", "day_pos", "batch_pos",
    "repeat_video", "repeat_author", "gap",
]

CROSSES = [
    ("tab", "duration"),
    ("tab", "tag"),
    ("hour", "tab"),
    ("day_pos", "tab"),
    ("batch_pos", "tab"),
    ("repeat_video", "day_pos"),
    ("repeat_author", "day_pos"),
    ("repeat_author", "tab"),
    ("gap", "day_pos"),
    ("upload", "duration"),
]


def cross_code(context, cards, left, right):
    return context[left] * cards[right] + context[right]


def fit_context_models(context, cards, y, weights):
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    base = float(np.sum(weights * y) / np.sum(weights))
    base_logit = float(logit(base))

    single_tables = {}
    single_counts = {}
    for name in SINGLES:
        table, counts = fit_rate_table(
            context[name],
            cards[name],
            y,
            weights,
            base,
            prior_strength=150.0,
        )
        single_tables[name] = table
        single_counts[name] = counts

    cross_tables = {}
    cross_counts = {}
    for left, right in CROSSES:
        code = cross_code(context, cards, left, right)
        cardinality = cards[left] * cards[right]
        table, counts = fit_rate_table(
            code,
            cardinality,
            y,
            weights,
            base,
            prior_strength=250.0,
        )
        cross_tables[(left, right)] = table
        cross_counts[(left, right)] = counts

    return {
        "base": base,
        "base_logit": base_logit,
        "single_tables": single_tables,
        "single_counts": single_counts,
        "cross_tables": cross_tables,
        "cross_counts": cross_counts,
    }


def predict_context_models(model, context, cards):
    n = next(iter(context.values())).size
    base_logit = model["base_logit"]

    single_delta = np.zeros(n, dtype=np.float64)
    for name in SINGLES:
        rates = model["single_tables"][name][context[name]]
        single_delta += logit(rates) - base_logit

    # Conditional-independence style empirical Bayes model.
    additive = base_logit + 0.24 * single_delta

    cross_logits = np.zeros(n, dtype=np.float64)
    cross_reliability = np.zeros(n, dtype=np.float64)
    for left, right in CROSSES:
        code = cross_code(context, cards, left, right)
        rates = model["cross_tables"][(left, right)][code]
        counts = model["cross_counts"][(left, right)][code]
        reliability = counts / (counts + 400.0)
        cross_logits += reliability * logit(rates)
        cross_reliability += reliability

    hierarchical = np.where(
        cross_reliability > 0,
        cross_logits / np.maximum(cross_reliability, EPS),
        base_logit,
    )
    hierarchical = 0.45 * hierarchical + 0.55 * (
        base_logit + 0.16 * single_delta
    )

    # A deterministic novelty/fatigue reranker structurally independent
    # of the label-smoothed target-statistic models.
    novelty = (
        -1.10 * np.log1p(context["repeat_video"].astype(np.float64))
        -0.45 * np.log1p(context["repeat_author"].astype(np.float64))
        -0.10 * np.log1p(context["day_pos"].astype(np.float64))
        -0.06 * context["batch_pos"].astype(np.float64)
    )

    return {
        "context_additive": additive,
        "context_hierarchical_cross": hierarchical,
        "novelty_diversification": novelty,
    }


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, user_ids))
    sorted_users = user_ids[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    groups = np.cumsum(starts_mask) - 1
    positions = np.arange(n, dtype=np.int64) - starts[groups]
    sizes = np.diff(np.append(starts, n))
    denominators = np.maximum(sizes[groups] - 1, 1)

    ranked_sorted = positions.astype(np.float64) / denominators
    ranked_sorted[sizes[groups] == 1] = 0.5

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


train = load("train")
valid = load("valid")

train_context, cards = make_context(train)
valid_context, valid_cards = make_context(valid)

for key in cards:
    if cards[key] != valid_cards[key]:
        raise RuntimeError("Context cardinality mismatch")

y_train = np.asarray(train.y, dtype=np.float64)

uniform_weights = np.ones(y_train.size, dtype=np.float64)
recent_weights = recency_weights(train.date, half_life=4.0)

uniform_model = fit_context_models(
    train_context, cards, y_train, uniform_weights
)
recent_model = fit_context_models(
    train_context, cards, y_train, recent_weights
)

uniform_predictions = predict_context_models(
    uniform_model, valid_context, cards
)
recent_predictions_raw = predict_context_models(
    recent_model, valid_context, cards
)

valid_own = {
    "uniform_additive": uniform_predictions["context_additive"],
    "uniform_hierarchical_cross":
        uniform_predictions["context_hierarchical_cross"],
    "recency_additive": recent_predictions_raw["context_additive"],
    "recency_hierarchical_cross":
        recent_predictions_raw["context_hierarchical_cross"],
    "novelty_diversification":
        recent_predictions_raw["novelty_diversification"],
}

shared_dir = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared_dir, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared_dir, "incumbent_test_scores.npy"
)
if not os.path.exists(inc_valid_path):
    raise FileNotFoundError("Trusted incumbent validation scores unavailable")
if not os.path.exists(inc_test_path):
    raise FileNotFoundError("Trusted incumbent test scores unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
if inc_valid.size != valid.user_id.size:
    raise RuntimeError("Incumbent validation length mismatch")

inc_valid_rank = within_user_rank(valid.user_id, inc_valid)

candidate_metrics = {}
candidate_scores = {}
candidate_family = {}
candidate_alpha = {}

inc_result = evaluate(valid.user_id, valid.y, inc_valid_rank)
candidate_metrics["trusted_incumbent"] = float(inc_result["primary"])
candidate_scores["trusted_incumbent"] = inc_valid_rank
candidate_family["trusted_incumbent"] = "recency_hierarchical_cross"
candidate_alpha["trusted_incumbent"] = 0.0

for family, own_scores in valid_own.items():
    own_rank = within_user_rank(valid.user_id, own_scores)

    standalone = evaluate(valid.user_id, valid.y, own_rank)
    standalone_name = family + "_standalone"
    candidate_metrics[standalone_name] = float(standalone["primary"])
    candidate_scores[standalone_name] = own_rank
    candidate_family[standalone_name] = family
    candidate_alpha[standalone_name] = 1.0

    for alpha in (0.05, 0.10, 0.20, 0.35, 0.50):
        blended = (
            (1.0 - alpha) * inc_valid_rank
            + alpha * own_rank
        )
        name = f"{family}_blend_{alpha:.2f}"
        result = evaluate(valid.user_id, valid.y, blended)
        candidate_metrics[name] = float(result["primary"])
        candidate_scores[name] = blended
        candidate_family[name] = family
        candidate_alpha[name] = alpha

winner_name = max(candidate_metrics, key=candidate_metrics.get)
winner_family = candidate_family[winner_name]
winner_alpha = candidate_alpha[winner_name]
valid_scores = candidate_scores[winner_name]
metrics = evaluate(valid.user_id, valid.y, valid_scores)

repeat_video = train_context["repeat_video"]
repeat_rates = {}
for value in range(4):
    mask = repeat_video == value
    repeat_rates[str(value)] = (
        float(y_train[mask].mean()) if np.any(mask) else None
    )
mask = repeat_video >= 4
repeat_rates["4+"] = float(y_train[mask].mean()) if np.any(mask) else None

day_pos = train_context["day_pos"]
position_rates = {}
for lo, hi, name in [
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 5, "3-5"),
    (6, 10, "6-10"),
    (11, 15, "11+"),
]:
    mask = (day_pos >= lo) & (day_pos <= hi)
    position_rates[name] = (
        float(y_train[mask].mean()) if np.any(mask) else None
    )

print(
    "FINDINGS context_target_rates "
    + json.dumps({
        "same_day_video_repeat": repeat_rates,
        "within_day_position": position_rates,
    }, sort_keys=True)
)
print("CANDIDATES " + json.dumps(candidate_metrics, sort_keys=True))
print(
    "FINDINGS winner "
    + json.dumps({
        "candidate": winner_name,
        "family": winner_family,
        "own_rank_weight": float(winner_alpha),
    }, sort_keys=True)
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(valid_scores, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(valid_own[winner_family], dtype=np.float64),
    )

test = load("test")
test_context, test_cards = make_context(test)
for key in cards:
    if cards[key] != test_cards[key]:
        raise RuntimeError("Test context cardinality mismatch")

uniform_test_predictions = predict_context_models(
    uniform_model, test_context, cards
)
recent_test_predictions = predict_context_models(
    recent_model, test_context, cards
)

test_own = {
    "uniform_additive":
        uniform_test_predictions["context_additive"],
    "uniform_hierarchical_cross":
        uniform_test_predictions["context_hierarchical_cross"],
    "recency_additive":
        recent_test_predictions["context_additive"],
    "recency_hierarchical_cross":
        recent_test_predictions["context_hierarchical_cross"],
    "novelty_diversification":
        recent_test_predictions["novelty_diversification"],
}

inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)
if inc_test.size != test.user_id.size:
    raise RuntimeError("Incumbent test length mismatch")

inc_test_rank = within_user_rank(test.user_id, inc_test)
own_test_rank = within_user_rank(
    test.user_id, test_own[winner_family]
)
test_scores = (
    (1.0 - winner_alpha) * inc_test_rank
    + winner_alpha * own_test_rank
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    "METRICS "
    + json.dumps({
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "gpu_seconds": float(elapsed),
    })
)