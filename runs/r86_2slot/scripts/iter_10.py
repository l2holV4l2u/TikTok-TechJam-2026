import os
import time
import json
import gc
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()

MARGINAL_FIELDS = [
    "video_id", "author_id", "tag", "duration_bucket",
    "upload_type", "music_type", "tab", "hour",
]
PAIR_FIELDS = [
    "video_id", "author_id", "tag", "duration_bucket",
    "upload_type", "music_type", "tab",
]

POP_WEIGHTS = {
    "video_id": 1.00,
    "author_id": 0.75,
    "tag": 0.35,
    "duration_bucket": 0.20,
    "upload_type": 0.25,
    "music_type": 0.15,
    "tab": 0.20,
    "hour": 0.10,
}

PAIR_WEIGHTS = {
    "video_id": 1.00,
    "author_id": 1.00,
    "tag": 0.65,
    "duration_bucket": 0.35,
    "upload_type": 0.40,
    "music_type": 0.25,
    "tab": 0.45,
}


class JoinedSplit:
    pass


def join_splits(a, b):
    out = JoinedSplit()
    required = sorted(set(MARGINAL_FIELDS + PAIR_FIELDS + ["user_id"]))
    out.X = {
        f: np.concatenate([
            np.asarray(a.X[f], dtype=np.int64),
            np.asarray(b.X[f], dtype=np.int64),
        ])
        for f in required
    }
    out.user_id = np.concatenate([
        np.asarray(a.user_id, dtype=np.int64),
        np.asarray(b.user_id, dtype=np.int64),
    ])
    out.video_id = np.concatenate([
        np.asarray(a.video_id, dtype=np.int64),
        np.asarray(b.video_id, dtype=np.int64),
    ])
    out.date = np.concatenate([
        np.asarray(a.date, dtype=np.int32),
        np.asarray(b.date, dtype=np.int32),
    ])
    return out


def ordinal_day(dates):
    d = np.asarray(dates, dtype=np.int64)
    month = (d // 100) % 100
    day = d % 100
    return day + np.where(month == 5, 30, 0)


def temporal_weights(dates, half_life):
    if half_life is None:
        return np.ones(len(dates), dtype=np.float64)
    day = ordinal_day(dates)
    age = day.max() - day
    w = np.exp2(-age.astype(np.float64) / float(half_life))
    return w / max(float(w.mean()), 1e-12)


def clipped_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(p) - np.log1p(-p)


def reduce_codes(codes, weights, weighted_labels):
    order = np.argsort(codes, kind="stable")
    sc = codes[order]
    sw = weights[order]
    sy = weighted_labels[order]

    starts_mask = np.empty(len(sc), dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sc[1:] != sc[:-1]
    starts = np.flatnonzero(starts_mask)

    unique = sc[starts]
    counts = np.add.reduceat(sw, starts)
    positives = np.add.reduceat(sy, starts)
    return unique, counts, positives


def sparse_lookup(unique, values, query):
    query = np.asarray(query, dtype=np.int64)
    pos = np.searchsorted(unique, query)
    found = pos < len(unique)
    safe = np.minimum(pos, max(len(unique) - 1, 0))
    if len(unique):
        found &= unique[safe] == query
    result = np.zeros(len(query), dtype=np.float64)
    if len(unique):
        result[found] = values[safe[found]]
    return result, found


class EBPack:
    def __init__(self, split, labels, half_life):
        labels = np.asarray(labels, dtype=np.float64)
        weights = temporal_weights(split.date, half_life)
        weighted_labels = weights * labels

        self.global_rate = float(
            weighted_labels.sum() / max(weights.sum(), 1e-12)
        )
        self.marginal = {}
        self.pairs = {}

        users = np.asarray(split.user_id, dtype=np.int64)

        for field in MARGINAL_FIELDS:
            x = np.asarray(split.X[field], dtype=np.int64)
            card = int(FEATURE_CARDINALITIES[field])
            count = np.bincount(
                x, weights=weights, minlength=card
            ).astype(np.float64)
            positive = np.bincount(
                x, weights=weighted_labels, minlength=card
            ).astype(np.float64)
            self.marginal[field] = (count, positive)

        for field in PAIR_FIELDS:
            x = np.asarray(split.X[field], dtype=np.int64)
            card = int(FEATURE_CARDINALITIES[field])
            codes = users * np.int64(card) + x
            unique, count, positive = reduce_codes(
                codes, weights, weighted_labels
            )
            self.pairs[field] = (card, unique, count, positive)

        self.half_life = half_life

    def marginal_rates(self, split, strength):
        result = {}
        g = self.global_rate
        for field in MARGINAL_FIELDS:
            x = np.asarray(split.X[field], dtype=np.int64)
            count, positive = self.marginal[field]
            c = count[x]
            s = positive[x]
            result[field] = (s + strength * g) / (c + strength)
        return result

    def score(self, split, marginal_strength, pair_strength,
              pair_scale, popularity=True):
        marginal = self.marginal_rates(split, marginal_strength)
        n = len(split.user_id)
        score = np.zeros(n, dtype=np.float64)

        if popularity:
            total_weight = 0.0
            for field, weight in POP_WEIGHTS.items():
                score += weight * clipped_logit(marginal[field])
                total_weight += abs(weight)
            score /= max(total_weight, 1e-12)

        if pair_scale != 0.0:
            users = np.asarray(split.user_id, dtype=np.int64)
            residual = np.zeros(n, dtype=np.float64)
            total_pair_weight = 0.0

            for field, field_weight in PAIR_WEIGHTS.items():
                x = np.asarray(split.X[field], dtype=np.int64)
                card, unique, count, positive = self.pairs[field]
                query = users * np.int64(card) + x

                pair_count, found_count = sparse_lookup(
                    unique, count, query
                )
                pair_positive, found_positive = sparse_lookup(
                    unique, positive, query
                )
                found = found_count & found_positive

                prior = marginal[field]
                posterior = (
                    pair_positive + pair_strength * prior
                ) / (pair_count + pair_strength)

                delta = clipped_logit(posterior) - clipped_logit(prior)
                delta[~found] = 0.0
                residual += field_weight * delta
                total_pair_weight += abs(field_weight)

            residual /= max(total_pair_weight, 1e-12)
            score += pair_scale * residual

        return score


def within_user_rank(user_ids, scores):
    users = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    rows = np.arange(n, dtype=np.int64)

    order = np.lexsort((rows, scores, users))
    su = users[order]

    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = su[1:] != su[:-1]
    starts = np.flatnonzero(starts_mask)
    group = np.cumsum(starts_mask) - 1
    sizes = np.diff(np.r_[starts, n])
    position = rows - starts[group]
    denom = np.maximum(sizes[group] - 1, 1)

    ranked_sorted = position.astype(np.float64) / denom
    ranked_sorted[sizes[group] == 1] = 0.5

    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def metric_primary(users, labels, scores):
    return float(evaluate(users, labels, scores)["primary"])


train = load("train")
valid = load("valid")
y_train = np.asarray(train.y, dtype=np.int8)
y_valid = np.asarray(valid.y, dtype=np.int8)
valid_users = np.asarray(valid.user_id, dtype=np.int64)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy")
).astype(np.float64)
inc_valid_rank = within_user_rank(valid_users, inc_valid)

# Each specification represents a materially different prediction rule:
# marginal popularity, hierarchical personal residual, or recent personal EB.
specifications = [
    {
        "name": "stable_marginal_eb",
        "half_life": None,
        "marginal_strength": 30.0,
        "pair_strength": 12.0,
        "pair_scale": 0.0,
        "popularity": True,
    },
    {
        "name": "stable_hierarchical_personal",
        "half_life": None,
        "marginal_strength": 30.0,
        "pair_strength": 12.0,
        "pair_scale": 0.85,
        "popularity": True,
    },
    {
        "name": "recent7_hierarchical_personal",
        "half_life": 7.0,
        "marginal_strength": 24.0,
        "pair_strength": 10.0,
        "pair_scale": 0.85,
        "popularity": True,
    },
    {
        "name": "recent3_hierarchical_personal",
        "half_life": 3.0,
        "marginal_strength": 20.0,
        "pair_strength": 8.0,
        "pair_scale": 0.75,
        "popularity": True,
    },
    {
        "name": "recent7_preference_residual",
        "half_life": 7.0,
        "marginal_strength": 24.0,
        "pair_strength": 10.0,
        "pair_scale": 1.0,
        "popularity": False,
    },
]

packs = {}
candidate_scores = {}
candidate_details = {}
best = None

for spec in specifications:
    key = spec["half_life"]
    if key not in packs:
        packs[key] = EBPack(train, y_train, key)

    own = packs[key].score(
        valid,
        marginal_strength=spec["marginal_strength"],
        pair_strength=spec["pair_strength"],
        pair_scale=spec["pair_scale"],
        popularity=spec["popularity"],
    )
    own_rank = within_user_rank(valid_users, own)
    own_primary = metric_primary(valid_users, y_valid, own_rank)

    family_best = None
    for alpha in [0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.0]:
        blended = alpha * own_rank + (1.0 - alpha) * inc_valid_rank
        metrics = evaluate(valid_users, y_valid, blended)
        primary = float(metrics["primary"])
        if family_best is None or primary > family_best["primary"]:
            family_best = {
                "primary": primary,
                "alpha": float(alpha),
                "scores": blended,
                "metrics": metrics,
                "own": own_rank,
                "own_primary": own_primary,
                "spec": spec,
            }

    candidate_scores[spec["name"] + "_raw"] = own_primary
    candidate_scores[spec["name"] + "_blend"] = family_best["primary"]
    candidate_details[spec["name"]] = {
        "raw": own_primary,
        "blend": family_best["primary"],
        "alpha": family_best["alpha"],
    }

    if best is None or family_best["primary"] > best["primary"]:
        best = family_best

    print(
        "FINDINGS %s raw=%.6f blend=%.6f alpha=%.2f"
        % (
            spec["name"],
            own_primary,
            family_best["primary"],
            family_best["alpha"],
        )
    )

print("CANDIDATES " + json.dumps(candidate_scores, sort_keys=True))

valid_scores = np.asarray(best["scores"], dtype=np.float64)
own_valid_scores = np.asarray(best["own"], dtype=np.float64)
final_metrics = evaluate(valid_users, y_valid, valid_scores)

out = os.environ.get("ITER_OUT")
if out:
    np.save(
        os.path.join(out, "scores_valid.npy"),
        valid_scores.astype(np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        own_valid_scores.astype(np.float64),
    )

winner = best["spec"]
winner_alpha = float(best["alpha"])

print(
    "FINDINGS winner=%s half_life=%s alpha=%.2f raw_primary=%.6f"
    % (
        winner["name"],
        str(winner["half_life"]),
        winner_alpha,
        best["own_primary"],
    )
)

# Permitted test refit: apply the selected recipe to train+validation labels.
te = load("test")
joined = join_splits(train, valid)
joined_y = np.concatenate([y_train, y_valid]).astype(np.int8)

test_pack = EBPack(joined, joined_y, winner["half_life"])
own_test = test_pack.score(
    te,
    marginal_strength=winner["marginal_strength"],
    pair_strength=winner["pair_strength"],
    pair_scale=winner["pair_scale"],
    popularity=winner["popularity"],
)
own_test_rank = within_user_rank(te.user_id, own_test)

inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy")
).astype(np.float64)
inc_test_rank = within_user_rank(te.user_id, inc_test)

test_scores = (
    winner_alpha * own_test_rank
    + (1.0 - winner_alpha) * inc_test_rank
)

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
print(
    'METRICS {"primary": %.10f, "gauc": %.10f, "ndcg@5": %.10f, '
    '"gpu_seconds": %.6f}'
    % (
        float(final_metrics["primary"]),
        float(final_metrics["gauc"]),
        float(final_metrics["ndcg@5"]),
        elapsed,
    )
)