import os
import time
import json
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate

START = time.time()
SEED = 20260831
np.random.seed(SEED)

train = load("train")
valid = load("valid")
test = load("test")

ytr = np.asarray(train.y, dtype=np.int8)
yva = np.asarray(valid.y, dtype=np.int8)
uva = np.asarray(valid.user_id, dtype=np.int64)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(shared, "incumbent_valid_scores.npy")
inc_test_path = os.path.join(shared, "incumbent_test_scores.npy")

if not os.path.exists(inc_valid_path) or not os.path.exists(inc_test_path):
    raise RuntimeError("Trusted incumbent predictions are unavailable")

inc_valid = np.asarray(np.load(inc_valid_path), dtype=np.float64)
inc_test = np.asarray(np.load(inc_test_path), dtype=np.float64)

if len(inc_valid) != len(uva) or len(inc_test) != len(test.user_id):
    raise RuntimeError("Trusted incumbent prediction length mismatch")


def group_order(users):
    users = np.asarray(users, dtype=np.int64)
    rows = np.arange(len(users), dtype=np.int64)
    order = np.lexsort((rows, users))
    sorted_users = users[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], len(order)]
    return order, starts, ends


def within_user_quality(users, scores):
    """Map scores to [0,1] within-user percentile quality."""
    users = np.asarray(users, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    rows = np.arange(len(scores), dtype=np.int64)

    order = np.lexsort((rows, scores, users))
    sorted_users = users[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    ends = np.r_[starts[1:], len(order)]
    lengths = ends - starts

    positions = (
        np.arange(len(order), dtype=np.float64)
        - np.repeat(starts, lengths)
    )
    denominator = np.maximum(np.repeat(lengths, lengths) - 1, 1)
    quality_sorted = positions / denominator

    result = np.empty(len(scores), dtype=np.float64)
    result[order] = quality_sorted
    return result


# ----------------------------------------------------------------------
# A stationary empirical-Bayes relevance model is fitted on the first ten
# train days. It is used only to select diversity strengths on the final
# three train days; validation labels never choose the reranker strengths.
# ----------------------------------------------------------------------
all_dates = np.unique(np.asarray(train.date, dtype=np.int64))
cut_date = all_dates[-3]
fit_mask = np.asarray(train.date, dtype=np.int64) < cut_date
hold_mask = ~fit_mask
hold_rows = np.flatnonzero(hold_mask)

FIT_FIELDS = [
    "video_id",
    "author_id",
    "tag",
    "tab",
    "duration_bucket",
    "upload_type",
    "onehot_feat3",
    "onehot_feat8",
]

fit_y = ytr[fit_mask].astype(np.float64)
fit_prior = float(fit_y.mean())
prior_logit = np.log(
    (fit_prior + 1.0e-6) / (1.0 - fit_prior + 1.0e-6)
)

hold_base = np.full(len(hold_rows), prior_logit, dtype=np.float64)

for field in FIT_FIELDS:
    card = int(FEATURE_CARDINALITIES[field])
    ids_fit = np.asarray(train.X[field], dtype=np.int64)[fit_mask]
    ids_hold = np.asarray(train.X[field], dtype=np.int64)[hold_mask]

    total = np.bincount(ids_fit, minlength=card).astype(np.float64)
    positive = np.bincount(
        ids_fit, weights=fit_y, minlength=card
    ).astype(np.float64)

    if card >= 5000:
        smoothing = 55.0
    elif card >= 500:
        smoothing = 35.0
    else:
        smoothing = 22.0

    rate = (positive + smoothing * fit_prior) / (total + smoothing)
    evidence = (
        np.log((rate + 1.0e-5) / (1.0 - rate + 1.0e-5))
        - prior_logit
    )
    reliability = total / (total + smoothing)
    hold_base += evidence[ids_hold] * reliability[ids_hold]

hold_users = np.asarray(train.user_id, dtype=np.int64)[hold_mask]
hold_labels = ytr[hold_mask]
hold_author = np.asarray(train.X["author_id"], dtype=np.int64)[hold_mask]
hold_tag = np.asarray(train.X["tag"], dtype=np.int64)[hold_mask]
hold_topic = np.asarray(train.X["onehot_feat3"], dtype=np.int64)[hold_mask]


def rerank(
    users,
    base_scores,
    author,
    tag,
    topic,
    method,
    strength=0.0,
    pool_size=10,
    top_k=5,
):
    """
    Rerank only the first top_k positions, drawing from the original top
    pool_size. Thus diversity can affect nDCG@5 while leaving the long tail
    and most pairwise comparisons unchanged.
    """
    users = np.asarray(users, dtype=np.int64)
    base_scores = np.asarray(base_scores, dtype=np.float64)
    author = np.asarray(author, dtype=np.int64)
    tag = np.asarray(tag, dtype=np.int64)
    topic = np.asarray(topic, dtype=np.int64)

    base_quality = within_user_quality(users, base_scores)
    result = base_quality.copy()

    order, starts, ends = group_order(users)

    for start, end in zip(starts, ends):
        group_rows = order[start:end]
        n = len(group_rows)
        if n <= 1:
            continue

        # Stable descending relevance order.
        local_order = np.lexsort(
            (group_rows, -base_scores[group_rows])
        )
        ranked = group_rows[local_order]

        pool_n = min(pool_size, n)
        choose_n = min(top_k, pool_n)
        pool = ranked[:pool_n]

        if pool_n <= 1:
            continue

        relevance = np.linspace(
            1.0, 0.0, pool_n, dtype=np.float64
        )
        available = np.ones(pool_n, dtype=bool)
        selected_local = []

        if method == "quota":
            author_counts = {}
            for _ in range(choose_n):
                choice = -1
                for j in range(pool_n):
                    if not available[j]:
                        continue
                    a = int(author[pool[j]])
                    if author_counts.get(a, 0) == 0:
                        choice = j
                        break
                if choice < 0:
                    choice = int(np.flatnonzero(available)[0])
                available[choice] = False
                selected_local.append(choice)
                a = int(author[pool[choice]])
                author_counts[a] = author_counts.get(a, 0) + 1
        else:
            for _ in range(choose_n):
                candidate_indices = np.flatnonzero(available)

                if not selected_local:
                    choice = int(candidate_indices[0])
                else:
                    selected_rows = pool[
                        np.asarray(selected_local, dtype=np.int64)
                    ]
                    candidate_rows = pool[candidate_indices]

                    sim = (
                        1.00
                        * (
                            author[candidate_rows, None]
                            == author[selected_rows][None, :]
                        )
                        + 0.40
                        * (
                            tag[candidate_rows, None]
                            == tag[selected_rows][None, :]
                        )
                        + 0.25
                        * (
                            topic[candidate_rows, None]
                            == topic[selected_rows][None, :]
                        )
                    )

                    if method == "max_similarity":
                        redundancy = sim.max(axis=1)
                    elif method == "cumulative":
                        redundancy = sim.sum(axis=1) / np.sqrt(
                            float(len(selected_local))
                        )
                    elif method == "coverage":
                        # Reward candidates covering attributes not represented
                        # by any already selected candidate.
                        redundancy = (
                            (sim > 0.0).mean(axis=1)
                            + 0.5 * sim.max(axis=1)
                        )
                    else:
                        raise ValueError("Unknown reranking method")

                    utility = (
                        relevance[candidate_indices]
                        - float(strength) * redundancy
                    )
                    choice = int(candidate_indices[np.argmax(utility)])

                available[choice] = False
                selected_local.append(choice)

        selected_local = np.asarray(selected_local, dtype=np.int64)
        remaining_pool = np.flatnonzero(available)
        new_pool = np.concatenate([selected_local, remaining_pool])
        new_order = np.concatenate([pool[new_pool], ranked[pool_n:]])

        # Scores encode exactly the newly formed ordering.
        quality = np.linspace(1.0, 0.0, n, dtype=np.float64)
        result[new_order] = quality

    return result


# Select one strength per learned reranking family exclusively on late train.
strength_grid = [0.015, 0.03, 0.06, 0.10, 0.16]
methods = ["max_similarity", "cumulative", "coverage"]
selected_strength = {}
internal_scores = {}

for method in methods:
    best_strength = strength_grid[0]
    best_primary = -np.inf

    for strength in strength_grid:
        candidate = rerank(
            hold_users,
            hold_base,
            hold_author,
            hold_tag,
            hold_topic,
            method=method,
            strength=strength,
        )
        metrics = evaluate(hold_users, hold_labels, candidate)
        score = float(metrics["primary"])
        internal_scores[f"{method}_{strength:.3f}"] = score

        if score > best_primary:
            best_primary = score
            best_strength = strength

    selected_strength[method] = best_strength

quota_hold = rerank(
    hold_users,
    hold_base,
    hold_author,
    hold_tag,
    hold_topic,
    method="quota",
)
internal_scores["quota"] = float(
    evaluate(hold_users, hold_labels, quota_hold)["primary"]
)

print(
    "FINDINGS "
    + json.dumps(
        {
            "late_train_rows": int(len(hold_rows)),
            "selected_strength": selected_strength,
            "internal_best": {
                method: max(
                    score
                    for name, score in internal_scores.items()
                    if name.startswith(method + "_")
                )
                for method in methods
            },
            "internal_quota": internal_scores["quota"],
        },
        sort_keys=True,
    )
)


def sample_arrays(sample):
    return (
        np.asarray(sample.user_id, dtype=np.int64),
        np.asarray(sample.X["author_id"], dtype=np.int64),
        np.asarray(sample.X["tag"], dtype=np.int64),
        np.asarray(sample.X["onehot_feat3"], dtype=np.int64),
    )


valid_users, valid_author, valid_tag, valid_topic = sample_arrays(valid)
test_users, test_author, test_tag, test_topic = sample_arrays(test)

valid_inc_rank = within_user_quality(valid_users, inc_valid)
test_inc_rank = within_user_quality(test_users, inc_test)

valid_family = {}
test_family = {}

for method in methods:
    strength = selected_strength[method]
    valid_family[method] = rerank(
        valid_users,
        inc_valid,
        valid_author,
        valid_tag,
        valid_topic,
        method=method,
        strength=strength,
    )
    test_family[method] = rerank(
        test_users,
        inc_test,
        test_author,
        test_tag,
        test_topic,
        method=method,
        strength=strength,
    )

valid_family["quota"] = rerank(
    valid_users,
    inc_valid,
    valid_author,
    valid_tag,
    valid_topic,
    method="quota",
)
test_family["quota"] = rerank(
    test_users,
    inc_test,
    test_author,
    test_tag,
    test_topic,
    method="quota",
)

# Compare each structurally different reranker alone and blended with the
# incumbent ranking. Blend weights are applied identically to hidden test.
candidate_metrics = {}
candidate_predictions = {}
candidate_test_predictions = {}
candidate_raw = {}

inc_metrics = evaluate(valid_users, yva, inc_valid)
candidate_metrics["incumbent"] = float(inc_metrics["primary"])
candidate_predictions["incumbent"] = inc_valid
candidate_test_predictions["incumbent"] = inc_test
candidate_raw["incumbent"] = valid_family["max_similarity"]

blend_weights = [0.25, 0.50, 0.75, 1.00]

for method in ["max_similarity", "cumulative", "coverage", "quota"]:
    family_valid = valid_family[method]
    family_test = test_family[method]

    for alpha in blend_weights:
        name = f"{method}_blend_{alpha:.2f}"
        valid_score = (
            (1.0 - alpha) * valid_inc_rank
            + alpha * family_valid
        )
        test_score = (
            (1.0 - alpha) * test_inc_rank
            + alpha * family_test
        )

        metrics = evaluate(valid_users, yva, valid_score)
        candidate_metrics[name] = float(metrics["primary"])
        candidate_predictions[name] = valid_score
        candidate_test_predictions[name] = test_score
        candidate_raw[name] = family_valid

best_name = max(candidate_metrics, key=candidate_metrics.get)
valid_scores = np.asarray(
    candidate_predictions[best_name], dtype=np.float64
)
test_scores = np.asarray(
    candidate_test_predictions[best_name], dtype=np.float64
)
raw_scores = np.asarray(candidate_raw[best_name], dtype=np.float64)

final_metrics = evaluate(valid_users, yva, valid_scores)

print("CANDIDATES " + json.dumps(candidate_metrics, sort_keys=True))
print(
    "FINDINGS "
    + json.dumps(
        {
            "winner": best_name,
            "winner_primary": float(final_metrics["primary"]),
            "incumbent_primary": float(inc_metrics["primary"]),
        },
        sort_keys=True,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        valid_scores.astype(np.float64),
    )
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        raw_scores.astype(np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        test_scores.astype(np.float64),
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