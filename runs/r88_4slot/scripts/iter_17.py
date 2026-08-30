import os
import time
import json
import numpy as np

from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate


START = time.time()
TOP_L = 12
TOP_K = 5


def rank_within_user(user_ids, scores):
    user_ids = np.asarray(user_ids)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.r_[0, np.flatnonzero(
        sorted_users[1:] != sorted_users[:-1]
    ) + 1]
    ends = np.r_[starts[1:], n]
    counts = ends - starts

    repeated_starts = np.repeat(starts, counts)
    repeated_counts = np.repeat(counts, counts)
    positions = (
        np.arange(n, dtype=np.float64) - repeated_starts
    )

    ranks = np.full(n, 0.5, dtype=np.float64)
    mask = repeated_counts > 1
    ranks[mask] = (
        positions[mask] / (repeated_counts[mask] - 1.0)
    )

    result = np.empty(n, dtype=np.float64)
    result[order] = ranks
    return result


def make_entity_codes(split, policy):
    if policy == "author":
        return [
            np.asarray(split.X["author_id"], dtype=np.int64)
        ]
    if policy == "video":
        return [
            np.asarray(split.X["video_id"], dtype=np.int64)
        ]
    if policy == "tag":
        return [
            np.asarray(split.X["tag"], dtype=np.int64)
        ]
    if policy == "author_tag":
        return [
            np.asarray(split.X["author_id"], dtype=np.int64),
            np.asarray(split.X["tag"], dtype=np.int64),
        ]
    if policy == "video_author_tag":
        return [
            np.asarray(split.X["video_id"], dtype=np.int64),
            np.asarray(split.X["author_id"], dtype=np.int64),
            np.asarray(split.X["tag"], dtype=np.int64),
        ]
    raise ValueError(policy)


def diversity_rerank(user_ids, base_scores, entity_arrays, penalty):
    """
    Greedily reorder only the incumbent's top TOP_L candidates. At each
    of the first TOP_K positions, subtract a redundancy cost for each
    entity value already represented. The original score multiset is
    preserved within every user, so this is purely a ranking policy.
    """
    user_ids = np.asarray(user_ids)
    base_scores = np.asarray(base_scores, dtype=np.float64)
    n = len(base_scores)
    row = np.arange(n, dtype=np.int64)

    # Users contiguous, with candidates initially sorted best first.
    order = np.lexsort((row, -base_scores, user_ids))
    sorted_users = user_ids[order]
    starts = np.r_[0, np.flatnonzero(
        sorted_users[1:] != sorted_users[:-1]
    ) + 1]
    ends = np.r_[starts[1:], n]

    result = base_scores.copy()

    for start, end in zip(starts, ends):
        size = int(end - start)
        if size <= 1:
            continue

        group = order[start:end]
        limit = min(size, TOP_L)
        candidates = list(group[:limit])
        selected = []
        selected_counts = [dict() for _ in entity_arrays]

        choose_count = min(TOP_K, limit)
        for position in range(choose_count):
            best_j = 0
            best_utility = -1e100

            for j, idx in enumerate(candidates):
                # Rank utility has unit range over the local candidate
                # window, making the penalty transferable across users.
                original_position = position + j
                base_utility = 1.0 - (
                    original_position / max(limit - 1, 1)
                )

                redundancy = 0.0
                for field_no, values in enumerate(entity_arrays):
                    value = int(values[idx])
                    redundancy += selected_counts[field_no].get(
                        value, 0
                    )

                utility = base_utility - penalty * redundancy
                if utility > best_utility:
                    best_utility = utility
                    best_j = j

            idx = candidates.pop(best_j)
            selected.append(idx)
            for field_no, values in enumerate(entity_arrays):
                value = int(values[idx])
                counts = selected_counts[field_no]
                counts[value] = counts.get(value, 0) + 1

        # Candidates not selected for the first five slots retain their
        # incumbent relative order.
        selected_set = set(selected)
        remaining_top = [
            idx for idx in group[:limit]
            if idx not in selected_set
        ]
        new_order = (
            selected + remaining_top + list(group[limit:])
        )

        # Assign the user's original descending score levels to the new
        # ordering. Tiny deterministic offsets eliminate accidental ties.
        levels = np.sort(base_scores[group])[::-1]
        for position, idx in enumerate(new_order):
            result[idx] = levels[position] - 1e-12 * position

    return result


def top5_duplicate_statistics(split, scores):
    users = np.asarray(split.user_id)
    scores = np.asarray(scores, dtype=np.float64)
    row = np.arange(len(scores), dtype=np.int64)
    order = np.lexsort((row, -scores, users))
    su = users[order]
    starts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]

    author = np.asarray(split.X["author_id"], dtype=np.int64)
    video = np.asarray(split.X["video_id"], dtype=np.int64)
    tag = np.asarray(split.X["tag"], dtype=np.int64)

    eligible = 0
    author_dup = 0
    video_dup = 0
    tag_dup = 0

    for start, end in zip(starts, ends):
        k = min(5, int(end - start))
        if k < 2:
            continue
        idx = order[start:start + k]
        eligible += 1
        author_dup += int(len(np.unique(author[idx])) < k)
        video_dup += int(len(np.unique(video[idx])) < k)
        tag_dup += int(len(np.unique(tag[idx])) < k)

    denom = max(eligible, 1)
    return {
        "eligible_users": int(eligible),
        "author_duplicate_share": author_dup / denom,
        "video_duplicate_share": video_dup / denom,
        "tag_duplicate_share": tag_dup / denom,
    }


valid = load("valid")
valid_users = np.asarray(valid.user_id)
valid_y = np.asarray(valid.y, dtype=np.int8)

shared = os.environ.get("SHARED_ARTIFACTS", "")
inc_valid_path = os.path.join(
    shared, "incumbent_valid_scores.npy"
)
inc_test_path = os.path.join(
    shared, "incumbent_test_scores.npy"
)

if not (
    os.path.exists(inc_valid_path)
    and os.path.exists(inc_test_path)
):
    raise RuntimeError("Trusted incumbent predictions unavailable")

inc_valid = np.asarray(
    np.load(inc_valid_path), dtype=np.float64
)
if len(inc_valid) != len(valid_users):
    raise RuntimeError("Incumbent validation length mismatch")

inc_valid_rank = rank_within_user(valid_users, inc_valid)
inc_metrics = evaluate(valid_users, valid_y, inc_valid_rank)

duplicate_stats = top5_duplicate_statistics(valid, inc_valid)
print(
    "FINDINGS " + json.dumps(
        {"incumbent_top5_redundancy": duplicate_stats},
        sort_keys=True,
    )
)

policies = [
    "author",
    "video",
    "tag",
    "author_tag",
    "video_author_tag",
]
penalties = [-0.20, -0.10, 0.08, 0.16, 0.28, 0.42]
blend_alphas = [0.50, 0.75, 1.00]

candidate_log = {
    "trusted_incumbent": float(inc_metrics["primary"])
}

best_primary = float(inc_metrics["primary"])
best_policy = None
best_penalty = 0.0
best_alpha = 0.0
best_metrics = inc_metrics
best_valid_scores = inc_valid_rank.copy()
best_raw_valid = inc_valid_rank.copy()

for policy in policies:
    entity_arrays = make_entity_codes(valid, policy)
    policy_best = -np.inf
    policy_best_penalty = None
    policy_best_alpha = None

    for penalty in penalties:
        raw = diversity_rerank(
            valid_users,
            inc_valid,
            entity_arrays,
            float(penalty),
        )
        raw_rank = rank_within_user(valid_users, raw)

        for alpha in blend_alphas:
            blended = (
                (1.0 - alpha) * inc_valid_rank
                + alpha * raw_rank
            )
            metrics = evaluate(
                valid_users, valid_y, blended
            )
            primary = float(metrics["primary"])

            if primary > policy_best:
                policy_best = primary
                policy_best_penalty = float(penalty)
                policy_best_alpha = float(alpha)

            if primary > best_primary:
                best_primary = primary
                best_policy = policy
                best_penalty = float(penalty)
                best_alpha = float(alpha)
                best_metrics = metrics
                best_valid_scores = blended.copy()
                best_raw_valid = raw_rank.copy()

    candidate_log[policy + "_best"] = float(policy_best)
    candidate_log[policy + "_penalty"] = float(
        policy_best_penalty
    )
    candidate_log[policy + "_alpha"] = float(
        policy_best_alpha
    )

candidate_log["winner_penalty"] = float(best_penalty)
candidate_log["winner_alpha"] = float(best_alpha)
candidate_log["winner_policy_code"] = float(
    -1 if best_policy is None else policies.index(best_policy)
)

print("CANDIDATES " + json.dumps(candidate_log, sort_keys=True))
print(
    "FINDINGS " + json.dumps(
        {
            "winner_policy": (
                "incumbent" if best_policy is None
                else best_policy
            ),
            "winner_penalty": best_penalty,
            "winner_alpha": best_alpha,
        },
        sort_keys=True,
    )
)

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid_scores, dtype=np.float64),
    )
    # The raw transformed ranking is saved because the reported score
    # can be a rank blend with the externally supplied incumbent.
    np.save(
        os.path.join(out, "scores_valid_raw.npy"),
        np.asarray(best_raw_valid, dtype=np.float64),
    )

test = load("test")
inc_test = np.asarray(
    np.load(inc_test_path), dtype=np.float64
)
test_users = np.asarray(test.user_id)

if len(inc_test) != len(test_users):
    raise RuntimeError("Incumbent test length mismatch")

inc_test_rank = rank_within_user(test_users, inc_test)

if best_policy is None or best_alpha == 0.0:
    test_scores = inc_test_rank
else:
    test_entities = make_entity_codes(test, best_policy)
    raw_test = diversity_rerank(
        test_users,
        inc_test,
        test_entities,
        best_penalty,
    )
    raw_test_rank = rank_within_user(test_users, raw_test)
    test_scores = (
        (1.0 - best_alpha) * inc_test_rank
        + best_alpha * raw_test_rank
    )

if out:
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(test_scores, dtype=np.float64),
    )

elapsed = time.time() - START
final = {
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}
print("METRICS " + json.dumps(final))