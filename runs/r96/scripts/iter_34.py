import os
import time
import json
import glob
import numpy as np

from pipeline.data import load
from pipeline.history import historical_features
from pipeline.evaluate import evaluate


START = time.time()


def grouped_rank_features(user_ids, scores):
    """Return within-user ascending percentile and descending average-tie rank."""
    users = np.asarray(user_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    n = len(values)

    finite = np.isfinite(values)
    if not np.all(finite):
        replacement = float(np.nanmedian(values[finite])) if np.any(finite) else 0.0
        values = np.where(finite, values, replacement)

    row = np.arange(n, dtype=np.int64)
    order = np.lexsort((row, values, users))
    su = users[order]
    sv = values[order]

    user_start_flag = np.empty(n, dtype=bool)
    user_start_flag[0] = True
    user_start_flag[1:] = su[1:] != su[:-1]

    user_end_flag = np.empty(n, dtype=bool)
    user_end_flag[-1] = True
    user_end_flag[:-1] = su[:-1] != su[1:]

    user_start = np.maximum.accumulate(
        np.where(user_start_flag, np.arange(n, dtype=np.int64), 0)
    )
    user_end = np.minimum.accumulate(
        np.where(user_end_flag, np.arange(n, dtype=np.int64), n - 1)[::-1]
    )[::-1]
    user_size = user_end - user_start + 1

    tie_start_flag = np.empty(n, dtype=bool)
    tie_start_flag[0] = True
    tie_start_flag[1:] = (
        (su[1:] != su[:-1]) |
        (sv[1:] != sv[:-1])
    )
    tie_end_flag = np.empty(n, dtype=bool)
    tie_end_flag[-1] = True
    tie_end_flag[:-1] = (
        (su[:-1] != su[1:]) |
        (sv[:-1] != sv[1:])
    )

    tie_start = np.maximum.accumulate(
        np.where(tie_start_flag, np.arange(n, dtype=np.int64), 0)
    )
    tie_end = np.minimum.accumulate(
        np.where(tie_end_flag, np.arange(n, dtype=np.int64), n - 1)[::-1]
    )[::-1]

    average_position = 0.5 * (
        (tie_start - user_start).astype(np.float64) +
        (tie_end - user_start).astype(np.float64)
    )
    percentile_sorted = (average_position + 0.5) / user_size
    descending_rank_sorted = user_size.astype(np.float64) - average_position

    percentile = np.empty(n, dtype=np.float64)
    descending_rank = np.empty(n, dtype=np.float64)
    percentile[order] = percentile_sorted
    descending_rank[order] = descending_rank_sorted
    return percentile, descending_rank


def history_consensus(split_name, user_ids):
    histories = historical_features(split_name, key="video_id")
    author_histories = historical_features(split_name, key="author_id")

    rank_columns = []
    weights = []
    used = []

    for prefix, collection in (
        ("video", histories),
        ("author", author_histories),
    ):
        for key in sorted(collection):
            lower = key.lower()
            if "rate" not in lower:
                continue

            values = np.asarray(collection[key], dtype=np.float64)
            if len(values) != len(user_ids):
                continue

            percentile, _ = grouped_rank_features(user_ids, values)
            weight = 3.0 if "long_view" in lower else 1.0
            # Entity-level long-view histories are more specific than auxiliary
            # engagement rates, while the latter break many popularity ties.
            if prefix == "video":
                weight *= 1.15

            rank_columns.append(percentile)
            weights.append(weight)
            used.append(prefix + ":" + key)

    if not rank_columns:
        return np.zeros(len(user_ids), dtype=np.float64), []

    matrix = np.stack(rank_columns, axis=1)
    weight_array = np.asarray(weights, dtype=np.float64)
    weight_array /= weight_array.sum()
    consensus = matrix @ weight_array
    return consensus, used


def load_archived_pairs(run_root, n_valid, n_test):
    """Find explicitly named reusable score/prediction arrays with test siblings."""
    pairs = []
    if not run_root or not os.path.isdir(run_root):
        return pairs

    patterns = [
        os.path.join(run_root, "**", "*valid*score*.npy"),
        os.path.join(run_root, "**", "*score*valid*.npy"),
        os.path.join(run_root, "**", "*valid*pred*.npy"),
        os.path.join(run_root, "**", "*pred*valid*.npy"),
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern, recursive=True))

    seen = set()
    for valid_path in sorted(set(candidates)):
        if valid_path in seen:
            continue
        seen.add(valid_path)

        base = os.path.basename(valid_path)
        possible_bases = {
            base.replace("valid", "test"),
            base.replace("validation", "test"),
            base.replace("val", "test"),
        }
        possible_paths = [
            os.path.join(os.path.dirname(valid_path), b)
            for b in possible_bases
            if b != base
        ]
        test_path = next((p for p in possible_paths if os.path.isfile(p)), None)
        if test_path is None:
            continue

        try:
            va = np.load(valid_path, allow_pickle=False)
            te = np.load(test_path, allow_pickle=False)
            va = np.asarray(va, dtype=np.float64).reshape(-1)
            te = np.asarray(te, dtype=np.float64).reshape(-1)
        except Exception:
            continue

        if len(va) != n_valid or len(te) != n_test:
            continue
        if not np.all(np.isfinite(va)) or not np.all(np.isfinite(te)):
            continue

        name = os.path.relpath(valid_path, run_root)
        pairs.append((name, va, te))

    return pairs


def rank_correlation(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ac = a - a.mean()
    bc = b - b.mean()
    denominator = np.sqrt(np.dot(ac, ac) * np.dot(bc, bc))
    if denominator <= 0:
        return 0.0
    return float(np.dot(ac, bc) / denominator)


valid = load("valid")
test = load("test")

shared = os.environ.get("SHARED_ARTIFACTS")
inc_valid = np.load(
    os.path.join(shared, "incumbent_valid_scores.npy"),
    allow_pickle=False,
).astype(np.float64).reshape(-1)
inc_test = np.load(
    os.path.join(shared, "incumbent_test_scores.npy"),
    allow_pickle=False,
).astype(np.float64).reshape(-1)

inc_valid_borda, inc_valid_rank = grouped_rank_features(
    valid.user_id, inc_valid
)
inc_test_borda, inc_test_rank = grouped_rank_features(
    test.user_id, inc_test
)

history_valid, history_keys = history_consensus("valid", valid.user_id)
history_test, _ = history_consensus("test", test.user_id)
history_valid_borda, history_valid_rank = grouped_rank_features(
    valid.user_id, history_valid
)
history_test_borda, history_test_rank = grouped_rank_features(
    test.user_id, history_test
)

experts = {
    "history_feedback_consensus": (
        history_valid,
        history_test,
        history_valid_borda,
        history_test_borda,
        history_valid_rank,
        history_test_rank,
    )
}

archived = load_archived_pairs(
    os.environ.get("RUN_ARTIFACTS"),
    len(valid.user_id),
    len(test.user_id),
)

# Retain only genuinely complementary archived predictions. If iteration #29
# was persisted, its low correlation with the incumbent should place it first.
archive_findings = {}
for index, (path, va, te) in enumerate(archived):
    va_borda, va_rank = grouped_rank_features(valid.user_id, va)
    te_borda, te_rank = grouped_rank_features(test.user_id, te)
    correlation = rank_correlation(inc_valid_borda, va_borda)
    archive_findings[path] = correlation
    if correlation < 0.995:
        name = "archived_%02d" % index
        experts[name] = (
            va, te, va_borda, te_borda, va_rank, te_rank
        )

candidates = {}
candidate_test = {}
candidate_source = {}

def add_candidate(name, valid_scores, test_scores, source):
    metrics = evaluate(valid.user_id, valid.y, valid_scores)
    candidates[name] = (
        float(metrics["primary"]),
        metrics,
        np.asarray(valid_scores, dtype=np.float64),
    )
    candidate_test[name] = np.asarray(test_scores, dtype=np.float64)
    candidate_source[name] = source


add_candidate(
    "trusted_incumbent",
    inc_valid,
    inc_test,
    "history_feedback_consensus",
)

# Standalone historical consensus records whether the complementary expert is
# useful by itself, while all competitive candidates preserve the incumbent.
add_candidate(
    "history_feedback_consensus_standalone",
    history_valid,
    history_test,
    "history_feedback_consensus",
)

for expert_name, bundle in experts.items():
    raw_va, raw_te, va_borda, te_borda, va_rank, te_rank = bundle

    # Borda preserves ordering information throughout the slate and therefore
    # primarily protects GAUC.
    for alpha in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        va = (1.0 - alpha) * inc_valid_borda + alpha * va_borda
        te = (1.0 - alpha) * inc_test_borda + alpha * te_borda
        add_candidate(
            f"{expert_name}_borda_{alpha:.2f}",
            va,
            te,
            expert_name,
        )

    # Reciprocal-rank fusion gives disproportionate influence to agreement at
    # the first few positions, directly targeting nDCG@5. Incumbent weights
    # above one make disagreement conservative.
    for k in (1.0, 5.0, 20.0, 60.0):
        for incumbent_weight in (1.0, 2.0, 4.0):
            va = (
                incumbent_weight / (k + inc_valid_rank) +
                1.0 / (k + va_rank)
            )
            te = (
                incumbent_weight / (k + inc_test_rank) +
                1.0 / (k + te_rank)
            )
            add_candidate(
                f"{expert_name}_rrf_k{int(k)}_iw{int(incumbent_weight)}",
                va,
                te,
                expert_name,
            )

# If a diverse archived model exists, also test a three-way fixed RRF. This
# lets train-only historical feedback resolve disagreements between it and the
# incumbent without fitting a temporal stacker.
archive_names = [name for name in experts if name.startswith("archived_")]
for archive_name in archive_names:
    _, _, _, _, archive_va_rank, archive_te_rank = experts[archive_name]
    for k in (5.0, 20.0, 60.0):
        va = (
            2.0 / (k + inc_valid_rank) +
            1.0 / (k + archive_va_rank) +
            0.5 / (k + history_valid_rank)
        )
        te = (
            2.0 / (k + inc_test_rank) +
            1.0 / (k + archive_te_rank) +
            0.5 / (k + history_test_rank)
        )
        add_candidate(
            f"{archive_name}_history_threeway_rrf_k{int(k)}",
            va,
            te,
            archive_name,
        )

best_name = max(candidates, key=lambda name: candidates[name][0])
_, best_metrics, best_valid = candidates[best_name]
best_test = candidate_test[best_name]
raw_source = candidate_source[best_name]

if raw_source in experts:
    raw_valid = experts[raw_source][0]
else:
    raw_valid = history_valid

print("CANDIDATES " + json.dumps(
    {name: score for name, (score, _, _) in candidates.items()},
    sort_keys=True,
))
print("FINDINGS " + json.dumps({
    "best_candidate": best_name,
    "history_features_used": history_keys,
    "history_feature_count": len(history_keys),
    "archived_pairs_found": len(archived),
    "archived_incumbent_rank_correlations": archive_findings,
    "history_incumbent_rank_correlation": rank_correlation(
        inc_valid_borda, history_valid_borda
    ),
    "mechanism": "fixed Borda and top-heavy reciprocal-rank aggregation",
}, sort_keys=True))

out = os.environ.get("ITER_OUT")
if out:
    os.makedirs(out, exist_ok=True)
    np.save(
        os.path.join(out, "scores_valid.npy"),
        np.asarray(best_valid, dtype=np.float64),
    )
    np.save(
        os.path.join(out, "scores_test.npy"),
        np.asarray(best_test, dtype=np.float64),
    )
    if best_name != "history_feedback_consensus_standalone":
        np.save(
            os.path.join(out, "scores_valid_raw.npy"),
            np.asarray(raw_valid, dtype=np.float64),
        )

elapsed = time.time() - START
print("METRICS " + json.dumps({
    "primary": float(best_metrics["primary"]),
    "gauc": float(best_metrics["gauc"]),
    "ndcg@5": float(best_metrics["ndcg@5"]),
    "gpu_seconds": float(elapsed),
}))