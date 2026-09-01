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


def within_user_rank(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    start_positions = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    local_positions = (
        np.arange(n, dtype=np.float32)
        - start_positions.astype(np.float32)
    )

    ends = np.empty(n, dtype=bool)
    ends[-1] = True
    ends[:-1] = sorted_users[:-1] != sorted_users[1:]
    end_positions = np.flatnonzero(ends)
    sizes = np.diff(np.r_[-1, end_positions]).astype(np.float32)

    group_indices = np.cumsum(starts, dtype=np.int64) - 1
    denominators = np.maximum(sizes[group_indices] - 1.0, 1.0)

    result = np.empty(n, dtype=np.float32)
    result[order] = local_positions / denominators
    return result


def descending_positions(user_ids, scores):
    user_ids = np.asarray(user_ids, dtype=np.int64)
    scores = np.asarray(scores)
    n = len(scores)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, -scores, user_ids))
    sorted_users = user_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_users[1:] != sorted_users[:-1]

    start_positions = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    positions_sorted = np.arange(n, dtype=np.int64) - start_positions

    positions = np.empty(n, dtype=np.int32)
    positions[order] = positions_sorted.astype(np.int32)
    return positions


def previous_higher_ranked_occurrences(user_ids, entity_ids, positions):
    """
    For every row, count how many rows from the same user and entity have a
    better incumbent rank. This forms a slate-level redundancy signal without
    using any evaluation labels.
    """
    user_ids = np.asarray(user_ids, dtype=np.int64)
    entity_ids = np.asarray(entity_ids, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int32)

    n = len(user_ids)
    row = np.arange(n, dtype=np.int64)

    order = np.lexsort((row, positions, entity_ids, user_ids))
    sorted_users = user_ids[order]
    sorted_entities = entity_ids[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    starts[1:] = (
        (sorted_users[1:] != sorted_users[:-1])
        | (sorted_entities[1:] != sorted_entities[:-1])
    )

    start_positions = np.maximum.accumulate(
        np.where(starts, np.arange(n, dtype=np.int64), 0)
    )
    occurrences_sorted = np.arange(n, dtype=np.int64) - start_positions

    # Entity zero denotes unknown, not a genuine common entity.
    occurrences_sorted[sorted_entities == 0] = 0

    occurrences = np.empty(n, dtype=np.int16)
    occurrences[order] = np.minimum(
        occurrences_sorted, np.iinfo(np.int16).max
    ).astype(np.int16)
    return occurrences


def build_components(split, base_scores, needed_fields):
    uid = np.asarray(split.user_id, dtype=np.int64)
    base_rank = within_user_rank(uid, base_scores)
    position = descending_positions(uid, base_rank)

    # Changes are intentionally concentrated near the head. This makes the
    # correction nDCG-oriented while leaving almost all pairwise comparisons
    # (and hence GAUC) unchanged.
    head_gate = np.exp(
        -np.asarray(position, dtype=np.float32) / 24.0
    ).astype(np.float32)
    head_gate[position >= 160] = 0.0

    occurrences = {}
    for field in needed_fields:
        occurrences[field] = previous_higher_ranked_occurrences(
            uid,
            np.asarray(split.X[field], dtype=np.int64),
            position,
        )

    return base_rank, position, head_gate, occurrences


def correction_from_spec(position, head_gate, occurrences, spec):
    correction = np.zeros(len(position), dtype=np.float32)

    for field, coefficient, transform in spec["parts"]:
        occ = np.asarray(occurrences[field], dtype=np.float32)

        if transform == "binary":
            redundancy = (occ > 0).astype(np.float32)
        elif transform == "log":
            redundancy = np.log1p(occ).astype(np.float32)
        elif transform == "linear":
            redundancy = np.minimum(occ, 5.0).astype(np.float32)
        elif transform == "after_second":
            redundancy = np.maximum(occ - 1.0, 0.0).astype(np.float32)
        elif transform == "sqrt":
            redundancy = np.sqrt(occ).astype(np.float32)
        else:
            raise ValueError("unknown transform: " + transform)

        correction -= float(coefficient) * redundancy * head_gate

    # Some families use a sharper top-only quota, structurally different from
    # the smoothly decaying MMR penalty.
    if spec.get("top_limit") is not None:
        correction[np.asarray(position) >= int(spec["top_limit"])] = 0.0

    return correction


# The family definitions differ in how redundancy enters the prediction:
# smooth logarithmic MMR, hard one-repeat quotas, delayed quotas, and
# multi-taxonomy coverage.
SPECS = {
    "soft_creator_mmr_003": {
        "parts": [("author_id", 0.003, "log")],
    },
    "soft_creator_mmr_007": {
        "parts": [("author_id", 0.007, "log")],
    },
    "soft_video_mmr_005": {
        "parts": [("video_id", 0.005, "log")],
    },
    "soft_video_creator_mmr": {
        "parts": [
            ("video_id", 0.005, "log"),
            ("author_id", 0.004, "log"),
        ],
    },
    "hard_creator_quota_top40": {
        "parts": [("author_id", 0.018, "binary")],
        "top_limit": 40,
    },
    "hard_video_quota_top40": {
        "parts": [("video_id", 0.022, "binary")],
        "top_limit": 40,
    },
    "delayed_creator_quota": {
        "parts": [("author_id", 0.012, "after_second")],
        "top_limit": 80,
    },
    "taxonomy_xquad": {
        "parts": [
            ("tag", 0.0025, "sqrt"),
            ("duration_bucket", 0.0015, "binary"),
            ("author_id", 0.0040, "log"),
        ],
    },
    "content_source_coverage": {
        "parts": [
            ("tag", 0.0015, "binary"),
            ("upload_type", 0.0015, "binary"),
            ("author_id", 0.0050, "log"),
        ],
    },
    "balanced_submodular_slate": {
        "parts": [
            ("video_id", 0.0045, "log"),
            ("author_id", 0.0045, "log"),
            ("tag", 0.0010, "sqrt"),
            ("duration_bucket", 0.0008, "binary"),
        ],
    },
}

needed_fields = sorted({
    field
    for spec in SPECS.values()
    for field, _, _ in spec["parts"]
})

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

valid = load("valid")
valid_uid = np.asarray(valid.user_id, dtype=np.int64)
valid_y = np.asarray(valid.y)

inc_valid = np.load(inc_valid_path, mmap_mode="r")
base_valid, valid_position, valid_gate, valid_occurrences = (
    build_components(valid, inc_valid, needed_fields)
)

base_metrics = evaluate(valid_uid, valid_y, base_valid)
candidate_scores = {
    "trusted_sequence_control": float(base_metrics["primary"])
}

best_name = "trusted_sequence_control"
best_primary = float(base_metrics["primary"])
best_valid = base_valid.copy()
best_correction = None

for name, spec in SPECS.items():
    correction = correction_from_spec(
        valid_position,
        valid_gate,
        valid_occurrences,
        spec,
    )
    candidate = (base_valid + correction).astype(np.float32)
    metrics = evaluate(valid_uid, valid_y, candidate)
    primary = float(metrics["primary"])
    candidate_scores[name] = primary

    print(
        "FINDINGS family=%s primary=%.6f gauc=%.6f ndcg5=%.6f "
        "delta_primary=%+.6f delta_gauc=%+.6f delta_ndcg5=%+.6f"
        % (
            name,
            primary,
            float(metrics["gauc"]),
            float(metrics["ndcg@5"]),
            primary - float(base_metrics["primary"]),
            float(metrics["gauc"]) - float(base_metrics["gauc"]),
            float(metrics["ndcg@5"]) - float(base_metrics["ndcg@5"]),
        ),
        flush=True,
    )

    if primary > best_primary:
        best_primary = primary
        best_name = name
        best_valid = candidate.copy()
        best_correction = correction.copy()

final_metrics = evaluate(valid_uid, valid_y, best_valid)

print(
    "FINDINGS winner=%s control_primary=%.6f winner_primary=%.6f"
    % (
        best_name,
        float(base_metrics["primary"]),
        float(final_metrics["primary"]),
    ),
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
    if best_correction is not None:
        np.save(
            os.path.join(OUT, "scores_valid_raw.npy"),
            np.asarray(best_correction, dtype=np.float64),
        )

del inc_valid
del base_valid
del valid_position
del valid_gate
del valid_occurrences
del best_valid
del best_correction
del valid
gc.collect()

test = load("test")
inc_test = np.load(inc_test_path, mmap_mode="r")

if best_name == "trusted_sequence_control":
    test_scores = within_user_rank(test.user_id, inc_test)
else:
    selected_spec = SPECS[best_name]
    selected_fields = sorted({
        field for field, _, _ in selected_spec["parts"]
    })
    base_test, test_position, test_gate, test_occurrences = (
        build_components(test, inc_test, selected_fields)
    )
    test_correction = correction_from_spec(
        test_position,
        test_gate,
        test_occurrences,
        selected_spec,
    )
    test_scores = (base_test + test_correction).astype(np.float32)

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