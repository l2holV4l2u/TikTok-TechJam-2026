"""Deterministic incumbent retention and rank blending for trusted predictions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _within_user_rank(user_id, scores) -> np.ndarray:
    """Map scores to [0,1] average rank within each query without using labels."""
    from pipeline.evaluate import _sorted_by_user

    user_id = np.asarray(user_id)
    scores = np.asarray(scores, dtype=np.float64)
    if len(user_id) == 0:
        return np.empty(0, dtype=np.float64)
    order = _sorted_by_user(user_id, scores)
    users = user_id[order]
    new = np.empty(len(users), dtype=bool)
    new[0] = True
    new[1:] = users[1:] != users[:-1]
    starts = np.flatnonzero(new)
    group = np.cumsum(new) - 1
    sizes = np.diff(np.r_[starts, len(users)])
    ordinal = np.arange(len(users)) - starts[group]
    values = scores[order]
    new_tie = np.empty(len(users), dtype=bool)
    new_tie[0] = True
    new_tie[1:] = (users[1:] != users[:-1]) | (values[1:] != values[:-1])
    tie = np.cumsum(new_tie) - 1
    tie_starts = np.flatnonzero(new_tie)
    tie_ends = np.r_[tie_starts[1:] - 1, len(users) - 1]
    average_ordinal = 0.5 * (ordinal[tie_starts] + ordinal[tie_ends])
    ranked = np.zeros(len(users), dtype=np.float64)
    ranked[order] = average_ordinal[tie] / np.maximum(sizes[group] - 1, 1)
    return ranked


def retain_or_blend(metrics: dict, evaluator, artifacts: Path, run_dir: Path,
                    iter_id: int) -> dict:
    """Select candidate/incumbent/rank-blend on validation and apply it unchanged to test."""
    candidate_valid = getattr(evaluator, "last_scores", None)
    candidate_test = getattr(evaluator, "last_test_scores", None)
    meta_path = artifacts / "incumbent.json"
    valid_path = artifacts / "incumbent_valid_scores.npy"
    test_path = artifacts / "incumbent_test_scores.npy"
    if (candidate_valid is None or candidate_test is None or not meta_path.exists()
            or not valid_path.exists() or not test_path.exists()):
        return metrics

    try:
        incumbent_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        incumbent_valid = np.asarray(np.load(valid_path, allow_pickle=False), dtype=np.float64)
        incumbent_test = np.asarray(np.load(test_path, allow_pickle=False), dtype=np.float64)
    except (OSError, ValueError, json.JSONDecodeError):
        return metrics
    if incumbent_valid.shape != candidate_valid.shape or incumbent_test.shape != candidate_test.shape:
        return metrics

    from pipeline.data import load
    from pipeline.evaluate import evaluate
    valid, test = load("valid"), load("test")
    inc_v_rank = _within_user_rank(valid.user_id, incumbent_valid)
    can_v_rank = _within_user_rank(valid.user_id, candidate_valid)

    raw_primary = float(metrics["primary"])
    options = {0.0: float(incumbent_meta["valid_primary"]), 1.0: raw_primary}
    valid_blends = {}
    for alpha in (0.25, 0.5, 0.75):
        scores = (1.0 - alpha) * inc_v_rank + alpha * can_v_rank
        valid_blends[alpha] = scores
        options[alpha] = float(evaluate(valid.user_id, valid.y, scores)["primary"])
    alpha = max(options, key=options.get)

    if alpha == 0.0:
        selected_valid, selected_test = incumbent_valid, incumbent_test
    elif alpha == 1.0:
        selected_valid, selected_test = candidate_valid, candidate_test
    else:
        inc_t_rank = _within_user_rank(test.user_id, incumbent_test)
        can_t_rank = _within_user_rank(test.user_id, candidate_test)
        selected_valid = valid_blends[alpha]
        selected_test = (1.0 - alpha) * inc_t_rank + alpha * can_t_rank

    selected_metrics = evaluate(valid.user_id, valid.y, selected_valid)
    selected_metrics.update({
        "gpu_seconds": metrics.get("gpu_seconds", 0.0),
        "device": metrics.get("device", "cpu"),
        "raw_candidate_primary": raw_primary,
        "harness_blend_alpha": float(alpha),
    })
    evaluator.last_scores = np.asarray(selected_valid, dtype=np.float64)
    evaluator.last_test_scores = np.asarray(selected_test, dtype=np.float64)
    np.save(run_dir / "scripts" / f"iter_{iter_id}_out" / "scores_valid.npy",
            evaluator.last_scores)
    np.save(run_dir / "scripts" / f"iter_{iter_id}_out" / "scores_test.npy",
            evaluator.last_test_scores)
    with (run_dir / "harness_ensembles.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"iter_id": iter_id, "selected_alpha": alpha,
                            "candidate_primary": raw_primary,
                            "incumbent_primary": options[0.0],
                            "selected_primary": selected_metrics["primary"],
                            "grid": options}) + "\n")
    return selected_metrics
