"""Several solution lineages advancing at once, under one convergence rule.

The organizers' rule is singular: "a run is converged when validation score has not improved by
more than eps over the last N consecutive iterations". There is no per-agent counter in it, so
n lineages cannot each carry one and keep going after the others stop -- the checkpoint that
produced would sit past the point where the run was over. What n lineages CAN do is share a
turn: the loop advances in turns, each turn launches one script per slot, and the turn's score
is the best of them. One curve, one counter, n times the search per unit of the three
non-improving turns the rule allows.

A slot that stops paying is therefore recycled, not stopped. That is a resource decision
inside a turn, which the rule says nothing about.

Why bother: the measured bottleneck on this benchmark is not search breadth, it is that
everything found correlates. The run logs record components at 0.94+ rank correlation, MMoE at
0.9888 against plain DeepFM, and blends that gained nothing as a result. A portfolio is worth
its cost only if its slots disagree, so `pairwise_rank_correlation` is not instrumentation
here -- it is the acceptance test for the whole design.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Above this, two slots are ranking validation the same way and the second one is buying
# nothing. Liu & Yao's negative correlation learning is the principled fix; this is the alarm.
CORR_ALERT = 0.95


@dataclass
class Slot:
    """One lineage. `parent` is the node its next script edits."""

    slot_id: int
    parent: object | None = None          # agent.tree.Node
    stale: int = 0                        # turns without beating this slot's own best
    best: float = float("-inf")
    origin: str = "fresh"                 # "fresh" | "revived"
    seed_note: str = ""                   # consultant note this slot was refilled with
    lineage: list[int] = field(default_factory=list)   # iter_ids it has produced
    last_hypothesis: str = ""
    last_valid_scores: object | None = None           # np.ndarray, for correlation


def pairwise_rank_correlation(score_arrays: dict[int, np.ndarray], user_id) -> dict:
    """Within-user rank correlation between slots' validation predictions.

    Ranks are taken inside each user, not globally, because the metric only ever compares rows
    belonging to the same user -- two models can disagree completely about absolute level and
    still produce an identical ranking, which is the case that matters here.
    """
    from .ensemble import _within_user_rank

    usable = {k: np.asarray(v, dtype=np.float64)
              for k, v in (score_arrays or {}).items() if v is not None}
    if len(usable) < 2:
        return {"pairs": {}, "mean": None, "max": None, "n_slots": len(usable)}
    ranked = {k: _within_user_rank(user_id, v) for k, v in usable.items()}
    pairs: dict[str, float] = {}
    for a, b in itertools.combinations(sorted(ranked), 2):
        x, y = ranked[a], ranked[b]
        if x.std() == 0 or y.std() == 0:
            # a constant ranking has no correlation defined; treat it as fully redundant
            # rather than emitting NaN into the mean the gate reads
            pairs[f"{a}-{b}"] = 1.0
            continue
        pairs[f"{a}-{b}"] = float(np.corrcoef(x, y)[0, 1])
    values = list(pairs.values())
    return {"pairs": pairs, "mean": float(np.mean(values)), "max": float(np.max(values)),
            "n_slots": len(usable)}


def log_portfolio(run_dir: Path, record: dict) -> None:
    """Append one line per turn to portfolio.jsonl -- the evidence the portfolio earned its cost."""
    path = Path(run_dir) / "portfolio.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=float) + "\n")


def demo() -> None:
    users = np.repeat(np.arange(50), 4)
    rng = np.random.default_rng(0)
    a = rng.random(users.size)

    same = pairwise_rank_correlation({0: a, 1: a.copy()}, users)
    assert abs(same["mean"] - 1.0) < 1e-9, same
    assert same["max"] >= CORR_ALERT, "identical slots must trip the alarm"

    b = rng.random(users.size)
    diff = pairwise_rank_correlation({0: a, 1: b}, users)
    assert abs(diff["mean"]) < 0.3, diff

    one = pairwise_rank_correlation({0: a}, users)
    assert one["mean"] is None and one["pairs"] == {}, "one slot has no pair to compare"
    assert pairwise_rank_correlation({}, users)["mean"] is None

    three = pairwise_rank_correlation({0: a, 1: b, 2: rng.random(users.size)}, users)
    assert set(three["pairs"]) == {"0-1", "0-2", "1-2"}, three["pairs"]

    flat = pairwise_rank_correlation({0: np.zeros(users.size), 1: a}, users)
    assert flat["mean"] == 1.0, "a constant ranking counts as fully redundant, not NaN"
    print("ok  portfolio.pairwise_rank_correlation")


if __name__ == "__main__":
    demo()
