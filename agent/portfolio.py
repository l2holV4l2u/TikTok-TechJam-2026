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


# A slot gets this many turns without beating its own best before it is archived and its place
# reused. Deliberately below the run-level patience: a lineage should be recycled well before
# the run itself is at risk of ending.
SLOT_PATIENCE = 2
# Refill alternates: three fresh drafts, then one revival from the archive. Exploration is the
# default because a fresh draft is the only thing that can find a family nothing has tried;
# revival exists so a line that stalled with an idea left in it is not lost.
FRESH_PER_REVIVE = 3
# How hard to punish a revival candidate for agreeing with what is already live. Reviving the
# top scorer when it correlates 0.97 with a running slot spends a slot to learn nothing.
LAMBDA_CORR = 0.5


@dataclass
class Slot:
    """One lineage. `parent` is the node its next script edits."""

    slot_id: int
    parent: object | None = None          # agent.tree.Node
    stale: int = 0                        # turns without beating this slot's own best
    best: float = float("-inf")
    origin: str = "fresh"                 # "fresh" | "revived"
    seed_note: str = ""                   # note this slot was refilled with
    lineage: list[int] = field(default_factory=list)   # iter_ids it has produced
    last_hypothesis: str = ""
    last_valid_scores: object | None = None           # np.ndarray, post-blend, for the blend pool
    last_test_scores: object | None = None            # np.ndarray, for the portfolio blend
    # This slot's OWN model, before retain_or_blend folds it toward the incumbent. The
    # correlation gate must read this: post-blend arrays answer "how similar are the
    # harness's retention decisions", not "do the lineages disagree".
    last_candidate_scores: object | None = None       # np.ndarray, pre-blend, for correlation
    # A revived slot must start from the node it was revived onto, not from whatever
    # tree.select would hand it; cleared once it has been used for one proposal.
    pending_parent: object | None = None
    # Crash and rejection feedback belongs to the lineage that produced it. Shared across the
    # portfolio it would tell a slot to "fix" a traceback from a script it never wrote, and
    # to keep a hypothesis that was never its own.
    feedback: str | None = None


@dataclass
class ArchiveEntry:
    """A retired lineage, kept for two different reasons.

    As a REVIVAL source it is a starting point with an idea left in it. As an ENSEMBLE MEMBER
    its stored predictions are the more valuable half: a set of converged, decorrelated models
    is exactly the input the harness blender has never had, and the correlation ceiling is the
    measured reason blends on this benchmark gain nothing.
    """

    entry_id: int
    slot_id: int
    turn_retired: int
    iter_id: int | None                   # the tree node this lineage ended on
    hypothesis: str
    primary: float
    note: str
    valid_path: str | None = None
    test_path: str | None = None

    def to_json(self) -> dict:
        return {"entry_id": self.entry_id, "slot_id": self.slot_id,
                "turn_retired": self.turn_retired, "iter_id": self.iter_id,
                "hypothesis": self.hypothesis, "primary": self.primary, "note": self.note,
                "valid_path": self.valid_path, "test_path": self.test_path}


@dataclass
class Archive:
    """Retired lineages, their predictions, and why each one stopped."""

    run_dir: Path
    entries: list[ArchiveEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)

    def _dir(self) -> Path:
        d = Path(self.run_dir) / "archive"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def add(self, slot: Slot, turn: int, note: str,
            valid_scores=None, test_scores=None) -> ArchiveEntry:
        entry_id = len(self.entries)
        valid_path = test_path = None
        if valid_scores is not None:
            valid_path = str(self._dir() / f"entry_{entry_id}_valid.npy")
            np.save(valid_path, np.asarray(valid_scores, dtype=np.float64))
        if test_scores is not None:
            test_path = str(self._dir() / f"entry_{entry_id}_test.npy")
            np.save(test_path, np.asarray(test_scores, dtype=np.float64))
        entry = ArchiveEntry(
            entry_id=entry_id, slot_id=slot.slot_id, turn_retired=turn,
            iter_id=slot.lineage[-1] if slot.lineage else None,
            hypothesis=slot.last_hypothesis, primary=slot.best, note=note,
            valid_path=valid_path, test_path=test_path)
        self.entries.append(entry)
        with (Path(self.run_dir) / "archive.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_json(), default=float) + "\n")
        return entry

    def top(self, k: int = 5) -> list[ArchiveEntry]:
        return sorted(self.entries, key=lambda e: -e.primary)[:k]

    def valid_scores(self, entry: ArchiveEntry):
        if not entry.valid_path or not Path(entry.valid_path).exists():
            return None
        return np.load(entry.valid_path, allow_pickle=False)

    def test_scores(self, entry: ArchiveEntry):
        if not entry.test_path or not Path(entry.test_path).exists():
            return None
        return np.load(entry.test_path, allow_pickle=False)

    def summary(self, k: int = 5) -> str:
        if not self.entries:
            return ""
        rows = [f"  #{e.entry_id} (turn {e.turn_retired}, primary {e.primary:.4f}) "
                f"{e.hypothesis[:90]}" for e in self.top(k)]
        return "LINES ALREADY RETIRED THIS RUN:\n" + "\n".join(rows)

    def best_revival(self, live_scores: dict, user_id,
                     lam: float = LAMBDA_CORR) -> ArchiveEntry | None:
        """argmax(primary - lam * max within-user rank correlation with any live slot).

        Not simply the top scorer. A stored line that ranks validation the way a running slot
        already does is a slot spent on a result the portfolio already holds.
        """
        if not self.entries:
            return None
        from .ensemble import _within_user_rank

        live = [np.asarray(v, dtype=np.float64)
                for v in (live_scores or {}).values() if v is not None]
        if not live:
            return max(self.entries, key=lambda e: e.primary)
        live_ranked = [_within_user_rank(user_id, v) for v in live]
        best, best_value = None, -float("inf")
        for entry in self.entries:
            scores = self.valid_scores(entry)
            penalty = 0.0
            if scores is not None and len(scores) == len(user_id):
                mine = _within_user_rank(user_id, scores)
                corrs = [abs(float(np.corrcoef(mine, other)[0, 1]))
                         for other in live_ranked
                         if mine.std() > 0 and other.std() > 0]
                penalty = max(corrs) if corrs else 1.0
            value = entry.primary - lam * penalty
            if value > best_value:
                best, best_value = entry, value
        return best


@dataclass
class RefillState:
    """Where the explore/exploit alternation currently stands."""

    fresh_since_revive: int = 0
    revivals: int = 0
    fresh: int = 0


def refill(slot_id: int, archive: Archive, live_scores: dict, user_id, state: RefillState,
           tree=None, fresh_per_revive: int = FRESH_PER_REVIVE,
           exhausted: bool = False) -> tuple[Slot, dict]:
    """Replace a retired slot: a fresh draft by default, a revival every Nth refill.

    Returns (slot, reason) so the choice is explainable in the run log rather than looking
    like a lost lineage.
    """
    want_revival = bool(archive) and (state.fresh_since_revive >= fresh_per_revive or exhausted)
    if want_revival:
        entry = archive.best_revival(live_scores, user_id)
        node = tree.get(entry.iter_id) if (tree is not None and entry is not None
                                           and entry.iter_id is not None) else None
        if entry is not None:
            state.fresh_since_revive = 0
            state.revivals += 1
            return (Slot(slot_id=slot_id, parent=node, pending_parent=node, origin="revived",
                         seed_note=entry.note, best=float("-inf")),
                    {"choice": "revived", "entry_id": entry.entry_id,
                     "entry_primary": entry.primary, "exhausted": exhausted})
    state.fresh_since_revive += 1
    state.fresh += 1
    return (Slot(slot_id=slot_id, origin="fresh"),
            {"choice": "fresh", "fresh_since_revive": state.fresh_since_revive})


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
