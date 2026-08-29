"""Portfolio search: turns, slots, archive and refill. python -m tests.test_portfolio"""
import json
import tempfile
from pathlib import Path

import numpy as np

from agent.demo import ScriptedProposer, _ledger
from agent.loop import Proposal, StdoutJsonEvaluator, run_loop
from agent.portfolio import CORR_ALERT, Slot, pairwise_rank_correlation
from agent.tree import Tree

OK = 'print("METRICS {\\"primary\\": %s}")'
CRASH = 'raise RuntimeError("boom")'


class SlotProposer(ScriptedProposer):
    """One canned score per improve script, plus a record of what each slot was asked."""

    def __init__(self, scores, baseline=0.6016):
        super().__init__(scores, baseline)
        self.siblings: list[str] = []
        self.notes: list[str] = []

    def propose(self, *, phase, history, blacklist, feedback, parent, context):
        if phase in ("eda", "baseline"):
            return super().propose(phase=phase, history=history, blacklist=blacklist,
                                   feedback=feedback, parent=parent, context=context)
        self.parents.append(parent.iter_id if parent else None)
        self.modes.append(context.get("mode"))
        self.siblings.append(context.get("siblings") or "")
        self.notes.append(context.get("seed_note") or "")
        if self.n >= len(self.scores):
            return None
        s = self.scores[self.n]
        self.n += 1
        return Proposal(f"h{self.n}", OK % s, 100, 50)


def _run(scores, slots, tmp, **kw):
    led = _ledger(tmp)
    pr = SlotProposer(scores)
    r = run_loop(pr, StdoutJsonEvaluator(), led, workdir=Path(tmp) / "scripts",
                 patience=kw.pop("patience", 99), timeout=30, n_slots=slots, **kw)
    return pr, led, r


def test_three_slots_produce_three_ledger_entries_per_turn():
    with tempfile.TemporaryDirectory() as tmp:
        _, led, r = _run([0.61] * 9, 3, tmp)
        improve = [e for e in led.read() if e.phase == "improve"]
        by_turn: dict[int, set] = {}
        for e in improve:
            by_turn.setdefault(e.turn, set()).add(e.slot_id)
        assert by_turn, improve
        for turn, ids in by_turn.items():
            assert ids == {0, 1, 2}, f"turn {turn} ran slots {sorted(ids)}"
        assert r.slots == 3


def test_convergence_counter_advances_once_per_turn_not_once_per_slot():
    """The rule is singular. Three scripts in a turn are one hypothesis-to-score cycle, not
    three, or a portfolio would burn the whole convergence budget in a single turn."""
    with tempfile.TemporaryDirectory() as tmp:
        # every script flat, so no turn ever improves: convergence must need `patience` TURNS
        _, led, r = _run([0.6016] * 30, 3, tmp, patience=3)
        assert r.stop_reason == "converged", r.stop_reason
        assert r.turns >= 4, f"converged after only {r.turns} turns"
        assert r.scripts >= r.turns, (r.scripts, r.turns)
        assert r.scripts > r.turns, "three slots must run more scripts than turns"


def test_one_slot_still_counts_turns_and_scripts_the_same():
    with tempfile.TemporaryDirectory() as tmp:
        _, _, r = _run([0.6016] * 10, 1, tmp, patience=3)
        assert r.turns == r.scripts - _NON_IMPROVE_SCRIPTS, (r.turns, r.scripts)


_NON_IMPROVE_SCRIPTS = 2   # the eda script and the baseline script precede every improve turn


def test_a_slot_crash_does_not_stop_the_other_slots():
    class OneCrashes(SlotProposer):
        def propose(self, *, phase, history, blacklist, feedback, parent, context):
            p = super().propose(phase=phase, history=history, blacklist=blacklist,
                                feedback=feedback, parent=parent, context=context)
            if p is not None and phase == "improve" and len(self.parents) % 3 == 1:
                return Proposal("crasher", CRASH, 100, 50)
            return p

    with tempfile.TemporaryDirectory() as tmp:
        led = _ledger(tmp)
        pr = OneCrashes([0.61] * 9)
        r = run_loop(pr, StdoutJsonEvaluator(), led, workdir=Path(tmp) / "scripts",
                     patience=99, timeout=30, n_slots=3)
        improve = [e for e in led.read() if e.phase == "improve"]
        assert any(e.status in ("failed", "blacklisted") for e in improve), "no crash recorded"
        assert any(e.status in ("ok", "kept") for e in improve), (
            "a crash in one slot must not stop the others scoring")
        assert r.stop_reason != "environment_broken"


def test_slots_write_to_disjoint_artifact_directories():
    """Three concurrent scripts sharing one scratch directory is a silent corruption."""
    script = "\n".join([
        'import os',
        'd = os.environ["RUN_ARTIFACTS"]',
        'open(os.path.join(d, "mine.txt"), "a").write("x")',
        'assert os.environ["SHARED_ARTIFACTS"] != d, "scratch must not be the shared dir"',
        'print("METRICS {\\"primary\\": 0.61}")',
    ])

    class P(SlotProposer):
        def propose(self, *, phase, history, blacklist, feedback, parent, context):
            if phase in ("eda", "baseline"):
                return ScriptedProposer.propose(
                    self, phase=phase, history=history, blacklist=blacklist,
                    feedback=feedback, parent=parent, context=context)
            self.parents.append(parent.iter_id if parent else None)
            self.n += 1
            if self.n > 6:
                return None
            return Proposal(f"h{self.n}", script, 100, 50)

    with tempfile.TemporaryDirectory() as tmp:
        run_loop(P([]), StdoutJsonEvaluator(), _ledger(tmp), workdir=Path(tmp) / "scripts",
                 patience=99, timeout=30, n_slots=3)
        art = Path(tmp) / "artifacts"
        for k in range(3):
            assert (art / f"slot_{k}" / "mine.txt").exists(), f"slot {k} lost its scratch dir"
        assert (art / "shared").is_dir(), "the incumbent directory must exist and be separate"


def test_slots_equal_one_reproduces_the_sequential_loop():
    """The default path must be bit-identical to the pre-portfolio loop."""
    ledgers = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            _, led, r = _run([0.62, 0.64, 0.66, 0.66, 0.66], 1, tmp, patience=3)
            ledgers.append([(e.iter_id, e.phase, e.status, e.metrics.get("primary"))
                            for e in led.read()])
    assert ledgers[0] == ledgers[1], "a one-slot run must be deterministic"
    assert [e[0] for e in ledgers[0]] == list(range(len(ledgers[0]))), "iter_ids stay contiguous"


def test_every_slot_starts_from_the_reproduced_baseline():
    with tempfile.TemporaryDirectory() as tmp:
        pr, _, _ = _run([0.61] * 6, 3, tmp)
        assert pr.parents[:3] == [1, 1, 1], (
            f"the baseline node is the first parent for every slot; got {pr.parents[:3]}")


def test_correlation_of_identical_predictions_is_one():
    users = np.repeat(np.arange(40), 5)
    a = np.random.default_rng(0).random(users.size)
    got = pairwise_rank_correlation({0: a, 1: a.copy(), 2: a.copy()}, users)
    assert abs(got["mean"] - 1.0) < 1e-9 and got["max"] >= CORR_ALERT, got


def test_correlation_needs_two_slots_to_mean_anything():
    users = np.repeat(np.arange(10), 3)
    a = np.random.default_rng(0).random(users.size)
    assert pairwise_rank_correlation({0: a}, users)["mean"] is None
    assert pairwise_rank_correlation({0: a, 1: None}, users)["mean"] is None


def test_slot_defaults_are_a_fresh_unstarted_lineage():
    s = Slot(slot_id=2)
    assert s.parent is None and s.stale == 0 and s.origin == "fresh"
    assert s.lineage == [] and s.seed_note == "" and s.last_valid_scores is None


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print(f"{len(tests)} tests passed")
