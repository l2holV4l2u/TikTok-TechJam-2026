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


class ArrayEvaluator(StdoutJsonEvaluator):
    """Scores from stdout, plus the prediction arrays the correlation gate needs.

    StdoutJsonEvaluator exposes no `last_scores`, so a portfolio driven by it has nothing to
    correlate -- correct, but it means the gate cannot be exercised without something that
    supplies arrays. `spread` controls how much the slots disagree, which is the axis the whole
    Phase 2 decision turns on.
    """

    def __init__(self, spread: float = 0.0, n_rows: int | None = None):
        self.spread = spread
        self.calls = 0
        self.last_scores = None
        self.last_test_scores = None
        self._n = n_rows

    def _rows(self) -> int:
        if self._n is None:
            try:
                from pipeline.data import load
                self._n = len(load("valid").user_id)
            except Exception:
                self._n = 64
        return self._n

    def evaluate(self, result, iter_out=None):
        metrics = super().evaluate(result, iter_out)
        if metrics is None:
            self.last_scores = self.last_test_scores = None
            return None
        n = self._rows()
        base = np.random.default_rng(0).random(n)
        noise = np.random.default_rng(1000 + self.calls).random(n)
        self.last_scores = (1.0 - self.spread) * base + self.spread * noise
        self.last_test_scores = self.last_scores[: max(1, n // 2)]
        self.calls += 1
        return metrics


def _run(scores, slots, tmp, evaluator=None, **kw):
    led = _ledger(tmp)
    pr = SlotProposer(scores)
    r = run_loop(pr, evaluator or StdoutJsonEvaluator(), led,
                 workdir=Path(tmp) / "scripts",
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


def test_siblings_are_disclosed_to_every_slot_and_never_to_itself():
    with tempfile.TemporaryDirectory() as tmp:
        pr, _, _ = _run([0.61] * 9, 3, tmp)
        # turn 1 has nothing to disclose; from turn 2 each slot sees the other two
        later = pr.siblings[3:6]
        assert later and all(s for s in later), f"siblings missing after turn 1: {later}"
        for slot_id, text in enumerate(later):
            assert f"slot {slot_id}:" not in text, f"slot {slot_id} was told about itself"
            others = {0, 1, 2} - {slot_id}
            for o in others:
                assert f"slot {o}:" in text, f"slot {slot_id} was not told about slot {o}"


def _portfolio_records(tmp):
    path = Path(tmp) / "portfolio.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_correlation_is_logged_once_per_turn():
    with tempfile.TemporaryDirectory() as tmp:
        _run([0.61] * 9, 3, tmp, evaluator=ArrayEvaluator(spread=0.9))
        recs = _portfolio_records(tmp)
        assert recs, "no portfolio record written"
        turns = [x["turn"] for x in recs]
        assert turns == sorted(set(turns)), f"one record per turn, got {turns}"
        for rec in recs:
            assert set(rec["correlation"]["pairs"]) == {"0-1", "0-2", "1-2"}, rec
            assert len(rec["scripts"]) == 3


def test_identical_slots_raise_the_alert_flag():
    """The gate has to fire on the case it exists for: three copies of one agent."""
    with tempfile.TemporaryDirectory() as tmp:
        _, _, r = _run([0.61] * 9, 3, tmp, evaluator=ArrayEvaluator(spread=0.0))
        recs = _portfolio_records(tmp)
        assert recs and all(rec["alert"] for rec in recs), recs
        assert r.mean_slot_correlation is not None
        assert r.mean_slot_correlation > CORR_ALERT, r.mean_slot_correlation


def test_disagreeing_slots_do_not_raise_the_alert():
    """The complement: the gate must not fire on a portfolio that is actually working."""
    with tempfile.TemporaryDirectory() as tmp:
        _, _, r = _run([0.61] * 9, 3, tmp, evaluator=ArrayEvaluator(spread=1.0))
        recs = _portfolio_records(tmp)
        assert recs and not any(rec["alert"] for rec in recs), recs
        assert r.mean_slot_correlation < CORR_ALERT, r.mean_slot_correlation


def test_one_slot_writes_no_portfolio_log():
    with tempfile.TemporaryDirectory() as tmp:
        _run([0.61] * 4, 1, tmp, evaluator=ArrayEvaluator(spread=0.5))
        assert not (Path(tmp) / "portfolio.jsonl").exists(), (
            "a single slot has no pair to correlate and must not write an empty log")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print(f"{len(tests)} tests passed")
