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


def _portfolio_records(tmp, event="turn"):
    """Records of one kind from portfolio.jsonl. The log carries turn summaries and refill
    events; filtering by kind keeps a test about one from breaking when the other is added."""
    path = Path(tmp) / "portfolio.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [r for r in rows if event is None or r.get("event") == event]


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


def _archive(tmp):
    from agent.portfolio import Archive
    return Archive(run_dir=Path(tmp))


def _slot_with(tmp, slot_id=0, best=0.60, hyp="h", lineage=(1,), scores=None):
    s = Slot(slot_id=slot_id, best=best, last_hypothesis=hyp)
    s.lineage = list(lineage)
    s.last_valid_scores = scores
    return s


def test_slot_archives_after_two_non_improving_turns():
    with tempfile.TemporaryDirectory() as tmp:
        # flat scores: no slot ever beats its own best, so each hits slot_patience
        _, _, r = _run([0.6016] * 30, 3, tmp, patience=99,
                       evaluator=ArrayEvaluator(spread=0.9), slot_patience=2)
        assert r.archived >= 2, f"expected retirements, got {r.archived}"
        rows = [json.loads(l) for l in
                (Path(tmp) / "archive.jsonl").read_text().splitlines() if l.strip()]
        assert rows and all("stalled after" in x["note"] for x in rows), rows


def test_archiving_a_slot_does_not_stop_the_run():
    """Slot stagnation is a resource decision, not a stopping rule. The organizers' counter
    belongs to the run, and nothing a slot does may shorten it."""
    with tempfile.TemporaryDirectory() as tmp:
        _, _, r = _run([0.6016] * 30, 3, tmp, patience=3,
                       evaluator=ArrayEvaluator(spread=0.9), slot_patience=1)
        assert r.stop_reason == "converged", r.stop_reason
        assert r.archived >= 1, "the fixture must actually archive something"
        # convergence still took the full window of TURNS despite churn underneath it
        assert r.turns >= 4, r.turns


def test_refill_is_fresh_until_the_alternation_calls_for_a_revival():
    from agent.portfolio import RefillState, refill

    with tempfile.TemporaryDirectory() as tmp:
        arch = _archive(tmp)
        arch.add(_slot_with(tmp, best=0.605), turn=1, note="n")
        state = RefillState()
        choices = [refill(0, arch, {}, None, state, fresh_per_revive=3)[1]["choice"]
                   for _ in range(4)]
        assert choices == ["fresh", "fresh", "fresh", "revived"], choices
        assert state.revivals == 1 and state.fresh == 3


def test_refill_falls_back_to_fresh_when_the_archive_is_empty():
    from agent.portfolio import RefillState, refill

    with tempfile.TemporaryDirectory() as tmp:
        state = RefillState(fresh_since_revive=99)
        slot, why = refill(1, _archive(tmp), {}, None, state, fresh_per_revive=1)
        assert why["choice"] == "fresh" and slot.origin == "fresh"


def test_revival_prefers_a_decorrelated_entry_over_a_higher_scoring_correlated_one():
    """Reviving the top scorer when it duplicates a live slot spends a slot to learn nothing."""
    from agent.portfolio import RefillState, refill

    users = np.repeat(np.arange(60), 5)
    rng = np.random.default_rng(0)
    live_scores = rng.random(users.size)
    with tempfile.TemporaryDirectory() as tmp:
        arch = _archive(tmp)
        # higher primary, but ranks validation exactly like the running slot
        arch.add(_slot_with(tmp, slot_id=0, best=0.610), turn=1, note="twin",
                 valid_scores=live_scores.copy())
        # lower primary, independent
        arch.add(_slot_with(tmp, slot_id=1, best=0.606), turn=1, note="different",
                 valid_scores=rng.random(users.size))
        chosen = arch.best_revival({9: live_scores}, users)
        assert chosen.note == "different", (chosen.entry_id, chosen.primary, chosen.note)

        state = RefillState(fresh_since_revive=3)
        slot, why = refill(2, arch, {9: live_scores}, users, state, fresh_per_revive=3)
        assert why["choice"] == "revived" and slot.seed_note == "different"
        assert slot.origin == "revived"


def test_revival_takes_the_top_scorer_when_nothing_is_live_to_compare_against():
    with tempfile.TemporaryDirectory() as tmp:
        arch = _archive(tmp)
        arch.add(_slot_with(tmp, best=0.601), turn=1, note="low")
        arch.add(_slot_with(tmp, best=0.609), turn=1, note="high")
        assert arch.best_revival({}, None).note == "high"


def test_a_revived_slot_resumes_from_its_archived_node_not_the_search_leader():
    from agent.portfolio import RefillState, refill
    from agent.tree import Node, Tree

    tree = Tree()
    tree.add(Node(4, None, "archived line", "code", 0.604))
    tree.add(Node(9, None, "the leader", "code", 0.620))
    with tempfile.TemporaryDirectory() as tmp:
        arch = _archive(tmp)
        arch.add(_slot_with(tmp, best=0.604, lineage=(4,)), turn=1, note="resume me")
        state = RefillState(fresh_since_revive=3)
        slot, why = refill(0, arch, {}, None, state, tree=tree, fresh_per_revive=3)
        assert why["choice"] == "revived"
        assert slot.pending_parent is not None and slot.pending_parent.iter_id == 4, (
            "a revived slot must start from the line it is resuming")


def test_all_slots_archived_in_one_turn_still_refills_to_n():
    with tempfile.TemporaryDirectory() as tmp:
        _, _, r = _run([0.6016] * 24, 3, tmp, patience=99,
                       evaluator=ArrayEvaluator(spread=0.9), slot_patience=1)
        recs = _portfolio_records(tmp, event="turn")
        for rec in recs:
            assert len(rec["slots"]) == 3, f"portfolio shrank below three slots: {rec}"
        assert r.archived >= 3, r.archived


def test_archive_persists_predictions_for_the_ensemble_pool():
    """The archive's more valuable job is as a blend pool, so the arrays must survive."""
    with tempfile.TemporaryDirectory() as tmp:
        arch = _archive(tmp)
        scores = np.arange(10, dtype=np.float64)
        entry = arch.add(_slot_with(tmp), turn=2, note="n",
                         valid_scores=scores, test_scores=scores[:4])
        assert np.array_equal(arch.valid_scores(entry), scores)
        assert np.array_equal(arch.test_scores(entry), scores[:4])
        assert "retired" in arch.summary().lower()


def _consult_recorder(notes_by_slot=None, claims=None):
    """A consult_fn that records its calls and returns canned notes."""
    calls = []

    def fn(knowledge, slots, results, archive, correlation, stale, patience):
        calls.append({"slots": [s.slot_id for s in slots], "results": list(results),
                      "correlation": correlation, "stale": stale,
                      "archive_size": len(archive) if archive else 0})
        if claims is not None:
            knowledge.claims = list(claims)
        return 11, 7, dict(notes_by_slot or {})

    return fn, calls


def test_consultant_is_called_exactly_once_per_turn():
    with tempfile.TemporaryDirectory() as tmp:
        fn, calls = _consult_recorder()
        led = _ledger(tmp)
        pr = SlotProposer([0.61] * 9)
        r = run_loop(pr, ArrayEvaluator(spread=0.9), led, workdir=Path(tmp) / "scripts",
                     patience=99, timeout=30, n_slots=3, consult_fn=fn)
        assert len(calls) == r.turns, (len(calls), r.turns)
        assert all(c["slots"] == [0, 1, 2] for c in calls), calls


def test_consultant_sees_every_slots_result_for_the_turn():
    with tempfile.TemporaryDirectory() as tmp:
        fn, calls = _consult_recorder()
        run_loop(SlotProposer([0.61] * 9), ArrayEvaluator(spread=0.9), _ledger(tmp),
                 workdir=Path(tmp) / "scripts", patience=99, timeout=30, n_slots=3,
                 consult_fn=fn)
        assert calls and all(len(c["results"]) == 3 for c in calls), (
            [len(c["results"]) for c in calls])
        assert all("primary" in r for c in calls for r in c["results"]), calls[0]
        assert calls[0]["correlation"] is not None, "the correlation record must reach it"


def test_every_live_slot_receives_its_note():
    notes = {0: "slot 1 covers recency", 1: "slot 0 covers crosses", 2: "try the loss stage"}
    with tempfile.TemporaryDirectory() as tmp:
        fn, _ = _consult_recorder(notes)
        pr = SlotProposer([0.61] * 12)
        run_loop(pr, ArrayEvaluator(spread=0.9), _ledger(tmp),
                 workdir=Path(tmp) / "scripts", patience=99, timeout=30, n_slots=3,
                 consult_fn=fn)
        # turn 1 has no note yet; from turn 2 each slot carries the one addressed to it
        assert set(pr.notes[3:6]) == set(notes.values()), pr.notes[3:6]


def test_the_consultant_replaces_per_experiment_revision_not_adds_to_it():
    """One call per turn, not n. Belief revision is already about a third of a run's
    requests and Feasibility is scored on tokens."""
    with tempfile.TemporaryDirectory() as tmp:
        revisions = []
        fn, calls = _consult_recorder()
        run_loop(SlotProposer([0.61] * 9), ArrayEvaluator(spread=0.9), _ledger(tmp),
                 workdir=Path(tmp) / "scripts", patience=99, timeout=30, n_slots=3,
                 consult_fn=fn,
                 revise_fn=lambda *a: (revisions.append(a) or (0, 0)))
        improve_turns = len(calls)
        assert improve_turns > 0
        # revise_fn still runs for the baseline phase, but never for an improve turn
        assert len(revisions) <= 1, f"{len(revisions)} per-experiment revisions alongside the consultant"


def test_one_slot_keeps_the_single_trajectory_belief_revision():
    with tempfile.TemporaryDirectory() as tmp:
        revisions = []
        fn, calls = _consult_recorder()
        run_loop(SlotProposer([0.61] * 4), ArrayEvaluator(spread=0.9), _ledger(tmp),
                 workdir=Path(tmp) / "scripts", patience=99, timeout=30, n_slots=1,
                 consult_fn=fn,
                 revise_fn=lambda *a: (revisions.append(a) or (0, 0)))
        assert calls == [], "a single slot has nothing to synthesise across"
        assert len(revisions) > 1, "and must keep its own per-experiment revision"


def test_consultant_tokens_are_counted_in_the_run_total():
    with tempfile.TemporaryDirectory() as tmp:
        fn, calls = _consult_recorder()
        r = run_loop(SlotProposer([0.61] * 9), ArrayEvaluator(spread=0.9), _ledger(tmp),
                     workdir=Path(tmp) / "scripts", patience=99, timeout=30, n_slots=3,
                     consult_fn=fn)
        assert r.llm_tokens_in >= 11 * len(calls), (r.llm_tokens_in, len(calls))
        assert r.llm_tokens_out >= 7 * len(calls)


def test_crash_feedback_goes_only_to_the_slot_that_crashed():
    """Shared feedback would tell a slot to fix a traceback from a script it never wrote."""
    seen: list[tuple[int, str | None]] = []

    class Watcher(SlotProposer):
        def propose(self, *, phase, history, blacklist, feedback, parent, context):
            if phase == "improve":
                # slot order within a turn is stable, so index by call count
                seen.append((len(self.parents) % 3, feedback))
                if len(self.parents) % 3 == 0 and len(self.parents) < 3:
                    self.parents.append(parent.iter_id if parent else None)
                    self.modes.append(context.get("mode"))
                    self.siblings.append(context.get("siblings") or "")
                    self.notes.append(context.get("seed_note") or "")
                    self.n += 1
                    return Proposal("crasher", CRASH, 100, 50)
            return super().propose(phase=phase, history=history, blacklist=blacklist,
                                   feedback=feedback, parent=parent, context=context)

    with tempfile.TemporaryDirectory() as tmp:
        run_loop(Watcher([0.61] * 12), ArrayEvaluator(spread=0.9), _ledger(tmp),
                 workdir=Path(tmp) / "scripts", patience=99, timeout=30, n_slots=3)
    # slot 0 crashed on turn 1; on turn 2 only slot 0 may carry failure feedback
    turn2 = seen[3:6]
    assert turn2, seen
    got = {slot: fb for slot, fb in turn2}
    assert got.get(0), "the slot that crashed must receive its own traceback"
    assert not got.get(1) and not got.get(2), (
        f"feedback leaked to slots that did not fail: {got}")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print(f"{len(tests)} tests passed")


# ------------------------------------------------------------------ Phase 3: archive & refill


# ------------------------------------------------------------------ Phase 4: the consultant
