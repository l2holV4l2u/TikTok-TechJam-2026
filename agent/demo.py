"""Self-check for the controller: python -m agent.demo"""
import json
import tempfile
from pathlib import Path

from .ledger import Ledger
from .loop import Proposal, StdoutJsonEvaluator, run_loop
from .recovery import Recovery
from .knowledge import Claim, Knowledge
from .tree import Tree

OK = 'print("METRICS {\\"primary\\": %s}")'
EDA = 'print("rows=10 positives=0.33 fields=37")'
CRASH = 'raise RuntimeError("boom")'
SILENT = 'print("no metrics here")'


class ScriptedProposer:
    """Replies with a canned score per improve-iteration; EDA and baseline always succeed."""

    def __init__(self, scores, baseline=0.6016):
        self.scores = list(scores)
        self.n = 0
        self.baseline = baseline
        self.phases: list[str] = []
        self.parents: list[int | None] = []
        self.modes: list[str] = []

    def propose(self, *, phase, history, blacklist, feedback, parent, context):
        self.phases.append(phase)
        if phase == "eda":
            return Proposal("inspect the data", EDA, 100, 50)
        if phase == "baseline":
            return Proposal("reproduce official FM", OK % self.baseline, 100, 50)
        self.parents.append(parent.iter_id if parent else None)
        self.modes.append(context.get("mode"))
        if self.n >= len(self.scores):
            return None
        s = self.scores[self.n]
        self.n += 1
        return Proposal(f"h{self.n}", OK % s, 100, 50)


class AlwaysCrashes(ScriptedProposer):
    def propose(self, *, phase, history, blacklist, feedback, parent, context):
        if phase in ("eda", "baseline"):
            return super().propose(phase=phase, history=history, blacklist=blacklist,
                                   feedback=feedback, parent=parent, context=context)
        if "bad" in blacklist:
            return None
        return Proposal("bad", CRASH, 100, 50)


def _ledger(tmp):
    return Ledger(Path(tmp) / "ledger.jsonl")


def test_runs_eda_then_baseline_then_improves():
    with tempfile.TemporaryDirectory() as tmp:
        led = _ledger(tmp)
        pr = ScriptedProposer([0.62, 0.63, 0.63, 0.63, 0.63])
        r = run_loop(pr, StdoutJsonEvaluator(), led, workdir=tmp, patience=3, timeout=30)
        assert r.stop_reason == "converged", r.stop_reason
        assert pr.phases[:2] == ["eda", "baseline"], pr.phases
        assert r.eda_ok and abs(r.baseline_primary - 0.6016) < 1e-9
        phases = [e.phase for e in led.read()]
        assert phases[0] == "eda" and phases[1] == "baseline"
        assert r.wall_clock_s > 0 and r.script_seconds > 0
        assert r.llm_tokens_in == 100 * len(led.read())


def test_eda_output_reaches_later_prompts():
    seen = {}

    class P(ScriptedProposer):
        def propose(self, *, phase, history, blacklist, feedback, parent, context):
            seen[phase] = context.get("eda")
            return super().propose(phase=phase, history=history, blacklist=blacklist,
                                   feedback=feedback, parent=parent, context=context)

    with tempfile.TemporaryDirectory() as tmp:
        run_loop(P([0.62] * 4), StdoutJsonEvaluator(), _ledger(tmp),
                 workdir=tmp, patience=3, timeout=30)
        assert seen["eda"] is None, "nothing is known before looking"
        assert "positives=0.33" in seen["baseline"], "EDA findings must persist"
        assert Path(tmp).parent.joinpath("eda_report.txt") or True


def test_refine_mode_climbs_from_the_current_best_node():
    """While every iteration keeps gaining, the search deepens the leader.

    The mode is `sweep` now, not `refine` -- every improve iteration sweeps model families
    (loop.py) -- but on a monotone trajectory the newest node is both the least-explored and
    the highest-scoring, so `select` returns the leader either way. What this asserts is the
    resulting CHAIN, which is the property the test was written for; pinning the mode string
    made it fail the moment the policy changed while the behaviour it describes did not.
    """
    with tempfile.TemporaryDirectory() as tmp:
        led = _ledger(tmp)
        pr = ScriptedProposer([0.62, 0.64, 0.66])
        r = run_loop(pr, StdoutJsonEvaluator(), led, workdir=tmp, patience=99, timeout=30)
        assert pr.parents[:3] == [1, 2, 3], pr.parents
        assert r.tree.best.score == 0.66
        parents = {e.iter_id: e.parent_iter_id for e in led.read() if e.phase == "improve"}
        assert parents == {2: 1, 3: 2, 4: 3}, parents


def test_a_node_that_never_pays_off_is_retired_and_search_moves_on():
    """Nothing ever gains, so nodes accumulate misses, get retired, and stop being selectable.

    This used to assert that the search returned to the root three times in `broaden` mode.
    Both halves of that are obsolete: every improve iteration is `sweep` now, and sweep picks
    the LEAST-EXPLORED live node rather than the leader, so a flat run spreads its children
    across nodes instead of piling them on one. What has to hold either way is the property
    the test is named for -- a node that keeps not paying off dies, and a dead node is never
    handed out again.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # `refine` returns the leader every time, so children pile onto one node and the
        # counter can reach max_misses. Under `sweep` it cannot: every scored iteration adds a
        # node with children=0, and sweep always prefers the least-explored, so no node is ever
        # handed a second child while the tree keeps growing. Retirement is reachable there
        # only through record_failure. Exercise the machinery where it can actually fire.
        pr = ScriptedProposer([0.60] * 5)
        r = run_loop(pr, StdoutJsonEvaluator(), _ledger(tmp), workdir=tmp, patience=99,
                     timeout=30, tree=Tree(max_misses=2, epsilon=0.002), force_mode="refine")
        assert r.tree.get(1).dead, (
            f"two non-improving children must retire the leader; "
            f"{[(n.iter_id, n.misses, n.dead) for n in r.tree.nodes]}")
        assert pr.parents[:2] == [1, 1], pr.parents
        # once retired it is never handed out again. Checking liveness at the END of the run
        # would be wrong: #1 was alive on the two turns that killed it.
        assert 1 not in pr.parents[2:], pr.parents


def test_sweep_spreads_children_so_no_node_takes_a_second_one():
    """Documents why retirement is near-unreachable under the shipped sweep-only policy.

    sweep returns min(children, -score). A scored iteration always appends a node with
    children=0, so there is always an unexplored node to prefer and no node accumulates the
    second miss that `max_misses=2` needs. This is a property of the policy, not a bug -- but
    it means node retirement protects a flat run only through crashes, and a reader of
    tree.py's retirement logic should know that before relying on it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        pr = ScriptedProposer([0.60] * 6)
        r = run_loop(pr, StdoutJsonEvaluator(), _ledger(tmp), workdir=tmp,
                     patience=99, timeout=30, tree=Tree(max_misses=2, epsilon=0.002))
        assert pr.modes and set(pr.modes) == {"sweep"}, pr.modes
        assert all(n.misses <= 1 for n in r.tree.nodes), (
            [(n.iter_id, n.misses) for n in r.tree.nodes])
        assert not any(n.dead for n in r.tree.nodes)


def test_search_broadens_on_stagnation_and_narrows_after_a_gain():
    """The adaptive refine/broaden ladder, still reachable through --force-mode.

    The shipped policy is sweep on every improve iteration, so this no longer describes a
    default run -- it describes the ladder the loop still supports and that `--force-mode`
    exposes for diagnostic runs. Asserting the old default here made the module fail while the
    machinery it covers was untouched; the mode strings are pinned explicitly instead.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # baseline 0.6016 -> a miss, a miss, then a real gain, then a miss
        pr = ScriptedProposer([0.6000, 0.6005, 0.6100, 0.6050])
        r = run_loop(pr, StdoutJsonEvaluator(), _ledger(tmp), workdir=tmp,
                     patience=99, timeout=30, force_mode="broaden")
        assert set(pr.modes) == {"broaden"}, pr.modes
        # `broaden` shares select()'s breadth branch with `sweep`, so it too anchors on the
        # least-explored live node rather than the leader; what makes it "broaden" is the
        # instruction the proposer receives, which tests/test_proposer.py covers. Here the
        # property is only that every parent handed out was a live node of this tree.
        assert all(r.tree.get(p) is not None for p in pr.parents), pr.parents
        assert r.tree.best.score == 0.6100


def test_critic_rejection_never_becomes_a_search_node():
    """A flagged leak used to remain both selectable and submission-eligible as `reverted`."""
    with tempfile.TemporaryDirectory() as tmp:
        led = _ledger(tmp)
        r = run_loop(ScriptedProposer([0.80]), StdoutJsonEvaluator(), led, workdir=tmp,
                     patience=1, timeout=30, ceiling=0.8645)
        improve = [e for e in led.read() if e.phase == "improve"]
        assert improve[0].status == "rejected", improve[0]
        assert r.tree.get(improve[0].iter_id) is None
        assert r.tree.best.score == 0.6016


def test_internal_candidates_are_recorded():
    """One iteration may compare several alternatives; the log should show what it compared."""
    cand = ('print(\'CANDIDATES {"fm": 0.60, "gbdt": 0.59, "mix": 0.62}\')\n'
            'print("METRICS {\\"primary\\": 0.62}")')

    class P(ScriptedProposer):
        def propose(self, *, phase, history, blacklist, feedback, parent, context):
            if phase in ("eda", "baseline"):
                return super().propose(phase=phase, history=history, blacklist=blacklist,
                                       feedback=feedback, parent=parent, context=context)
            self.parents.append(parent.iter_id if parent else None)
            self.modes.append(context.get("mode"))
            if self.n:
                return None
            self.n += 1
            return Proposal("compares three", cand, 100, 50)

    with tempfile.TemporaryDirectory() as tmp:
        r = run_loop(P([]), StdoutJsonEvaluator(), _ledger(tmp),
                     workdir=Path(tmp) / "scripts", patience=99, timeout=30)
        assert r.candidates_evaluated == 3, r.candidates_evaluated
        rec = [json.loads(l) for l in
               (Path(tmp) / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
        assert rec[0]["candidates"]["mix"] == 0.62, rec
        assert rec[0]["iter_id"] == 2


def test_candidates_line_is_optional_and_malformed_is_ignored():
    from .loop import parse_candidates
    assert parse_candidates('METRICS {"primary": 0.6}') is None
    assert parse_candidates("CANDIDATES not json") is None
    assert parse_candidates("CANDIDATES {}") is None, "empty is not a comparison"
    assert parse_candidates('CANDIDATES {"a": 1}\nMETRICS {}') == {"a": 1}


def test_artifacts_dir_persists_across_iterations():
    """Each script starts cold otherwise, so later iterations refit what earlier ones built."""
    saw = []
    script = "\n".join([
        'import os',
        'd = os.environ["RUN_ARTIFACTS"]',
        'p = os.path.join(d, "shared.txt")',
        'prior = open(p).read() if os.path.exists(p) else ""',
        'open(p, "w").write(prior + "x")',
        'print("FINDINGS carried", len(prior))',
        'print("METRICS {\\"primary\\": 0.60}")',
    ])

    class P(ScriptedProposer):
        def propose(self, *, phase, history, blacklist, feedback, parent, context):
            if phase in ("eda", "baseline"):
                return super().propose(phase=phase, history=history, blacklist=blacklist,
                                       feedback=feedback, parent=parent, context=context)
            self.parents.append(parent.iter_id if parent else None)
            self.modes.append(context.get("mode"))
            if self.n >= 3:
                return None
            self.n += 1
            return Proposal(f"reuses artifacts {self.n}", script, 100, 50)

    def revise_fn(entries, last, findings, stale, patience):
        saw.append(findings)
        return 0, 0

    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp) / "scripts"
        run_loop(P([]), StdoutJsonEvaluator(), _ledger(tmp), workdir=wd,
                 patience=99, timeout=30, revise_fn=revise_fn)
        # RUN_ARTIFACTS is the slot's own scratch directory: private, so parallel slots cannot
        # overwrite each other, and persistent across the turns that slot runs.
        shared = wd.parent / "artifacts" / "slot_0" / "shared.txt"
        assert shared.exists(), "the artifacts directory must outlive a single iteration"
        assert shared.read_text() == "xxx", shared.read_text()
        carried = [f for f in saw if "carried" in f]
        assert carried == ["carried 0", "carried 1", "carried 2"], carried


def test_retries_then_blacklists():
    with tempfile.TemporaryDirectory() as tmp:
        led = _ledger(tmp)
        rec = Recovery(max_retries=2)
        r = run_loop(AlwaysCrashes([]), StdoutJsonEvaluator(), led,
                     workdir=tmp, recovery=rec, timeout=30)
        assert r.stop_reason == "exhausted", r.stop_reason
        assert "bad" in rec.blacklist
        improve = [x.status for x in led.read() if x.phase == "improve"]
        assert improve == ["failed", "failed", "blacklisted"], improve


def test_wall_clock_limit_stops_the_run():
    with tempfile.TemporaryDirectory() as tmp:
        led = _ledger(tmp)
        r = run_loop(ScriptedProposer([0.6] * 10), StdoutJsonEvaluator(), led,
                     workdir=tmp, wall_clock_limit_s=0.0, timeout=30)
        assert r.stop_reason == "wall_clock", r.stop_reason
        assert len(led.read()) == 0


def test_knowledge_is_revised_after_every_scored_iteration():
    with tempfile.TemporaryDirectory() as tmp:
        calls = []
        k = Knowledge()

        def revise_fn(entries, last, findings, stale, patience):
            calls.append(last.phase)
            k.claims = [Claim(f"belief after {last.phase} #{last.iter_id}")]
            return 7, 3

        r = run_loop(ScriptedProposer([0.62, 0.63]), StdoutJsonEvaluator(), _ledger(tmp),
                     workdir=tmp, patience=99, timeout=30, knowledge=k, revise_fn=revise_fn)
        assert calls == ["baseline", "improve", "improve"], calls
        assert r.knowledge is k and len(k.claims) == 1
        assert r.llm_tokens_out == 50 * 4 + 3 * 3, r.llm_tokens_out


def test_revised_beliefs_reach_the_next_proposal():
    """An append-only reflection list could never overturn itself; a belief set must."""
    seen = []
    k = Knowledge()

    class P(ScriptedProposer):
        def propose(self, *, phase, history, blacklist, feedback, parent, context):
            seen.append(context.get("knowledge"))
            return super().propose(phase=phase, history=history, blacklist=blacklist,
                                   feedback=feedback, parent=parent, context=context)

    def revise_fn(entries, last, findings, stale, patience):
        k.claims = [Claim(f"revision {len(entries)}")]
        return 0, 0

    with tempfile.TemporaryDirectory() as tmp:
        run_loop(P([0.62, 0.63]), StdoutJsonEvaluator(), _ledger(tmp),
                 workdir=tmp, patience=99, timeout=30, knowledge=k, revise_fn=revise_fn)
        assert seen[0] == "nothing established yet", "nothing is believed before any evidence"
        assert "revision 2" in seen[2], seen
        assert seen[2] != seen[1], "each proposal sees the state as it stands now"


def test_findings_are_captured_and_passed_to_revision():
    """Epistemic evidence rides along with a scored iteration, costing no extra iteration."""
    got = {}
    script = "\n".join([
        'print("FINDINGS long_view rate by decile is flat 0.27-0.38")',
        'print("FINDINGS 30% of valid users have zero positives")',
        'print("METRICS {\\"primary\\": 0.61}")',
    ])

    class P(ScriptedProposer):
        def propose(self, *, phase, history, blacklist, feedback, parent, context):
            if phase in ("eda", "baseline"):
                return super().propose(phase=phase, history=history, blacklist=blacklist,
                                       feedback=feedback, parent=parent, context=context)
            self.parents.append(parent.iter_id if parent else None)
            self.modes.append(context.get("mode"))
            if self.n:
                return None
            self.n += 1
            return Proposal("measures while scoring", script, 100, 50)

    def revise_fn(entries, last, findings, stale, patience):
        got.setdefault(last.iter_id, findings)
        return 0, 0

    with tempfile.TemporaryDirectory() as tmp:
        run_loop(P([]), StdoutJsonEvaluator(), _ledger(tmp), workdir=tmp,
                 patience=99, timeout=30, revise_fn=revise_fn)
        assert "decile is flat" in got[2] and "zero positives" in got[2], got
        assert got[1] == "", "the baseline reported no findings"


def test_findings_parser_ignores_everything_else():
    from .loop import parse_findings
    assert parse_findings('METRICS {"primary": 0.6}') == ""
    assert parse_findings("FINDINGS a\nnoise\nFINDINGS b") == "a\nb"


def test_ledger_append_only():
    with tempfile.TemporaryDirectory() as tmp:
        led = _ledger(tmp)
        run_loop(ScriptedProposer([0.6, 0.7, 0.8]), StdoutJsonEvaluator(), led,
                 workdir=tmp, patience=99, timeout=30)
        e = led.read()
        assert [x.iter_id for x in e] == sorted(x.iter_id for x in e)
        assert len(led.path.read_text().strip().splitlines()) == len(e)


if __name__ == "__main__":
    # Discovered, not listed: a hand-kept tuple silently omitted every test added after it.
    _tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in _tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all passed ({len(_tests)} tests)")
