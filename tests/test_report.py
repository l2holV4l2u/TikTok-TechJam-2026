"""Report rendering, including the deliverables the spec names. python -m tests.test_report"""
import io
import json
import contextlib
import tempfile
from pathlib import Path

import numpy as np

import report_run


def _entry(iter_id, **kw):
    e = {"iter_id": iter_id, "parent_iter_id": None, "tier": 0, "hypothesis": f"h{iter_id}",
         "diff": "print(1)\n", "metrics": {"primary": 0.60}, "gpu_seconds": 1.0,
         "tokens_in": 10, "tokens_out": 5, "status": "kept", "error": None,
         "phase": "improve", "timestamp": 0.0, "slot_id": None, "turn": None}
    e.update(kw)
    return e


def _render(fn, *args, **kw) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kw)
    return buf.getvalue()


def test_report_renders_a_unified_diff_per_iteration():
    """The spec lists "the code diff applied" as a required per-iteration log field.

    The ledger has always held each script in full plus its parent id, so the diff was
    derivable; it was simply never rendered.
    """
    rows = [_entry(1, diff="a = 1\nb = 2\n"),
            _entry(2, parent_iter_id=1, diff="a = 1\nb = 99\nc = 3\n")]
    out = _render(report_run._code_diffs, Path("."), rows)
    assert "## Code diff applied" in out
    assert "**#1 -> #2**" in out
    assert "-b = 2" in out and "+b = 99" in out and "+c = 3" in out
    assert "```diff" in out


def test_an_iteration_that_changed_nothing_renders_no_empty_section():
    """An empty section reads as a deliverable that produced nothing, which is worse than
    omitting it -- the truth is that no iteration changed its parent."""
    rows = [_entry(1, diff="same\n"), _entry(2, parent_iter_id=1, diff="same\n")]
    assert _render(report_run._code_diffs, Path("."), rows) == ""


def test_diffs_are_truncated_so_one_iteration_cannot_swamp_the_report():
    rows = [_entry(1, diff="x\n" * 500),
            _entry(2, parent_iter_id=1, diff="y\n" * 500)]
    out = _render(report_run._code_diffs, Path("."), rows, max_lines=10)
    assert "more diff lines" in out
    assert out.count("\n") < 60, "a single diff must not swamp the report"


def _full_report(rows, meta) -> str:
    """Drive report_run.main() over a temporary run directory, the way a user does."""
    import sys
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        (run / "ledger.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        (run / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        argv = sys.argv
        sys.argv = ["report_run.py", str(run)]
        try:
            return _render(report_run.main)
        finally:
            sys.argv = argv


def test_report_renders_every_slot_lane():
    """A portfolio run's log has to say which lineage produced each experiment."""
    rows = [_entry(2, slot_id=0, turn=1, hypothesis="cross network"),
            _entry(3, slot_id=1, turn=1, hypothesis="recency weighting"),
            _entry(4, slot_id=2, turn=1, hypothesis="multi task heads")]
    out = _full_report(rows, {"slots": 3, "turns": 1, "scripts": 3, "iteration_cap": 50})
    assert "| # | turn | slot | phase |" in out, out[:600]
    for slot_id, hyp in enumerate(["cross network", "recency weighting", "multi task heads"]):
        assert f"| 1 | {slot_id} |" in out, f"slot {slot_id} lane missing"
        assert hyp in out


def test_a_single_slot_report_keeps_the_original_table():
    rows = [_entry(2), _entry(3)]
    out = _full_report(rows, {"slots": 1, "turns": 2, "scripts": 2, "iteration_cap": 50})
    assert "| # | phase | parent |" in out
    assert "| # | turn | slot |" not in out, "no lanes when there is one lineage"


def test_report_states_both_turn_and_script_counts():
    rows = [_entry(2, slot_id=0, turn=1), _entry(3, slot_id=1, turn=1)]
    out = _full_report(rows, {"slots": 2, "turns": 1, "scripts": 2, "iteration_cap": 50,
                              "archived": 0, "revivals": 0})
    assert "Turns:" in out and "scripts executed:" in out, out[:400]


def test_portfolio_section_reports_turns_scripts_and_correlation():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        (run / "portfolio.jsonl").write_text("\n".join(json.dumps(r) for r in [
            {"event": "turn", "turn": 1, "scripts": [2, 3, 4],
             "correlation": {"mean": 0.42, "max": 0.55, "pairs": {"0-1": 0.42}},
             "alert": False},
            {"event": "turn", "turn": 2, "scripts": [5, 6, 7],
             "correlation": {"mean": 0.97, "max": 0.99, "pairs": {"0-1": 0.97}},
             "alert": True},
            {"event": "refill", "turn": 2, "slot_id": 1, "archived_primary": 0.6041,
             "choice": "revived", "entry_id": 0, "entry_primary": 0.6039},
            {"event": "portfolio_blend", "turn": 2, "accepted": True, "pool_size": 5,
             "members": ["slot_0", "archive_1"], "fold_a_gain": 0.004,
             "fold_b_gain": 0.003, "reason": "gained on fold A and held on fold B"},
        ]), encoding="utf-8")
        meta = {"slots": 3, "turns": 2, "scripts": 6, "iteration_cap": 50,
                "archived": 1, "revivals": 1}
        out = _render(report_run._portfolio_sections, run, meta, [])

    assert "## Portfolio" in out
    assert "Turns: **2**" in out and "scripts executed: **6**" in out, out
    assert "Lineages archived: **1**" in out and "revived from the archive: **1**" in out
    # the correlation trace is the evidence the portfolio earned its cost
    assert "0.4200" in out and "0.9700" in out
    assert "**YES**" in out, "an above-threshold turn must be flagged"
    # refill and blend tables
    assert "revived" in out and "entry #0" in out
    assert "slot_0, archive_1" in out and "+0.0040" in out and "+0.0030" in out


def test_a_refused_blend_is_reported_with_its_reason():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        (run / "portfolio.jsonl").write_text(json.dumps(
            {"event": "portfolio_blend", "turn": 1, "accepted": False, "pool_size": 3,
             "members": [], "fold_a_gain": 0.19, "fold_b_gain": -0.33,
             "reason": "the blend won the selection fold and lost the confirmation fold"}),
            encoding="utf-8")
        out = _render(report_run._portfolio_sections, run, {"slots": 3}, [])
    assert "no" in out and "confirmation fold" in out
    assert "-0.3300" in out, "the losing fold-B gain must be visible, not hidden"


def test_a_single_lineage_run_reports_no_portfolio_sections():
    with tempfile.TemporaryDirectory() as tmp:
        out = _render(report_run._portfolio_sections, Path(tmp), {"slots": 1}, [])
    assert out == "", "a single-slot run's report must be unchanged"


def test_counts_are_reported_even_before_a_turn_has_been_logged():
    """The counts come from run_meta; the tables need the per-turn log. A portfolio run that
    crashed before its first turn must still report what it was configured to do."""
    with tempfile.TemporaryDirectory() as tmp:
        out = _render(report_run._portfolio_sections, Path(tmp),
                      {"slots": 3, "turns": 0, "scripts": 2, "iteration_cap": 50}, [])
    assert "## Portfolio" in out and "Lineages advanced per turn: **3**" in out
    assert "Did the lineages actually disagree?" not in out, "no log, no correlation table"


def test_report_leaves_test_predictions_unscored():
    """Generating a deliverable must not turn into a hidden-label evaluation path."""
    import sys

    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        row = _entry(1, metrics={"primary": 0.61, "gauc": 0.62, "ndcg@5": 0.60}, status="ok")
        (run / "ledger.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (run / "run_meta.json").write_text(json.dumps({"slots": 1}), encoding="utf-8")
        score_dir = run / "scripts" / "iter_1_out"
        score_dir.mkdir(parents=True)
        np.save(score_dir / "scores_test.npy", np.array([0.1, 0.2]))
        argv = sys.argv
        sys.argv = ["report_run.py", str(run)]
        try:
            out = _render(report_run.main)
        finally:
            sys.argv = argv

    assert "hidden test (this submission) | unscored" in out
    assert "computed once by the organizers" in out


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print(f"{len(tests)} tests passed")
