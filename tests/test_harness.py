"""Integrity tests for generated-code outputs and critic decisions."""
import contextlib
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

from agent.executor import RunResult
from agent.loop import SavedScoresEvaluator
from pipeline.evaluate import evaluate

USERS = np.array([1, 1, 2, 2], dtype=np.int64)
LABELS = np.array([1, 0, 0, 1], dtype=np.int8)
SCORES = np.array([0.9, 0.1, 0.2, 0.8], dtype=np.float64)


@contextlib.contextmanager
def _fake_splits():
    previous = sys.modules.get("pipeline.data")
    module = types.ModuleType("pipeline.data")
    valid = types.SimpleNamespace(user_id=USERS, y=LABELS)
    test = types.SimpleNamespace(user_id=USERS[:2], y=LABELS[:2])
    module.load = lambda name: valid if name == "valid" else test
    sys.modules["pipeline.data"] = module
    try:
        yield
    finally:
        if previous is None:
            del sys.modules["pipeline.data"]
        else:
            sys.modules["pipeline.data"] = previous


def _result(metrics: dict) -> RunResult:
    return RunResult(True, 0, "METRICS " + json.dumps(metrics), "", 0.1, False)


def test_saved_predictions_are_recomputed_and_retained():
    metrics = evaluate(USERS, LABELS, SCORES)
    with _fake_splits(), tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        np.save(out / "scores_valid.npy", SCORES)
        np.save(out / "scores_test.npy", np.array([0.3, 0.4]))
        evaluator = SavedScoresEvaluator()
        got = evaluator.evaluate(_result(metrics), out)
    assert got["primary"] == metrics["primary"]
    assert np.array_equal(evaluator.last_scores, SCORES)


def test_fabricated_metric_and_missing_test_scores_are_rejected():
    metrics = evaluate(USERS, LABELS, SCORES)
    with _fake_splits(), tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        np.save(out / "scores_valid.npy", SCORES)
        evaluator = SavedScoresEvaluator()
        assert evaluator.evaluate(_result(metrics), out) is None
        assert "Missing scores_test.npy" in evaluator.last_error
        np.save(out / "scores_test.npy", np.array([0.3, 0.4]))
        metrics["primary"] = 0.123
        assert evaluator.evaluate(_result(metrics), out) is None
        assert "does not match" in evaluator.last_error


def test_a_crashed_run_still_records_what_it_measured():
    """A run that dies mid-flight must leave run_meta.json behind.

    Cross-run memory only reads runs that have that file, so without it a crash discards every
    experiment the run actually scored. r57 lost a +0.0035 DeepFM result exactly this way when
    all three API keys hit their daily cap.
    """
    import run_agent
    from agent.memory import distil

    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d) / "runs" / "rX"
        boom = RuntimeError("all API keys exhausted their daily quota")
        original_loop, original_base = run_agent.run_loop, run_agent._dry_run_baseline
        original_argv = sys.argv
        run_agent.run_loop = lambda *a, **k: (_ for _ in ()).throw(boom)
        run_agent._dry_run_baseline = lambda: 0.5715
        sys.argv = ["run_agent", "--dry-run", "--run-dir", str(run_dir), "--no-memory"]
        try:
            run_agent.main()
        except RuntimeError as exc:
            assert exc is boom, "the crash must propagate, not be swallowed"
        else:
            raise AssertionError("the crash must propagate")
        finally:
            run_agent.run_loop, run_agent._dry_run_baseline = original_loop, original_base
            sys.argv = original_argv

        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        assert meta["crashed"] is True
        assert "daily quota" in meta["stop_reason"], meta["stop_reason"]
        assert "api_surface" in meta, "memory reads api_surface to date a run's capabilities"

        ledger = run_dir / "ledger.jsonl"
        ledger.write_text(json.dumps({"hypothesis": "deepfm over safe fields", "status": "ok",
                                      "phase": "improve", "metrics": {"primary": 0.6046}}),
                          encoding="utf-8")
        assert "deepfm over safe fields" in distil(run_dir.parent, baseline=0.6016), (
            "a crashed run's measured experiments must still reach the next run")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print(f"{len(tests)} tests passed")
