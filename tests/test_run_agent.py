"""Focused run_agent tests. Run with: python -m tests.test_run_agent"""
import contextlib
import csv
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

from agent.ledger import Entry, Ledger
from run_agent import _write_submission


class _ForbiddenY:
    def __getattribute__(self, name):
        raise AssertionError("submission generation accessed test labels")


class _FakeTestSplit:
    def __init__(self):
        self.user_id = np.asarray([10, 10, 20], dtype=np.int64)
        self.video_id = np.asarray([101, 102, 201], dtype=np.int64)

    def __len__(self):
        return len(self.user_id)

    @property
    def y(self):
        raise AssertionError("submission generation accessed test labels")


@contextlib.contextmanager
def _fake_test_data():
    prev_data = sys.modules.get("pipeline.data")
    prev_eval = sys.modules.get("pipeline.evaluate")
    data = types.ModuleType("pipeline.data")
    data.load = lambda split: _FakeTestSplit()
    evaluate = types.ModuleType("pipeline.evaluate")
    evaluate.evaluate = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("submission generation scored test labels"))
    sys.modules["pipeline.data"] = data
    sys.modules["pipeline.evaluate"] = evaluate
    try:
        yield
    finally:
        if prev_data is None:
            del sys.modules["pipeline.data"]
        else:
            sys.modules["pipeline.data"] = prev_data
        if prev_eval is None:
            del sys.modules["pipeline.evaluate"]
        else:
            sys.modules["pipeline.evaluate"] = prev_eval


def test_write_submission_does_not_touch_or_score_test_labels():
    with tempfile.TemporaryDirectory() as d, _fake_test_data():
        run_dir = Path(d)
        out = run_dir / "scripts" / "iter_1_out"
        out.mkdir(parents=True)
        np.save(out / "scores_test.npy", np.asarray([0.3, 0.2, 0.9], dtype=np.float64))
        ledger = Ledger(run_dir / "ledger.jsonl")
        ledger.append(Entry(1, None, 0, "test hypothesis", "code",
                            {"primary": 0.61}, 1.0, 10, 5, "ok"))

        got = _write_submission(run_dir, ledger, baseline_test=0.0)

        assert got["test_scored"] is False
        assert "test_primary" not in got and "test_delta" not in got
        with (run_dir / "submission.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["row_id", "user_id", "video_id", "score"]
        assert rows[1:] == [["0", "10", "101", "0.3"],
                            ["1", "10", "102", "0.2"],
                            ["2", "20", "201", "0.9"]]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok: {t.__name__}")
    print(f"{len(tests)} tests passed")
