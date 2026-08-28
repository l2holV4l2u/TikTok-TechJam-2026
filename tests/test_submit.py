"""Run: python -m tests.test_submit. Plain asserts, no pytest, no real dataset."""
import contextlib
import csv
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

from pipeline.submit import check_submission, make_submission

_USER_IDS = [10, 10, 20, 30, 30]
_VIDEO_IDS = [1, 2, 3, 4, 5]


class _FakeSplit:
    def __init__(self, user_ids, video_ids):
        self.user_id = np.asarray(user_ids)
        self.video_id = np.asarray(video_ids)


@contextlib.contextmanager
def _fake_data_module(user_ids=_USER_IDS, video_ids=_VIDEO_IDS):
    """Install a stub pipeline.data module so tests never touch the real dataset."""
    prev = sys.modules.get("pipeline.data")
    mod = types.ModuleType("pipeline.data")
    mod.load = lambda split: _FakeSplit(user_ids, video_ids)
    sys.modules["pipeline.data"] = mod
    try:
        yield
    finally:
        if prev is None:
            del sys.modules["pipeline.data"]
        else:
            sys.modules["pipeline.data"] = prev


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def _valid_rows():
    return [[i, _USER_IDS[i], _VIDEO_IDS[i], 0.1 * i] for i in range(len(_USER_IDS))]


def test_make_then_check_round_trips_valid():
    with _fake_data_module(), tempfile.TemporaryDirectory() as d:
        out = str(Path(d) / "sub.csv")
        make_submission(out, split="eval")
        ok, reason = check_submission(out, split="eval")
        assert ok, reason


def test_valid_handwritten_file_passes():
    with _fake_data_module(), tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "sub.csv")
        _write_csv(path, ["row_id", "user_id", "video_id", "score"], _valid_rows())
        ok, reason = check_submission(path)
        assert ok, reason


def test_rejects_wrong_header():
    with _fake_data_module(), tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "sub.csv")
        _write_csv(path, ["id", "user_id", "video_id", "score"], _valid_rows())
        ok, reason = check_submission(path)
        assert not ok
        assert "header" in reason


def test_rejects_row_count_mismatch():
    with _fake_data_module(), tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "sub.csv")
        rows = _valid_rows()[:-1]  # drop the last row
        _write_csv(path, ["row_id", "user_id", "video_id", "score"], rows)
        ok, reason = check_submission(path)
        assert not ok
        assert "row count" in reason


def test_rejects_row_id_gap():
    with _fake_data_module(), tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "sub.csv")
        rows = _valid_rows()
        rows[2][0] = 99  # break monotonicity with a gap
        _write_csv(path, ["row_id", "user_id", "video_id", "score"], rows)
        ok, reason = check_submission(path)
        assert not ok
        assert "row_id" in reason


def test_rejects_row_id_non_monotonic():
    with _fake_data_module(), tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "sub.csv")
        rows = _valid_rows()
        rows[1][0], rows[2][0] = rows[2][0], rows[1][0]  # swap two row_ids out of order
        _write_csv(path, ["row_id", "user_id", "video_id", "score"], rows)
        ok, reason = check_submission(path)
        assert not ok
        assert "row_id" in reason


def test_rejects_misaligned_user_id():
    with _fake_data_module(), tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "sub.csv")
        rows = _valid_rows()
        rows[0][1] = 999999  # user_id no longer matches the split at this row_id
        _write_csv(path, ["row_id", "user_id", "video_id", "score"], rows)
        ok, reason = check_submission(path)
        assert not ok
        assert "misaligned" in reason


def test_rejects_misaligned_video_id():
    with _fake_data_module(), tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "sub.csv")
        rows = _valid_rows()
        rows[3][2] = 999999
        _write_csv(path, ["row_id", "user_id", "video_id", "score"], rows)
        ok, reason = check_submission(path)
        assert not ok
        assert "misaligned" in reason


def test_rejects_non_numeric_score():
    with _fake_data_module(), tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "sub.csv")
        rows = _valid_rows()
        rows[1][3] = "not_a_number"
        _write_csv(path, ["row_id", "user_id", "video_id", "score"], rows)
        ok, reason = check_submission(path)
        assert not ok
        assert "not numeric" in reason


def test_rejects_nan_score():
    with _fake_data_module(), tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "sub.csv")
        rows = _valid_rows()
        rows[0][3] = "nan"
        _write_csv(path, ["row_id", "user_id", "video_id", "score"], rows)
        ok, reason = check_submission(path)
        assert not ok
        assert "NaN" in reason or "Inf" in reason


def test_rejects_inf_score():
    with _fake_data_module(), tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "sub.csv")
        rows = _valid_rows()
        rows[4][3] = "inf"
        _write_csv(path, ["row_id", "user_id", "video_id", "score"], rows)
        ok, reason = check_submission(path)
        assert not ok
        assert "NaN" in reason or "Inf" in reason


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok: {t.__name__}")
    print(f"{len(tests)} tests passed")
