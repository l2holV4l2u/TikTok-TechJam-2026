"""Leakage and leave-one-out tests for train-derived entity histories."""
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from pipeline.data import load
from pipeline.history import clear_cache, historical_features
from pipeline.synth import build as build_synth


def _fixture():
    root = Path(tempfile.mkdtemp(prefix="history_cache_"))
    previous = os.environ.get("KUAIRAND_CACHE_DIR")
    os.environ["KUAIRAND_CACHE_DIR"] = str(root)
    build_synth(root, sizes={"train": 300, "valid": 80, "test": 80}, seed=13)
    clear_cache()
    return root, previous


def _restore(root, previous):
    clear_cache()
    if previous is None:
        os.environ.pop("KUAIRAND_CACHE_DIR", None)
    else:
        os.environ["KUAIRAND_CACHE_DIR"] = previous
    shutil.rmtree(root, ignore_errors=True)


def test_validation_and_test_labels_cannot_change_history_features():
    root, previous = _fixture()
    hidden = os.environ.get("AGENT_HIDE_TEST_LABELS")
    try:
        before_valid = historical_features("valid")
        np.save(root / "valid" / "y.npy", 1 - np.load(root / "valid" / "y.npy"))
        after_valid = historical_features("valid")
        for name in before_valid:
            assert np.array_equal(before_valid[name], after_valid[name]), name

        os.environ["AGENT_HIDE_TEST_LABELS"] = "1"
        got = historical_features("test")
        assert got and all(len(v) == 80 for v in got.values())
    finally:
        if hidden is None:
            os.environ.pop("AGENT_HIDE_TEST_LABELS", None)
        else:
            os.environ["AGENT_HIDE_TEST_LABELS"] = hidden
        _restore(root, previous)


def test_singleton_train_entities_receive_leave_one_out_prior():
    root, previous = _fixture()
    try:
        train = load("train")
        ids = np.asarray(train.X["video_id"])
        counts = np.bincount(ids)
        singleton = np.flatnonzero((ids != 0) & (counts[ids] == 1))[0]
        got = historical_features("train")["video_id_long_view_rate"]
        assert abs(float(got[singleton]) - float(np.mean(train.y))) < 1e-6
        assert np.isfinite(got).all()
    finally:
        _restore(root, previous)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print(f"{len(tests)} tests passed")
