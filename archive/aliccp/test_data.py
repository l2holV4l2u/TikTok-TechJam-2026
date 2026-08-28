"""Plain-assert tests for pipeline.data. Run with: python -m tests.test_data"""
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from pipeline.data import FEATURE_CARDINALITIES, MIN_CONV_POSITIVES, load, write_cache
from pipeline.synth import generate_split


def _use_temp_cache():
    tmp = Path(tempfile.mkdtemp(prefix="aliccp_cache_"))
    prev = os.environ.get("ALICCP_CACHE_DIR")
    os.environ["ALICCP_CACHE_DIR"] = str(tmp)
    return tmp, prev


def _restore(tmp: Path, prev):
    if prev is None:
        os.environ.pop("ALICCP_CACHE_DIR", None)
    else:
        os.environ["ALICCP_CACHE_DIR"] = prev
    shutil.rmtree(tmp, ignore_errors=True)


def test_missing_cache_raises():
    tmp, prev = _use_temp_cache()
    try:
        try:
            load("train")
            assert False, "expected FileNotFoundError"
        except FileNotFoundError as e:
            assert "train" in str(e)
            assert "python -m pipeline.synth" in str(e)
    finally:
        _restore(tmp, prev)


def test_determinism():
    tmp, prev = _use_temp_cache()
    try:
        # exact (not random) label counts, so pool sizes are known safely above the MIN_*_POSITIVES floors
        n = 5_000
        rng = np.random.default_rng(1)
        X = {f: rng.integers(1, c, size=n, dtype=np.int64) for f, c in FEATURE_CARDINALITIES.items()}
        y_click = np.zeros(n, dtype=np.int8)
        y_click[:200] = 1  # 4% click rate
        y_conv = np.zeros(n, dtype=np.int8)
        y_conv[:30] = 1  # 30 conversions, all inside the click=1 rows
        write_cache(tmp, "train", X, y_click, y_conv)

        s1 = load("train", fraction=0.2, seed=42)
        s2 = load("train", fraction=0.2, seed=42)
        for feat in FEATURE_CARDINALITIES:
            assert np.array_equal(s1.X[feat], s2.X[feat])
        assert np.array_equal(s1.y_click, s2.y_click)
        assert np.array_equal(s1.y_conv, s2.y_conv)

        # labels are fixed by the per-stratum counts, so a new seed changes which rows are drawn, not the label vector
        s3 = load("train", fraction=0.2, seed=43)
        assert np.array_equal(s1.y_click, s3.y_click)
        assert any(not np.array_equal(s1.X[f], s3.X[f]) for f in FEATURE_CARDINALITIES)
    finally:
        _restore(tmp, prev)


def test_stratification_holds_and_guarantees_positives():
    tmp, prev = _use_temp_cache()
    try:
        n = 1_000_000
        X, yc, yv = generate_split(n=n, seed=7)
        write_cache(tmp, "train", X, yc, yv)

        true_click_rate = float(yc.mean())
        clicked = yc == 1
        total_conv = int(yv[clicked].sum())
        assert total_conv > MIN_CONV_POSITIVES, "fixture too small to prove both floor and proportional paths"

        for fraction in (0.01, 0.10):
            s = load("train", fraction=fraction, seed=5)
            got_click_rate = float(s.y_click.mean())
            assert abs(got_click_rate - true_click_rate) < 0.01

            n_conv = int(s.y_conv.sum())
            assert n_conv >= MIN_CONV_POSITIVES, "stratified subsample must guarantee a minimum positive count"

            naive_expected = total_conv * fraction  # what plain random sampling would expect -- can round to 0
            assert n_conv >= naive_expected
    finally:
        _restore(tmp, prev)


def test_guard_raises_when_infeasible():
    tmp, prev = _use_temp_cache()
    try:
        n = 5_000
        rng = np.random.default_rng(0)
        X = {f: rng.integers(1, c, size=n, dtype=np.int64) for f, c in FEATURE_CARDINALITIES.items()}
        y_click = np.zeros(n, dtype=np.int8)
        y_click[:200] = 1  # 4% click rate
        y_conv = np.zeros(n, dtype=np.int8)
        y_conv[:3] = 1  # only 3 conversions total, below MIN_CONV_POSITIVES
        write_cache(tmp, "train", X, y_click, y_conv)

        try:
            load("train", fraction=0.5, seed=0)
            assert False, "expected ValueError for too few positives to subsample safely"
        except ValueError as e:
            assert "conv" in str(e).lower()
    finally:
        _restore(tmp, prev)


def main():
    tests = [
        test_missing_cache_raises,
        test_determinism,
        test_stratification_holds_and_guarantees_positives,
        test_guard_raises_when_infeasible,
    ]
    for t in tests:
        t()
        print(f"ok: {t.__name__}")
    print("all tests passed")


if __name__ == "__main__":
    main()
