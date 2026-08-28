"""Deterministic controller-side incumbent ensemble tests."""
import contextlib
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

from agent.ensemble import _within_user_rank, retain_or_blend
from pipeline.evaluate import evaluate


@contextlib.contextmanager
def _fake_data(valid, test):
    previous = sys.modules.get("pipeline.data")
    module = types.ModuleType("pipeline.data")
    module.load = lambda name: valid if name == "valid" else test
    sys.modules["pipeline.data"] = module
    try:
        yield
    finally:
        if previous is None:
            del sys.modules["pipeline.data"]
        else:
            sys.modules["pipeline.data"] = previous


def test_complementary_candidate_is_blended_and_saved_for_submission():
    users = np.repeat(np.arange(100), 20)
    labels = (np.random.default_rng(1).random(len(users)) < 0.3).astype(np.int8)
    rng = np.random.default_rng(0)
    incumbent = labels + rng.normal(0, 1.8, len(labels))
    candidate = labels + rng.normal(0, 1.8, len(labels))
    valid = types.SimpleNamespace(user_id=users, y=labels)
    test = types.SimpleNamespace(user_id=users[::-1], y=labels[::-1])

    with tempfile.TemporaryDirectory() as tmp, _fake_data(valid, test):
        run = Path(tmp)
        artifacts = run / "artifacts"
        out = run / "scripts" / "iter_2_out"
        artifacts.mkdir()
        out.mkdir(parents=True)
        np.save(artifacts / "incumbent_valid_scores.npy", incumbent)
        np.save(artifacts / "incumbent_test_scores.npy", incumbent[::-1])
        incumbent_primary = evaluate(users, labels, incumbent)["primary"]
        (artifacts / "incumbent.json").write_text(json.dumps({
            "iter_id": 1, "valid_primary": incumbent_primary,
        }), encoding="utf-8")
        evaluator = types.SimpleNamespace(last_scores=candidate,
                                          last_test_scores=candidate[::-1])
        raw = evaluate(users, labels, candidate)
        got = retain_or_blend(raw, evaluator, artifacts, run, 2)

        assert got["harness_blend_alpha"] == 0.5, got
        assert got["primary"] > max(raw["primary"], incumbent_primary)
        assert np.array_equal(np.load(out / "scores_test.npy"), evaluator.last_test_scores)
        record = json.loads((run / "harness_ensembles.jsonl").read_text())
        assert record["selected_alpha"] == 0.5


def test_rank_transform_preserves_ties():
    ranks = _within_user_rank(
        np.array([1, 1, 1, 2, 2]), np.array([0.0, 0.0, 1.0, 4.0, 4.0]))
    assert ranks[0] == ranks[1] == 0.25
    assert ranks[2] == 1.0
    assert ranks[3] == ranks[4] == 0.5


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print(f"{len(tests)} tests passed")
