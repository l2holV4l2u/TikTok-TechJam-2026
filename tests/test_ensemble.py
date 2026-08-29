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


def _pool(seed=0, n_users=500, per_user=6):
    from agent.selection import split_validation
    rng = np.random.default_rng(seed)
    users = np.repeat(np.arange(n_users), per_user)
    y = (rng.random(users.size) < 0.35).astype(np.int8)
    a, b = split_validation(users)
    return users, y, a, b, rng


def test_blend_never_scores_below_the_incumbent_on_fold_a():
    from agent.ensemble import blend_portfolio
    users, y, a, b, rng = _pool()
    inc = y + rng.normal(0, 2.0, users.size)
    members = {f"m{i}": (y + rng.normal(0, 2.0, users.size), None) for i in range(3)}
    got = blend_portfolio(members, inc, None, users, y, a, b)
    assert got["accepted"], got
    assert got["fold_a_gain"] > 0, got
    assert got["members"], "a complementary pool must produce a blend"


def test_blend_rejected_when_it_loses_on_fold_b():
    """The guard the whole phase exists for: winning the selection split is not winning."""
    from agent.ensemble import blend_portfolio
    users, y, a, b, rng = _pool()
    inc = y + rng.normal(0, 2.0, users.size)
    curse = inc.copy()
    curse[a] = y[a] + rng.normal(0, 0.05, int(a.sum()))
    curse[b] = -y[b] + rng.normal(0, 0.05, int(b.sum()))
    got = blend_portfolio({"curse": (curse, None)}, inc, None, users, y, a, b)
    assert not got["accepted"], got
    assert got["fold_a_gain"] > 0 > got["fold_b_gain"], got
    assert got["valid"] is None and "confirmation fold" in got["reason"]


def test_duplicate_members_add_nothing():
    from agent.ensemble import blend_portfolio
    users, y, a, b, rng = _pool()
    inc = y + rng.normal(0, 2.0, users.size)
    got = blend_portfolio({"copy": (inc.copy(), None)}, inc, None, users, y, a, b)
    assert not got["accepted"], "a copy of the incumbent cannot improve on it"
    assert got["members"] == []


def test_blend_stops_at_max_members():
    from agent.ensemble import blend_portfolio
    users, y, a, b, rng = _pool()
    inc = y + rng.normal(0, 2.0, users.size)
    members = {f"m{i}": (y + rng.normal(0, 2.0, users.size), None) for i in range(12)}
    got = blend_portfolio(members, inc, None, users, y, a, b, max_members=2)
    assert len(got["members"]) <= 2, got["members"]


def test_test_scores_use_the_same_members_and_weights_as_validation():
    """Applied, never re-selected: choosing on test would be fitting the hidden split."""
    from agent.ensemble import blend_portfolio, _within_user_rank
    users, y, a, b, rng = _pool(n_users=200)
    t_users = np.repeat(np.arange(120), 5)
    inc_v = y + rng.normal(0, 2.0, users.size)
    inc_t = rng.normal(0, 1.0, t_users.size)
    members = {f"m{i}": (y + rng.normal(0, 2.0, users.size),
                         rng.normal(0, 1.0, t_users.size)) for i in range(3)}
    got = blend_portfolio(members, inc_v, inc_t, users, y, a, b, test_user_id=t_users)
    assert got["accepted"] and got["test"] is not None, got
    expected = np.mean([_within_user_rank(t_users, inc_t)]
                       + [_within_user_rank(t_users, members[m][1]) for m in got["members"]],
                       axis=0)
    assert np.allclose(got["test"], expected), "test must use the selected members, equally weighted"


def test_blend_returns_no_test_scores_rather_than_a_wrong_shape():
    from agent.ensemble import blend_portfolio
    users, y, a, b, rng = _pool(n_users=200)
    t_users = np.repeat(np.arange(80), 5)
    inc_v = y + rng.normal(0, 2.0, users.size)
    members = {"m": (y + rng.normal(0, 2.0, users.size), rng.normal(0, 1, 3))}  # wrong length
    got = blend_portfolio(members, inc_v, rng.normal(0, 1, t_users.size), users, y, a, b,
                          test_user_id=t_users)
    assert got["test"] is None, "a mismatched member must not silently produce test scores"


def test_blend_is_deterministic():
    from agent.ensemble import blend_portfolio
    outs = []
    for _ in range(2):
        users, y, a, b, rng = _pool(seed=5)
        inc = y + rng.normal(0, 2.0, users.size)
        members = {f"m{i}": (y + rng.normal(0, 2.0, users.size), None) for i in range(4)}
        outs.append(blend_portfolio(members, inc, None, users, y, a, b))
    assert outs[0]["members"] == outs[1]["members"]
    assert abs(outs[0]["fold_a_gain"] - outs[1]["fold_a_gain"]) < 1e-12


def test_a_member_of_the_wrong_length_is_ignored_not_fatal():
    from agent.ensemble import blend_portfolio
    users, y, a, b, rng = _pool()
    inc = y + rng.normal(0, 2.0, users.size)
    members = {"good": (y + rng.normal(0, 2.0, users.size), None),
               "bad": (np.array([1.0, 2.0]), None)}
    got = blend_portfolio(members, inc, None, users, y, a, b)
    assert "bad" not in got["members"], got["members"]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print(f"{len(tests)} tests passed")


# --------------------------------------------------- Phase 5: portfolio blend + fold guard
