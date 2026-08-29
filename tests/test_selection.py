"""Fold-split and acceptance guard: python -m tests.test_selection"""
import numpy as np

from agent.ledger import Entry
from agent.selection import accept, primary_on, split_validation, user_folds


def _fixture(seed=0, n_users=400, per_user=6):
    rng = np.random.default_rng(seed)
    users = np.repeat(np.arange(n_users), per_user)
    y = (rng.random(users.size) < 0.35).astype(np.int8)
    return users, y, rng


def test_folds_never_split_a_user():
    users, _, _ = _fixture()
    a, b = split_validation(users)
    assert not (a & b).any(), "a row cannot be in both folds"
    assert (a | b).all(), "every row must land in a fold"
    for u in np.unique(users):
        m = users == u
        assert a[m].all() or b[m].all(), f"user {u} spans both folds"


def test_folds_are_deterministic_given_seed():
    users, _, _ = _fixture()
    assert np.array_equal(split_validation(users, seed=7)[0],
                          split_validation(users, seed=7)[0])
    assert not np.array_equal(split_validation(users, seed=7)[0],
                              split_validation(users, seed=8)[0])


def test_folds_are_near_balanced():
    users, _, _ = _fixture()
    a, _ = split_validation(users)
    share = a.mean()
    assert 0.4 < share < 0.6, share


def test_matches_the_analysis_packages_implementation():
    """agent.selection reimplements blend.weights.user_folds so the shipped agent does not
    depend on the analysis package. They must not drift apart."""
    from blend.weights import user_folds as blend_user_folds

    users, _, _ = _fixture()
    for seed in (0, 3, 11):
        assert np.array_equal(user_folds(users, 2, seed), blend_user_folds(users, 2, seed))


def test_accept_passes_a_genuine_win_on_both():
    users, y, rng = _fixture()
    a, b = split_validation(users)
    incumbent = rng.random(users.size)
    genuine = y + rng.normal(0, 0.6, users.size)
    ok, detail = accept(genuine, incumbent, users, y, a, b)
    assert ok, detail
    assert detail["fold_a_gain"] > 0 and detail["fold_b_gain"] > 0, detail


def test_accept_rejects_a_fold_a_win_that_loses_on_fold_b():
    """The whole point of the guard: winning the split is not winning the task."""
    users, y, rng = _fixture()
    a, b = split_validation(users)
    incumbent = rng.random(users.size)
    curse = incumbent.copy()
    curse[a] = y[a] + rng.normal(0, 0.05, int(a.sum()))     # near-perfect on A
    curse[b] = -y[b] + rng.normal(0, 0.05, int(b.sum()))    # inverted on B
    ok, detail = accept(curse, incumbent, users, y, a, b)
    assert not ok, detail
    assert detail["fold_a_gain"] > 0, "the fixture must actually win fold A"
    assert detail["fold_b_gain"] < 0, "and must actually lose fold B"


def test_accept_rejects_a_gain_inside_epsilon():
    users, y, rng = _fixture()
    a, b = split_validation(users)
    incumbent = y + rng.normal(0, 0.8, users.size)
    ok, _ = accept(incumbent.copy(), incumbent, users, y, a, b)
    assert not ok, "an identical candidate gains nothing and must not be accepted"


def test_accept_refuses_an_unscorable_fold_instead_of_guessing():
    users, y, _ = _fixture()
    a, b = split_validation(users)
    empty = np.zeros_like(a)
    ok, detail = accept(np.ones(users.size), np.zeros(users.size), users, y, a, empty)
    assert not ok and not np.isfinite(detail["fold_b_gain"]), detail


def test_primary_on_restricts_to_the_mask():
    users, y, rng = _fixture()
    a, _ = split_validation(users)
    scores = rng.random(users.size)
    from pipeline.evaluate import evaluate
    assert abs(primary_on(users, y, scores, a)
               - evaluate(users[a], y[a], scores[a])["primary"]) < 1e-12


def test_old_ledger_lines_still_parse_without_slot_id():
    """slot_id/turn were added for the portfolio; every ledger written before it must load."""
    import json
    from dataclasses import asdict

    legacy = {"iter_id": 3, "parent_iter_id": 1, "tier": 0, "hypothesis": "h", "diff": "c",
              "metrics": {"primary": 0.6}, "gpu_seconds": 1.0, "tokens_in": 10,
              "tokens_out": 5, "status": "ok", "error": None, "phase": "improve",
              "timestamp": 0.0}
    e = Entry(**legacy)
    assert e.slot_id is None and e.turn is None
    round_tripped = Entry(**json.loads(json.dumps(asdict(e))))
    assert round_tripped == e


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok: {test.__name__}")
    print(f"{len(tests)} tests passed")
