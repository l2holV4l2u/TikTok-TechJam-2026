"""Run: python -m tests.test_weights. Plain asserts, no pytest, no data files."""
import numpy as np

from blend.weights import search, cv_search, honest_estimate, user_folds


def _rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(np.float64)


def _synth(n_users=60, rows_per_user=12, seed=0):
    """Users with a latent quality score; labels are noisy draws from it."""
    rng = np.random.default_rng(seed)
    user_ids = np.repeat(np.arange(n_users), rows_per_user)
    n = user_ids.size
    latent = rng.normal(size=n)
    prob = 1.0 / (1.0 + np.exp(-latent))
    labels = (rng.random(n) < prob).astype(np.int64)
    return user_ids, labels, latent, rng


def test_user_folds_never_splits_a_user():
    rng = np.random.default_rng(1)
    user_ids = rng.integers(0, 40, size=2000)  # unsorted, many repeats per user
    fold = user_folds(user_ids, folds=5, seed=7)
    user_fold = {}
    for u in np.unique(user_ids):
        f = np.unique(fold[user_ids == u])
        assert f.size == 1, f"user {u} split across folds"
        user_fold[u] = f[0]
    counts = np.bincount(list(user_fold.values()), minlength=5)
    assert counts.max() - counts.min() <= 1, "fold sizes (by unique user count) should be near-balanced"


def test_user_folds_deterministic_given_seed():
    user_ids = np.repeat(np.arange(30), 5)
    a = user_folds(user_ids, folds=4, seed=3)
    b = user_folds(user_ids, folds=4, seed=3)
    assert np.array_equal(a, b)


def test_weights_nonnegative_and_sum_to_one():
    user_ids, labels, latent, rng = _synth(seed=2)
    comps = {
        "a": _rank(latent + rng.normal(scale=0.5, size=latent.size)),
        "b": _rank(latent + rng.normal(scale=0.5, size=latent.size)),
        "c": _rank(rng.normal(size=latent.size)),
    }
    for method, kw in [("grid", {"step": 0.1}), ("random", {"n_evals": 200, "seed": 0}),
                        ("slsqp", {"restarts": 4, "seed": 0})]:
        out = search(comps, user_ids, labels, method=method, **kw)
        w = out["weights"]
        vals = np.array(list(w.values()))
        assert np.all(vals >= -1e-9), f"{method}: negative weight {w}"
        assert abs(vals.sum() - 1.0) < 1e-6, f"{method}: weights sum to {vals.sum()}, not 1"


def test_noise_component_gets_near_zero_weight():
    user_ids, labels, latent, rng = _synth(n_users=80, rows_per_user=15, seed=3)
    informative = _rank(latent)          # perfectly ranks the labels' generating signal
    noise = _rank(rng.normal(size=latent.size))  # unrelated to labels
    comps = {"informative": informative, "noise": noise}
    out = search(comps, user_ids, labels, method="grid", step=0.05)
    w = out["weights"]
    assert w["informative"] > 0.85, f"informative weight too low: {w}"
    assert w["noise"] < 0.15, f"noise weight too high: {w}"


def test_search_recovers_known_optimum_by_symmetry():
    """Two equally-good, independently-noisy views of the same latent signal:
    averaging them (denoising) should beat using either alone, so the optimal
    weight is known by construction to sit at 0.5/0.5."""
    user_ids, labels, latent, rng = _synth(n_users=100, rows_per_user=15, seed=4)
    comp1 = _rank(latent + rng.normal(scale=1.5, size=latent.size))
    comp2 = _rank(latent + rng.normal(scale=1.5, size=latent.size))
    comps = {"comp1": comp1, "comp2": comp2}
    out = search(comps, user_ids, labels, method="grid", step=0.05)
    w = out["weights"]
    assert abs(w["comp1"] - 0.5) < 0.2, f"expected near-symmetric optimum, got {w}"

    solo1 = search({"comp1": comp1}, user_ids, labels, method="grid")["score"]
    solo2 = search({"comp2": comp2}, user_ids, labels, method="grid")["score"]
    assert out["score"] >= max(solo1, solo2) - 1e-9, "blend should not be worse than either solo component"


def test_random_and_slsqp_find_comparable_optimum_to_grid():
    user_ids, labels, latent, rng = _synth(n_users=80, rows_per_user=15, seed=5)
    comps = {
        "a": _rank(latent + rng.normal(scale=0.7, size=latent.size)),
        "b": _rank(rng.normal(size=latent.size)),
    }
    grid = search(comps, user_ids, labels, method="grid", step=0.05)["score"]
    rand = search(comps, user_ids, labels, method="random", n_evals=300, seed=0)["score"]
    slsqp = search(comps, user_ids, labels, method="slsqp", restarts=6, seed=0)["score"]
    assert rand >= grid - 0.01, f"random search much worse than grid: {rand} vs {grid}"
    assert slsqp >= grid - 0.01, f"slsqp much worse than grid: {slsqp} vs {grid}"


def test_grid_rejects_too_many_components():
    user_ids, labels, latent, rng = _synth(seed=6)
    comps = {str(i): _rank(rng.normal(size=latent.size)) for i in range(5)}
    try:
        search(comps, user_ids, labels, method="grid")
        assert False, "expected ValueError for grid with >4 components"
    except ValueError:
        pass


def test_cv_search_folds_by_user_and_returns_oof_score():
    user_ids, labels, latent, rng = _synth(n_users=90, rows_per_user=12, seed=7)
    comps = {
        "a": _rank(latent + rng.normal(scale=0.5, size=latent.size)),
        "b": _rank(rng.normal(size=latent.size)),
    }
    out = cv_search(comps, user_ids, labels, folds=5, seed=0, method="grid", step=0.1)
    assert len(out["fold_weights"]) == 5
    for w in out["fold_weights"]:
        vals = np.array(list(w.values()))
        assert np.all(vals >= -1e-9) and abs(vals.sum() - 1.0) < 1e-6
    assert 0.0 <= out["out_of_fold_score"] <= 1.0


def test_honest_estimate_shows_positive_optimism_with_many_noise_components():
    """With few users and many purely-noise components, in-sample weight fitting can
    chase validation noise; the honest OOF estimate should not, so optimism > 0."""
    user_ids, labels, latent, rng = _synth(n_users=120, rows_per_user=10, seed=8)
    n = labels.size
    comps = {f"noise{i}": _rank(rng.normal(size=n)) for i in range(20)}
    out = honest_estimate(comps, user_ids, labels, folds=5, seed=0, method="random", n_evals=250)
    assert out["optimism"] > 0.0, f"expected positive optimism from overfit noise, got {out}"
    assert abs((out["in_sample"] - out["out_of_fold"]) - out["optimism"]) < 1e-9


def test_honest_estimate_low_optimism_with_one_component():
    """A single component has no weight freedom (always weight 1.0), so in-sample
    and out-of-fold should closely agree -- optimism should be small."""
    user_ids, labels, latent, rng = _synth(n_users=90, rows_per_user=12, seed=9)
    comps = {"only": _rank(latent + rng.normal(scale=0.5, size=latent.size))}
    out = honest_estimate(comps, user_ids, labels, folds=5, seed=0, method="grid")
    assert abs(out["optimism"]) < 0.03, f"expected near-zero optimism for a single component, got {out}"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
