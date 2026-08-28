"""Rank-blend weight search with user-grouped CV, to keep validation-set weight
selection from overfitting into an unearned score.

Weights are non-negative and sum to 1. `search` fits on whatever rows it is given
(in-sample, optimistic). `cv_search`/`honest_estimate` fold by USER so no user's
rows appear in both the fitting fold and the fold used to score that fit -- GAUC
and nDCG@5 are both per-user, so a row-level split would leak and understate the
real overfit risk.
"""
import numpy as np
from pipeline.evaluate import evaluate


def _blend(component_ranks: dict, weights: dict) -> np.ndarray:
    names = list(component_ranks)
    return sum(weights[name] * component_ranks[name] for name in names)


def _score(component_ranks, user_ids, labels, weights) -> float:
    return evaluate(user_ids, labels, _blend(component_ranks, weights))["primary"]


def user_folds(user_ids, folds: int, seed: int) -> np.ndarray:
    """Deterministic per-row fold id, 0..folds-1, constant within each user_id."""
    user_ids = np.asarray(user_ids)
    uniq = np.unique(user_ids)
    rng = np.random.default_rng(seed)
    order = rng.permutation(uniq.size)
    fold_of_user = np.empty(uniq.size, dtype=np.int64)
    fold_of_user[order] = np.arange(uniq.size) % folds  # near-exactly balanced fold sizes
    return fold_of_user[np.searchsorted(uniq, user_ids)]


def _grid_search(component_ranks, user_ids, labels, step=0.1):
    names = list(component_ranks)
    n = len(names)
    k = max(1, round(1.0 / step))
    best_w, best_s, n_evals = None, -np.inf, 0

    def rec(i, remaining, acc):
        nonlocal best_w, best_s, n_evals
        if i == n - 1:
            w = {name: p / k for name, p in zip(names, acc + [remaining])}
            s = _score(component_ranks, user_ids, labels, w)
            n_evals += 1
            if s > best_s:
                best_s, best_w = s, w
            return
        for v in range(remaining + 1):
            rec(i + 1, remaining - v, acc + [v])

    rec(0, k, [])
    return best_w, best_s, n_evals


def _random_search(component_ranks, user_ids, labels, n_evals=500, seed=0, alpha=1.0):
    names = list(component_ranks)
    n = len(names)
    rng = np.random.default_rng(seed)
    best_w, best_s = None, -np.inf
    for _ in range(n_evals):
        w = dict(zip(names, rng.dirichlet(np.full(n, alpha))))
        s = _score(component_ranks, user_ids, labels, w)
        if s > best_s:
            best_s, best_w = s, w
    return best_w, best_s, n_evals


def _slsqp_search(component_ranks, user_ids, labels, restarts=8, seed=0, maxiter=60):
    from scipy.optimize import minimize
    names = list(component_ranks)
    n = len(names)
    rng = np.random.default_rng(seed)

    def project(x):
        x = np.clip(x, 0.0, None)
        s = x.sum()
        return x / s if s > 0 else np.full(n, 1.0 / n)

    def neg_score(x):
        return -_score(component_ranks, user_ids, labels, dict(zip(names, project(x))))

    cons = [{"type": "eq", "fun": lambda x: x.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n
    best_w, best_s, n_evals = None, -np.inf, 0
    for _ in range(restarts):
        x0 = rng.dirichlet(np.full(n, 1.0))
        res = minimize(neg_score, x0, method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": maxiter, "ftol": 1e-6})
        n_evals += res.nfev
        w = dict(zip(names, project(res.x)))
        s = _score(component_ranks, user_ids, labels, w)
        n_evals += 1
        if s > best_s:
            best_s, best_w = s, w
    return best_w, best_s, n_evals


def search(component_ranks: dict, user_ids, labels, method: str = "grid", **kw) -> dict:
    """Fit blend weights on the rows given. IN-SAMPLE: whatever split you pass is what
    gets fit and scored, so this alone always overstates transfer to a held-out split."""
    user_ids = np.asarray(user_ids)
    labels = np.asarray(labels)
    n = len(component_ranks)
    if method == "grid":
        if n > 4:
            raise ValueError(f"grid search supports <=4 components, got {n}; use method='random' or 'slsqp'")
        w, s, n_evals = _grid_search(component_ranks, user_ids, labels, step=kw.get("step", 0.1))
    elif method == "random":
        w, s, n_evals = _random_search(component_ranks, user_ids, labels, n_evals=kw.get("n_evals", 500),
                                        seed=kw.get("seed", 0), alpha=kw.get("alpha", 1.0))
    elif method == "slsqp":
        w, s, n_evals = _slsqp_search(component_ranks, user_ids, labels, restarts=kw.get("restarts", 8),
                                       seed=kw.get("seed", 0), maxiter=kw.get("maxiter", 60))
    else:
        raise ValueError(f"unknown method {method!r}")
    return {"weights": w, "score": s, "method": method, "n_evals": n_evals}


def cv_search(component_ranks: dict, user_ids, labels, folds: int = 5, seed: int = 0,
              method: str = "grid", **kw) -> dict:
    """Fit weights on folds-1 folds, predict the held-out fold, repeat per fold.
    Folding is by user_id so a user's rows never split across fit/score. Returns an
    honest out-of-fold score: concatenate every fold's held-out predictions (each row
    scored with weights that never saw it) and evaluate once over the full set."""
    user_ids = np.asarray(user_ids)
    labels = np.asarray(labels)
    names = list(component_ranks)
    row_fold = user_folds(user_ids, folds, seed)
    oof = np.zeros(labels.shape[0], dtype=np.float64)
    fold_weights = []
    for k in range(folds):
        held = row_fold == k
        fit = ~held
        fit_ranks = {name: component_ranks[name][fit] for name in names}
        kw_fold = dict(kw)
        kw_fold.setdefault("seed", seed + k)  # distinct default draws per fold, still deterministic
        w = search(fit_ranks, user_ids[fit], labels[fit], method=method, **kw_fold)["weights"]
        fold_weights.append(w)
        held_ranks = {name: component_ranks[name][held] for name in names}
        oof[held] = _blend(held_ranks, w)
    oof_score = evaluate(user_ids, labels, oof)["primary"]
    return {"fold_weights": fold_weights, "out_of_fold_score": oof_score, "folds": folds,
            "seed": seed, "method": method}


def honest_estimate(component_ranks: dict, user_ids, labels, folds: int = 5, seed: int = 0,
                     method: str = "grid", **kw) -> dict:
    """in_sample: fit+score on the whole split (what a naive report would show).
    out_of_fold: cv_search's honest estimate. optimism = in_sample - out_of_fold,
    i.e. how much of the in-sample gain would NOT be expected to transfer."""
    in_sample = search(component_ranks, user_ids, labels, method=method, **kw)["score"]
    cv = cv_search(component_ranks, user_ids, labels, folds=folds, seed=seed, method=method, **kw)
    out_of_fold = cv["out_of_fold_score"]
    return {"in_sample": in_sample, "out_of_fold": out_of_fold, "optimism": in_sample - out_of_fold}
