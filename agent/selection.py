"""Guard against the winner's curse when more candidates are compared on one validation split.

Selecting the maximum of k candidates on a finite validation set returns a score inflated by
selection alone, and that inflation does not transfer to test. Against this benchmark's
reported 5-seed sigma of 0.0008 the expected inflation is roughly +0.00114 at k=8, +0.00145 at
k=18 and +0.00180 at k=50 -- so a portfolio that triples the number of candidates compared
buys about +0.0003 of validation that is not real. The run's own history shows the failure
mode already: r74 has the better validation (0.6049 vs 0.6047) and the worse hidden test
(0.5991 vs 0.5998).

The remedy is not to compare fewer things. It is to stop letting one split both propose and
confirm: choose on fold A, require the choice not to fall apart on fold B. Folds are grouped by
user because both scored metrics are per-user means -- splitting a user's rows across folds
would leak the very thing being held out.
"""
from __future__ import annotations

import numpy as np

EPSILON = 0.002  # the organizers' convergence epsilon, reused as the "materially worse" bar


def user_folds(user_id, folds: int = 2, seed: int = 0) -> np.ndarray:
    """Per-row fold id in 0..folds-1, constant within each user_id.

    Deliberately identical to blend.weights.user_folds. It is reimplemented rather than
    imported so the shipped agent does not depend on the analysis package;
    tests/test_selection.py asserts the two never drift apart.
    """
    user_id = np.asarray(user_id)
    uniq = np.unique(user_id)
    rng = np.random.default_rng(seed)
    order = rng.permutation(uniq.size)
    fold_of_user = np.empty(uniq.size, dtype=np.int64)
    fold_of_user[order] = np.arange(uniq.size) % folds  # near-exactly balanced fold sizes
    return fold_of_user[np.searchsorted(uniq, user_id)]


def split_validation(user_id, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (fold_a, fold_b), user-grouped so no user spans both."""
    fold = user_folds(user_id, 2, seed)
    return fold == 0, fold == 1


def primary_on(user_id, labels, scores, mask) -> float:
    """The scored primary restricted to one fold. NaN when the fold has no ranked user."""
    from pipeline.evaluate import evaluate

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    got = evaluate(np.asarray(user_id)[mask], np.asarray(labels)[mask],
                   np.asarray(scores, dtype=np.float64)[mask])
    return float(got["primary"])


def accept(candidate, incumbent, user_id, labels, fold_a, fold_b,
           epsilon: float = EPSILON) -> tuple[bool, dict]:
    """Is `candidate` a real improvement on `incumbent`, or a fold-A artefact?

    Accepted when it gains more than `epsilon` on the selection fold AND does not lose by more
    than `epsilon` on the confirmation fold. The second half is what a single-split comparison
    cannot say: a candidate that wins A and collapses on B won on the split, not on the task.

    Returns (accepted, detail) so the caller can log exactly why, which is the whole point --
    a rejected blend has to be explainable in the run log or it looks like a lost gain.
    """
    a_new = primary_on(user_id, labels, candidate, fold_a)
    a_old = primary_on(user_id, labels, incumbent, fold_a)
    b_new = primary_on(user_id, labels, candidate, fold_b)
    b_old = primary_on(user_id, labels, incumbent, fold_b)
    gain_a = a_new - a_old
    gain_b = b_new - b_old
    # NaN on either fold means it could not be scored there; refuse rather than guess
    ok = bool(np.isfinite(gain_a) and np.isfinite(gain_b)
              and gain_a > epsilon and gain_b > -epsilon)
    return ok, {"fold_a_gain": gain_a, "fold_b_gain": gain_b,
                "fold_a_candidate": a_new, "fold_a_incumbent": a_old,
                "fold_b_candidate": b_new, "fold_b_incumbent": b_old,
                "epsilon": epsilon, "accepted": ok}


def demo() -> None:
    rng = np.random.default_rng(0)
    users = np.repeat(np.arange(400), 6)
    y = (rng.random(users.size) < 0.35).astype(np.int8)
    a, b = split_validation(users)
    assert not (a & b).any() and (a | b).all(), "folds must partition every row"
    for u in np.unique(users):
        m = users == u
        assert a[m].all() or b[m].all(), "a user must sit entirely in one fold"

    incumbent = rng.random(users.size)
    genuine = y + rng.normal(0, 0.6, users.size)          # better on both folds
    ok, detail = accept(genuine, incumbent, users, y, a, b)
    assert ok, detail

    # a candidate that only wins on the selection fold must be refused
    fold_only = incumbent.copy()
    fold_only[a] = y[a] + rng.normal(0, 0.05, int(a.sum()))
    fold_only[b] = -y[b] + rng.normal(0, 0.05, int(b.sum()))
    ok, detail = accept(fold_only, incumbent, users, y, a, b)
    assert not ok and detail["fold_a_gain"] > 0 > detail["fold_b_gain"], detail
    print("ok  selection.accept")


if __name__ == "__main__":
    demo()
