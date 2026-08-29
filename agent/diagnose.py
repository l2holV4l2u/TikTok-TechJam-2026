"""Where an iteration's model loses, measured on its own validation predictions.

The agent has `evaluate(per_user=True)` available and used it in 3 of 358 scripts. It proposes
improvements without ever measuring which users it ranks badly. This computes that breakdown
from the iteration's own scores and hands it back as context, so the next proposal can target a
measured weakness instead of a guess.

This is the agent's own error profile, not a human finding: every number comes from the model
the agent just trained, and no conclusion is drawn for it.
"""
import numpy as np

from pipeline.evaluate import evaluate

BINS = ((1, 1), (2, 2), (3, 4), (5, 7), (8, 15), (16, 40), (41, 10 ** 9))


def segment_report(user_id, labels, scores) -> str:
    """A compact table of nDCG@5 against the per-segment ceiling, bucketed by impressions."""
    user_id = np.asarray(user_id)
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if user_id.size == 0:
        return ""

    ours = evaluate(user_id, labels, scores, per_user=True)["per_user"]
    # scoring the label itself is a perfect ranking: the best any model could do per user
    ceil = evaluate(user_id, labels, np.asarray(labels, dtype=np.float64), per_user=True)["per_user"]

    n = np.asarray(ours["n_impressions"])
    nd, ndc = np.asarray(ours["ndcg@5"]), np.asarray(ceil["ndcg@5"])
    auc = np.asarray(ours["auc"])
    total_gap = float((ndc - nd).sum())
    if total_gap <= 0:
        return ""

    rows = ["impressions  users   nDCG@5   ceiling   gap   share_of_total_gap   GAUC"]
    for lo, hi in BINS:
        m = (n >= lo) & (n <= hi)
        if not m.any():
            continue
        label = str(lo) if lo == hi else (f"{lo}+" if hi > 10 ** 8 else f"{lo}-{hi}")
        seg_auc = auc[m]
        seg_auc = seg_auc[np.isfinite(seg_auc)]
        a = f"{seg_auc.mean():.4f}" if seg_auc.size else "  n/a "
        rows.append(f"{label:<12} {m.sum():>5}  {nd[m].mean():.4f}   {ndc[m].mean():.4f}  "
                    f"{(ndc[m]-nd[m]).mean():.4f}   {100*(ndc[m]-nd[m]).sum()/total_gap:>5.1f}%   {a}")
    zero = int((np.asarray(ours["n_positives"]) == 0).sum())
    rows.append(f"({zero} of {n.size} users have no positive label: their nDCG is 0 for any "
                f"model and no method can change it)")
    return "\n".join(rows)


def drift_report() -> str:
    """How far train differs from validation, measured on the splits themselves.

    A deliberately over-powered model reaches 0.9245 primary in-sample on train and 0.5868 on
    validation -- worse than the five-field baseline it dwarfs in capacity. Capacity and
    feature expressiveness are therefore not the limit; transfer across the date boundary is.
    The controller measured that boundary and said nothing about it, so every proposal was
    aimed at capacity. These are distribution facts about the splits, not a recommendation.
    """
    import numpy as np
    from pipeline.data import load

    rows = ["split   rows        users   rows/user(med)  users with no positive"]
    for name in ("train", "valid"):
        s = load(name)
        u = np.asarray(s.user_id)
        order = np.argsort(u, kind="stable")
        us, ys = u[order], np.asarray(s.y)[order]
        starts = np.flatnonzero(np.r_[True, us[1:] != us[:-1]])
        sizes = np.diff(np.r_[starts, us.size])
        pos = np.add.reduceat(ys.astype(np.int64), starts)
        rows.append(f"{name:<7} {us.size:>9,}  {sizes.size:>6,}  {np.median(sizes):>13.0f}  "
                    f"{100.0 * (pos == 0).mean():>20.1f}%")
    rows.append("Test is the ten days after validation; the same drift continues into it.")
    return chr(10).join(rows)


def demo() -> None:
    rng = np.random.default_rng(0)
    users = np.repeat(np.arange(300), rng.integers(1, 30, 300))
    y = (rng.random(users.size) < 0.3).astype(np.int8)

    perfect = segment_report(users, y, y.astype(float))
    assert perfect == "", "a perfect ranking has no gap and should report nothing"

    noise = segment_report(users, y, rng.random(users.size))
    assert "impressions" in noise and "share_of_total_gap" in noise
    shares = [float(l.split("%")[0].split()[-1]) for l in noise.splitlines()[1:] if "%" in l]
    assert abs(sum(shares) - 100.0) < 0.5, shares          # shares must partition the gap
    assert segment_report([], [], []) == ""                 # empty input must not raise
    print("ok  segment_report")
    print(noise)


if __name__ == "__main__":
    demo()
