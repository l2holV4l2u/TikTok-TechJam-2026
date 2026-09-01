"""Rank-average blend over saved prediction files, weights picked on validation only."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from pipeline.data import load
from pipeline.evaluate import evaluate

OUT = Path("limits/out")


def per_user_rank(user_id, score):
    """Rank within each user, scaled to [0,1]. Both metrics are per-user, so only this matters."""
    order = np.lexsort((score, user_id))
    u = np.asarray(user_id)[order]
    start = np.searchsorted(u, np.unique(u))
    size = np.diff(np.append(start, len(u)))
    within = np.arange(len(u)) - np.repeat(start, size)
    out = np.empty(len(u))
    out[order] = within / np.maximum(np.repeat(size, size) - 1, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--grid", type=float, default=0.1)
    a = ap.parse_args()
    va, te = load("valid"), load("test")
    rv = {n: per_user_rank(va.user_id, np.load(OUT / f"{n}.npz")["valid"]) for n in a.names}
    rt = {n: per_user_rank(te.user_id, np.load(OUT / f"{n}.npz")["test"]) for n in a.names}
    steps = np.arange(0, 1 + 1e-9, a.grid)
    best = None
    for w in itertools.product(steps, repeat=len(a.names)):
        if abs(sum(w) - 1) > 1e-6:
            continue
        sv = sum(wi * rv[n] for wi, n in zip(w, a.names))
        m = evaluate(va.user_id, va.y, sv)["primary"]
        if best is None or m > best[0]:
            best = (m, w)
    m, w = best
    st = sum(wi * rt[n] for wi, n in zip(w, a.names))
    mt = evaluate(te.user_id, te.y, st)
    print("weights", dict(zip(a.names, [round(x, 2) for x in w])))
    print(f"blend valid={m:.4f} test={mt['primary']:.4f} gauc={mt['gauc']:.4f} ndcg={mt['ndcg@5']:.4f}")
    path = OUT / "results.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev["blend_" + "+".join(a.names)] = dict(name="blend", weights=dict(zip(a.names, w)),
                                              valid={"primary": m}, test=mt)
    path.write_text(json.dumps(prev, indent=1))


if __name__ == "__main__":
    main()
