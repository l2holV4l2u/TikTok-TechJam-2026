"""Final stack: average within model family, then weight families on validation only."""
import itertools
import json

import numpy as np

from limits.blend import per_user_rank
from pipeline.data import load
from pipeline.evaluate import evaluate

FAMILIES = {
    "deepfm": [f"deepfm_s{i}" for i in range(6)],
    "dcn": [f"dcn_s{i}" for i in range(3)],
    "gbdt": [f"gbdt_s{i}" for i in range(5)],
    "fm": ["fm_bce", "fm_bce_s1", "fm_bce_s2"],
    "itemcf": ["itemcf"],
}


def family_ranks(split, names, uid):
    return np.mean([per_user_rank(uid, np.load(f"limits/out/{n}.npz")[split]) for n in names], axis=0)


def main(extra=()):
    va, te = load("valid"), load("test")
    fams = dict(FAMILIES)
    for name in extra:
        fams[name] = [name]
    RV = {f: family_ranks("valid", n, va.user_id) for f, n in fams.items()}
    RT = {f: family_ranks("test", n, te.user_id) for f, n in fams.items()}
    for f in fams:
        print(f"  {f:8s} valid={evaluate(va.user_id, va.y, RV[f])['primary']:.4f} "
              f"test={evaluate(te.user_id, te.y, RT[f])['primary']:.4f}")
    keys = list(fams)
    steps = np.arange(0, 1.0001, 0.1)
    best = None
    for w in itertools.product(steps, repeat=len(keys)):
        if abs(sum(w) - 1) > 1e-6:
            continue
        m = evaluate(va.user_id, va.y, sum(wi * RV[k] for wi, k in zip(w, keys)))["primary"]
        if best is None or m > best[0]:
            best = (m, w)
    m, w = best
    mt = evaluate(te.user_id, te.y, sum(wi * RT[k] for wi, k in zip(w, keys)))
    print("weights", {k: round(x, 2) for k, x in zip(keys, w)})
    print(f"STACK valid={m:.4f} test={mt['primary']:.4f} gauc={mt['gauc']:.4f} ndcg@5={mt['ndcg@5']:.4f}")
    return m, mt


if __name__ == "__main__":
    import sys
    main(extra=tuple(sys.argv[1:]))
