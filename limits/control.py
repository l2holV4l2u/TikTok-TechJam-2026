"""Control for the in-distribution CV: same training-set SIZE, but from the earlier train window.

If a same-size out-of-distribution sample scores as well as the in-distribution one, then the
date boundary is not what is holding the score down -- the feature set is.
"""
import argparse

import lightgbm as lgb
import numpy as np

from limits.features import build
from limits.probe import BASE
from pipeline.evaluate import evaluate

GROUPS = ["item_cat", "ctx", "item_num", "item_ids", "user_cat", "user_ids", "user_num"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=136_000)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    Xtr, names, cat, tr = build("train", GROUPS, hist=False, sess=True)
    Xte, _, _, te = build("test", GROUPS, hist=False, sess=True)
    ytr = np.asarray(tr.y, dtype=np.float32)
    scores = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ytr), size=a.n, replace=False)
        ds = lgb.Dataset(Xtr[idx], label=ytr[idx], feature_name=names, categorical_feature=cat)
        m = lgb.train(BASE, ds, num_boost_round=a.rounds)
        r = evaluate(te.user_id, te.y, m.predict(Xte))
        scores.append(r["primary"])
        print(f"  seed {seed}: test primary={r['primary']:.4f}", flush=True)
    print(f"train-subsample n={a.n}: test primary {np.mean(scores):.4f} +- {np.std(scores):.4f}")


if __name__ == "__main__":
    main()
