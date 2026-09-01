"""How much of the wall is drift, and how much is the features themselves?

Fit and score inside ONE split with user-disjoint k-fold CV. No date boundary is crossed, so
the result is what these features are worth when the train and score distributions match.
The gap against the same model trained on the real (earlier) train split is the cost of drift.
"""
import argparse

import lightgbm as lgb
import numpy as np

from limits.features import build
from limits.probe import BASE
from pipeline.evaluate import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=300)
    a = ap.parse_args()
    groups = ["item_cat", "ctx", "item_num", "item_ids", "user_cat", "user_ids", "user_num"]
    X, names, cat, sp = build(a.split, groups, hist=False, sess=True)
    y = np.asarray(sp.y, dtype=np.float32)
    uid = np.asarray(sp.user_id)
    users = np.unique(uid)
    fold_of_user = np.arange(len(users)) % a.folds
    fold = fold_of_user[np.searchsorted(users, uid)]

    oof = np.zeros(len(y))
    for f in range(a.folds):
        tr, va = fold != f, fold == f
        ds = lgb.Dataset(X[tr], label=y[tr], feature_name=names, categorical_feature=cat)
        m = lgb.train(BASE, ds, num_boost_round=a.rounds)
        oof[va] = m.predict(X[va])
        print(f"  fold {f} done", flush=True)
    r = evaluate(uid, y, oof)
    print(f"{a.split} in-distribution {a.folds}-fold CV: gauc={r['gauc']:.4f} "
          f"ndcg@5={r['ndcg@5']:.4f} primary={r['primary']:.4f}")


if __name__ == "__main__":
    main()
