"""Hand-built model probe: how high can the score go if a human optimizes directly?

Off the submission path. Reads test labels on purpose -- the point is to bound what is
achievable, not to produce an entry. Reports both the honest number (valid-selected
iteration) and the oracle number (best iteration by test), so the gap is visible.

Run: python -m limits.probe [--only name,name] [--rounds 800]
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from limits.features import build
from pipeline.evaluate import evaluate

OUT = Path("limits/out")
BASE = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
            verbose=-1, num_threads=16, seed=0, deterministic=True)

ITEM = ["item_cat", "ctx", "item_num"]
ALL = ITEM + ["item_ids", "user_cat", "user_ids", "user_num"]

CONFIGS = {
    "hist_only":     dict(groups=[]),
    "item":          dict(groups=ITEM),
    "item_ids":      dict(groups=ITEM + ["item_ids"]),
    "item_user":     dict(groups=ITEM + ["user_cat", "user_num"]),
    "everything":    dict(groups=ALL),
    "item_rank":     dict(groups=ITEM, objective="lambdarank"),
    "item_recency":  dict(groups=ITEM, decay=0.15),
    "item_deep":     dict(groups=ITEM, num_leaves=255, min_data_in_leaf=20, lambda_l2=1.0),
    "item_tiny":     dict(groups=ITEM, num_leaves=8, min_data_in_leaf=1000, lambda_l2=50.0),
    "no_hist":       dict(groups=ALL, hist=False),
    "cats":          dict(groups=ITEM + ["item_ids", "user_cat", "user_num"], hist=False),
    "cats_hist":     dict(groups=ITEM + ["item_ids", "user_cat", "user_num"]),
    "cats_aff":      dict(groups=ITEM + ["item_ids", "user_cat", "user_num"], hist=False, aff=True),
    "cats_slow":     dict(groups=ITEM + ["item_ids", "user_cat", "user_num"], hist=False,
                          learning_rate=0.02, num_leaves=63),
    "best":          dict(groups=ALL, hist=False, sess=True, aff=True, ctxnorm=True),
    "no_hist_sess":  dict(groups=ALL, hist=False, sess=True),
    "cats_sess":     dict(groups=ITEM + ["item_ids", "user_cat", "user_num"], hist=False, sess=True),
    "cats_sess_ctx": dict(groups=ITEM + ["item_ids", "user_cat", "user_num"], hist=False, sess=True, ctxnorm=True),
    "sess_only":     dict(groups=ITEM, hist=False, sess=True),
    "item_aff":      dict(groups=ITEM, aff=True),
    "item_aff_slow": dict(groups=ITEM, aff=True, learning_rate=0.02, num_leaves=15),
    "aff_rank":      dict(groups=ITEM, aff=True, objective="lambdarank"),
}


def _weights(dates, decay):
    day = np.asarray(dates, dtype=np.int64)
    order = np.unique(day)
    age = (len(order) - 1) - np.searchsorted(order, day)
    return np.exp(-decay * age).astype(np.float32)


def run(name, cfg, rounds, step):
    groups, hist, aff = cfg["groups"], cfg.get("hist", True), cfg.get("aff", False)
    sess, ctxnorm = cfg.get("sess", False), cfg.get("ctxnorm", False)
    params = {**BASE, **{k: v for k, v in cfg.items() if k not in ("groups", "hist", "decay", "aff", "sess", "ctxnorm")}}
    t0 = time.time()
    Xtr, names, cat, tr = build("train", groups, hist, aff, sess=sess, ctxnorm=ctxnorm)
    Xva, _, _, va = build("valid", groups, hist, aff, sess=sess, ctxnorm=ctxnorm)
    Xte, _, _, te = build("test", groups, hist, aff, sess=sess, ctxnorm=ctxnorm)
    ytr = np.asarray(tr.y, dtype=np.float32)

    w = _weights(tr.date, cfg["decay"]) if "decay" in cfg else None
    if params["objective"] == "lambdarank":
        order = np.argsort(np.asarray(tr.user_id), kind="stable")
        Xtr, ytr = Xtr[order], ytr[order]
        w = w[order] if w is not None else None
        grp = np.bincount(np.searchsorted(np.unique(tr.user_id), np.asarray(tr.user_id)[order]))
        params.update(lambdarank_truncation_level=20, label_gain=[0, 1])
    else:
        grp = None

    dtrain = lgb.Dataset(Xtr, label=ytr, weight=w, group=grp, feature_name=names,
                         categorical_feature=cat, free_raw_data=False)
    model = lgb.train(params, dtrain, num_boost_round=rounds)

    rows = []
    ks = sorted(set(list(range(5, 51, 5)) + list(range(60, rounds + 1, step))))
    for k in ks:
        pv = model.predict(Xva, num_iteration=k)
        pt = model.predict(Xte, num_iteration=k)
        rows.append((k, evaluate(va.user_id, va.y, pv), evaluate(te.user_id, te.y, pt)))
    best = max(rows, key=lambda r: r[1]["primary"])
    oracle = max(rows, key=lambda r: r[2]["primary"])

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}.npz",
             valid=model.predict(Xva, num_iteration=best[0]),
             test=model.predict(Xte, num_iteration=best[0]))
    res = dict(name=name, rounds=best[0], n_features=len(names), secs=round(time.time() - t0, 1),
               valid=best[1], test=best[2], oracle_test=oracle[2]["primary"], oracle_rounds=oracle[0])
    _record(res)
    print(f"{name:14s} k={best[0]:4d} valid={best[1]['primary']:.4f} "
          f"test={best[2]['primary']:.4f} (oracle {oracle[2]['primary']:.4f}) {res['secs']}s", flush=True)
    return res


def _record(res):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev[res["name"]] = res
    path.write_text(json.dumps(prev, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--step", type=int, default=20)
    a = ap.parse_args()
    picked = a.only.split(",") if a.only else list(CONFIGS)
    for n in picked:
        run(n, CONFIGS[n], a.rounds, a.step)


if __name__ == "__main__":
    main()
