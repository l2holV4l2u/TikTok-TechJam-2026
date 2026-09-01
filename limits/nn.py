"""Embedding models with a within-user pairwise loss.

Both scored metrics are per-user means, so the training signal that matches them is
"rank this user's positives above this user's negatives", not global calibration.
Sampling a pair by drawing a positive uniformly reproduces GAUC's positive-count weighting.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from limits.features import build as build_tab
from pipeline.data import FEATURE_CARDINALITIES, load
from pipeline.evaluate import evaluate
from pipeline.models import build as build_model

OUT = Path("limits/out")
OFFICIAL = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
WIDE = OFFICIAL + ["video_type", "upload_type", "music_type", "tag", "hour",
                   "user_active_degree", "register_days_bucket"]


def _pairs(user_id, y, rng, n):
    """Sample (pos_row, neg_row) within the same user; positives drawn uniformly."""
    order = np.argsort(user_id, kind="stable")
    u = user_id[order]
    start = np.searchsorted(u, np.unique(u))
    end = np.append(start[1:], len(u))
    yo = y[order]
    # ragged per-user positive/negative row lists, flattened with offsets so sampling is vectorised
    is_pos = yo == 1
    pos_flat, neg_flat, pstart, nstart, plen, nlen = [], [], [], [], [], []
    for s_, e_ in zip(start, end):
        seg, m = order[s_:e_], is_pos[s_:e_]
        pstart.append(len(pos_flat))
        pos_flat.extend(seg[m].tolist()); plen.append(int(m.sum()))
        nstart.append(len(neg_flat)); neg_flat.extend(seg[~m].tolist()); nlen.append(int((~m).sum()))
    pos_flat = np.array(pos_flat); neg_flat = np.array(neg_flat)
    pstart = np.array(pstart); nstart = np.array(nstart)
    plen = np.array(plen); nlen = np.array(nlen)
    keep = np.flatnonzero((plen > 0) & (nlen > 0))
    w = plen[keep] / plen[keep].sum()
    g = keep[rng.choice(len(keep), size=n, p=w)]
    p = pos_flat[pstart[g] + (rng.random(n) * plen[g]).astype(np.int64)]
    q = neg_flat[nstart[g] + (rng.random(n) * nlen[g]).astype(np.int64)]
    return p, q


def _tensors(sp, fields):
    return {f: torch.from_numpy(np.asarray(sp.X[f], dtype=np.int64)) for f in fields}


@torch.no_grad()
def _predict(model, T, n, bs=200_000):
    model.eval()
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, bs):
        out[i:i + bs] = model({f: v[i:i + bs] for f, v in T.items()}).numpy()
    return out


def _concat(splits):
    """Glue several splits into one training set (used to refit on train+valid for test)."""
    parts = [load(s) for s in splits]
    from types import SimpleNamespace
    return SimpleNamespace(
        X={f: np.concatenate([np.asarray(p.X[f]) for p in parts]) for f in parts[0].X},
        y=np.concatenate([np.asarray(p.y) for p in parts]),
        user_id=np.concatenate([np.asarray(p.user_id) for p in parts]),
        date=np.concatenate([np.asarray(p.date) for p in parts]))


def run(name="fm_bpr", fields=OFFICIAL, embed_dim=16, lr=3e-3, epochs=8, batch=8192,
        pairs_per_epoch=2_000_000, loss="bpr", l2=1e-6, seed=0, model_name="fm", save=True,
        fit_splits=("train",), min_date=None, select_epoch=None):
    torch.manual_seed(seed)
    torch.set_num_threads(16)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    tr = _concat(fit_splits) if len(fit_splits) > 1 else load("train")
    if min_date is not None:  # recency: drop training days older than min_date
        keep = np.asarray(tr.date) >= min_date
        from types import SimpleNamespace
        tr = SimpleNamespace(X={f: np.asarray(v)[keep] for f, v in tr.X.items()},
                             y=np.asarray(tr.y)[keep], user_id=np.asarray(tr.user_id)[keep],
                             date=np.asarray(tr.date)[keep])
    va, te = load("valid"), load("test")
    cards = {f: FEATURE_CARDINALITIES[f] for f in fields}
    model = build_model(model_name, cards, embed_dim=embed_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    Ttr, Tva, Tte = _tensors(tr, fields), _tensors(va, fields), _tensors(te, fields)
    ytr = np.asarray(tr.y, dtype=np.float32)
    uid = np.asarray(tr.user_id)

    rows = []
    for ep in range(1, epochs + 1):
        model.train()
        if loss == "bpr":
            p, q = _pairs(uid, ytr, rng, pairs_per_epoch)
            for i in range(0, len(p), batch):
                pi = torch.from_numpy(p[i:i + batch]); qi = torch.from_numpy(q[i:i + batch])
                sp_ = model({f: v[pi] for f, v in Ttr.items()})
                sq_ = model({f: v[qi] for f, v in Ttr.items()})
                l = torch.nn.functional.softplus(sq_ - sp_).mean()
                opt.zero_grad(); l.backward(); opt.step()
        else:
            perm = rng.permutation(len(ytr))
            yt = torch.from_numpy(ytr)
            for i in range(0, len(perm), batch):
                bi = torch.from_numpy(perm[i:i + batch])
                s = model({f: v[bi] for f, v in Ttr.items()})
                l = torch.nn.functional.binary_cross_entropy_with_logits(s, yt[bi])
                opt.zero_grad(); l.backward(); opt.step()
        pv = _predict(model, Tva, len(va.y)); pt = _predict(model, Tte, len(te.y))
        mv, mt = evaluate(va.user_id, va.y, pv), evaluate(te.user_id, te.y, pt)
        rows.append((ep, mv, mt, pv, pt))
        print(f"  ep{ep} valid={mv['primary']:.4f} test={mt['primary']:.4f}", flush=True)

    # fitting on valid makes the valid metric in-sample, so the epoch must come from elsewhere
    best = rows[select_epoch - 1] if select_epoch else max(rows, key=lambda r: r[1]["primary"])
    oracle = max(rows, key=lambda r: r[2]["primary"])
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / f"{name}.npz", valid=best[3], test=best[4])
    res = dict(name=name, epoch=best[0], secs=round(time.time() - t0, 1), fields=len(fields),
               valid=best[1], test=best[2], oracle_test=oracle[2]["primary"])
    print(f"{name:14s} ep={best[0]} valid={best[1]['primary']:.4f} test={best[2]['primary']:.4f} "
          f"(oracle {oracle[2]['primary']:.4f}) {res['secs']}s", flush=True)
    return res


def _record(res):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev[res["name"]] = res
    path.write_text(json.dumps(prev, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="fm_bpr")
    ap.add_argument("--fields", default="official", choices=["official", "wide"])
    ap.add_argument("--model", default="fm")
    ap.add_argument("--loss", default="bpr")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--pairs", type=int, default=2_000_000)
    ap.add_argument("--fit-splits", default="train")
    ap.add_argument("--min-date", type=int, default=None)
    ap.add_argument("--select-epoch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    _record(run(name=a.name, fields=OFFICIAL if a.fields == "official" else WIDE,
                model_name=a.model, loss=a.loss, epochs=a.epochs,
                embed_dim=a.embed_dim, lr=a.lr, pairs_per_epoch=a.pairs,
                fit_splits=tuple(a.fit_splits.split(",")), min_date=a.min_date,
                select_epoch=a.select_epoch, seed=a.seed))
