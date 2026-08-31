import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")

def grouped_auc(scores, pos, neg):
    keep = (pos + neg) > 0
    scores, pos, neg = scores[keep], pos[keep], neg[keep]
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return np.nan
    order = np.argsort(scores, kind="mergesort")
    scores, pos, neg = scores[order], pos[order], neg[order]
    starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
    gp = np.add.reduceat(pos, starts)
    gn = np.add.reduceat(neg, starts)
    neg_before = np.cumsum(gn) - gn
    return float(np.sum(gp * (neg_before + 0.5 * gn)) / (P * N))

def user_summary(s):
    u, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    nq = np.quantile(n, [0.5, 0.9, 0.99])
    pq = np.quantile(p, [0.5, 0.9, 0.99])
    return len(u), nq, pq, 100.0 * np.mean(p == 0)

def date_summary(s):
    return ",".join(
        f"{int(d)}:{np.mean(s.y[s.date == d]):.3f}"
        for d in np.unique(s.date)
    )

print(f"DATA tr={len(tr.y)} va={len(va.y)} y={tr.y.mean():.4f}/{va.y.mean():.4f}")
print("DATEtr " + date_summary(tr))
print("DATEva " + date_summary(va))

for name, s in [("USERtr", tr), ("USERva", va)]:
    nu, nq, pq, z = user_summary(s)
    print(f"{name} n={nu} rows50/90/99={nq.astype(int)} pos50/90/99={pq.astype(int)} zero={z:.1f}%")

order = np.lexsort((np.arange(len(tr.y)), tr.time_ms, tr.user_id))
same_batch = ((tr.user_id[order][1:] == tr.user_id[order][:-1]) &
              (tr.time_ms[order][1:] == tr.time_ms[order][:-1]))
print(f"SHAPE cat={tr.X[next(iter(tr.X))].shape} num={tr.num[next(iter(tr.num))].shape} "
      f"fields={len(tr.X)}/{len(tr.num)} same_user_time_adj={same_batch.mean():.3f}")

for f in ["user_id", "video_id", "author_id"]:
    C = FEATURE_CARDINALITIES[f]
    seen = np.bincount(tr.X[f], minlength=C) > 0
    vu = np.unique(va.X[f])
    print(f"OVERLAP {f} unseen_rows={100*np.mean(~seen[va.X[f]]):.2f}% "
          f"unseen_valid_ids={100*np.mean(~seen[vu]):.2f}%")

for f, a in tr.num.items():
    finite = np.isfinite(a)
    q = np.nanquantile(a, [0.5, 0.95]) if finite.any() else [np.nan, np.nan]
    corr = np.corrcoef(a[finite].astype(np.float64),
                       tr.y[finite].astype(np.float64))[0, 1] if finite.sum() > 2 else np.nan
    vb = va.num[f]
    print(f"NUM {f} miss={100*np.mean(~finite):.1f}% q50/95={q[0]:.2g}/{q[1]:.2g} "
          f"corr={corr:.3f} va50={np.nanmedian(vb):.2g}")

dates = np.unique(tr.date)
early = tr.date < dates[-3]
late = ~early
prior = float(tr.y[early].mean())

for f in tr.X:
    C = FEATURE_CARDINALITIES[f]
    x, xv = tr.X[f], va.X[f]
    full_count = np.bincount(x, minlength=C)
    valid_count = np.bincount(xv, minlength=C)
    seen = full_count > 0

    xe, ye = x[early], tr.y[early]
    ec = np.bincount(xe, minlength=C).astype(np.float64)
    ep = np.bincount(xe, weights=ye, minlength=C)
    rates = (ep + 20.0 * prior) / (ec + 20.0)

    xl, yl = x[late], tr.y[late]
    lp = np.bincount(xl, weights=yl, minlength=C)
    ln = np.bincount(xl, weights=1 - yl, minlength=C)
    auc = grouped_auc(rates, lp, ln)

    print(f"F {f} C/t/v={C}/{seen.sum()}/{np.count_nonzero(valid_count)} "
          f"top={100*full_count.max()/len(x):.1f}% uv={100*np.mean(~seen[xv]):.1f}% A={auc:.3f}")

h = historical_features("train", key="video_id")
desc = ",".join(f"{k}:{np.asarray(v).shape}/{np.asarray(v).dtype}" for k, v in h.items())
print("HIST video " + desc[:350])