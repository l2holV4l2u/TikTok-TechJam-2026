import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.evaluate import evaluate
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")  # Features only; never inspect test labels.

print(f"ROWS train={len(tr.user_id)} valid={len(va.user_id)} test={len(te.user_id)}")
print(f"SHAPES Xfields={len(tr.X)} Xscalar={all(np.asarray(x).shape == tr.y.shape for x in tr.X.values())} "
      f"numfields={len(tr.num)} y={tr.y.shape} uid={tr.user_id.shape} time={tr.time_ms.shape}")
print(f"LABEL train={tr.y.mean():.4f} valid={va.y.mean():.4f} "
      f"pos={int(tr.y.sum())}/{int(va.y.sum())}")

def user_summary(name, uid, y):
    _, inv, n = np.unique(uid, return_inverse=True, return_counts=True)
    p = np.bincount(inv, weights=y, minlength=len(n))
    qn = np.percentile(n, [10, 25, 50, 75, 90, 99])
    qp = np.percentile(p, [10, 25, 50, 75, 90, 99])
    print(f"USER {name} N={len(n)} impQ={np.round(qn,1).tolist()} posQ={np.round(qp,1).tolist()}")
    print(f"USERLABEL {name} zero={(p==0).mean():.3f} all={(p==n).mean():.3f} "
          f"eligible={((p>0)&(p<n)).mean():.3f} meanPos={p.mean():.2f}")

user_summary("train", tr.user_id, tr.y)
user_summary("valid", va.user_id, va.y)

train_dates = np.unique(tr.date)
valid_dates = np.unique(va.date)
tr_dm = [tr.y[tr.date == d].mean() for d in train_dates]
va_dm = [va.y[va.date == d].mean() for d in valid_dates]
print("DATE train=" + ",".join(f"{d}:{m:.3f}" for d, m in zip(train_dates, tr_dm)))
print("DATE valid=" + ",".join(f"{d}:{m:.3f}" for d, m in zip(valid_dates, va_dm)))

def novelty(name, a, b, c):
    seen = np.unique(a)
    vb = np.isin(b, seen)
    vc = np.isin(c, seen)
    print(f"NOVEL {name} uniq={len(seen)}/{len(np.unique(b))}/{len(np.unique(c))} "
          f"rowNewV={1-vb.mean():.3f} rowNewT={1-vc.mean():.3f}")

novelty("user", tr.user_id, va.user_id, te.user_id)
novelty("video", tr.video_id, va.video_id, te.video_id)
novelty("author", tr.X["author_id"], va.X["author_id"], te.X["author_id"])

order = np.lexsort((np.arange(len(tr.user_id)), tr.time_ms, tr.user_id))
same = ((tr.user_id[order][1:] == tr.user_id[order][:-1]) &
        (tr.time_ms[order][1:] == tr.time_ms[order][:-1]))
print(f"TIME trainRange={tr.time_ms.min()}..{tr.time_ms.max()} adjacentSameBatch={same.mean():.3f}")

base = float(tr.y.mean())
print("FIELD format: name K train/valid/test_unique zeroV newV smoothedTE_primary")
for name in tr.X:
    xt = np.asarray(tr.X[name])
    xv = np.asarray(va.X[name])
    xe = np.asarray(te.X[name])
    k = FEATURE_CARDINALITIES[name]
    cnt = np.bincount(xt, minlength=k).astype(np.float64)
    pos = np.bincount(xt, weights=tr.y, minlength=k)
    mean = (pos + 20.0 * base) / (cnt + 20.0)
    pred = mean[xv]
    met = evaluate(va.user_id, va.y, pred)
    seen = cnt > 0
    newv = np.mean((xv != 0) & (~seen[xv]))
    print(f"F {name} K={k} U={len(np.unique(xt))}/{len(np.unique(xv))}/{len(np.unique(xe))} "
          f"z={np.mean(xv==0):.3f} n={newv:.3f} P={met['primary']:.4f}")

for name, x in tr.num.items():
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    z = np.log1p(np.maximum(x[ok], 0))
    yy = tr.y[ok].astype(np.float64)
    corr = np.corrcoef(z, yy)[0, 1] if z.size and np.std(z) > 0 else 0.0
    q = np.percentile(x[ok], [10, 50, 90, 99]) if ok.any() else [np.nan] * 4
    m0 = np.nanmean(x[(tr.y == 0) & ok])
    m1 = np.nanmean(x[(tr.y == 1) & ok])
    print(f"NUM {name} miss={1-ok.mean():.3f} Q={np.round(q,1).tolist()} "
          f"mean0/1={m0:.1f}/{m1:.1f} logCorr={corr:.3f}")

for entity in ("video_id", "author_id"):
    h = historical_features("valid", key=entity)
    desc = ",".join(f"{k}:{np.asarray(v).shape}:{np.isfinite(v).mean():.2f}"
                    for k, v in h.items())
    print(f"HIST {entity} {desc}")