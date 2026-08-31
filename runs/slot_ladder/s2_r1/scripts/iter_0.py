import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")

def corr(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return 0.0
    a, b = a[ok], b[ok]
    a -= a.mean()
    b -= b.mean()
    den = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / den) if den > 0 else 0.0

def user_stats(s):
    _, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    qn = np.quantile(n, [0, .25, .5, .75, .9, .99, 1])
    qp = np.quantile(p, [0, .25, .5, .75, .9, .99, 1])
    return n, p, qn, qp

print(f"ROWS tr/va/te={len(tr.user_id)}/{len(va.user_id)}/{len(te.user_id)} "
      f"users={np.unique(tr.user_id).size}/{np.unique(va.user_id).size}/{np.unique(te.user_id).size}")
shape_ok = all(np.asarray(x).shape == (len(tr.user_id),) for x in tr.X.values())
print(f"SCHEMA Xfields={len(tr.X)} scalar_per_row={shape_ok} num={list(tr.num)} aux_n={len(tr.aux)}")

for name, s in [("tr", tr), ("va", va)]:
    n, p, qn, qp = user_stats(s)
    print(f"USER {name} y={s.y.mean():.4f} noPos={(p==0).mean():.3f} "
          f"allPos={(p==n).mean():.3f} mixed={((p>0)&(p<n)).mean():.3f}")
    print(f"USERQ {name} n={np.round(qn,1).tolist()} pos={np.round(qp,1).tolist()}")

for name, s in [("tr", tr), ("va", va)]:
    ds = []
    for d in np.unique(s.date):
        z = s.date == d
        ds.append(f"{int(d)%10000:04d}:{z.sum()}:{s.y[z].mean():.3f}")
    print(f"DATE {name} day:n:rate " + " ".join(ds))

for k in ["user_id", "video_id", "author_id"]:
    a = np.unique(tr.X[k])
    b = np.unique(va.X[k])
    c = np.unique(te.X[k])
    print(f"OVERLAP {k} uniq tr/va/te={a.size}/{b.size}/{c.size} "
          f"newU va/te={(~np.isin(b,a)).mean():.3f}/{(~np.isin(c,a)).mean():.3f}")

global_rate = float(tr.y.mean())
records = []
for k in tr.X:
    xt = np.asarray(tr.X[k], dtype=np.int64)
    xv = np.asarray(va.X[k], dtype=np.int64)
    xe = np.asarray(te.X[k], dtype=np.int64)
    card = int(FEATURE_CARDINALITIES[k])
    cnt = np.bincount(xt, minlength=card)
    pos = np.bincount(xt, weights=tr.y, minlength=card)
    rate = (pos + 20.0 * global_rate) / (cnt + 20.0)
    rv = corr(rate[xv], va.y)
    seen = cnt > 0
    uv = float((~seen[xv]).mean())
    ue = float((~seen[xe]).mean())
    top = float(cnt.max() / len(xt))
    records.append((abs(rv), k, card, np.count_nonzero(cnt),
                    np.unique(xv).size, np.unique(xe).size, top, uv, ue, rv))

print("FIELDS sorted by |valid corr|: card, unique tr/va/te, trainTop, unseenRows va/te, corr")
for rec in sorted(records, reverse=True):
    _, k, card, ut, uvn, uen, top, uv, ue, rv = rec
    print(f"F {k} c={card} u={ut}/{uvn}/{uen} top={top:.3f} "
          f"new={uv:.3f}/{ue:.3f} r={rv:+.3f}")

print("NUM train missing,q10/q50/q90; valid/test median; valid log-corr")
for k in tr.num:
    a = np.asarray(tr.num[k], np.float64)
    b = np.asarray(va.num[k], np.float64)
    c = np.asarray(te.num[k], np.float64)
    finite = np.isfinite(a)
    q = np.nanquantile(a, [.1, .5, .9]) if finite.any() else [np.nan] * 3
    med = float(q[1]) if finite.any() else 0.0
    bs = np.where(np.isfinite(b), b, med)
    bs = np.sign(bs) * np.log1p(np.abs(bs))
    print(f"N {k} miss={1-finite.mean():.3f} q={np.round(q,2).tolist()} "
          f"medV/T={np.nanmedian(b):.2f}/{np.nanmedian(c):.2f} r={corr(bs,va.y):+.3f}")

for key in ["video_id", "author_id"]:
    h = historical_features("valid", key=key)
    names = list(h)
    print(f"HIST {key} n={len(names)} keys={','.join(names)[:220]}")
    parts = []
    for name in names[:8]:
        x = np.asarray(h[name])
        f = np.isfinite(x)
        med = float(np.median(x[f])) if f.any() else np.nan
        parts.append(f"{name}:sh{x.shape},nan{1-f.mean():.2f},m{med:.3g}")
    print("HSTAT " + " | ".join(parts)[:330])