import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES

tr = load("train")
va = load("valid")

def user_stats(s, name):
    _, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    qn = np.quantile(n, [0, .25, .5, .75, .9, .99, 1])
    qp = np.quantile(p, [0, .25, .5, .75, .9, .99, 1])
    print(f"USR {name} U={len(n)} rowsQ={qn.astype(int).tolist()} posQ={qp.astype(int).tolist()}")
    print(f"USR {name} zero={np.mean(p==0):.3f} all={np.mean(p==n):.3f} mixed={np.mean((p>0)&(p<n)):.3f}")

def signal(x, y, alpha=20.0):
    m = int(x.max()) + 1
    n = np.bincount(x, minlength=m).astype(np.float64)
    p = np.bincount(x, weights=y, minlength=m)
    mu = float(np.mean(y))
    r = (p + alpha * mu) / (n + alpha)
    den = max(mu * (1.0 - mu), 1e-12)
    return float(np.sum(n * (r - mu) ** 2) / (len(y) * den)), int(np.count_nonzero(n)), float(n.max() / len(y))

print(f"ROWS train={len(tr.y)} valid={len(va.y)} Xfields={len(tr.X)} num={len(tr.num)}")
print(f"SHAPE Xscalar={all(v.shape==(len(tr.y),) for v in tr.X.values())} "
      f"dtype={next(iter(tr.X.values())).dtype} y={tr.y.shape}/{tr.y.dtype} "
      f"time={tr.time_ms.shape}/{tr.time_ms.dtype}")
print(f"LABEL train={tr.y.mean():.5f} pos={int(tr.y.sum())} valid={va.y.mean():.5f} pos={int(va.y.sum())}")
user_stats(tr, "tr")
user_stats(va, "va")

for s, name in [(tr, "tr"), (va, "va")]:
    ds = []
    for d in np.unique(s.date):
        z = s.date == d
        ds.append(f"{d%10000:04d}:{z.sum()}/{s.y[z].mean():.3f}")
    print(f"DATE {name} " + " ".join(ds))

for name in sorted(tr.X):
    xt, xv = tr.X[name], va.X[name]
    st, ot, top = signal(xt, tr.y)
    sv, ov, _ = signal(xv, va.y)
    seen = np.zeros(max(int(xt.max()), int(xv.max())) + 1, dtype=bool)
    seen[xt] = True
    if len(seen):
        seen[0] = False
    unseen = float(np.mean(~seen[xv]))
    print(f"F {name} K={FEATURE_CARDINALITIES[name]} O={ot}/{ov} M={top:.3f} U={unseen:.3f} S={st:.3f}/{sv:.3f}")

for name in sorted(tr.num):
    x = np.asarray(tr.num[name], dtype=np.float64)
    ok = np.isfinite(x)
    if ok.any():
        q = np.quantile(x[ok], [.01, .5, .99])
        yy = tr.y[ok].astype(np.float64)
        corr = np.corrcoef(np.log1p(np.maximum(x[ok], 0)), yy)[0, 1]
        print(f"N {name} miss={1-ok.mean():.3f} q01/50/99={q[0]:.2g}/{q[1]:.2g}/{q[2]:.2g} logcorr={corr:.3f}")
    else:
        print(f"N {name} all_missing")

for key in ["user_id", "video_id", "author_id"]:
    a, b = tr.X[key], va.X[key]
    seen = np.zeros(max(int(a.max()), int(b.max())) + 1, dtype=bool)
    seen[a] = True
    if len(seen):
        seen[0] = False
    uv = np.unique(b)
    print(f"NOV {key} row={np.mean(~seen[b]):.3f} ids={np.mean(~seen[uv]):.3f} valid_ids={len(uv)}")