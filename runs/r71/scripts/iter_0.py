import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")

def qstr(a):
    q = np.quantile(a, [0, .25, .5, .75, .9, .99, 1])
    return "/".join(f"{x:.0f}" for x in q)

def user_stats(s, name):
    _, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y).astype(np.int64)
    z = np.mean(p == 0)
    allp = np.mean(p == n)
    mixed = np.mean((p > 0) & (p < n))
    print(f"USER {name} U={len(n)} rowsQ={qstr(n)} posQ={qstr(p)} "
          f"zero={z:.3f} all={allp:.3f} mixed={mixed:.3f}")

def date_stats(s, name):
    d, inv = np.unique(s.date, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    vals = ",".join(f"{x}:{r/n0:.3f}" for x, r, n0 in zip(d, p, n))
    print(f"DATE {name} {vals}")

def unseen_fraction(ref_unique, x):
    idx = np.searchsorted(ref_unique, x)
    hit = np.zeros(len(x), dtype=bool)
    ok = idx < len(ref_unique)
    hit[ok] = ref_unique[idx[ok]] == x[ok]
    return 1.0 - hit.mean()

def normalized_mi(x, y):
    n = np.bincount(x)
    p = np.bincount(x, weights=y, minlength=len(n))
    used = n > 0
    n = n[used].astype(np.float64)
    p = p[used]
    total = n.sum()
    py = y.mean()
    hy = -(py*np.log(py + 1e-15) + (1-py)*np.log(1-py + 1e-15))
    mi = 0.0
    for c, pc in ((p, py), (n-p, 1-py)):
        m = c > 0
        joint = c[m] / total
        mi += np.sum(joint * np.log((c[m] / n[m]) / pc))
    return mi / hy if hy > 0 else 0.0

print(f"SHAPE train={len(tr.user_id)} valid={len(va.user_id)} test={len(te.user_id)} "
      f"Xfields={len(tr.X)} num={len(tr.num)} scalarX={all(np.asarray(v).ndim==1 for v in tr.X.values())}")
print(f"LABEL train pos={tr.y.sum()} rate={tr.y.mean():.4f} "
      f"valid pos={va.y.sum()} rate={va.y.mean():.4f}")
date_stats(tr, "tr")
date_stats(va, "va")
user_stats(tr, "tr")
user_stats(va, "va")

for key in ("user_id", "video_id", "author_id"):
    utr = np.unique(tr.X[key])
    utrva = np.union1d(utr, np.unique(va.X[key]))
    print(f"COLD {key} valid_vs_tr={unseen_fraction(utr, va.X[key]):.3f} "
          f"test_vs_trva={unseen_fraction(utrva, te.X[key]):.3f}")

for name in sorted(tr.num):
    a = np.asarray(tr.num[name], dtype=np.float64)
    b = np.asarray(va.num[name], dtype=np.float64)
    good = np.isfinite(a)
    corr = np.corrcoef(np.log1p(np.maximum(a[good], 0)), tr.y[good])[0, 1] if good.sum() > 2 else np.nan
    qs = np.quantile(a[good], [0, .5, .9, .99, 1])
    print(f"NUM {name} miss={1-good.mean():.3f}/{np.mean(~np.isfinite(b)):.3f} "
          f"q={','.join(f'{x:.1f}' for x in qs)} logcorr={corr:.3f}")

for name in sorted(tr.X):
    x, xv, xt = tr.X[name], va.X[name], te.X[name]
    u = np.unique(x)
    uv = np.unique(xv)
    ut_ref = np.union1d(u, uv)
    cnt = np.bincount(x)
    nz = cnt[cnt > 0]
    top = nz.max() / len(x)
    singleton = np.mean(nz == 1)
    print(f"F {name} C={FEATURE_CARDINALITIES[name]} tr={len(u)} va={len(uv)} "
          f"top={top:.3f} s1={singleton:.3f} uv={unseen_fraction(u,xv):.3f} "
          f"ut={unseen_fraction(ut_ref,xt):.3f} nmi={normalized_mi(x,tr.y):.4f}")

for key in ("video_id", "author_id"):
    h = historical_features("valid", key=key)
    desc = ",".join(f"{k}:{np.asarray(v).shape}/{np.asarray(v).dtype}" for k, v in h.items())
    print(f"HIST {key} n={len(h)} {desc}"[:350])