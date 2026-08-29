import numpy as np
from scipy.stats import rankdata
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")

def qstr(x):
    q = np.quantile(x, [0, .25, .5, .75, .9, .99, 1])
    return "/".join(f"{v:.0f}" for v in q)

def user_stats(s):
    _, inv = np.unique(s.user_id, return_inverse=True)
    nr = np.bincount(inv)
    np_ = np.bincount(inv, weights=s.y)
    return nr, np_

def auc(y, score):
    y = np.asarray(y, dtype=np.int8)
    p = int(y.sum())
    n = len(y) - p
    if p == 0 or n == 0:
        return np.nan
    r = rankdata(score, method="average")
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)

def date_rates(s):
    ds = np.unique(s.date)
    return ",".join(f"{str(int(d))[-2:]}:{s.y[s.date == d].mean():.3f}" for d in ds)

def novelty(name):
    c = FEATURE_CARDINALITIES[name]
    tc = np.bincount(tr.X[name], minlength=c)
    v = va.X[name]
    return np.mean(tc[v] == 0), len(np.unique(tr.X[name])), len(np.unique(v))

print(f"SHAPE tr={len(tr.y)} va={len(va.y)} X={len(tr.X)} "
      f"scalar={all(np.asarray(x).shape == (len(tr.y),) for x in tr.X.values())}")
for tag, s in [("TR", tr), ("VA", va)]:
    nr, np_ = user_stats(s)
    print(f"{tag} y={s.y.mean():.4f} users={len(nr)} zero={np.mean(np_==0):.3f} "
          f"rowsQ={qstr(nr)} posQ={qstr(np_)}")
print("TRdate " + date_rates(tr))
print("VAdate " + date_rates(va))

for name in ["user_id", "video_id", "author_id"]:
    n, ut, uv = novelty(name)
    print(f"NEW {name} rows={n:.3f} uniq={ut}/{uv}")

mu = float(tr.y.mean())
for name in tr.X:
    c = FEATURE_CARDINALITIES[name]
    xt = tr.X[name]
    xv = va.X[name]
    cnt = np.bincount(xt, minlength=c)
    pos = np.bincount(xt, weights=tr.y, minlength=c)
    rate = (pos + 20.0 * mu) / (cnt + 20.0)
    pred = rate[xv]
    unseen = np.mean(cnt[xv] == 0)
    top = cnt.max() / len(xt)
    print(f"F {name} C{c} U{np.count_nonzero(cnt)}/{len(np.unique(xv))} "
          f"N{unseen:.3f} T{top:.3f} A{auc(va.y,pred):.3f}")

for name in tr.num:
    a = np.asarray(tr.num[name], dtype=np.float64)
    b = np.asarray(va.num[name], dtype=np.float64)
    finite_a = np.isfinite(a)
    finite_b = np.isfinite(b)
    med = np.nanmedian(a)
    z = np.log1p(np.maximum(np.where(finite_b, b, med), 0))
    r = np.corrcoef(z, va.y)[0, 1] if np.std(z) > 0 else 0.0
    vals = a[finite_a]
    q50, q95 = np.quantile(vals, [.5, .95]) if len(vals) else (np.nan, np.nan)
    print(f"N {name} M{1-finite_a.mean():.3f}/{1-finite_b.mean():.3f} "
          f"Q{q50:.1f}/{q95:.1f} R{r:.3f}")

for key in ["video_id", "author_id"]:
    h = historical_features("valid", key=key)
    shapes = sorted(set(np.asarray(v).shape for v in h.values()))
    print(f"H {key} keys={','.join(sorted(h))} shapes={shapes}")