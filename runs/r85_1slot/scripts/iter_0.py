import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")

def pct(x):
    return 100.0 * float(x)

def qstr(x, qs=(0, .25, .5, .75, .9, .99, 1)):
    z = np.quantile(x, qs)
    return ",".join(f"{v:.1f}" for v in z)

def user_stats(s, name):
    u, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    eligible = ((p > 0) & (p < n)).mean()
    print(f"USER {name} U={len(u)} rowsQ={qstr(n)} posQ={qstr(p)} "
          f"zero={pct((p==0).mean()):.1f}% elig={pct(eligible):.1f}%")

def day_stats(s, name):
    ds = np.unique(s.date)
    vals = []
    for d in ds:
        m = s.date == d
        vals.append(f"{int(d)%10000:04d}:{m.sum()//1000}k/{s.y[m].mean():.3f}")
    print(f"DAY {name} " + " ".join(vals))

def overlap(key, a, b, name):
    xa, xb = getattr(a, key), getattr(b, key)
    seen = np.unique(xa)
    newrow = ~np.isin(xb, seen)
    ub = np.unique(xb)
    newuniq = ~np.isin(ub, seen)
    print(f"NEW {name}.{key} row={pct(newrow.mean()):.1f}% "
          f"uniq={pct(newuniq.mean()):.1f}% ({newuniq.sum()}/{len(ub)})")

def normalized_mi(x, y, k):
    n = len(y)
    cnt = np.bincount(x, minlength=k).astype(np.float64)
    pos = np.bincount(x, weights=y, minlength=k).astype(np.float64)
    neg = cnt - pos
    py1 = float(y.sum())
    py0 = n - py1
    mi = 0.0
    for cell, py in ((pos, py1), (neg, py0)):
        m = cell > 0
        mi += np.sum((cell[m] / n) *
                     np.log((cell[m] * n) / (cnt[m] * py)))
    p = py1 / n
    h = -(p * np.log(p) + (1-p) * np.log(1-p)) if 0 < p < 1 else 1.0
    return 100.0 * mi / h

print(f"ROWS train={len(tr.user_id)} valid={len(va.user_id)} test={len(te.user_id)}")
print(f"LABEL train pos={tr.y.sum()} rate={tr.y.mean():.4f} "
      f"valid pos={va.y.sum()} rate={va.y.mean():.4f}")
print(f"X fields={len(tr.X)} NUM={list(tr.num)} "
      f"scalar_shapes={sorted(set((v.shape, str(v.dtype)) for v in tr.X.values()))}")
user_stats(tr, "train")
user_stats(va, "valid")
day_stats(tr, "train")
day_stats(va, "valid")

for key in ("user_id", "video_id"):
    overlap(key, tr, va, "valid")
    overlap(key, tr, te, "test")
overlap("author_id", tr, va, "valid")
overlap("author_id", tr, te, "test")

print("FIELD format: name K/Otrain/Ovalid/Otest unseenV%/unseenT% topTrain% MItrain%/MIvalid%")
for name in sorted(tr.X):
    k = FEATURE_CARDINALITIES[name]
    xt, xv, xe = tr.X[name], va.X[name], te.X[name]
    ct = np.bincount(xt, minlength=k)
    cv = np.bincount(xv, minlength=k)
    ce = np.bincount(xe, minlength=k)
    seen = ct > 0
    uv = (~seen[xv]).mean()
    ue = (~seen[xe]).mean()
    mit = normalized_mi(xt, tr.y, k)
    miv = normalized_mi(xv, va.y, k)
    print(f"F {name} {k}/{np.count_nonzero(ct)}/{np.count_nonzero(cv)}/"
          f"{np.count_nonzero(ce)} {pct(uv):.1f}/{pct(ue):.1f} "
          f"{pct(ct.max()/len(xt)):.1f} {mit:.2f}/{miv:.2f}")

for name in tr.num:
    a, b, c = tr.num[name], va.num[name], te.num[name]
    finite = a[np.isfinite(a)]
    qs = np.quantile(finite, [.01, .5, .99]) if len(finite) else [np.nan]*3
    print(f"NUM {name} missT/V/E={pct(~np.isfinite(a).mean()):.1f}/"
          f"{pct(~np.isfinite(b).mean()):.1f}/{pct(~np.isfinite(c).mean()):.1f}% "
          f"q01/50/99={qs[0]:.2g}/{qs[1]:.2g}/{qs[2]:.2g}")

try:
    hv = historical_features("valid", key="video_id")
    desc = ",".join(f"{k}:{np.asarray(v).shape}/{np.asarray(v).dtype}"
                    for k, v in hv.items())
    print("HIST valid.video " + desc)
except Exception as e:
    print("HIST error " + type(e).__name__ + ":" + str(e)[:100])