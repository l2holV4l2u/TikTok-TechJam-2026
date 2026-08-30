import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")

def pct(x):
    return 100.0 * float(x)

def qstr(x, qs=(0, .25, .5, .75, .9, .99, 1)):
    return ",".join(f"{v:.1f}" for v in np.quantile(x, qs))

def user_stats(s, name):
    u, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    eligible = (p > 0) & (p < n)
    print(f"USER {name} U={len(u)} rowsQ={qstr(n)} posQ={qstr(p)} "
          f"zero={pct(np.mean(p == 0)):.1f}% eligible={pct(eligible.mean()):.1f}%")

def day_stats(s, name):
    vals = []
    for d in np.unique(s.date):
        m = s.date == d
        vals.append(f"{int(d)%10000:04d}:{m.sum()//1000}k/{s.y[m].mean():.3f}")
    print(f"DAY {name} " + " ".join(vals))

def get_field(s, key):
    if key in s.X:
        return s.X[key]
    return getattr(s, key)

def overlap(key, a, b, name):
    xa = get_field(a, key)
    xb = get_field(b, key)
    seen = np.unique(xa)
    novel_rows = (xb != 0) & ~np.isin(xb, seen)
    ub = np.unique(xb[xb != 0])
    novel_ids = ~np.isin(ub, seen)
    print(f"NEW {name}.{key} rows={pct(novel_rows.mean()):.1f}% "
          f"ids={pct(novel_ids.mean()) if len(ub) else 0:.1f}% "
          f"zero={pct(np.mean(xb == 0)):.1f}%")

def normalized_mi(x, y, k):
    n = len(y)
    cnt = np.bincount(x, minlength=k).astype(np.float64)
    pos = np.bincount(x, weights=y, minlength=k).astype(np.float64)
    neg = cnt - pos
    py1 = float(np.sum(y))
    py0 = n - py1
    mi = 0.0
    for cell, py in ((pos, py1), (neg, py0)):
        if py <= 0:
            continue
        m = cell > 0
        mi += np.sum((cell[m] / n) *
                     np.log((cell[m] * n) / (cnt[m] * py)))
    p = py1 / n
    h = -(p*np.log(p) + (1-p)*np.log1p(-p)) if 0 < p < 1 else 1.0
    return 100.0 * mi / h

def numeric_summary(name):
    a, b, c = tr.num[name], va.num[name], te.num[name]
    fa, fb, fc = np.isfinite(a), np.isfinite(b), np.isfinite(c)
    vals = a[fa]
    qs = np.quantile(vals, [.01, .5, .99]) if len(vals) else [np.nan]*3

    def logcorr(x, mask, y):
        if mask.sum() < 2:
            return np.nan
        z = np.log1p(np.maximum(x[mask].astype(np.float64), 0))
        return np.corrcoef(z, y[mask])[0, 1] if np.std(z) > 0 else 0.0

    ct = logcorr(a, fa, tr.y)
    cv = logcorr(b, fb, va.y)
    print(f"NUM {name} miss={pct((~fa).mean()):.1f}/{pct((~fb).mean()):.1f}/"
          f"{pct((~fc).mean()):.1f}% q01/50/99={qs[0]:.2g}/{qs[1]:.2g}/"
          f"{qs[2]:.2g} logcorr={ct:.3f}/{cv:.3f}")

print(f"ROWS T/V/E={len(tr.user_id)}/{len(va.user_id)}/{len(te.user_id)}")
print(f"LABEL T={int(tr.y.sum())}/{tr.y.mean():.4f} "
      f"V={int(va.y.sum())}/{va.y.mean():.4f}")
shape_types = sorted({(tuple(v.shape), str(v.dtype)) for v in tr.X.values()})
print(f"SCHEMA cat={len(tr.X)} num={list(tr.num)} Xshape_dtype={shape_types}")

user_stats(tr, "train")
user_stats(va, "valid")
day_stats(tr, "train")
day_stats(va, "valid")

for key in ("user_id", "video_id", "author_id"):
    overlap(key, tr, va, "valid")
    overlap(key, tr, te, "test")

cold_u = ~np.isin(va.user_id, np.unique(tr.user_id))
cold_v = ~np.isin(va.video_id, np.unique(tr.video_id))
print(f"COLD valid user/video/either/both={pct(cold_u.mean()):.1f}/"
      f"{pct(cold_v.mean()):.1f}/{pct((cold_u|cold_v).mean()):.1f}/"
      f"{pct((cold_u&cold_v).mean()):.1f}%")

print("FIELD name K seenT/V/E zeroV/T novelV/T topT MI_T/V")
for name in sorted(tr.X):
    k = FEATURE_CARDINALITIES[name]
    xt, xv, xe = tr.X[name], va.X[name], te.X[name]
    ct = np.bincount(xt, minlength=k)
    cv = np.bincount(xv, minlength=k)
    ce = np.bincount(xe, minlength=k)
    seen = ct > 0
    nv = (xv != 0) & ~seen[xv]
    ne = (xe != 0) & ~seen[xe]
    print(f"F {name} {k} {np.count_nonzero(ct)}/{np.count_nonzero(cv)}/"
          f"{np.count_nonzero(ce)} {pct(np.mean(xv==0)):.1f}/"
          f"{pct(np.mean(xe==0)):.1f} {pct(nv.mean()):.1f}/{pct(ne.mean()):.1f} "
          f"{pct(ct.max()/len(xt)):.1f} "
          f"{normalized_mi(xt,tr.y,k):.2f}/{normalized_mi(xv,va.y,k):.2f}")

for name in tr.num:
    numeric_summary(name)

try:
    hv = historical_features("valid", key="video_id")
    keys = ",".join(hv.keys())
    layouts = sorted({(tuple(np.asarray(v).shape), str(np.asarray(v).dtype))
                      for v in hv.values()})
    print(f"HIST valid.video fields={len(hv)} keys={keys}")
    print(f"HIST layouts={layouts}")
except Exception as e:
    print(f"HIST error={type(e).__name__}:{str(e)[:120]}")