import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")

def qstr(a):
    return "/".join(f"{x:.0f}" for x in np.quantile(a, [0.1, 0.5, 0.9, 0.99]))

def user_summary(s, name):
    _, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    print(f"USR|{name}|U={len(n)}|rowsq={qstr(n)}|posq={qstr(p)}"
          f"|zero={100*np.mean(p==0):.1f}%|mixed={100*np.mean((p>0)&(p<n)):.1f}%")

def cramer_binary(x, y, k):
    n = len(y)
    py = float(np.mean(y))
    if n < 2 or py <= 0 or py >= 1:
        return 0.0
    c = np.bincount(x, minlength=k).astype(np.float64)
    p = np.bincount(x, weights=y, minlength=k)
    use = c > 0
    d = p[use] - c[use] * py
    chi2 = np.sum(d * d / (c[use] * py * (1.0 - py)))
    cats = int(use.sum())
    phi2 = chi2 / n
    corrected = max(0.0, phi2 - (cats - 1.0) / (n - 1.0))
    return float(np.sqrt(corrected))

def daily(s, name):
    ds, inv = np.unique(s.date, return_inverse=True)
    cnt = np.bincount(inv)
    pos = np.bincount(inv, weights=s.y)
    vals = ",".join(f"{int(d)%10000:04d}:{p/c:.3f}"
                    for d, p, c in zip(ds, pos, cnt))
    print(f"DAY|{name}|{vals}")

def overlap(a, b, entity):
    av = np.asarray(getattr(a, entity))
    bv = np.asarray(getattr(b, entity))
    seen = np.unique(av)
    known = np.isin(bv, seen)
    ub = np.unique(bv)
    useen = np.isin(ub, seen)
    return 100*(1-known.mean()), 100*useen.mean()

print(f"SHAPE|train={len(tr.user_id)}|valid={len(va.user_id)}|test={len(te.user_id)}"
      f"|X={len(tr.X)} scalar_fields|num={len(tr.num)}")
print(f"LABEL|train={tr.y.mean():.5f} ({tr.y.sum()})"
      f"|valid={va.y.mean():.5f} ({va.y.sum()})")
user_summary(tr, "train")
user_summary(va, "valid")
daily(tr, "train")
daily(va, "valid")

for ent in ("user_id", "video_id"):
    uv, sv = overlap(tr, va, ent)
    ut, st = overlap(tr, te, ent)
    print(f"OVL|{ent}|unseen_rows valid/test={uv:.2f}/{ut:.2f}%"
          f"|seen_unique valid/test={sv:.2f}/{st:.2f}%")
for ent in ("user_id", "video_id"):
    a = np.asarray(getattr(tr, ent))
    b = np.asarray(getattr(va, ent))
    seen = np.isin(b, np.unique(a))
    r1 = va.y[seen].mean() if seen.any() else np.nan
    r0 = va.y[~seen].mean() if (~seen).any() else np.nan
    print(f"SHIFT|valid {ent}|label seen/unseen={r1:.4f}/{r0:.4f}"
          f"|unseen_n={(~seen).sum()}")

print("F legend: K=declared,O=train/valid/test observed,U=valid/test unseen-row%,"
      "M=train mode%,V=train/valid bias-corrected Cramer")
for name in tr.X:
    xt = np.asarray(tr.X[name])
    xv = np.asarray(va.X[name])
    xe = np.asarray(te.X[name])
    k = int(FEATURE_CARDINALITIES[name])
    seen = np.zeros(k, dtype=bool)
    seen[np.unique(xt)] = True
    uv = np.mean((xv == 0) | ~seen[xv])
    ue = np.mean((xe == 0) | ~seen[xe])
    ot = np.unique(xt).size
    ov = np.unique(xv).size
    oe = np.unique(xe).size
    mode = np.bincount(xt, minlength=k).max() / len(xt)
    vt = cramer_binary(xt, tr.y, k)
    vv = cramer_binary(xv, va.y, k)
    print(f"F|{name}|K={k}|O={ot}/{ov}/{oe}|U={100*uv:.1f}/{100*ue:.1f}"
          f"|M={100*mode:.1f}|V={vt:.3f}/{vv:.3f}")

for name in tr.num:
    a = np.asarray(tr.num[name], dtype=np.float64)
    b = np.asarray(va.num[name], dtype=np.float64)
    fa, fb = np.isfinite(a), np.isfinite(b)
    qa = np.quantile(a[fa], [0.5, 0.95]) if fa.any() else [np.nan, np.nan]
    qb = np.quantile(b[fb], [0.5, 0.95]) if fb.any() else [np.nan, np.nan]
    za = np.log1p(np.maximum(a[fa], 0))
    zb = np.log1p(np.maximum(b[fb], 0))
    ca = np.corrcoef(za, tr.y[fa])[0, 1] if za.std() else 0.0
    cb = np.corrcoef(zb, va.y[fb])[0, 1] if zb.std() else 0.0
    print(f"N|{name}|miss={100*(~fa).mean():.1f}/{100*(~fb).mean():.1f}"
          f"|med95={qa[0]:.0f},{qa[1]:.0f}/{qb[0]:.0f},{qb[1]:.0f}"
          f"|logcorr={ca:.3f}/{cb:.3f}")

for key in ("video_id", "author_id"):
    h = historical_features("train", key=key)
    desc = ",".join(f"{n}:{np.asarray(v).dtype}:{np.asarray(v).shape}"
                    for n, v in h.items())
    print(f"HIST|{key}|{desc}")