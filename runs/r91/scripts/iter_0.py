import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")

def date_summary(s):
    parts = []
    for d in np.unique(s.date):
        m = s.date == d
        parts.append(f"{str(int(d))[-4:]}:{m.sum()},{s.y[m].mean():.3f}")
    return " ".join(parts)

def user_summary(s, name):
    c = np.bincount(s.user_id)
    p = np.bincount(s.user_id, weights=s.y)
    active = c > 0
    c, p = c[active], p[active]
    q = np.percentile(c, [10, 50, 90, 99])
    pq = np.percentile(p, [10, 50, 90, 99])
    print(f"USER {name} n={len(c)} rowsQ={q.astype(int).tolist()} "
          f"posQ={pq.astype(int).tolist()} zero={np.mean(p==0):.3f} "
          f"all={np.mean(p==c):.3f}")

def adjusted_assoc(x, y, card):
    n = len(y)
    cnt = np.bincount(x, minlength=card).astype(np.float64)
    pos = np.bincount(x, weights=y, minlength=card)
    used = cnt > 0
    cnt, pos = cnt[used], pos[used]
    p = float(np.mean(y))
    if p <= 0 or p >= 1:
        return 0.0
    chi2 = np.sum((pos - cnt * p) ** 2 / (cnt * p * (1.0 - p)))
    k = len(cnt)
    phi2_adj = max(0.0, chi2 / n - (k - 1.0) / max(n - 1.0, 1.0))
    return float(np.sqrt(phi2_adj))

print(f"BASIC rows={len(tr.y)}/{len(va.y)}/{len(te.user_id)} "
      f"pos={tr.y.mean():.4f}/{va.y.mean():.4f} "
      f"fields={len(tr.X)} nums={len(tr.num)} "
      f"scalar={all(np.asarray(v).shape==(len(tr.y),) for v in tr.X.values())}")
print("DATE tr " + date_summary(tr))
print("DATE va " + date_summary(va))
user_summary(tr, "tr")
user_summary(va, "va")

order = np.lexsort((np.arange(len(tr.y)), tr.time_ms, tr.user_id))
u, t = tr.user_id[order], tr.time_ms[order]
tie_pairs = np.mean((u[1:] == u[:-1]) & (t[1:] == t[:-1]))
print(f"TIME range={tr.time_ms.min()}..{tr.time_ms.max()} adjacent_same_ts={tie_pairs:.4f}")

for name in tr.X:
    card = FEATURE_CARDINALITIES[name]
    xt, xv, xe = tr.X[name], va.X[name], te.X[name]
    counts = np.bincount(xt, minlength=card)
    seen = counts > 0
    ut = int(seen.sum())
    uv = int(np.count_nonzero(np.bincount(xv, minlength=card)))
    ue = int(np.count_nonzero(np.bincount(xe, minlength=card)))
    unseen_v = float(np.mean(~seen[xv]))
    unseen_e = float(np.mean(~seen[xe]))
    dominant = float(counts.max() / len(xt))
    at = adjusted_assoc(xt, tr.y, card)
    av = adjusted_assoc(xv, va.y, card)
    print(f"F {name} c={card} u={ut}/{uv}/{ue} "
          f"new={unseen_v:.3f}/{unseen_e:.3f} dom={dominant:.3f} "
          f"a={at:.3f}/{av:.3f}")

for name in tr.num:
    a = np.asarray(tr.num[name], dtype=np.float64)
    b = np.asarray(va.num[name], dtype=np.float64)
    ma, mb = np.isfinite(a), np.isfinite(b)
    za = np.log1p(np.maximum(a[ma], 0))
    zb = np.log1p(np.maximum(b[mb], 0))
    ca = np.corrcoef(za, tr.y[ma])[0, 1] if ma.sum() > 2 and np.std(za) else 0.0
    cb = np.corrcoef(zb, va.y[mb])[0, 1] if mb.sum() > 2 and np.std(zb) else 0.0
    med0 = np.nanmedian(a[tr.y == 0])
    med1 = np.nanmedian(a[tr.y == 1])
    print(f"N {name} miss={1-ma.mean():.3f}/{1-mb.mean():.3f} "
          f"med0/1={med0:.1f}/{med1:.1f} corr={ca:.3f}/{cb:.3f}")

for key in ("video_id", "author_id"):
    try:
        h = historical_features("train", key=key)
        desc = ",".join(f"{k}:{np.asarray(v).dtype}" for k, v in h.items())
        print(f"HIST {key} n={len(h)} keys={desc}"[:380])
    except Exception as e:
        print(f"HIST {key} ERROR={type(e).__name__}:{str(e)[:100]}")