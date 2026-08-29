import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")

print(f"rows tr/va/te={len(tr.user_id)}/{len(va.user_id)}/{len(te.user_id)}")
print(f"label train mean={tr.y.mean():.5f} pos={int(tr.y.sum())}; valid mean={va.y.mean():.5f} pos={int(va.y.sum())}")
print(f"X fields={len(tr.X)} num_fields={len(tr.num)}")
scalar_ok = all(np.asarray(x).shape == (len(tr.user_id),) for x in tr.X.values())
print(f"X scalar-per-row={scalar_ok}; dtypes={sorted(set(str(x.dtype) for x in tr.X.values()))}")

def user_summary(name, s):
    u, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    qn = np.quantile(n, [0, .25, .5, .75, .9, .99, 1])
    qp = np.quantile(p, [0, .25, .5, .75, .9, .99, 1])
    print(f"{name} users={len(u)} imp_q={np.round(qn,1).tolist()}")
    print(f"{name} pos_q={np.round(qp,1).tolist()} zero={np.mean(p==0):.3f} all={np.mean(p==n):.3f}")

user_summary("train", tr)
user_summary("valid", va)

dates = np.unique(tr.date)
date_text = ",".join(f"{d}:{tr.y[tr.date==d].mean():.3f}" for d in dates)
print("train date rates=" + date_text)
dates = np.unique(va.date)
date_text = ",".join(f"{d}:{va.y[va.date==d].mean():.3f}" for d in dates)
print("valid date rates=" + date_text)

def mutual_information(x, y, card):
    n = len(y)
    cnt = np.bincount(x, minlength=card).astype(np.float64)
    pos = np.bincount(x, weights=y, minlength=card).astype(np.float64)
    neg = cnt - pos
    py1 = y.mean()
    out = 0.0
    m = pos > 0
    out += np.sum((pos[m] / n) * np.log2((pos[m] / cnt[m]) / py1))
    m = neg > 0
    out += np.sum((neg[m] / n) * np.log2((neg[m] / cnt[m]) / (1.0-py1)))
    return out

def temporal_consistency(x, y, date, card):
    cut = np.sort(np.unique(date))[len(np.unique(date)) // 2]
    a = date < cut
    ca = np.bincount(x[a], minlength=card).astype(float)
    cb = np.bincount(x[~a], minlength=card).astype(float)
    pa = np.bincount(x[a], weights=y[a], minlength=card)
    pb = np.bincount(x[~a], weights=y[~a], minlength=card)
    m = (ca >= 20) & (cb >= 20)
    if m.sum() < 3:
        return np.nan, int(m.sum())
    ra, rb = pa[m]/ca[m], pb[m]/cb[m]
    w = np.sqrt(ca[m]*cb[m])
    ma, mb = np.average(ra, weights=w), np.average(rb, weights=w)
    cov = np.average((ra-ma)*(rb-mb), weights=w)
    den = np.sqrt(np.average((ra-ma)**2, weights=w) *
                  np.average((rb-mb)**2, weights=w))
    return (cov/den if den > 0 else np.nan), int(m.sum())

print("field: card uniq(tr/v/t) unseen-row%(v/t) zero-v% MIbits half-rate-r(n)")
for name in tr.X:
    xt, xv, xe = tr.X[name], va.X[name], te.X[name]
    card = FEATURE_CARDINALITIES[name]
    seen = np.zeros(card, dtype=bool)
    seen[np.unique(xt)] = True
    seen[0] = False
    uv = 100*np.mean(~seen[xv])
    ue = 100*np.mean(~seen[xe])
    mi = mutual_information(xt, tr.y, card)
    tc, ns = temporal_consistency(xt, tr.y, tr.date, card)
    print(f"{name}: {card} {len(np.unique(xt))}/{len(np.unique(xv))}/{len(np.unique(xe))} "
          f"{uv:.1f}/{ue:.1f} {100*np.mean(xv==0):.1f} {mi:.4f} {tc:.2f}({ns})")

for name in tr.num:
    a, b = np.asarray(tr.num[name]), np.asarray(va.num[name])
    finite = a[np.isfinite(a)]
    q = np.quantile(finite, [.01, .5, .9, .99]) if len(finite) else [np.nan]*4
    print(f"num {name}: nan tr/v={np.mean(~np.isfinite(a)):.3f}/{np.mean(~np.isfinite(b)):.3f} "
          f"q01/50/90/99={np.round(q,2).tolist()}")

for key in ("video_id", "author_id"):
    try:
        h = historical_features("valid", key=key)
        names = list(h.keys())
        desc = []
        for k in names:
            z = np.asarray(h[k])
            desc.append(f"{k}:{z.shape}/{z.dtype}/nan{np.mean(~np.isfinite(z)):.2f}")
        text = ";".join(desc)
        print(f"history {key} n={len(names)} " + text[:650])
    except Exception as e:
        print(f"history {key} ERROR {type(e).__name__}:{str(e)[:120]}")

print("aux keys(leakage-only)=" + ",".join(sorted(tr.aux.keys()))[:350])