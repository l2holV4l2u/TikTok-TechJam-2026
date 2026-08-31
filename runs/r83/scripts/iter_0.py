import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")

def compact(n):
    n = int(n)
    if n >= 1_000_000:
        return f"{n/1e6:.2f}m"
    if n >= 1_000:
        return f"{n/1e3:.1f}k"
    return str(n)

def user_stats(s):
    u, inv, rows = np.unique(s.user_id, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=s.y, minlength=len(u))
    return (len(u), np.median(rows), np.percentile(rows, 90),
            np.median(pos), np.percentile(pos, 90),
            100*np.mean(pos == 0), 100*np.mean(pos == rows),
            100*np.mean((pos > 0) & (pos < rows)))

def date_line(s):
    d, inv, cnt = np.unique(s.date, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=s.y)
    return " ".join(f"{x}:{p/c:.3f}" for x, p, c in zip(d, pos, cnt))

def shrunk_eta(x, y, alpha=20.0):
    p = float(np.mean(y))
    n = np.bincount(x)
    sy = np.bincount(x, weights=y, minlength=len(n))
    rate = (sy + alpha*p) / (n + alpha)
    var = np.sum(n * (rate - p)**2) / len(y)
    return 100.0 * var / max(p*(1-p), 1e-12)

def entity_overlap(name):
    a = tr.X[name]
    b = va.X[name]
    seen = np.zeros(max(int(a.max()), int(b.max())) + 1, dtype=bool)
    seen[np.unique(a)] = True
    ub = np.unique(b)
    return 100*np.mean(~seen[b]), 100*np.mean(~seen[ub])

print(f"rows train={len(tr.user_id)} valid={len(va.user_id)}")
print(f"labels train p={tr.y.mean():.4f} pos={tr.y.sum()} valid p={va.y.mean():.4f} pos={va.y.sum()}")
print("dates train", date_line(tr))
print("dates valid", date_line(va))

for name, s in [("train", tr), ("valid", va)]:
    z = user_stats(s)
    print(f"users {name}: n={z[0]} rows med/p90={z[1]:.0f}/{z[2]:.0f} "
          f"pos med/p90={z[3]:.0f}/{z[4]:.0f} no/all/mixed%={z[5]:.1f}/{z[6]:.1f}/{z[7]:.1f}")

for f in ["user_id", "video_id", "author_id"]:
    rr, uu = entity_overlap(f)
    print(f"unseen-valid {f}: rows={rr:.2f}% unique_ids={uu:.2f}%")

shapes_ok = all(np.asarray(x).shape == (len(tr.user_id),) and np.asarray(x).ndim == 1
                for x in tr.X.values())
print(f"X fields={len(tr.X)} all_scalar_1d={shapes_ok} sample_dtype={tr.X['user_id'].dtype}")
print(f"numeric fields={list(tr.num)} aux_fields={len(tr.aux)} aux_names={sorted(tr.aux)[:12]}")

# Sort once to estimate whether each field can vary within a user's impression list.
order = np.argsort(va.user_id, kind="stable")
us = va.user_id[order]
starts = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1]
counts = np.diff(np.r_[starts, len(us)])

print("field: cardinality observed_train/valid unseenRow% dominant% withinUserVar% etaTrain/Valid%")
for f in sorted(tr.X):
    xt = tr.X[f]
    xv = va.X[f]
    card = FEATURE_CARDINALITIES[f]
    nt = np.bincount(xt)
    ut = np.count_nonzero(nt)
    uv = len(np.unique(xv))
    unseen = np.ones(max(len(nt), int(xv.max()) + 1), dtype=bool)
    unseen[:len(nt)] = (nt == 0)
    unrow = 100*np.mean(unseen[xv])
    dom = 100*nt.max()/len(xt)

    xs = xv[order]
    first = np.repeat(xs[starts], counts)
    within = 100*np.mean(xs != first)

    et = shrunk_eta(xt, tr.y)
    ev = shrunk_eta(xv, va.y)
    print(f"{f}: C{compact(card)} {compact(ut)}/{compact(uv)} "
          f"un{unrow:.1f} dom{dom:.1f} var{within:.1f} eta{et:.2f}/{ev:.2f}")

for f in tr.num:
    a = np.asarray(tr.num[f], dtype=np.float64)
    miss = ~np.isfinite(a)
    good = a[~miss]
    q = np.percentile(good, [10, 50, 90]) if len(good) else [np.nan]*3
    m0 = np.nanmean(a[tr.y == 0])
    m1 = np.nanmean(a[tr.y == 1])
    print(f"num {f}: miss={100*miss.mean():.1f}% q10/50/90={q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f} y0/y1={m0:.1f}/{m1:.1f}")

h = historical_features("valid", key="video_id")
desc = ",".join(f"{k}:{np.asarray(v).shape}/{np.asarray(v).dtype}" for k, v in h.items())
print("history valid video_id", desc)