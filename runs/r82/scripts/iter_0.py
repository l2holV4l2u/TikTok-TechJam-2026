import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features
from pipeline.evaluate import evaluate

tr, va, te = load("train"), load("valid"), load("test")
mu = float(tr.y.mean())

print(f"ROWS train={len(tr.y)} valid={len(va.y)} test={len(te.user_id)}")
print(f"SHAPES X={len(tr.X)} fields; sample={next(iter(tr.X.values())).shape}, y={tr.y.shape}, date={tr.date.shape}, time={tr.time_ms.shape}")
print(f"NUM keys={','.join(tr.num.keys())}; AUX n={len(tr.aux)} (outcomes, inspection only)")

def user_summary(s, name):
    u, inv, n = np.unique(s.user_id, return_inverse=True, return_counts=True)
    p = np.bincount(inv, weights=s.y, minlength=len(u))
    qn = np.quantile(n, [0, .25, .5, .75, .9, .99, 1])
    qp = np.quantile(p, [0, .25, .5, .75, .9, .99, 1])
    print(f"USER {name} n={len(u)} y={s.y.mean():.4f} zero={np.mean(p==0):.3f} all={np.mean(p==n):.3f}")
    print(f"UROWS {name} q0/25/50/75/90/99/max=" + "/".join(f"{x:.0f}" for x in qn))
    print(f"UPOS  {name} q0/25/50/75/90/99/max=" + "/".join(f"{x:.0f}" for x in qp))

user_summary(tr, "tr")
user_summary(va, "va")

tu = np.unique(tr.user_id)
tv = np.unique(tr.video_id)
print(f"UNSEEN valid rows user={np.mean(~np.isin(va.user_id,tu)):.3f} video={np.mean(~np.isin(va.video_id,tv)):.3f}")
print(f"UNSEEN valid ids user={np.mean(~np.isin(np.unique(va.user_id),tu)):.3f} video={np.mean(~np.isin(np.unique(va.video_id),tv)):.3f}")
print(f"UNSEEN test rows user={np.mean(~np.isin(te.user_id,tu)):.3f} video={np.mean(~np.isin(te.video_id,tv)):.3f}")

for s, name in [(tr, "tr"), (va, "va")]:
    vals = []
    for d in np.unique(s.date):
        m = s.date == d
        vals.append(f"{str(d)[4:]}:{s.y[m].mean():.3f}")
    print("DATE " + name + " " + " ".join(vals))

print("FIELD format name card:seen_tr/seen_va unseenV topShare P/G")
for name in tr.X:
    card = FEATURE_CARDINALITIES[name]
    x = tr.X[name]
    xv = va.X[name]
    cnt = np.bincount(x, minlength=card)
    pos = np.bincount(x, weights=tr.y, minlength=card)
    rate = (pos + 20.0 * mu) / (cnt + 20.0)
    pred = rate[xv]
    met = evaluate(va.user_id, va.y, pred)
    unseen = np.mean(cnt[xv] == 0)
    top = cnt.max() / len(x)
    print(f"F {name} {card}:{np.count_nonzero(cnt)}/{np.unique(xv).size} u={unseen:.3f} t={top:.3f} P={met['primary']:.3f} G={met['gauc']:.3f}")

for name in tr.num:
    a = np.asarray(tr.num[name], dtype=np.float64)
    b = np.asarray(va.num[name], dtype=np.float64)
    good = np.isfinite(a)
    qs = np.unique(np.quantile(a[good], np.linspace(0, 1, 33)))
    edges = qs[1:-1]
    bt = np.searchsorted(edges, np.nan_to_num(a, nan=-np.inf)) + 1
    bv = np.searchsorted(edges, np.nan_to_num(b, nan=-np.inf)) + 1
    bt[~good] = 0
    bv[~np.isfinite(b)] = 0
    cnt = np.bincount(bt, minlength=len(edges) + 2)
    pos = np.bincount(bt, weights=tr.y, minlength=len(edges) + 2)
    pred = ((pos + 50 * mu) / (cnt + 50))[bv]
    met = evaluate(va.user_id, va.y, pred)
    print(f"N {name} miss={np.mean(~good):.3f}/{np.mean(~np.isfinite(b)):.3f} bins={len(edges)+1} P={met['primary']:.3f} G={met['gauc']:.3f}")

for entity in ("video_id", "author_id"):
    h = historical_features("valid", key=entity)
    results = []
    for name, z in h.items():
        z = np.asarray(z, dtype=np.float64)
        fill = np.nanmedian(z) if np.any(np.isfinite(z)) else 0.0
        z = np.nan_to_num(z, nan=fill, posinf=fill, neginf=fill)
        m1 = evaluate(va.user_id, va.y, z)
        m2 = evaluate(va.user_id, va.y, -z)
        best = m1 if m1["primary"] >= m2["primary"] else m2
        results.append((best["primary"], best["gauc"], name, z.shape))
    results.sort(reverse=True)
    desc = " ".join(f"{n}:{p:.3f}/{g:.3f}" for p, g, n, _ in results[:6])
    shape = results[0][3] if results else ()
    print(f"HIST {entity} n={len(h)} shape={shape} topP/G {desc}")