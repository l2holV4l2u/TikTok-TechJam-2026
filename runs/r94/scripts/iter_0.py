import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")

print(f"SHAPE train={len(tr.user_id)} valid={len(va.user_id)} test={len(te.user_id)} "
      f"Xfields={len(tr.X)} num={len(tr.num)}")
print(f"ARRAY Xscalar={all(np.asarray(v).shape == (len(tr.user_id),) for v in tr.X.values())} "
      f"uid={tr.user_id.dtype} time={tr.time_ms.dtype} y={tr.y.dtype}")

def user_stats(s):
    u, inv, cnt = np.unique(s.user_id, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=s.y, minlength=len(u))
    return (len(u), np.median(cnt), np.quantile(cnt, .9), np.median(pos),
            np.mean(pos == 0), np.mean(pos == cnt), float(np.mean(s.y)))

for name, s in [("train", tr), ("valid", va)]:
    a = user_stats(s)
    print(f"USER {name} n={a[0]} rows50/90={a[1]:.0f}/{a[2]:.0f} pos50={a[3]:.0f} "
          f"zero={a[4]:.3f} all={a[5]:.3f} rate={a[6]:.4f}")
print(f"TEST n={len(te.user_id)} users={np.unique(te.user_id).size}")

def day_rates(s):
    d, inv = np.unique(s.date, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    return " ".join(f"{str(x)[4:]}:{100*y/z:.1f}" for x, y, z in zip(d, p, n))

print("DAY train% " + day_rates(tr))
print("DAY valid% " + day_rates(va))

def cold_rows(train_ids, other_ids):
    seen = np.unique(train_ids)
    return np.mean(~np.isin(other_ids, seen)), np.unique(other_ids).size

for key, atr, ava, ate in [
    ("user", tr.user_id, va.user_id, te.user_id),
    ("video", tr.video_id, va.video_id, te.video_id),
    ("author", tr.X["author_id"], va.X["author_id"], te.X["author_id"])
]:
    cv, uv = cold_rows(atr, ava)
    ct, ut = cold_rows(atr, ate)
    print(f"COLD {key} valid={cv:.3f}(U{uv}) test={ct:.3f}(U{ut})")

order = np.lexsort((np.arange(len(tr.user_id)), tr.time_ms, tr.user_id))
same_batch = ((tr.user_id[order][1:] == tr.user_id[order][:-1]) &
              (tr.time_ms[order][1:] == tr.time_ms[order][:-1]))
print(f"TIME adjacent_same_timestamp={same_batch.mean():.3f} "
      f"range={tr.time_ms.min()}..{tr.time_ms.max()}")
del order, same_batch

vu, vinv, vcnt = np.unique(va.user_id, return_inverse=True, return_counts=True)
vmean = np.bincount(vinv, weights=va.y) / vcnt
yc = va.y.astype(np.float64) - vmean[vinv]
yy = np.dot(yc, yc)

def within_corr(z):
    z = np.asarray(z, dtype=np.float64)
    zm = np.bincount(vinv, weights=z, minlength=len(vu)) / vcnt
    zc = z - zm[vinv]
    den = np.sqrt(np.dot(zc, zc) * yy)
    return 0.0 if den == 0 else float(np.dot(zc, yc) / den)

print("NUM name miss tr50/tr95 Wcorr(log1p, valid)")
for name in tr.num:
    x = tr.num[name].astype(np.float64)
    finite = np.isfinite(x)
    med = np.nanmedian(x)
    q95 = np.nanquantile(x, .95)
    xv = np.nan_to_num(va.num[name].astype(np.float64), nan=med)
    z = np.log1p(np.maximum(xv, 0))
    print(f"N {name} M={1-finite.mean():.3f} Q={med:.1f}/{q95:.1f} W={within_corr(z):+.3f}")

for key in ("video_id", "author_id"):
    h = historical_features("train", key=key)
    ks = ",".join(sorted(h.keys()))
    ok = all(np.asarray(v).shape == (len(tr.user_id),) for v in h.values())
    print(f"HIST {key} n={len(h)} rowshape={ok} keys={ks}"[:220])

print(f"AUX forbidden outcomes n={len(tr.aux)} keys={','.join(sorted(tr.aux)[:10])}")
print("FIELD C=card U=train/valid/test unique Z=valid cold-row D=train dominant W=within-user corr")
p0 = float(np.mean(tr.y))
for name in tr.X:
    C = FEATURE_CARDINALITIES[name]
    xt, xv, xe = tr.X[name], va.X[name], te.X[name]
    cnt = np.bincount(xt, minlength=C)
    pos = np.bincount(xt, weights=tr.y, minlength=C)
    seen = cnt > 0
    rate = (pos + 20.0 * p0) / (cnt + 20.0)
    score = rate[xv]
    u1 = np.count_nonzero(seen)
    u2 = np.unique(xv).size
    u3 = np.unique(xe).size
    cold = np.mean(~seen[xv])
    dom = cnt.max() / len(xt)
    print(f"F {name} C={C} U={u1}/{u2}/{u3} Z={cold:.2f} D={dom:.2f} W={within_corr(score):+.3f}")