import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES

tr = load("train")
va = load("valid")
te = load("test")  # Feature inspection only; never access te.y.

def qstr(x):
    q = np.quantile(x, [0, .25, .5, .75, .9, .99, 1])
    return "/".join(f"{z:.0f}" for z in q)

def user_summary(s, name):
    u, inv, rows = np.unique(s.user_id, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=s.y, minlength=len(u))
    zero = np.mean(pos == 0) * 100
    mixed = np.mean((pos > 0) & (pos < rows)) * 100
    print(f"USER {name} n={len(u)} rowsQ={qstr(rows)} posQ={qstr(pos)} "
          f"zero={zero:.1f}% mixed={mixed:.1f}%")
    return u, inv

def row_overlap(x, train_unique):
    p = np.searchsorted(train_unique, x)
    ok = p < len(train_unique)
    hit = np.zeros(len(x), dtype=bool)
    hit[ok] = train_unique[p[ok]] == x[ok]
    return 100 * hit.mean()

print(f"ROWS train={len(tr.y)} valid={len(va.y)} test={len(te.user_id)}")
print(f"LABEL train={tr.y.mean():.5f} valid={va.y.mean():.5f}")
print(f"DATES train={tr.date.min()}-{tr.date.max()} "
      f"valid={va.date.min()}-{va.date.max()} test={te.date.min()}-{te.date.max()}")

day_stats = []
for d in np.unique(tr.date):
    m = tr.date == d
    day_stats.append(f"{d % 100:02d}:{tr.y[m].mean():.3f}")
print("TRAIN_DAY_RATE " + " ".join(day_stats))

tu, ti = user_summary(tr, "train")
vu, vi = user_summary(va, "valid")
print(f"ENTITY user validSeenRows={row_overlap(va.user_id, tu):.1f}% "
      f"testSeenRows={row_overlap(te.user_id, tu):.1f}%")
tv = np.unique(tr.video_id)
print(f"ENTITY video trainUnique={len(tv)} validUnique={len(np.unique(va.video_id))} "
      f"testUnique={len(np.unique(te.video_id))} "
      f"validSeenRows={row_overlap(va.video_id, tv):.1f}% "
      f"testSeenRows={row_overlap(te.video_id, tv):.1f}%")

names = sorted(tr.X)
shape_ok = all(np.asarray(tr.X[n]).shape == (len(tr.y),) for n in names)
print(f"X fields={len(names)} scalarPerRow={shape_ok} "
      f"ndims={sorted(set(np.asarray(tr.X[n]).ndim for n in names))} "
      f"dtypes={sorted(set(str(np.asarray(tr.X[n]).dtype) for n in names))}")
aux_desc = ",".join(f"{k}:{np.asarray(v).shape}" for k, v in sorted(tr.aux.items()))
print(f"AUX n={len(tr.aux)} shapes={aux_desc[:300]}")

# Prepare validation user order to measure whether a field can vary within a ranking group.
vo = np.argsort(va.user_id, kind="stable")
vios = vi[vo]
same_user = vios[1:] == vios[:-1]

p0 = np.clip(tr.y.mean(), 1e-7, 1 - 1e-7)
base_ll = -np.mean(va.y * np.log(p0) + (1 - va.y) * np.log1p(-p0))
alpha = 20.0

print("FIELD name C Utr/Uva/Ute unseenV/T% max% varUserV% dLLx1e3")
records = []
for name in names:
    xt = np.asarray(tr.X[name])
    xv = np.asarray(va.X[name])
    xe = np.asarray(te.X[name])
    card = int(FEATURE_CARDINALITIES[name])

    cnt = np.bincount(xt, minlength=card)
    sy = np.bincount(xt, weights=tr.y, minlength=card)
    pred = (sy[xv] + alpha * p0) / (cnt[xv] + alpha)
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    ll = -np.mean(va.y * np.log(pred) + (1 - va.y) * np.log1p(-pred))
    dll = 1000 * (base_ll - ll)

    unseen_v = 100 * np.mean(cnt[xv] == 0)
    unseen_t = 100 * np.mean(cnt[xe] == 0)
    max_share = 100 * cnt.max() / len(xt)

    xs = xv[vo]
    changed = same_user & (xs[1:] != xs[:-1])
    variable = np.zeros(len(vu), dtype=bool)
    variable[vios[1:][changed]] = True
    var_pct = 100 * variable.mean()

    rec = (dll, name, card, np.count_nonzero(cnt),
           len(np.unique(xv)), len(np.unique(xe)),
           unseen_v, unseen_t, max_share, var_pct)
    records.append(rec)

for dll, name, card, ut, uv, ue, zv, zt, mx, vr in sorted(records, reverse=True):
    print(f"F {name} {card} {ut}/{uv}/{ue} {zv:.1f}/{zt:.1f} "
          f"{mx:.1f} {vr:.1f} {dll:+.2f}")