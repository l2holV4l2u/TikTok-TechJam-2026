import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
yt = tr.y.astype(np.float64)
yv = va.y.astype(np.float64)

print(f"ROWS train={len(yt)} valid={len(yv)} Xfields={len(tr.X)} num={len(tr.num)}")
shape_set = sorted({(a.ndim, str(a.dtype), len(a)) for a in tr.X.values()})
print(f"X_LAYOUT={shape_set} scalar_per_row={all(a.ndim == 1 for a in tr.X.values())}")
print(f"LABEL train={yt.mean():.4f} ({int(yt.sum())}) valid={yv.mean():.4f} ({int(yv.sum())})")

def user_summary(s, y, name):
    u, inv = np.unique(s.user_id, return_inverse=True)
    nr = np.bincount(inv)
    np_ = np.bincount(inv, weights=y)
    rq = np.percentile(nr, [10, 25, 50, 75, 90, 99])
    pq = np.percentile(np_, [10, 25, 50, 75, 90, 99])
    print(f"USER {name} n={len(u)} rows_q={rq.astype(int).tolist()}")
    print(f"USER {name} pos_q={np.round(pq,1).tolist()} zero={np.mean(np_==0):.3f} all={np.mean(np_==nr):.3f}")

user_summary(tr, yt, "tr")
user_summary(va, yv, "va")

def overlap(name, a, b):
    ua = np.unique(a)
    ub = np.unique(b)
    seen = np.isin(b, ua)
    seen_u = np.isin(ub, ua)
    print(f"OVERLAP {name} trU={len(ua)} vaU={len(ub)} unseen_rows={1-seen.mean():.3f} unseen_U={1-seen_u.mean():.3f}")

overlap("user", tr.user_id, va.user_id)
overlap("video", tr.video_id, va.video_id)
overlap("author", tr.X["author_id"], va.X["author_id"])

for s, y, nm in [(tr, yt, "tr"), (va, yv, "va")]:
    ds = []
    for d in np.unique(s.date):
        m = s.date == d
        ds.append(f"{str(int(d))[-4:]}:{y[m].mean():.3f}")
    print(f"DATE {nm} " + " ".join(ds))

def mutual_info_millibits(x, y, size):
    n = np.bincount(x, minlength=size).astype(np.float64)
    p = np.bincount(x, weights=y, minlength=size)
    q = n - p
    ny1, ny0, total = p.sum(), q.sum(), n.sum()
    mi = 0.0
    m = p > 0
    mi += np.sum((p[m] / total) * np.log2((p[m] * total) / (n[m] * ny1)))
    m = q > 0
    mi += np.sum((q[m] / total) * np.log2((q[m] * total) / (n[m] * ny0)))
    return 1000.0 * mi

rows = []
for name in tr.X:
    xt, xv = tr.X[name], va.X[name]
    size = max(FEATURE_CARDINALITIES[name], int(xt.max()) + 1, int(xv.max()) + 1)
    ct = np.bincount(xt, minlength=size)
    cv = np.bincount(xv, minlength=size)
    new = np.mean(ct[xv] == 0)
    top = ct.max() / len(xt)
    mi = mutual_info_millibits(xt, yt, size)
    rows.append((mi, name, FEATURE_CARDINALITIES[name],
                 int(np.count_nonzero(ct)), int(np.count_nonzero(cv)), new, top))

print("FIELDS sorted_by_train_MI: K, train/valid used IDs, valid-new-row%, train-top%")
for mi, name, k, ut, uv, new, top in sorted(rows, reverse=True):
    print(f"F {name:22s} K={k} U={ut}/{uv} new={100*new:.1f} top={100*top:.1f} MI={mi:.2f}")

for name, x in tr.num.items():
    a = x.astype(np.float64)
    miss = ~np.isfinite(a)
    z = a[~miss]
    med = np.median(z) if len(z) else 0.0
    p95 = np.percentile(z, 95) if len(z) else 0.0
    filled = np.where(miss, med, a)
    corr = np.corrcoef(np.log1p(np.maximum(filled, 0)), yt)[0, 1]
    print(f"NUM {name:25s} miss={miss.mean():.3f} med={med:.1f} p95={p95:.1f} logcorr={corr:.3f}")

print("AUX keys=" + ",".join(sorted(tr.aux.keys()))[:220])
for key in ("video_id", "author_id"):
    h = historical_features("valid", key=key)
    desc = ",".join(f"{k}:{np.asarray(v).shape}/{np.asarray(v).dtype}" for k, v in h.items())
    print(f"HIST {key} nkeys={len(h)} " + desc[:280])