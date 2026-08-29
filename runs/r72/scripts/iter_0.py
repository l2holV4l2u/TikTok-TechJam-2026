import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")


def qstr(a):
    q = np.quantile(a, [0, .25, .5, .75, .9, .99, 1])
    return "/".join(f"{x:.0f}" for x in q)


def user_summary(s, name):
    _, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    print(
        f"USER {name} U={len(n)} rows_q={qstr(n)} pos_q={qstr(p)} "
        f"zero={np.mean(p == 0):.3f} all={np.mean(p == n):.3f}"
    )


def date_summary(s, name):
    ds = []
    for d in np.unique(s.date):
        z = s.date == d
        ds.append(f"{d}:{z.sum()}:{s.y[z].mean():.3f}")
    print(f"DATE {name} " + " ".join(ds))


def overlap(field):
    a, b = tr.X[field], va.X[field]
    k = max(FEATURE_CARDINALITIES[field], int(a.max()) + 1, int(b.max()) + 1)
    seen = np.zeros(k, dtype=bool)
    seen[a] = True
    unseen = ~seen[b]
    bu = np.unique(b)
    return unseen.mean(), np.mean(~seen[bu]), len(np.unique(a)), len(bu)


def corrected_cramers(x, y):
    counts = np.bincount(x)
    pos = np.bincount(x, weights=y, minlength=len(counts))
    nz = counts > 0
    counts, pos = counts[nz], pos[nz]
    n = counts.sum()
    p = pos.sum() / n
    if p <= 0 or p >= 1 or len(counts) <= 1:
        return 0.0
    phi2 = np.sum(counts * (pos / counts - p) ** 2) / (n * p * (1 - p))
    k = len(counts)
    corr = (k - 1) / max(n - 1, 1)
    phi2c = max(0.0, phi2 - corr)
    kcorr = k - (k - 1) ** 2 / max(n - 1, 1)
    rcorr = 2.0 - 1.0 / max(n - 1, 1)
    den = max(min(kcorr - 1, rcorr - 1), 1e-12)
    return float(np.sqrt(phi2c / den))


def numeric_assoc(x, y):
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    if ok.sum() < 2 or np.std(x[ok]) == 0:
        return ok.mean(), 0.0
    c = np.corrcoef(np.log1p(np.maximum(x[ok], 0)), y[ok])[0, 1]
    return ok.mean(), float(c)


print(
    f"SHAPE train={len(tr.user_id)} valid={len(va.user_id)} "
    f"Xfields={len(tr.X)} numfields={len(tr.num)}"
)
first = next(iter(tr.X))
print(
    f"ARRAY X[{first}]={tr.X[first].shape}/{tr.X[first].dtype} "
    f"y={tr.y.shape}/{tr.y.dtype} uid={tr.user_id.shape}/{tr.user_id.dtype} "
    f"time={tr.time_ms.shape}/{tr.time_ms.dtype}"
)
print(
    f"LABEL train_pos={tr.y.sum()} rate={tr.y.mean():.4f} "
    f"valid_pos={va.y.sum()} rate={va.y.mean():.4f}"
)
user_summary(tr, "train")
user_summary(va, "valid")
date_summary(tr, "train")
date_summary(va, "valid")

for f in ("user_id", "video_id", "author_id", "tag", "music_type"):
    rr, ur, ut, uv = overlap(f)
    print(f"OVERLAP {f} trU={ut} vaU={uv} unseen_rows={rr:.3f} unseen_ids={ur:.3f}")

for f in sorted(tr.X):
    a, b = tr.X[f], va.X[f]
    ca = np.bincount(a)
    cb = np.bincount(b)
    ua, ub = np.count_nonzero(ca), np.count_nonzero(cb)
    dom_a = ca.max() / len(a)
    dom_b = cb.max() / len(b)
    aa = corrected_cramers(a, tr.y)
    ab = corrected_cramers(b, va.y)
    print(
        f"F {f} K={FEATURE_CARDINALITIES[f]} U={ua}/{ub} "
        f"dom={dom_a:.2f}/{dom_b:.2f} V={aa:.3f}/{ab:.3f}"
    )

for f in sorted(tr.num):
    ft, ct = numeric_assoc(tr.num[f], tr.y)
    fv, cv = numeric_assoc(va.num[f], va.y)
    print(f"N {f} finite={ft:.3f}/{fv:.3f} logcorr={ct:+.3f}/{cv:+.3f}")

for entity in ("video_id", "author_id"):
    h = historical_features("valid", key=entity)
    parts = []
    for name, x in sorted(h.items()):
        x = np.asarray(x, dtype=np.float64)
        ok = np.isfinite(x)
        if ok.sum() > 1 and np.std(x[ok]) > 0:
            c = np.corrcoef(x[ok], va.y[ok])[0, 1]
        else:
            c = 0.0
        parts.append(f"{name}:{ok.mean():.2f}:{c:+.3f}")
    line = f"H {entity} shape={next(iter(h.values())).shape} " + ",".join(parts)
    print(line[:700])

order = np.lexsort((np.arange(len(tr.user_id)), tr.time_ms, tr.user_id))
u = tr.user_id[order]
t = tr.time_ms[order]
same_user = u[1:] == u[:-1]
tied = same_user & (t[1:] == t[:-1])
print(
    f"TIME train_same_timestamp_adjacent={tied.sum()}/{same_user.sum()} "
    f"fraction={tied.sum()/max(same_user.sum(),1):.4f}"
)