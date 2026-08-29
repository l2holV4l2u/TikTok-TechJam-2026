import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
y = tr.y.astype(np.float64)
yv = va.y.astype(np.float64)

print(f"SHAPE train={len(y)} valid={len(yv)} X={len(tr.X)} num={len(tr.num)}")
scalar_ok = all(np.asarray(v).shape == (len(y),) for v in tr.X.values())
print(f"X_SCALAR_PER_ROW={scalar_ok} fields={','.join(tr.X.keys())}")
print(f"LABEL train_rate={y.mean():.4f} valid_rate={yv.mean():.4f}")

def user_stats(s, labels, tag):
    u, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=labels)
    qn = np.quantile(n, [0, .25, .5, .75, .9, .99]).astype(int)
    qp = np.quantile(p, [0, .25, .5, .75, .9, .99])
    print(f"USER {tag} U={len(u)} rows_q={qn.tolist()} pos_q={np.round(qp,2).tolist()}")
    print(f"USER {tag} zero={(p==0).mean():.3f} all={(p==n).mean():.3f} mixed={((p>0)&(p<n)).mean():.3f}")

user_stats(tr, y, "tr")
user_stats(va, yv, "va")

def overlap(a, b, name):
    seen = np.unique(a)
    newrow = ~np.isin(b, seen)
    newuniq = ~np.isin(np.unique(b), seen)
    print(f"OVERLAP {name} new_rows={newrow.mean():.4f} new_unique={newuniq.mean():.4f}")

overlap(tr.user_id, va.user_id, "user")
overlap(tr.video_id, va.video_id, "video")
overlap(tr.X["author_id"], va.X["author_id"], "author")

for s, labels, tag in [(tr, y, "tr"), (va, yv, "va")]:
    vals = []
    for d in np.unique(s.date):
        m = s.date == d
        vals.append(f"{str(int(d))[-4:]}:{m.sum()}/{labels[m].mean():.3f}")
    print(f"DATE {tag} " + ",".join(vals))

py = y.mean()
hy = -(py*np.log(py) + (1-py)*np.log(1-py))

def categorical_summary(name):
    x = np.asarray(tr.X[name], dtype=np.int64)
    xv = np.asarray(va.X[name], dtype=np.int64)
    C = int(FEATURE_CARDINALITIES[name])
    c = np.bincount(x, minlength=C).astype(np.float64)
    p = np.bincount(x, weights=y, minlength=C)
    cv = np.bincount(xv, minlength=C)
    nz = c > 0
    mi = 0.0
    p1 = p[nz]
    p0 = c[nz] - p1
    cc = c[nz]
    m = p1 > 0
    mi += np.sum((p1[m]/len(y))*np.log((p1[m]/len(y))/((cc[m]/len(y))*py)))
    m = p0 > 0
    mi += np.sum((p0[m]/len(y))*np.log((p0[m]/len(y))/((cc[m]/len(y))*(1-py))))
    new = (c[xv] == 0).mean()
    print(f"F {name}|C{C} T{np.count_nonzero(c)} V{np.count_nonzero(cv)} "
          f"new{new:.3f} z{(xv==0).mean():.3f} top{c.max()/len(y):.3f} MI{mi/hy:.4f}")

for name in tr.X:
    categorical_summary(name)

for name in tr.num:
    a = np.asarray(tr.num[name], dtype=np.float64)
    b = np.asarray(va.num[name], dtype=np.float64)
    ok = np.isfinite(a)
    q = np.nanquantile(a, [.01, .5, .99])
    corr = np.corrcoef(a[ok], y[ok])[0, 1] if ok.sum() > 2 and np.std(a[ok]) > 0 else 0.0
    print(f"N {name}|missT{1-ok.mean():.3f} missV{np.isnan(b).mean():.3f} "
          f"q01/50/99={q[0]:.2g}/{q[1]:.2g}/{q[2]:.2g} corr={corr:.3f}")

for key in ["video_id", "author_id"]:
    h = historical_features("valid", key=key)
    ranked = []
    for name, a in h.items():
        a = np.asarray(a)
        ok = np.isfinite(a)
        c = np.corrcoef(a[ok], yv[ok])[0, 1] if ok.sum() > 2 and np.std(a[ok]) > 0 else 0.0
        ranked.append((abs(c), name, c, a.shape, 1-ok.mean()))
    ranked.sort(reverse=True)
    desc = ";".join(f"{n}:{c:.3f}/miss{m:.2f}" for _, n, c, _, m in ranked[:8])
    print(f"HIST {key} n={len(h)} shape={next(iter(h.values())).shape} top={desc}")

print("AUX_OUTCOMES_NOT_FEATURES=" + ",".join(sorted(tr.aux.keys())))