import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")
y = tr.y.astype(np.float64)

def qtext(a):
    q = np.percentile(a, [0, 25, 50, 75, 90, 99, 100])
    return "/".join(f"{x:.0f}" for x in q)

def user_summary(s, name):
    u, inv, cnt = np.unique(s.user_id, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=s.y, minlength=len(u))
    print(f"USER {name} n={len(u)} rowsQ={qtext(cnt)} posQ={qtext(pos)} "
          f"zero={np.mean(pos==0):.3f} all={np.mean(pos==cnt):.3f} "
          f"mixed={np.mean((pos>0)&(pos<cnt)):.3f}")

def dates(s, name):
    vals = []
    for d in np.unique(s.date):
        m = s.date == d
        vals.append(f"{str(int(d))[-4:]}:{m.sum()}@{s.y[m].mean():.3f}")
    print(f"DATE {name} " + ",".join(vals))

def overlap(key):
    a = getattr(tr, key) if hasattr(tr, key) else tr.X[key]
    b = getattr(va, key) if hasattr(va, key) else va.X[key]
    au = np.unique(a)
    bu = np.unique(b)
    row_new = np.mean(~np.isin(b, au))
    uniq_new = np.mean(~np.isin(bu, au))
    print(f"COLD {key} trainU={len(au)} validU={len(bu)} "
          f"validRowNew={row_new:.3f} validUniqNew={uniq_new:.3f}")

print(f"SHAPE train={len(y)} valid={len(va.user_id)} test={len(te.user_id)} "
      f"Xfields={len(tr.X)} numfields={len(tr.num)}")
print(f"LABEL train pos={int(y.sum())} rate={y.mean():.4f} "
      f"valid pos={int(va.y.sum())} rate={va.y.mean():.4f}")
print("XSCALAR " + str(all(np.asarray(v).shape == (len(y),) for v in tr.X.values())) +
      f" dtype={next(iter(tr.X.values())).dtype}")
user_summary(tr, "train")
user_summary(va, "valid")
dates(tr, "train")
dates(va, "valid")
for k in ("user_id", "video_id", "author_id"):
    overlap(k)

p = y.mean()
hy = -(p*np.log(max(p, 1e-15)) + (1-p)*np.log(max(1-p, 1e-15)))
for name in tr.X:
    x = tr.X[name]
    ux, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=y, minlength=len(ux))
    pc = np.clip(pos / cnt, 1e-12, 1-1e-12)
    mi = np.sum((cnt / len(y)) * (
        pc*np.log(pc/p) + (1-pc)*np.log((1-pc)/(1-p))
    ))
    vu = np.unique(va.X[name])
    tu = np.unique(te.X[name])
    newv = np.mean(~np.isin(va.X[name], ux))
    singleton_rows = np.sum(cnt == 1) / len(y)
    print(f"F {name} C={FEATURE_CARDINALITIES[name]} "
          f"U={len(ux)}/{len(vu)}/{len(tu)} z={np.mean(x==0):.3f} "
          f"newV={newv:.3f} IG={mi/hy:.4f} s1={singleton_rows:.3f}")

for name, a0 in tr.num.items():
    a = np.asarray(a0, dtype=np.float64)
    ok = np.isfinite(a)
    if ok.sum():
        z = np.log1p(np.maximum(a[ok], 0))
        corr = np.corrcoef(z, y[ok])[0, 1] if np.std(z) > 0 else 0.0
        qs = np.percentile(a[ok], [1, 50, 99])
        print(f"N {name} miss={1-ok.mean():.3f} "
              f"q1/50/99={qs[0]:.2g}/{qs[1]:.2g}/{qs[2]:.2g} "
              f"logcorr={corr:.4f}")

for key in ("video_id", "author_id"):
    h = historical_features("train", key=key)
    desc = ",".join(f"{k}:{np.asarray(v).dtype}{np.asarray(v).shape}"
                    for k, v in h.items())
    print(f"HIST {key} {desc}")