import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
yt = tr.y.astype(np.float64)
yv = va.y.astype(np.float64)

def qstr(a):
    q = np.quantile(a, [0, .25, .5, .75, .9, .99, 1])
    return "/".join(f"{x:.0f}" for x in q)

def user_summary(s, y, name):
    u, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=y)
    print(f"USER {name} n={len(u)} rows_q={qstr(n)} pos_q={qstr(p)} "
          f"zero={np.mean(p==0):.3f} all={np.mean(p==n):.3f}")

def date_summary(s, y, name):
    ds = []
    for d in np.unique(s.date):
        m = s.date == d
        ds.append(f"{str(int(d))[-4:]}:{m.sum()}/{y[m].mean():.3f}")
    print(f"DATE {name} " + ",".join(ds))

def empirical_nmi(x, y, card):
    n = len(y)
    cnt = np.bincount(x, minlength=card).astype(np.float64)
    pos = np.bincount(x, weights=y, minlength=card)
    nz = cnt > 0
    p = pos[nz] / cnt[nz]
    hcond = np.sum(cnt[nz] * (-(p*np.log(p, where=p>0, out=np.zeros_like(p))
             + (1-p)*np.log(1-p, where=(1-p)>0, out=np.zeros_like(p))))) / n
    b = y.mean()
    hy = -(b*np.log(b) + (1-b)*np.log(1-b))
    return max(0.0, (hy-hcond) / hy) if hy > 0 else 0.0

print(f"ROWS train={len(yt)} valid={len(yv)} label={yt.mean():.4f}/{yv.mean():.4f}")
print(f"SHAPES X_fields={len(tr.X)} num_fields={len(tr.num)} "
      f"all_X_scalar={all(np.asarray(x).shape==(len(yt),) for x in tr.X.values())}")
date_summary(tr, yt, "tr")
date_summary(va, yv, "va")
user_summary(tr, yt, "tr")
user_summary(va, yv, "va")

for key, at, av in [
    ("user", tr.user_id, va.user_id),
    ("video", tr.video_id, va.video_id),
    ("author", tr.X["author_id"], va.X["author_id"])
]:
    seen = np.unique(at)
    cold = ~np.isin(av, seen)
    print(f"COLD {key} row={cold.mean():.3f} unique={np.mean(~np.isin(np.unique(av),seen)):.3f}")

base = int(max(tr.video_id.max(), va.video_id.max())) + 1
ptr = tr.user_id.astype(np.int64) * base + tr.video_id
pva = va.user_id.astype(np.int64) * base + va.video_id
uptr = np.unique(ptr)
print(f"PAIR train_repeat_rows={1-len(uptr)/len(ptr):.3f} valid_seen_rows={np.isin(pva,uptr).mean():.3f}")

print("FEATURE name C=declared O=train/valid U=valid-row-unseen% D=max-share% I=empirical-NMI(train/valid)")
for name in sorted(tr.X):
    xt = tr.X[name]
    xv = va.X[name]
    card = FEATURE_CARDINALITIES[name]
    ct = np.bincount(xt, minlength=card)
    cv = np.bincount(xv, minlength=card)
    seen = ct > 0
    unseen = (~seen[xv]).mean() * 100
    dom = ct.max() / len(xt) * 100
    it = empirical_nmi(xt, yt, card) * 100
    iv = empirical_nmi(xv, yv, card) * 100
    print(f"F {name:24s} C{card} O{np.count_nonzero(ct)}/{np.count_nonzero(cv)} "
          f"U{unseen:.1f} D{dom:.1f} I{it:.2f}/{iv:.2f}")

for name in sorted(tr.num):
    x = tr.num[name].astype(np.float64)
    ok = np.isfinite(x)
    if ok.sum():
        lo, hi = np.quantile(x[ok], [.2, .8])
        lr = yt[ok & (x <= lo)].mean()
        hr = yt[ok & (x >= hi)].mean()
        med = np.median(x[ok])
        print(f"NUM {name:26s} miss={1-ok.mean():.3f} med={med:.2f} "
              f"q20/80={lo:.2f}/{hi:.2f} ylo/hi={lr:.3f}/{hr:.3f}")

for key in ("video_id", "author_id"):
    h = historical_features("train", key=key)
    keys = ",".join(sorted(h.keys()))
    shapes = sorted(set(str(np.asarray(v).shape) for v in h.values()))
    print(f"HIST {key} nfeat={len(h)} shapes={';'.join(shapes)} keys={keys}")