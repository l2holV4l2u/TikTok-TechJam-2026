import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
y = tr.y.astype(np.float64)
yv = va.y.astype(np.float64)
n = len(y)

print(f"rows train={n} valid={len(yv)} fields={len(tr.X)} nums={len(tr.num)}")
print(f"X_scalar={all(np.asarray(v).shape == (n,) for v in tr.X.values())} "
      f"dtypes={sorted(set(str(np.asarray(v).dtype) for v in tr.X.values()))}")
print(f"label train={y.mean():.4f} valid={yv.mean():.4f}")

def date_rates(s, yy):
    ds = np.unique(s.date)
    return " ".join(f"{str(int(d))[-4:]}:{yy[s.date == d].mean():.3f}" for d in ds)

print("date_train " + date_rates(tr, y))
print("date_valid " + date_rates(va, yv))

def user_summary(s, yy, name):
    u, inv, cnt = np.unique(s.user_id, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=yy)
    cq = np.quantile(cnt, [0, .25, .5, .75, .9, .99]).astype(int)
    pq = np.quantile(pos, [0, .25, .5, .75, .9, .99])
    print(f"users_{name}={len(u)} rows_q={cq.tolist()} "
          f"pos_q={np.round(pq,1).tolist()} zero={np.mean(pos == 0):.3f}")

user_summary(tr, y, "tr")
user_summary(va, yv, "va")

def overlap(name, a, b):
    ua = np.unique(a)
    ub = np.unique(b)
    seen = np.isin(b, ua)
    uniq_seen = np.isin(ub, ua)
    print(f"overlap_{name} row_unseen={1-seen.mean():.3f} "
          f"unique_unseen={1-uniq_seen.mean():.3f} trU={len(ua)} vaU={len(ub)}")

overlap("user", tr.user_id, va.user_id)
overlap("video", tr.video_id, va.video_id)
overlap("author", tr.X["author_id"], va.X["author_id"])

p = y.mean()
ln2 = np.log(2.0)
print("FIELD c/card tuniq vuniq new top ami_mbit smooth_sd")
for name in tr.X:
    xt = np.asarray(tr.X[name])
    xv = np.asarray(va.X[name])
    card = FEATURE_CARDINALITIES[name]
    cnt = np.bincount(xt, minlength=card).astype(np.float64)
    pos = np.bincount(xt, weights=y, minlength=card)
    obs = cnt > 0
    seen = obs[xv]
    top = cnt.max() / n
    c1 = pos
    c0 = cnt - pos
    mi = 0.0
    for c, py in ((c1, p), (c0, 1.0 - p)):
        z = c > 0
        mi += np.sum((c[z] / n) * np.log2((c[z] * n) / (cnt[z] * n * py)))
    k = int(obs.sum())
    bias = (k - 1) / (2.0 * n * ln2)
    ami = max(0.0, mi - bias) * 1000.0
    rate = (pos[obs] + 20.0 * p) / (cnt[obs] + 20.0)
    sd = np.sqrt(np.sum(cnt[obs] * (rate - p) ** 2) / n)
    print(f"F {name} {card}/{k}/{np.unique(xv).size} "
          f"{1-seen.mean():.3f} {top:.3f} {ami:.3f} {sd:.3f}")

print("NUM nan q10/q50/q90 corr")
for name, arr0 in tr.num.items():
    arr = np.asarray(arr0, dtype=np.float64)
    ok = np.isfinite(arr)
    q = np.quantile(arr[ok], [.1, .5, .9])
    a = np.log1p(np.maximum(arr[ok], 0))
    b = y[ok]
    ac = a - a.mean()
    bc = b - b.mean()
    den = np.sqrt(np.sum(ac * ac) * np.sum(bc * bc))
    corr = np.sum(ac * bc) / den if den else 0.0
    print(f"N {name} {1-ok.mean():.3f} "
          f"{q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f} {corr:.3f}")

def hist_report(key):
    h = historical_features("valid", key=key)
    vals = []
    for name, arr0 in h.items():
        arr = np.asarray(arr0, dtype=np.float64)
        ok = np.isfinite(arr)
        if ok.sum() < 2 or np.std(arr[ok]) == 0:
            corr = 0.0
        else:
            a = arr[ok] - arr[ok].mean()
            b = yv[ok] - yv[ok].mean()
            den = np.sqrt(np.sum(a*a) * np.sum(b*b))
            corr = np.sum(a*b) / den if den else 0.0
        vals.append((abs(corr), name, corr, 1-ok.mean(), arr.shape))
    vals.sort(reverse=True)
    top = ";".join(f"{z[1]}:{z[2]:.3f}/nan{z[3]:.2f}" for z in vals[:4])
    print(f"HIST {key} n={len(vals)} shape={vals[0][4] if vals else None} top={top}")

hist_report("video_id")
hist_report("author_id")