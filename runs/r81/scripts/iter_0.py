import gc
import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
yt = tr.y.astype(np.float64)
yv = va.y.astype(np.float64)
p = float(yt.mean())

print(f"ROWS train={len(yt)} valid={len(yv)} pos_rate={p:.4f}/{yv.mean():.4f}")
print(f"X fields={len(tr.X)} sample_shape={next(iter(tr.X.values())).shape} "
      f"dtype={next(iter(tr.X.values())).dtype} num={list(tr.num)}")
print(f"AUX outcome-only n={len(tr.aux)} names={','.join(sorted(tr.aux)[:18])}")

def date_line(s, y):
    d, inv = np.unique(s.date, return_inverse=True)
    n = np.bincount(inv)
    z = np.bincount(inv, weights=y)
    return " ".join(f"{x}:{int(nn)}/{zz/nn:.3f}" for x, nn, zz in zip(d, n, z))

print("DATE train", date_line(tr, yt))
print("DATE valid", date_line(va, yv))

def user_summary(name, s, y):
    u, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    z = np.bincount(inv, weights=y)
    qn = np.quantile(n, [0, .25, .5, .75, .9, .99, 1])
    qp = np.quantile(z, [0, .5, .75, .9, .99, 1])
    print(f"USER {name} n={len(u)} rows_q={','.join(f'{x:.0f}' for x in qn)} "
          f"pos_q={','.join(f'{x:.0f}' for x in qp)} zero={(z==0).mean():.3f} "
          f"all={(z==n).mean():.3f}")

user_summary("train", tr, yt)
user_summary("valid", va, yv)

tu = np.unique(tr.user_id)
tv = np.unique(tr.video_id)
print(f"COLD valid_user rows={(~np.isin(va.user_id,tu)).mean():.3f} "
      f"ids={(~np.isin(np.unique(va.user_id),tu)).mean():.3f} "
      f"video_rows={(~np.isin(va.video_id,tv)).mean():.3f} "
      f"video_ids={(~np.isin(np.unique(va.video_id),tv)).mean():.3f}")
del tu, tv
gc.collect()

base = -np.mean(yv*np.log(p) + (1-yv)*np.log1p(-p))
print("CAT name card trU vaU zeroT unseenV dLLx1e3 corrV")
for name in sorted(tr.X):
    x = tr.X[name]
    z = va.X[name]
    m = int(x.max()) + 1
    cnt = np.bincount(x, minlength=m)
    pos = np.bincount(x, weights=yt, minlength=m)
    pred = np.full(len(z), p, dtype=np.float64)
    known = z < m
    kz = z[known]
    seen = cnt[kz] > 0
    ii = np.flatnonzero(known)[seen]
    ids = kz[seen]
    pred[ii] = (pos[ids] + 20.0*p) / (cnt[ids] + 20.0)
    pred = np.clip(pred, 1e-6, 1-1e-6)
    ll = -np.mean(yv*np.log(pred) + (1-yv)*np.log1p(-pred))
    sd = pred.std()
    corr = np.corrcoef(pred, yv)[0, 1] if sd > 0 else 0.0
    unseen = 1.0 - len(ii)/len(z)
    print(f"C {name} {FEATURE_CARDINALITIES[name]} {np.count_nonzero(cnt)} "
          f"{len(np.unique(z))} {(x==0).mean():.2f} {unseen:.2f} "
          f"{1000*(base-ll):+.2f} {corr:+.3f}")
    del cnt, pos, pred
    gc.collect()

print("NUM name missT missV q01/q50/q99_train corr_log_valid")
for name in sorted(tr.num):
    a = tr.num[name].astype(np.float64)
    b = va.num[name].astype(np.float64)
    fa, fb = np.isfinite(a), np.isfinite(b)
    q = np.quantile(a[fa], [.01, .5, .99]) if fa.any() else [np.nan]*3
    if fb.sum() > 1:
        sb = np.sign(b[fb]) * np.log1p(np.abs(b[fb]))
        c = np.corrcoef(sb, yv[fb])[0, 1] if sb.std() else 0.0
    else:
        c = 0.0
    print(f"N {name} {1-fa.mean():.3f} {1-fb.mean():.3f} "
          f"{q[0]:.2g}/{q[1]:.2g}/{q[2]:.2g} {c:+.3f}")

for key in ("video_id", "author_id"):
    ht = historical_features("train", key=key)
    hv = historical_features("valid", key=key)
    desc = []
    for k in sorted(ht):
        a = np.asarray(ht[k])
        desc.append(f"{k}:{a.shape}/{a.dtype}")
    print(f"HIST {key} train={len(yt)} valid={len(yv)} " + " ".join(desc))
    print(f"HISTV {key} " + " ".join(f"{k}:{np.asarray(hv[k]).shape}" for k in sorted(hv)))