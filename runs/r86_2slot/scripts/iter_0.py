import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")

def user_summary(s, name):
    u, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    qn = np.quantile(n, [0, .25, .5, .75, .9, .99]).astype(int)
    qp = np.quantile(p, [0, .25, .5, .75, .9, .99])
    zero = np.mean(p == 0)
    mixed = np.mean((p > 0) & (p < n))
    print(f"USER {name} U={len(u)} rowsQ={qn.tolist()} posQ={np.round(qp,1).tolist()} "
          f"zero={zero:.3f} mixed={mixed:.3f}")

def adjusted_mi(x, y, card):
    n = np.bincount(x, minlength=card).astype(np.float64)
    p = np.bincount(x, weights=y, minlength=card)
    used = n > 0
    r = np.divide(p[used], n[used])
    eps = 1e-15
    py = np.mean(y)
    hy = -(py*np.log2(py+eps) + (1-py)*np.log2(1-py+eps))
    hc = -(r*np.log2(r+eps) + (1-r)*np.log2(1-r+eps))
    mi = hy - np.sum(n[used] * hc) / len(y)
    bias = (used.sum() - 1) / (2 * len(y) * np.log(2))
    return max(0.0, mi - bias), n

print(f"SHAPE train={len(tr.y)} valid={len(va.y)} X={len(tr.X)} num={len(tr.num)}")
print(f"LABEL train={tr.y.mean():.5f} ({tr.y.sum()}) valid={va.y.mean():.5f} ({va.y.sum()})")
print("ARRAYS X1D=" + str(all(np.asarray(v).shape == (len(tr.y),) for v in tr.X.values())) +
      " time=" + str(tr.time_ms.shape) + " date=" + str(tr.date.shape))
user_summary(tr, "train")
user_summary(va, "valid")

for name, split in [("train", tr), ("valid", va)]:
    ds = np.unique(split.date)
    rates = [f"{d}:{split.y[split.date == d].mean():.3f}" for d in ds]
    print("DATE " + name + " " + " ".join(rates))

for key in ["user_id", "video_id", "author_id"]:
    tx = np.asarray(tr.X[key])
    vx = np.asarray(va.X[key])
    tu = np.unique(tx[tx != 0])
    vu = np.unique(vx[vx != 0])
    known = (vx != 0) & np.isin(vx, tu)
    new_ids = np.mean(~np.isin(vu, tu)) if len(vu) else 0.0
    print(f"OVERLAP {key} trainIDs={len(tu)} validIDs={len(vu)} "
          f"validKnownRows={known.mean():.3f} newValidIDs={new_ids:.3f} zeroRows={np.mean(vx==0):.3f}")

order = np.lexsort((np.arange(len(tr.y)), tr.time_ms, tr.user_id))
same = ((tr.user_id[order][1:] == tr.user_id[order][:-1]) &
        (tr.time_ms[order][1:] == tr.time_ms[order][:-1]))
print(f"TIME trainRangeMs={tr.time_ms.min()}..{tr.time_ms.max()} sameUserTimestampRows={same.mean():.4f}")

print("FIELDS name card seenT/seenV unseenVrow topT adjMI_T/V_millibits")
for f in tr.X:
    card = FEATURE_CARDINALITIES[f]
    mit, nt = adjusted_mi(tr.X[f], tr.y, card)
    miv, nv = adjusted_mi(va.X[f], va.y, card)
    seen_t = np.count_nonzero(nt)
    seen_v = np.count_nonzero(nv)
    unseen = np.mean((va.X[f] == 0) | (nt[va.X[f]] == 0))
    top = nt.max() / len(tr.y)
    print(f"F {f} {card} {seen_t}/{seen_v} {unseen:.3f} {top:.3f} {1000*mit:.2f}/{1000*miv:.2f}")

for f, x in tr.num.items():
    x = np.asarray(x)
    ok = np.isfinite(x)
    z = x[ok].astype(np.float64)
    y = tr.y[ok].astype(np.float64)
    q = np.quantile(z, [.1, .5, .9]) if len(z) else [np.nan]*3
    corr = np.corrcoef(z, y)[0, 1] if len(z) > 1 and np.std(z) > 0 else np.nan
    lo, hi = np.quantile(z, [.25, .75]) if len(z) else [0, 0]
    rlo = y[z <= lo].mean() if len(z) else np.nan
    rhi = y[z >= hi].mean() if len(z) else np.nan
    print(f"NUM {f} miss={1-ok.mean():.3f} q10/50/90={q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f} "
          f"corr={corr:.3f} yQ1/Q4={rlo:.3f}/{rhi:.3f}")

for key in ["video_id", "author_id"]:
    h = historical_features("valid", key=key)
    desc = []
    for k, v in h.items():
        a = np.asarray(v)
        desc.append(f"{k}[{a.shape[0]},nan={np.mean(~np.isfinite(a)):.2f}]")
    print("HIST " + key + " " + " ".join(desc))