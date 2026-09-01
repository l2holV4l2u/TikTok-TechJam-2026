import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES

tr = load("train")
va = load("valid")
te = load("test")  # Features only; never access test labels.

yt = np.asarray(tr.y, dtype=np.int8)
yv = np.asarray(va.y, dtype=np.int8)

print(f"ROWS train={len(yt)} valid={len(yv)} test={len(te.user_id)}")
print(f"LABEL rate train={yt.mean():.5f} valid={yv.mean():.5f}")
print("DATES train=" + ",".join(
    f"{d}:{yt[tr.date == d].mean():.3f}" for d in np.unique(tr.date)
))
print("DATES valid=" + ",".join(
    f"{d}:{yv[va.date == d].mean():.3f}" for d in np.unique(va.date)
))
print("DATES test=" + ",".join(map(str, np.unique(te.date))))

def group_summary(name, ids, y):
    _, inv, n = np.unique(ids, return_inverse=True, return_counts=True)
    p = np.bincount(inv, weights=y, minlength=len(n))
    mixed = (p > 0) & (p < n)
    qn = np.percentile(n, [10, 25, 50, 75, 90, 99]).astype(int)
    qp = np.percentile(p, [10, 25, 50, 75, 90, 99]).astype(int)
    pos_weight = p[mixed].sum() / max(p.sum(), 1)
    print(
        f"{name} users={len(n)} zero={np.mean(p==0):.3f} "
        f"allpos={np.mean(p==n):.3f} mixed={mixed.mean():.3f} "
        f"auc_poswt={pos_weight:.3f} n_q={qn.tolist()} p_q={qp.tolist()}"
    )

group_summary("USER train", tr.user_id, yt)
group_summary("USER valid", va.user_id, yv)

def overlap(name, train_ids, valid_ids, test_ids):
    seen = np.unique(train_ids)
    vu = np.unique(valid_ids)
    tu = np.unique(test_ids)
    vrow = 1.0 - np.isin(valid_ids, seen).mean()
    trow = 1.0 - np.isin(test_ids, seen).mean()
    vnew = 1.0 - np.isin(vu, seen).mean()
    tnew = 1.0 - np.isin(tu, seen).mean()
    print(
        f"OVERLAP {name} train_unique={len(seen)} "
        f"valid_unique={len(vu)} test_unique={len(tu)} "
        f"new_unique={vnew:.3f}/{tnew:.3f} new_rows={vrow:.3f}/{trow:.3f}"
    )

overlap("user", tr.user_id, va.user_id, te.user_id)
overlap("video", tr.video_id, va.video_id, te.video_id)

names = list(tr.X)
shape_ok = all(
    np.asarray(tr.X[k]).ndim == 1 and len(tr.X[k]) == len(yt)
    for k in names
)
print(
    f"X fields={len(names)} scalar_per_row={shape_ok} "
    f"card_sum={sum(FEATURE_CARDINALITIES[k] for k in names)} "
    f"card_max={max(FEATURE_CARDINALITIES[k] for k in names)}"
)

aux_keys = sorted(tr.aux)
aux_shapes = sorted({
    (np.asarray(v).ndim, tuple(np.asarray(v).shape[1:]), str(np.asarray(v).dtype))
    for v in tr.aux.values()
})
print("AUX leakage_outcomes keys=" + ",".join(aux_keys))
print(f"AUX shape_signatures={aux_shapes}")

# Bias-corrected mutual information after pooling categories with support <20.
# This reduces the misleading apparent signal from singleton/high-cardinality IDs.
def robust_mi(count, pos, min_count=20):
    count = count.astype(np.float64, copy=False)
    pos = pos.astype(np.float64, copy=False)
    keep = count >= min_count
    ns = count[keep]
    ps = pos[keep]
    rare_n = count[~keep].sum()
    rare_p = pos[~keep].sum()
    if rare_n:
        ns = np.append(ns, rare_n)
        ps = np.append(ps, rare_p)
    good = ns > 0
    ns, ps = ns[good], ps[good]
    n = ns.sum()
    P = ps.sum()
    N = n - P
    if n == 0 or P == 0 or N == 0:
        return 0.0
    neg = ns - ps
    mi = 0.0
    m = ps > 0
    mi += np.sum((ps[m] / n) * np.log2((ps[m] * n) / (ns[m] * P)))
    m = neg > 0
    mi += np.sum((neg[m] / n) * np.log2((neg[m] * n) / (ns[m] * N)))
    bias = (len(ns) - 1) / (2.0 * n * np.log(2.0))
    return max(0.0, mi - bias)

print("FIELD format: c=declared a=train/valid/test active u=valid/test unseen-row top=train-share mi=train/valid bits")
for k in names:
    x = np.asarray(tr.X[k], dtype=np.int64)
    xv = np.asarray(va.X[k], dtype=np.int64)
    xe = np.asarray(te.X[k], dtype=np.int64)
    L = int(max(x.max(initial=0), xv.max(initial=0), xe.max(initial=0))) + 1

    ct = np.bincount(x, minlength=L)
    pt = np.bincount(x[yt == 1], minlength=L)
    cv = np.bincount(xv, minlength=L)
    pv = np.bincount(xv[yv == 1], minlength=L)
    ce = np.bincount(xe, minlength=L)

    seen = ct > 0
    uv = np.mean(~seen[xv])
    ue = np.mean(~seen[xe])
    top = ct.max() / len(x)
    mit = robust_mi(ct, pt)
    miv = robust_mi(cv, pv)

    print(
        f"F {k} c={FEATURE_CARDINALITIES[k]} "
        f"a={np.count_nonzero(ct)}/{np.count_nonzero(cv)}/{np.count_nonzero(ce)} "
        f"u={uv:.3f}/{ue:.3f} top={top:.3f} mi={mit:.5f}/{miv:.5f}"
    )