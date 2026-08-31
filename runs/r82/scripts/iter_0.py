import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")

print(f"ROWS train={len(tr.user_id)} valid={len(va.user_id)} test={len(te.user_id)}")
print(f"LABEL train_rate={tr.y.mean():.5f} train_pos={int(tr.y.sum())} "
      f"valid_rate={va.y.mean():.5f} valid_pos={int(va.y.sum())}")

def date_rates(s):
    parts = []
    for d in np.unique(s.date):
        m = s.date == d
        parts.append(f"{int(d)%100:02d}:{m.sum()}/{s.y[m].mean():.3f}")
    return " ".join(parts)

print("DATE train " + date_rates(tr))
print("DATE valid " + date_rates(va))

def user_summary(name, s):
    u, inv, cnt = np.unique(s.user_id, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=s.y, minlength=len(u))
    rq = np.quantile(cnt, [0.1, 0.5, 0.9, 0.99])
    pq = np.quantile(pos, [0.1, 0.5, 0.9, 0.99])
    zero = np.mean(pos == 0)
    allp = np.mean(pos == cnt)
    mixed = np.mean((pos > 0) & (pos < cnt))
    print(f"USER {name} n={len(u)} rows_q10/50/90/99={rq.astype(int).tolist()} "
          f"pos_q={pq.tolist()} zero={zero:.3f} all={allp:.3f} mixed={mixed:.3f}")

user_summary("train", tr)
user_summary("valid", va)

def overlap(field):
    c = FEATURE_CARDINALITIES[field]
    seen = np.zeros(c, dtype=bool)
    seen[np.asarray(tr.X[field])] = True
    vu = np.unique(va.X[field])
    tu = np.unique(te.X[field])
    v_unq = np.mean(~seen[vu])
    t_unq = np.mean(~seen[tu])
    v_row = np.mean(~seen[va.X[field]])
    t_row = np.mean(~seen[te.X[field]])
    print(f"OV {field} V_unq={v_unq:.3f} V_row={v_row:.3f} "
          f"T_unq={t_unq:.3f} T_row={t_row:.3f}")

for f in ("user_id", "video_id", "author_id"):
    overlap(f)

shape_ok = all(np.asarray(x).ndim == 1 and len(x) == len(tr.user_id)
               for x in tr.X.values())
print(f"X fields={len(tr.X)} scalar_per_row={shape_ok} "
      f"example_shape={tr.X['video_id'].shape}")

n = len(tr.y)
nv = len(va.user_id)
nt = len(te.user_id)
py1 = tr.y.mean()
py0 = 1.0 - py1

for f in tr.X:
    c = FEATURE_CARDINALITIES[f]
    xt = tr.X[f]
    xv = va.X[f]
    xe = te.X[f]
    ct = np.bincount(xt, minlength=c).astype(np.float64)
    cp = np.bincount(xt, weights=tr.y, minlength=c)
    cv = np.bincount(xv, minlength=c).astype(np.float64)
    ce = np.bincount(xe, minlength=c).astype(np.float64)

    pc = ct / n
    p1 = cp / n
    p0 = (ct - cp) / n
    mi = 0.0
    m = p1 > 0
    mi += np.sum(p1[m] * np.log2(p1[m] / (pc[m] * py1)))
    m = p0 > 0
    mi += np.sum(p0[m] * np.log2(p0[m] / (pc[m] * py0)))

    unseen = ct == 0
    vn = cv[unseen].sum() / nv
    tn = ce[unseen].sum() / nt
    tv = 0.5 * np.abs(ct / n - cv / nv).sum()
    dom = ct.max() / n
    used = np.count_nonzero(ct)
    print(f"F {f} c/u={c}/{used} d={dom:.3f} mi={mi*1000:.2f} "
          f"vn={vn:.3f} tn={tn:.3f} tv={tv:.3f}")

for f in tr.num:
    a = np.asarray(tr.num[f], dtype=np.float64)
    b = np.asarray(va.num[f], dtype=np.float64)
    ma = np.isfinite(a)
    mb = np.isfinite(b)
    q = np.nanquantile(a, [0.1, 0.5, 0.9])
    vb = np.nanmedian(b)
    z = np.log1p(np.maximum(a[ma], 0))
    corr = np.corrcoef(z, tr.y[ma])[0, 1] if ma.sum() > 2 and z.std() > 0 else 0.0
    print(f"N {f} missT={1-ma.mean():.3f} q={np.round(q,2).tolist()} "
          f"missV={1-mb.mean():.3f} medV={vb:.2f} logcorr={corr:.3f}")

for key in ("video_id", "author_id"):
    h = historical_features("valid", key=key)
    desc = []
    for name, arr in h.items():
        a = np.asarray(arr)
        desc.append(f"{name}:{a.shape}/{a.dtype}")
    print(f"H {key} " + " ".join(desc))