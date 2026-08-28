import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES

tr = load("train")
va = load("valid")
y = np.asarray(tr.y, dtype=np.float64)
yv = np.asarray(va.y, dtype=np.float64)

def qstr(a):
    q = np.quantile(a, [0, .25, .5, .75, .9, .99, 1])
    return "/".join(f"{v:.0f}" for v in q)

def group_stats(ids, labels):
    _, inv = np.unique(ids, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=labels)
    mixed = (p > 0) & (p < n)
    return n, p, mixed

print(f"rows train={len(y)} valid={len(yv)} fields={len(tr.X)}")
print(f"label_rate train={y.mean():.5f} valid={yv.mean():.5f} delta={yv.mean()-y.mean():+.5f}")
print("X_scalar=" + str(all(np.asarray(x).ndim == 1 and len(x) == len(y)
                            for x in tr.X.values())))
print("aux_keys_outcomes_only=" + ",".join(sorted(tr.aux.keys())))

for tag, s, lab in [("train", tr, y), ("valid", va, yv)]:
    n, p, mixed = group_stats(np.asarray(s.user_id), lab)
    total_pos = p.sum()
    pos_share = p[mixed].sum() / total_pos if total_pos else 0
    print(f"{tag}_users={len(n)} rows_q={qstr(n)} pos_q={qstr(p)}")
    print(f"{tag}_users zero={np.mean(p==0):.3f} all={np.mean(p==n):.3f} "
          f"mixed={mixed.mean():.3f} mixed_pos_share={pos_share:.3f}")

tu = np.unique(tr.user_id)
tv = np.unique(tr.video_id)
vu_new = ~np.isin(va.user_id, tu)
vv_new = ~np.isin(va.video_id, tv)
print(f"valid_unseen user_rows={vu_new.mean():.4f} user_ids={np.mean(~np.isin(np.unique(va.user_id),tu)):.4f}")
print(f"valid_unseen video_rows={vv_new.mean():.4f} video_ids={np.mean(~np.isin(np.unique(va.video_id),tv)):.4f}")
print("valid_cold_rows oldUoldV/newUoldV/oldUnewV/newUnewV="
      + "/".join(f"{z:.3f}" for z in [
          np.mean(~vu_new & ~vv_new), np.mean(vu_new & ~vv_new),
          np.mean(~vu_new & vv_new), np.mean(vu_new & vv_new)]))

umatch, vmatch = [], []
for name, x in tr.X.items():
    a = np.asarray(x)
    if a.shape == np.asarray(tr.user_id).shape and np.array_equal(a, tr.user_id):
        umatch.append(name)
    if a.shape == np.asarray(tr.video_id).shape and np.array_equal(a, tr.video_id):
        vmatch.append(name)
print(f"raw_id_exact_matches user={umatch} video={vmatch}")
print("FIELD columns: name C=declared T/V=seen ids new=valid row unseen z=zero top=maxfreq E=effective_ids sd=smoothed-rate-SD r=count>=100 rate range")

base = y.mean()
for name in sorted(tr.X):
    x = np.asarray(tr.X[name])
    xv = np.asarray(va.X[name])
    C = int(FEATURE_CARDINALITIES[name])
    if x.ndim != 1:
        print(f"F {name} shape={x.shape} valid_shape={xv.shape}")
        continue
    size = max(C, int(x.max(initial=0)) + 1, int(xv.max(initial=0)) + 1)
    cnt = np.bincount(x, minlength=size).astype(np.float64)
    pos = np.bincount(x, weights=y, minlength=size)
    occ = cnt > 0
    unseen = cnt[xv] == 0
    q = cnt[occ] / len(x)
    eff = np.exp(-np.sum(q * np.log(q)))
    rate = (pos + 100.0 * base) / (cnt + 100.0)
    sd = np.sqrt(np.sum(cnt[occ] * (rate[occ] - base) ** 2) / len(x))
    dense = cnt >= 100
    if dense.any():
        raw = pos[dense] / cnt[dense]
        rr = f"{raw.min():.3f}-{raw.max():.3f}"
    else:
        rr = "NA"
    print(f"F {name} C={C} T/V={occ.sum()}/{np.unique(xv).size} "
          f"new={unseen.mean():.3f} z={np.mean(x==0):.3f} "
          f"top={q.max():.3f} E={eff:.0f} sd={sd:.4f} r={rr}")