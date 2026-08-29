import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES

tr = load("train")
va = load("valid")

print(f"ROWS train={len(tr.y)} valid={len(va.y)} features={len(tr.X)}")
print(f"LABEL train_rate={tr.y.mean():.5f} train_pos={tr.y.sum()} "
      f"valid_rate={va.y.mean():.5f} valid_pos={va.y.sum()}")

def date_summary(s):
    d, inv, n = np.unique(s.date, return_inverse=True, return_counts=True)
    p = np.bincount(inv, weights=s.y)
    return ",".join(f"{int(x)}:{int(c)}/{q/c:.3f}" for x, c, q in zip(d, n, p))

print("DATES_TR " + date_summary(tr))
print("DATES_VA " + date_summary(va))

def user_summary(s, tag, keep_inverse=False):
    uid, inv, cnt = np.unique(s.user_id, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=s.y).astype(np.int64)
    q = [0, 10, 25, 50, 75, 90, 99, 100]
    cq = np.percentile(cnt, q)
    pq = np.percentile(pos, q)
    mixed = np.mean((pos > 0) & (pos < cnt))
    zero = np.mean(pos == 0)
    print(f"USER_{tag} n={len(uid)} zero_pos={zero:.4f} mixed={mixed:.4f} "
          f"rows_mean={cnt.mean():.1f} pos_mean={pos.mean():.1f}")
    print(f"USER_{tag}_Q q={q} rows={np.round(cq,1).tolist()} "
          f"pos={np.round(pq,1).tolist()}")
    return (uid, inv, cnt, pos) if keep_inverse else uid

utr = user_summary(tr, "TR")
uva, vinv, vcnt, vpos = user_summary(va, "VA", True)

def overlap(train_ids, valid_ids):
    a = np.unique(train_ids)
    idx = np.searchsorted(a, valid_ids)
    seen = (idx < len(a))
    seen[seen] &= a[idx[seen]] == valid_ids[seen]
    return seen.mean(), len(np.unique(valid_ids)), len(a)

ur, uvn, utn = overlap(tr.user_id, va.user_id)
vr, vvn, vtn = overlap(tr.video_id, va.video_id)
print(f"OVERLAP user_seen_rows={ur:.4f} train_unique={utn} valid_unique={uvn} "
      f"video_seen_rows={vr:.4f} train_unique={vtn} valid_unique={vvn}")

xdims = sorted(set((x.ndim, x.shape[0], str(x.dtype)) for x in tr.X.values()))
print(f"X_LAYOUT signatures={xdims} keys_match_cardinalities="
      f"{set(tr.X)==set(FEATURE_CARDINALITIES)}")
aux_desc = ",".join(f"{k}:{v.shape}/{v.dtype}" for k, v in tr.aux.items())
print("AUX " + aux_desc)

# User-centered validation correlation estimates within-user ranking usefulness.
vy = va.y.astype(np.float64)
user_ymean = vpos[vinv] / vcnt[vinv]
yc = vy - user_ymean
ycss = np.dot(yc, yc)
base = float(tr.y.mean())
eps = 1e-7
base_ll = -np.mean(vy * np.log(np.clip(va.y.mean(), eps, 1-eps)) +
                   (1-vy) * np.log(np.clip(1-va.y.mean(), eps, 1-eps)))

for name in sorted(tr.X):
    xt = tr.X[name]
    xv = va.X[name]
    card = FEATURE_CARDINALITIES[name]
    size = max(card, int(xt.max()) + 1, int(xv.max()) + 1)
    cnt = np.bincount(xt, minlength=size)
    pos = np.bincount(xt, weights=tr.y, minlength=size)
    rates = (pos + 20.0 * base) / (cnt + 20.0)
    pred = rates[xv]
    unseen = np.mean(cnt[xv] == 0)
    ntr = int(np.count_nonzero(cnt))
    nva = int(np.count_nonzero(np.bincount(xv, minlength=size)))
    top = cnt.max() / len(xt)
    ll = -np.mean(vy*np.log(np.clip(pred, eps, 1-eps)) +
                  (1-vy)*np.log(np.clip(1-pred, eps, 1-eps)))
    psum = np.bincount(vinv, weights=pred, minlength=len(vcnt))
    pc = pred - (psum / vcnt)[vinv]
    pcss = np.dot(pc, pc)
    wcorr = np.dot(yc, pc) / np.sqrt(ycss * pcss) if pcss > 0 else 0.0
    print(f"F {name} C={card} T={ntr} V={nva} U={unseen:.3f} "
          f"TOP={top:.3f} DLL={base_ll-ll:.4f} WC={wcorr:.3f}")