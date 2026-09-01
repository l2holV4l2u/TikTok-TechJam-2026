import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES

tr = load("train")
va = load("valid")
te = load("test")  # Features only; never access test labels.

print(f"ROWS train={len(tr.y)} valid={len(va.y)} test={len(te.user_id)}")
print(f"LABEL train_rate={tr.y.mean():.5f} train_pos={int(tr.y.sum())} "
      f"valid_rate={va.y.mean():.5f} valid_pos={int(va.y.sum())}")

def user_stats(tag, uid, y):
    users, inv, rows = np.unique(uid, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=y, minlength=len(users))
    rq = np.quantile(rows, [0, .25, .5, .75, .9, .99, 1])
    pq = np.quantile(pos, [0, .25, .5, .75, .9, .99, 1])
    zero = np.mean(pos == 0)
    mixed = np.mean((pos > 0) & (pos < rows))
    allpos = np.mean(pos == rows)
    print(f"USER {tag} n={len(users)} zero={zero:.3f} mixed={mixed:.3f} "
          f"allpos={allpos:.3f}")
    print(f"UROWS {tag} q0/25/50/75/90/99/max=" +
          "/".join(f"{x:.0f}" for x in rq))
    print(f"UPOS {tag} q0/25/50/75/90/99/max=" +
          "/".join(f"{x:.0f}" for x in pq))

user_stats("train", tr.user_id, tr.y)
user_stats("valid", va.user_id, va.y)

def novelty(name, train_ids, split_ids, tag):
    utrain = np.unique(train_ids)
    usplit = np.unique(split_ids)
    row_new = 1.0 - np.isin(split_ids, utrain, assume_unique=False).mean()
    ent_new = 1.0 - np.isin(usplit, utrain, assume_unique=True).mean()
    print(f"NEW {name} {tag} row={row_new:.4f} entity={ent_new:.4f} "
          f"uniq_train={len(utrain)} uniq_split={len(usplit)}")

novelty("user", tr.user_id, va.user_id, "valid")
novelty("video", tr.video_id, va.video_id, "valid")
novelty("user", tr.user_id, te.user_id, "test")
novelty("video", tr.video_id, te.video_id, "test")

fields = list(tr.X)
bad_shapes = []
for name in fields:
    for tag, s, n in (("tr", tr, len(tr.y)), ("va", va, len(va.y)),
                      ("te", te, len(te.user_id))):
        a = np.asarray(s.X[name])
        if a.ndim != 1 or len(a) != n:
            bad_shapes.append(f"{name}:{tag}{a.shape}")
print(f"X fields={len(fields)} all_row_scalar={not bad_shapes} "
      f"exceptions={','.join(bad_shapes) if bad_shapes else 'none'}")
print("AUX outcome_only keys=" + ",".join(sorted(tr.aux.keys())))
aux_shapes = sorted({str(np.asarray(v).shape) for v in tr.aux.values()})
print("AUX train_shapes=" + ",".join(aux_shapes))
print("FIELD name C Utr/Uva/Ute seenV/seenT z dom MIraw/MIadj_bits")

p1 = float(tr.y.mean())
n = len(tr.y)
ln2 = np.log(2.0)

for name in fields:
    x = np.asarray(tr.X[name], dtype=np.int64)
    xv = np.asarray(va.X[name], dtype=np.int64)
    xt = np.asarray(te.X[name], dtype=np.int64)
    C = int(FEATURE_CARDINALITIES[name])
    size = max(C, int(x.max(initial=0)) + 1,
               int(xv.max(initial=0)) + 1, int(xt.max(initial=0)) + 1)

    cnt = np.bincount(x, minlength=size).astype(np.float64)
    pos = np.bincount(x, weights=tr.y, minlength=size)
    nz = cnt > 0
    c = cnt[nz]
    r = pos[nz] / c
    q = c / n

    t1 = np.zeros_like(r)
    t0 = np.zeros_like(r)
    m1 = r > 0
    m0 = r < 1
    if 0 < p1 < 1:
        t1[m1] = r[m1] * np.log2(r[m1] / p1)
        t0[m0] = (1-r[m0]) * np.log2((1-r[m0]) / (1-p1))
    mi = float(np.sum(q * (t1 + t0)))
    bias = (int(nz.sum()) - 1) / (2.0 * n * ln2)
    mia = max(0.0, mi - bias)

    seen_v = float(np.mean(cnt[xv] > 0))
    seen_t = float(np.mean(cnt[xt] > 0))
    uv = len(np.unique(xv))
    ut = len(np.unique(xt))
    zero = float(np.mean(x == 0))
    dom = float(cnt.max() / n)
    print(f"F {name} {C} {int(nz.sum())}/{uv}/{ut} "
          f"{seen_v:.3f}/{seen_t:.3f} {zero:.3f} {dom:.3f} "
          f"{mi:.5f}/{mia:.5f}")