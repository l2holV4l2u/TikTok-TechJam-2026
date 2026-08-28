import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES

tr = load("train")
va = load("valid")

print(f"ROWS train={len(tr.y)} valid={len(va.y)} Xfields={len(tr.X)} num={len(tr.num)}")
print(f"SHAPES y={tr.y.shape}/{tr.y.dtype} user={tr.user_id.shape} video={tr.video_id.shape}")
scalar_ok = all(np.asarray(v).ndim == 1 and len(v) == len(tr.y) for v in tr.X.values())
print(f"X_SCALAR_PER_ROW={scalar_ok} aux_keys={','.join(sorted(tr.aux.keys()))}")

def split_summary(tag, s):
    users, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    qn = np.quantile(n, [0, .25, .5, .75, .9, .99, 1])
    qp = np.quantile(p, [0, .25, .5, .75, .9, .99, 1])
    zero = np.mean(p == 0)
    allp = np.mean(p == n)
    elig = np.mean((p > 0) & (p < n))
    print(f"{tag} pos={s.y.mean():.4f} users={len(users)} videos={len(np.unique(s.video_id))}")
    print(f"{tag} user_n q0/25/50/75/90/99/max=" +
          "/".join(f"{x:.0f}" for x in qn))
    print(f"{tag} user_pos q0/25/50/75/90/99/max=" +
          "/".join(f"{x:.0f}" for x in qp))
    print(f"{tag} users zero={zero:.3f} all={allp:.3f} auc_eligible={elig:.3f}")

split_summary("TR", tr)
split_summary("VA", va)

tu = np.unique(tr.user_id)
tv = np.unique(tr.video_id)
vu = np.unique(va.user_id)
vv = np.unique(va.video_id)
print(f"COLD valid_rows user={np.mean(~np.isin(va.user_id,tu)):.4f} "
      f"video={np.mean(~np.isin(va.video_id,tv)):.4f} "
      f"either={np.mean((~np.isin(va.user_id,tu)) | (~np.isin(va.video_id,tv))):.4f}")
print(f"COLD valid_unique user={np.mean(~np.isin(vu,tu)):.4f} video={np.mean(~np.isin(vv,tv)):.4f}")

def daily(tag, s):
    vals = []
    for d in np.unique(s.date):
        z = s.y[s.date == d]
        vals.append(f"{d}:{len(z)}/{z.mean():.3f}")
    print(tag + "_DAY rows/pos " + " ".join(vals))

daily("TR", tr)
daily("VA", va)

order = np.lexsort((np.arange(len(tr.y)), tr.time_ms, tr.user_id))
same_u = tr.user_id[order][1:] == tr.user_id[order][:-1]
same_t = tr.time_ms[order][1:] == tr.time_ms[order][:-1]
print(f"TIME same-user adjacent timestamp ties={np.mean(same_u & same_t):.4f}")

py = float(tr.y.mean())
hy = -(py*np.log(py) + (1-py)*np.log(1-py))
print("CAT name card seenTR seenVA zeroVA novelVA topTR MI/H")
for name in sorted(tr.X):
    a = np.asarray(tr.X[name])
    b = np.asarray(va.X[name])
    ids, inv = np.unique(a, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    pos = np.bincount(inv, weights=tr.y).astype(np.float64)
    neg = cnt - pos
    n = cnt.sum()
    pc = cnt / n
    mi = 0.0
    good = pos > 0
    mi += np.sum((pos[good]/n) * np.log((pos[good]/n)/(pc[good]*py)))
    good = neg > 0
    mi += np.sum((neg[good]/n) * np.log((neg[good]/n)/(pc[good]*(1-py))))
    novel = (b != 0) & (~np.isin(b, ids))
    print(f"C {name} {FEATURE_CARDINALITIES[name]} {len(ids)} {len(np.unique(b))} "
          f"{np.mean(b==0):.3f} {np.mean(novel):.3f} {cnt.max()/n:.3f} {mi/hy:.4f}")

num_stats = []
for name in sorted(tr.num):
    x = np.asarray(tr.num[name], dtype=np.float64)
    finite = np.isfinite(x)
    miss = 1.0 - finite.mean()
    if finite.sum() > 10 and np.std(x[finite]) > 0:
        z = np.sign(x[finite]) * np.log1p(np.abs(x[finite]))
        corr = np.corrcoef(z, tr.y[finite])[0, 1]
    else:
        corr = 0.0
    miss_gap = (tr.y[~finite].mean() - tr.y[finite].mean()
                if finite.any() and (~finite).any() else 0.0)
    num_stats.append((abs(corr), name, miss, corr, miss_gap))

num_stats.sort(reverse=True)
for chunk in (num_stats[:11], num_stats[11:]):
    print("NUM " + " ".join(
        f"{name}:m{miss:.2f},r{corr:+.3f},g{gap:+.3f}"
        for _, name, miss, corr, gap in chunk
    ))