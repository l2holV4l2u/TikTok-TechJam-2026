import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES

tr = load("train")
va = load("valid")
te = load("test")  # inspect inputs only; never access te.y

lines = []
lines.append(
    f"ROWS train={len(tr.user_id)} valid={len(va.user_id)} test={len(te.user_id)} "
    f"X={len(tr.X)} num={len(tr.num)} aux={len(tr.aux)}"
)
shape_ok = all(
    a.ndim == 1 and len(a) == len(tr.user_id)
    for a in list(tr.X.values()) + list(tr.num.values())
)
lines.append(
    f"REPRESENTATION scalar_per_row={shape_ok} "
    f"X_dtypes={sorted(set(str(a.dtype) for a in tr.X.values()))} "
    f"num_dtypes={sorted(set(str(a.dtype) for a in tr.num.values()))}"
)

def date_rates(s):
    d, inv = np.unique(s.date, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    return " ".join(f"{int(x)%10000:04d}:{q/r:.3f}" for x, q, r in zip(d, p, n))

lines.append(f"LABEL rate train={tr.y.mean():.4f} valid={va.y.mean():.4f}")
lines.append("DATE train " + date_rates(tr))
lines.append("DATE valid " + date_rates(va))

def user_summary(s, tag):
    u, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y).astype(np.int64)
    qn = np.quantile(n, [0, .25, .5, .75, .9, .99, 1])
    qp = np.quantile(p, [0, .25, .5, .75, .9, .99, 1])
    zero = np.mean(p == 0)
    allp = np.mean(p == n)
    mixed = (p > 0) & (p < n)
    eligible_pos = p[mixed].sum() / max(1, p.sum())
    lines.append(
        f"USER {tag} n={len(u)} rows_q={','.join(f'{x:.0f}' for x in qn)} "
        f"pos_q={','.join(f'{x:.0f}' for x in qp)}"
    )
    lines.append(
        f"USER {tag} zero={zero:.3f} all={allp:.3f} mixed={mixed.mean():.3f} "
        f"GAUC_pos_eligible={eligible_pos:.3f}"
    )
    return u

tu = user_summary(tr, "train")
vu = user_summary(va, "valid")

def membership(x, sorted_values):
    j = np.searchsorted(sorted_values, x)
    return (j < len(sorted_values)) & (sorted_values[np.minimum(j, len(sorted_values)-1)] == x)

tv = np.unique(tr.video_id)
vv = np.unique(va.video_id)
seen_u = membership(va.user_id, tu)
seen_v = membership(va.video_id, tv)
lines.append(
    f"OVERLAP valid_users_unique_seen={membership(vu,tu).mean():.3f} "
    f"rows_seen={seen_u.mean():.3f}; videos_unique_seen={membership(vv,tv).mean():.3f} "
    f"rows_seen={seen_v.mean():.3f}"
)
lines.append(
    f"COLD rows both_seen={(seen_u&seen_v).mean():.3f} "
    f"user_only={(seen_u&~seen_v).mean():.3f} video_only={(~seen_u&seen_v).mean():.3f} "
    f"neither={(~seen_u&~seen_v).mean():.3f}"
)

def adjusted_eta(counts, sums):
    mask = counts > 0
    n = counts[mask].astype(np.float64)
    sy = sums[mask].astype(np.float64)
    N = n.sum()
    p = sy.sum() / N
    var_y = p * (1.0 - p)
    if var_y <= 0 or mask.sum() <= 1:
        return 0.0
    between = np.sum(sy * sy / n) / N - p * p
    null_bias = var_y * (mask.sum() - 1) / N
    return float(np.sqrt(max(0.0, between - null_bias) / var_y))

cat_records = []
for name in sorted(tr.X):
    card = int(FEATURE_CARDINALITIES[name])
    xt, xv, xe = tr.X[name], va.X[name], te.X[name]
    ct = np.bincount(xt, minlength=card)
    st = np.bincount(xt, weights=tr.y, minlength=card)
    cv = np.bincount(xv, minlength=card)
    sv = np.bincount(xv, weights=va.y, minlength=card)
    ce = np.bincount(xe, minlength=card)
    unseen_v = np.mean(ct[xv] == 0)
    et = adjusted_eta(ct, st)
    ev = adjusted_eta(cv, sv)
    cat_records.append(
        f"{name}={card}/{np.count_nonzero(ct)}/{np.count_nonzero(cv)}/"
        f"{np.count_nonzero(ce)},uv{unseen_v:.3f},e{et:.3f}/{ev:.3f}"
    )

lines.append("CAT format=name card/usedTrain/usedValid/usedTest,validUnseenRow,adjEtaTrain/Valid")
for i in range(0, len(cat_records), 2):
    lines.append("CAT " + " | ".join(cat_records[i:i+2]))

def numeric_stats(x, y):
    ok = np.isfinite(x)
    miss = 1.0 - ok.mean()
    if ok.sum() < 2:
        return miss, 0.0, 0.0
    z = np.log1p(np.maximum(x[ok].astype(np.float64), 0.0))
    yy = y[ok].astype(bool)
    sd = z.std()
    if sd == 0 or yy.all() or (~yy).all():
        corr = 0.0
    else:
        p = yy.mean()
        corr = (z[yy].mean() - z[~yy].mean()) * np.sqrt(p * (1-p)) / sd
    return miss, sd, corr

num_records = []
for name in sorted(tr.num):
    mt, sdt, rt = numeric_stats(tr.num[name], tr.y)
    mv, sdv, rv = numeric_stats(va.num[name], va.y)
    num_records.append(
        f"{name}=m{mt:.2f}/{mv:.2f},sd{sdt:.2f},r{rt:+.3f}/{rv:+.3f}"
    )

lines.append("NUM format=name missingTrain/Valid,logSDTrain,pointBiserialTrain/Valid")
for i in range(0, len(num_records), 2):
    lines.append("NUM " + " | ".join(num_records[i:i+2]))

for line in lines:
    print(line)