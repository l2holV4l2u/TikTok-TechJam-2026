import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES

tr = load("train")
va = load("valid")
te = load("test")  # Feature inspection only; never access te.y.

def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a -= a.mean()
    b -= b.mean()
    d = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / d) if d > 0 else 0.0

def user_stats(s):
    _, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    qn = np.quantile(n, [0, .25, .5, .75, .9, .99, 1])
    qp = np.quantile(p, [0, .25, .5, .75, .9, .99, 1])
    return len(n), qn, qp, np.mean(p == 0), np.mean((p > 0) & (p < n))

def fmtq(x):
    return "/".join(f"{v:.0f}" for v in x)

def overlap_rows(train_ids, other_ids):
    seen = np.unique(train_ids)
    return float(np.mean(np.isin(other_ids, seen, assume_unique=False)))

def binary_mi(ids, y):
    n = len(y)
    cnt = np.bincount(ids)
    pos = np.bincount(ids, weights=y, minlength=len(cnt))
    py = float(np.mean(y))
    h = 0.0
    if 0 < py < 1:
        h = -py*np.log2(py) - (1-py)*np.log2(1-py)
    good = cnt > 0
    q = np.zeros_like(pos, dtype=np.float64)
    q[good] = pos[good] / cnt[good]
    hc = np.zeros_like(q)
    mid = good & (q > 0) & (q < 1)
    hc[mid] = -q[mid]*np.log2(q[mid]) - (1-q[mid])*np.log2(1-q[mid])
    return float(h - np.sum(cnt * hc) / n)

def date_line(s):
    d, inv = np.unique(s.date, return_inverse=True)
    rates = np.bincount(inv, weights=s.y) / np.bincount(inv)
    return " ".join(f"{int(x)}:{r:.3f}" for x, r in zip(d, rates))

print(f"ROWS train={len(tr.y)} valid={len(va.y)} test={len(te.user_id)}")
print(f"LABEL train={tr.y.mean():.4f} valid={va.y.mean():.4f}")
print("DATE_T " + date_line(tr))
print("DATE_V " + date_line(va))

for tag, s in [("T", tr), ("V", va)]:
    nu, qn, qp, zero, eligible = user_stats(s)
    print(f"USER_{tag} n={nu} rowsQ={fmtq(qn)} posQ={fmtq(qp)} "
          f"zero={zero:.3f} eligible={eligible:.3f}")

print(f"OVERLAP valid userRows={overlap_rows(tr.user_id, va.user_id):.3f} "
      f"videoRows={overlap_rows(tr.video_id, va.video_id):.3f} "
      f"test userRows={overlap_rows(tr.user_id, te.user_id):.3f} "
      f"videoRows={overlap_rows(tr.video_id, te.video_id):.3f}")

order = np.lexsort((np.arange(len(tr.y)), tr.time_ms, tr.user_id))
ou = tr.user_id[order]
ot = tr.time_ms[order]
same_user = ou[1:] == ou[:-1]
gaps = ot[1:][same_user] - ot[:-1][same_user]
print(f"ORDER priorRowFrac={same_user.sum()/len(tr.y):.3f} "
      f"sameTimestampFrac={np.mean(gaps == 0):.3f} "
      f"gapMsQ50/90/99={fmtq(np.quantile(gaps, [.5,.9,.99]))}")

cat_names = sorted(tr.X)
num_names = sorted(tr.num)
scalar_ok = all(np.asarray(tr.X[k]).shape == (len(tr.y),) for k in cat_names)
print(f"SCHEMA cats={len(cat_names)} nums={len(num_names)} scalarCat={scalar_ok}")
print("CAT key=name:K,activeT/V/E,newRowV/E,MItrain,TEcorrV")

cat_tokens = []
prior = float(tr.y.mean())
for name in cat_names:
    xt, xv, xe = tr.X[name], va.X[name], te.X[name]
    ut = np.unique(xt)
    av, ae = len(np.unique(xv)), len(np.unique(xe))
    max_id = int(max(xt.max(), xv.max(), xe.max()))
    seen = np.zeros(max_id + 1, dtype=bool)
    seen[ut] = True
    new_v = np.mean(~seen[xv])
    new_e = np.mean(~seen[xe])

    cnt = np.bincount(xt, minlength=max_id + 1).astype(np.float64)
    pos = np.bincount(xt, weights=tr.y, minlength=max_id + 1)
    rate = (pos + 20.0 * prior) / (cnt + 20.0)
    pred = rate[xv]
    token = (f"{name}:{FEATURE_CARDINALITIES[name]},"
             f"{len(ut)}/{av}/{ae},{new_v:.2f}/{new_e:.2f},"
             f"{binary_mi(xt, tr.y):.3g},{corr(pred, va.y):+.3f}")
    cat_tokens.append(token)

for i in range(0, len(cat_tokens), 3):
    print("C " + " ".join(cat_tokens[i:i+3]))

print("NUM key=name:missingT/V/E,corrLogT/V,corrMissingV")
num_tokens = []
for name in num_names:
    at = np.asarray(tr.num[name], dtype=np.float64)
    av = np.asarray(va.num[name], dtype=np.float64)
    ae = np.asarray(te.num[name], dtype=np.float64)
    mt, mv, me = ~np.isfinite(at), ~np.isfinite(av), ~np.isfinite(ae)
    finite = at[~mt]
    med = float(np.median(finite)) if len(finite) else 0.0
    zt = np.log1p(np.maximum(np.where(mt, med, at), 0.0))
    zv = np.log1p(np.maximum(np.where(mv, med, av), 0.0))
    token = (f"{name}:{mt.mean():.2f}/{mv.mean():.2f}/{me.mean():.2f},"
             f"{corr(zt,tr.y):+.3f}/{corr(zv,va.y):+.3f},"
             f"{corr(mv.astype(float),va.y):+.3f}")
    num_tokens.append((max(abs(corr(zt, tr.y)), abs(corr(zv, va.y))), token))

num_tokens.sort(reverse=True)
for i in range(0, len(num_tokens), 3):
    print("N " + " ".join(x[1] for x in num_tokens[i:i+3]))