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
    print(f"USER {name} U={len(u)} rows_q={qn.tolist()} pos_q={np.round(qp,1).tolist()} "
          f"zero={np.mean(p==0):.3f} mixed={np.mean((p>0)&(p<n)):.3f}")

def daily(s, name):
    vals = []
    for d in np.unique(s.date):
        z = s.y[s.date == d]
        vals.append(f"{d%10000}:{len(z)}/{z.mean():.3f}")
    print(f"DAY {name} " + ",".join(vals))

def overlap(name):
    a = tr.X[name]
    b = va.X[name]
    seen = np.zeros(max(FEATURE_CARDINALITIES[name], int(a.max()) + 1), dtype=bool)
    seen[np.unique(a)] = True
    row_unseen = np.mean(~seen[b])
    ub = np.unique(b)
    uniq_unseen = np.mean(~seen[ub])
    print(f"COLD {name} row_unseen={row_unseen:.3f} valid_ids_unseen={uniq_unseen:.3f}")

def shrunk_eta(x, y, alpha=20.0):
    n = np.bincount(x)
    sy = np.bincount(x, weights=y, minlength=len(n))
    mu = float(np.mean(y))
    m = (sy + alpha * mu) / (n + alpha)
    den = len(y) * mu * (1.0 - mu)
    return np.sqrt(np.sum(n * (m - mu) ** 2) / den) if den > 0 else 0.0

print(f"SCHEMA X={len(tr.X)} num={list(tr.num)} X_shapes={sorted(set(v.shape for v in tr.X.values()))}")
print(f"SPLIT train n={len(tr.user_id)} pos={tr.y.mean():.4f} dates={tr.date.min()}-{tr.date.max()}")
print(f"SPLIT valid n={len(va.user_id)} pos={va.y.mean():.4f} dates={va.date.min()}-{va.date.max()}")
user_summary(tr, "train")
user_summary(va, "valid")
daily(tr, "train")
daily(va, "valid")
for k in ["user_id", "video_id", "author_id"]:
    overlap(k)

for k in tr.num:
    x = tr.num[k].astype(np.float64)
    ok = np.isfinite(x)
    q = np.quantile(x[ok], [.01, .5, .99]) if np.any(ok) else [np.nan] * 3
    m0 = np.nanmean(x[tr.y == 0])
    m1 = np.nanmean(x[tr.y == 1])
    print(f"NUM {k} miss={1-ok.mean():.3f} q01/50/99={q[0]:.3g}/{q[1]:.3g}/{q[2]:.3g} "
          f"mean0/1={m0:.3g}/{m1:.3g}")

print("CAT name card usedT usedV unseenRow maxShare eta20T eta20V")
for k in sorted(tr.X):
    a, b = tr.X[k], va.X[k]
    card = FEATURE_CARDINALITIES[k]
    cnt = np.bincount(a, minlength=card)
    unseen = np.mean(cnt[b] == 0)
    print(f"CAT {k} {card} {np.count_nonzero(cnt)} {len(np.unique(b))} "
          f"{unseen:.3f} {cnt.max()/len(a):.3f} "
          f"{shrunk_eta(a,tr.y):.3f} {shrunk_eta(b,va.y):.3f}")

for key in ["video_id", "author_id"]:
    htr = historical_features("train", key=key)
    hva = historical_features("valid", key=key)
    shapes = sorted(set(tuple(np.asarray(v).shape) for v in htr.values()))
    miss = [f"{k}:{np.mean(~np.isfinite(v)):.2f}" for k, v in hva.items()]
    print(f"HIST {key} keys={','.join(htr.keys())} train_shapes={shapes}")
    print(f"HISTMISS {key} " + ",".join(miss))

print(f"AUX keys={','.join(sorted(tr.aux.keys()))} shapes={sorted(set(v.shape for v in tr.aux.values()))}")