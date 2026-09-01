import numpy as np
from scipy.stats import rankdata
from pipeline.data import load, FEATURE_CARDINALITIES

tr = load("train")
va = load("valid")
te = load("test")  # inspect covariates only; never access test labels

def qstr(x):
    q = np.quantile(x, [0, .25, .5, .75, .9, .99, 1])
    return "/".join(f"{v:.0f}" for v in q)

def user_summary(s, name):
    _, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    print(f"{name} users={len(n)} y={s.y.mean():.4f} rowsQ={qstr(n)} posQ={qstr(p)} "
          f"zero={(p==0).mean():.3f} all={(p==n).mean():.3f} mixed={((p>0)&(p<n)).mean():.3f}")

def overlap(train_ids, other_ids):
    seen = np.unique(train_ids)
    return 1.0 - np.isin(other_ids, seen).mean(), 1.0 - np.isin(np.unique(other_ids), seen).mean()

def auc_binary(y, score):
    r = rankdata(score, method="average")
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

print(f"ROWS train={len(tr.y)} valid={len(va.y)} test={len(te.user_id)} fields={len(tr.X)}")
print(f"X scalar_per_row={all(np.asarray(x).ndim==1 and len(x)==len(tr.y) for x in tr.X.values())} "
      f"keys_match={set(tr.X)==set(va.X)==set(te.X)}")
user_summary(tr, "TRAIN")
user_summary(va, "VALID")

for key, a, b in [
    ("user", tr.user_id, va.user_id),
    ("video", tr.video_id, va.video_id),
]:
    rr, uu = overlap(a, b)
    print(f"VALID_COLD {key} row={rr:.4f} distinct={uu:.4f} trainU={len(np.unique(a))} validU={len(np.unique(b))}")
for key, a, b in [
    ("user", tr.user_id, te.user_id),
    ("video", tr.video_id, te.video_id),
]:
    rr, uu = overlap(a, b)
    print(f"TEST_COLD {key} row={rr:.4f} distinct={uu:.4f} testU={len(np.unique(b))}")

aux_desc = ",".join(f"{k}:{np.asarray(v).dtype}{tuple(np.asarray(v).shape)}"
                    for k, v in tr.aux.items())
print("AUX " + aux_desc[:500])

base = float(tr.y.mean())
eps = 1e-7
vbase = -(va.y * np.log(base) + (1 - va.y) * np.log(1 - base)).mean()
alpha = 20.0

for name in tr.X:
    xt = np.asarray(tr.X[name], dtype=np.int64)
    xv = np.asarray(va.X[name], dtype=np.int64)
    xe = np.asarray(te.X[name], dtype=np.int64)
    K = int(FEATURE_CARDINALITIES[name])

    cnt = np.bincount(xt, minlength=K)
    pos = np.bincount(xt, weights=tr.y, minlength=K)
    rate = (pos + alpha * base) / (cnt + alpha)
    pv = rate[xv]
    pv = np.clip(pv, eps, 1 - eps)

    auc = auc_binary(va.y, pv)
    ll = -(va.y * np.log(pv) + (1 - va.y) * np.log(1 - pv)).mean()
    lift = (vbase - ll) / vbase
    seen = cnt > 0
    dom = cnt.max() / len(xt)
    uv = (~seen[xv]).mean()
    ue = (~seen[xe]).mean()
    print(f"F {name} K={K} U={np.unique(xt).size}/{np.unique(xv).size}/{np.unique(xe).size} "
          f"d={dom:.3f} uv={uv:.3f} ut={ue:.3f} A={auc:.3f} L={lift:.3f}")