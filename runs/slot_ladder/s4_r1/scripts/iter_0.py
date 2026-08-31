import numpy as np
from scipy.stats import rankdata
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")

def auc(y, score):
    y = np.asarray(y, dtype=np.int8)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score, method="average")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def qstr(x):
    q = np.quantile(x, [0, .25, .5, .75, .9, .99, 1])
    return "/".join(f"{v:.0f}" for v in q)

def user_summary(name, s):
    _, inv = np.unique(s.user_id, return_inverse=True)
    nr = np.bincount(inv)
    np_ = np.bincount(inv, weights=s.y)
    zero = np.mean(np_ == 0)
    mixed = np.mean((np_ > 0) & (np_ < nr))
    print(f"USER {name} n={len(nr)} y={s.y.mean():.4f} rowsQ={qstr(nr)} "
          f"posQ={qstr(np_)} zero={zero:.3f} mixed={mixed:.3f}")

print(f"SHAPE train={len(tr.y)} valid={len(va.y)} X={len(tr.X)} num={len(tr.num)} "
      f"x_scalar={all(np.asarray(v).shape == (len(tr.y),) for v in tr.X.values())}")
print(f"AUX outcome-only n={len(tr.aux)} keys={','.join(sorted(tr.aux)[:12])}")
user_summary("train", tr)
user_summary("valid", va)

for name, s in [("train", tr), ("valid", va)]:
    ds = []
    for d in np.unique(s.date):
        m = s.date == d
        ds.append(f"{d % 10000}:{m.sum()}:{s.y[m].mean():.3f}")
    print(f"DATE {name} day:rows:rate " + ",".join(ds))

overlap = []
for f in ["user_id", "video_id", "author_id"]:
    a, b = tr.X[f], va.X[f]
    seen = np.unique(a[a != 0])
    unseen = (b == 0) | ~np.isin(b, seen, assume_unique=False)
    uv = np.unique(b[b != 0])
    unseen_u = ~np.isin(uv, seen, assume_unique=False)
    overlap.append(f"{f}:rowU={unseen.mean():.3f},idU={unseen_u.mean():.3f}")
print("COLD " + " ".join(overlap))

print("NUM name miss trainQ50/Q90/Q99 logCorrY validMiss")
for name in sorted(tr.num):
    x = np.asarray(tr.num[name], dtype=np.float64)
    z = np.asarray(va.num[name], dtype=np.float64)
    ok = np.isfinite(x)
    vals = x[ok]
    qs = np.quantile(vals, [.5, .9, .99]) if len(vals) else [np.nan] * 3
    lx = np.log1p(np.maximum(vals, 0))
    corr = np.corrcoef(lx, tr.y[ok])[0, 1] if np.std(lx) > 0 else 0.0
    print(f"NUM {name} {1-ok.mean():.3f} "
          f"{qs[0]:.1f}/{qs[1]:.1f}/{qs[2]:.1f} {corr:+.3f} "
          f"{1-np.isfinite(z).mean():.3f}")

dates = np.unique(tr.date)
hold_dates = dates[-3:]
early = ~np.isin(tr.date, hold_dates)
hold = ~early
prior = float(tr.y[early].mean())
cat_lines = []

for name in tr.X:
    x = np.asarray(tr.X[name])
    xv = np.asarray(va.X[name])
    mx = int(max(x.max(initial=0), xv.max(initial=0)))
    full = np.bincount(x, minlength=mx + 1)
    cnt = np.bincount(x[early], minlength=mx + 1)
    pos = np.bincount(x[early], weights=tr.y[early], minlength=mx + 1)
    rate = (pos + 20.0 * prior) / (cnt + 20.0)
    hauc = auc(tr.y[hold], rate[x[hold]])
    unseen = (xv == 0) | (full[xv] == 0)
    appeared = int(np.count_nonzero(full))
    top = float(full.max() / len(x))
    card = int(FEATURE_CARDINALITIES[name])
    cat_lines.append((hauc, name, card, appeared, top, unseen.mean()))

print("CAT sorted temporal-holdout target-rate AUC; card/seen topShare validRowUnseen")
for hauc, name, card, appeared, top, unseen in sorted(cat_lines, reverse=True):
    print(f"CAT {name} auc={hauc:.3f} card={card}/{appeared} "
          f"top={top:.3f} vunseen={unseen:.3f}")

hist = historical_features("train", key="video_id")
print(f"HIST video keys={len(hist)} names={','.join(sorted(hist))}")
for name in sorted(hist)[:5]:
    x = np.asarray(hist[name])
    ok = np.isfinite(x)
    q = np.quantile(x[ok], [.5, .9, .99]) if ok.any() else [np.nan] * 3
    print(f"HIST {name} shape={x.shape} miss={1-ok.mean():.3f} "
          f"q={q[0]:.3g}/{q[1]:.3g}/{q[2]:.3g}")