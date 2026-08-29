import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")

lines = []

def qstr(a, qs=(0, .5, .9, .99)):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return "NA"
    return "/".join(f"{x:.3g}" for x in np.quantile(a, qs))

def user_stats(s, name):
    u, inv = np.unique(s.user_id, return_inverse=True)
    nr = np.bincount(inv)
    np_ = np.bincount(inv, weights=s.y)
    lines.append(
        f"USR {name} n={len(u)} rowsQ={qstr(nr,(0,.5,.9,.99))} "
        f"posQ={qstr(np_,(0,.5,.9,.99))} zero={(np_==0).mean():.3f} "
        f"all={(np_==nr).mean():.3f} mixed={((np_>0)&(np_<nr)).mean():.3f}"
    )

lines.append(
    f"SHAPE tr={len(tr.user_id)} va={len(va.user_id)} te={len(te.user_id)} "
    f"X={len(tr.X)} num={len(tr.num)}"
)
scalar_ok = all(np.asarray(v).ndim == 1 and len(v) == len(tr.user_id)
                for v in tr.X.values())
lines.append(f"X_LAYOUT scalar_per_row={scalar_ok} example={next(iter(tr.X.values())).shape}")
lines.append(
    f"LABEL tr={tr.y.mean():.4f}({int(tr.y.sum())}) "
    f"va={va.y.mean():.4f}({int(va.y.sum())})"
)
user_stats(tr, "tr")
user_stats(va, "va")

def novelty(key):
    a = np.asarray(getattr(tr, key))
    b = np.asarray(getattr(va, key))
    c = np.asarray(getattr(te, key))
    au = np.unique(a)
    return (
        len(au), len(np.unique(b)), len(np.unique(c)),
        100 * np.mean(~np.isin(b, au)),
        100 * np.mean(~np.isin(c, au))
    )

for key in ("user_id", "video_id"):
    x = novelty(key)
    lines.append(f"NEW {key} U={x[0]}/{x[1]}/{x[2]} row%va/te={x[3]:.2f}/{x[4]:.2f}")
x = novelty("author_id")
lines.append(f"NEW author_id U={x[0]}/{x[1]}/{x[2]} row%va/te={x[3]:.2f}/{x[4]:.2f}")

# Date-wise prevalence exposes drift without fitting any predictor.
def daily(s):
    z = []
    for d in np.unique(s.date):
        m = s.date == d
        z.append(f"{str(int(d))[-4:]}:{m.sum()//1000}k/{s.y[m].mean():.3f}")
    return " ".join(z)

lines.append("DAY tr mmdd:rows/rate " + daily(tr))
lines.append("DAY va mmdd:rows/rate " + daily(va))

# Continuous-feature scale and missingness.
for name in tr.num:
    a, b = tr.num[name], va.num[name]
    lines.append(
        f"NUM {name} miss={np.mean(~np.isfinite(a)):.2f}/{np.mean(~np.isfinite(b)):.2f} "
        f"q0/50/90/99={qstr(a)}"
    )

# Bias-corrected correlation ratio. Subtracts the null category-count effect,
# avoiding the misleading apparent signal of singleton-heavy ID fields.
def adj_eta(x, y, cardinality):
    size = max(int(cardinality), int(np.max(x)) + 1)
    cnt = np.bincount(x, minlength=size).astype(np.float64)
    sy = np.bincount(x, weights=y, minlength=size)
    nz = cnt > 0
    p = float(np.mean(y))
    var = p * (1.0 - p)
    if var <= 0:
        return 0.0
    ssb = np.sum((sy[nz] - cnt[nz] * p) ** 2 / cnt[nz])
    k = int(nz.sum())
    corrected = max(0.0, ssb - max(0, k - 1) * var)
    denom = max(1e-12, len(y) * var - max(0, k - 1) * var)
    return float(np.sqrt(min(1.0, corrected / denom)))

feat_lines = []
for name in tr.X:
    a, b, c = tr.X[name], va.X[name], te.X[name]
    C = FEATURE_CARDINALITIES[name]
    size = max(int(C), int(a.max()) + 1, int(b.max()) + 1, int(c.max()) + 1)
    seen = np.zeros(size, dtype=bool)
    seen[np.unique(a)] = True
    cnt = np.bincount(a, minlength=size)
    feat_lines.append((
        adj_eta(a, tr.y, C),
        f"CAT {name} C={C} U={np.unique(a).size}/{np.unique(b).size}/{np.unique(c).size} "
        f"new%={100*np.mean(~seen[b]):.1f}/{100*np.mean(~seen[c]):.1f} "
        f"z%={100*np.mean(a==0):.1f}/{100*np.mean(b==0):.1f} "
        f"max%={100*cnt.max()/len(a):.1f} eta={adj_eta(a,tr.y,C):.3f}/{adj_eta(b,va.y,C):.3f}"
    ))

lines.append("CAT legend C=declared U=tr/va/te new%=va/te z%=tr/va max%=train eta=tr/va")
for _, text in sorted(feat_lines, reverse=True):
    lines.append(text)

# Check whether ordered within-user behavior is potentially useful.
order = np.lexsort((np.arange(len(tr.user_id)), tr.time_ms, tr.user_id))
u = tr.user_id[order]
y = tr.y[order]
tm = tr.time_ms[order]
same = u[1:] == u[:-1]
prev = y[:-1][same]
nxt = y[1:][same]
p0 = nxt[prev == 0].mean() if np.any(prev == 0) else np.nan
p1 = nxt[prev == 1].mean() if np.any(prev == 1) else np.nan
ties = np.mean(tm[1:][same] == tm[:-1][same])
lines.append(f"SEQ transitions={same.sum()} P(y|prev0/1)={p0:.3f}/{p1:.3f} timestamp_ties={ties:.3f}")

# Inspect supplied historical feature interfaces and scalar shapes.
for key in ("video_id", "author_id"):
    h = historical_features("valid", key=key)
    desc = ",".join(
        f"{k}:{np.asarray(v).shape}/{np.asarray(v).dtype}"
        for k, v in h.items()
    )
    lines.append(f"HIST {key} {desc}")

# Keep output below the phase's truncation budget while preserving core diagnostics.
out = []
used = 0
for line in lines:
    need = len(line) + 1
    if used + need <= 3950:
        out.append(line)
        used += need
print("\n".join(out))