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
    if not len(a):
        return "NA"
    return "/".join(f"{v:.3g}" for v in np.quantile(a, qs))

def user_stats(s, tag):
    _, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    lines.append(
        f"USR {tag} n={len(n)} rows={qstr(n)} pos={qstr(p)} "
        f"zero={(p==0).mean():.3f} all={(p==n).mean():.3f} "
        f"mixed={((p>0)&(p<n)).mean():.3f}"
    )

lines.append(
    f"SHAPE tr/va/te={len(tr.user_id)}/{len(va.user_id)}/{len(te.user_id)} "
    f"X={len(tr.X)} num={len(tr.num)}"
)
scalar = all(np.asarray(x).ndim == 1 and len(x) == len(tr.user_id)
             for x in tr.X.values())
lines.append(f"LAYOUT X_scalar_per_row={scalar} sample={next(iter(tr.X.values())).shape}")
lines.append(
    f"LABEL tr={tr.y.mean():.4f}({int(tr.y.sum())}) "
    f"va={va.y.mean():.4f}({int(va.y.sum())})"
)
user_stats(tr, "tr")
user_stats(va, "va")

def values(s, key):
    if key == "user_id":
        return np.asarray(s.user_id)
    if key == "video_id":
        return np.asarray(s.video_id)
    return np.asarray(s.X[key])

def novelty(key):
    a, b, c = values(tr, key), values(va, key), values(te, key)
    au = np.unique(a)
    return (
        len(au), len(np.unique(b)), len(np.unique(c)),
        100 * np.mean(~np.isin(b, au)),
        100 * np.mean(~np.isin(c, au))
    )

for key in ("user_id", "video_id", "author_id"):
    x = novelty(key)
    lines.append(
        f"NEW {key} U={x[0]}/{x[1]}/{x[2]} "
        f"row_va/te={x[3]:.2f}/{x[4]:.2f}%"
    )

def daily(s):
    z = []
    for d in np.unique(s.date):
        m = s.date == d
        z.append(f"{int(d)%10000:04d}:{m.sum()//1000}k/{s.y[m].mean():.3f}")
    return " ".join(z)

lines.append("DAY tr " + daily(tr))
lines.append("DAY va " + daily(va))

for name in tr.num:
    a = np.asarray(tr.num[name], dtype=np.float64)
    b = np.asarray(va.num[name], dtype=np.float64)
    finite = np.isfinite(a)
    corr = np.nan
    logcorr = np.nan
    if finite.sum() > 1 and np.std(a[finite]) > 0:
        corr = np.corrcoef(a[finite], tr.y[finite])[0, 1]
        if np.min(a[finite]) >= 0:
            logcorr = np.corrcoef(np.log1p(a[finite]), tr.y[finite])[0, 1]
    lines.append(
        f"NUM {name} miss={np.mean(~finite):.2f}/{np.mean(~np.isfinite(b)):.2f} "
        f"q={qstr(a)} r={corr:.3f}/{logcorr:.3f}"
    )

def adj_eta(x, y, cardinality):
    x = np.asarray(x, dtype=np.int64)
    size = max(int(cardinality), int(x.max()) + 1)
    cnt = np.bincount(x, minlength=size).astype(np.float64)
    sy = np.bincount(x, weights=y, minlength=size)
    nz = cnt > 0
    p = float(np.mean(y))
    var = p * (1 - p)
    if var <= 0:
        return 0.0
    ssb = np.sum((sy[nz] - cnt[nz] * p) ** 2 / cnt[nz])
    df = max(0, int(nz.sum()) - 1)
    corrected = max(0.0, ssb - df * var)
    denom = max(1e-12, (len(y) - df) * var)
    return float(np.sqrt(min(1.0, corrected / denom)))

features = []
for name in tr.X:
    a = np.asarray(tr.X[name])
    b = np.asarray(va.X[name])
    c = np.asarray(te.X[name])
    ua = np.unique(a)
    cnt = np.bincount(a)
    et = adj_eta(a, tr.y, FEATURE_CARDINALITIES[name])
    ev = adj_eta(b, va.y, FEATURE_CARDINALITIES[name])
    text = (
        f"CAT {name} C={FEATURE_CARDINALITIES[name]} "
        f"U={len(ua)}/{len(np.unique(b))}/{len(np.unique(c))} "
        f"N={100*np.mean(~np.isin(b,ua)):.1f}/{100*np.mean(~np.isin(c,ua)):.1f} "
        f"Z={100*np.mean(a==0):.1f}/{100*np.mean(b==0):.1f} "
        f"M={100*cnt.max()/len(a):.1f} E={et:.3f}/{ev:.3f}"
    )
    features.append((et, text))

lines.append("CAT C=card U=tr/va/te N=new_va/te Z=zero_tr/va M=max% E=eta_tr/va")
for _, text in sorted(features, key=lambda z: z[0], reverse=True):
    lines.append(text)

order = np.lexsort((np.arange(len(tr.user_id)), tr.time_ms, tr.user_id))
u = tr.user_id[order]
y = tr.y[order]
tm = tr.time_ms[order]
same = u[1:] == u[:-1]
prev, nxt = y[:-1][same], y[1:][same]
p0 = nxt[prev == 0].mean() if np.any(prev == 0) else np.nan
p1 = nxt[prev == 1].mean() if np.any(prev == 1) else np.nan
ties = np.mean(tm[1:][same] == tm[:-1][same])
lines.append(
    f"SEQ n={same.sum()} Pnext_prev0/1={p0:.3f}/{p1:.3f} ties={ties:.3f}"
)

for key in ("video_id", "author_id"):
    h = historical_features("valid", key=key)
    desc = ",".join(f"{k}:{np.asarray(v).shape}" for k, v in h.items())
    lines.append(f"HIST {key} {desc}")

out, used = [], 0
for line in lines:
    if used + len(line) + 1 <= 3990 and len(out) < 60:
        out.append(line)
        used += len(line) + 1
print("\n".join(out))