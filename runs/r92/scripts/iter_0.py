import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")
lines = []

def user_summary(s, labeled):
    _, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    qn = np.quantile(n, [0, .25, .5, .75, .9, .99]).astype(int)
    text = f"N={len(n)} rowsQ={qn.tolist()}"
    if labeled:
        p = np.bincount(inv, weights=s.y)
        qp = np.quantile(p, [0, .25, .5, .75, .9, .99])
        text += (f" posQ={np.round(qp,1).tolist()} zero={np.mean(p==0):.3f}"
                 f" mixed={np.mean((p>0)&(p<n)):.3f}")
    return text

lines.append(f"rows train/valid/test={len(tr.user_id)}/{len(va.user_id)}/{len(te.user_id)}")
lines.append(f"label rate train/valid={tr.y.mean():.4f}/{va.y.mean():.4f}")
lines.append("users train " + user_summary(tr, True))
lines.append("users valid " + user_summary(va, True))
lines.append("users test  " + user_summary(te, False))

def daily(s):
    ds = np.unique(s.date)
    return " ".join(f"{int(d)%10000:04d}:{np.mean(s.y[s.date==d]):.3f}"
                    for d in ds)

lines.append("daily train " + daily(tr))
lines.append("daily valid " + daily(va))
lines.append(f"X scalar check: fields={len(tr.X)} shapes="
             f"{len(set(a.shape for a in tr.X.values()))} dtypes="
             f"{sorted(set(str(a.dtype) for a in tr.X.values()))}")

def corrected_eta(x, y, card):
    n = np.bincount(x, minlength=card).astype(np.float64)
    p = np.bincount(x, weights=y, minlength=card)
    used = n > 0
    mu = float(np.mean(y))
    between = np.sum(n[used] * (p[used] / n[used] - mu) ** 2)
    null = mu * (1.0 - mu) * max(int(used.sum()) - 1, 0)
    den = len(y) * mu * (1.0 - mu)
    return np.sqrt(max(0.0, (between - null) / max(den, 1e-12)))

lines.append("FIELD: C=card U=train/valid/test unique z=unknown-row valid/test "
             "e=bias-corrected label association train/valid m=max train share")
for name in tr.X:
    a, b, c = tr.X[name], va.X[name], te.X[name]
    card = FEATURE_CARDINALITIES[name]
    ct = np.bincount(a, minlength=card)
    seen = ct > 0
    uv = (b == 0) | (~seen[b])
    ue = (c == 0) | (~seen[c])
    uniq = (np.count_nonzero(ct), np.unique(b).size, np.unique(c).size)
    et = corrected_eta(a, tr.y, card)
    ev = corrected_eta(b, va.y, card)
    mx = ct.max() / len(a)
    lines.append(f"{name} C={card} U={uniq[0]}/{uniq[1]}/{uniq[2]} "
                 f"z={uv.mean():.3f}/{ue.mean():.3f} "
                 f"e={et:.3f}/{ev:.3f} m={mx:.3f}")

lines.append("NUM: missing train/valid, train p01/p50/p99, corr(log1p,value,label) tr/va")
for name in tr.num:
    a = tr.num[name].astype(np.float64)
    b = va.num[name].astype(np.float64)
    fa, fb = np.isfinite(a), np.isfinite(b)
    q = np.quantile(a[fa], [.01, .5, .99]) if fa.any() else [np.nan] * 3
    ca = np.corrcoef(np.log1p(np.maximum(a[fa], 0)), tr.y[fa])[0, 1] if fa.sum() > 2 else np.nan
    cb = np.corrcoef(np.log1p(np.maximum(b[fb], 0)), va.y[fb])[0, 1] if fb.sum() > 2 else np.nan
    lines.append(f"{name} miss={1-fa.mean():.3f}/{1-fb.mean():.3f} "
                 f"q={q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f} c={ca:.3f}/{cb:.3f}")

for key in ("video_id", "author_id"):
    h = historical_features("train", key=key)
    names = ",".join(h.keys())
    shapes = sorted(set(str(v.shape) for v in h.values()))
    lines.append(f"hist {key}: n={len(h)} shapes={shapes} keys={names[:150]}")

text = "\n".join(lines)
print(text[:3990])