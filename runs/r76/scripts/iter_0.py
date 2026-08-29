import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")
lines = []

def q(x, ps=(.5, .9, .99)):
    return "/".join(f"{np.quantile(x, p):.1f}" for p in ps)

def user_stats(s, name):
    _, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    lines.append(
        f"USR {name} U={len(n)} n50/90/99={q(n)} "
        f"p50/90/99={q(p)} zero={np.mean(p==0):.3f} "
        f"all={np.mean(p==n):.3f} mixed={np.mean((p>0)&(p<n)):.3f}"
    )

def dates(s, name):
    ds = []
    for d in np.unique(s.date):
        m = s.date == d
        ds.append(f"{d}:{m.sum()}/{s.y[m].mean():.3f}")
    lines.append(f"DATE {name} " + " ".join(ds))

def adjusted_signal(x, y, card):
    n = np.bincount(x, minlength=card).astype(np.float64)
    p = np.bincount(x, weights=y, minlength=card)
    used = n > 0
    mu = y.mean()
    if mu <= 0 or mu >= 1:
        return 0.0, int(used.sum()), 1.0
    raw = np.sum((p[used] - n[used] * mu) ** 2 / n[used])
    raw /= len(y) * mu * (1.0 - mu)
    bias = max(0.0, (used.sum() - 1) / len(y))
    adj = max(0.0, (raw - bias) / max(1e-12, 1.0 - bias))
    return adj, int(used.sum()), n.max() / len(y)

def sequence_stats(s, name):
    idx = np.lexsort((np.arange(len(s.user_id)), s.time_ms, s.user_id))
    u, t, y = s.user_id[idx], s.time_ms[idx], s.y[idx]
    same = u[1:] == u[:-1]
    yp, yc = y[:-1][same], y[1:][same]
    gaps = (t[1:] - t[:-1])[same] / 1000.0
    p1 = yc[yp == 1].mean() if np.any(yp == 1) else np.nan
    p0 = yc[yp == 0].mean() if np.any(yp == 0) else np.nan
    lines.append(
        f"SEQ {name} transitions={same.sum()} tie={np.mean(gaps==0):.3f} "
        f"gap_s50/90={np.quantile(gaps,.5):.1f}/{np.quantile(gaps,.9):.1f} "
        f"P1|prev1={p1:.3f} P1|prev0={p0:.3f}"
    )

all_scalar = all(np.asarray(v).ndim == 1 and len(v) == len(tr.y) for v in tr.X.values())
lines.append(
    f"SHAPE train={len(tr.y)} valid={len(va.y)} test={len(te.user_id)} "
    f"fields={len(tr.X)} nums={len(tr.num)} scalar1d={all_scalar}"
)
lines.append(
    f"LABEL train={tr.y.mean():.4f} ({tr.y.sum()}) "
    f"valid={va.y.mean():.4f} ({va.y.sum()})"
)
user_stats(tr, "T")
user_stats(va, "V")
dates(tr, "T")
dates(va, "V")

entity_masks = {}
parts = []
for name, a, b, c in [
    ("user", tr.user_id, va.user_id, te.user_id),
    ("video", tr.video_id, va.video_id, te.video_id),
    ("author", tr.X["author_id"], va.X["author_id"], te.X["author_id"]),
]:
    seen_t = np.unique(a)
    mv = np.isin(b, seen_t)
    seen_tv = np.unique(np.concatenate((a, b)))
    mt = np.isin(c, seen_tv)
    entity_masks[name] = mv
    parts.append(f"{name}:Vseen={mv.mean():.3f},EseenTV={mt.mean():.3f}")
lines.append("OVERLAP " + " ".join(parts))
lines.append(
    "COLD valid_rate "
    + " ".join(
        f"{k}:seen={va.y[m].mean():.3f},new={va.y[~m].mean():.3f}"
        for k, m in entity_masks.items()
        if np.any(m) and np.any(~m)
    )
)

sequence_stats(tr, "T")
sequence_stats(va, "V")

for key in ("video_id", "author_id"):
    h = historical_features("valid", key=key)
    desc = ",".join(f"{k}:{np.asarray(v).shape}" for k, v in h.items())
    lines.append(f"HIST {key} " + desc[:260])

for name in sorted(tr.num):
    a, b = tr.num[name], va.num[name]
    fa, fb = np.isfinite(a), np.isfinite(b)
    za = np.log1p(np.maximum(a[fa].astype(np.float64), 0))
    zb = np.log1p(np.maximum(b[fb].astype(np.float64), 0))
    ca = np.corrcoef(za, tr.y[fa])[0, 1] if fa.sum() > 2 and za.std() else 0
    cb = np.corrcoef(zb, va.y[fb])[0, 1] if fb.sum() > 2 and zb.std() else 0
    lines.append(
        f"NUM {name} missT/V={1-fa.mean():.3f}/{1-fb.mean():.3f} "
        f"q50/95={np.quantile(a[fa],.5):.1f}/{np.quantile(a[fa],.95):.1f} "
        f"logcorrT/V={ca:.3f}/{cb:.3f}"
    )

for name in sorted(tr.X):
    card = FEATURE_CARDINALITIES[name]
    st, at, top = adjusted_signal(tr.X[name], tr.y, card)
    sv, av, _ = adjusted_signal(va.X[name], va.y, card)
    ae = np.unique(te.X[name]).size
    zv = np.mean(va.X[name] == 0)
    ze = np.mean(te.X[name] == 0)
    lines.append(
        f"F {name} C={card} A={at}/{av}/{ae} Z={zv:.2f}/{ze:.2f} "
        f"top={top:.2f} S={st:.4f}/{sv:.4f}"
    )

text = "\n".join(lines)
print(text[:3990])