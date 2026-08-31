import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
lines = []

def date_rates(s):
    out = []
    for d in np.unique(s.date):
        z = s.y[s.date == d]
        out.append(f"{str(int(d))[-4:]}:{z.mean():.3f}")
    return " ".join(out)

def user_summary(name, s):
    u, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=np.asarray(s.y, dtype=np.float64))
    qn = np.quantile(n, [0, .25, .5, .75, .9, .99]).astype(int)
    qp = np.quantile(p, [0, .25, .5, .75, .9, .99])
    zero = np.mean(p == 0)
    mixed = np.mean((p > 0) & (p < n))
    lines.append(
        f"USER {name} U={len(u)} rowsq={qn.tolist()} "
        f"posq={np.round(qp,1).tolist()} zero={zero:.3f} mixed={mixed:.3f}"
    )

lines.append(
    f"DATA train={len(tr)} p={np.mean(tr.y):.4f} valid={len(va)} "
    f"p={np.mean(va.y):.4f} X={len(tr.X)} num={len(tr.num)}"
)
lines.append("DATES tr " + date_rates(tr))
lines.append("DATES va " + date_rates(va))
user_summary("tr", tr)
user_summary("va", va)

# Confirm that categorical features are scalar row-aligned arrays.
scalar_ok = all(
    np.asarray(x).ndim == 1 and len(x) == len(tr)
    for x in tr.X.values()
)
lines.append(
    f"SHAPE categorical_scalar={scalar_ok} example={next(iter(tr.X.values())).shape} "
    f"time={tr.time_ms.shape} y={tr.y.shape}"
)

# Same-user equal timestamps indicate feed batches and limits of exact chronology.
order = np.lexsort((np.arange(len(tr)), tr.time_ms, tr.user_id))
uo = tr.user_id[order]
to = tr.time_ms[order]
ties = (uo[1:] == uo[:-1]) & (to[1:] == to[:-1])
lines.append(f"TIME adjacent_same_user_timestamp={ties.mean():.4f}")

# Numeric availability, scale, and simple train-only marginal association.
for name, x0 in tr.num.items():
    x = np.asarray(x0, dtype=np.float64)
    ok = np.isfinite(x)
    if ok.any():
        q = np.quantile(x[ok], [.01, .5, .99])
        lx = np.log1p(np.maximum(x[ok], 0))
        yy = np.asarray(tr.y, dtype=np.float64)[ok]
        corr = np.corrcoef(lx, yy)[0, 1] if np.std(lx) > 0 else 0.0
    else:
        q, corr = [np.nan] * 3, np.nan
    lines.append(
        f"NUM {name} miss={1-ok.mean():.3f} "
        f"q01/50/99={','.join(f'{v:.1f}' for v in q)} logcorr={corr:.3f}"
    )

# Inspect exact historical feature names once; all arrays should be row-aligned.
hv = historical_features("train", key="video_id")
ha = historical_features("train", key="author_id")
vkeys = ",".join(hv.keys())
akeys = ",".join(ha.keys())
lines.append(f"HIST video n={len(hv)} aligned={all(len(x)==len(tr) for x in hv.values())} keys={vkeys}")
lines.append(
    f"HIST author n={len(ha)} aligned={all(len(x)==len(tr) for x in ha.values())} "
    f"keys={'same' if akeys == vkeys else akeys}"
)

# Train-only bias-adjusted categorical eta^2. The df correction suppresses
# spurious apparent signal from extremely high-cardinality fields.
y = np.asarray(tr.y, dtype=np.float64)
pbar = y.mean()
den = len(y) * pbar * (1.0 - pbar)
lines.append("FIELD: K=declared ot/ov=observed top=maxfreq sing=id-singleton new=valid-row-unseen eta=adj*1000")
for name in tr.X:
    xt = np.asarray(tr.X[name])
    xv = np.asarray(va.X[name])
    ids, inv, cnt = np.unique(xt, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=y)
    bss = np.sum(pos * pos / cnt) - y.sum() ** 2 / len(y)
    noise = pbar * (1.0 - pbar) * max(len(ids) - 1, 0)
    eta = max(0.0, (bss - noise) / den) if den else 0.0
    j = np.searchsorted(ids, xv)
    seen = (j < len(ids))
    seen[seen] &= ids[j[seen]] == xv[seen]
    lines.append(
        f"F {name} K={FEATURE_CARDINALITIES[name]} ot={len(ids)} "
        f"ov={len(np.unique(xv))} top={cnt.max()/len(xt):.3f} "
        f"sing={np.mean(cnt==1):.3f} new={np.mean(~seen):.3f} eta={1000*eta:.2f}"
    )

text = "\n".join(lines)
# Preserve the complete field table if unusually long, shortening history metadata first.
if len(text) > 3950:
    lines = [ln if not ln.startswith("HIST ") else ln[:150] for ln in lines]
    text = "\n".join(lines)
print(text[:3990])