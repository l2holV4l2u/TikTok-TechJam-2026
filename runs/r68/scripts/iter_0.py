import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")
lines = []

def qstr(x):
    return "/".join(f"{v:.0f}" for v in np.quantile(x, [0, .25, .5, .75, .9, .99, 1]))

def corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x)
    if ok.sum() < 3 or np.std(x[ok]) == 0:
        return 0.0
    return float(np.corrcoef(x[ok], y[ok])[0, 1])

def adjusted_eta(x, y):
    x = np.asarray(x, dtype=np.int64)
    ncat = int(x.max()) + 1
    cnt = np.bincount(x, minlength=ncat).astype(np.float64)
    pos = np.bincount(x, weights=y, minlength=ncat)
    used = cnt > 0
    mu = float(np.mean(y))
    var = mu * (1.0 - mu)
    if var == 0:
        return 0.0
    rates = pos[used] / cnt[used]
    r2 = np.sum(cnt[used] * (rates - mu) ** 2) / (len(y) * var)
    k = int(used.sum())
    return float(1.0 - (1.0 - r2) * (len(y) - 1) / max(1, len(y) - k))

def user_stats(s, label):
    _, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    lines.append(
        f"{label}_USER users={len(n)} rows_q={qstr(n)} pos_q={qstr(p)} "
        f"zero={np.mean(p==0):.3f} all={np.mean(p==n):.3f} "
        f"mixed={np.mean((p>0)&(p<n)):.3f}"
    )

lines.append(
    f"SHAPES train={len(tr.user_id)} valid={len(va.user_id)} test={len(te.user_id)} "
    f"Xfields={len(tr.X)} num={len(tr.num)}"
)
lines.append(
    f"ARRAYS X_scalar={all(np.asarray(v).shape==(len(tr.user_id),) for v in tr.X.values())} "
    f"uid={tr.user_id.shape}/{tr.user_id.dtype} time={tr.time_ms.shape}/{tr.time_ms.dtype}"
)
lines.append(
    f"LABEL train_rate={np.mean(tr.y):.5f} train_pos={int(np.sum(tr.y))} "
    f"valid_rate={np.mean(va.y):.5f} valid_pos={int(np.sum(va.y))}"
)

for name, s in [("TR", tr), ("VA", va)]:
    ds = []
    for d in np.unique(s.date):
        m = s.date == d
        ds.append(f"{d}:{np.mean(s.y[m]):.3f}")
    lines.append(name + "_DATE " + " ".join(ds))

user_stats(tr, "TR")
user_stats(va, "VA")

for key in ["user_id", "video_id", "author_id"]:
    a = tr.X[key]
    seen = np.zeros(FEATURE_CARDINALITIES[key], dtype=bool)
    seen[np.unique(a)] = True
    if len(seen):
        seen[0] = False
    for tag, s in [("va", va), ("te", te)]:
        x = s.X[key]
        unseen = ~seen[x]
        lines.append(
            f"OVERLAP {key}/{tag} row_unseen={np.mean(unseen):.4f} "
            f"zero={np.mean(x==0):.4f} uniq={len(np.unique(x))} card={len(seen)}"
        )

order = np.lexsort((np.arange(len(tr.user_id)), tr.time_ms, tr.user_id))
u = tr.user_id[order]
tm = tr.time_ms[order]
same_user = u[1:] == u[:-1]
lines.append(
    f"TIME train_unique_ms={len(np.unique(tr.time_ms))} "
    f"adjacent_same_timestamp_given_user={np.mean((tm[1:]==tm[:-1])[same_user]):.4f}"
)

cat_stats = []
for name in tr.X:
    xt, xv = tr.X[name], va.X[name]
    card = FEATURE_CARDINALITIES[name]
    seen = np.zeros(card, dtype=bool)
    seen[np.unique(xt)] = True
    if card:
        seen[0] = False
    counts = np.bincount(xt, minlength=card)
    cat_stats.append({
        "name": name,
        "et": adjusted_eta(xt, tr.y),
        "ev": adjusted_eta(xv, va.y),
        "un": float(np.mean(~seen[xv])),
        "dom": float(counts.max() / len(xt)),
        "ut": int(np.count_nonzero(counts)),
        "uv": int(len(np.unique(xv))),
        "card": card
    })

sig = sorted(cat_stats, key=lambda z: z["ev"], reverse=True)
lines.append("CAT_SIGNAL_TOP name:etaV/etaT " +
             " ".join(f"{z['name']}:{z['ev']:.3f}/{z['et']:.3f}" for z in sig[:15]))
lines.append("CAT_SIGNAL_BOTTOM " +
             " ".join(f"{z['name']}:{z['ev']:.3f}/{z['et']:.3f}" for z in sig[-10:]))

nov = sorted(cat_stats, key=lambda z: z["un"], reverse=True)
lines.append("CAT_NOVELTY name:unseenV,uniqT/uniqV/card " +
             " ".join(f"{z['name']}:{z['un']:.2f},{z['ut']}/{z['uv']}/{z['card']}"
                      for z in nov[:12]))
dom = sorted(cat_stats, key=lambda z: z["dom"], reverse=True)
lines.append("CAT_DOMINANT " +
             " ".join(f"{z['name']}:{z['dom']:.3f}" for z in dom[:12]))

for name in tr.num:
    xtr = np.asarray(tr.num[name], dtype=np.float64)
    xva = np.asarray(va.num[name], dtype=np.float64)
    ok = np.isfinite(xtr)
    transform = np.log1p(np.maximum(xtr, 0))
    miss_rate = np.mean(tr.y[~ok]) if np.any(~ok) else np.nan
    vals = xtr[ok]
    qs = np.quantile(vals, [0, .5, .9, .99, 1]) if len(vals) else np.zeros(5)
    lines.append(
        f"NUM {name} missT/V={np.mean(~ok):.3f}/{np.mean(~np.isfinite(xva)):.3f} "
        f"logcorrT={corr(transform,tr.y):.3f} miss_y={miss_rate:.3f} "
        f"q0/50/90/99/max=" + "/".join(f"{v:.1f}" for v in qs)
    )

for key in ["video_id", "author_id"]:
    h = historical_features("valid", key=key)
    hs = []
    for name, x in h.items():
        x = np.asarray(x)
        hs.append((abs(corr(x, va.y)), corr(x, va.y), name,
                   np.mean(~np.isfinite(x)), x.shape))
    hs.sort(reverse=True)
    lines.append(
        f"HIST_{key} n={len(h)} top_abs_corr " +
        " ".join(f"{name}:{c:.3f},miss={miss:.2f},shape={shape}"
                 for _, c, name, miss, shape in hs[:8])
    )

text = "\n".join(lines)
print(text[:3950])