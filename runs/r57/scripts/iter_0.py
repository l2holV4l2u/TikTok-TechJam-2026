import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")  # Inputs only; never access te.y.
lines = []

lines.append(
    f"ROWS train={len(tr.user_id)} valid={len(va.user_id)} test={len(te.user_id)} "
    f"cat={len(tr.X)} num={len(tr.num)}"
)
first = next(iter(tr.X))
scalar_ok = all(
    np.asarray(a).ndim == 1 and len(a) == len(tr.user_id)
    for a in tr.X.values()
)
lines.append(
    f"SHAPE categorical_scalar={scalar_ok} example={first}:{tr.X[first].shape}/"
    f"{tr.X[first].dtype}"
)

def user_summary(s, name):
    users, inv, cnt = np.unique(s.user_id, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=np.asarray(s.y, dtype=np.float64))
    qn = np.quantile(cnt, [0.1, 0.5, 0.9, 0.99])
    qp = np.quantile(pos, [0.1, 0.5, 0.9, 0.99])
    zero = np.mean(pos == 0)
    allp = np.mean(pos == cnt)
    mixed = np.mean((pos > 0) & (pos < cnt))
    lines.append(
        f"USER {name} n={len(users)} y={np.mean(s.y):.4f} "
        f"rows_q10/50/90/99={','.join(f'{x:.0f}' for x in qn)} "
        f"pos_q={','.join(f'{x:.0f}' for x in qp)} "
        f"zero/all/mixed={zero:.3f}/{allp:.3f}/{mixed:.3f}"
    )

user_summary(tr, "tr")
user_summary(va, "va")

def daily(s, name):
    out = []
    for d in np.unique(s.date):
        m = s.date == d
        out.append(f"{str(int(d))[-2:]}:{np.mean(s.y[m]):.3f}")
    lines.append(f"DAYRATE {name} " + " ".join(out))

daily(tr, "tr")
daily(va, "va")

def overlap(key):
    a = np.asarray(tr.X[key])
    atr = np.unique(a[a != 0])
    for s, nm in ((va, "v"), (te, "t")):
        x = np.asarray(s.X[key])
        unseen = (x == 0) | ~np.isin(x, atr, assume_unique=False)
        lines.append(
            f"OVERLAP {key} {nm}: unseen_rows={np.mean(unseen):.3f} "
            f"zero={np.mean(x == 0):.3f} ids={len(np.unique(x))}"
        )

for key in ("user_id", "video_id", "author_id"):
    overlap(key)

# How much sequential or repeat structure exists inside train.
uid = np.asarray(tr.user_id)
tm = np.asarray(tr.time_ms)
row = np.arange(len(uid), dtype=np.int64)
o = np.lexsort((row, tm, uid))
same_batch_extra = np.mean(
    (uid[o][1:] == uid[o][:-1]) & (tm[o][1:] == tm[o][:-1])
)
vid = np.asarray(tr.video_id)
o2 = np.lexsort((vid, uid))
repeat_pair_extra = np.mean(
    (uid[o2][1:] == uid[o2][:-1]) & (vid[o2][1:] == vid[o2][:-1])
)
lines.append(
    f"SEQUENCE same_user_timestamp_extra={same_batch_extra:.3f} "
    f"repeat_user_video_extra={repeat_pair_extra:.3f}"
)

# Numeric availability, scale, and simple signed-log point-biserial association.
for name, x0 in tr.num.items():
    x = np.asarray(x0, dtype=np.float64)
    ok = np.isfinite(x)
    vals = x[ok]
    if len(vals):
        q50, q99 = np.quantile(vals, [0.5, 0.99])
        z = np.sign(vals) * np.log1p(np.abs(vals))
        yy = np.asarray(tr.y, dtype=np.float64)[ok]
        corr = np.corrcoef(z, yy)[0, 1] if np.std(z) > 0 else 0.0
    else:
        q50 = q99 = corr = np.nan
    miss = ~ok
    delta = (
        float(np.mean(tr.y[miss]) - np.mean(tr.y[ok]))
        if np.any(miss) and np.any(ok) else 0.0
    )
    lines.append(
        f"NUM {name} miss={np.mean(miss):.3f} med/p99={q50:.1f}/{q99:.1f} "
        f"logcorr={corr:.3f} miss_dy={delta:+.3f}"
    )

def weighted_corr(x, y, w):
    sw = np.sum(w)
    if sw <= 0:
        return np.nan
    mx = np.sum(w * x) / sw
    my = np.sum(w * y) / sw
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)
    if vx <= 0 or vy <= 0:
        return np.nan
    return np.sum(w * (x - mx) * (y - my)) / np.sqrt(vx * vy)

lines.append("FIELD name C/T/V zV% newV% topT% etaT% rho(k)")
yt = np.asarray(tr.y, dtype=np.float64)
yv = np.asarray(va.y, dtype=np.float64)
p = np.mean(yt)
total_var = len(yt) * p * (1.0 - p)

for name in tr.X:
    xt = np.asarray(tr.X[name], dtype=np.int64)
    xv = np.asarray(va.X[name], dtype=np.int64)
    card = max(
        int(FEATURE_CARDINALITIES[name]),
        int(xt.max(initial=0)) + 1,
        int(xv.max(initial=0)) + 1,
    )
    ct = np.bincount(xt, minlength=card).astype(np.float64)
    st = np.bincount(xt, weights=yt, minlength=card)
    cv = np.bincount(xv, minlength=card).astype(np.float64)
    sv = np.bincount(xv, weights=yv, minlength=card)

    seen_t = int(np.count_nonzero(ct))
    seen_v = int(np.count_nonzero(cv))
    rt = np.divide(st, ct, out=np.full(card, p), where=ct > 0)
    rv = np.divide(sv, cv, out=np.zeros(card), where=cv > 0)

    between = np.sum(ct[ct > 0] * (rt[ct > 0] - p) ** 2)
    noise = max(seen_t - 1, 0) * p * (1.0 - p)
    eta = max(0.0, (between - noise) / max(total_var, 1e-12))

    stable = (ct >= 50) & (cv >= 20)
    k = int(np.sum(stable))
    rho = weighted_corr(rt[stable], rv[stable],
                        np.minimum(ct[stable], cv[stable])) if k >= 2 else np.nan
    novel = np.mean(ct[np.minimum(xv, card - 1)] == 0)
    top = np.max(ct) / len(xt)
    rtxt = "nan" if not np.isfinite(rho) else f"{rho:.2f}"
    lines.append(
        f"{name} {card}/{seen_t}/{seen_v} {100*np.mean(xv==0):.1f} "
        f"{100*novel:.1f} {100*top:.1f} {100*eta:.2f} {rtxt}({k})"
    )

# Inspect available leakage-safe historical feature names and array shapes.
for key in ("video_id", "author_id"):
    try:
        h = historical_features("valid", key=key)
        desc = ",".join(
            f"{n}:{np.asarray(a).shape}" for n, a in h.items()
        )
        if len(desc) > 300:
            desc = desc[:297] + "..."
        lines.append(f"HISTORY {key} {desc}")
    except Exception as e:
        lines.append(f"HISTORY {key} error={type(e).__name__}:{str(e)[:80]}")

text = "\n".join(lines)
print(text[:3990])