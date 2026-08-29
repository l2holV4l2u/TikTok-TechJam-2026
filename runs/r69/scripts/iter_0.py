import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")
lines = []

lines.append(
    f"ROWS train={len(tr.user_id)} valid={len(va.user_id)} test={len(te.user_id)} "
    f"features={len(tr.X)} nums={len(tr.num)}"
)
shapes = sorted(set((v.ndim, v.shape[0]) for v in tr.X.values()))
lines.append(f"X_SHAPES={shapes} scalar_per_row={all(v.ndim == 1 for v in tr.X.values())}")
lines.append(f"LABEL train={tr.y.mean():.4f} valid={va.y.mean():.4f}")

def date_rates(s):
    out = []
    for d in np.unique(s.date):
        m = s.date == d
        out.append(f"{d % 10000:04d}:{m.sum()//1000}k/{s.y[m].mean():.3f}")
    return " ".join(out)

lines.append("TRAIN_DATES " + date_rates(tr))
lines.append("VALID_DATES " + date_rates(va))

def user_summary(name, s):
    _, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    qn = np.percentile(n, [10, 50, 90, 99])
    qp = np.percentile(p, [10, 50, 90, 99])
    zero = np.mean(p == 0)
    allp = np.mean(p == n)
    mixed = np.mean((p > 0) & (p < n))
    pos_weight_eligible = p[(p > 0) & (p < n)].sum() / max(p.sum(), 1)
    lines.append(
        f"USER_{name} n={len(n)} rows_q10/50/90/99="
        f"{qn[0]:.0f}/{qn[1]:.0f}/{qn[2]:.0f}/{qn[3]:.0f} "
        f"pos_q={qp[0]:.0f}/{qp[1]:.0f}/{qp[2]:.0f}/{qp[3]:.0f} "
        f"zero/all/mix={zero:.3f}/{allp:.3f}/{mixed:.3f} "
        f"GAUC_pos_cov={pos_weight_eligible:.3f}"
    )

user_summary("TR", tr)
user_summary("VA", va)

def overlap(name, a, b):
    ua = np.unique(a)
    ub = np.unique(b)
    seen = np.isin(b, ua, assume_unique=False)
    useen = np.isin(ub, ua, assume_unique=True)
    lines.append(
        f"OVERLAP_{name} trainU={len(ua)} evalU={len(ub)} "
        f"new_rows={1-seen.mean():.4f} new_entities={1-useen.mean():.4f}"
    )

overlap("user_V", tr.user_id, va.user_id)
overlap("user_T", tr.user_id, te.user_id)
overlap("video_V", tr.video_id, va.video_id)
overlap("video_T", tr.video_id, te.video_id)
overlap("author_V", tr.X["author_id"], va.X["author_id"])

base_p = np.clip(tr.y.mean(), 1e-6, 1 - 1e-6)
base_ll = -np.mean(va.y * np.log(base_p) + (1 - va.y) * np.log(1 - base_p))
lines.append("FIELDS C=declared U=train/valid/test z=zero-row(valid/test) new=no-train-row gain=valid-LLx1e3")

for name in tr.X:
    xt = tr.X[name]
    xv = va.X[name]
    xe = te.X[name]
    mx = int(max(xt.max(initial=0), xv.max(initial=0), xe.max(initial=0)))
    cnt = np.bincount(xt, minlength=mx + 1)
    pos = np.bincount(xt, weights=tr.y, minlength=mx + 1)
    uv = np.count_nonzero(np.bincount(xv))
    ue = np.count_nonzero(np.bincount(xe))
    ut = np.count_nonzero(cnt)
    pred = (pos[xv] + 20.0 * base_p) / (cnt[xv] + 20.0)
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    ll = -np.mean(va.y * np.log(pred) + (1 - va.y) * np.log(1 - pred))
    gain = 1000.0 * (base_ll - ll)
    lines.append(
        f"F {name} C={FEATURE_CARDINALITIES[name]} U={ut}/{uv}/{ue} "
        f"z={np.mean(xv==0):.3f}/{np.mean(xe==0):.3f} "
        f"new={np.mean(cnt[xv]==0):.3f} gain={gain:+.2f}"
    )

def corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    m = np.isfinite(x)
    if m.sum() < 2:
        return np.nan
    xx = x[m]
    yy = np.asarray(y, dtype=np.float64)[m]
    if xx.std() == 0 or yy.std() == 0:
        return 0.0
    return np.corrcoef(xx, yy)[0, 1]

for name in tr.num:
    a = tr.num[name]
    b = va.num[name]
    finite = np.isfinite(a)
    q = np.nanpercentile(a, [5, 50, 95]) if finite.any() else [np.nan] * 3
    c_tr = corr(np.log1p(np.maximum(a, 0)), tr.y)
    c_va = corr(np.log1p(np.maximum(b, 0)), va.y)
    lines.append(
        f"NUM {name} miss={np.mean(~finite):.3f}/{np.mean(~np.isfinite(b)):.3f} "
        f"q5/50/95={q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f} corr={c_tr:+.3f}/{c_va:+.3f}"
    )

lines.append("AUX(outcomes,do_not_use) " + ",".join(
    f"{k}:{np.asarray(v).shape}" for k, v in tr.aux.items()
)[:500])

for key in ("video_id", "author_id"):
    h = historical_features("valid", key=key)
    ranked = []
    for name, arr in h.items():
        arr = np.asarray(arr)
        ranked.append((abs(corr(arr, va.y)), name, corr(arr, va.y),
                       np.mean(~np.isfinite(arr)), arr.shape))
    ranked.sort(reverse=True)
    desc = ";".join(
        f"{n}:r={c:+.3f},nan={miss:.3f},sh={shape}"
        for _, n, c, miss, shape in ranked[:8]
    )
    lines.append(f"HIST_{key} keys={len(h)} top {desc}")

for line in lines[:60]:
    print(line)