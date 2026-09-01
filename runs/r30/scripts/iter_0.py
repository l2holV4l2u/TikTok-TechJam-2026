import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES

tr = load("train")
va = load("valid")

def qfmt(x):
    q = np.quantile(x, [0, .5, .9, .99, 1])
    return "/".join(f"{v:.0f}" for v in q)

def user_summary(split, tag):
    users, inv, rows = np.unique(
        split.user_id, return_inverse=True, return_counts=True
    )
    pos = np.bincount(inv, weights=split.y, minlength=len(users))
    zero = np.mean(pos == 0)
    allpos = np.mean(pos == rows)
    mixed = (pos > 0) & (pos < rows)
    eligible_pos = pos[mixed].sum() / max(float(pos.sum()), 1.0)
    print(
        f"USER {tag} n={len(users)} rows_q0/50/90/99/max={qfmt(rows)} "
        f"pos_q={qfmt(pos)} zero={zero:.3f} all={allpos:.3f} "
        f"mixed={mixed.mean():.3f} pos_in_mixed={eligible_pos:.3f}"
    )

def overlap(train_ids, valid_ids, name):
    tu = np.unique(train_ids)
    vu = np.unique(valid_ids)
    unseen_rows = np.mean(~np.isin(valid_ids, tu))
    unseen_unique = np.mean(~np.isin(vu, tu))
    print(
        f"OVERLAP {name} train_u={len(tu)} valid_u={len(vu)} "
        f"valid_unseen_rows={unseen_rows:.4f} valid_unseen_ids={unseen_unique:.4f}"
    )

def weighted_sd(rate, count):
    w = count.astype(np.float64)
    if w.sum() == 0:
        return 0.0
    mu = np.dot(w, rate) / w.sum()
    return float(np.sqrt(np.dot(w, (rate - mu) ** 2) / w.sum()))

def weighted_corr(a, b, w):
    w = w.astype(np.float64)
    if len(w) < 2 or w.sum() == 0:
        return np.nan
    w /= w.sum()
    ma, mb = np.dot(w, a), np.dot(w, b)
    da, db = a - ma, b - mb
    den = np.sqrt(np.dot(w, da * da) * np.dot(w, db * db))
    return float(np.dot(w, da * db) / den) if den > 0 else np.nan

print(
    f"ROWS train={len(tr.y)} valid={len(va.y)} "
    f"features={len(tr.X)} train_rate={tr.y.mean():.5f} "
    f"valid_rate={va.y.mean():.5f}"
)
print(
    f"LABEL train_counts={np.bincount(tr.y, minlength=2).tolist()} "
    f"valid_counts={np.bincount(va.y, minlength=2).tolist()}"
)

shapes_tr = sorted(set((x.ndim, x.shape[0], str(x.dtype)) for x in tr.X.values()))
shapes_va = sorted(set((x.ndim, x.shape[0], str(x.dtype)) for x in va.X.values()))
bad = [k for k, x in tr.X.items() if x.ndim != 1 or len(x) != len(tr.y)]
print(f"X_SHAPES train={shapes_tr} valid={shapes_va} nonscalar_or_bad={bad}")

aux_names = sorted(set(tr.aux) | set(va.aux))
print(f"AUX count={len(aux_names)}")
joined = ",".join(aux_names)
for i in range(0, len(joined), 180):
    print("AUX_KEYS " + joined[i:i + 180])

user_summary(tr, "train")
user_summary(va, "valid")
overlap(tr.user_id, va.user_id, "user")
overlap(tr.video_id, va.video_id, "video")

print("FIELD columns: card uniq_tr/va new% zero% top% smoothedSD_tr/va corr(n>=20)")
for name in sorted(tr.X):
    xt = np.asarray(tr.X[name], dtype=np.int64)
    xv = np.asarray(va.X[name], dtype=np.int64)
    card = max(
        int(FEATURE_CARDINALITIES[name]),
        int(xt.max(initial=0)) + 1,
        int(xv.max(initial=0)) + 1,
    )
    ct = np.bincount(xt, minlength=card)
    cv = np.bincount(xv, minlength=card)
    pt = np.bincount(xt, weights=tr.y, minlength=card)
    pv = np.bincount(xv, weights=va.y, minlength=card)

    alpha = 20.0
    rt = (pt + alpha * tr.y.mean()) / (ct + alpha)
    rv = (pv + alpha * va.y.mean()) / (cv + alpha)
    sdt = weighted_sd(rt, ct)
    sdv = weighted_sd(rv, cv)

    supported = (ct >= 20) & (cv >= 20)
    rawt = pt[supported] / ct[supported]
    rawv = pv[supported] / cv[supported]
    corr = weighted_corr(rawt, rawv, np.minimum(ct[supported], cv[supported]))

    new_rows = cv[ct == 0].sum() / len(xv)
    zero_rows = np.mean(xv == 0)
    top = ct.max(initial=0) / len(xt)
    corr_s = "nan" if not np.isfinite(corr) else f"{corr:.2f}"
    print(
        f"F {name} c={card} u={np.count_nonzero(ct)}/{np.count_nonzero(cv)} "
        f"new={100*new_rows:.1f} z={100*zero_rows:.1f} top={100*top:.1f} "
        f"sd={sdt:.3f}/{sdv:.3f} r={corr_s}({supported.sum()})"
    )