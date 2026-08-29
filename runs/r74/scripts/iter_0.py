import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")

def entropy_assoc(x, y, card):
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    size = max(int(card), int(x.max()) + 1)
    cnt = np.bincount(x, minlength=size).astype(np.float64)
    pos = np.bincount(x, weights=y, minlength=size)
    keep = cnt > 0
    q = np.divide(pos[keep], cnt[keep])
    eps = 1e-15
    hcond = np.sum(cnt[keep] * (
        -q * np.log(np.clip(q, eps, 1.0))
        -(1.0-q) * np.log(np.clip(1.0-q, eps, 1.0))
    )) / len(y)
    p = float(y.mean())
    hy = -p*np.log(max(p, eps)) - (1-p)*np.log(max(1-p, eps))
    return 0.0 if hy == 0 else max(0.0, 1.0 - hcond/hy), cnt

def user_stats(s):
    _, inv = np.unique(s.user_id, return_inverse=True)
    nr = np.bincount(inv)
    np_ = np.bincount(inv, weights=s.y).astype(np.int64)
    qrows = np.quantile(nr, [0, .1, .5, .9, .99, 1]).astype(int)
    qpos = np.quantile(np_, [0, .1, .5, .9, .99, 1]).astype(int)
    return len(nr), qrows, qpos, np.mean(np_ == 0), np.mean(np_ == nr), np.mean((np_ > 0) & (np_ < nr))

def date_line(s):
    parts = []
    for d in np.unique(s.date):
        m = s.date == d
        parts.append(f"{d}:{m.sum()}/{s.y[m].mean():.3f}")
    return " ".join(parts)

def overlap(name):
    a, b = tr.X[name], va.X[name]
    size = max(FEATURE_CARDINALITIES[name], int(a.max())+1, int(b.max())+1)
    seen = np.bincount(a, minlength=size) > 0
    seen[0] = False
    return np.mean(~seen[b]), np.mean(b == 0), np.count_nonzero(seen)

print(f"shape train={len(tr.user_id)} valid={len(va.user_id)} X={len(tr.X)} num={len(tr.num)} aux={len(tr.aux)}")
print(f"label train pos={tr.y.sum()} rate={tr.y.mean():.5f}; valid pos={va.y.sum()} rate={va.y.mean():.5f}")
for nm, s in [("train", tr), ("valid", va)]:
    nu, qr, qp, z, a, mix = user_stats(s)
    print(f"user {nm} n={nu} rows_q={qr.tolist()} pos_q={qp.tolist()} zero={z:.3f} all={a:.3f} mixed={mix:.3f}")
print("dates train " + date_line(tr))
print("dates valid " + date_line(va))
for f in ["user_id", "video_id", "author_id"]:
    u, z, nseen = overlap(f)
    print(f"cold {f}: valid_rows_not_seen={u:.3f} id0={z:.3f} train_seen_ids={nseen}")
scalar_ok = all(np.asarray(v).shape == (len(tr.user_id),) for v in tr.X.values())
print(f"representation categorical_scalar_per_row={scalar_ok} time_shape={tr.time_ms.shape} date_shape={tr.date.shape}")
print("CAT fields: card, train_unique, valid_unique, valid_unseen_row, train_topfreq, train_id0, NMI(train/valid)")
for f in tr.X:
    card = FEATURE_CARDINALITIES[f]
    nt, ct = entropy_assoc(tr.X[f], tr.y, card)
    nv, cv = entropy_assoc(va.X[f], va.y, card)
    size = max(len(ct), len(cv))
    seen = np.zeros(size, dtype=bool)
    seen[:len(ct)] = ct > 0
    seen[0] = False
    bx = va.X[f]
    unseen = np.mean(~seen[bx])
    top = ct.max() / len(tr.y)
    zero = ct[0] / len(tr.y)
    print(f"C {f} {card} {np.count_nonzero(ct)} {np.count_nonzero(cv)} {unseen:.3f} {top:.3f} {zero:.3f} {nt:.3f}/{nv:.3f}")
print("NUM fields: missing(train/valid), train q10/q50/q90, corr(log1p,value with y train/valid)")
for f in tr.num:
    vals = []
    for s in (tr, va):
        x = np.asarray(s.num[f], dtype=np.float64)
        ok = np.isfinite(x)
        z = np.log1p(np.maximum(x[ok], 0))
        corr = np.corrcoef(z, s.y[ok])[0, 1] if len(z) and np.std(z) > 0 else 0.0
        vals.append((1-ok.mean(), corr))
    xt = np.asarray(tr.num[f], dtype=np.float64)
    q = np.nanquantile(xt, [.1, .5, .9])
    print(f"N {f} miss={vals[0][0]:.3f}/{vals[1][0]:.3f} q={q.round(2).tolist()} corr={vals[0][1]:.3f}/{vals[1][1]:.3f}")
for key in ("video_id", "author_id"):
    try:
        h = historical_features("train", key=key)
        desc = ",".join(f"{k}:{np.asarray(v).shape}/{np.asarray(v).dtype}" for k, v in h.items())
        print(f"HIST {key} {desc}")
    except Exception as e:
        print(f"HIST {key} ERROR {type(e).__name__}:{e}")