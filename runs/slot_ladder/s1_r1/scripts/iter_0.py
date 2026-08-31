import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")

print("rows train/valid/test", len(tr.user_id), len(va.user_id), len(te.user_id),
      "Xfields", len(tr.X), "num", len(tr.num))
print("X scalar shapes", {k: (v.shape, str(v.dtype)) for k, v in list(tr.X.items())[:2]})

def daily(s, with_y=True):
    d, inv, n = np.unique(s.date, return_inverse=True, return_counts=True)
    if with_y:
        p = np.bincount(inv, weights=s.y, minlength=len(d)) / n
        return ",".join(f"{x}:{c}/{r:.3f}" for x, c, r in zip(d, n, p))
    return ",".join(f"{x}:{c}" for x, c in zip(d, n))

print("train day rows/rate", daily(tr))
print("valid day rows/rate", daily(va))
print("test day rows", daily(te, False))
print("label rate train/valid", round(float(tr.y.mean()), 5), round(float(va.y.mean()), 5))

def user_stats(s):
    _, inv = np.unique(s.user_id, return_inverse=True)
    nr = np.bincount(inv)
    np_ = np.bincount(inv, weights=s.y)
    q = [0, .25, .5, .75, .9, .99, 1]
    return len(nr), np.quantile(nr, q), np.quantile(np_, q), np.mean(np_ == 0), np.mean(np_ == nr)

for name, s in [("train", tr), ("valid", va)]:
    nu, qr, qp, z, a = user_stats(s)
    print(name, "users", nu, "rowsQ", np.round(qr, 1), "posQ", np.round(qp, 1),
          "zero/all", round(float(z), 4), round(float(a), 4))

def unseen_rows(a, b, card):
    seen = np.zeros(card, dtype=bool)
    seen[np.unique(a)] = True
    return float(np.mean(~seen[b]))

print("row unseen valid/test user video author",
      *[f"{unseen_rows(tr.X[k], s.X[k], FEATURE_CARDINALITIES[k]):.3f}"
        for k in ("user_id", "video_id", "author_id") for s in (va, te)])

for k, x in tr.num.items():
    ok = np.isfinite(x)
    z = np.log1p(np.maximum(x[ok].astype(np.float64), 0))
    corr = np.corrcoef(z, tr.y[ok])[0, 1] if ok.sum() > 2 and z.std() else 0
    qs = np.quantile(x[ok], [0, .5, .9, .99, 1]) if ok.any() else np.full(5, np.nan)
    print("NUM", k, "miss", f"{1-ok.mean():.3f}", "Q", np.round(qs, 2),
          "logcorr", f"{corr:.3f}")

try:
    h = historical_features("train", key="video_id")
    print("HIST video", ",".join(f"{k}:{v.shape}/{v.dtype}" for k, v in h.items()))
except Exception as e:
    print("HIST unavailable", type(e).__name__)

# Leakage-free utility: fit smoothed one-field target rates on early train,
# measure held-out last-three-day log-loss gain over an early-train constant.
days = np.unique(tr.date)
fit = tr.date < days[-3]
hold = ~fit
yf = tr.y[fit].astype(np.float64)
yh = tr.y[hold].astype(np.float64)
g = float(yf.mean())
eps = 1e-7
base = -np.mean(yh*np.log(g) + (1-yh)*np.log(1-g))
print("CAT name card activeT/V/E unseenV/E mode% temporal_logloss_gain_x1000")
for k in tr.X:
    card = FEATURE_CARDINALITIES[k]
    x = tr.X[k]
    cnt = np.bincount(x[fit], minlength=card).astype(np.float64)
    pos = np.bincount(x[fit], weights=yf, minlength=card)
    rate = (pos + 20*g) / (cnt + 20)
    p = np.clip(rate[x[hold]], eps, 1-eps)
    loss = -np.mean(yh*np.log(p) + (1-yh)*np.log(1-p))
    seen = np.bincount(x, minlength=card) > 0
    uv = np.mean(~seen[va.X[k]])
    ue = np.mean(~seen[te.X[k]])
    mode = np.bincount(x, minlength=card).max() / len(x)
    print("CAT", k, card, np.unique(x).size, np.unique(va.X[k]).size,
          np.unique(te.X[k]).size, f"{uv:.3f}/{ue:.3f}", f"{mode:.3f}",
          f"{1000*(base-loss):.2f}")