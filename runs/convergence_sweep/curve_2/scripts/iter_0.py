import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")

def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return 0.0
    a = a[ok] - a[ok].mean()
    b = b[ok] - b[ok].mean()
    den = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / den) if den > 0 else 0.0

def user_summary(s, name):
    _, inv = np.unique(s.user_id, return_inverse=True)
    rows = np.bincount(inv)
    pos = np.bincount(inv, weights=s.y)
    rq = np.quantile(rows, [0.25, 0.5, 0.75, 0.9])
    pq = np.quantile(pos, [0.5, 0.75, 0.9])
    zero = np.mean(pos == 0)
    mixed = np.mean((pos > 0) & (pos < rows))
    print(f"{name} n={len(s.y)} y={s.y.mean():.4f} users={len(rows)} "
          f"rowsQ={rq.astype(int).tolist()} posQ={pq.tolist()} "
          f"zero={zero:.3f} mixed={mixed:.3f}")

def date_summary(s, name):
    ds = []
    for d in np.unique(s.date):
        m = s.date == d
        ds.append(f"{int(d)%10000}:{m.sum()}/{s.y[m].mean():.3f}")
    print(name + "_dates " + " ".join(ds))

def overlap(name, a, b):
    seen = np.unique(a)
    row_seen = np.isin(b, seen)
    ub = np.unique(b)
    unique_seen = np.isin(ub, seen)
    print(f"OV {name} trU={len(seen)} vaU={len(ub)} "
          f"vaRowUn={1-row_seen.mean():.3f} vaUniUn={1-unique_seen.mean():.3f}")

print(f"SHAPE train={len(tr.user_id)} valid={len(va.user_id)} "
      f"X={len(tr.X)} num={len(tr.num)} aux={len(tr.aux)}")
user_summary(tr, "TR")
user_summary(va, "VA")
date_summary(tr, "TR")
date_summary(va, "VA")
overlap("user", tr.user_id, va.user_id)
overlap("video", tr.video_id, va.video_id)
overlap("author", tr.X["author_id"], va.X["author_id"])

scalar_ok = all(np.asarray(x).shape == (len(tr.user_id),) for x in tr.X.values())
print(f"XSCALAR={scalar_ok} sample_dtype={tr.X['user_id'].dtype} "
      f"time={tr.time_ms.shape}/{tr.time_ms.dtype}")

base = float(tr.y.mean())
feature_rows = []
for name in sorted(tr.X):
    xt = np.asarray(tr.X[name], dtype=np.int64)
    xv = np.asarray(va.X[name], dtype=np.int64)
    card = int(FEATURE_CARDINALITIES[name])
    cnt = np.bincount(xt, minlength=card).astype(np.float64)
    pos = np.bincount(xt, weights=tr.y, minlength=card)
    rate = (pos + 20.0 * base) / (cnt + 20.0)
    pred = rate[xv]
    seen_t = int(np.count_nonzero(cnt))
    seen_v = int(np.unique(xv).size)
    unseen = float(np.mean(cnt[xv] == 0))
    dominant = float(cnt.max() / len(xt))
    r = corr(pred, va.y)
    feature_rows.append((name, card, seen_t, seen_v, unseen, dominant, r))

for name, card, st, sv, unseen, dominant, r in feature_rows:
    print(f"F {name} C={card} T={st} V={sv} "
          f"U={unseen:.3f} D={dominant:.3f} R={r:+.3f}")

for name in sorted(tr.num):
    x = np.asarray(tr.num[name], dtype=np.float64)
    xv = np.asarray(va.num[name], dtype=np.float64)
    finite = np.isfinite(x)
    q = np.quantile(x[finite], [0.1, 0.5, 0.9]) if finite.any() else [np.nan] * 3
    z = np.log1p(np.maximum(xv, 0))
    print(f"N {name} missT={1-finite.mean():.3f} "
          f"missV={np.mean(~np.isfinite(xv)):.3f} "
          f"Q={np.round(q,1).tolist()} RlogV={corr(z,va.y):+.3f}")

for key in ("video_id", "author_id"):
    h = historical_features("valid", key=key)
    desc = ",".join(f"{k}:{np.asarray(v).shape}/{np.asarray(v).dtype}"
                    for k, v in h.items())
    print(f"H {key} n={len(h)} {desc[:260]}")

aux_names = ",".join(sorted(tr.aux))
print(f"AUX outcomes_only n={len(tr.aux)} names={aux_names[:220]}")