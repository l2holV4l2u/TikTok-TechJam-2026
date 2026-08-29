import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")

print(f"ROWS tr={len(tr.user_id)} va={len(va.user_id)} te={len(te.user_id)}")
print(f"X fields={len(tr.X)} num={len(tr.num)} scalar_shapes="
      f"{all(np.asarray(x).shape == (len(tr.user_id),) for x in tr.X.values())}")

def user_summary(s, name):
    users, inv = np.unique(s.user_id, return_inverse=True)
    nr = np.bincount(inv)
    np_ = np.bincount(inv, weights=s.y)
    rq = np.quantile(nr, [0.25, 0.5, 0.75, 0.95])
    pq = np.quantile(np_, [0.25, 0.5, 0.75, 0.95])
    print(f"USER {name} n={len(users)} y={s.y.mean():.4f} no+={(np_==0).mean():.3f} "
          f"rowsq={rq.astype(int).tolist()} posq={pq.astype(int).tolist()}")

user_summary(tr, "tr")
user_summary(va, "va")

def day_line(s, name):
    ds = np.unique(s.date)
    vals = [f"{int(d)%100:02d}:{s.y[s.date==d].mean():.3f}" for d in ds]
    print(f"DAY {name} " + " ".join(vals))

day_line(tr, "tr")
day_line(va, "va")

def novelty(key):
    a = np.asarray(tr.X[key])
    b = np.asarray(va.X[key])
    c = np.asarray(te.X[key])
    C = FEATURE_CARDINALITIES[key]
    seen = np.bincount(a, minlength=C) > 0
    print(f"NOVEL {key} vaRows={np.mean(~seen[b]):.3f} teRows={np.mean(~seen[c]):.3f} "
          f"vaIDs={np.mean(~seen[np.unique(b)]):.3f} teIDs={np.mean(~seen[np.unique(c)]):.3f}")

for key in ("user_id", "video_id", "author_id"):
    novelty(key)

p0 = float(np.clip(tr.y.mean(), 1e-6, 1 - 1e-6))
base_ll = -np.mean(va.y * np.log(p0) + (1 - va.y) * np.log(1 - p0))
alpha = 20.0
print("CAT name C usedT/V/E unseenV/E topT dLL(train-smoothed->valid)")

for name in tr.X:
    C = FEATURE_CARDINALITIES[name]
    xt = np.asarray(tr.X[name])
    xv = np.asarray(va.X[name])
    xe = np.asarray(te.X[name])
    cnt = np.bincount(xt, minlength=C)
    pos = np.bincount(xt, weights=tr.y, minlength=C)
    rate = (pos + alpha * p0) / (cnt + alpha)
    pred = np.clip(rate[xv], 1e-6, 1 - 1e-6)
    ll = -np.mean(va.y * np.log(pred) + (1 - va.y) * np.log(1 - pred))
    cv = np.bincount(xv, minlength=C)
    ce = np.bincount(xe, minlength=C)
    unseen_v = np.mean(cnt[xv] == 0)
    unseen_e = np.mean(cnt[xe] == 0)
    top = cnt.max() / len(xt)
    print(f"C {name} {C} {np.count_nonzero(cnt)}/{np.count_nonzero(cv)}/"
          f"{np.count_nonzero(ce)} {unseen_v:.3f}/{unseen_e:.3f} "
          f"{top:.3f} {base_ll-ll:+.5f}")

print("NUM name missT/V median p99 dLL(20-bin train->valid)")
for name in tr.num:
    x = np.asarray(tr.num[name], dtype=np.float64)
    z = np.asarray(va.num[name], dtype=np.float64)
    good = np.isfinite(x)
    if good.any():
        edges = np.unique(np.quantile(x[good], np.linspace(0, 1, 21)))
        bt = np.searchsorted(edges[1:-1], x, side="right") + 1
        bv = np.searchsorted(edges[1:-1], z, side="right") + 1
        bt[~good] = 0
        bv[~np.isfinite(z)] = 0
        nb = len(edges) + 1
        cnt = np.bincount(bt, minlength=nb)
        pos = np.bincount(bt, weights=tr.y, minlength=nb)
        rate = (pos + alpha * p0) / (cnt + alpha)
        pred = np.clip(rate[bv], 1e-6, 1 - 1e-6)
        ll = -np.mean(va.y * np.log(pred) + (1-va.y) * np.log(1-pred))
        med, p99 = np.nanquantile(x, [0.5, 0.99])
        print(f"N {name} {np.mean(~good):.3f}/{np.mean(~np.isfinite(z)):.3f} "
              f"{med:.3g} {p99:.3g} {base_ll-ll:+.5f}")

for key in ("video_id", "author_id"):
    h = historical_features("valid", key=key)
    desc = ",".join(f"{k}:{np.asarray(v).shape}/{np.asarray(v).dtype}" for k, v in h.items())
    print(f"HIST {key} {desc}"[:500])