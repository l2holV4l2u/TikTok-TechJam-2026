import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")

def split_summary(name, s, with_y):
    u, cnt = np.unique(s.user_id, return_counts=True)
    msg = f"{name}: n={len(s.user_id)} users={len(u)} videos={np.unique(s.video_id).size}"
    if with_y:
        y = s.y.astype(np.float64)
        msg += f" pos={y.mean():.4f}"
    print(msg)

def user_stats(name, s):
    u, inv, cnt = np.unique(s.user_id, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=s.y, minlength=len(u))
    qn = np.quantile(cnt, [0, .25, .5, .75, .9, .99]).astype(int)
    qp = np.quantile(pos, [0, .25, .5, .75, .9, .99])
    print(f"{name}_user rows_q={qn.tolist()} pos_q={np.round(qp,2).tolist()} "
          f"zero={np.mean(pos==0):.3f} all={np.mean(pos==cnt):.3f}")

def daily(s):
    vals = []
    for d in np.unique(s.date):
        z = s.date == d
        vals.append(f"{int(d)%10000:04d}:{z.sum()}/{s.y[z].mean():.3f}")
    return " ".join(vals)

def entity_overlap(name, a, b, c):
    seen = np.zeros(int(max(a.max(), b.max(), c.max())) + 1, dtype=bool)
    seen[a] = True
    rv = np.mean(~seen[b])
    rt = np.mean(~seen[c])
    print(f"overlap {name}: uniq={np.unique(a).size}/{np.unique(b).size}/{np.unique(c).size} "
          f"unseen_row V={rv:.3f} T={rt:.3f}")

def entropy01(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def cat_info(x, y, cardinality):
    n = np.bincount(x, minlength=cardinality).astype(np.float64)
    p = np.bincount(x, weights=y, minlength=cardinality).astype(np.float64)
    common = n >= 50
    nc, pc = n[common], p[common]
    nr, pr = n[~common].sum(), p[~common].sum()
    if nr:
        nc, pc = np.r_[nc, nr], np.r_[pc, pr]
    rates = np.divide(pc, nc, out=np.zeros_like(pc), where=nc > 0)
    cond = np.sum(nc * np.array([entropy01(v) for v in rates])) / len(x)
    return entropy01(y.mean()) - cond, n

def hist_report(key):
    try:
        h = historical_features("train", key=key)
        print(f"history {key} keys={','.join(h.keys())}"[:650])
        rows = []
        yy = tr.y.astype(np.float64)
        for k, v in h.items():
            a = np.asarray(v)
            if a.ndim != 1 or len(a) != len(yy):
                continue
            ok = np.isfinite(a)
            corr = 0.0
            if ok.sum() > 2 and np.std(a[ok]) > 0:
                corr = float(np.corrcoef(a[ok], yy[ok])[0, 1])
            rows.append((abs(corr), k, corr, 1-ok.mean()))
        rows.sort(reverse=True)
        print("history_top " + " ".join(
            f"{k}:r={r:+.3f},nan={m:.2f}" for _, k, r, m in rows[:8]))
    except Exception as e:
        print(f"history {key} ERROR={type(e).__name__}:{str(e)[:120]}")

print(f"schema X_fields={len(tr.X)} num_fields={len(tr.num)} "
      f"X_scalar={all(np.asarray(v).ndim==1 and len(v)==len(tr.user_id) for v in tr.X.values())} "
      f"dtypes={sorted(set(str(v.dtype) for v in tr.X.values()))}")
split_summary("train", tr, True)
split_summary("valid", va, True)
split_summary("test", te, False)
print("train_day " + daily(tr))
print("valid_day " + daily(va))
user_stats("train", tr)
user_stats("valid", va)

entity_overlap("user", tr.user_id, va.user_id, te.user_id)
entity_overlap("video", tr.video_id, va.video_id, te.video_id)
entity_overlap("author", tr.X["author_id"], va.X["author_id"], te.X["author_id"])

y = tr.y.astype(np.float64)
field_rows = []
for name in tr.X:
    card = FEATURE_CARDINALITIES[name]
    mi, counts = cat_info(tr.X[name], y, card)
    seen = counts > 0
    uv = float(np.mean(~seen[va.X[name]]))
    ut = float(np.mean(~seen[te.X[name]]))
    obs = int(seen.sum())
    dom = float(counts.max() / len(y))
    field_rows.append((mi, name, obs, card, dom, uv, ut))

field_rows.sort(reverse=True)
print("cat legend: MI=rare-pooled(>=50) train bits; obs/card; dom; unseen-row V/T")
for mi, name, obs, card, dom, uv, ut in field_rows[:15]:
    print(f"cat {name}: MI={mi:.5f} {obs}/{card} dom={dom:.3f} unseen={uv:.3f}/{ut:.3f}")
print("cat_rank_all " + ",".join(f"{name}:{mi:.4g}" for mi, name, *_ in field_rows))

for name, a0 in tr.num.items():
    a = np.asarray(a0, dtype=np.float64)
    ok = np.isfinite(a)
    q = np.quantile(a[ok], [.1, .5, .9, .99]) if ok.any() else np.full(4, np.nan)
    corr = 0.0
    if ok.sum() > 2:
        z = np.log1p(np.maximum(a[ok], 0))
        if np.std(z) > 0:
            corr = float(np.corrcoef(z, y[ok])[0, 1])
    vmiss = np.mean(~np.isfinite(np.asarray(va.num[name])))
    print(f"num {name}: missT/V={1-ok.mean():.3f}/{vmiss:.3f} "
          f"q={np.round(q,2).tolist()} logcorr={corr:+.3f}")

hist_report("video_id")
hist_report("author_id")