import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")

def pct(a, qs=(0, 25, 50, 75, 90, 99, 100)):
    return ",".join(f"{q}:{v:.0f}" for q, v in zip(qs, np.percentile(a, qs)))

print(f"ROWS train={len(tr.y)} valid={len(va.y)} test={len(te.user_id)}")
print(f"X fields={len(tr.X)} sample_shape={next(iter(tr.X.values())).shape} "
      f"dtype={next(iter(tr.X.values())).dtype} num_fields={list(tr.num)}")
print(f"DATES train={np.unique(tr.date).tolist()} valid={np.unique(va.date).tolist()} "
      f"test={np.unique(te.date).tolist()}")

for name, s in [("train", tr), ("valid", va)]:
    users, inv = np.unique(s.user_id, return_inverse=True)
    nr = np.bincount(inv)
    np_ = np.bincount(inv, weights=np.asarray(s.y, dtype=np.float64))
    eligible = np.mean((np_ > 0) & (np_ < nr))
    print(f"LABEL {name} rate={np.mean(s.y):.5f} users={len(users)} "
          f"zero_pos={np.mean(np_==0):.4f} auc_eligible={eligible:.4f}")
    print(f"USER_ROWS {name} {pct(nr)}")
    print(f"USER_POS {name} {pct(np_)}")

for name, s in [("train", tr), ("valid", va)]:
    vals = []
    for d in np.unique(s.date):
        m = s.date == d
        vals.append(f"{d%10000:04d}:{np.mean(s.y[m]):.4f}")
    print(f"DATE_RATE {name} " + " ".join(vals))

def entity_coverage(field):
    a, b, c = tr.X[field], va.X[field], te.X[field]
    card = FEATURE_CARDINALITIES[field]
    seen = np.zeros(card, dtype=bool)
    seen[np.unique(a)] = True
    cnt = np.bincount(a, minlength=card)
    used = int(np.count_nonzero(cnt))
    dom = float(cnt.max() / len(a))
    uv = float(np.mean(~seen[b]))
    ut = float(np.mean(~seen[c]))
    return card, used, dom, uv, ut

for f in ["user_id", "video_id", "author_id", "tag", "music_type"]:
    card, used, dom, uv, ut = entity_coverage(f)
    print(f"COVER {f} card={card} train_ids={used} dominant={dom:.4f} "
          f"unseen_rows valid={uv:.4f} test={ut:.4f}")

days = np.unique(tr.date)
fit_mask = tr.date < days[-3]
hold_mask = ~fit_mask
yf = np.asarray(tr.y[fit_mask], dtype=np.float64)
yh = np.asarray(tr.y[hold_mask], dtype=np.float64)
prior = np.clip(yf.mean(), 1e-6, 1 - 1e-6)
base_ll = -np.mean(yh*np.log(prior) + (1-yh)*np.log1p(-prior))
field_stats = []

for f, x in tr.X.items():
    card = FEATURE_CARDINALITIES[f]
    xf = x[fit_mask]
    cnt = np.bincount(xf, minlength=card).astype(np.float64)
    pos = np.bincount(xf, weights=yf, minlength=card)
    pred = (pos + 20.0*prior) / (cnt + 20.0)
    ph = np.clip(pred[x[hold_mask]], 1e-6, 1 - 1e-6)
    ll = -np.mean(yh*np.log(ph) + (1-yh)*np.log1p(-ph))
    fullcnt = np.bincount(x, minlength=card)
    used = np.count_nonzero(fullcnt)
    dominant = fullcnt.max()/len(x)
    seen = fullcnt > 0
    unseen_v = np.mean(~seen[va.X[f]])
    field_stats.append((base_ll-ll, f, card, used, dominant, unseen_v))

field_stats.sort(reverse=True)
fmt = lambda z: f"{z[1]}:{1e4*z[0]:+.1f}/{z[3]}/{z[4]:.2f}/{100*z[5]:.1f}%"
print("CAT_TE_GAIN_TOP gain1e4/ids/dom/unseenV " +
      " ".join(fmt(z) for z in field_stats[:12]))
print("CAT_TE_GAIN_BOTTOM gain1e4/ids/dom/unseenV " +
      " ".join(fmt(z) for z in field_stats[-10:]))

coverage_rank = sorted(field_stats, key=lambda z: z[5], reverse=True)
print("CAT_UNSEEN_TOP " + " ".join(
    f"{z[1]}:{100*z[5]:.2f}%" for z in coverage_rank[:10]))
constant_rank = sorted(field_stats, key=lambda z: z[4], reverse=True)
print("CAT_DOMINANT_TOP " + " ".join(
    f"{z[1]}:{100*z[4]:.1f}%" for z in constant_rank[:10]))

for f, a in tr.num.items():
    a = np.asarray(a, dtype=np.float64)
    finite = np.isfinite(a)
    neg = finite & (tr.y == 0)
    pos = finite & (tr.y == 1)
    q = np.nanpercentile(a, [1, 50, 99])
    print(f"NUM {f} missing={1-finite.mean():.3f} q1/50/99="
          f"{q[0]:.2g}/{q[1]:.2g}/{q[2]:.2g} "
          f"mean0/1={a[neg].mean():.3g}/{a[pos].mean():.3g}")

for key in ["video_id", "author_id"]:
    h = historical_features("train", key=key)
    summaries = []
    y = np.asarray(tr.y, dtype=np.float64)
    for name, arr in h.items():
        arr = np.asarray(arr)
        if arr.ndim != 1 or len(arr) != len(y):
            summaries.append(f"{name}:shape{arr.shape}")
            continue
        ok = np.isfinite(arr)
        if not np.any(ok):
            summaries.append(f"{name}:all_nan")
            continue
        x = arr[ok].astype(np.float64)
        yy = y[ok]
        sx, sy = x.std(), yy.std()
        corr = 0.0 if sx == 0 or sy == 0 else np.mean((x-x.mean())*(yy-yy.mean()))/(sx*sy)
        summaries.append(f"{name}:{corr:+.3f}/{1-ok.mean():.2f}")
    print(f"HISTORY {key} corr/missing " + " ".join(summaries))

print("AUX_KEYS outcome_only=" + ",".join(sorted(tr.aux.keys())))