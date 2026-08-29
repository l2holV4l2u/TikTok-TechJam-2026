import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")

def qstr(a):
    a = np.asarray(a)
    return "/".join(f"{x:.1f}" for x in np.quantile(a, [0, .25, .5, .75, .9, .99, 1]))

def binary_mi_bits(x, y, card):
    n = len(y)
    cnt = np.bincount(x, minlength=card).astype(np.float64)
    pos = np.bincount(x, weights=y, minlength=card).astype(np.float64)
    neg = cnt - pos
    py1 = y.mean()
    py0 = 1.0 - py1
    mi = 0.0
    m = pos > 0
    mi += np.sum((pos[m] / n) * np.log2((pos[m] / cnt[m]) / py1))
    m = neg > 0
    mi += np.sum((neg[m] / n) * np.log2((neg[m] / cnt[m]) / py0))
    return mi

def entity_overlap(name):
    a = np.unique(tr.X[name])
    av = va.X[name]
    at = te.X[name]
    seen_v = np.isin(av, a)
    seen_t = np.isin(at, a)
    return (len(a), len(np.unique(av)), len(np.unique(at)),
            1 - seen_v.mean(), 1 - seen_t.mean())

print(f"ROWS train/valid/test={len(tr.y)}/{len(va.y)}/{len(te.X['user_id'])}")
print(f"LABEL train/valid rate={tr.y.mean():.4f}/{va.y.mean():.4f}")
print("DATES train=" + ",".join(
    f"{d}:{tr.y[tr.date == d].mean():.3f}" for d in np.unique(tr.date)))
print("DATES valid=" + ",".join(
    f"{d}:{va.y[va.date == d].mean():.3f}" for d in np.unique(va.date)))
print(f"TEST dates={int(te.date.min())}-{int(te.date.max())}")

uid = tr.X["user_id"]
n = np.bincount(uid)
p = np.bincount(uid, weights=tr.y)
keep = n > 0
n, p = n[keep], p[keep]
print(f"TRAIN users={len(n)} rows_q0/25/50/75/90/99/max={qstr(n)}")
print(f"TRAIN user_pos_q={qstr(p)} zero={np.mean(p==0):.3f} "
      f"allpos={np.mean(p==n):.3f} mixed={np.mean((p>0)&(p<n)):.3f}")

uid = va.X["user_id"]
n = np.bincount(uid)
p = np.bincount(uid, weights=va.y)
keep = n > 0
n, p = n[keep], p[keep]
print(f"VALID users={len(n)} rows_q={qstr(n)} pos_q={qstr(p)}")
print(f"VALID user zero={np.mean(p==0):.3f} allpos={np.mean(p==n):.3f} "
      f"mixed={np.mean((p>0)&(p<n)):.3f}")

for name in ["user_id", "video_id", "author_id"]:
    ut, uv, ue, cv, ct = entity_overlap(name)
    print(f"COLD {name} uniq_tr/va/te={ut}/{uv}/{ue} "
          f"unseen_row_va/te={cv:.3f}/{ct:.3f}")

sample_name = next(iter(tr.X))
print(f"SHAPE X[{sample_name}]={tr.X[sample_name].shape},{tr.X[sample_name].dtype} "
      f"y={tr.y.shape} date={tr.date.shape} time={tr.time_ms.shape}")
print("CAT columns: card,uniq_tr/va,zero_va,topshare_tr,MI_millibits")
rows = []
for name in sorted(tr.X):
    x = tr.X[name]
    xv = va.X[name]
    card = FEATURE_CARDINALITIES[name]
    counts = np.bincount(x, minlength=card)
    mi = binary_mi_bits(x, tr.y, card)
    rows.append((mi, name, card, np.count_nonzero(counts),
                 len(np.unique(xv)), np.mean(xv == 0), counts.max() / len(x)))
for mi, name, card, u, uv, z, top in sorted(rows, reverse=True):
    print(f"C {name} {card},{u}/{uv},{z:.3f},{top:.3f},{1000*mi:.3f}")

print("NUM columns: missing,median,p90,p99,point_biserial")
for name in sorted(tr.num):
    x = np.asarray(tr.num[name], dtype=np.float64)
    ok = np.isfinite(x)
    xx = np.log1p(np.maximum(x[ok], 0))
    yy = tr.y[ok].astype(np.float64)
    corr = np.corrcoef(xx, yy)[0, 1] if len(xx) > 1 and xx.std() > 0 else np.nan
    qq = np.quantile(x[ok], [.5, .9, .99])
    print(f"N {name} {1-ok.mean():.3f},{qq[0]:.2f},{qq[1]:.2f},"
          f"{qq[2]:.2f},{corr:.4f}")

for key in ["video_id", "author_id"]:
    h = historical_features("valid", key=key)
    desc = []
    for name, a in h.items():
        a = np.asarray(a)
        desc.append(f"{name}:{a.shape}/{a.dtype}/nan{np.mean(~np.isfinite(a)):.3f}")
    print(f"HIST {key} " + " ".join(desc))

same_time = np.mean(
    (va.X["user_id"][1:] == va.X["user_id"][:-1]) &
    (va.time_ms[1:] == va.time_ms[:-1])
)
print(f"TIME valid adjacent_same_user_timestamp={same_time:.4f} "
      f"range_ms={int(va.time_ms.min())}-{int(va.time_ms.max())}")