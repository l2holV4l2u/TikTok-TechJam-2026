import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")

print("SHAPES train=%d valid=%d test=%d fields=%d num=%d" %
      (len(tr.user_id), len(va.user_id), len(te.user_id), len(tr.X), len(tr.num)))
print("SCALAR_CHECK " + " ".join(
    "%s:%s" % (k, tuple(v.shape)) for k, v in list(tr.X.items())[:5]))


def user_summary(name, s):
    u, inv, cnt = np.unique(s.user_id, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=s.y, minlength=len(u))
    qn = np.quantile(cnt, [0, .25, .5, .75, .9, .99, 1])
    qp = np.quantile(pos, [0, .25, .5, .75, .9, .99, 1])
    print("%s y=%.4f users=%d impQ=%s posQ=%s zero=%.1f%% all=%.1f%% mixed=%.1f%%" %
          (name, s.y.mean(), len(u),
           np.array2string(qn, precision=0, separator=","),
           np.array2string(qp, precision=0, separator=","),
           100*np.mean(pos == 0), 100*np.mean(pos == cnt),
           100*np.mean((pos > 0) & (pos < cnt))))


user_summary("TRAIN", tr)
user_summary("VALID", va)

for name, s in [("TR_DATE", tr), ("VA_DATE", va)]:
    d, inv = np.unique(s.date, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y) / n
    print(name + " " + " ".join("%d:%.3f" % (x, y) for x, y in zip(d, p)))


def novelty(name, a, b, c):
    ua = np.unique(a)
    bv = ~np.isin(b, ua)
    ct = ~np.isin(c, ua)
    print("NOVEL %-10s uniqTr=%d validRows=%.1f%% validUniq=%.1f%% testRows=%.1f%% testUniq=%.1f%%" %
          (name, len(ua), 100*bv.mean(),
           100*np.mean(~np.isin(np.unique(b), ua)), 100*ct.mean(),
           100*np.mean(~np.isin(np.unique(c), ua))))


novelty("user_id", tr.user_id, va.user_id, te.user_id)
novelty("video_id", tr.video_id, va.video_id, te.video_id)
novelty("author_id", tr.X["author_id"], va.X["author_id"], te.X["author_id"])


def mutual_info_bits(x, y, card):
    n = len(y)
    total = np.bincount(x, minlength=card).astype(np.float64)
    pos = np.bincount(x, weights=y, minlength=card).astype(np.float64)
    neg = total - pos
    py1 = y.mean()
    py0 = 1.0 - py1
    mi = 0.0
    for cell, py in ((pos, py1), (neg, py0)):
        ok = cell > 0
        pc = cell[ok] / n
        px = total[ok] / n
        mi += np.sum(pc * np.log2(pc / (px * py)))
    return mi


rows = []
for f in tr.X:
    x = tr.X[f]
    card = FEATURE_CARDINALITIES[f]
    used = np.unique(x).size
    cnt = np.bincount(x, minlength=card)
    dom = cnt.max() / len(x)
    seen = cnt > 0
    vv = va.X[f]
    tt = te.X[f]
    uv = np.mean((vv >= len(seen)) | ~seen[np.minimum(vv, len(seen)-1)])
    ut = np.mean((tt >= len(seen)) | ~seen[np.minimum(tt, len(seen)-1)])
    mi = mutual_info_bits(x, tr.y, card)
    rows.append((mi, f, used, card, dom, uv, ut))

print("CATEGORICAL sorted by train mutual-information; mi=milli-bits")
for mi, f, used, card, dom, uv, ut in sorted(rows, reverse=True):
    print("C %-25s k=%d/%d dom=%4.1f uv=%4.1f ut=%4.1f mi=%7.3f" %
          (f, used, card, 100*dom, 100*uv, 100*ut, 1000*mi))

for f, x in tr.num.items():
    z = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(z)
    zz = np.log1p(np.maximum(z[ok], 0))
    yy = tr.y[ok].astype(np.float64)
    corr = np.corrcoef(zz, yy)[0, 1] if zz.std() > 0 else 0.0
    q = np.quantile(z[ok], [.01, .5, .99])
    print("N %-24s miss=%4.1f%% q01/50/99=%g,%g,%g logcorr=%+.4f" %
          (f, 100*(~ok).mean(), q[0], q[1], q[2], corr))

for key in ("video_id", "author_id"):
    h = historical_features("valid", key=key)
    desc = ",".join("%s:%s" % (k, tuple(np.asarray(v).shape)) for k, v in h.items())
    print("HIST %s keys=%d %s" % (key, len(h), desc[:500]))