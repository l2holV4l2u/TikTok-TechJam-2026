import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES
from pipeline.history import historical_features

tr = load("train")
va = load("valid")
te = load("test")

def date_summary(s, with_y):
    ds, cnt = np.unique(s.date, return_counts=True)
    if not with_y:
        return " ".join(f"{d}:{n}" for d, n in zip(ds, cnt))
    return " ".join(
        f"{d}:{n}/{s.y[s.date == d].mean():.3f}"
        for d, n in zip(ds, cnt)
    )

def user_summary(s):
    u, inv = np.unique(s.user_id, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    qn = np.quantile(n, [.1, .5, .9, .99])
    qp = np.quantile(p, [.1, .5, .9, .99])
    return (len(u), qn, qp, np.mean(p == 0), np.mean(p == n))

def row_unseen(train_values, other_values, cardinality):
    seen = np.zeros(cardinality, dtype=bool)
    seen[np.unique(train_values)] = True
    return np.mean(~seen[other_values])

def distinct_overlap(train_values, other_values, cardinality):
    seen = np.zeros(cardinality, dtype=bool)
    seen[np.unique(train_values)] = True
    u = np.unique(other_values)
    return np.mean(seen[u])

def categorical_mi_corrected(x, y, cardinality):
    n = len(y)
    cnt = np.bincount(x, minlength=cardinality).astype(np.float64)
    pos = np.bincount(x, weights=y, minlength=cardinality)
    neg = cnt - pos
    P = float(y.sum())
    N = n - P
    mi = 0.0
    m = pos > 0
    mi += np.sum((pos[m] / n) * np.log2((pos[m] * n) / (cnt[m] * P)))
    m = neg > 0
    mi += np.sum((neg[m] / n) * np.log2((neg[m] * n) / (cnt[m] * N)))
    k = np.count_nonzero(cnt)
    correction = (k - 1) / (2.0 * n * np.log(2.0))
    return max(0.0, mi - correction), cnt, pos

def supported_rate_sd(cnt, pos, min_count=50):
    m = cnt >= min_count
    if not np.any(m):
        return np.nan, 0.0
    rates = pos[m] / cnt[m]
    w = cnt[m]
    mean = np.average(rates, weights=w)
    sd = np.sqrt(np.average((rates - mean) ** 2, weights=w))
    return sd, w.sum() / cnt.sum()

def varying_user_fraction(user, x, cardinality):
    pair = user.astype(np.int64) * np.int64(cardinality) + x.astype(np.int64)
    pairs = np.unique(pair)
    pair_users = pairs // np.int64(cardinality)
    _, counts = np.unique(pair_users, return_counts=True)
    return np.mean(counts > 1)

print(f"SHAPE tr={len(tr.y)} va={len(va.y)} te={len(te.user_id)} fields={len(tr.X)} num={len(tr.num)}")
print(f"ARRAY X_example={next(iter(tr.X.values())).shape}/{next(iter(tr.X.values())).dtype} "
      f"y={tr.y.shape}/{tr.y.dtype} time={tr.time_ms.shape}/{tr.time_ms.dtype}")
print(f"LABEL tr={tr.y.mean():.5f} va={va.y.mean():.5f}")
print("DATES_TR date:n/rate " + date_summary(tr, True))
print("DATES_VA date:n/rate " + date_summary(va, True))
print("DATES_TE date:n " + date_summary(te, False))

for name, s in [("TR", tr), ("VA", va)]:
    nu, qn, qp, zero, allpos = user_summary(s)
    print(f"USER_{name} n={nu} rows_q={qn.astype(int).tolist()} "
          f"pos_q={np.round(qp,1).tolist()} zero={zero:.3f} all={allpos:.3f}")

for name in ["user_id", "video_id", "author_id"]:
    c = FEATURE_CARDINALITIES[name]
    print(f"COLD_{name} va_row={row_unseen(tr.X[name], va.X[name], c):.3f} "
          f"va_id={1-distinct_overlap(tr.X[name], va.X[name], c):.3f} "
          f"te_row={row_unseen(tr.X[name], te.X[name], c):.3f} "
          f"te_id={1-distinct_overlap(tr.X[name], te.X[name], c):.3f}")

print("FIELD columns: k=train/valid/test|configured u=unseen-row-valid/test "
      "top=train-mode-share mi=bias-corrected-bits sd=rate-SD(count>=50)/coverage "
      "mv=valid-users-with->1-value")
for name in tr.X:
    c = FEATURE_CARDINALITIES[name]
    xt, xv, xe = tr.X[name], va.X[name], te.X[name]
    mi, cnt, pos = categorical_mi_corrected(xt, tr.y, c)
    sd, cov = supported_rate_sd(cnt, pos)
    kt, kv, ke = len(np.unique(xt)), len(np.unique(xv)), len(np.unique(xe))
    uv = row_unseen(xt, xv, c)
    ue = row_unseen(xt, xe, c)
    top = cnt.max() / len(xt)
    mv = varying_user_fraction(va.user_id, xv, c)
    print(f"F {name} k={kt}/{kv}/{ke}|{c} u={uv:.2f}/{ue:.2f} "
          f"top={top:.2f} mi={mi:.4f} sd={sd:.3f}/{cov:.2f} mv={mv:.2f}")

print("NUM columns: missing train/valid, train finite q01/q50/q99, corr(log1p,label)")
for name in tr.num:
    a = tr.num[name].astype(np.float64)
    b = va.num[name]
    finite = np.isfinite(a)
    q = np.quantile(a[finite], [.01, .5, .99])
    z = np.log1p(np.maximum(a[finite], 0))
    corr = np.corrcoef(z, tr.y[finite])[0, 1] if np.std(z) > 0 else 0.0
    print(f"N {name} miss={1-finite.mean():.3f}/{np.mean(~np.isfinite(b)):.3f} "
          f"q={np.round(q,2).tolist()} corr={corr:.3f}")

for key in ["video_id", "author_id"]:
    h = historical_features("train", key=key)
    shapes = sorted(set((tuple(v.shape), str(v.dtype)) for v in h.values()))
    print(f"HIST_{key} keys={','.join(sorted(h.keys()))}")
    print(f"HIST_{key} representations={shapes}")