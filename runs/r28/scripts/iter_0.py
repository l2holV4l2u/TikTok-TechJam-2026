import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES

tr = load("train")
va = load("valid")
yt = np.asarray(tr.y, dtype=np.int8)
yv = np.asarray(va.y, dtype=np.int8)

print(f"ROWS train={len(yt)} valid={len(yv)} fields={len(tr.X)}")
print(f"LABEL rate train={yt.mean():.5f} valid={yv.mean():.5f} drift={yv.mean()-yt.mean():+.5f}")

def user_summary(tag, uid, y):
    _, inv, cnt = np.unique(uid, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=y, minlength=len(cnt))
    zero = pos == 0
    allp = pos == cnt
    mixed = ~(zero | allp)
    cq = np.percentile(cnt, [10, 50, 90, 99])
    pq = np.percentile(pos, [10, 50, 90, 99])
    wp = pos[mixed].sum() / max(pos.sum(), 1)
    print(f"USER {tag} n={len(cnt)} rows_q10/50/90/99={cq.astype(int).tolist()}")
    print(f"USER {tag} pos_q10/50/90/99={pq.round(1).tolist()} zero/all/mix%="
          f"{100*zero.mean():.1f}/{100*allp.mean():.1f}/{100*mixed.mean():.1f} gauc_pos_cov={wp:.3f}")

user_summary("train", np.asarray(tr.user_id), yt)
user_summary("valid", np.asarray(va.user_id), yv)

def overlap(name, a, b):
    ua = np.unique(a)
    ub = np.unique(b)
    hit_rows = np.isin(b, ua)
    hit_unique = np.isin(ub, ua)
    print(f"OVERLAP {name} trainU={len(ua)} validU={len(ub)} "
          f"valid_unseen_rows%={100*(~hit_rows).mean():.2f} "
          f"valid_unseen_ids%={100*(~hit_unique).mean():.2f}")

overlap("user", np.asarray(tr.user_id), np.asarray(va.user_id))
overlap("video", np.asarray(tr.video_id), np.asarray(va.video_id))

bad = []
for k in tr.X:
    a, b = np.asarray(tr.X[k]), np.asarray(va.X[k])
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(yt) or len(b) != len(yv):
        bad.append((k, a.shape, b.shape))
print(f"X_LAYOUT all_scalar_per_row={not bad} exceptions={bad}")
aux_desc = [f"{k}:{np.asarray(v).shape[1:] or 'scalar'}/{np.asarray(v).dtype}"
            for k, v in tr.aux.items()]
print("AUX " + ",".join(aux_desc))

p = float(yt.mean())
den = len(yt) * max(p * (1.0 - p), 1e-12)
for k in tr.X:
    x = np.asarray(tr.X[k], dtype=np.int64)
    z = np.asarray(va.X[k], dtype=np.int64)
    card = int(FEATURE_CARDINALITIES[k])
    size = max(card, int(x.max(initial=0)) + 1, int(z.max(initial=0)) + 1)
    cnt = np.bincount(x, minlength=size)
    ps = np.bincount(x, weights=yt, minlength=size)
    obs = cnt > 0
    rate = np.zeros(size, dtype=np.float64)
    rate[obs] = ps[obs] / cnt[obs]
    raw = float(np.sum(cnt[obs] * (rate[obs] - p) ** 2) / den)
    reliable = cnt >= 20
    e20 = float(np.sum(cnt[reliable] * (rate[reliable] - p) ** 2) / den)
    cov20 = float(cnt[reliable].sum() / len(x))
    newv = float(np.mean(cnt[z] == 0))
    dom = float(cnt.max(initial=0) / len(x))
    print(f"F {k} C{card} U{obs.sum()}/{np.unique(z).size} "
          f"Z{100*np.mean(x==0):.1f}/{100*np.mean(z==0):.1f} "
          f"N{100*newv:.1f} D{100*dom:.1f} E{raw:.3f}/{e20:.3f} Q{100*cov20:.0f}")