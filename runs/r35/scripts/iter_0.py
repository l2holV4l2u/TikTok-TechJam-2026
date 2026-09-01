import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES

tr = load("train")
va = load("valid")

def qstr(a):
    q = np.quantile(a, [0, .25, .5, .75, .9, .99, 1])
    return "/".join(f"{x:.0f}" for x in q)

def grouped_stats(ids, y, tag):
    _, inv, cnt = np.unique(ids, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=y, minlength=len(cnt))
    zero = np.mean(pos == 0) * 100
    allp = np.mean(pos == cnt) * 100
    mixed = np.mean((pos > 0) & (pos < cnt)) * 100
    print(f"{tag}_GROUPS n={len(cnt)} rowsQ={qstr(cnt)} posQ={qstr(pos)} "
          f"zero={zero:.1f}% all={allp:.1f}% mixed={mixed:.1f}%")
    return cnt, pos

def overlap_line(name, train_ids, valid_ids):
    ut = np.unique(train_ids)
    uv = np.unique(valid_ids)
    unseen_rows = np.mean(~np.isin(valid_ids, ut, assume_unique=False)) * 100
    unseen_entities = np.mean(~np.isin(uv, ut, assume_unique=True)) * 100
    print(f"{name}_OVERLAP trainU={len(ut)} validU={len(uv)} "
          f"validUnseenRows={unseen_rows:.2f}% validUnseenEntities={unseen_entities:.2f}%")

def date_line(split, tag):
    parts = []
    for d in np.unique(split.date):
        m = split.date == d
        parts.append(f"{int(d)%10000:04d}:{m.sum()}/{split.y[m].mean():.3f}")
    print(f"{tag}_DATE n/rate " + " ".join(parts))

print(f"ROWS train={len(tr.y)} valid={len(va.y)} fields={len(tr.X)}")
print(f"LABEL trainPos={tr.y.sum()} rate={tr.y.mean():.5f} "
      f"validPos={va.y.sum()} rate={va.y.mean():.5f}")
print(f"ARRAYS Xscalar={all(np.asarray(v).ndim == 1 and len(v) == len(tr.y) for v in tr.X.values())} "
      f"Xdtypes={sorted({str(np.asarray(v).dtype) for v in tr.X.values()})} "
      f"user={tr.user_id.shape}/{tr.user_id.dtype} video={tr.video_id.shape}/{tr.video_id.dtype} "
      f"date={tr.date.shape}/{tr.date.dtype}")
aux_names = sorted(tr.aux)
print(f"AUX outcomesOnly count={len(aux_names)} names=" + ",".join(aux_names)[:500])

grouped_stats(tr.user_id, tr.y, "TRAIN_USER")
grouped_stats(va.user_id, va.y, "VALID_USER")
grouped_stats(tr.video_id, tr.y, "TRAIN_VIDEO")
grouped_stats(va.video_id, va.y, "VALID_VIDEO")
overlap_line("USER", tr.user_id, va.user_id)
overlap_line("VIDEO", tr.video_id, va.video_id)
date_line(tr, "TRAIN")
date_line(va, "VALID")

n = len(tr.y)
ny1 = float(tr.y.sum())
ny0 = float(n - ny1)
field_rows = []

for name in tr.X:
    x = np.asarray(tr.X[name], dtype=np.int64)
    xv = np.asarray(va.X[name], dtype=np.int64)
    mx = int(max(x.max(initial=0), xv.max(initial=0)))
    cnt = np.bincount(x, minlength=mx + 1).astype(np.float64)
    pos = np.bincount(x, weights=tr.y, minlength=mx + 1)
    neg = cnt - pos

    mi = 0.0
    if ny1 > 0:
        m = pos > 0
        mi += np.sum((pos[m] / n) * np.log2((pos[m] * n) / (cnt[m] * ny1)))
    if ny0 > 0:
        m = neg > 0
        mi += np.sum((neg[m] / n) * np.log2((neg[m] * n) / (cnt[m] * ny0)))

    observed = cnt > 0
    unseen = np.mean(~observed[xv]) * 100
    top = cnt.max(initial=0) / n * 100
    singleton_rows = np.sum(cnt == 1) / n * 100
    zero_share = np.mean(x == 0) * 100
    field_rows.append((
        mi, name, FEATURE_CARDINALITIES.get(name, mx + 1),
        int(observed.sum()), int(np.unique(xv).size),
        unseen, zero_share, top, singleton_rows
    ))

print("FIELDS sorted_by_empirical_MI: C=config T/V=observed U=valid-unseen-row "
      "Z=id0 Q=top-category S=singleton-row M=MIbits")
for mi, name, card, nt, nv, unseen, zero, top, singleton in sorted(field_rows, reverse=True):
    print(f"F {name} C{card} T{nt} V{nv} U{unseen:.1f} Z{zero:.1f} "
          f"Q{top:.1f} S{singleton:.1f} M{mi:.5f}")