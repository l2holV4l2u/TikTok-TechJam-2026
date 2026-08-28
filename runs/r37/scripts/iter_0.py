import numpy as np
from pipeline.data import load, FEATURE_CARDINALITIES

tr = load("train")
va = load("valid")
lines = []

def date_summary(s):
    d, inv = np.unique(s.date, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=s.y)
    return " ".join(f"{str(int(x))[-4:]}:{int(nn)}/{pp/nn:.3f}"
                    for x, nn, pp in zip(d, n, p))

def group_summary(ids, y):
    _, inv = np.unique(ids, return_inverse=True)
    n = np.bincount(inv)
    p = np.bincount(inv, weights=y).astype(np.int64)
    qn = np.quantile(n, [0, .25, .5, .75, .9, .99, 1])
    qp = np.quantile(p, [0, .25, .5, .75, .9, .99, 1])
    mixed = (p > 0) & (p < n)
    pos_weight = p.sum()
    eligible_weight = p[mixed].sum() / pos_weight if pos_weight else 0
    return (
        len(n), n.mean(), qn, qp, np.mean(p == 0), np.mean(p == n),
        np.mean(mixed), eligible_weight
    )

def row_seen_fraction(train_ids, valid_ids):
    u = np.unique(train_ids)
    j = np.searchsorted(u, valid_ids)
    seen = j < len(u)
    seen[seen] &= (u[j[seen]] == valid_ids[seen])
    return seen.mean(), len(u), len(np.unique(valid_ids))

# Robust association ratio: categories with <50 rows are pooled together.
def cat_stats(x, y, configured):
    x = np.asarray(x)
    if x.ndim != 1 or len(x) != len(y):
        return None
    mx = int(x.max(initial=0))
    size = max(int(configured), mx + 1)
    cnt = np.bincount(x, minlength=size).astype(np.float64)
    pos = np.bincount(x, weights=y, minlength=size)
    used = cnt > 0
    top = cnt.max(initial=0) / len(x)
    p = float(np.mean(y))
    common = cnt >= 50
    between = np.sum(cnt[common] * (pos[common] / cnt[common] - p) ** 2)
    rn = cnt[~common].sum()
    if rn:
        rp = pos[~common].sum() / rn
        between += rn * (rp - p) ** 2
    denom = len(x) * p * (1 - p)
    eta = np.sqrt(between / denom) if denom > 0 else 0.0
    return int(used.sum()), float(top), float(eta), float(np.mean(x == 0))

lines.append(
    f"ROWS train={len(tr.y)} valid={len(va.y)} "
    f"features={len(tr.X)} yrate={tr.y.mean():.5f}/{va.y.mean():.5f}"
)
lines.append("DATES train " + date_summary(tr))
lines.append("DATES valid " + date_summary(va))

for name, ids_tr, ids_va in [
    ("USER", tr.user_id, va.user_id),
    ("VIDEO", tr.video_id, va.video_id),
]:
    seen, nu_tr, nu_va = row_seen_fraction(ids_tr, ids_va)
    lines.append(
        f"{name} unique={nu_tr}/{nu_va} valid_row_seen={seen:.4f} "
        f"cold={1-seen:.4f}"
    )

for split_name, s in [("TR_USER", tr), ("VA_USER", va)]:
    k, mean_n, qn, qp, z, a, mix, ew = group_summary(s.user_id, s.y)
    lines.append(
        f"{split_name} n={k} imp_mean={mean_n:.1f} "
        f"imp_q=({','.join(f'{v:.0f}' for v in qn)}) "
        f"pos_q=({','.join(f'{v:.0f}' for v in qp)})"
    )
    lines.append(
        f"{split_name} zero={z:.3f} allpos={a:.3f} mixed={mix:.3f} "
        f"GAUC_pos_weight_eligible={ew:.3f}"
    )

scalar = [k for k, v in tr.X.items()
          if np.asarray(v).ndim == 1 and len(v) == len(tr.y)]
nonscalar = [(k, np.asarray(v).shape, str(np.asarray(v).dtype))
             for k, v in tr.X.items() if k not in scalar]
lines.append(
    f"X scalar_per_row={len(scalar)}/{len(tr.X)} "
    f"nonscalar={nonscalar if nonscalar else 'none'}"
)

aux_desc = [
    f"{k}:{np.asarray(v).shape}/{np.asarray(v).dtype}"
    for k, v in tr.aux.items()
]
aux_text = " ".join(aux_desc)
if len(aux_text) > 420:
    aux_text = aux_text[:417] + "..."
lines.append(f"AUX outcome_only n={len(aux_desc)} {aux_text}")

lines.append("FIELD C Utr/Uva topTr zeroTr/V etaTr/V (eta pools count<50)")
for name in sorted(tr.X):
    a = np.asarray(tr.X[name])
    b = np.asarray(va.X[name])
    c = int(FEATURE_CARDINALITIES.get(name, max(
        int(a.max(initial=0)), int(b.max(initial=0))) + 1))
    st = cat_stats(a, tr.y, c)
    sv = cat_stats(b, va.y, c)
    if st is None or sv is None:
        lines.append(f"F {name} shape={a.shape}/{b.shape} C={c}")
    else:
        lines.append(
            f"F {name} C={c} U={st[0]}/{sv[0]} top={st[1]:.3f} "
            f"z={st[3]:.3f}/{sv[3]:.3f} eta={st[2]:.3f}/{sv[2]:.3f}"
        )

# Stay below the inspection channel's character budget without losing field order.
text = "\n".join(lines)
if len(text) > 3950:
    compact = []
    for x in lines:
        x = x.replace("0.", ".")
        compact.append(x)
    text = "\n".join(compact)
print(text[:3990])