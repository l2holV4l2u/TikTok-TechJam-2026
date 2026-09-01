"""User x item-attribute affinity, fit on train only (leave-one-out on train rows).

The metric is a per-user mean, so features constant within a user cannot move it. Only
terms that vary across a user's own impressions can -- these do.
"""
import numpy as np

from pipeline.data import FEATURE_CARDINALITIES, load

_CACHE = {}


def _train_tables(attr, fit_splits=("train",)):
    key = (attr, fit_splits)
    if key in _CACHE:
        return _CACHE[key]
    uid, aid, y = [], [], []
    for s in fit_splits:
        sp = load(s)
        uid.append(np.asarray(sp.X["user_id"], dtype=np.int64))
        aid.append(np.asarray(sp.X[attr], dtype=np.int64))
        y.append(np.asarray(sp.y, dtype=np.float64))
    uid, aid, y = np.concatenate(uid), np.concatenate(aid), np.concatenate(y)
    card = int(FEATURE_CARDINALITIES[attr])
    comb = uid * card + aid
    uniq, inv = np.unique(comb, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    pos = np.bincount(inv, weights=y)
    ucnt = np.bincount(uid, minlength=int(FEATURE_CARDINALITIES["user_id"])).astype(np.float64)
    upos = np.bincount(uid, weights=y, minlength=int(FEATURE_CARDINALITIES["user_id"]))
    out = (card, uniq, cnt, pos, ucnt, upos, float(y.mean()))
    _CACHE[key] = out
    return out


def affinity(split_name, attr, smoothing=5.0, fit_splits=("train",)):
    """Return {name: array} of user-attribute affinity features for ``split_name``."""
    card, uniq, cnt, pos, ucnt, upos, prior = _train_tables(attr, fit_splits)
    sp = load(split_name)
    uid = np.asarray(sp.X["user_id"], dtype=np.int64)
    aid = np.asarray(sp.X[attr], dtype=np.int64)
    comb = uid * card + aid
    idx = np.searchsorted(uniq, comb)
    idx = np.clip(idx, 0, len(uniq) - 1)
    hit = uniq[idx] == comb
    c = np.where(hit, cnt[idx], 0.0)
    p = np.where(hit, pos[idx], 0.0)
    uc, up = ucnt[uid].copy(), upos[uid].copy()
    if split_name in fit_splits:  # leave-one-out: a row must not see its own label
        own = np.asarray(sp.y, dtype=np.float64)
        c, p, uc, up = c - 1.0, p - own, uc - 1.0, up - own
        c, uc = np.maximum(c, 0.0), np.maximum(uc, 0.0)
    user_rate = (up + 10.0 * prior) / (uc + 10.0)
    rate = (p + smoothing * user_rate) / (c + smoothing)
    return {f"aff_{attr}_resid": (rate - user_rate).astype(np.float32),
            f"aff_{attr}_count": np.log1p(c).astype(np.float32)}
