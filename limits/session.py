"""Session-context features: where a row sits in the user's own impression stream.

All are known at serving time -- position and the gap to the PREVIOUS impression. The gap to
the next impression is excluded: it encodes how long this video was watched, i.e. the label.
Each varies within a user's candidate list, which is the only kind of feature a per-user
metric can respond to.
"""
import numpy as np

from pipeline.data import load


def features(split_name):
    sp = load(split_name)
    uid = np.asarray(sp.user_id, dtype=np.int64)
    t = np.asarray(sp.time_ms, dtype=np.int64)
    n = len(uid)
    order = np.lexsort((t, uid))
    u = uid[order]
    ts = t[order]
    start = np.searchsorted(u, np.unique(u))
    size = np.diff(np.append(start, n))
    grp = np.repeat(np.arange(len(size)), size)
    pos = np.arange(n) - np.repeat(start, size)

    prev_t = np.roll(ts, 1)
    same = grp == np.roll(grp, 1)
    gap = np.where(same, (ts - prev_t) / 1000.0, np.nan)
    since_first = (ts - np.repeat(ts[start], size)) / 1000.0

    out = {}
    def put(name, arr):
        full = np.empty(n, dtype=np.float32)
        full[order] = arr
        out[name] = full
    put("sess_pos", pos.astype(np.float32))
    put("sess_pos_frac", (pos / np.maximum(np.repeat(size, size) - 1, 1)).astype(np.float32))
    put("sess_len", np.log1p(np.repeat(size, size)).astype(np.float32))
    put("sess_gap_log", np.log1p(np.clip(gap, 0, None)).astype(np.float32))
    put("sess_since_first_log", np.log1p(np.clip(since_first, 0, None)).astype(np.float32))
    return out
