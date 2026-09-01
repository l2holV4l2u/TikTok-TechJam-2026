"""Item-item collaborative filtering with time decay -- a non-parametric family neither the
FM nor the GBDT covers.

Score(u, i) = sum over the user's earlier train interactions j of sim(i, j), where sim is
cosine co-occurrence over user sets, weighted by how recently the user saw j.
Fitted on train only; a valid/test row is scored from that user's train history.
"""
import numpy as np
from scipy import sparse

from pipeline.data import load


def _train_matrix(positives_only=True):
    tr = load("train")
    u = np.asarray(tr.X["user_id"], dtype=np.int64)
    v = np.asarray(tr.X["video_id"], dtype=np.int64)
    y = np.asarray(tr.y, dtype=np.float64)
    t = np.asarray(tr.time_ms, dtype=np.float64)
    keep = (y > 0) if positives_only else np.ones(len(y), bool)
    return u[keep], v[keep], t[keep]


def scores(split_name, half_life_days=7.0, topk=200, shrink=10.0):
    u, v, t = _train_matrix()
    nu, nv = int(u.max()) + 1, int(v.max()) + 1
    # recency weight of each training interaction, decayed to the end of the train window
    age_d = (t.max() - t) / (1000 * 3600 * 24)
    w = 0.5 ** (age_d / half_life_days)

    M = sparse.csr_matrix((w, (u, v)), shape=(nu, nv))
    norm = np.sqrt(np.asarray(M.multiply(M).sum(axis=0)).ravel()) + shrink
    Mn = M.multiply(sparse.csr_matrix(1.0 / norm))          # column-normalised
    S = (Mn.T @ Mn).tocsr()                                  # item-item cosine similarity
    S.setdiag(0.0)
    S.eliminate_zeros()

    if topk:                                                 # keep the k strongest neighbours per item
        S = S.tolil()
        for i in range(nv):
            row = S.rows[i]
            if len(row) > topk:
                d = np.array(S.data[i]); keep = np.argsort(-d)[:topk]
                S.rows[i] = [row[j] for j in keep]; S.data[i] = [d[j] for j in keep]
        S = S.tocsr()

    sp = load(split_name)
    uu = np.asarray(sp.X["user_id"], dtype=np.int64)
    vv = np.asarray(sp.X["video_id"], dtype=np.int64)
    P = (M @ S).tocsr()                                      # user x item affinity
    uu_c = np.clip(uu, 0, P.shape[0] - 1)
    vv_c = np.clip(vv, 0, P.shape[1] - 1)
    return np.asarray(P[uu_c, vv_c]).ravel().astype(np.float64)


if __name__ == "__main__":
    import json
    from pipeline.evaluate import evaluate
    out = {}
    for hl in (3.0, 7.0, 1e6):
        sv = scores("valid", half_life_days=hl)
        st = scores("test", half_life_days=hl)
        va, te = load("valid"), load("test")
        mv, mt = evaluate(va.user_id, va.y, sv), evaluate(te.user_id, te.y, st)
        print(f"itemcf hl={hl}: valid={mv['primary']:.4f} test={mt['primary']:.4f}", flush=True)
        out[hl] = (mv["primary"], sv, st)
    best = max(out, key=lambda k: out[k][0])
    np.savez("limits/out/itemcf.npz", valid=out[best][1], test=out[best][2])
    print("saved half_life =", best)
