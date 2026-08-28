"""Component pool for the rank-blend study.

Every component trains on the "train" split only, early-stops against "valid",
and never touches "test" for any fitting or selection decision -- test ranks are
produced purely for later reporting. Results cache to blend/artifacts/ so repeat
weight-search runs don't retrain.

    from blend.components import available, fit_predict
    out = fit_predict("fm_k16", splits, seed=0)   # {"valid": ranks, "test": ranks, "meta": {...}}

`splits` is a dict {"train": Split, "valid": Split, "test": Split} from pipeline.data.load,
passed in by the caller so multiple components can share one load of the data.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

from pipeline.data import FEATURE_CARDINALITIES as FC
from pipeline.evaluate import evaluate

ART_DIR = Path(__file__).parent / "artifacts"
ALL_FIELDS = list(FC)
FM5_FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]


def rank(a) -> np.ndarray:
    return np.argsort(np.argsort(a)).astype(np.float64)


def _fm_offsets(fields):
    off = np.cumsum([0] + [FC[f] for f in fields[:-1]]).astype(np.int64)
    tot = int(sum(FC[f] for f in fields))
    return off, tot


def _fm_mat(s, fields, off):
    return np.stack([np.minimum(s.X[f], FC[f] - 1) + off[i] for i, f in enumerate(fields)], 1)


def _gb_mat(s):
    return np.stack([s.X[f] for f in ALL_FIELDS], 1).astype(np.int32)


def _groups_of(uids):
    o = np.argsort(uids, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(uids[o]) != 0])
    return np.diff(np.r_[starts, len(o)]), o


class _FM(nn.Module):
    """Single shared embedding table over concatenated field offsets, FM pairwise term."""

    def __init__(self, tot, k):
        super().__init__()
        self.emb = nn.Embedding(tot, k)
        self.bias = nn.Embedding(tot, 1)
        nn.init.normal_(self.emb.weight, std=0.01)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x):
        e = self.emb(x)
        return 0.5 * ((e.sum(1) ** 2) - (e ** 2).sum(1)).sum(1) + self.bias(x).sum((1, 2))


class _DCN(nn.Module):
    """FM term plus a shallow DCN-v2 style cross network over the same field embeddings."""

    def __init__(self, tot, n_fields, k=16, cross_layers=2):
        super().__init__()
        self.emb = nn.Embedding(tot, k)
        self.bias = nn.Embedding(tot, 1)
        nn.init.normal_(self.emb.weight, std=0.01)
        nn.init.zeros_(self.bias.weight)
        in_dim = n_fields * k
        self.cw = nn.ParameterList([nn.Parameter(torch.empty(in_dim, in_dim)) for _ in range(cross_layers)])
        self.cb = nn.ParameterList([nn.Parameter(torch.zeros(in_dim)) for _ in range(cross_layers)])
        for w in self.cw:
            nn.init.xavier_uniform_(w)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x):
        e = self.emb(x)  # (B, F, K)
        fm = 0.5 * ((e.sum(1) ** 2) - (e ** 2).sum(1)).sum(1) + self.bias(x).sum((1, 2))
        x0 = e.flatten(1)
        xl = x0
        for w, b in zip(self.cw, self.cb):
            xl = x0 * (xl @ w.T + b) + xl
        return fm + self.head(xl).squeeze(-1)


def _scores(m, X):
    m.eval()
    with torch.no_grad():
        return torch.cat([m(X[i:i + 65536]) for i in range(0, len(X), 65536)]).numpy()


def _fit_torch(model_fn, Xtr, ytr, Xva, va, Xt, seed, max_epochs=40, patience=4, time_budget_s=240):
    """Early-stops on validation primary; also hard-stops on a wall-clock budget so a wide-field
    model that never triggers patience (e.g. fm_all37) can't blow the ~5min-per-component rule."""
    torch.manual_seed(seed)
    m = model_fn()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lf = nn.BCEWithLogitsLoss()
    best, bad, state = -1.0, 0, None
    t0 = time.perf_counter()
    for _ in range(max_epochs):
        m.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), 8192):
            b = perm[i:i + 8192]
            opt.zero_grad()
            lf(m(Xtr[b]), ytr[b]).backward()
            opt.step()
        p = evaluate(va.user_id, va.y, _scores(m, Xva))["primary"]
        if p > best + 1e-5:
            best, bad, state = p, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if time.perf_counter() - t0 > time_budget_s:
            break
    m.load_state_dict(state)
    return rank(_scores(m, Xva)), rank(_scores(m, Xt))


def _fit_lgb(Xtr, ytr, Xva, yva, Xt, objective, seed, groups=None, **hp):
    cat = list(range(Xtr.shape[1]))
    p = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
             verbose=-1, num_threads=8, objective=objective,
             seed=seed, feature_fraction_seed=seed, bagging_seed=seed)
    p.update(hp)
    if objective == "binary":
        p.setdefault("bagging_fraction", 0.8)
        p.setdefault("bagging_freq", 1)
        d = lgb.Dataset(Xtr, ytr, categorical_feature=cat, free_raw_data=False)
        dv = lgb.Dataset(Xva, yva, reference=d)
    else:
        p["metric"] = "ndcg"
        p["ndcg_eval_at"] = [5]
        gtr, gva, otr, ova = groups
        d = lgb.Dataset(Xtr[otr], ytr[otr], group=gtr, categorical_feature=cat, free_raw_data=False)
        dv = lgb.Dataset(Xva[ova], yva[ova], group=gva, reference=d)
    m = lgb.train(p, d, 600, valid_sets=[dv], callbacks=[lgb.early_stopping(40, verbose=False)])
    it = m.best_iteration
    return rank(m.predict(Xva, num_iteration=it)), rank(m.predict(Xt, num_iteration=it))


def _run_fm(splits, seed, fields, k):
    tr, va, te = splits["train"], splits["valid"], splits["test"]
    off, tot = _fm_offsets(fields)
    Xtr = torch.from_numpy(_fm_mat(tr, fields, off))
    ytr = torch.from_numpy(tr.y.astype(np.float32))
    Xva = torch.from_numpy(_fm_mat(va, fields, off))
    Xte = torch.from_numpy(_fm_mat(te, fields, off))
    return _fit_torch(lambda: _FM(tot, k), Xtr, ytr, Xva, va, Xte, seed)


def _run_dcn(splits, seed):
    tr, va, te = splits["train"], splits["valid"], splits["test"]
    fields = FM5_FIELDS
    off, tot = _fm_offsets(fields)
    Xtr = torch.from_numpy(_fm_mat(tr, fields, off))
    ytr = torch.from_numpy(tr.y.astype(np.float32))
    Xva = torch.from_numpy(_fm_mat(va, fields, off))
    Xte = torch.from_numpy(_fm_mat(te, fields, off))
    return _fit_torch(lambda: _DCN(tot, len(fields), k=16), Xtr, ytr, Xva, va, Xte, seed)


def _run_lgb(splits, seed, objective, **hp):
    tr, va, te = splits["train"], splits["valid"], splits["test"]
    Gtr, Gva, Gte = _gb_mat(tr), _gb_mat(va), _gb_mat(te)
    ytr_i = tr.y.astype(np.int32)
    yva = va.y.astype(np.int32)
    groups = None
    if objective == "lambdarank":
        gtr, otr = _groups_of(tr.user_id)
        gva, ova = _groups_of(va.user_id)
        groups = (gtr, gva, otr, ova)
    return _fit_lgb(Gtr, ytr_i, Gva, yva, Gte, objective, seed, groups, **hp)


def _run_itempop(splits, seed):
    """Per-video train-only positive rate, keyed on the train-fit video_id vocab id."""
    tr, va, te = splits["train"], splits["valid"], splits["test"]
    n = FC["video_id"]
    vid = np.asarray(tr.X["video_id"])
    y = np.asarray(tr.y, dtype=np.float64)
    counts = np.bincount(vid, minlength=n)
    pos = np.bincount(vid, weights=y, minlength=n)
    rate = np.zeros(n)
    nz = counts > 0
    rate[nz] = pos[nz] / counts[nz]
    return rank(rate[np.asarray(va.X["video_id"])]), rank(rate[np.asarray(te.X["video_id"])])


_BUILDERS = {
    "fm_k16": lambda splits, seed: _run_fm(splits, seed, FM5_FIELDS, 16),
    "fm_k32": lambda splits, seed: _run_fm(splits, seed, FM5_FIELDS, 32),
    "fm_k8": lambda splits, seed: _run_fm(splits, seed, FM5_FIELDS, 8),
    "fm_all37": lambda splits, seed: _run_fm(splits, seed, ALL_FIELDS, 16),
    "lgb_binary": lambda splits, seed: _run_lgb(splits, seed, "binary"),
    "lgb_lambdarank": lambda splits, seed: _run_lgb(splits, seed, "lambdarank"),
    "lgb_deep": lambda splits, seed: _run_lgb(splits, seed, "binary", num_leaves=255,
                                               learning_rate=0.03, min_data_in_leaf=50),
    "lgb_shallow": lambda splits, seed: _run_lgb(splits, seed, "binary", num_leaves=31, learning_rate=0.1),
    "itempop": _run_itempop,
    "dcn": _run_dcn,
}


def available() -> list[str]:
    return list(_BUILDERS)


def _cache_path(name, seed) -> Path:
    return ART_DIR / f"{name}_seed{seed}.npz"


def fit_predict(name: str, splits: dict, seed: int = 0) -> dict:
    if name not in _BUILDERS:
        raise ValueError(f"unknown component {name!r}, choices: {available()}")

    path = _cache_path(name, seed)
    if path.exists():
        d = np.load(path, allow_pickle=True)
        return {"valid": d["valid"], "test": d["test"], "meta": json.loads(str(d["meta"]))}

    t0 = time.perf_counter()
    va_rank, te_rank = _BUILDERS[name](splits, seed)
    meta = {"seconds": time.perf_counter() - t0}

    ART_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, valid=va_rank, test=te_rank, meta=np.array(json.dumps(meta)))
    return {"valid": va_rank, "test": te_rank, "meta": meta}
