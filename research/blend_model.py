"""FM + LightGBM(binary) + LightGBM(lambdarank) rank-blend.

Neither family beats the FM baseline alone; blending their ranks does, because their
error profiles differ (LGB is stronger on nDCG, FM on GAUC).

python blend_model.py --split test --out submission.csv
"""
import argparse, time
import numpy as np, torch, torch.nn as nn, lightgbm as lgb

from pipeline.data import load, FEATURE_CARDINALITIES as FC
from pipeline.evaluate import evaluate

FM_FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
OFF = np.cumsum([0] + [FC[f] for f in FM_FIELDS[:-1]]).astype(np.int64)
TOT = sum(FC[f] for f in FM_FIELDS)
ALL = list(FC)


def fm_mat(s):
    return np.stack([np.minimum(s.X[f], FC[f] - 1) + OFF[i] for i, f in enumerate(FM_FIELDS)], 1)


def gb_mat(s):
    return np.stack([s.X[f] for f in ALL], 1).astype(np.int32)


class FM(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        self.emb = nn.Embedding(TOT, k); self.bias = nn.Embedding(TOT, 1)
        nn.init.normal_(self.emb.weight, std=0.01); nn.init.zeros_(self.bias.weight)

    def forward(self, x):
        e = self.emb(x)
        return 0.5 * ((e.sum(1) ** 2) - (e ** 2).sum(1)).sum(1) + self.bias(x).sum((1, 2))


def _scores(m, X):
    m.eval()
    with torch.no_grad():
        return torch.cat([m(X[i:i + 65536]) for i in range(0, len(X), 65536)]).numpy()


def rank(a):
    return np.argsort(np.argsort(a)).astype(np.float64)


def fit_fm(Xtr, ytr, Xva, va, Xt, seeds=5):
    """Early-stops each seed on validation, returns rank-averaged val and target scores."""
    rv, rt = [], []
    for s in range(seeds):
        torch.manual_seed(s)
        m = FM(); opt = torch.optim.Adam(m.parameters(), lr=1e-3); lf = nn.BCEWithLogitsLoss()
        best, bad, state = -1.0, 0, None
        for _ in range(40):
            m.train()
            perm = torch.randperm(len(ytr))
            for i in range(0, len(perm), 8192):
                b = perm[i:i + 8192]
                opt.zero_grad(); lf(m(Xtr[b]), ytr[b]).backward(); opt.step()
            p = evaluate(va.user_id, va.y, _scores(m, Xva))["primary"]
            if p > best + 1e-5:
                best, bad = p, 0
                state = {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= 4:
                    break
        m.load_state_dict(state)
        rv.append(rank(_scores(m, Xva))); rt.append(rank(_scores(m, Xt)))
    return np.mean(rv, 0), np.mean(rt, 0)


def fit_lgb(Xtr, ytr, Xva, yva, Xt, objective, groups=None):
    cat = list(range(Xtr.shape[1]))
    p = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
             feature_fraction=0.8, verbose=-1, num_threads=8, objective=objective)
    if objective == "binary":
        p.update(bagging_fraction=0.8, bagging_freq=1)
        d = lgb.Dataset(Xtr, ytr, categorical_feature=cat, free_raw_data=False)
        dv = lgb.Dataset(Xva, yva, reference=d)
    else:
        p.update(metric="ndcg", ndcg_eval_at=[5])
        gtr, gva, otr, ova = groups
        d = lgb.Dataset(Xtr[otr], ytr[otr], group=gtr, categorical_feature=cat, free_raw_data=False)
        dv = lgb.Dataset(Xva[ova], yva[ova], group=gva, reference=d)
    m = lgb.train(p, d, 600, valid_sets=[dv], callbacks=[lgb.early_stopping(40, verbose=False)])
    it = m.best_iteration
    return rank(m.predict(Xva, num_iteration=it)), rank(m.predict(Xt, num_iteration=it))


def groups_of(uids):
    o = np.argsort(uids, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(uids[o]) != 0])
    return np.diff(np.r_[starts, len(o)]), o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    t0 = time.perf_counter()
    tr, va = load("train"), load("valid")
    target = va if args.split == "valid" else load(args.split)
    yva = va.y.astype(np.int32)

    fm_v, fm_t = fit_fm(torch.from_numpy(fm_mat(tr)), torch.from_numpy(tr.y.astype(np.float32)),
                        torch.from_numpy(fm_mat(va)), va, torch.from_numpy(fm_mat(target)), args.seeds)
    print(f"FM         valid {evaluate(va.user_id, yva, fm_v)['primary']:.4f}")

    Gtr, Gva, Gt = gb_mat(tr), gb_mat(va), gb_mat(target)
    ytr_i = tr.y.astype(np.int32)
    gb_v, gb_t = fit_lgb(Gtr, ytr_i, Gva, yva, Gt, "binary")
    print(f"LGB-binary valid {evaluate(va.user_id, yva, gb_v)['primary']:.4f}")

    gtr, otr = groups_of(tr.user_id); gva, ova = groups_of(va.user_id)
    lr_v, lr_t = fit_lgb(Gtr, ytr_i, Gva, yva, Gt, "lambdarank", (gtr, gva, otr, ova))
    print(f"LGB-lambda valid {evaluate(va.user_id, yva, lr_v)['primary']:.4f}")

    # weights chosen on validation over a coarse simplex
    best, bw = -1.0, None
    for a in np.arange(0, 1.01, 0.1):
        for b in np.arange(0, 1.01 - a + 1e-9, 0.1):
            c = 1 - a - b
            p = evaluate(va.user_id, yva, a * fm_v + b * gb_v + c * lr_v)["primary"]
            if p > best:
                best, bw = p, (a, b, c)
    a, b, c = bw
    print(f"\nbest weights fm={a:.1f} lgbbin={b:.1f} lgblam={c:.1f}  valid primary {best:.4f}")

    final = a * fm_t + b * gb_t + c * lr_t
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write("row_id,user_id,video_id,score\n")
        for i, (u, v, s) in enumerate(zip(target.user_id, target.video_id, final)):
            f.write(f"{i},{u},{v},{s}\n")
    print(f"wrote {args.out} rows={len(final):,} ({time.perf_counter()-t0:.0f}s)")

    r = evaluate(target.user_id, target.y, final)
    base = 0.6016 if args.split == "valid" else 0.5946
    print(f"{args.split} self-scored: primary {r['primary']:.4f} gauc {r['gauc']:.4f} ndcg@5 {r['ndcg@5']:.4f}")
    print(f"official baseline {args.split} {base:.4f}   DELTA {r['primary']-base:+.4f}")


if __name__ == "__main__":
    main()
