"""Train the best-known config, score a split, write a submission CSV.

python make_submission.py --split test --out submission.csv
"""
import argparse, json, time
import numpy as np, torch, torch.nn as nn

from pipeline.data import load, FEATURE_CARDINALITIES as FC
from pipeline.evaluate import evaluate

FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]
OFF = np.cumsum([0] + [FC[f] for f in FIELDS[:-1]]).astype(np.int64)
TOT = sum(FC[f] for f in FIELDS)


def mat(s):
    return np.stack([np.minimum(s.X[f], FC[f] - 1) + OFF[i] for i, f in enumerate(FIELDS)], 1)


class FM(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        self.emb = nn.Embedding(TOT, k); self.bias = nn.Embedding(TOT, 1)
        nn.init.normal_(self.emb.weight, std=0.01); nn.init.zeros_(self.bias.weight)

    def forward(self, x):
        e = self.emb(x)
        return 0.5 * ((e.sum(1) ** 2) - (e ** 2).sum(1)).sum(1) + self.bias(x).sum((1, 2))


def score_split(m, X):
    m.eval()
    with torch.no_grad():
        return torch.cat([m(X[i:i + 65536]) for i in range(0, len(X), 65536)]).numpy()


def train_one(Xtr, ytr, Xva, va, seed, max_epochs=40, patience=4):
    torch.manual_seed(seed)
    m = FM(); opt = torch.optim.Adam(m.parameters(), lr=1e-3); lf = nn.BCEWithLogitsLoss()
    best, best_state, bad = -1.0, None, 0
    for _ in range(max_epochs):
        m.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), 8192):
            b = perm[i:i + 8192]
            opt.zero_grad(); lf(m(Xtr[b]), ytr[b]).backward(); opt.step()
        p = evaluate(va.user_id, va.y, score_split(m, Xva))["primary"]
        if p > best + 1e-5:
            best, bad = p, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    return m, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    t0 = time.perf_counter()
    tr, va = load("train"), load("valid")
    Xtr = torch.from_numpy(mat(tr)); ytr = torch.from_numpy(tr.y.astype(np.float32))
    Xva = torch.from_numpy(mat(va))
    target = va if args.split == "valid" else load(args.split)
    Xt = Xva if args.split == "valid" else torch.from_numpy(mat(target))

    ranks, vals = [], []
    for s in range(args.seeds):
        m, v = train_one(Xtr, ytr, Xva, va, s)
        vals.append(v)
        sc = score_split(m, Xt)
        ranks.append(np.argsort(np.argsort(sc)))   # rank-average: scale-free across seeds
        print(f"  seed {s}: valid primary {v:.4f}")
    final = np.mean(ranks, 0)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write("row_id,user_id,video_id,score\n")
        for i, (u, v, sc) in enumerate(zip(target.user_id, target.video_id, final)):
            f.write(f"{i},{u},{v},{sc}\n")

    print(f"\nwrote {args.out}  rows={len(final):,}  ({time.perf_counter()-t0:.0f}s)")
    print(f"valid primary: mean {np.mean(vals):.4f} +- {np.std(vals):.4f}")
    # we hold the public labels for the test window, so we can self-score before submitting
    r = evaluate(target.user_id, target.y, final)
    print(f"{args.split} self-scored: primary {r['primary']:.4f} gauc {r['gauc']:.4f} ndcg@5 {r['ndcg@5']:.4f}")
    base = 0.6016 if args.split == "valid" else 0.5946
    print(f"official baseline {args.split}: {base:.4f}   delta = {r['primary']-base:+.4f}")


if __name__ == "__main__":
    main()
