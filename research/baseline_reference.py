"""Run the organizers' baseline recipe on whichever KuaiRand variant the cache holds.

The baseline is specified as a recipe -- FM, k=16, lr=0.001, five fields -- not as a number, so
it runs unchanged on KuaiRand-1K. On Pure it should land near the published 0.6016, which is the
sanity check that makes the 1K number trustworthy as an internally-anchored reference.

  python -m research.baseline_reference                      # Pure
  KUAIRAND_VARIANT=1k KUAIRAND_CACHE_DIR=data/cache_1k \
      python -m research.baseline_reference --epochs 4       # 1K
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.data import FEATURE_CARDINALITIES, load
from pipeline.evaluate import evaluate
from pipeline.models import FM

# kuairand-starter-kit/baseline_scores.json: fm_official.config.fields
FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]


def _tensors(split, device):
    return {f: torch.as_tensor(np.asarray(split.X[f]), dtype=torch.long, device=device) for f in FIELDS}


@torch.no_grad()
def _predict(model, x, n, batch):
    model.eval()
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, batch):
        xb = {f: v[i:i + batch] for f, v in x.items()}
        out[i:i + batch] = torch.sigmoid(model(xb)).float().cpu().numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write the measured reference to this JSON path")
    # Reference scores must not move when the active environment changes. CPU also measured
    # slightly faster for this embedding-heavy baseline on the available RTX 4050.
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    t0 = time.perf_counter()

    tr, va, te = load("train"), load("valid"), load("test")
    print(f"train {len(tr.y):,}  valid {len(va.y):,}  test {len(te.y):,}  device {device}")

    cards = {f: FEATURE_CARDINALITIES[f] for f in FIELDS}
    model = FM(cards, embed_dim=args.k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    x_tr = _tensors(tr, device)
    y_tr = torch.as_tensor(np.asarray(tr.y), dtype=torch.float32, device=device)
    x_va, x_te = _tensors(va, device), _tensors(te, device)
    n = len(tr.y)

    best = {"primary": float("-inf")}
    best_state = None
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                model({f: v[idx] for f, v in x_tr.items()}), y_tr[idx])
            loss.backward()
            opt.step()
        m = evaluate(va.user_id, va.y, _predict(model, x_va, len(va.y), args.batch_size))
        print(f"  epoch {ep+1}: valid primary {m['primary']:.4f}  gauc {m['gauc']:.4f}  ndcg@5 {m['ndcg@5']:.4f}")
        if m["primary"] > best["primary"]:
            best = m
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    t = evaluate(te.user_id, te.y, _predict(model, x_te, len(te.y), args.batch_size))
    wall = time.perf_counter() - t0

    ref = {"valid_primary": best["primary"], "valid_gauc": best["gauc"], "valid_ndcg@5": best["ndcg@5"],
           "test_primary": t["primary"], "test_gauc": t["gauc"], "test_ndcg@5": t["ndcg@5"],
           "fields": FIELDS, "k": args.k, "lr": args.lr, "epochs": args.epochs,
           "wall_clock_s": wall}
    print(f"\nREFERENCE valid {best['primary']:.4f}   test {t['primary']:.4f}   ({wall/60:.1f} min)")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(ref, f, indent=2)
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
