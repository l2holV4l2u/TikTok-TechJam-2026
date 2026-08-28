"""CLI: python -m pipeline.train --model fm --epochs 3 --fraction 0.1 --out scores.csv"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.data import FEATURE_CARDINALITIES, Split, load
from pipeline.evaluate import evaluate
from pipeline.models import build


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["fm", "deepfm", "dcnv2", "din"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fraction", type=float, default=1.0, help="subsample fraction of the train split")
    p.add_argument("--out", default=None, help="path to write a user_id,video_id,score submission CSV")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def subsample(split: Split, fraction: float, seed: int) -> Split:
    n = len(split.y)
    if fraction >= 1.0:
        return split
    k = max(1, int(round(n * fraction)))
    idx = np.random.default_rng(seed).choice(n, size=k, replace=False)
    return Split(
        user_id=split.user_id[idx],
        video_id=split.video_id[idx],
        X={f: v[idx] for f, v in split.X.items()},
        y=split.y[idx],
        aux={f: v[idx] for f, v in split.aux.items()},
    )


def to_tensors(X: dict, device) -> dict:
    # np.array(v) copies out of data.py's read-only mmap -- avoids the non-writable-tensor warning.
    return {f: torch.tensor(np.array(v), dtype=torch.int64, device=device) for f, v in X.items()}


@torch.no_grad()
def predict(model, X: dict, batch_size: int) -> np.ndarray:
    model.eval()
    n = len(next(iter(X.values())))
    out = []
    for i in range(0, n, batch_size):
        xb = {f: v[i:i + batch_size] for f, v in X.items()}
        out.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    return np.concatenate(out)


def write_submission(path: str, user_id, video_id, scores):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "video_id", "score"])
        for u, v, s in zip(user_id, video_id, scores):
            w.writerow([int(u), int(v), float(s)])


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu")

    train = subsample(load("train"), args.fraction, args.seed)
    valid = load("valid")

    model = build(args.model, FEATURE_CARDINALITIES, embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    wall_start = time.perf_counter()

    x_train = to_tensors(train.X, device)
    y_train = torch.tensor(np.array(train.y), dtype=torch.float32, device=device)
    x_valid = to_tensors(valid.X, device)
    n_train = len(train.y)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"{args.model}_best.pt")

    best_primary, best_metrics, best_scores = float("-inf"), None, None
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = {f: v[idx] for f, v in x_train.items()}
            yb = y_train[idx]
            optimizer.zero_grad(set_to_none=True)
            logit = model(xb)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            loss.backward()
            optimizer.step()

        scores = predict(model, x_valid, args.batch_size)
        metrics = evaluate(valid.user_id, valid.y, scores)
        if metrics["primary"] > best_primary:
            best_primary, best_metrics, best_scores = metrics["primary"], metrics, scores
            torch.save(model.state_dict(), ckpt_path)

    gpu_seconds = time.perf_counter() - wall_start

    if best_metrics is None:
        best_metrics, best_scores = metrics, scores  # every epoch scored worse than -inf never happens, but guard anyway

    if args.out:
        write_submission(args.out, valid.user_id, valid.video_id, best_scores)

    result = {
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg@5": float(best_metrics["ndcg@5"]),
        "gpu_seconds": float(gpu_seconds),
        "tokens_in": 0,
        "tokens_out": 0,
    }
    print("METRICS " + json.dumps(result))


if __name__ == "__main__":
    main()
