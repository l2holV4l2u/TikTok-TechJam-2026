"""CLI: python -m pipeline.train --model esmm --fraction 0.01 --epochs 2 [...]"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from pipeline.data import FEATURE_CARDINALITIES, load
from pipeline.evaluate import ctr_cvr_auc
from pipeline.models import build


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["shared_bottom", "esmm", "mmoe", "ple"])
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="valid")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(split_x: dict, y_click, y_conv, device):
    x = {k: torch.as_tensor(v, dtype=torch.int64, device=device) for k, v in split_x.items()}
    return x, torch.as_tensor(y_click, dtype=torch.float32, device=device), \
        torch.as_tensor(y_conv, dtype=torch.float32, device=device)


def batch_loss(model, x, y_click, y_conv, is_esmm):
    ctr_logit, cvr_logit = model(x)
    ctr_loss = F.binary_cross_entropy_with_logits(ctr_logit, y_click)
    if is_esmm:
        # ESMM's defining trick: supervise CVR only through CTCVR = p_ctr * p_cvr over ALL impressions,
        # since y_conv is only meaningful where y_click == 1 this avoids the CVR sample-selection bias.
        p_ctcvr = torch.sigmoid(ctr_logit) * torch.sigmoid(cvr_logit)
        target = y_click * y_conv
        cvr_loss = F.binary_cross_entropy(p_ctcvr.clamp(1e-7, 1 - 1e-7), target)
    else:
        clicked = y_click == 1
        if clicked.any():
            cvr_loss = F.binary_cross_entropy_with_logits(cvr_logit[clicked], y_conv[clicked])
        else:
            cvr_loss = torch.zeros((), device=ctr_logit.device)
    return ctr_loss + cvr_loss


@torch.no_grad()
def predict(model, x, batch_size):
    model.eval()
    n = len(next(iter(x.values())))
    ctr_out, cvr_out = [], []
    for i in range(0, n, batch_size):
        xb = {k: v[i:i + batch_size] for k, v in x.items()}
        ctr_logit, cvr_logit = model(xb)
        ctr_out.append(torch.sigmoid(ctr_logit).cpu().numpy())
        cvr_out.append(torch.sigmoid(cvr_logit).cpu().numpy())
    return np.concatenate(ctr_out), np.concatenate(cvr_out)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_cuda = device.type == "cuda"

    train = load(args.train_split, fraction=args.fraction, seed=args.seed)
    val = load(args.val_split, fraction=args.fraction, seed=args.seed)

    model = build(args.model, FEATURE_CARDINALITIES, embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    is_esmm = args.model == "esmm"

    if use_cuda:
        start_evt, end_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_evt.record()
    else:
        wall_start = time.perf_counter()

    x_train, y_click_train, y_conv_train = to_device(train.X, train.y_click, train.y_conv, device)
    n_train = len(train.y_click)

    best_score, best_metrics, best_state = float("-inf"), None, None
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"{args.model}_best.pt")

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = {k: v[idx] for k, v in x_train.items()}
            yb_click, yb_conv = y_click_train[idx], y_conv_train[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_cuda):
                loss = batch_loss(model, xb, yb_click, yb_conv, is_esmm)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        x_val = {k: torch.as_tensor(v, dtype=torch.int64, device=device) for k, v in val.X.items()}
        p_ctr, p_cvr = predict(model, x_val, args.batch_size)
        metrics = ctr_cvr_auc(val.y_click, val.y_conv, p_ctr, p_cvr)

        score = np.nanmean([metrics["ctr_auc"], metrics["cvr_auc"]])
        if not np.isnan(score) and score > best_score:
            best_score, best_metrics = score, metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)

    if use_cuda:
        end_evt.record()
        torch.cuda.synchronize()
        gpu_seconds = start_evt.elapsed_time(end_evt) / 1000.0
    else:
        gpu_seconds = time.perf_counter() - wall_start

    if best_metrics is None:
        best_metrics = metrics  # every epoch scored nan (e.g. degenerate tiny --fraction); fall back to last

    result = {"ctr_auc": best_metrics["ctr_auc"], "cvr_auc": best_metrics["cvr_auc"], "gpu_seconds": gpu_seconds}
    print("METRICS " + json.dumps(result))


if __name__ == "__main__":
    main()
