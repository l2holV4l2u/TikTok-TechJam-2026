"""Train-only metric-aligned neural ranker for KuaiRand-Pure.

This module is intentionally outside the generated agent harness. It trains for
a fixed, predeclared number of epochs on the official train split and writes
validation/test score arrays without reading test labels or test outcomes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.data import AUX_DTYPES, FEATURE_CARDINALITIES, NUMERIC_FEATURES, Split, load
from pipeline.evaluate import evaluate

DEFAULT_FIELDS = (
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "tag",
    "duration_bucket",
    "hour",
    "upload_type",
    "music_type",
    "video_type",
    "onehot_feat3",
    "onehot_feat8",
    "user_active_degree",
    "register_days_bucket",
)


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def categorical_matrix(split: Split, fields: tuple[str, ...]) -> np.ndarray:
    missing = [field for field in fields if field not in split.X]
    if missing:
        raise KeyError(f"unknown categorical fields: {missing}")
    return np.ascontiguousarray(
        np.column_stack([np.asarray(split.X[field], dtype=np.int32) for field in fields]),
        dtype=np.int32,
    )


@dataclass
class NumericPreprocessor:
    names: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, split: Split, names: tuple[str, ...] | None = None) -> "NumericPreprocessor":
        available = split.num or {}
        names = tuple(name for name in (names or NUMERIC_FEATURES) if name in available)
        if not names:
            return cls((), np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32))

        values = np.column_stack([np.asarray(available[name], dtype=np.float32) for name in names])
        center = np.nanmedian(values, axis=0).astype(np.float32)
        center = np.where(np.isfinite(center), center, 0.0).astype(np.float32)
        filled = np.where(np.isfinite(values), values, center)
        scale = np.nanpercentile(filled, 75, axis=0) - np.nanpercentile(filled, 25, axis=0)
        scale = np.where(scale > 1e-6, scale, np.nanstd(filled, axis=0))
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
        return cls(names, center, scale)

    def transform(self, split: Split) -> np.ndarray:
        n = len(split)
        if not self.names:
            return np.zeros((n, 0), dtype=np.float32)
        available = split.num or {}
        values = np.column_stack([np.asarray(available[name], dtype=np.float32) for name in self.names])
        values = np.where(np.isfinite(values), values, self.center)
        values = (values - self.center) / self.scale
        return np.clip(values, -8.0, 8.0).astype(np.float32, copy=False)


def binary_aux_matrix(split: Split, names: tuple[str, ...]) -> np.ndarray:
    if not names:
        return np.zeros((len(split), 0), dtype=np.float32)
    bad = [name for name in names if AUX_DTYPES.get(name) != "int8" or name not in split.aux]
    if bad:
        raise ValueError(f"auxiliary targets must be binary train outcomes, got {bad}")
    cols = [np.asarray(split.aux[name], dtype=np.float32) for name in names]
    return np.column_stack(cols).astype(np.float32, copy=False)


def user_groups(user_ids: np.ndarray, labels: np.ndarray | None = None, require_mixed: bool = False) -> list[np.ndarray]:
    users = np.asarray(user_ids)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    if sorted_users.size == 0:
        return []

    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], sorted_users.size]
    groups: list[np.ndarray] = []
    labels_arr = None if labels is None else np.asarray(labels)
    for start, end in zip(starts, ends):
        rows = order[start:end].astype(np.int64, copy=True)
        if rows.size < 2:
            continue
        if require_mixed and labels_arr is not None:
            y = labels_arr[rows]
            if not (np.any(y > 0) and np.any(y <= 0)):
                continue
        groups.append(rows)
    return groups


def _sample_rows_from_group(
    rows: np.ndarray,
    labels: np.ndarray,
    max_slate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if rows.size <= max_slate:
        return rows

    y = labels[rows]
    pos = rows[y > 0]
    neg = rows[y <= 0]
    chosen: list[int] = []
    if pos.size:
        chosen.append(int(rng.choice(pos)))
    if neg.size and len(chosen) < max_slate:
        chosen.append(int(rng.choice(neg)))

    remaining = np.setdiff1d(rows, np.asarray(chosen, dtype=np.int64), assume_unique=False)
    fill = max_slate - len(chosen)
    if fill > 0:
        chosen.extend(int(x) for x in rng.choice(remaining, size=fill, replace=False))
    rng.shuffle(chosen)
    return np.asarray(chosen, dtype=np.int64)


class WholeUserSlateSampler:
    """Seeded whole-user sampler; row truncation never splits by date/session."""

    def __init__(
        self,
        user_ids: np.ndarray,
        labels: np.ndarray,
        max_slate: int,
        users_per_epoch: int | None,
        seed: int,
    ) -> None:
        self.labels = np.asarray(labels)
        self.max_slate = int(max_slate)
        self.users_per_epoch = users_per_epoch
        self.groups = user_groups(user_ids, self.labels, require_mixed=True)
        self.rng = np.random.default_rng(seed)
        if not self.groups:
            raise ValueError("no train users with both positive and negative labels")

    def epoch(self) -> list[np.ndarray]:
        n_users = len(self.groups)
        limit = n_users if self.users_per_epoch is None else min(int(self.users_per_epoch), n_users)
        user_order = self.rng.choice(n_users, size=limit, replace=False)
        return [
            _sample_rows_from_group(self.groups[int(i)], self.labels, self.max_slate, self.rng)
            for i in user_order
        ]


class SlateRanker(nn.Module):
    def __init__(
        self,
        fields: tuple[str, ...],
        cardinalities: dict[str, int],
        numeric_dim: int = 0,
        embed_dim: int = 16,
        hidden_dims: tuple[int, ...] = (128, 64),
        n_aux: int = 0,
    ) -> None:
        super().__init__()
        self.fields = fields
        self.embeddings = nn.ModuleList([nn.Embedding(int(cardinalities[field]), embed_dim) for field in fields])
        self.linear = nn.ModuleList([nn.Embedding(int(cardinalities[field]), 1) for field in fields])
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.02)
        for emb in self.linear:
            nn.init.zeros_(emb.weight)

        in_dim = len(fields) * embed_dim + numeric_dim
        layers: list[nn.Module] = []
        last = in_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(last, hidden), nn.SiLU(), nn.LayerNorm(hidden)])
            last = hidden
        self.body = nn.Sequential(*layers) if layers else nn.Identity()
        self.main_head = nn.Linear(last, 1)
        self.aux_head = nn.Linear(last, n_aux) if n_aux else None
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        embedded = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        if x_num is not None and x_num.shape[1]:
            features = torch.cat([embedded, x_num], dim=1)
        else:
            features = embedded
        hidden = self.body(features)
        first_order = sum(emb(x_cat[:, i]).squeeze(1) for i, emb in enumerate(self.linear))
        main = self.bias + first_order + self.main_head(hidden).squeeze(1)
        aux = None if self.aux_head is None else self.aux_head(hidden)
        return main, aux


def lambda_ndcg_loss(scores: torch.Tensor, labels: torch.Tensor, topk: int = 5, sigma: float = 1.0) -> torch.Tensor:
    labels = labels.float()
    n = int(labels.numel())
    if n < 2 or torch.count_nonzero(labels > 0).item() == 0:
        return scores.sum() * 0.0

    gains = torch.pow(2.0, labels) - 1.0
    ideal_gains, _ = torch.sort(gains, descending=True)
    ideal_limit = min(int(topk), n)
    ideal_discount = 1.0 / torch.log2(torch.arange(ideal_limit, device=scores.device, dtype=torch.float32) + 2.0)
    idcg = torch.sum(ideal_gains[:ideal_limit] * ideal_discount)
    if float(idcg.detach().cpu()) <= 0.0:
        return scores.sum() * 0.0

    with torch.no_grad():
        _, order = torch.sort(scores.detach(), descending=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(n, device=scores.device)
        discount = 1.0 / torch.log2(rank.float() + 2.0)
        top_focus = (rank[:, None] < topk) | (rank[None, :] < topk)
        better = labels[:, None] > labels[None, :]
        delta = torch.abs(discount[:, None] - discount[None, :])
        delta = delta * torch.abs(gains[:, None] - gains[None, :]) / idcg.clamp_min(1e-12)
        weights = torch.where(better & top_focus, delta, torch.zeros_like(delta))

    if torch.count_nonzero(weights).item() == 0:
        return scores.sum() * 0.0
    diff = scores[:, None] - scores[None, :]
    losses = F.softplus(-float(sigma) * diff) / float(sigma)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


def pairwise_auc_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pos = scores[labels > 0]
    neg = scores[labels <= 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return scores.sum() * 0.0
    return F.softplus(-(pos[:, None] - neg[None, :])).mean()


@torch.inference_mode()
def predict(
    model: SlateRanker,
    x_cat: np.ndarray,
    x_num: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    out = np.empty(x_cat.shape[0], dtype=np.float32)
    for start in range(0, x_cat.shape[0], batch_size):
        end = min(start + batch_size, x_cat.shape[0])
        cat = torch.as_tensor(x_cat[start:end], dtype=torch.long, device=device)
        num = torch.as_tensor(x_num[start:end], dtype=torch.float32, device=device)
        logits, _ = model(cat, num)
        out[start:end] = logits.detach().cpu().numpy()
    return out


def train_one_epoch(
    model: SlateRanker,
    optimizer: torch.optim.Optimizer,
    sampler: WholeUserSlateSampler,
    x_cat: np.ndarray,
    x_num: np.ndarray,
    y: np.ndarray,
    aux: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model.train()
    slates = sampler.epoch()
    total = 0.0
    steps = 0
    for start in range(0, len(slates), args.user_batch_size):
        optimizer.zero_grad(set_to_none=True)
        slate_batch = slates[start:start + args.user_batch_size]
        lengths = [len(rows) for rows in slate_batch]
        rows = np.concatenate(slate_batch)
        cat = torch.as_tensor(x_cat[rows], dtype=torch.long, device=device)
        num = torch.as_tensor(x_num[rows], dtype=torch.float32, device=device)
        targets = torch.as_tensor(y[rows], dtype=torch.float32, device=device)
        logits, aux_logits = model(cat, num)
        aux_targets = (torch.as_tensor(aux[rows], dtype=torch.float32, device=device)
                       if aux.shape[1] and args.aux_weight > 0.0 else None)
        batch_losses = []
        offset = 0
        for length in lengths:
            sl = slice(offset, offset + length)
            target = targets[sl]
            slate_logits = logits[sl]
            loss = (
                args.lambda_weight * lambda_ndcg_loss(slate_logits, target, topk=args.topk, sigma=args.lambda_sigma)
                + args.pairwise_weight * pairwise_auc_loss(slate_logits, target)
            )
            if args.bce_weight > 0.0:
                loss = loss + args.bce_weight * F.binary_cross_entropy_with_logits(slate_logits, target)
            if aux_targets is not None and aux_logits is not None:
                loss = loss + args.aux_weight * F.binary_cross_entropy_with_logits(
                    aux_logits[sl], aux_targets[sl])
            batch_losses.append(loss)
            offset += length

        step_loss = torch.stack(batch_losses).mean()
        step_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total += float(step_loss.detach().cpu())
        steps += 1
    return total / max(steps, 1)


def train_metric_ranker(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    torch.set_num_threads(min(args.threads, max(1, os.cpu_count() or 1)))
    device = torch.device("cpu")

    os.environ["AGENT_HIDE_TEST_LABELS"] = "1"
    train = load("train")
    valid = load("valid")
    test = load("test")

    fields = parse_csv(args.fields)
    aux_names = parse_csv(args.aux_targets)
    hidden_dims = tuple(int(x) for x in parse_csv(args.hidden_dims))

    x_train_cat = categorical_matrix(train, fields)
    x_valid_cat = categorical_matrix(valid, fields)
    x_test_cat = categorical_matrix(test, fields)
    num = NumericPreprocessor.fit(train)
    x_train_num = num.transform(train)
    x_valid_num = num.transform(valid)
    x_test_num = num.transform(test)
    y_train = np.asarray(train.y, dtype=np.float32)
    y_valid = np.asarray(valid.y, dtype=np.int8)
    aux_train = binary_aux_matrix(train, aux_names)

    users_per_epoch = None if args.max_users_per_epoch <= 0 else args.max_users_per_epoch
    sampler = WholeUserSlateSampler(
        train.user_id,
        y_train,
        max_slate=args.max_slate,
        users_per_epoch=users_per_epoch,
        seed=args.seed + 101,
    )
    model = SlateRanker(
        fields,
        FEATURE_CARDINALITIES,
        numeric_dim=x_train_num.shape[1],
        embed_dim=args.embed_dim,
        hidden_dims=hidden_dims,
        n_aux=len(aux_names),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    scores: np.ndarray | None = None
    metrics: dict[str, float] | None = None
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, optimizer, sampler, x_train_cat, x_train_num, y_train, aux_train, args, device)
        scores = predict(model, x_valid_cat, x_valid_num, args.pred_batch_size, device)
        metrics = evaluate(valid.user_id, y_valid, scores)
        history.append({"epoch": epoch, "loss": loss, **{k: float(metrics[k]) for k in ("primary", "gauc", "ndcg@5")}})
        print("EPOCH " + json.dumps(history[-1], sort_keys=True), flush=True)
    assert metrics is not None and scores is not None

    test_scores = predict(model, x_test_cat, x_test_num, args.pred_batch_size, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "scores_valid.npy", scores.astype(np.float64))
    np.save(out_dir / "scores_test.npy", test_scores.astype(np.float64))
    if args.save_model:
        torch.save({"state_dict": model.state_dict(), "fields": fields, "numeric": num}, out_dir / "metric_ranker.pt")

    return {
        "primary": float(metrics["primary"]),
        "gauc": float(metrics["gauc"]),
        "ndcg@5": float(metrics["ndcg@5"]),
        "fixed_epochs": int(args.epochs),
        "n_train_users": int(len(sampler.groups)),
        "fields": fields,
        "aux_targets": aux_names,
        "numeric_features": num.names,
        "history": history,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="runs/metric_ranker")
    p.add_argument("--epochs", type=int, default=1,
                   help="fixed before training; validation never selects a checkpoint")
    p.add_argument("--max-users-per-epoch", type=int, default=6000, help="0 means all eligible train users")
    p.add_argument("--max-slate", type=int, default=64)
    p.add_argument("--user-batch-size", type=int, default=16)
    p.add_argument("--pred-batch-size", type=int, default=16384)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=2e-5)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--hidden-dims", default="128,64")
    p.add_argument("--fields", default=",".join(DEFAULT_FIELDS))
    p.add_argument("--aux-targets", default="is_click,is_like")
    p.add_argument("--lambda-weight", type=float, default=0.55)
    p.add_argument("--pairwise-weight", type=float, default=0.35)
    p.add_argument("--bce-weight", type=float, default=0.10)
    p.add_argument("--aux-weight", type=float, default=0.08)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--lambda-sigma", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--save-model", action="store_true")
    args = p.parse_args(argv)
    if args.max_slate < 2:
        raise ValueError("--max-slate must be at least 2")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    return args


def main(argv: list[str] | None = None) -> dict[str, object]:
    start = time.perf_counter()
    args = parse_args(argv)
    result = train_metric_ranker(args)
    result["gpu_seconds"] = float(time.perf_counter() - start)
    print(
        "METRICS "
        + json.dumps(
            {key: result[key] for key in ("primary", "gauc", "ndcg@5", "gpu_seconds")},
            sort_keys=True,
        ),
        flush=True,
    )
    print(
        "FINDINGS "
        + json.dumps(
            {
                "fixed_epochs": result["fixed_epochs"],
                "fields": result["fields"],
                "aux_targets": result["aux_targets"],
                "numeric_features": result["numeric_features"],
                "n_train_users": result["n_train_users"],
                "test_labels_read": False,
                "test_aux_read": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    main()
