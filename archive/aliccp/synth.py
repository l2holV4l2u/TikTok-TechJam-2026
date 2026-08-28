"""Synthetic AliCCP-shaped cache for testing without the real dataset."""
import argparse
from pathlib import Path

import numpy as np

from pipeline.data import FEATURE_CARDINALITIES, write_cache

CLICK_RATE = 0.04
CONV_GIVEN_CLICK_RATE = 0.005


def generate_split(n: int, seed: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = {feat: rng.integers(1, card, size=n, dtype=np.int64) for feat, card in FEATURE_CARDINALITIES.items()}
    y_click = (rng.random(n) < CLICK_RATE).astype(np.int8)
    y_conv = np.zeros(n, dtype=np.int8)
    click_idx = np.flatnonzero(y_click)
    conv_hits = rng.random(click_idx.size) < CONV_GIVEN_CLICK_RATE
    y_conv[click_idx[conv_hits]] = 1
    return X, y_click, y_conv


def build(out_dir, sizes: dict[str, int] | None = None, seed: int = 0) -> None:
    sizes = sizes or {"train": 50_000, "valid": 10_000, "test": 10_000}
    for i, (split, n) in enumerate(sizes.items()):
        X, y_click, y_conv = generate_split(n, seed=seed + i)
        write_cache(out_dir, split, X, y_click, y_conv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build a synthetic AliCCP-shaped cache for dev/testing.")
    ap.add_argument("--out-dir", default="data/cache")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(Path(args.out_dir), seed=args.seed)
    print(f"synthetic cache written to {args.out_dir}")
