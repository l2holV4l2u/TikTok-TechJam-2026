"""Synthetic KuaiRand-Pure-shaped cache for testing without the real dataset."""
import argparse
from pathlib import Path

import numpy as np

from pipeline.data import AUX_DTYPES, FEATURE_CARDINALITIES, write_cache

CLICK_RATE = 0.46  # matches the real train split's is_click rate


def generate_split(n: int, seed: int):
    rng = np.random.default_rng(seed)
    user_id = rng.integers(0, 27_285, size=n, dtype=np.int64)
    video_id = rng.integers(0, 7_583, size=n, dtype=np.int64)
    X = {feat: rng.integers(1, card, size=n, dtype=np.int64) for feat, card in FEATURE_CARDINALITIES.items()}
    y = (rng.random(n) < CLICK_RATE).astype(np.int8)

    aux: dict[str, np.ndarray] = {}
    for name, dtype in AUX_DTYPES.items():
        if dtype == "int8":
            # secondary feedback is rarer than the click itself, and only possible on clicked rows
            aux[name] = ((rng.random(n) < 0.05) & (y == 1)).astype(np.int8)
        else:
            aux[name] = np.where(y == 1, rng.exponential(scale=5000.0, size=n), 0.0).astype(np.float32)
    return user_id, video_id, X, y, aux


def build(out_dir, sizes: dict[str, int] | None = None, seed: int = 0) -> None:
    sizes = sizes or {"train": 50_000, "valid": 10_000, "test": 10_000}
    for i, (split, n) in enumerate(sizes.items()):
        user_id, video_id, X, y, aux = generate_split(n, seed=seed + i)
        write_cache(out_dir, split, user_id, video_id, X, y, aux)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build a synthetic KuaiRand-Pure-shaped cache for dev/testing.")
    ap.add_argument("--out-dir", default="data/cache")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(Path(args.out_dir), seed=args.seed)
    print(f"synthetic cache written to {args.out_dir}")
