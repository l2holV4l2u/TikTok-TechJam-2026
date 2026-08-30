"""Synthetic KuaiRand-Pure-shaped cache for testing without the real dataset.

The shape has to match what `pipeline.data.load` actually returns, not a subset of it. Twice
now the loader gained a channel -- `date`, then `num` -- and this generator did not, so
`Split.date`/`Split.num` came back None on every synthetic cache and the tests covering them
either skipped silently or failed. Anything `load` reads is produced here.
"""
import argparse
from pathlib import Path

import numpy as np

from pipeline.data import (AUX_DTYPES, FEATURE_CARDINALITIES, NUMERIC_FEATURES, write_cache)

CLICK_RATE = 0.46  # matches the real train split's is_click rate

# The organizers' date windows, so a synthetic split spans the same boundaries the real one
# does and code that filters on `date` behaves the same way against it.
_WINDOWS = {
    "train": (20220409, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}


def _local_noon_ms(yyyymmdd: int) -> int:
    """Epoch ms at local midday on the given date, so a round trip through
    datetime.fromtimestamp reproduces exactly that date."""
    import datetime

    y, m, d = yyyymmdd // 10000, (yyyymmdd // 100) % 100, yyyymmdd % 100
    return int(datetime.datetime(y, m, d, 12, 0).timestamp() * 1000)


def generate_split(n: int, seed: int, split: str = "train"):
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

    lo, hi = _WINDOWS.get(split, _WINDOWS["train"])
    days = _dates_between(lo, hi)
    day_index = rng.integers(0, len(days), size=n)
    date = days[day_index].astype(np.int32)
    # time_ms must land on the same calendar day as `date` when read back, and the loader's
    # consumers read it with datetime.fromtimestamp -- which is LOCAL time. Anchoring at local
    # midday and jittering +-6h keeps every row inside its own day under any timezone or DST
    # shift; anchoring at UTC midnight silently moved rows a day either side.
    noon_ms = np.array([_local_noon_ms(int(d)) for d in days], dtype=np.int64)
    time_ms = (noon_ms[day_index]
               + rng.integers(-6 * 3_600_000, 6 * 3_600_000, size=n)).astype(np.int64)

    # Continuous channels. A real cache carries NaN where a user is absent from the side table,
    # so a fraction is NaN here too -- code that forgets to handle it must fail in tests, not
    # only against the real data.
    num: dict[str, np.ndarray] = {}
    for name in NUMERIC_FEATURES:
        scale = 60_000.0 if name == "duration_ms" else 500.0
        col = np.abs(rng.exponential(scale=scale, size=n)).astype(np.float32)
        col[rng.random(n) < 0.02] = np.nan
        num[name] = col
    return user_id, video_id, X, y, aux, date, time_ms, num


def _dates_between(lo: int, hi: int) -> np.ndarray:
    """Every YYYYMMDD between two inclusive bounds, without a calendar dependency."""
    out, y, m, d = [], lo // 10000, (lo // 100) % 100, lo % 100
    length = (31, 29 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 28,
              31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    while y * 10000 + m * 100 + d <= hi:
        out.append(y * 10000 + m * 100 + d)
        d += 1
        if d > length[m - 1]:
            d, m = 1, m + 1
            if m > 12:
                m, y = 1, y + 1
    return np.array(out, dtype=np.int64)


def build(out_dir, sizes: dict[str, int] | None = None, seed: int = 0) -> None:
    sizes = sizes or {"train": 50_000, "valid": 10_000, "test": 10_000}
    for i, (split, n) in enumerate(sizes.items()):
        user_id, video_id, X, y, aux, date, time_ms, num = generate_split(
            n, seed=seed + i, split=split)
        write_cache(out_dir, split, user_id, video_id, X, y, aux,
                    date=date, time_ms=time_ms, num=num)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build a synthetic KuaiRand-Pure-shaped cache for dev/testing.")
    ap.add_argument("--out-dir", default="data/cache")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(Path(args.out_dir), seed=args.seed)
    print(f"synthetic cache written to {args.out_dir}")
