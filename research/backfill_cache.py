"""Add newly-exposed columns to an existing cache without a full rebuild.

`build_cache` writes everything, but rebuilding KuaiRand-1K costs 24 minutes and the raw CSVs
are already on disk. This re-reads them in the cache's own row order and writes only the columns
that were added later -- time_ms and the Split.num channel.

  python -m research.backfill_cache                                  # Pure
  KUAIRAND_VARIANT=1k KUAIRAND_CACHE_DIR=data/cache_1k \
      python -m research.backfill_cache --raw data/raw/KuaiRand-1K/data

Never run this against a cache a live agent run is reading: the generated scripts load these
arrays per iteration, so rewriting them mid-run changes the data underneath an experiment.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from pipeline.data import (NUMERIC_FEATURES, NUMERIC_LOG, NUMERIC_USER,
                           NUMERIC_USER_SRC, NUMERIC_VIDEO_STAT,
                           _EXPECTED_ROWS, _TEST_RANGE, _TRAIN_RANGE, _VALID_RANGE,
                           _TRAIN_LOG, _VALID_TEST_LOG, _USER_FEATURES_FILE, _VIDEO_STAT_FILE,
                           _cache_root, _iter_log_rows, _load_user_features, _load_video_stats)

NAN = float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/KuaiRand-Pure/data")
    args = ap.parse_args()
    raw, cache = Path(args.raw), _cache_root()

    users = _load_user_features(raw / _USER_FEATURES_FILE)
    stats = _load_video_stats(raw / _VIDEO_STAT_FILE)
    print(f"user records {len(users):,} | videos with statistics {len(stats):,}"
          f"{' -- statistics file absent, those columns skipped' if not stats else ''}")
    names = NUMERIC_LOG + NUMERIC_USER + (NUMERIC_VIDEO_STAT if stats else ())
    empty = tuple([NAN] * len(NUMERIC_VIDEO_STAT))

    for split, log, rng in (("train", _TRAIN_LOG, _TRAIN_RANGE),
                            ("valid", _VALID_TEST_LOG, _VALID_RANGE),
                            ("test", _VALID_TEST_LOG, _TEST_RANGE)):
        t0, n = time.perf_counter(), _EXPECTED_ROWS[split]
        d = cache / split
        tms = np.lib.format.open_memmap(d / "time_ms.npy", mode="w+", dtype=np.int64, shape=(n,))
        num = {f: np.lib.format.open_memmap(d / f"num_{f}.npy", mode="w+",
               dtype=np.float32, shape=(n,)) for f in names}
        i = 0
        for row, idx in _iter_log_rows(raw / log, *rng):
            uid, vid = int(row[idx["user_id"]]), int(row[idx["video_id"]])
            tms[i] = int(row[idx["time_ms"]])
            u = users.get(uid, {})
            for f in NUMERIC_LOG:
                try:
                    num[f][i] = float(row[idx[f]])
                except (ValueError, KeyError):
                    num[f][i] = NAN
            for src, f in zip(NUMERIC_USER_SRC, NUMERIC_USER):
                try:
                    num[f][i] = float(u.get(src, ""))
                except ValueError:
                    num[f][i] = NAN
            if stats:
                st = stats.get(vid, empty)
                for j, f in enumerate(NUMERIC_VIDEO_STAT):
                    num[f][i] = st[j]
            i += 1
        if i != n:
            raise ValueError(f"{split}: wrote {i} rows, cache holds {n} -- row order diverged")
        for a in (tms, *num.values()):
            a.flush()
        dead = [f for f, a in num.items() if np.isnan(np.asarray(a)).all()]
        print(f"  {split:6s} {n:>9,} rows in {time.perf_counter()-t0:5.1f}s | "
              f"{len(num)} numeric columns"
              + (f" | WARNING all-NaN: {dead}" if dead else ""))


if __name__ == "__main__":
    main()
