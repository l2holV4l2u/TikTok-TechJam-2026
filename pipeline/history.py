"""Leakage-safe entity histories derived exclusively from the train split.

The dataset's supplied video statistics average outcomes over the full month, which overlaps
the fixed validation and test windows. This module recovers the useful mechanism without the
future-window leak: aggregate only training rows, and use leave-one-out values on training rows
so an example never receives its own target as an input.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from pipeline.data import AUX_DTYPES, FEATURE_CARDINALITIES, _cache_root, load

ALLOWED_KEYS = ("video_id", "author_id")


def _signal_arrays(train) -> dict[str, np.ndarray]:
    signals = {"long_view": np.asarray(train.y, dtype=np.float64)}
    for name, dtype in AUX_DTYPES.items():
        value = np.asarray(train.aux[name], dtype=np.float64)
        if dtype != "int8":
            value = np.log1p(np.maximum(value, 0.0))
        signals[name] = value
    return signals


def _artifact_path(key: str, n_rows: int) -> Path | None:
    root = os.environ.get("RUN_ARTIFACTS")
    return Path(root) / f"train_history_{key}_{n_rows}.npz" if root else None


@lru_cache(maxsize=8)
def _table(cache_root: str, key: str) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    if key not in ALLOWED_KEYS:
        raise ValueError(f"history key must be one of {ALLOWED_KEYS}, got {key!r}")
    train = load("train")
    entity = np.asarray(train.X[key], dtype=np.int64)
    size = int(FEATURE_CARDINALITIES[key])
    artifact = _artifact_path(key, len(entity))

    if artifact is not None and artifact.exists():
        try:
            with np.load(artifact, allow_pickle=False) as saved:
                count = np.asarray(saved["count"], dtype=np.float64)
                sums = {name: np.asarray(saved[f"sum_{name}"], dtype=np.float64)
                        for name in ("long_view", *AUX_DTYPES)}
                priors = {name: float(saved[f"prior_{name}"]) for name in sums}
            if len(count) == size:
                return count, sums, priors
        except (OSError, ValueError, KeyError):
            pass

    count = np.bincount(entity, minlength=size).astype(np.float64)
    signals = _signal_arrays(train)
    sums = {name: np.bincount(entity, weights=value, minlength=size).astype(np.float64)
            for name, value in signals.items()}
    priors = {name: float(np.mean(value)) for name, value in signals.items()}

    if artifact is not None:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        payload = {"count": count}
        payload.update({f"sum_{name}": value for name, value in sums.items()})
        payload.update({f"prior_{name}": np.array(value) for name, value in priors.items()})
        np.savez(artifact, **payload)
    return count, sums, priors


def historical_features(split_name: str, key: str = "video_id",
                        smoothing: float = 20.0) -> dict[str, np.ndarray]:
    """Return per-row train-only history features for ``train|valid|test``.

    Binary outcomes become smoothed rates; continuous feedback becomes a smoothed mean after
    log1p. Entity id 0 is OOV and deliberately receives only global priors rather than an
    aggregate that mixes unrelated tail entities.
    """
    if split_name not in {"train", "valid", "test"}:
        raise ValueError("split_name must be train, valid, or test")
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")

    target = load(split_name)
    entity = np.asarray(target.X[key], dtype=np.int64)
    count, sums, priors = _table(str(_cache_root().resolve()), key)
    seen = entity != 0
    denom = count[entity].copy()
    current = _signal_arrays(target) if split_name == "train" else None
    if current is not None:
        denom -= 1.0

    out = {f"{key}_train_count_log1p": np.where(
        seen, np.log1p(np.maximum(denom, 0.0)), 0.0).astype(np.float32)}
    for name, total in sums.items():
        numerator = total[entity].copy()
        if current is not None:
            numerator -= current[name]
        value = (numerator + smoothing * priors[name]) / (denom + smoothing)
        value[~seen] = priors[name]
        suffix = "rate" if name == "long_view" or AUX_DTYPES.get(name) == "int8" else "logmean"
        out[f"{key}_{name}_{suffix}"] = value.astype(np.float32)
    return out


def clear_cache() -> None:
    """Test/helper hook for switching KUAIRAND_CACHE_DIR inside one Python process."""
    _table.cache_clear()
