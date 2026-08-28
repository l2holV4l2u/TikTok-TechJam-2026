"""AliCCP cache format: contiguous-int feature arrays + click/conversion labels as memmapped .npy."""
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reduced 19-field sparse schema from docs/nise-aliccp-research.md (torch-rechub's
# preprocess_ali_ccp.py field list: 101,121,122,124,125,126,127,128,129,205,206,207,
# 210,216,508,509,702,853,301). Names kept as raw Alimama field codes since the doc
# does not confirm per-field semantics or cardinalities -- every value below is a guess.
# Excludes the 4 multi-hot behavior-sequence fields (109_14/110_14/127_14/150_14): this
# schema is scalar-per-row (see Split.X), sequences need different modeling than a
# single int64 per feature.
FEATURE_CARDINALITIES: dict[str, int] = {
    "f101": 200_000,  # ASSUMPTION: user_id-like, ~400K raw users, capped post freq-filter
    "f121": 20,        # ASSUMPTION: discretized user profile attribute
    "f122": 20,        # ASSUMPTION: discretized user profile attribute
    "f124": 20,        # ASSUMPTION: discretized user profile attribute
    "f125": 20,        # ASSUMPTION: discretized user profile attribute
    "f126": 20,        # ASSUMPTION: discretized user profile attribute
    "f127": 20,        # ASSUMPTION: discretized user profile attribute
    "f128": 20,        # ASSUMPTION: discretized user profile attribute
    "f129": 20,        # ASSUMPTION: discretized user profile attribute
    "f210": 5_000,      # ASSUMPTION: user intention/interest node id
    "f205": 500_000,   # ASSUMPTION: item/adgroup id-like, ~4.3M raw items, capped
    "f206": 15_000,    # ASSUMPTION: category id
    "f207": 100_000,   # ASSUMPTION: campaign/shop id
    "f216": 50_000,    # ASSUMPTION: brand id
    "f301": 10,         # ASSUMPTION: ad slot position (Merlin loader's "context: position")
    "f508": 20_000,    # ASSUMPTION: user-item cross/combination feature
    "f509": 20_000,    # ASSUMPTION: user-item cross/combination feature
    "f702": 20_000,    # ASSUMPTION: user-item cross/combination feature
    "f853": 20_000,    # ASSUMPTION: user-item cross/combination feature
}

# ASSUMPTION: per docs/nise-aliccp-research.md these 10 fields are the "common"
# (user/context) side, joined in via common_feature_index; the rest live directly
# on each sample_skeleton row (item side + cross features + position).
_COMMON_FIELDS = {"f101", "f121", "f122", "f124", "f125", "f126", "f127", "f128", "f129", "f210"}
_SAMPLE_FIELDS = set(FEATURE_CARDINALITIES) - _COMMON_FIELDS

MIN_CLICK_POSITIVES = 30  # floor on click=1 rows kept per subsample
MIN_CONV_POSITIVES = 15   # floor on click=1,conv=1 rows kept per subsample -- conversions are rare enough that fraction*count can round to 0


@dataclass
class Split:
    X: dict[str, np.ndarray]
    y_click: np.ndarray
    y_conv: np.ndarray


def _cache_root() -> Path:
    return Path(os.environ.get("ALICCP_CACHE_DIR", "data/cache"))  # env override lets tests use an isolated cache


def _take(pool: np.ndarray, fraction: float, min_required: int, rng: np.random.Generator, label: str) -> np.ndarray:
    if pool.size < min_required:
        raise ValueError(f"stratum {label!r} has only {pool.size} rows, need >= {min_required} to subsample safely")
    k = min(max(math.ceil(pool.size * fraction), min_required), pool.size)  # ceil, never round a rare-positive stratum down
    return rng.choice(pool, size=k, replace=False)


def _stratified_indices(y_click: np.ndarray, y_conv: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    y_click = np.asarray(y_click)
    y_conv = np.asarray(y_conv)
    no_click = np.flatnonzero(y_click == 0)
    click_no_conv = np.flatnonzero((y_click == 1) & (y_conv == 0))
    click_conv = np.flatnonzero((y_click == 1) & (y_conv == 1))

    rng = np.random.default_rng(seed)
    idx = np.concatenate([
        _take(no_click, fraction, min(1, no_click.size), rng, "no_click"),
        _take(click_no_conv, fraction, MIN_CLICK_POSITIVES, rng, "click_no_conv"),
        _take(click_conv, fraction, MIN_CONV_POSITIVES, rng, "click_conv"),
    ])
    idx.sort()
    return idx


def load(split: str, fraction: float = 1.0, seed: int = 0) -> Split:
    if split not in ("train", "valid", "test"):
        raise ValueError(f"split must be one of train/valid/test, got {split!r}")
    if not (0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    split_dir = _cache_root() / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"no cache for split {split!r} at {split_dir}. Build one with: "
            "python -m pipeline.synth (synthetic dev cache), or "
            "pipeline.data.build_cache(raw_dir, out_dir) for real AliCCP data"
        )

    y_click = np.load(split_dir / "y_click.npy", mmap_mode="r")
    y_conv = np.load(split_dir / "y_conv.npy", mmap_mode="r")
    X = {feat: np.load(split_dir / f"X_{feat}.npy", mmap_mode="r") for feat in FEATURE_CARDINALITIES}

    if fraction >= 1.0:
        return Split(X=X, y_click=y_click, y_conv=y_conv)

    idx = _stratified_indices(y_click, y_conv, fraction, seed)
    return Split(
        X={feat: arr[idx] for feat, arr in X.items()},
        y_click=y_click[idx],
        y_conv=y_conv[idx],
    )


def write_cache(out_dir, split: str, X: dict[str, np.ndarray], y_click: np.ndarray, y_conv: np.ndarray) -> None:
    split_dir = Path(out_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for feat, arr in X.items():
        np.save(split_dir / f"X_{feat}.npy", np.asarray(arr, dtype=np.int64))
    np.save(split_dir / "y_click.npy", np.asarray(y_click, dtype=np.int8))
    np.save(split_dir / "y_conv.npy", np.asarray(y_conv, dtype=np.int8))


def _parse_feature_list(s: str) -> dict[str, str]:
    # ASSUMPTION: raw entries are "field_id:feat_id:feat_value" triplets joined by ','
    out: dict[str, str] = {}
    for entry in s.split(","):
        parts = entry.split(":")
        if len(parts) != 3:
            continue
        field_id, feat_id = parts[0], parts[1]
        key = f"f{field_id}"
        if key in FEATURE_CARDINALITIES:
            out[key] = feat_id
    return out


def _load_common_features(path: Path) -> dict[str, dict[str, str]]:
    # ASSUMPTION: headerless CSV "common_feature_index,feature_num,feature_list"
    common: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            common[row[0]] = _parse_feature_list(row[2])
    return common


def _load_or_build_vocabs(raw_dir: Path, vocab_dir: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] | None = None
    vocabs: dict[str, dict[str, int]] = {}
    for feat, card in FEATURE_CARDINALITIES.items():
        path = vocab_dir / f"{feat}.json"
        if path.exists():
            vocabs[feat] = json.loads(path.read_text(encoding="utf-8"))
            continue
        if counts is None:
            counts = _count_raw_values(raw_dir)
        ranked = sorted(counts[feat], key=counts[feat].get, reverse=True)[: card - 1]
        vocab = {v: i + 1 for i, v in enumerate(ranked)}  # 0 reserved for unseen/OOV
        path.write_text(json.dumps(vocab), encoding="utf-8")
        vocabs[feat] = vocab
    return vocabs


def _count_raw_values(raw_dir: Path) -> dict[str, dict[str, int]]:
    counts = {feat: {} for feat in FEATURE_CARDINALITIES}
    common = _load_common_features(raw_dir / "common_features_train.csv")
    with (raw_dir / "sample_skeleton_train.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            values = dict(common.get(row[3], {}))
            values.update(_parse_feature_list(row[5]))
            for feat, val in values.items():
                counts[feat][val] = counts[feat].get(val, 0) + 1
    return counts


def _assign_valid(sample_id: str, valid_frac: float = 0.1) -> bool:
    # ASSUMPTION: AliCCP ships only train/test (confirmed docs/nise-aliccp-research.md);
    # carve a deterministic valid split out of train via a stable hash of sample_id.
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 1000) < int(valid_frac * 1000)


def _convert(skeleton_path: Path, common_path: Path, dst_dir: Path, vocabs: dict[str, dict[str, int]], keep) -> None:
    common = _load_common_features(common_path)

    n = 0
    with skeleton_path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if keep(row[0]):
                n += 1

    dst_dir.mkdir(parents=True, exist_ok=True)
    arrays = {feat: np.lib.format.open_memmap(dst_dir / f"X_{feat}.npy", mode="w+", dtype=np.int64, shape=(n,))
              for feat in FEATURE_CARDINALITIES}
    y_click = np.lib.format.open_memmap(dst_dir / "y_click.npy", mode="w+", dtype=np.int8, shape=(n,))
    y_conv = np.lib.format.open_memmap(dst_dir / "y_conv.npy", mode="w+", dtype=np.int8, shape=(n,))

    i = 0
    with skeleton_path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            sample_id, click, conversion, common_idx, _feature_num, feature_list = row[0], row[1], row[2], row[3], row[4], row[5]
            if not keep(sample_id):
                continue
            values = dict(common.get(common_idx, {}))
            values.update(_parse_feature_list(feature_list))
            for feat in FEATURE_CARDINALITIES:
                arrays[feat][i] = vocabs[feat].get(values.get(feat), 0)
            y_click[i] = int(click)
            y_conv[i] = int(conversion)
            i += 1

    for arr in arrays.values():
        arr.flush()
    y_click.flush()
    y_conv.flush()


def build_cache(raw_dir, out_dir) -> None:
    """Convert raw AliCCP sample_skeleton_{split}.csv + common_features_{split}.csv into the memmap cache."""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    vocab_dir = out_dir / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)

    vocabs = _load_or_build_vocabs(raw_dir, vocab_dir)

    train_skeleton = raw_dir / "sample_skeleton_train.csv"
    train_common = raw_dir / "common_features_train.csv"
    _convert(train_skeleton, train_common, out_dir / "train", vocabs, keep=lambda sid: not _assign_valid(sid))
    _convert(train_skeleton, train_common, out_dir / "valid", vocabs, keep=_assign_valid)
    _convert(raw_dir / "sample_skeleton_test.csv", raw_dir / "common_features_test.csv", out_dir / "test", vocabs, keep=lambda sid: True)
