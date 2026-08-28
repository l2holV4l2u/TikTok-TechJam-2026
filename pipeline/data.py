"""KuaiRand-Pure cache format: contiguous-int feature arrays + click label as memmapped .npy.

Splits are organizer-fixed by date, taken from the two provided log files:
  train = log_standard_4_08_to_4_21_pure.csv, all rows (1,141,112)
  valid = log_standard_4_22_to_5_08_pure.csv, date 20220422-20220428 (124,909)
  test  = log_standard_4_22_to_5_08_pure.csv, date 20220429-20220508 (170,588)
`date` in the raw CSV is a plain decimal YYYYMMDD string (e.g. "20220411"), so
fixed-width lexicographic comparison against the bounds below is exact.

Row order per split is the source log file's line order (date-filtered, never
re-sorted) -- this is the order submission row_id indexes into.

LEAKAGE WARNING: play_time_ms, is_click, profile_stay_time, comment_stay_time,
is_profile_enter, is_like, is_follow, is_comment, is_forward, is_hate are post-click
outcomes of the row being scored. They live in Split.aux ONLY, never in Split.X --
using them as inputs to predict long_view on the same row leaks the label.
"""
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Capacities are upper-bound embedding budgets, not exact counts -- the real vocab
# (built from TRAIN ONLY, see _build_vocabs_and_edges) may use fewer ids. 0 = OOV/unseen.
FEATURE_CARDINALITIES: dict[str, int] = {
    "user_id": 30_000,
    "video_id": 8_000,
    "tab": 20,
    "hour": 32,
    "user_active_degree": 16,
    "is_lowactive_period": 4,
    "is_live_streamer": 4,
    "is_video_author": 4,
    "follow_user_num_range": 16,
    "fans_user_num_range": 16,
    "friend_user_num_range": 16,
    "register_days_range": 16,
    "onehot_feat0": 8,
    "onehot_feat1": 12,
    "onehot_feat2": 64,
    "onehot_feat3": 1600,
    "onehot_feat4": 24,
    "onehot_feat5": 48,
    "onehot_feat6": 8,
    "onehot_feat7": 160,
    "onehot_feat8": 512,
    "onehot_feat9": 12,
    "onehot_feat10": 8,
    "onehot_feat11": 8,
    "onehot_feat12": 8,
    "onehot_feat13": 8,
    "onehot_feat14": 8,
    "onehot_feat15": 8,
    "onehot_feat16": 8,
    "onehot_feat17": 8,
    "author_id": 7_000,
    "video_type": 8,
    "upload_type": 20,
    "music_type": 12,
    "tag": 64,
    "duration_bucket": 12,          # quantile bucket of the log row's own duration_ms (pre-click, safe)
    "register_days_bucket": 12,     # quantile bucket of the user's account age (pre-click, safe)
}

# Continuous features. Split.X is categorical ids only, so until now every numeric quantity in
# the dataset was unreachable: raw video length, the user's real follower counts, and the whole
# of video_features_statistic (52 columns the harness never opened). All of these are attributes
# known BEFORE the impression is scored -- they describe the user and the video, not what
# happened on this row -- which is what separates them from the post-click signals in aux.
#
# The video statistics carry no declared time window. Measured as standalone rankers they score
# train 0.5910 > valid 0.5804 > test 0.5740, i.e. worst on the split they would leak into, and
# their `counts` field spans 45-181 daily records against a 30-day log -- both consistent with
# historical attributes rather than anything fitted on the evaluation window.
NUMERIC_LOG: tuple[str, ...] = ("duration_ms",)
# Prefixed because `follow_user_num` exists in BOTH user_features (people this user follows)
# and video_features_statistic (people who followed via this video). They are different
# quantities; unprefixed, the video column silently overwrote the user column and the brief
# described one while the cache held the other.
NUMERIC_USER_SRC: tuple[str, ...] = ("follow_user_num", "fans_user_num", "friend_user_num",
                                     "register_days")
NUMERIC_USER: tuple[str, ...] = tuple("user_" + c for c in NUMERIC_USER_SRC)
NUMERIC_VIDEO_STAT: tuple[str, ...] = (
    # every column of video_features_statistic. We screened these individually and only
    # three reach 0.55 as standalone rankers -- but that screen scores each column ALONE,
    # and a feature can be uninformative by itself and useful in a cross. The FM exists to
    # find those. Judging them for the agent is the mistake; exposing them is cheap
    # (~180MB of float32) and lets it decide.
    "counts", "show_cnt", "show_user_num", "play_cnt", "play_user_num", "play_duration",
    "complete_play_cnt", "complete_play_user_num", "valid_play_cnt", "valid_play_user_num",
    "long_time_play_cnt", "long_time_play_user_num", "short_time_play_cnt",
    "short_time_play_user_num", "play_progress", "comment_stay_duration", "like_cnt",
    "like_user_num", "click_like_cnt", "double_click_cnt", "cancel_like_cnt",
    "cancel_like_user_num", "comment_cnt", "comment_user_num", "direct_comment_cnt",
    "reply_comment_cnt", "delete_comment_cnt", "delete_comment_user_num", "comment_like_cnt",
    "comment_like_user_num", "follow_cnt", "follow_user_num", "cancel_follow_cnt",
    "cancel_follow_user_num", "share_cnt", "share_user_num", "download_cnt",
    "download_user_num", "report_cnt", "report_user_num", "reduce_similar_cnt",
    "reduce_similar_user_num", "collect_cnt", "collect_user_num", "cancel_collect_cnt",
    "cancel_collect_user_num", "direct_comment_user_num", "reply_comment_user_num",
    "share_all_cnt", "share_all_user_num", "outsite_share_all_cnt",
)
NUMERIC_FEATURES: tuple[str, ...] = NUMERIC_LOG + NUMERIC_USER + NUMERIC_VIDEO_STAT


# Post-click feedback signals -> dtype. aux-only, see LEAKAGE WARNING above.
AUX_DTYPES: dict[str, str] = {
    "is_like": "int8",
    "is_follow": "int8",
    "is_comment": "int8",
    "is_forward": "int8",
    "is_hate": "int8",
    "is_click": "int8",
    "is_profile_enter": "int8",
    "play_time_ms": "float32",
    "profile_stay_time": "float32",
    "comment_stay_time": "float32",
}

_BUCKET_FEATURES = ("duration_bucket", "register_days_bucket")
_VOCAB_FEATURES = tuple(f for f in FEATURE_CARDINALITIES if f not in _BUCKET_FEATURES)
N_QUANTILE_BUCKETS = 10  # deciles fit on train only; bucket id = searchsorted(edges, v) + 1

# KUAIRAND_VARIANT picks which release to read: "pure" (the scored task) or "1k" (the bonus
# transfer set). Both ship identical column headers and date windows, so only the filenames,
# the row counts and the item-vocab budgets differ.
_VARIANT = os.environ.get("KUAIRAND_VARIANT", "pure").lower()

_TRAIN_LOG = f"log_standard_4_08_to_4_21_{_VARIANT}.csv"
_VALID_TEST_LOG = f"log_standard_4_22_to_5_08_{_VARIANT}.csv"
_USER_FEATURES_FILE = f"user_features_{_VARIANT}.csv"
_VIDEO_FEATURES_FILE = f"video_features_basic_{_VARIANT}.csv"
_VIDEO_STAT_FILE = f"video_features_statistic_{_VARIANT}.csv"

_TRAIN_RANGE = ("20220408", "20220421")
_VALID_RANGE = ("20220422", "20220428")
_TEST_RANGE = ("20220429", "20220508")
_ROWS_BY_VARIANT = {
    "pure": {"train": 1_141_112, "valid": 124_909, "test": 170_588},
    "1k": {"train": 5_055_984, "valid": 2_524_980, "test": 4_132_081},
}
_EXPECTED_ROWS = _ROWS_BY_VARIANT[_VARIANT]

if _VARIANT == "1k":
    # 1K logs 1,000 users against the full catalogue: 2.1M distinct train videos at 2.4
    # impressions each, so most ids carry no learnable signal. The cap keeps the head and
    # sends the tail to OOV; Pure's 8,000 budget would have discarded almost everything.
    FEATURE_CARDINALITIES["video_id"] = 250_000
    FEATURE_CARDINALITIES["author_id"] = 150_000


@dataclass
class Split:
    user_id: np.ndarray               # int64, (n,) raw ids, for GAUC grouping and submission
    video_id: np.ndarray              # int64, (n,) raw ids, for submission
    X: dict[str, np.ndarray]          # feature name -> int64 (n,), contiguous ids, 0 = unseen/OOV
    y: np.ndarray                     # int8 (n,), long_view -- the officially scored label
    aux: dict[str, np.ndarray]        # other feedback signals, int8/float32
    date: np.ndarray | None = None    # int32 (n,), YYYYMMDD of the impression
    time_ms: np.ndarray | None = None # int64 (n,), epoch ms of the impression (pre-click, safe)
    num: dict[str, np.ndarray] | None = None  # float32 (n,), continuous features, NaN = unknown


LABEL = "long_view"  # starter kit data.py: LABEL = 'long_view'; the prose saying "click" is wrong


def _cache_root() -> Path:
    return Path(os.environ.get("KUAIRAND_CACHE_DIR", "data/cache"))  # env override lets tests use an isolated cache


def load(split: str) -> Split:
    if split not in ("train", "valid", "test"):
        raise ValueError(f"split must be one of train/valid/test, got {split!r}")

    split_dir = _cache_root() / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"no cache for split {split!r} at {split_dir}. Build one with: "
            "python -m pipeline.synth (synthetic dev cache), or "
            "pipeline.data.build_cache(raw_dir, out_dir) for real KuaiRand-Pure data"
        )

    user_id = np.load(split_dir / "user_id.npy", mmap_mode="r")
    video_id = np.load(split_dir / "video_id.npy", mmap_mode="r")
    y = np.load(split_dir / "y.npy", mmap_mode="r")
    X = {feat: np.load(split_dir / f"X_{feat}.npy", mmap_mode="r") for feat in FEATURE_CARDINALITIES}
    aux = {name: np.load(split_dir / f"aux_{name}.npy", mmap_mode="r") for name in AUX_DTYPES}
    # the impression date. It is what the splits are defined by, and the train window is 13 days
    # wide, so withholding it silently rules out recency weighting and time-based validation --
    # a whole family of methods aimed at the drift between the train and evaluation windows.
    date_path = split_dir / "date.npy"
    date = np.load(date_path, mmap_mode="r") if date_path.exists() else None
    # the impression timestamp. `date` only orders across days and `hour` is time-of-day, so
    # without this a user's impressions cannot be put in order at all -- which rules out every
    # sequence model, session split, inter-arrival feature and recency decay. It is the
    # impression time, known before the click, so it is a feature and not an outcome.
    tms_path = split_dir / "time_ms.npy"
    time_ms = np.load(tms_path, mmap_mode="r") if tms_path.exists() else None
    num = {f: np.load(split_dir / f"num_{f}.npy", mmap_mode="r")
           for f in NUMERIC_FEATURES if (split_dir / f"num_{f}.npy").exists()} or None
    return Split(user_id=user_id, video_id=video_id, X=X, y=y, aux=aux, date=date,
                 time_ms=time_ms, num=num)


def write_cache(out_dir, split: str, user_id: np.ndarray, video_id: np.ndarray,
                 X: dict[str, np.ndarray], y: np.ndarray, aux: dict[str, np.ndarray]) -> None:
    """Write a fully-materialized split (used by pipeline.synth; build_cache streams instead)."""
    split_dir = Path(out_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "user_id.npy", np.asarray(user_id, dtype=np.int64))
    np.save(split_dir / "video_id.npy", np.asarray(video_id, dtype=np.int64))
    np.save(split_dir / "y.npy", np.asarray(y, dtype=np.int8))
    for feat, arr in X.items():
        np.save(split_dir / f"X_{feat}.npy", np.asarray(arr, dtype=np.int64))
    for name, arr in aux.items():
        np.save(split_dir / f"aux_{name}.npy", np.asarray(arr, dtype=AUX_DTYPES[name]))


# ---------------------------------------------------------------------------
# Real KuaiRand-Pure ingestion
# ---------------------------------------------------------------------------

def _load_user_features(path: Path) -> dict[int, dict[str, str]]:
    users: dict[int, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {h: i for i, h in enumerate(header)}
        onehot_cols = [f"onehot_feat{i}" for i in range(18)]
        for row in r:
            uid = int(row[idx["user_id"]])
            rec = {c: row[idx[c]] for c in
                   ("user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author",
                    "follow_user_num_range", "fans_user_num_range", "friend_user_num_range", "register_days_range")}
            rec["register_days"] = row[idx["register_days"]]  # numeric, kept raw for quantile bucketing
            # the raw counts behind the *_range buckets. Without these NUMERIC_USER resolves to
            # "" and lands in Split.num as 100% NaN -- which is exactly what the agent's own EDA
            # reported (m1.00) on the first run after the numeric channel was added.
            for c in ("follow_user_num", "fans_user_num", "friend_user_num"):
                if c in idx:
                    rec[c] = row[idx[c]]
            for c in onehot_cols:
                rec[c] = row[idx[c]]
            users[uid] = rec
    return users


# video record layout, kept as a tuple rather than a dict: 1K has 2.7M videos and a dict per
# video costs ~3 GB of the 16 GB box before a single row is converted.
_V_AUTHOR, _V_TYPE, _V_UPLOAD, _V_MUSIC, _V_TAG = range(5)
_EMPTY_VIDEO = ("", "", "", "", "")


def _load_video_features(path: Path) -> dict[int, tuple[str, ...]]:
    videos: dict[int, tuple[str, ...]] = {}
    intern = sys.intern
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {h: i for i, h in enumerate(header)}
        for row in r:
            vid = int(row[idx["video_id"]])
            tag_raw = row[idx["tag"]]
            videos[vid] = (
                intern(row[idx["author_id"]]),
                intern(row[idx["video_type"]]),
                intern(row[idx["upload_type"]]),
                intern(row[idx["music_type"]]),
                intern(tag_raw.split(",")[0] if tag_raw else ""),  # first tag only; multi-hot not modeled
            )
    return videos



def _load_video_stats(path: Path) -> dict[int, tuple[float, ...]]:
    """video_features_statistic: per-video historical aggregates. Missing file -> no numerics.

    KuaiRand-1K ships this as a 3.4GB file covering 2.7M videos; a cache built without it simply
    omits those columns rather than failing, and Split.num reports only what is present.
    """
    if not path.exists():
        return {}
    out: dict[int, tuple[float, ...]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {h: i for i, h in enumerate(header)}
        cols = [idx[c] for c in NUMERIC_VIDEO_STAT if c in idx]
        if len(cols) != len(NUMERIC_VIDEO_STAT):
            return {}
        for row in r:
            vals = []
            for c in cols:
                try:
                    vals.append(float(row[c]))
                except (ValueError, IndexError):
                    vals.append(float("nan"))
            out[int(row[idx["video_id"]])] = tuple(vals)
    return out


def _row_values(uid: int, vid: int, hourmin: int, tab: str,
                 users: dict[int, dict[str, str]], videos: dict[int, tuple[str, ...]]) -> dict[str, str]:
    u = users.get(uid, {})
    v = videos.get(vid, _EMPTY_VIDEO)
    vals = {
        "user_id": str(uid),
        "video_id": str(vid),
        "tab": tab,
        "hour": str(hourmin // 100),
        "user_active_degree": u.get("user_active_degree", ""),
        "is_lowactive_period": u.get("is_lowactive_period", ""),
        "is_live_streamer": u.get("is_live_streamer", ""),
        "is_video_author": u.get("is_video_author", ""),
        "follow_user_num_range": u.get("follow_user_num_range", ""),
        "fans_user_num_range": u.get("fans_user_num_range", ""),
        "friend_user_num_range": u.get("friend_user_num_range", ""),
        "register_days_range": u.get("register_days_range", ""),
        "author_id": v[_V_AUTHOR],
        "video_type": v[_V_TYPE],
        "upload_type": v[_V_UPLOAD],
        "music_type": v[_V_MUSIC],
        "tag": v[_V_TAG],
    }
    for i in range(18):
        vals[f"onehot_feat{i}"] = u.get(f"onehot_feat{i}", "")
    return vals


def _iter_log_rows(path: Path, date_lo: str, date_hi: str):
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {h: i for i, h in enumerate(header)}
        for row in r:
            if date_lo <= row[idx["date"]] <= date_hi:
                yield row, idx


def _build_vocabs_and_edges(train_log: Path, users: dict, videos: dict, vocab_dir: Path):
    counts = {f: {} for f in _VOCAB_FEATURES}
    durations: list[float] = []
    reg_days: list[float] = []

    for row, idx in _iter_log_rows(train_log, *_TRAIN_RANGE):
        uid = int(row[idx["user_id"]])
        vid = int(row[idx["video_id"]])
        vals = _row_values(uid, vid, int(row[idx["hourmin"]]), row[idx["tab"]], users, videos)
        for f in _VOCAB_FEATURES:
            counts[f][vals[f]] = counts[f].get(vals[f], 0) + 1
        durations.append(float(row[idx["duration_ms"]]))
        reg_days.append(float(users.get(uid, {}).get("register_days", 0.0) or 0.0))

    vocabs: dict[str, dict[str, int]] = {}
    for f in _VOCAB_FEATURES:
        cap = FEATURE_CARDINALITIES[f]
        ranked = sorted(counts[f], key=counts[f].get, reverse=True)[: cap - 1]  # 0 reserved for OOV
        vocab = {v: i + 1 for i, v in enumerate(ranked)}
        (vocab_dir / f"{f}.json").write_text(json.dumps(vocab), encoding="utf-8")
        vocabs[f] = vocab

    quantiles = np.linspace(0, 1, N_QUANTILE_BUCKETS + 1)[1:-1]
    duration_edges = np.quantile(np.array(durations), quantiles).tolist()
    reg_edges = np.quantile(np.array(reg_days), quantiles).tolist()
    (vocab_dir / "duration_bucket_edges.json").write_text(json.dumps(duration_edges), encoding="utf-8")
    (vocab_dir / "register_days_bucket_edges.json").write_text(json.dumps(reg_edges), encoding="utf-8")
    return vocabs, duration_edges, reg_edges


def _convert(log_path: Path, date_lo: str, date_hi: str, split: str,
             users: dict, videos: dict, vocabs: dict, duration_edges: list[float], reg_edges: list[float],
             out_dir: Path, stats: dict[int, tuple[float, ...]] | None = None) -> None:
    n_expected = _EXPECTED_ROWS[split]
    split_dir = Path(out_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)

    user_id_arr = np.lib.format.open_memmap(split_dir / "user_id.npy", mode="w+", dtype=np.int64, shape=(n_expected,))
    video_id_arr = np.lib.format.open_memmap(split_dir / "video_id.npy", mode="w+", dtype=np.int64, shape=(n_expected,))
    y_arr = np.lib.format.open_memmap(split_dir / "y.npy", mode="w+", dtype=np.int8, shape=(n_expected,))
    date_arr = np.lib.format.open_memmap(split_dir / "date.npy", mode="w+", dtype=np.int32, shape=(n_expected,))
    time_ms_arr = np.lib.format.open_memmap(split_dir / "time_ms.npy", mode="w+", dtype=np.int64, shape=(n_expected,))
    stats = stats or {}
    num_names = NUMERIC_LOG + NUMERIC_USER + (NUMERIC_VIDEO_STAT if stats else ())
    num_arrs = {f: np.lib.format.open_memmap(split_dir / f"num_{f}.npy", mode="w+",
                dtype=np.float32, shape=(n_expected,)) for f in num_names}
    _empty_stat = tuple([float("nan")] * len(NUMERIC_VIDEO_STAT))
    X_arrs = {f: np.lib.format.open_memmap(split_dir / f"X_{f}.npy", mode="w+", dtype=np.int64, shape=(n_expected,))
              for f in FEATURE_CARDINALITIES}
    aux_arrs = {name: np.lib.format.open_memmap(split_dir / f"aux_{name}.npy", mode="w+",
                dtype=np.dtype(dtype), shape=(n_expected,)) for name, dtype in AUX_DTYPES.items()}

    duration_edges_arr = np.array(duration_edges)
    reg_edges_arr = np.array(reg_edges)

    i = 0
    for row, idx in _iter_log_rows(log_path, date_lo, date_hi):
        uid = int(row[idx["user_id"]])
        vid = int(row[idx["video_id"]])
        vals = _row_values(uid, vid, int(row[idx["hourmin"]]), row[idx["tab"]], users, videos)

        user_id_arr[i] = uid
        video_id_arr[i] = vid
        date_arr[i] = int(row[idx["date"]])
        time_ms_arr[i] = int(row[idx["time_ms"]])
        for f in NUMERIC_LOG:
            try:
                num_arrs[f][i] = float(row[idx[f]])
            except (ValueError, KeyError):
                num_arrs[f][i] = float("nan")
        u_rec = users.get(uid, {})
        for src, f in zip(NUMERIC_USER_SRC, NUMERIC_USER):
            try:
                num_arrs[f][i] = float(u_rec.get(src, ""))
            except ValueError:
                num_arrs[f][i] = float("nan")
        if stats:
            st = stats.get(vid, _empty_stat)
            for j, f in enumerate(NUMERIC_VIDEO_STAT):
                num_arrs[f][i] = st[j]
        for f, vocab in vocabs.items():
            X_arrs[f][i] = vocab.get(vals[f], 0)

        duration_ms = float(row[idx["duration_ms"]])
        reg_days = float(users.get(uid, {}).get("register_days", 0.0) or 0.0)
        X_arrs["duration_bucket"][i] = int(np.searchsorted(duration_edges_arr, duration_ms)) + 1
        X_arrs["register_days_bucket"][i] = int(np.searchsorted(reg_edges_arr, reg_days)) + 1

        y_arr[i] = int(row[idx[LABEL]] != "0")
        for name in AUX_DTYPES:
            aux_arrs[name][i] = float(row[idx[name]])

        i += 1

    if i != n_expected:
        raise ValueError(f"split {split!r}: expected {n_expected} rows from {log_path}, got {i} -- date filter mismatch")

    for arr in (user_id_arr, video_id_arr, y_arr, date_arr, time_ms_arr,
                *X_arrs.values(), *aux_arrs.values(), *num_arrs.values()):
        arr.flush()


def build_cache(raw_dir, out_dir) -> None:
    """Convert raw KuaiRand-Pure CSVs into the memmap cache. Vocabs are built from train only."""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    vocab_dir = out_dir / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)

    users = _load_user_features(raw_dir / _USER_FEATURES_FILE)
    videos = _load_video_features(raw_dir / _VIDEO_FEATURES_FILE)
    stats = _load_video_stats(raw_dir / _VIDEO_STAT_FILE)

    train_log = raw_dir / _TRAIN_LOG
    valid_test_log = raw_dir / _VALID_TEST_LOG

    have_vocabs = all((vocab_dir / f"{f}.json").exists() for f in _VOCAB_FEATURES)
    have_edges = (vocab_dir / "duration_bucket_edges.json").exists() and (vocab_dir / "register_days_bucket_edges.json").exists()
    if have_vocabs and have_edges:
        vocabs = {f: json.loads((vocab_dir / f"{f}.json").read_text(encoding="utf-8")) for f in _VOCAB_FEATURES}
        duration_edges = json.loads((vocab_dir / "duration_bucket_edges.json").read_text(encoding="utf-8"))
        reg_edges = json.loads((vocab_dir / "register_days_bucket_edges.json").read_text(encoding="utf-8"))
    else:
        vocabs, duration_edges, reg_edges = _build_vocabs_and_edges(train_log, users, videos, vocab_dir)

    _convert(train_log, *_TRAIN_RANGE, "train", users, videos, vocabs, duration_edges, reg_edges, out_dir, stats)
    _convert(valid_test_log, *_VALID_RANGE, "valid", users, videos, vocabs, duration_edges, reg_edges, out_dir, stats)
    _convert(valid_test_log, *_TEST_RANGE, "test", users, videos, vocabs, duration_edges, reg_edges, out_dir, stats)
