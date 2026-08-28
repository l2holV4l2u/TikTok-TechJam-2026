"""Plain-assert tests for pipeline.data. Run with: python -m tests.test_data"""
import csv
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from pipeline.data import (
    AUX_DTYPES,
    FEATURE_CARDINALITIES,
    _build_vocabs_and_edges,
    _load_user_features,
    _load_video_features,
    build_cache,
    load,
)
from pipeline.synth import build as build_synth

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REAL_RAW_DIR = _REPO_ROOT / "data" / "raw" / "KuaiRand-Pure" / "data"

# long_view is the scored label, not a leak column; is_click is a post-impression outcome
_LEAK_COLUMNS = {
    "play_time_ms", "is_click", "profile_stay_time", "comment_stay_time",
    "is_profile_enter", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
}

_USER_HEADER = [
    "user_id", "user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author",
    "follow_user_num", "follow_user_num_range", "fans_user_num", "fans_user_num_range",
    "friend_user_num", "friend_user_num_range", "register_days", "register_days_range",
] + [f"onehot_feat{i}" for i in range(18)]

_VIDEO_HEADER = [
    "video_id", "author_id", "video_type", "upload_dt", "upload_type", "visible_status",
    "video_duration", "server_width", "server_height", "music_id", "music_type", "tag",
]

_LOG_HEADER = [
    "user_id", "video_id", "date", "hourmin", "time_ms", "is_click", "is_like", "is_follow",
    "is_comment", "is_forward", "is_hate", "long_view", "play_time_ms", "duration_ms",
    "profile_stay_time", "comment_stay_time", "is_profile_enter", "is_rand", "tab",
]


def _use_temp_cache():
    tmp = Path(tempfile.mkdtemp(prefix="kuairand_cache_"))
    prev = os.environ.get("KUAIRAND_CACHE_DIR")
    os.environ["KUAIRAND_CACHE_DIR"] = str(tmp)
    return tmp, prev


def _restore(tmp: Path, prev):
    if prev is None:
        os.environ.pop("KUAIRAND_CACHE_DIR", None)
    else:
        os.environ["KUAIRAND_CACHE_DIR"] = prev
    shutil.rmtree(tmp, ignore_errors=True)


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _user_row(user_id: int) -> list:
    row = {h: "0" for h in _USER_HEADER}
    row["user_id"] = str(user_id)
    row["user_active_degree"] = "full_active"
    row["register_days_range"] = "31-60"
    row["register_days"] = "45"
    return [row[h] for h in _USER_HEADER]


def _video_row(video_id: int) -> list:
    row = {h: "0" for h in _VIDEO_HEADER}
    row["video_id"] = str(video_id)
    row["author_id"] = "1"
    row["video_type"] = "NORMAL"
    row["upload_type"] = "Web"
    row["music_type"] = "9.0"
    row["tag"] = "1"
    return [row[h] for h in _VIDEO_HEADER]


def _log_row(user_id: int, video_id: int, date: str, is_click: int) -> list:
    row = {h: "0" for h in _LOG_HEADER}
    row["user_id"] = str(user_id)
    row["video_id"] = str(video_id)
    row["date"] = date
    row["hourmin"] = "1230"
    row["time_ms"] = "1649675512388"
    row["is_click"] = str(is_click)
    row["duration_ms"] = "10000"
    row["tab"] = "0"
    return [row[h] for h in _LOG_HEADER]


def test_missing_cache_raises():
    tmp, prev = _use_temp_cache()
    try:
        try:
            load("train")
            assert False, "expected FileNotFoundError"
        except FileNotFoundError as e:
            assert "train" in str(e)
            assert "python -m pipeline.synth" in str(e)
    finally:
        _restore(tmp, prev)


def test_determinism_of_row_order():
    tmp, prev = _use_temp_cache()
    try:
        build_synth(tmp, sizes={"train": 2_000, "valid": 500, "test": 500}, seed=3)
        s1 = load("train")
        s2 = load("train")
        assert np.array_equal(np.asarray(s1.user_id), np.asarray(s2.user_id))
        assert np.array_equal(np.asarray(s1.video_id), np.asarray(s2.video_id))
        assert np.array_equal(np.asarray(s1.y), np.asarray(s2.y))
        for feat in FEATURE_CARDINALITIES:
            assert np.array_equal(np.asarray(s1.X[feat]), np.asarray(s2.X[feat]))
    finally:
        _restore(tmp, prev)


def test_no_leak_columns_in_X_synth_split():
    tmp, prev = _use_temp_cache()
    try:
        build_synth(tmp, sizes={"train": 200, "valid": 50, "test": 50}, seed=1)
        s = load("train")
        assert not (set(s.X) & _LEAK_COLUMNS), f"leak columns found in X: {set(s.X) & _LEAK_COLUMNS}"
        assert set(AUX_DTYPES) & _LEAK_COLUMNS == _LEAK_COLUMNS  # every leak column is tracked as aux
        assert set(s.aux) == set(AUX_DTYPES)
    finally:
        _restore(tmp, prev)


def test_no_leak_columns_in_feature_cardinalities():
    assert not (set(FEATURE_CARDINALITIES) & _LEAK_COLUMNS)


def test_generated_process_can_size_but_not_read_test_labels():
    tmp, prev = _use_temp_cache()
    old = os.environ.get("AGENT_HIDE_TEST_LABELS")
    try:
        build_synth(tmp, sizes={"train": 20, "valid": 10, "test": 7}, seed=9)
        os.environ["AGENT_HIDE_TEST_LABELS"] = "1"
        test = load("test")
        assert len(test.y) == 7 and test.y.shape == (7,)
        try:
            np.asarray(test.y)
        except RuntimeError as exc:
            assert "test labels are hidden" in str(exc)
        else:
            raise AssertionError("generated code could read hidden-test labels")
        assert "is_click" in test.aux
        try:
            _ = test.aux["is_click"]
        except RuntimeError as exc:
            assert "post-impression outcomes are hidden" in str(exc)
        else:
            raise AssertionError("generated code could read hidden-test outcomes")
        assert len(load("valid").y) == 10, "validation labels remain available for selection"
    finally:
        if old is None:
            os.environ.pop("AGENT_HIDE_TEST_LABELS", None)
        else:
            os.environ["AGENT_HIDE_TEST_LABELS"] = old
        _restore(tmp, prev)


def test_vocab_built_from_train_only_unseen_maps_to_zero():
    tmp = Path(tempfile.mkdtemp(prefix="kuairand_rawfixture_"))
    try:
        raw_dir = tmp / "raw"
        raw_dir.mkdir()
        vocab_dir = tmp / "vocab"
        vocab_dir.mkdir()

        # user 0 appears in the train log; user 1 exists in the side table but only ever
        # shows up in valid/test logs -- its user_id must map to OOV (0) after train-only vocab fit
        _write_csv(raw_dir / "user_features_pure.csv", _USER_HEADER, [_user_row(0), _user_row(1)])
        _write_csv(raw_dir / "video_features_basic_pure.csv", _VIDEO_HEADER, [_video_row(0)])
        train_log = raw_dir / "log_standard_4_08_to_4_21_pure.csv"
        _write_csv(train_log, _LOG_HEADER, [_log_row(0, 0, "20220410", 1)])

        users = _load_user_features(raw_dir / "user_features_pure.csv")
        videos = _load_video_features(raw_dir / "video_features_basic_pure.csv")
        vocabs, _duration_edges, _reg_edges = _build_vocabs_and_edges(train_log, users, videos, vocab_dir)

        assert vocabs["user_id"].get("0", 0) != 0, "user seen in train must get a real vocab id"
        assert vocabs["user_id"].get("1", 0) == 0, "user never seen in train must map to OOV (0)"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_real_data_split_counts():
    if not _REAL_RAW_DIR.exists():
        print(f"skip: test_real_data_split_counts (no real data at {_REAL_RAW_DIR})")
        return

    tmp, prev = _use_temp_cache()
    try:
        build_cache(_REAL_RAW_DIR, tmp)
        expected = {"train": 1_141_112, "valid": 124_909, "test": 170_588}
        for split, n in expected.items():
            s = load(split)
            assert len(s.y) == n, f"{split}: expected {n} rows, got {len(s.y)}"
            assert len(s.user_id) == n
            assert len(s.video_id) == n
            for feat, arr in s.X.items():
                assert len(arr) == n, f"{split}: X[{feat!r}] has {len(arr)} rows, expected {n}"
    finally:
        _restore(tmp, prev)


def test_date_is_exposed_and_matches_the_official_windows():
    """The splits are defined by date, and the train window is 13 days wide.

    Withholding the date silently rules out recency weighting and time-based validation -- the
    methods aimed at the drift between windows, which a high-capacity probe showed is the
    binding constraint here (0.9245 in-sample, 0.5868 on validation). It must not appear in X,
    though: a date used as a categorical feature cannot generalise to a later window.
    """
    if not _REAL_RAW_DIR.exists():
        print(f"skip: test_date_is_exposed_and_matches_the_official_windows (no real data)")
        return
    windows = {"train": (20220409, 20220421), "valid": (20220422, 20220428),
               "test": (20220429, 20220508)}
    for split, (lo, hi) in windows.items():
        s = load(split)
        assert s.date is not None, f"{split}: date not exposed"
        assert len(s.date) == len(s.y), f"{split}: date length mismatch"
        assert int(s.date.min()) == lo and int(s.date.max()) == hi, (
            f"{split}: window {int(s.date.min())}..{int(s.date.max())} != {lo}..{hi}")
        assert "date" not in s.X, f"{split}: date must not be a model feature"
    assert len(set(load("train").date.tolist())) == 13, "train covers 13 distinct days"


def main():
    # discovered, not listed: three separate tests in this repo were written, never added to a
    # hand-maintained list, and silently never ran. A test that does not run is worse than no
    # test, because it reads as coverage.
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        note = t()
        print(f"ok: {t.__name__}" + (f" {note}" if note else ""))
    print(f"{len(tests)} tests passed")


def test_time_ms_is_exposed_and_orders_impressions():
    """Without a timestamp a user's impressions cannot be put in order at all.

    `date` only separates days and `hour` is time-of-day, so every sequence model, session
    split, inter-arrival feature and recency decay was inexpressible. time_ms is the impression
    time -- known before the click -- so it belongs in the Split, not in aux with the outcomes.
    """
    import datetime
    for split in ("train", "valid", "test"):
        try:
            sp = load(split)
        except FileNotFoundError:
            return "skip (no real cache)"
        if sp.time_ms is None:
            raise AssertionError(f"{split}: time_ms missing -- rebuild the cache")
        t = np.asarray(sp.time_ms)
        assert t.dtype == np.int64, t.dtype
        assert len(t) == len(sp.y)
        assert (t > 1_600_000_000_000).all() and (t < 1_700_000_000_000).all(), "not epoch ms"
        # the day each timestamp falls on must equal the cached date for the same row, or the
        # backfill wrote a misaligned column and every sequence built from it would be wrong
        d = np.asarray(sp.date)[:2000]
        got = np.array([int(datetime.datetime.fromtimestamp(v / 1000).strftime("%Y%m%d"))
                        for v in t[:2000]])
        assert (got == d).all(), f"{split}: time_ms does not line up with date"

    # it is a feature, never an outcome
    assert "time_ms" not in AUX_DTYPES


def test_numeric_channel_is_present_and_carries_no_post_click_signal():
    """Split.num is the continuous channel; it must never carry an outcome of the scored row.

    Every numeric here is an attribute of the user or the video known before the impression is
    served. The post-click signals (play_time_ms, is_click, stay times) stay in aux. Mixing them
    would leak the label, and the leak would be invisible because both are float arrays.
    """
    from pipeline.data import NUMERIC_FEATURES, AUX_DTYPES
    assert not (set(NUMERIC_FEATURES) & set(AUX_DTYPES)), "a post-click signal reached Split.num"
    for banned in ("play_time_ms", "is_click", "long_view", "profile_stay_time",
                   "comment_stay_time", "is_profile_enter", "is_like"):
        assert banned not in NUMERIC_FEATURES, banned

    try:
        tr = load("train")
    except FileNotFoundError:
        return "skip (no real cache)"
    if tr.num is None:
        raise AssertionError("no numeric features cached -- rebuild the cache")
    for name, arr in tr.num.items():
        a = np.asarray(arr)
        assert a.dtype == np.float32, (name, a.dtype)
        assert len(a) == len(tr.y), name
        assert not np.isinf(a).any(), f"{name} has infinities"
    # A column that is entirely NaN is a silently dead channel: it costs cache, appears in the
    # brief as an available feature, and carries nothing. Three of these shipped that way --
    # NUMERIC_USER named raw counts that _load_user_features never read off the CSV, and only
    # the agent's own EDA (reporting m1.00 missing) surfaced it.
    dead = [n for n, a in tr.num.items() if np.isnan(np.asarray(a)).all()]
    assert not dead, f"numeric features are 100% NaN: {dead}"
    flat = [n for n, a in tr.num.items() if np.nanstd(np.asarray(a)) == 0]
    assert not flat, f"numeric features are constant: {flat}"


if __name__ == "__main__":
    main()
