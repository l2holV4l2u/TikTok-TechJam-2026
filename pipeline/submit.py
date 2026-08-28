"""CLI matching the organizers' submit.py: --make writes an example submission, --check validates one.

Assumption (undocumented upstream): pipeline.data.load(split) returns an object exposing the
evaluation split's user ids and video/item ids, either as attributes or dict keys, under one of
a few common names. _field() below is defensive about which spelling is used so this file does
not have to be rewritten once pipeline/data.py's exact shape is settled by its owning agent.
"""
import argparse
import csv
import math
import sys

import numpy as np

HEADER = ["row_id", "user_id", "video_id", "score"]

_USER_ID_NAMES = ("user_id", "user_ids", "uid", "uids")
_VIDEO_ID_NAMES = ("video_id", "video_ids", "item_id", "item_ids", "vid", "vids")


def _field(obj, names):
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return np.asarray(obj[name])
        elif hasattr(obj, name):
            return np.asarray(getattr(obj, name))
    raise AttributeError(f"split object has none of {names} (type={type(obj).__name__})")


def _call_load(load_fn, split):
    try:
        return load_fn(split)
    except TypeError:
        return load_fn()  # some load() signatures take no split argument


def _load_eval_split(split: str):
    from pipeline.data import load  # lazy: keeps tests independent of the real dataset

    obj = _call_load(load, split)
    user_ids = _field(obj, _USER_ID_NAMES)
    video_ids = _field(obj, _VIDEO_ID_NAMES)
    if user_ids.shape[0] != video_ids.shape[0]:
        raise ValueError("split user_id/video_id arrays have mismatched lengths")
    return user_ids, video_ids


def make_submission(out_path: str, split: str = "eval", seed: int = 0) -> None:
    user_ids, video_ids = _load_eval_split(split)
    n = user_ids.shape[0]
    rng = np.random.default_rng(seed)
    scores = rng.random(n)  # placeholder scores; only relative order matters for the metrics
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for row_id in range(n):
            w.writerow([row_id, user_ids[row_id], video_ids[row_id], scores[row_id]])


def check_submission(path: str, split: str = "eval") -> tuple[bool, str]:
    user_ids, video_ids = _load_eval_split(split)
    n_expected = user_ids.shape[0]

    try:
        f = open(path, newline="", encoding="utf-8")
    except FileNotFoundError:
        return False, f"file not found: {path}"
    with f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return False, "file is empty, missing header"
        if header != HEADER:
            return False, f"header mismatch: expected {','.join(HEADER)!r}, got {','.join(header)!r}"

        rows = list(reader)

    if len(rows) != n_expected:
        return False, f"row count mismatch: expected {n_expected}, got {len(rows)}"

    for i, row in enumerate(rows):
        if len(row) != 4:
            return False, f"row {i} has {len(row)} fields, expected 4"

        row_id_str, user_id_str, video_id_str, score_str = row

        try:
            row_id = int(row_id_str)
        except ValueError:
            return False, f"row {i}: row_id {row_id_str!r} is not an integer"
        if row_id != i:
            return False, f"row_id gap or non-monotonicity at row {i}: expected row_id {i}, got {row_id}"

        expected_user = str(user_ids[i])
        expected_video = str(video_ids[i])
        if user_id_str != expected_user or video_id_str != expected_video:
            return False, (
                f"row {i}: user_id/video_id misaligned with evaluation split "
                f"(expected user_id={expected_user!r} video_id={expected_video!r}, "
                f"got user_id={user_id_str!r} video_id={video_id_str!r})"
            )

        try:
            score = float(score_str)
        except ValueError:
            return False, f"row {i}: score {score_str!r} is not numeric"
        if math.isnan(score) or math.isinf(score):
            return False, f"row {i}: score is NaN/Inf ({score_str!r})"

    return True, "ok"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build or validate a KuaiRand-Pure Track 2 submission CSV.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make", action="store_true", help="write an example submission")
    mode.add_argument("--check", action="store_true", help="validate an existing submission")
    p.add_argument("--split", default="eval", help="split name passed to pipeline.data.load (default: eval)")
    p.add_argument("--out", help="output CSV path for --make")
    p.add_argument("--in", dest="in_path", help="input CSV path for --check")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --make example scores")
    args = p.parse_args(argv)

    if args.make:
        if not args.out:
            p.error("--make requires --out")
        make_submission(args.out, split=args.split, seed=args.seed)
        print(f"wrote example submission to {args.out}")
        return 0

    if not args.in_path:
        p.error("--check requires --in")
    ok, reason = check_submission(args.in_path, split=args.split)
    print(reason)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
