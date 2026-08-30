"""Check every score the DEVPOST ablation table cites against the run records.

A number in a deliverable that no longer matches its run is the cheapest possible way to lose
credibility with a judge, and these tables were edited by hand across many runs. This re-derives
each row: from `run_meta.json` where it exists, and by re-scoring the run's own submission.csv
where it does not (r30's metadata was destroyed by an encoding crash after it had already
written its submission).

  python -m research.verify_claims                      # exits non-zero if any row disagrees
  python -m research.verify_claims --runs-root PATH     # submission CSVs are gitignored, so a
                                                        # fresh clone needs a local run archive
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

BASELINE_TEST = 0.5946
ROW = re.compile(r"\|\s*\*{0,2}(r\d+\w*)\*{0,2}\s*\|.*?\|\s*\*{0,2}(0\.\d{4})\*{0,2}\s*\|"
                 r"\s*\*{0,2}([+-]0\.\d{4})\*{0,2}\s*\|")


CORR = re.compile("^\|\s*(\d)\s*\|\s*([0-9.]+|[-—])\s*\|\s*([0-9.]+|[-—])\s*\|\s*$")


def _check_correlations(root: Path, roots, checked: int, bad: int):
    """The portfolio gate table cites per-turn slot correlations; re-read them from the runs.

    The gate is the reason Phases 3-5 were abandoned, so a wrong number here would misreport a
    negative result -- the one kind of claim nobody else will re-derive for us.
    """
    logged = {}
    for run in ("r82", "r83"):
        path = next((r / run / "portfolio.jsonl" for r in roots
                     if (r / run / "portfolio.jsonl").exists()), None)
        if path is None:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "turn" and (rec.get("correlation") or {}).get("mean") is not None:
                logged[(run, rec["turn"])] = float(rec["correlation"]["mean"])
    if not logged:
        return checked, bad
    for line in (root / "DEVPOST.md").read_text(encoding="utf-8").splitlines():
        m = CORR.match(line)
        if not m:
            continue
        turn = int(m.group(1))
        for run, claim in (("r82", m.group(2)), ("r83", m.group(3))):
            if claim in ("-", "—"):
                continue
            actual = logged.get((run, turn))
            checked += 1
            if actual is None:
                print(f"  {run} turn {turn}: UNVERIFIABLE -- no portfolio.jsonl record")
                bad += 1
                continue
            ok = abs(actual - float(claim)) < 1e-4
            print(f"  {run} turn {turn} correlation: claims {float(claim):.4f}  "
                  f"actual {actual:.4f}  {'OK' if ok else '<-- MISMATCH'}")
            bad += not ok
    return checked, bad


def _rescore(run_dir: Path):
    """Score a run's own submission.csv -- the fallback when its metadata was lost."""
    sub = run_dir / "submission.csv"
    if not sub.exists():
        return None
    from pipeline.data import load
    from pipeline.evaluate import evaluate
    te = load("test")
    with sub.open() as f:
        r = csv.reader(f)
        next(r)
        scores = np.array([float(x[3]) for x in r])
    if len(scores) != len(te.y):
        return None
    m = evaluate(te.user_id, te.y, scores)
    return m["primary"], m["primary"] - BASELINE_TEST


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=None,
                    help="comma-separated directories holding run folders; defaults to ./runs. "
                         "Submission CSVs are gitignored, and runs made in a worktree live "
                         "beside it, so a full check may need more than one archive.")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    roots = ([Path(x) for x in args.runs_root.split(",") if x.strip()]
             if args.runs_root else [root / "runs"])
    devpost = (root / "DEVPOST.md").read_text(encoding="utf-8")
    bad = 0
    checked = 0
    for line in devpost.splitlines():
        m = ROW.match(line)
        if not m:
            continue
        run, valid_claim, delta_claim = m.group(1), float(m.group(2)), float(m.group(3))
        # metadata and submission may live in different archives -- a worktree keeps the run
        # it produced, while gitignored CSVs stay only where they were written. Resolve each
        # independently rather than picking one directory and hoping it holds both.
        meta = next((r / run / "run_meta.json" for r in roots
                     if (r / run / "run_meta.json").exists()), roots[0] / run / "run_meta.json")
        run_dir = next((r / run for r in roots if (r / run / "submission.csv").exists()),
                       meta.parent)
        valid = delta = source = None
        if meta.exists():
            sub = (json.loads(meta.read_text(encoding="utf-8")).get("submission") or {})
            valid, delta = sub.get("valid_primary"), sub.get("test_delta")
            source = "run_meta"
        if delta is None:
            got = _rescore(run_dir)
            if got is not None:
                _, delta = got
                source = "re-scored submission.csv"
        checked += 1
        if delta is None:
            print(f"  {run}: UNVERIFIABLE -- no run_meta submission block and no submission.csv")
            bad += 1
            continue
        okd = abs(delta - delta_claim) < 1e-4
        okv = valid is None or abs(valid - valid_claim) < 1e-4
        print(f"  {run}: claims {valid_claim:.4f}/{delta_claim:+.4f}  "
              f"actual {'n/a' if valid is None else f'{valid:.4f}'}/{delta:+.4f}  "
              f"[{source}]  {'OK' if okd and okv else '<-- MISMATCH'}")
        bad += not (okd and okv)
    checked, bad = _check_correlations(root, roots, checked, bad)
    print(f"\n{checked} claim(s) checked, {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
