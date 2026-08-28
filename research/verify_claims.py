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
                    help="directory holding the run folders; defaults to ./runs. Submission CSVs "
                         "are excluded from git, so verifying r30 needs a local archive.")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    runs_root = Path(args.runs_root) if args.runs_root else root / "runs"
    devpost = (root / "DEVPOST.md").read_text(encoding="utf-8")
    bad = 0
    checked = 0
    for line in devpost.splitlines():
        m = ROW.match(line)
        if not m:
            continue
        run, valid_claim, delta_claim = m.group(1), float(m.group(2)), float(m.group(3))
        run_dir = runs_root / run
        meta = run_dir / "run_meta.json"
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
    print(f"\n{checked} rows checked, {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
