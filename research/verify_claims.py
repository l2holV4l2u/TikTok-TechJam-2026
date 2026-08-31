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


_CELLS = re.compile(r"^\|(.+)\|\s*$")


def _table_rows(md: str):
    """Every markdown table row in the document, as a list of stripped cells."""
    for line in md.splitlines():
        m = _CELLS.match(line)
        if m:
            yield [c.strip().strip("*") for c in m.group(1).split("|")]


def _resolve(run: str, roots):
    """Map a short run id in the write-up to its directory, which may carry a _Nslot suffix."""
    for r in roots:
        if (r / run).is_dir():
            return r / run
        hit = sorted(r.glob(f"{run}_*"))
        if hit:
            return hit[0]
    return None


def _check_correlations(root: Path, roots, checked: int, bad: int):
    """The slot-correlation table and the slot ladder, re-read from the runs themselves.

    These carry the retraction: the gate reversed a delete-the-subsystem decision, and the ladder
    picks the submitted run. A stale number in either would misreport the project's main finding,
    and unlike the ablation table nobody else re-derives them for us.
    """
    md = (root / "DEVPOST.md").read_text(encoding="utf-8")
    logged, header = {}, None
    for cells in _table_rows(md):
        if cells and cells[0] == "turn" and len(cells) > 2:
            header = [c.split()[0] for c in cells[1:]]
            continue
        if header and cells and cells[0].isdigit() and len(cells) == len(header) + 1:
            turn = int(cells[0])
            for run, claim in zip(header, cells[1:]):
                if not re.fullmatch(r"0\.\d+", claim):
                    continue
                d = _resolve(run, roots)
                actual = None
                if d is not None and (d / "portfolio.jsonl").exists():
                    for line in (d / "portfolio.jsonl").read_text(encoding="utf-8").splitlines():
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if rec.get("event") == "turn" and rec.get("turn") == turn:
                            mean = (rec.get("correlation") or {}).get("mean")
                            actual = None if mean is None else float(mean)
                checked += 1
                if actual is None:
                    print(f"  {run} turn {turn}: UNVERIFIABLE -- no portfolio.jsonl record")
                    bad += 1
                    continue
                ok = abs(actual - float(claim)) < 1e-4
                print(f"  {run} turn {turn} correlation: claims {float(claim):.4f}  "
                      f"actual {actual:.4f}  {'OK' if ok else '<-- MISMATCH'}")
                bad += not ok
            continue
        # the slot ladder: | slots | run | validation | test | delta | ... |
        if len(cells) >= 5 and re.fullmatch(r"r\d+", cells[1] or "") and                 re.fullmatch(r"0\.\d+", cells[2] or "") and re.fullmatch(r"0\.\d+", cells[3] or ""):
            run = cells[1]
            d = _resolve(run, roots)
            meta = None if d is None else d / "run_meta.json"
            checked += 1
            if meta is None or not meta.exists():
                print(f"  {run}: UNVERIFIABLE -- no run_meta.json")
                bad += 1
                continue
            sub = json.loads(meta.read_text(encoding="utf-8")).get("submission") or {}
            okv = abs(float(sub.get("valid_primary", -9)) - float(cells[2])) < 1e-5
            okt = abs(float(sub.get("test_primary", -9)) - float(cells[3])) < 1e-5
            print(f"  {run} ladder: claims valid {cells[2]} test {cells[3]}  "
                  f"actual valid {sub.get('valid_primary', float('nan')):.6f} "
                  f"test {sub.get('test_primary', float('nan')):.6f}  "
                  f"{'OK' if okv and okt else '<-- MISMATCH'}")
            bad += not (okv and okt)
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
