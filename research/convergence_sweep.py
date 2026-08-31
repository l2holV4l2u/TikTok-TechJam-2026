"""Find the (epsilon, N) that the convergence rule should use.

The organizers fix the *form* of the stopping rule -- "converged when the validation score has
not improved by more than eps over the last N consecutive iterations" -- but leave eps and N to
the entrant. This script measures what those two numbers are worth.

Method. A run launched with --patience 999 cannot converge, so it runs to the iteration cap and
its validation curve is *uncensored*: every candidate N is visible on it, instead of only those
smaller than wherever the rule happened to stop. The stopping rule is then a pure function of
that curve, so every (eps, N) pair can be replayed offline against a curve that was recorded
once. No cell in the grid costs a run.

Each cell is scored on two axes, because either alone is degenerate:

  * forgone validation -- best_curve[-1] - best_curve[stop]. Minimising this alone says "never
    stop", which is why it cannot stand by itself.
  * iterations consumed -- the price paid for that forgone number.

and then on the axis that actually decides it:

  * test primary at the stop point -- scored from the scores_test.npy of whichever iteration was
    the validation argmax when the rule fired. Validation forgone can be large while test is
    flat; r77/r78 showed exactly that for the refine and tune modes. Test is what the rule is
    ultimately being chosen to maximise.

Usage
    python -m research.convergence_sweep runs/convergence_sweep/curve_*   # the study
    python -m research.convergence_sweep --verify runs/r90 runs/r89_5slot # self-test only
    python -m research.convergence_sweep --no-test runs/...               # skip test scoring
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.loop import converged  # the rule itself, never a second copy of it

# The grid. eps spans well below the seed noise floor (sigma ~ 0.0008 on this benchmark) through
# the organizers' 0.002 and out to a coarse value; N runs from 2 to the largest value a 50-script
# curve can resolve without the answer running into the cap.
EPSILONS = [0.0005, 0.001, 0.002, 0.003, 0.005]
PATIENCES = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
OFFICIAL = (0.002, 3)


def load_curve(run_dir: Path) -> list[dict]:
    """Rebuild the harness's best_curve from a run's ledger.

    agent.loop appends one point per scored improve *turn* (loop.py:880), and the value appended
    is `selection_best`, a running maximum over the whole run. With --slots 1 a turn is a single
    script; with more, the turn's point is the best of its slots. Reconstructing rather than
    reading a stored curve keeps this honest: the same ledger a judge would read is the input.
    """
    rows = [json.loads(l) for l in (run_dir / "ledger.jsonl").open(encoding="utf-8") if l.strip()]
    scored = [r for r in rows
              if r.get("phase") == "improve"
              and r.get("status") in ("ok", "kept", "reverted")
              and "primary" in (r.get("metrics") or {})]

    by_turn: dict[int, list[dict]] = {}
    for r in scored:
        by_turn.setdefault(r.get("turn") or 0, []).append(r)

    curve, best, best_iter = [], float("-inf"), None
    for turn in sorted(by_turn):
        for r in sorted(by_turn[turn], key=lambda x: int(x["iter_id"])):
            if r["metrics"]["primary"] > best:
                best, best_iter = r["metrics"]["primary"], int(r["iter_id"])
        curve.append({"turn": turn, "best": best, "best_iter": best_iter,
                      "scripts": len(by_turn[turn])})
    return curve


def replay(curve: list[dict], patience: int, epsilon: float) -> int | None:
    """Index into `curve` where this rule would have stopped, or None if it never fires.

    The harness checks convergence at the *top* of a turn, against the curve built so far
    (loop.py:550), so a stop detected after k points means k points were paid for.
    """
    best = [p["best"] for p in curve]
    for k in range(1, len(best) + 1):
        if converged(best[:k], patience, epsilon):
            return k - 1
    return None


class TestScorer:
    """Scores an iteration's saved test predictions. Cached: distinct argmax iterations are few."""

    def __init__(self, enabled: bool = True):
        self.enabled, self.cache, self.te = enabled, {}, None

    def __call__(self, run_dir: Path, iter_id: int | None) -> float | None:
        if not self.enabled or iter_id is None:
            return None
        key = (str(run_dir), iter_id)
        if key in self.cache:
            return self.cache[key]
        path = run_dir / "scripts" / f"iter_{iter_id}_out" / "scores_test.npy"
        if not path.exists():
            self.cache[key] = None
            return None
        import numpy as np
        from pipeline.evaluate import evaluate
        if self.te is None:
            from pipeline.data import load
            self.te = load("test")
        val = evaluate(self.te.user_id, self.te.y, np.load(path))["primary"]
        self.cache[key] = val
        return val


def verify(run_dirs: list[Path]) -> int:
    """Self-test: replaying a run's OWN (eps, N) must reproduce the stop it actually recorded.

    A reconstruction that cannot do this is not measuring the harness's rule, and every number
    downstream of it would be fiction.
    """
    bad = 0
    print(f"{'run':<24} {'eps':>7} {'N':>4} {'turns':>6} {'replay':>7} {'recorded':>9}  verdict")
    for rd in run_dirs:
        meta = json.loads((rd / "run_meta.json").read_text(encoding="utf-8"))
        eps = meta.get("epsilon", 0.002)
        pat = meta.get("patience", 3)
        curve = load_curve(rd)
        stop = replay(curve, pat, eps)
        recorded = meta.get("stop_reason", "?")
        fired = stop is not None
        expected = recorded == "converged"
        ok = fired == expected
        bad += not ok
        print(f"{rd.name:<24} {eps:>7.4f} {pat:>4} {len(curve):>6} "
              f"{(str(stop + 1) if fired else '-'):>7} {recorded:>9}  "
              f"{'OK' if ok else 'MISMATCH'}")
    print(f"\n{len(run_dirs) - bad}/{len(run_dirs)} runs reproduce their recorded stop.")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--verify", action="store_true", help="self-test the reconstruction only")
    ap.add_argument("--no-test", action="store_true", help="skip hidden-test scoring")
    ap.add_argument("--out", type=Path, default=Path("reports/convergence_sweep"))
    args = ap.parse_args()

    if args.verify:
        return 1 if verify(args.runs) else 0

    curves = {rd.name: load_curve(rd) for rd in args.runs}
    for name, c in curves.items():
        if not c:
            print(f"WARNING: {name} has no scored improve turns", file=sys.stderr)
    scorer = TestScorer(enabled=not args.no_test)
    args.out.mkdir(parents=True, exist_ok=True)

    # --- curves.csv: the raw evidence, one row per curve point -------------------------------
    with (args.out / "curves.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["curve", "turn", "best_valid", "argmax_iter", "scripts_this_turn", "test"])
        for rd in args.runs:
            for p in curves[rd.name]:
                w.writerow([rd.name, p["turn"], f"{p['best']:.6f}", p["best_iter"],
                            p["scripts"], _fmt(scorer(rd, p["best_iter"]))])

    # --- the grid ----------------------------------------------------------------------------
    grid = []
    for eps in EPSILONS:
        for pat in PATIENCES:
            per_curve = []
            for rd in args.runs:
                c = curves[rd.name]
                if not c:
                    continue
                stop = replay(c, pat, eps)
                censored = stop is None
                idx = (len(c) - 1) if censored else stop
                per_curve.append({
                    "curve": rd.name,
                    "stop_turn": idx + 1,
                    "censored": censored,
                    "forgone": c[-1]["best"] - c[idx]["best"],
                    "test": scorer(rd, c[idx]["best_iter"]),
                    "test_full": scorer(rd, c[-1]["best_iter"]),
                })
            if per_curve:
                grid.append({"epsilon": eps, "patience": pat, "per_curve": per_curve})

    _write_grid(args.out, grid)
    _print_summary(grid, curves)
    return 0


def _fmt(v):
    return "" if v is None else f"{v:.6f}"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _write_grid(out: Path, grid: list[dict]) -> None:
    with (out / "grid_valid.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epsilon", "patience", "curve", "stop_turn", "censored", "forgone_valid"])
        for cell in grid:
            for pc in cell["per_curve"]:
                w.writerow([cell["epsilon"], cell["patience"], pc["curve"], pc["stop_turn"],
                            int(pc["censored"]), f"{pc['forgone']:.6f}"])
    with (out / "grid_test.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epsilon", "patience", "curve", "stop_turn", "test_at_stop",
                    "test_run_to_cap", "test_delta"])
        for cell in grid:
            for pc in cell["per_curve"]:
                d = (None if pc["test"] is None or pc["test_full"] is None
                     else pc["test"] - pc["test_full"])
                w.writerow([cell["epsilon"], cell["patience"], pc["curve"], pc["stop_turn"],
                            _fmt(pc["test"]), _fmt(pc["test_full"]), _fmt(d)])


def _print_summary(grid: list[dict], curves: dict) -> None:
    print(f"\n# Convergence sweep\n\n{len(curves)} curve(s): "
          + ", ".join(f"{k} ({len(v)} turns)" for k, v in curves.items()))
    censored = sum(pc["censored"] for cell in grid for pc in cell["per_curve"])
    if censored:
        print(f"\n**{censored} of {sum(len(c['per_curve']) for c in grid)} cells are CENSORED** "
              "-- the rule never fired before the cap, so those rows are lower bounds, not "
              "results. Re-run with a larger --iters, or drop the largest N.")

    print("\n## Mean over curves\n")
    print("| eps | N | stop turn | forgone valid | test at stop | vs run-to-cap |")
    print("|---|---|---|---|---|---|")
    for cell in grid:
        pcs = cell["per_curve"]
        mt = _mean([p["test"] for p in pcs])
        md = _mean([None if p["test"] is None or p["test_full"] is None
                    else p["test"] - p["test_full"] for p in pcs])
        flag = " *" if (cell["epsilon"], cell["patience"]) == OFFICIAL else ""
        cen = " (censored)" if any(p["censored"] for p in pcs) else ""
        print(f"| {cell['epsilon']}{flag} | {cell['patience']} | "
              f"{_mean([p['stop_turn'] for p in pcs]):.1f} | "
              f"{_mean([p['forgone'] for p in pcs]):.5f} | "
              f"{'-' if mt is None else f'{mt:.6f}'} | "
              f"{'-' if md is None else f'{md:+.6f}'}{cen} |")
    print("\n`*` = the organizers' example values.")


if __name__ == "__main__":
    raise SystemExit(main())
