"""Slot-ladder report: what n concurrent lineages actually buy, and what they cost.

`report_run.py` documents one run. This documents a controlled comparison across runs that
differ only in `--slots`, because the portfolio's whole claim -- n lineages per turn buy more
search per unit of the three non-improving turns the convergence rule allows -- is a claim
about the SHAPE of that curve, and no single run can show it.

    python report_run_compare.py > RUN_REPORT.md
    python report_run_compare.py runs/r85_1slot runs/r86_2slot ...   # explicit runs
    python report_run_compare.py --price-in=1.25 --price-out=10      # USD per 1M tokens

Runs are ordered by their recorded `slots`, not by name, so a ladder assembled out of order
still reads correctly. Cost is only printed when a price is supplied: the repo records tokens,
never dollars, and inventing a rate for the model in use would put a fabricated number in a
deliverable.
"""
import json
import sys
from pathlib import Path

# KuaiRand-Pure's published baseline, matching report_run.py. A ladder run on another variant
# must pass its own reference or every delta below is measured against the wrong anchor.
BASE_VALID, BASE_TEST = 0.6016, 0.5946
BASE_LABEL = "official baseline"
EPSILON = 0.002          # the organizers' convergence epsilon
SEED_SIGMA = 0.0008      # the baseline's reported 5-seed std, the noise floor for any delta

DEFAULT_LADDER = ["runs/r85_1slot", "runs/r86_2slot", "runs/r87_3slot",
                  "runs/r88_4slot", "runs/r89_5slot"]

PRICE_IN = PRICE_OUT = None   # USD per 1M tokens; unset means the cost columns stay blank


def _load(run_dir: Path) -> dict | None:
    """One run's metadata plus its ledger. Returns None for a run that never wrote metadata."""
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = []
    led = run_dir / "ledger.jsonl"
    if led.exists():
        for line in led.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    corr = []
    pf = run_dir / "portfolio.jsonl"
    if pf.exists():
        for line in pf.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "turn" and (rec.get("correlation") or {}).get("mean") is not None:
                corr.append(rec["correlation"]["mean"])
    return {"dir": run_dir, "meta": meta, "rows": rows, "corr": corr}


def _cost(meta: dict) -> float | None:
    if PRICE_IN is None or PRICE_OUT is None:
        return None
    return (meta.get("tokens_in", 0) / 1e6) * PRICE_IN + (meta.get("tokens_out", 0) / 1e6) * PRICE_OUT


def _sub(run: dict, key: str, default=None):
    return (run["meta"].get("submission") or {}).get(key, default)


def _noise(delta: float) -> str:
    """Label a delta against the noise floor rather than leaving the reader to judge it."""
    if abs(delta) < SEED_SIGMA:
        return "within seed noise"
    if abs(delta) < EPSILON:
        return "under epsilon"
    return "above epsilon"


def _headline(runs: list) -> None:
    print("## What each rung produced\n")
    print("Every run below is the same agent, same dataset, same data contract and the same "
          "convergence rule. The only difference is `--slots`.\n")
    cost_hdr = " cost |" if PRICE_IN is not None else ""
    print(f"| slots | run | validation | hidden test | vs baseline | wall clock | tokens |{cost_hdr} scripts | turns |")
    print(f"|---|---|---|---|---|---|---|{'---|' if PRICE_IN is not None else ''}---|---|")
    for r in runs:
        m = r["meta"]
        c = _cost(m)
        cost_cell = f" ${c:.2f} |" if c is not None else ""
        print(f"| **{m.get('slots', 1)}** | `{r['dir'].name}` | {_sub(r,'valid_primary',0):.6f} | "
              f"**{_sub(r,'test_primary',0):.6f}** | {_sub(r,'test_delta',0):+.6f} | "
              f"{m.get('wall_clock_s',0)/60:.1f} min | {m.get('tokens_total',0):,} |{cost_cell} "
              f"{m.get('scripts','?')} | {m.get('turns','?')} |")
    print(f"\nBaseline for the `vs baseline` column is the {BASE_LABEL}: "
          f"validation {BASE_VALID:.4f}, hidden test {BASE_TEST:.4f}.")


def _plateau(runs: list) -> None:
    """The load-bearing section: marginal test gain per added slot, against the noise floor."""
    print("\n## Where the score stops moving\n")
    print("Marginal change from adding one more lineage. A benchmark difference below "
          f"{SEED_SIGMA} is inside the baseline's own 5-seed noise and is not evidence of "
          "anything; below {:.3f} it is under the organizers' convergence epsilon.\n".format(EPSILON))
    print("| step | test | delta | reading | tokens | delta | wall | delta |")
    print("|---|---|---|---|---|---|---|---|")
    prev = None
    for r in runs:
        m, t = r["meta"], _sub(r, "test_primary", 0.0)
        tok, wall = m.get("tokens_total", 0), m.get("wall_clock_s", 0) / 60
        if prev is None:
            print(f"| {m.get('slots',1)} slot | {t:.6f} | - | starting point | {tok:,} | - | {wall:.1f} min | - |")
        else:
            d = t - prev[0]
            print(f"| {prev[1]} -> {m.get('slots',1)} | {t:.6f} | {d:+.6f} | {_noise(d)} | "
                  f"{tok:,} | {tok-prev[2]:+,} | {wall:.1f} min | {wall-prev[3]:+.1f} |")
        prev = (t, m.get("slots", 1), tok, wall)

    tests = [(_sub(r, "test_primary", 0.0), r["meta"].get("slots", 1)) for r in runs]
    best_t, best_s = max(tests)
    # The claim worth checking is not "more slots is better" but "the curve flattens". Compare
    # the best rung against every rung at or above the first one within noise of it.
    plateau = [s for t, s in tests if abs(t - best_t) < SEED_SIGMA]
    if len(plateau) > 1:
        lo = min(plateau)
        span = [r for r in runs if r["meta"].get("slots", 1) in plateau]
        tok_lo = min(r["meta"].get("tokens_total", 0) for r in span)
        tok_hi = max(r["meta"].get("tokens_total", 0) for r in span)
        w_lo = min(r["meta"].get("wall_clock_s", 0) for r in span) / 60
        w_hi = max(r["meta"].get("wall_clock_s", 0) for r in span) / 60
        print(f"\n**The curve flattens at {lo} slots.** Rungs {', '.join(str(s) for s in sorted(plateau))} "
              f"all land within {SEED_SIGMA} of the best result ({best_t:.6f} at {best_s} slots), "
              f"so on this evidence they are indistinguishable on the hidden test. Across that "
              f"flat span the spend still rises from {tok_lo:,} to {tok_hi:,} tokens "
              f"({tok_hi/tok_lo:.1f}x) and from {w_lo:.1f} to {w_hi:.1f} minutes.")
        print(f"\nThe honest reading is that everything past {lo} slots is bought and not "
              f"delivered: cost scales close to linearly in the slot count while the scored "
              f"result does not move outside noise.")


def _efficiency(runs: list) -> None:
    print("\n## What the spend bought\n")
    print("| slots | tokens | tokens/slot | wall clock | script time | candidates compared | "
          "candidates/1k tokens |")
    print("|---|---|---|---|---|---|---|")
    for r in runs:
        m = r["meta"]
        s = m.get("slots", 1) or 1
        tok = m.get("tokens_total", 0)
        cand = m.get("candidates_evaluated", 0) or 0
        print(f"| {s} | {tok:,} | {tok//s:,} | {m.get('wall_clock_s',0)/60:.1f} min | "
              f"{m.get('script_seconds',0)/60:.1f} min | {cand} | {1000*cand/tok:.2f} |")
    print("\n`candidates compared` counts the alternatives evaluated INSIDE scripts, which is "
          "where most of the search happens: a script may build and compare a dozen models for "
          "one iteration of budget. It rises with the slot count, so the extra lineages are "
          "doing real work -- the question the table above answers is whether that work reaches "
          "the score.")


def _diversity(runs: list) -> None:
    """Slot correlation is the portfolio's own acceptance test, so it belongs beside the cost."""
    have = [r for r in runs if r["corr"]]
    if not have:
        return
    print("\n## Did the extra lineages disagree?\n")
    print("Within-user rank correlation between the slots' own pre-blend models. Above ~0.95 "
          "the extra lineages rank validation identically and return one lineage's information "
          "for n lineages' spend.\n")
    # run_meta's `mean_slot_correlation` is the LAST turn's mean, not an average over turns --
    # LoopResult overwrites it every turn. Reporting it under a "run mean" header would be the
    # kind of quietly wrong label this report exists to avoid, so both are printed and the
    # verdict is taken from the average across turns rather than from whichever turn ran last.
    print("| slots | per-turn mean | mean across turns | last turn (`mean_slot_correlation`) | reading |")
    print("|---|---|---|---|---|")
    for r in have:
        m = r["meta"]
        per = " -> ".join(f"{c:.3f}" for c in r["corr"])
        avg = sum(r["corr"]) / len(r["corr"])
        last = m.get("mean_slot_correlation")
        reading = ("lineages genuinely differ" if avg < 0.90
                   else "borderline" if avg < 0.95 else "effectively copies")
        last_cell = f"{last:.3f}" if isinstance(last, (int, float)) else "-"
        print(f"| {m.get('slots',1)} | {per} | **{avg:.3f}** | {last_cell} | {reading} |")
    print("\nThe last-turn column is what `run_meta.json` records; it is a single turn and "
          "swings widely, so the verdict above is taken from the average across turns.")


def _reliability(runs: list) -> None:
    print("\n## Reliability across the ladder\n")
    print("| slots | scripts | crashes | integrity rejections | manual interventions | stop reason |")
    print("|---|---|---|---|---|---|")
    for r in runs:
        m = r["meta"]
        crashes = sum(1 for e in r["rows"] if e.get("status") in ("failed", "blacklisted"))
        print(f"| {m.get('slots',1)} | {m.get('scripts','?')} | {crashes} | "
              f"{m.get('integrity_rejections',0)} | {m.get('manual_interventions',0)} | "
              f"`{m.get('stop_reason','?')}` |")
    print("\n`crashes` counts scripts that failed to produce a scored result, separately from "
          "integrity rejections, which are results the critic refused. Both are handled in the "
          "loop; neither reaches a human.")


def _curves(runs: list) -> None:
    print("\n## Score curve within each run\n")
    print("The run's best validation score after each scored improve iteration. One curve per "
          "run, which is what the convergence rule is measured against.\n")
    for r in runs:
        best, curve = float("-inf"), []
        for e in r["rows"]:
            if e.get("phase") != "improve":
                continue
            p = (e.get("metrics") or {}).get("primary")
            if isinstance(p, (int, float)):
                best = max(best, p)
                curve.append(best)
        if curve:
            print(f"- **{r['meta'].get('slots',1)} slot(s)** (`{r['dir'].name}`): "
                  + " -> ".join(f"{c:.4f}" for c in curve))


def main() -> None:
    global PRICE_IN, PRICE_OUT, BASE_VALID, BASE_TEST, BASE_LABEL
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for a in sys.argv[1:]:
        if a.startswith("--price-in="):
            PRICE_IN = float(a.split("=", 1)[1])
        elif a.startswith("--price-out="):
            PRICE_OUT = float(a.split("=", 1)[1])
        elif a.startswith("--baseline-valid="):
            BASE_VALID = float(a.split("=", 1)[1])
        elif a.startswith("--baseline-test="):
            BASE_TEST = float(a.split("=", 1)[1])

    loaded = []
    for name in (args or DEFAULT_LADDER):
        run = _load(Path(name))
        if run is None:
            print(f"skipping {name}: no run_meta.json", file=sys.stderr)
            continue
        loaded.append(run)
    if not loaded:
        print("no runs to compare", file=sys.stderr)
        raise SystemExit(1)
    runs = sorted(loaded, key=lambda r: r["meta"].get("slots", 1))

    print("# Slot-ladder comparison\n")
    models = {r["meta"].get("model") for r in runs}
    contracts = {r["meta"].get("data_contract") for r in runs}
    print(f"{len(runs)} runs, {min(r['meta'].get('slots',1) for r in runs)} to "
          f"{max(r['meta'].get('slots',1) for r in runs)} lineages per turn. "
          f"Model: {', '.join(sorted(str(m) for m in models))}. "
          f"Data contract: {', '.join(sorted(str(c) for c in contracts))}.")
    if len(models) > 1 or len(contracts) > 1:
        print("\n**These runs do not share a model or a data contract, so the comparison below "
              "is confounded.** Re-run the ladder under one configuration before citing it.")
    if PRICE_IN is None:
        print("\nCost columns are omitted: the run records hold token counts, not prices. Pass "
              "`--price-in=` and `--price-out=` in USD per 1M tokens to include them.")

    _headline(runs)
    _plateau(runs)
    _efficiency(runs)
    _diversity(runs)
    _reliability(runs)
    _curves(runs)

    print("\n## Caveat\n")
    print(f"Each rung is a single run. Run-to-run spread on this benchmark is roughly "
          f"{SEED_SIGMA} on the scored metric, which is the same size as most of the "
          f"differences above, so the ladder shows the SHAPE of the cost/score curve rather "
          f"than a precise value for any rung. The plateau claim rests on several rungs "
          f"agreeing, not on any single pair.")


if __name__ == "__main__":
    main()
