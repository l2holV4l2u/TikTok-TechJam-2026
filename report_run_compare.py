"""Slot-ladder report: what n concurrent lineages actually buy, and what they cost.

`report_run.py` documents one run. This documents a controlled comparison across runs that
differ only in `--slots`, because the portfolio's whole claim -- n lineages per turn buy more
search per unit of the budget the convergence rule allows -- is a claim about the SHAPE of that
curve, and no single run can show it.

    python report_run_compare.py > RUN_REPORT.md
    python report_run_compare.py runs/slot_ladder/s1_r1 runs/slot_ladder/s2_r1 ...
    python report_run_compare.py --price-in=1.25 --price-out=10      # USD per 1M tokens

With no arguments it discovers every `runs/slot_ladder/s<slots>_r<repeat>` directory.

REPEATS. Run-to-run spread on this benchmark is about the same size as the differences the
ladder is trying to measure, so a one-run-per-rung ladder cannot separate a real plateau from
scatter. When a slot count has several repeats this report aggregates them, reports the observed
spread, and -- crucially -- uses that MEASURED spread as the noise floor instead of an assumed
constant. With one repeat per rung it falls back to the published seed sigma and says so.

Runs are ordered by their recorded `slots`, not by name, so a ladder assembled out of order
still reads correctly. Cost is only printed when a price is supplied: the repo records tokens,
never dollars, and inventing a rate for the model in use would put a fabricated number in a
deliverable.
"""
import json
import statistics as st
import sys
from pathlib import Path

# KuaiRand-Pure's published baseline, matching report_run.py. A ladder run on another variant
# must pass its own reference or every delta below is measured against the wrong anchor.
BASE_VALID, BASE_TEST = 0.6016, 0.5946
BASE_LABEL = "official baseline"
EPSILON = 0.002          # the convergence epsilon these runs used
SEED_SIGMA = 0.0008      # the baseline's reported 5-seed std; the fallback noise floor

LADDER_DIR = Path("runs/slot_ladder")

PRICE_IN = PRICE_OUT = None   # USD per 1M tokens; unset means the cost columns stay blank


# ----------------------------------------------------------------------------- loading

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


def _discover() -> list[str]:
    """Every s<slots>_r<repeat> run under the ladder directory, ordered by slots then repeat."""
    def key(p: Path):
        try:
            s, r = p.name[1:].split("_r")
            return (int(s), int(r))
        except (ValueError, IndexError):
            return (99, 99)
    return [str(p) for p in sorted(LADDER_DIR.glob("s*_r*"), key=key) if p.is_dir()]


def _cost(meta: dict) -> float | None:
    if PRICE_IN is None or PRICE_OUT is None:
        return None
    return (meta.get("tokens_in", 0) / 1e6) * PRICE_IN + (meta.get("tokens_out", 0) / 1e6) * PRICE_OUT


def _sub(run: dict, key: str, default=None):
    return (run["meta"].get("submission") or {}).get(key, default)


# ----------------------------------------------------------------------------- grouping

class Rung:
    """All repeats of one slot count, and the statistics that survive having only a few."""

    def __init__(self, slots: int, runs: list):
        self.slots, self.runs = slots, runs
        self.n = len(runs)
        self.tests = [_sub(r, "test_primary", 0.0) for r in runs]
        self.valids = [_sub(r, "valid_primary", 0.0) for r in runs]
        self.tokens = [r["meta"].get("tokens_total", 0) for r in runs]
        self.walls = [r["meta"].get("wall_clock_s", 0) / 60 for r in runs]

    @property
    def test(self): return sum(self.tests) / self.n

    @property
    def valid(self): return sum(self.valids) / self.n

    @property
    def tok(self): return sum(self.tokens) / self.n

    @property
    def wall(self): return sum(self.walls) / self.n

    @property
    def spread(self):
        """Max-min across repeats. None at n=1, where the run gives no spread information."""
        return (max(self.tests) - min(self.tests)) if self.n > 1 else None

    @property
    def sem(self):
        return (st.stdev(self.tests) / self.n ** 0.5) if self.n > 1 else None

    def cell(self, vals, fmt="{:.6f}"):
        """A value with its spread when repeats exist, the bare value when they do not."""
        if self.n == 1:
            return fmt.format(vals[0])
        return f"{fmt.format(sum(vals)/self.n)} ({fmt.format(min(vals))}-{fmt.format(max(vals))})"


def _rungs(runs: list) -> list:
    by = {}
    for r in runs:
        by.setdefault(r["meta"].get("slots", 1), []).append(r)
    return [Rung(s, by[s]) for s in sorted(by)]


def _noise_floor(rungs: list) -> tuple[float, str]:
    """The yardstick a difference is judged against.

    Prefer the spread actually observed across repeats -- it measures this agent on this dataset
    -- and fall back to the published seed sigma only when no rung has been repeated.
    """
    sems = [r.sem for r in rungs if r.sem is not None]
    if sems:
        f = max(sems)
        return f, (f"the largest standard error measured across repeats ({f:.5f}); "
                   f"{sum(1 for r in rungs if r.n > 1)} of {len(rungs)} rungs were repeated")
    return SEED_SIGMA, (f"the baseline's published 5-seed sigma ({SEED_SIGMA}), used because no "
                        f"rung has been repeated -- repeat the ladder to measure it directly")


# ----------------------------------------------------------------------------- sections

def _config(runs: list) -> None:
    """Everything that must be identical for the comparison to mean anything."""
    print("## Configuration\n")
    fields = [("model", "model"), ("data_contract", "data contract"),
              ("epsilon", "epsilon"), ("patience", "N (patience)"),
              ("iteration_cap", "iteration cap"), ("dataset", "dataset")]
    mixed = []
    print("| setting | value |")
    print("|---|---|")
    for key, label in fields:
        vals = {r["meta"].get(key) for r in runs}
        shown = ", ".join(sorted(str(v) for v in vals))
        if len(vals) > 1:
            mixed.append(label)
            shown = f"**MIXED: {shown}**"
        print(f"| {label} | {shown} |")
    print(f"| runs compared | {len(runs)} |")
    if mixed:
        print(f"\n**These runs differ in {', '.join(mixed)}, so the comparison below is "
              f"confounded.** Re-run the ladder under one configuration before citing it.")
    bad = [r for r in runs if r["meta"].get("stop_reason") != "converged"]
    if bad:
        print(f"\n**{len(bad)} run(s) did not converge normally** and may distort every average "
              f"below: " + ", ".join(f"`{r['dir'].name}` ({r['meta'].get('stop_reason')})"
                                     for r in bad))
    if PRICE_IN is None:
        print("\nCost columns are omitted: the run records hold token counts, not prices. Pass "
              "`--price-in=` and `--price-out=` in USD per 1M tokens to include them.")


def _headline(rungs: list) -> None:
    print("\n## What each rung produced\n")
    print("Same agent, same dataset, same data contract, same convergence rule. The only "
          "difference is `--slots`. Where a slot count has repeats, the cell shows the mean "
          "with the observed range in brackets.\n")
    cost_hdr = " cost |" if PRICE_IN is not None else ""
    print(f"| slots | runs | validation | hidden test | vs baseline | wall clock | tokens |"
          f"{cost_hdr} scripts | turns |")
    print(f"|---|---|---|---|---|---|---|{'---|' if PRICE_IN is not None else ''}---|---|")
    for g in rungs:
        c = [_cost(r["meta"]) for r in g.runs]
        cost_cell = f" ${sum(x for x in c if x)/g.n:.2f} |" if c and c[0] is not None else ""
        scripts = "/".join(str(r["meta"].get("scripts", "?")) for r in g.runs)
        turns = "/".join(str(r["meta"].get("turns", "?")) for r in g.runs)
        print(f"| **{g.slots}** | {g.n} | {g.cell(g.valids)} | **{g.cell(g.tests)}** | "
              f"{g.test - BASE_TEST:+.6f} | {g.cell(g.walls, '{:.1f}')} min | "
              f"{g.cell(g.tokens, '{:,.0f}')} |{cost_cell} {scripts} | {turns} |")
    print(f"\nBaseline for the `vs baseline` column is the {BASE_LABEL}: "
          f"validation {BASE_VALID:.4f}, hidden test {BASE_TEST:.4f}. `scripts` and `turns` are "
          f"listed per run rather than averaged, because they are counts of work done and an "
          f"average of them describes no run that was actually executed.")


def _plateau(rungs: list, floor: float, floor_note: str) -> None:
    """The load-bearing section: marginal test gain per added slot, against the noise floor."""
    print("\n## Where the score stops moving\n")
    print(f"Marginal change from adding one more lineage. The noise floor here is {floor_note}. "
          f"A difference below it is not evidence of anything.\n")
    print("| step | test | delta | reading | tokens | delta | wall | delta |")
    print("|---|---|---|---|---|---|---|---|")
    prev = None
    for g in rungs:
        if prev is None:
            print(f"| {g.slots} slot | {g.test:.6f} | - | starting point | {g.tok:,.0f} | - | "
                  f"{g.wall:.1f} min | - |")
        else:
            d = g.test - prev.test
            reading = ("within noise" if abs(d) < floor
                       else "under epsilon" if abs(d) < EPSILON else "above epsilon")
            print(f"| {prev.slots} -> {g.slots} | {g.test:.6f} | {d:+.6f} | {reading} | "
                  f"{g.tok:,.0f} | {g.tok-prev.tok:+,.0f} | {g.wall:.1f} min | "
                  f"{g.wall-prev.wall:+.1f} |")
        prev = g

    best = max(rungs, key=lambda g: g.test)
    flat = [g for g in rungs if abs(g.test - best.test) < floor]
    if len(flat) > 1:
        lo = min(g.slots for g in flat)
        tok_lo, tok_hi = min(g.tok for g in flat), max(g.tok for g in flat)
        w_lo, w_hi = min(g.wall for g in flat), max(g.wall for g in flat)
        plural = "slot" if lo == 1 else "slots"
        print(f"\n**The curve is flat from {lo} {plural} onward.** Rungs "
              f"{', '.join(str(g.slots) for g in flat)} all land within {floor:.5f} of the best "
              f"result ({best.test:.6f} at {best.slots} slots), so on this evidence they are "
              f"indistinguishable on the hidden test. Across that flat span the spend still "
              f"rises from {tok_lo:,.0f} to {tok_hi:,.0f} tokens ({tok_hi/tok_lo:.1f}x) and from "
              f"{w_lo:.1f} to {w_hi:.1f} minutes.")
        print(f"\nThe honest reading is that everything past {lo} {plural} is bought and not "
              f"delivered: cost scales close to linearly in the slot count while the scored "
              f"result does not move outside noise.")
    else:
        print(f"\nNo two rungs land within {floor:.5f} of each other, so this ladder does not "
              f"show a plateau. Either the slot count genuinely matters here or the rungs are "
              f"too few to tell.")


def _efficiency(rungs: list) -> None:
    print("\n## What the spend bought\n")
    print("| slots | tokens | tokens/slot | wall clock | script time | candidates compared | "
          "candidates/1k tokens |")
    print("|---|---|---|---|---|---|---|")
    for g in rungs:
        cand = sum((r["meta"].get("candidates_evaluated") or 0) for r in g.runs) / g.n
        script = sum(r["meta"].get("script_seconds", 0) for r in g.runs) / g.n / 60
        print(f"| {g.slots} | {g.tok:,.0f} | {g.tok/g.slots:,.0f} | {g.wall:.1f} min | "
              f"{script:.1f} min | {cand:.0f} | {1000*cand/g.tok:.2f} |")
    print("\n`candidates compared` counts the alternatives evaluated INSIDE scripts, which is "
          "where most of the search happens: a script may build and compare a dozen models for "
          "one iteration of budget. It rises with the slot count, so the extra lineages are "
          "doing real work -- the question the table above answers is whether that work reaches "
          "the score.")


def _diversity(rungs: list) -> None:
    """Slot correlation is the portfolio's own acceptance test, so it belongs beside the cost."""
    have = [(g, r) for g in rungs for r in g.runs if r["corr"]]
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
    print("| slots | run | per-turn mean | mean across turns | last turn (`mean_slot_correlation`) | reading |")
    print("|---|---|---|---|---|---|")
    for g, r in have:
        per = " -> ".join(f"{c:.3f}" for c in r["corr"])
        avg = sum(r["corr"]) / len(r["corr"])
        last = r["meta"].get("mean_slot_correlation")
        reading = ("lineages genuinely differ" if avg < 0.90
                   else "borderline" if avg < 0.95 else "effectively copies")
        last_cell = f"{last:.3f}" if isinstance(last, (int, float)) else "-"
        print(f"| {g.slots} | `{r['dir'].name}` | {per} | **{avg:.3f}** | {last_cell} | {reading} |")
    print("\nThe last-turn column is what `run_meta.json` records; it is a single turn and "
          "swings widely, so the verdict above is taken from the average across turns.")


def _reliability(rungs: list) -> None:
    print("\n## Reliability across the ladder\n")
    print("| slots | run | scripts | crashes | integrity rejections | manual interventions | stop reason |")
    print("|---|---|---|---|---|---|---|")
    for g in rungs:
        for r in g.runs:
            m = r["meta"]
            crashes = sum(1 for e in r["rows"] if e.get("status") in ("failed", "blacklisted"))
            print(f"| {g.slots} | `{r['dir'].name}` | {m.get('scripts','?')} | {crashes} | "
                  f"{m.get('integrity_rejections',0)} | {m.get('manual_interventions',0)} | "
                  f"`{m.get('stop_reason','?')}` |")
    print("\n`crashes` counts scripts that failed to produce a scored result, separately from "
          "integrity rejections, which are results the critic refused. Both are handled in the "
          "loop; neither reaches a human.")


def _curves(rungs: list) -> None:
    print("\n## Score curve within each run\n")
    print("The run's best validation score after each scored improve iteration. This is the "
          "curve the convergence rule is measured against.\n")
    for g in rungs:
        for r in g.runs:
            best, curve = float("-inf"), []
            for e in r["rows"]:
                if e.get("phase") != "improve":
                    continue
                p = (e.get("metrics") or {}).get("primary")
                if isinstance(p, (int, float)):
                    best = max(best, p)
                    curve.append(best)
            if curve:
                print(f"- **{g.slots} slot(s)** (`{r['dir'].name}`): "
                      + " -> ".join(f"{c:.4f}" for c in curve))


def _generalization(rungs: list, floor: float) -> None:
    """Does the extra search buy validation or test?

    The portfolio compares more candidates per turn, and every candidate is compared on the same
    validation split. The more you compare, the more the winner is inflated by validation noise
    it does not carry to test -- the winner's curse. If that is happening here, the gap between
    validation and test should widen with the slot count while test itself stays flat.
    """
    print("\n## Validation vs test: is the extra search buying score or noise?\n")
    print("`gap` is validation minus hidden test for the submitted iteration. A gap that widens "
          "with the slot count, while test does not improve, means the added lineages are "
          "winning on validation noise rather than finding transferable signal.\n")
    print("| slots | candidates compared | validation | hidden test | gap | gap vs 1 slot |")
    print("|---|---|---|---|---|---|")
    base_gap = None
    rows = []
    for g in rungs:
        cand = sum((r["meta"].get("candidates_evaluated") or 0) for r in g.runs) / g.n
        gap = g.valid - g.test
        if base_gap is None:
            base_gap = gap
        rows.append((g.slots, cand, gap))
        print(f"| {g.slots} | {cand:.0f} | {g.valid:.6f} | {g.test:.6f} | {gap:.6f} | "
              f"{gap - base_gap:+.6f} |")
    # Rank correlation between candidates compared and the generalisation gap. Spearman on a
    # handful of points is weak evidence, so it is reported with its n and nothing more.
    if len(rows) >= 3:
        def rank(vals):
            order = sorted(range(len(vals)), key=lambda i: vals[i])
            rk = [0] * len(vals)
            for pos, i in enumerate(order):
                rk[i] = pos
            return rk
        rc, rg = rank([r[1] for r in rows]), rank([r[2] for r in rows])
        n = len(rows)
        d2 = sum((a - b) ** 2 for a, b in zip(rc, rg))
        rho = 1 - 6 * d2 / (n * (n * n - 1))
        print(f"\nRank correlation between candidates compared and the validation-test gap: "
              f"**{rho:+.2f}** over {n} rungs. Positive means more comparison produces more "
              f"validation inflation. At n={n} this is suggestive at best.")
    spread = max(r[2] for r in rows) - min(r[2] for r in rows)
    print(f"\nThe gap spans {spread:.6f} across the ladder, against a noise floor of "
          f"{floor:.5f}. " + ("That is larger than the floor, so the differences in "
          "generalisation are worth taking seriously." if spread > floor else
          "That is inside the floor, so no rung generalises measurably better than another."))


def _concentration(rungs: list) -> None:
    """Where in a run the score actually moves.

    If nearly all the gain lands in the first few iterations, the budget spent after that is
    the real waste -- independent of how many lineages are running.
    """
    print("\n## Where the gain happens inside a run\n")
    print("Share of each run's total validation gain reached by iteration k. The convergence "
          "rule stops these runs at turn 4-5; this shows how much was already banked by then.\n")
    print("| slots | run | improve iters | total gain | by #1 | by #2 | by #4 | last gain at |")
    print("|---|---|---|---|---|---|---|---|")
    for g in rungs:
        for r in g.runs:
            best, curve = float("-inf"), []
            for e in r["rows"]:
                if e.get("phase") != "improve":
                    continue
                p = (e.get("metrics") or {}).get("primary")
                if isinstance(p, (int, float)):
                    best = max(best, p)
                    curve.append(best)
            if len(curve) < 2:
                continue
            total = curve[-1] - curve[0]
            last_gain = max(i for i in range(len(curve))
                            if i == 0 or curve[i] > curve[i - 1]) + 1
            def share(k):
                if total <= 0 or k >= len(curve):
                    return "-"
                return f"{(curve[k] - curve[0]) / total:.0%}"
            print(f"| {g.slots} | `{r['dir'].name}` | {len(curve)} | {total:+.5f} | "
                  f"{share(1)} | {share(2)} | {share(4)} | #{last_gain} of {len(curve)} |")
    print("\n`last gain at` is the final iteration that set a new best. A run whose last gain "
          "is well before its final iteration spent the remainder finding nothing.")


def _frontier(rungs: list, floor: float) -> None:
    """What each rung costs per unit of score it actually delivers."""
    print("\n## Cost of the result\n")
    best = max(rungs, key=lambda g: g.test)
    cheapest = min(rungs, key=lambda g: g.tok)
    print("| slots | tokens | wall | test | tokens per 0.0001 of test delta | verdict |")
    print("|---|---|---|---|---|---|")
    for g in rungs:
        d = g.test - BASE_TEST
        per = f"{g.tok / (d / 0.0001):,.0f}" if d > 0 else "-"
        if abs(g.test - best.test) < floor and g.tok <= cheapest.tok * 1.05:
            verdict = "**best value**"
        elif abs(g.test - best.test) < floor:
            verdict = f"ties the best for {g.tok/cheapest.tok:.1f}x the cost"
        else:
            verdict = "below the best"
        print(f"| {g.slots} | {g.tok:,.0f} | {g.wall:.1f} min | {g.test:.6f} | {per} | {verdict} |")
    print(f"\nThe `tokens per 0.0001` column divides the whole run's spend by the test gain "
          f"over the {BASE_LABEL}. It is a blunt figure -- it charges the baseline "
          f"reproduction and EDA to the improvement -- but it is the number that decides "
          f"whether a rung is worth running.")


def _time_split(rungs: list) -> None:
    """Where wall-clock time goes, which is what decides whether runs can be parallelised."""
    print("\n## Where the wall-clock time goes\n")
    print("`script` is time executing the agent's code; the remainder is dominated by LLM "
          "latency. The split matters operationally: a run that is mostly latency parallelises "
          "almost free, while one that is mostly compute contends for cores.\n")
    print("| slots | wall | script time | script share | remainder (LLM etc.) |")
    print("|---|---|---|---|---|")
    for g in rungs:
        script = sum(r["meta"].get("script_seconds", 0) for r in g.runs) / g.n / 60
        share = script / g.wall if g.wall else 0
        print(f"| {g.slots} | {g.wall:.1f} min | {script:.1f} min | {share:.0%} | "
              f"{g.wall - script:.1f} min |")
    shares = [(g.slots, sum(r["meta"].get("script_seconds", 0) for r in g.runs)
               / g.n / 60 / (g.wall or 1)) for g in rungs]
    hi_s, hi_v = max(shares, key=lambda x: x[1])
    lo_s, lo_v = min(shares, key=lambda x: x[1])
    print(f"\nCompute share ranges from {lo_v:.0%} ({lo_s} slots) to {hi_v:.0%} ({hi_s} slots)"
          + ("" if hi_s == max(s for s, _ in shares) else
             f" -- it does not rise monotonically with the slot count, because what a script "
             f"costs depends on the model the agent chose to write, not only on how many "
             f"scripts run")
          + ". Slots already execute concurrently inside a run (`agent/loop.py:737`) and no "
            "thread counts are pinned, so running several ladder rungs at once would contend "
            "for cores, and would contend by different amounts per rung. That is why this "
            "ladder is run sequentially: the wall-clock and token columns above are the "
            "comparison, and overlapping the runs would corrupt them.")


def _caveat(rungs: list, floor: float) -> None:
    print("\n## Caveat\n")
    reps = {g.n for g in rungs}
    if reps == {1}:
        print(f"Each rung is a single run. Run-to-run spread on this benchmark is roughly "
              f"{SEED_SIGMA} on the scored metric -- the same size as most of the differences "
              f"above -- so this ladder shows the SHAPE of the cost/score curve rather than a "
              f"precise value for any rung, and any plateau claim rests on several rungs "
              f"agreeing rather than on any single pair. Repeating each rung would replace that "
              f"assumed noise figure with a measured one.")
    else:
        worst = max(rungs, key=lambda g: g.spread or 0)
        print(f"Repeats per rung: {', '.join(f'{g.slots} slots x{g.n}' for g in rungs)}. The "
              f"widest observed spread within one slot count is {worst.spread:.5f} "
              f"(at {worst.slots} slots), against a noise floor of {floor:.5f} used above. "
              f"Differences between rungs smaller than that spread are not attributable to the "
              f"slot count.")


# ----------------------------------------------------------------------------- entry

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
    for name in (args or _discover()):
        run = _load(Path(name))
        if run is None:
            print(f"skipping {name}: no run_meta.json", file=sys.stderr)
            continue
        loaded.append(run)
    if not loaded:
        print("no runs to compare", file=sys.stderr)
        raise SystemExit(1)
    rungs = _rungs(loaded)
    floor, floor_note = _noise_floor(rungs)

    print("# Slot-ladder comparison\n")
    print(f"How many solution lineages the agent should advance per turn. {len(loaded)} runs "
          f"across {len(rungs)} slot counts, {min(g.slots for g in rungs)} to "
          f"{max(g.slots for g in rungs)} lineages per turn.\n")
    _config(loaded)
    _headline(rungs)
    _plateau(rungs, floor, floor_note)
    _generalization(rungs, floor)
    _concentration(rungs)
    _frontier(rungs, floor)
    _efficiency(rungs)
    _time_split(rungs)
    _diversity(rungs)
    _reliability(rungs)
    _curves(rungs)
    _caveat(rungs, floor)


if __name__ == "__main__":
    main()
