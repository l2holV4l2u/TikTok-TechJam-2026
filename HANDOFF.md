# Handoff — portfolio search

What was built, how to run it, what is proven, and what is not.

Implements `docs/portfolio-plan.md`, Phases 0–6. Branch `mon`, seven commits, not pushed.

---

## 1. The one-paragraph version

The loop can now advance **n solution lineages per turn** instead of one, without leaving the
organizers' convergence rule. A turn launches one script per slot, the turn's score is the best
of them, and that single curve drives the one counter the rules recognise. A slot that stops
paying is archived and its place refilled — three fresh drafts to one revival, and the revival
is chosen for score *and* for disagreeing with what is already live. One consultant call per
turn synthesises across the slots and the archive. Every turn the controller blends the
incumbent, the live slots and the archive, choosing members on one half of validation and
confirming on the other. **The default is still `--slots 1`, which is the old sequential loop,
byte-identical.**

## 2. How to run it

```bash
python run_agent.py --run-dir runs/rN --slots 3        # portfolio
python run_agent.py --run-dir runs/rN                  # unchanged sequential loop (default)
python report_run.py runs/rN > RUN_REPORT.md
```

Offline, no network, no dataset needed beyond a synthetic cache:

```bash
python -m pipeline.synth --out-dir data/cache
python run_agent.py --dry-run --run-dir /tmp/x --iters 12 --no-memory --slots 3
```

`--slots` accepts 1, 2 or 3 only. The cap is deliberate; see §5.

## 3. What each new file does

| File | Role |
|---|---|
| `agent/portfolio.py` | `Slot`, `Archive`, `ArchiveEntry`, `refill`, `pairwise_rank_correlation`. The portfolio's state and its acceptance test. |
| `agent/selection.py` | User-grouped fold split and `accept()` — the winner's-curse guard. |
| `agent/consultant.py` | One LLM call per turn: shared belief set plus one note per slot. |
| `agent/ensemble.py` | `blend_portfolio()` added beside the existing `retain_or_blend`. |
| `tests/test_portfolio.py` | 31 tests: turns, slots, archive, refill, consultant. |
| `tests/test_selection.py` | 10 tests: folds and the acceptance guard. |
| `tests/test_report.py` | 10 tests: slot lanes, portfolio sections, code diffs. |

New run artifacts, all under `runs/<id>/`:

```
portfolio.jsonl        one record per turn (correlation), plus refill and blend events
archive.jsonl          retired lineages, with why each stopped
archive/entry_N_*.npy  their predictions — the blend pool
portfolio_blend/       the accepted blend's scores and provenance
artifacts/slot_N/      per-slot scratch (was one shared directory)
artifacts/shared/      the trusted incumbent, controller-written only
```

## 4. The rule, and why the design is shaped this way

The spec is singular: *"**a run** is converged when validation score has not improved by more
than ε over the last N = 3 consecutive iterations."* There is no per-lineage counter in it. So
slots do not get their own — a slot that stalls is **recycled, not stopped**, which is a
resource decision inside a turn and something the rule says nothing about.

"Iteration" has two defensible readings once a turn launches several scripts, so **both are
reported** rather than the flattering one:

- `turns` — one hypothesis-to-score cycle, Figure 1's iteration, what convergence measures.
- `scripts` — every script executed, the stricter reading of the 50-iteration cap.

At three slots converging in ~6 turns that is 18 scripts, under the cap either way.

## 5. Why `--slots` stops at 3

Selecting the max of *k* candidates on one validation split returns a number inflated by the
selection itself. Against the baseline's reported 5-seed σ of 0.0008:

| k | inflation |
|---|---|
| 8 (the sequential loop today) | +0.00114 |
| 18 (three slots, ~6 turns) | +0.00145 |
| 50 | +0.00180 |

Three slots cost about **+0.0003** of validation that will not transfer, against an effect of
roughly +0.005 on test. Eight would start eating the result. The run's own history already
shows the failure mode: r74 has the better validation (0.6049 vs 0.6047) and the worse hidden
test (0.5991 vs 0.5998).

`agent/selection.py::accept` is the guard: a candidate must gain on fold A **and** not collapse
on fold B, folds grouped by user because both metrics are per-user means.

## 6. Status of each phase

| Phase | State | Evidence |
|---|---|---|
| 0 — safety rails | done | per-slot artifact dirs asserted by test; `tests/test_selection.py` 10 tests |
| 1 — parallel turns | done | 3 scripts/turn, one convergence curve, `--slots 1` reproduces the old ledger |
| 2 — diversity + gate | **done, not yet answered** | see §7 |
| 3 — archive & refill | done | 30-script dry run: 9 archived, alternation `fresh×3 → revived → fresh×2 → revived` |
| 4 — consultant | done | 12-script dry run: 12 propose calls, **4** consultant calls (one per turn) |
| 5 — portfolio blend | done | greedy forward selection, fold-B confirmation, fold-A-only winner refused |
| 6 — reporting | done | slot lanes, portfolio section, correlation trace, code diffs |

## 7. The open question — read this before trusting Phases 3–5

**Phase 2 is a go/no-go gate and it has not been answered.** The machinery is built and
tested; the *measurement* requires a real run against the real dataset, which needs the 195 MB
KuaiRand-Pure download and API credentials. Neither is present here.

The gate: run three turns at `--slots 3`, then read `mean_slot_correlation` from
`run_meta.json` or the correlation table in the report.

- **< 0.90** → proceed. The portfolio is doing what it is for.
- **0.90–0.95** → tighten `_SIBLINGS` in `agent/proposer.py`, re-run once, decide.
- **> 0.95** → **stop.** You have three expensive copies of one agent. Keep Phases 0–1 —
  parallelism still buys wall-clock — and abandon 3–5. Record the number; a negative result
  here is a genuine finding and belongs in the write-up, not in a drawer.

This matters because the measured bottleneck on this benchmark is not search breadth. It is
that everything found correlates: components at 0.94+, MMoE at 0.9888 against plain DeepFM,
and blends that gained nothing as a result. The portfolio is built for the **decorrelated
ensemble pool**, not for the wider search. If the pool is not decorrelated, it has no reason to
exist.

What the dry run *can* show is that the gate works: its canned proposer returns the same script
three times and the correlation reports 1.0000 with `alert: true`, and the blend then declines
every turn with *"no member improved fold A"*. Refusing to blend copies of one model is the
correct outcome.

## 8. Out-of-plan repairs

Five test modules failed from a clean checkout before any of this started, and none of the
failures was about the behaviour under test. They were fixed first, in their own commit
(`93c9884`), because "the suite still passes" would otherwise have meant nothing.

- **Test runners had drifted from the tests.** `tests/test_proposer.py` listed a renamed test,
  so the module died on import while three real tests below the list never ran.
  `agent/demo.py` and `tests/test_harness.py` kept their runner mid-file, silently skipping
  everything defined after it — two tests in `test_harness`, covering the sweep-mode change
  itself. All three now discover tests from `globals()` at the end of the file.
- **Four assertions in `agent/demo.py` pinned the pre-sweep search policy** and described a
  loop that no longer exists. They now assert the behaviour they were written for, and reach
  the refine/broaden ladder through `--force-mode`.
- **`pipeline/synth.py` was two channels behind `pipeline/data.py`.** `load()` reads `date`,
  `time_ms` and `num`; the synthetic cache wrote none of them, so those tests could not pass on
  any machine without the real dataset.
- **`blend/__init__.py` imported lightgbm eagerly**, making the pure-numpy `blend.weights`
  unimportable without it.

Result: **12 pass / 5 fail → 17 pass / 0 fail**, before a line of the plan was written.

## 9. Verified

```
23 test modules, 0 failures
  agent.{tree,knowledge,memory,demo,critic,diagnose,portfolio,selection,consultant}
  tests.{proposer,llm,evaluate,submit,data,models,executor,harness,history,
         ensemble,weights,selection,portfolio,report}
```

End to end at every slot count, offline:

| | exit | stop | turns | scripts | archived | report | submission |
|---|---|---|---|---|---|---|---|
| `--slots 1` | 0 | converged | 4 | 6 | 0 | ok | valid |
| `--slots 2` | 0 | converged | 4 | 10 | 2 | ok | valid |
| `--slots 3` | 0 | max_iters | 4 | 12 | 3 | ok | valid |

`python -m pipeline.submit --check` passes on the three-slot submission. The committed r70,
r59 and r38_1k reports still render, and r70 now shows the four real code diffs its iterations
applied.

## 10. Not done, and known

- **The gate is unanswered** (§7). Everything in Phases 3–5 is conditional on it.
- **Wall-clock speedup is unmeasured.** The dry run's scripts finish in milliseconds, so it
  says nothing about contention. Expect 1.5–2× throughput at three slots on one CPU box, not
  3×; the scripts are CPU-bound and will contend.
- **The consultant has never run against a real model.** Its parsing, its failure modes and its
  token cost are covered by tests with a scripted `complete`, but no real reply has been seen.
- **`blend_portfolio` uses equal weights.** Deliberate — a searched weight vector overfits the
  split the plan is trying to protect — but it is a choice, not a proven optimum.
- **Two pre-existing dead imports** remain in `run_agent.py` (`NUMERIC_FEATURES`, `_os`). Left
  alone as out of scope.
- **Nothing is pushed.** Seven commits sit on `mon`.

## 11. Commits

```
93c9884  Restore the test suite to green before building on it
894df1c  Advance n solution lineages per turn under one convergence counter   (Phases 0–1)
4dacf27  Tell each slot what its siblings are attempting, and measure whether it helped  (2)
06ddbbe  Archive a stalled lineage and reuse its slot, decorrelating the revival          (3)
7cfaf7d  Synthesise across the slots once per turn instead of per experiment              (4)
a8583e5  Blend the whole portfolio, choosing on one fold and confirming on the other      (5)
b9051b0  Show the portfolio in the run report, and render the code diff the spec asks for (6)
aa21934  Keep crash feedback with the lineage that produced it
```

The last one is a bug the portfolio introduced, found in review: `feedback` was run-scoped, so
a crash in one slot handed a different slot a traceback from a script it never wrote along with
"keep the same hypothesis". It is now per-slot, with a test that crashes exactly one.
