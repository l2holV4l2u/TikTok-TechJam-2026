# DOOMS — 3-minute pitch

Deck: `Copy of Dark Theme by Slidesgo.pdf` (8 slides, ends on divider 03).
Budget: 180s. Word counts assume ~150 wpm.

Each slide gets a **thesis** — the one claim it exists to prove. If a sentence on the
slide doesn't serve its thesis, cut it. If the script drifts off the thesis, cut the script.

---

## S1 · Title — 12s

**Thesis:** *We built an ML researcher, not an ML model.*

> DOOMS is an autonomous machine learning research agent.
> Give it a dataset it has never seen and a metric, and it does the whole research loop
> itself — reads the data, forms a hypothesis, writes the code, runs it, and decides
> what to try next.
> We pointed it at a Kuaishou short-video recommender benchmark.

---

## S2 · Results — 25s

**Thesis:** *It beat the official baseline, and the headline is that nobody helped it.*

> It beat the organizers' official baseline on the hidden test set.
>
> But the number we care about is the last line. **No human in the loop.**
> Nobody debugged a script, nobody picked a model, nobody restarted a failed run.
> The agent that produced that result never asked us anything.
>
> So — how?

*Delivery: land on "no human in the loop," then straight into the divider. Don't dwell
on the token and wall-clock figures; they're on screen, and they read as cost, not win.*

---

## S3 · Divider 01 "How we did it?" — 3s

**Thesis:** *transition.* Say nothing, or just "How it works."

---

## S4 · Solution Architecture — 50s

**Thesis:** *The agent proposes; only the harness judges. That separation is the design.*

> **[harness → gpt 5.6 sol]**
> The harness runs the loop. It hands the model the task and the tools —
> and deliberately nothing about what works on this data.
> So every idea you see is the agent's own, not ours smuggled in through a prompt.
>
> **[code script → score]**
> It writes a complete experiment, and here's the important part —
> we throw away the score it reports and re-score it ourselves.
> It also can't see the test answers at all; the data physically isn't there.
> The agent proposes. Only the harness judges.
>
> **[DOOMS → slots → DOOM'S HISTORY]**
> And it doesn't chase one idea. DOOMS runs three at once.
> Every result becomes a node in DOOM'S TREE.
> When a line stops paying, it retires to DOOM'S HISTORY — and when we bring one back,
> we take the most **different** one, not the best one.
> A second opinion that agrees with you is worth nothing.

*Delivery: this is the heart of the pitch. Pause before "Only the harness judges."*

---

## S5 · Divider 02 "How we optimize it?" — 3s

**Thesis:** *transition.*

---

## S6 · Portfolio mode — 25s

**Thesis:** *Parallel slots are only worth it if they actually disagree — so we measured that.*

> Running slots in parallel is easy. Making them worth the cost is not.
> Three lineages that reach the same answer cost three times as much
> and tell you one thing.
>
> So every turn we measure how much the slots actually disagree,
> and the best of the turn carries forward.
> It's a real acceptance test for the design, not a dashboard.

---

## S7 · Optimal E and N — 22s

**Thesis:** *We didn't guess the stopping rule — we measured all 80 of them.*

> The run stops when the score stops improving. But *by how much*, and *for how long*,
> are ours to choose — and they decide whether a run dies early or wastes an hour.
>
> So we swept every combination and scored each one on the hidden test set.
> Eighty cells. The answer came back E = 0.002, N = 3.
> We kept those — now because we measured it, not because it was the default.

*Delivery: "not because it was the default" is the point. Say it deliberately.*

---

## S8 · Divider 03 "How we visualize it?" — 3s

**Thesis:** *transition.*

---

## S9+ · Visualization — 25s  *(slides not built yet)*

**Thesis:** *Every claim we just made is re-derivable from the run records — here's the proof.*

> Everything we've claimed comes out of the run logs, not a write-up.
> The viewer reads them straight off disk and classifies every run itself —
> which ones count, which ones are disqualified, and why.
>
> Including the runs that failed. We didn't curate this.

---

## Close — 12s

**Thesis:** *Autonomy you can audit.*

> Most agent demos ask you to trust the output.
> DOOMS is built so you don't have to — it can't grade itself,
> and every number is re-derivable.
>
> Autonomy you can actually audit. That's DOOMS.

---

# Fix before you present

**S2 `+XX.XX` placeholder.** Our best eligible run is `r79`: **+0.00509** on hidden test
(0.59969 vs the official 0.59460). Verify with `python -m research.verify_claims`.

**S2 token / wall-clock figures don't reconcile.** The slide says ~6,767k tokens and
~420m. From the repo:

| source | tokens | wall-clock |
|---|---|---|
| sum of all `run_meta.json` (38 runs) | 4,811k | 809 min |
| sum of all `llm_calls.jsonl` (45 dirs) | 5,713k | — |
| **slide** | **6,767k** | **420 min** |

Neither reconciles. Decide what the figure counts — all runs, or only eligible ones —
and regenerate it. A judge who checks will find the gap.

**S7's 80-cell grid has no inputs in the repo.** `research/convergence_sweep.py` is
finished, but the three `runs/convergence_sweep/curve_*` inputs hold only 3–4 ledger rows
each and no `run_meta.json`, and there's no output under `reports/`. If the grid was made
elsewhere, commit the curves and the output before claiming it on stage.

**Don't claim full rule-compliance.** "Beat the official baseline" is verified and safe.
But `r79` ran under `train-plus-valid-v2`, which our own code now marks superseded, and no
run exists yet under `train-only-v3`. If asked specifically about the training-split rule,
say that plainly.
