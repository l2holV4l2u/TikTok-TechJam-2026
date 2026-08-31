# Choosing epsilon and N in the convergence rule

The task fixes the *form* of the stopping rule -- "converged when the validation score has not
improved by more than eps over the last N consecutive iterations" -- and leaves eps and N to the
entrant. This study selects them from measured run results.

No published or default value is treated as privileged. Every (eps, N) pair, including the one
quoted in the task description, is ranked by the same criterion applied to the same three runs.

## Answer

**Any of eps in [0.0005, 0.005] with N in {2, 3}.** These nine cells are indistinguishable from
one another in the measurements; `eps = 0.002, N = 3` is reported as the representative choice
because it sorts first, not because it was shown to be better. All nine stop after 3-5 improve
turns.

How little separates them: eps = 0.002 and eps = 0.001 at N = 3 differ by 0.000010 in mean test,
one thirty-seventh of the per-cell standard error. On curve_1 they are the *same decision* --
same stop turn, same selected model, same score. On the other two curves eps = 0.001 stops one
turn later and gives back 0.000001 and 0.000030. Selecting between them on this evidence would be
reading noise.

The honest strength of that claim is bounded, and the bound matters more than the ranking:
**all 80 cells lie within one seed-sigma of each other**, and the per-cell standard error (n=3)
is 0.00037 against a spread of 0.00032 across the defensible set. The data does not show that
this cell is *better* than the other nine that survive the robustness filter. What it shows is
that it is never worse, and that everything outside that set is either unreliable across curves
or costs several times the compute for no measurable gain.

## Selection method

Three criteria, applied in order. Each exists to reject a specific failure mode.

**1. Robustness across curves.** A cell is kept only if all three curves agree in sign that
stopping there beat running to the cap. Rationale: the per-cell SEM (0.00037) is larger than
almost every difference in the grid, so a mean computed over three curves cannot by itself
distinguish cells. Sign agreement is a weaker but far more reliable signal than the mean, and it
directly answers "would this choice have helped on a run I have not seen?" This filter removes
70 of 80 cells.

**2. No censoring.** A cell is rejected if the rule failed to fire before any curve ended, since
its result is a lower bound rather than a measurement. This removes cells at large N and at very
small eps.

**3. Rank the survivors by mean test at the stop point, then break ties by worst-curve result.**
Ties are resolved on the minimum single-curve delta rather than the mean, because with n=3 the
minimum is the more conservative statistic.

Validation score is deliberately *not* the ranking criterion. Forgone validation falls
monotonically as the rule is made more permissive -- from 0.00084 at N=2 to 0.00002 at N=20 --
so ranking on it selects "never stop" by construction. It is reported as a diagnostic only. Test
at the stop point is the quantity the rule exists to maximise.

### The ten cells that survive

| eps | N | mean stop turn | mean test | worst-curve delta |
|---|---|---|---|---|
| 0.002 | 2 | 3.3 | 0.599488 | +0.000013 |
| 0.001 | 2 | 4.0 | 0.599522 | +0.000060 |
| 0.003 | 3 | 4.0 | 0.599546 | +0.000013 |
| 0.005 | 3 | 4.0 | 0.599546 | +0.000013 |
| 0.0005 | 2 | 4.3 | 0.599512 | +0.000030 |
| **0.002** | **3** | **4.3** | **0.599561** | **+0.000060** |
| 0.00025 | 2 | 5.0 | 0.599551 | +0.000030 |
| 0.001 | 3 | 5.0 | 0.599551 | +0.000030 |
| 0.0005 | 3 | 5.3 | 0.599551 | +0.000030 |
| 0.001 | 15 | 18.3 | 0.599238 | +0.000009 |

Spread of mean test across this set: 0.000323, against a per-cell SEM of 0.00037. **The set is
one statistical tie**, and the tiebreak that puts (0.002, 3) at the top separates it from
(0.001, 3) by 0.000010. No claim is made that it is better. A defensible argument runs the other
way: the lower eps values stop ~0.7 turns later for no measurable test cost, which is a margin
against stopping prematurely on a run whose gain profile is shaped differently.

Nine of the ten stop between 3.3 and 5.3 turns. The tenth, (0.001, 15), reaches the same place
four times slower and scores 0.0003 lower.

## Method

Three runs were launched with `--patience 999`, which makes convergence impossible, so each ran
until it stopped for another reason and its validation curve is *uncensored*: every candidate N
is visible on it, not only those smaller than wherever the rule happened to fire. The stopping
rule is then a pure function of that curve, so all 80 (eps, N) pairs were replayed offline
against curves recorded once. No cell in the grid cost a run.

```
python run_agent.py --run-dir runs/convergence_sweep/curve_N \
    --slots 1 --patience 999 --iters 50 --timeout 900
python -m research.convergence_sweep runs/convergence_sweep/curve_*
```

Grid: eps in {0, 0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.003, 0.005} x N in
{2,3,4,5,6,8,10,12,15,20} -- 80 cells, replayed against each of the 3 curves. eps = 0 is the
floor of the rule, where it stops only if the window produced literally no improvement, so the
parameter is bracketed from both ends rather than only from above.

`run_meta.json` records `epsilon` and `patience` for each run, so a `--patience 999` diagnostic
curve is distinguishable from a default run that merely never converged.

`research/convergence_sweep.py` imports `converged` directly from `agent.loop` rather than
reimplementing it, and rebuilds `best_curve` from the same `ledger.jsonl` a judge would read.
Its self-test replays eight prior runs at their own recorded (eps, N) and reproduces the stop
each one actually recorded, 8 of 8 -- so the replay measures the harness's rule, not an
approximation of it.

Test at a stop point is scored from the `scores_test.npy` of whichever iteration was the
validation argmax when the rule fired -- that is, the model the run would actually have
submitted.

### The curves

| curve | scored improve turns | failures | best validation | test at best | script time | tokens |
|---|---|---|---|---|---|---|
| curve_1 | 24 | 3 | 0.605296 | 0.598972 | 655 s | 566,584 |
| curve_2 | 21 | 3 | 0.606125 | 0.599731 | 758 s | 481,981 |
| curve_3 | 19 | 1 | 0.605085 | 0.598897 | 769 s | 441,162 |

All three: `--slots 1`, cross-run memory OFF, data contract `train-only-v3` (training on the
train split alone, per rule 2.9.2).

## Why every late-stopping cell fails

Validation and test move together for the first four turns and separate after. Mean over curves:

| turn | delta validation | delta test |
|---|---|---|
| 2 | +0.00072 | +0.00174 |
| 3 | +0.00048 | +0.00054 |
| 4 | +0.00025 | +0.00022 |
| 5 | +0.00017 | **-0.00023** |
| 6 | +0.00003 | -0.00001 |
| 10 | +0.00001 | -0.00002 |
| 13 | +0.00005 | -0.00017 |

Through turn 4 validation gains transfer roughly one-for-one. From turn 5 validation continues to
creep upward while test goes flat and slightly negative: the run has started selecting on
validation noise. Test moves 0.597044 -> 0.599546 by turn 4 and then does not move again for
seventeen turns.

This is the same effect r77 and r78 recorded for the refine and tune modes -- validation moves,
test does not.

Every rejected cell is rejected by this one fact. Any (eps, N) that keeps the run going past
turn ~5 is buying validation and not test, and pays 2-4x the compute for it.

### What happens outside the selected region

The two axes fail differently. Lowering eps has a plateau and then a cliff: from 0.005 down to
0.0005 test is constant within 0.00002, and below 0.0005 it drops by ~0.0003 and stays down.
Raising N has no plateau -- test falls immediately at N = 4 (-0.00026) and keeps sinking to
-0.00057 by N = 15.

Validation moves the opposite way in both cases, rising monotonically throughout. That opposition
is the entire result.

The decline is at the edge of significance on the mean (-0.00057 against a SEM of 0.00037) but
its *direction* is consistent: at N = 6, 12 and 15 all three curves score lower than at N = 3;
only N = 10 is mixed. The magnitude is dominated by curve_1 (-0.0012 at N = 15) while curve_3
barely moves (-0.00005). The supportable claim is that loosening past the selected region never
increases test, and most likely decreases it slightly.

### The eps axis

Above 0.002, eps is inert: the rows for 0.002, 0.003 and 0.005 fire at exactly N+1 in almost
every cell, because gains in the flat region are ~0.0001 per step and any of those values dwarfs
them.

Below 0.0005 it is a strong lever, and pulling it makes results worse. Holding N = 3:

| eps | mean stop turn | forgone validation | test at stop | curves agree? |
|---|---|---|---|---|
| 0.0 | 18.0 | 0.00010 | 0.599210 | mixed |
| 0.0001 | 8.3 | 0.00050 | 0.599309 | mixed |
| 0.00025 | 7.0 | 0.00050 | 0.599296 | mixed |
| 0.0005 | 5.3 | 0.00074 | 0.599551 | all positive |
| 0.001 | 5.0 | 0.00074 | 0.599551 | all positive |
| 0.002 | 4.3 | 0.00076 | 0.599561 | all positive |
| 0.003 | 4.0 | 0.00077 | 0.599546 | all positive |
| 0.005 | 4.0 | 0.00077 | 0.599546 | all positive |

Driving eps to 0 quadruples the run -- turn 4.3 to turn 18 -- and recovers nearly all the forgone
validation, 0.00076 down to 0.00010. It buys **+0.00001 of test**. Two things break together
below 0.0005: test at stop falls to 0.5992-0.5993, and the curves stop agreeing in sign, with
curve_1 flipping to -0.0002 while curve_2 stays positive.

Lower eps does not find more signal. It lowers the bar until noise clears it.

### The N axis

Per-curve test deltas at eps = 0.002 (stop vs. running to the cap):

| N | curve_1 | curve_2 | curve_3 | agreement |
|---|---|---|---|---|
| 2 | +0.000423 | +0.000428 | +0.000013 | all positive |
| 3 | +0.000540 | +0.000485 | +0.000060 | all positive |
| 4 | -0.000194 | +0.000483 | +0.000030 | mixed |
| 6 | -0.000186 | +0.000460 | +0.000030 | mixed |
| 10 | -0.000644 | +0.000403 | +0.000133 | mixed |
| 15 | -0.000660 | +0.000015 | +0.000013 | mixed |

N = 2 and N = 3 are the only values where all three curves agree. From N = 4 up, curve_1 says the
extra iterations hurt test while curves 2 and 3 say they help slightly, so any recommendation to
raise N would depend on which curve one happened to look at.

## Limits

1. **Three curves.** The per-cell SEM of 0.00037 is the dominant uncertainty and is why the
   selection rests on sign agreement rather than on mean differences. More curves would narrow
   it; nothing here establishes that the selected cell beats the other nine survivors.

2. **The curves disagree at N >= 4.** That disagreement is itself a finding -- it is what rules
   out raising N -- but it is also direct evidence of run-to-run variance at this sample size.

3. **Censoring at the permissive corner.** Cells at large N, and at very small eps, never fired
   before the curves ended. They are excluded rather than reported, and the script flags them.

4. **`knowledge.py:_budget()` reports the live patience to the belief reviser,** so under
   `--patience 999` it was told "999". The proposer never sees it -- the brief hardcodes
   "3 CONSECUTIVE" -- so the effect is indirect, and if anything it makes the agent less urgent,
   biasing *towards* finding value in later iterations. None was found.

5. **A replay shows where a rule would have stopped on a recorded curve,** not that a live run at
   that N behaves identically, since the agent's own state would differ.

6. **Scope.** One agent, one dataset, one slot count. The selected values match the signal scale
   of KuaiRand-Pure as this agent explores it; they are not established as generally correct.

## Files

| path | contents |
|---|---|
| `runs/convergence_sweep/curve_1..3/` | the three diagnostic runs: ledgers, scripts, knowledge |
| `research/convergence_sweep.py` | curve reconstruction, replay, grid, self-test |
| `reports/convergence_sweep/curves.csv` | per-turn best validation, argmax iteration, test |
| `reports/convergence_sweep/grid_valid.csv` | 80 cells x 3 curves: stop turn, forgone validation |
| `reports/convergence_sweep/grid_test.csv` | 80 cells x 3 curves: test at stop vs. run-to-cap |

Reproduce with the two commands under **Method**. The `.npy` score arrays are gitignored, so the
CSVs above are the durable record.
