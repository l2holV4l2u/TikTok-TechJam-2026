# Bugs / design gaps found during review

Running log. One `##` section per finding, newest at bottom.

## Tree backtrack is dead code for scored misses

`agent/tree.py`'s docstring claims node retirement gives the search backtracking a linear walk
can't do: a node that keeps producing non-improving children is retired, and `select()` falls
back to the next-best live node.

For genuine scored misses, this never fires. `Tree.max_misses` and `run_loop`'s `patience` are
both `3`, checked against the same `epsilon` (0.002), on the same reigning-leader score. A
node's 3rd non-improving *scored* child is, by construction, also the run's 3rd consecutive
non-improvement of the global best -- `agent/loop.py:converged()` is checked at the *top* of
the next iteration, before `tree.select()` runs for it. So the run ends one step before any
fallback selection could take effect.

**The only path where backtrack actually executes:** crashes. A crashed/timed-out iteration
hits `continue` in `agent/loop.py` before `stale` is touched, so it never counts against
`patience`, only against the node's separate crash counter (`Tree.record_failure`, threshold
`2 * max_misses = 6`). A node can die from repeated crashes without spending any of the run's
convergence budget, and only then does `select()` actually return a different node.

**Fix, if the general-purpose backtrack is wanted:** set `max_misses < patience` so a node
retires before the run converges. As shipped, the mechanism as documented only rescues
crash-heavy branches, not genuinely-tried-and-failed ones.
**Resolved 29 Aug 2026.** `--max-misses` and `Tree.max_misses` are now `2` against a
`patience` of `3`, so a node retires one iteration before the run can converge and `select()`
reaches the fallback. Verified by `python -m agent.tree`.

## Cross-iteration search cannot pay for itself under this convergence rule

The fix above makes backtracking reachable, but it does not make it valuable. With
eps = 0.002 and N = 3 fixed by the organizers, a run gets roughly three to six iterations.
Tree search, node retirement and backtracking all need more samples than that to earn their
complexity; they are the right structure for a budget an order of magnitude larger.

What the runs actually show is that the search already lives somewhere else. The convergence
rule charges per ITERATION, not per model, so comparing candidates inside one script is free
and comparing them across scripts costs a life:

| run | iterations | candidates compared inside them |
|---|---|---|
| r59 | 6 | 161 |
| r70 | 6 | 39 |

A 4x spread means that search was incidental, not designed. The brief now states the
economics explicitly. The cross-iteration tree remains as an audit trail and as the mechanism
that survives crash-heavy branches, which is what it is actually good for here.

## Parent code truncation silently corrupts the reference script

`agent/proposer.py` sends the selected parent node's source in full, hard-truncated at
`MAX_CODE_CHARS = 14000` (`parent.code[:MAX_CODE_CHARS]`, a plain slice with no boundary
awareness). Measured across `runs/`: 35 of 442 generated scripts (8%) exceed 14,000 characters.

When one of those is selected as parent, the proposer is told "here is the parent script, make
a targeted edit" and shown a slice that ends mid-statement. Confirmed on
`runs/r39/scripts/iter_6.py` (22,889 chars): the cut lands inside a numpy slice expression,
`1:])`, mid-token. The model cannot see whatever runs after that point -- which may include the
`scores_test.npy` save, the leakage-safe aggregate, or the metrics line -- while being asked to
edit it as if it saw the whole thing.

Later, more-evolved scripts (which accumulate feature engineering across iterations) are the
ones most likely to cross the cap, so this disproportionately hits exactly the highest-value
parents. Measured prompt size overall already reaches ~12,000 tokens uncapped
(`agent/proposer.py:27-29`), so this is one symptom of a broader context-budget problem, not
an isolated one.

**Not yet fixed.** A smarter truncation (cut at a line/def boundary, or summarize the tail
instead of dropping it) would at least make the loss legible instead of silent.

## Runs converge at iteration 6 of 50 because only sweeps clear epsilon

The organizers allow 50 iterations or 6 h, whichever comes first. Measured on r70-r74, every
run stopped at iteration 6 having used 16 minutes -- 12% of the iteration cap and 4.5% of the
wall-clock ceiling. The stop was always `converged`, never the cap.

Per-iteration validation gain, all five runs:

| iteration kind | gain | clears eps = 0.002 |
|---|---|---|
| family sweep (r74, r71) | +0.00305, +0.00266 | yes |
| sweep that swept hyperparameters instead (r70, r72, r73) | +0.00127, +0.00046, +0.00038 | no |
| refine / broaden (15 of them) | +0.00000 .. +0.00042 | never |

No refine iteration has ever cleared epsilon. Three sub-epsilon iterations is exactly the
convergence condition, so the run ends three iterations after the last sweep -- every time.
The rule was not detecting a score ceiling; it was detecting that the harness stopped
sweeping after iteration 2.

Two causes, both now fixed:

1. `agent/loop.py` only ever entered `sweep` mode on the first improve iteration
   (`if phase == "improve" and not best_curve`). It now also sweeps on the first miss, and
   falls through to `broaden` on a second so the search can still leave the model stage.
2. `_SWEEP` described "structurally different model families" without naming any, and three
   of five runs read that as breadth over blend weights, feature subsets or fusion scales.
   The prompt now lists the families outright, states the negative case ("the same model at
   another width, depth, learning rate, seed, epoch count or feature subset is NOT a
   different family"), and carries the run's already-tried list so each sweep picks new ones.

**This also revises the entry above.** "Cross-iteration search cannot pay for itself under
this convergence rule" was argued from "a run gets roughly three to six iterations". That
premise was self-inflicted, not imposed by the rule. If sweeping keeps clearing epsilon the
run reaches the budget it was actually given, and node retirement and backtracking have the
sample count they need. The conclusion stands only for a run that stops sweeping.

## The convergence tail was spent wandering instead of exploiting

Follow-on from the entry above. Making stalls re-sweep fixes the wrong half of the schedule
on its own: a run must still spend `patience` = 3 sub-epsilon iterations before it is allowed
to stop, so that tail is spent either way. Across r70-r74 it went on unfocused refinement and
banked about 0.0005 in total.

The mode ladder is now explore-then-exploit, which is the standard schedule for a search
whose two loops are priced differently:

| stale | mode | what it searches |
|---|---|---|
| 0 | refine | the direction that just gained |
| 1 | sweep | model families not yet tried this run |
| >= 2 | tune | the best architecture's configuration space |

The two loops: an ITERATION costs one of three convergence lives and there are 5-30 of them;
a model inside a script costs only wall clock against the per-script timeout and is free
under the convergence rule. Every search strategy therefore belongs in the inner loop, and
the outer loop only picks which inner search to run.

`_TUNE_INSTRUCTION` asks for successive halving rather than a grid, because a grid does not
fit the timeout: ~16 configurations at one epoch, the best ~6 at a medium budget, the best ~2
at full. It also states the two failure modes -- taking the max over many configurations on
124,909 validation rows overfits that split, so a configuration good across its neighbours
beats an isolated peak; and a gain under 0.0008 is inside the baseline's own 5-seed std, so
the honest move is to return the incumbent unchanged.

**Correcting an earlier claim in this file and in the run reports.** "Tuning recovers less
than the gap between families" was argued from two measurements that do not cover this: the
0.0002 figure is seed-to-seed variation, and the 0.0000-0.0042 range is single-edit refine
iterations. Neither is a configuration sweep, which the harness had never run. The family gap
being larger than seed noise is still measured and still true; the claim that tuning cannot
pay was not.

## Refine and tune bought validation points that did not exist on the test set

r77 and r78 were run with `--patience 999` to disable the convergence stop, and every
iteration's saved `scores_valid.npy` / `scores_test.npy` were scored on both splits. The
hidden test set is what the ranking uses; validation only selects which checkpoint is
submitted. Scoring both is what exposed this -- the harness had only ever reported
validation.

| move | validation | hidden test |
|---|---|---|
| family sweeps 1-3 (boosted tree, cross network, multi-task) | +0.00168 | **+0.00224** |
| family sweeps 4-8 (ten further architectures) | +0.00009 | +0.00007 |
| refine: 9 -> 37 categorical fields (r77 #4) | +0.00021 | **-0.00001** |
| tune: successive halving (r76 #5, r77 #6) | +0.00018, refused | never shipped |

Two conclusions, and the second is the damaging one:

1. **Three complementary families capture the whole gain.** Seventeen architectures were
   tried across all seven groups -- FM, FFM, DeepFM, xDeepFM, DCN, DCN-V2, AutoInt, PNN, AFM,
   FiBiNET, NFM, Wide & Deep, DIN, GRU, SASRec, LightGBM binary and LambdaRank, MMoE, SVD,
   empirical Bayes. Everything after the third was inside seed noise. Once the blend holds a
   boosted tree, a cross network and a multi-task model they already disagree in enough
   different ways that a fourth architecture has no error mode left to correct.

2. **Refine and tune move validation without moving test.** The ranking is on test but the
   rules force the submission to be the validation-best checkpoint, so a mode that inflates
   validation actively steers the harness toward the worse model. r74 has better validation
   than r78 (0.604917 vs 0.604778) and worse test (0.599100 vs 0.599842) -- a 0.00074 gap,
   about 14% of the whole margin over the official baseline.

The leaked iteration r78 #9 is the same effect without ambiguity: reading `.aux` on the valid
split gained +0.00049 validation and lost 0.0001 test. The integrity check rejected it.

**Fixed.** Every improve iteration is now `sweep`. `refine`, `tune` and `broaden` remain
reachable through `--force-mode` for diagnostic runs, which is how these numbers were
measured. Best result on record is r78 iteration 7: test 0.599842, delta +0.00524.

## A chronological selection window does not beat full validation

Claimed earlier in this run's notes: selection is forced onto validation, validation and test
disagree, so choosing on the last days of validation -- the days nearest the test window --
should track test better. Test is 20220429-0508 and validation is 20220422-0428, so the last
validation days really are the closest thing to the evaluation window the agent may look at.

Measured with `research/selector_window.py` over all 49 iterations in r72-r81 that left both
`scores_valid.npy` and `scores_test.npy`:

| selector | Spearman vs test | test primary of its pick |
|---|---|---|
| full 7-day validation | 0.8659 | 0.599688 |
| last 4 days | 0.8397 | 0.599688 |
| last 3 days | 0.6637 | 0.599688 |
| last 2 days | 0.3273 | 0.599688 |

Full validation ranks iterations more like the hidden test than any truncation of it, and all
four selectors submit the same model. Averaged per run the last-2-day window is ahead by
0.00008, which is one iteration in r74 against the worst rank correlation of the four.

The premise was misapplied rather than wrong. The evidence for it was r74 beating r78 on
validation and losing on test -- a comparison BETWEEN runs, which the harness never makes. The
only choice it makes is between iterations within one run, and there full validation is a good
selector. Shrinking the window trades bias it does not have for variance it cannot afford.

Recorded so the axis is not re-proposed; `_SWEEP_INSTRUCTION` now marks it REFUTED.

## RETRACTED: "the portfolio diversity gate says no"

Everything below this heading, up to the next `##`, was written from a broken measurement and
its conclusion is wrong. It is kept rather than deleted because the retraction is the useful
part. The corrected result is in DEVPOST.md, "A portfolio of parallel lineages -- abandoned,
then reinstated when the measurement was fixed".

What the section below concluded: mean slot correlation > 0.95, so the lineages are copies of
one another and Phases 3-5 (archive, refill, portfolio blend) should be deleted.

What is actually true: measured on each slot's own pre-blend model, correlation runs 0.18-0.92
and *falls* across a run (r87: 0.914 -> 0.635 -> 0.764 -> 0.741). The lineages diverge. The
portfolio stays, and the 3-slot run r87 holds the best hidden-test score on record, +0.00581.

Two bugs produced the wrong number, and the section below only found the first:

1. The gate read post-blend arrays. When `retain_or_blend` discarded a slot's candidate it
   republished the incumbent, so two discarded slots held the identical array and correlated
   at 1.0000 by construction. Diagnosed correctly below.
2. **Not found below, and the causal one:** `slot.last_hypothesis` was assigned after scoring,
   so `_sibling_note` showed each slot what its siblings did LAST turn, and on turn 1 showed
   nothing. Every slot opened from an identical prompt. The slots were not only mismeasured as
   clones, they were being made into clones. Fixed on the `prem` branch by recording the
   hypothesis at propose time.

The specific error in reasoning: fix (1) was applied, the correlation stayed high, and that was
read as confirmation rather than as a second defect. One fix landing without moving the number
is weak evidence that the number is real -- it is equally consistent with a second bug behind
the first, which is what it was.

## The portfolio diversity gate: answered, and the answer is no

`docs/portfolio-plan.md` Phase 2 is a go/no-go gate. Three slots are only worth their cost if
they explore differently; if they converge on the same model the run is three expensive copies
of one agent. The rule set in HANDOFF.md §7: mean slot correlation < 0.90 proceed, 0.90-0.95
tighten `_SIBLINGS` and retry, > 0.95 stop and keep only Phases 0-1.

Two three-slot runs against the real dataset, r82 and r83:

| turn | r82 | r83 |
|---|---|---|
| 1 | 0.9431 | 0.9142 |
| 2 | (contaminated) | 0.9952 |
| 3 | (contaminated) | 0.9946 |
| 4 | (contaminated) | 0.9955 |

r82's turns 2-4 are not usable: see the bug below. r83 ran with the fix, and all four of its
readings are honest. The shape is the same in both runs -- slots start moderately diverse and
collapse by turn 2. r83's turn-2 pair 1-2 correlates at exactly 1.0000: two slots given
different hypotheses produced byte-identical predictions, because the best internal candidate
in each script was the same recency-weighted boosted tree.

Verdict: **> 0.95, stop.** Keep Phases 0-1 -- parallel turns still buy wall-clock, and r83
converged in 24.3 min against r82's 34.1 -- and abandon Phases 3-5. The archive, the refill
policy and `blend_portfolio` are built for a decorrelated ensemble pool that does not exist
here. Consistent with that, the portfolio blend declined on all eight turns across both runs
with "no member improved fold A", which is the correct outcome for a pool of near-copies.

This is the same bottleneck measured everywhere else on this benchmark: MMoE against plain
DeepFM at 0.9888, components at 0.94+, and blends that gained nothing. The constraint is not
search breadth.

### The gate was measuring the incumbent against itself

Found in r82, fixed before r83. `retain_or_blend` picks alpha over
{incumbent, 0.25, 0.5, 0.75, candidate} and overwrites `evaluator.last_scores` with whatever it
selected. At alpha 0.0 that array IS the incumbent. The slot then recorded it as its own
prediction, so:

- the correlation gate compared the incumbent with itself and reported a fake 1.0000 (r82
  turn 3: three different architectures -- DIN, spectral factorisation, DIN+pairwise -- all
  reporting the identical 0.60518);
- `_blend_portfolio_turn` was handed the incumbent as a candidate member, which can never
  improve a fold. Every "no member improved fold A" in r82 is partly this.

Slots now keep `raw_valid_scores` / `raw_test_scores`, what their own script produced.
Regression tests in `tests/test_selection.py`.

Still open: `retain_or_blend` also OVERWRITES `iter_N_out/scores_valid.npy` and
`scores_test.npy` on disk with the selected scores, so after a run the raw candidate is
unrecoverable from the artifacts. The in-memory fix repairs the gate and the blend; the audit
trail still loses what each script actually produced. It should be saved alongside, not over.

## Slot decorrelation does not predict the score either

The gate was worth fixing because it was measuring nothing. It does not follow that the number
it now produces is worth optimising, and r92 is the run that separates those two claims.

r92 ran the merged harness -- the `prem` branch's two portfolio fixes (pre-blend correlation,
sibling disclosure firing on turn 1) plus this branch's drift axis pointing recency weighting at
the main model's `sample_weight`. No earlier run had both. It produced the best decorrelation on
record, under the 0.90 "proceed" line on every turn:

    turn 1  0.7794      turn 3  0.6991
    turn 2  0.6687      turn 4  0.2757

and the worst hidden-test delta of any converged 3-slot run:

| run | mean correlation | valid | test | delta |
|---|---|---|---|---|
| r90 (gate broken, slots were near-clones) | ~0.99 | 0.605282 | 0.599904 | +0.00530 |
| r87 (gate fixed) | 0.741 | 0.605713 | 0.600410 | +0.00581 |
| r92 (gate fixed + drift axis) | 0.276 | 0.605454 | 0.599289 | +0.00469 |

Both directions are now measured. Slots that were literally copies of one another scored
+0.00530; slots that disagreed as much as anything we have produced scored +0.00469. Diversity
is not the axis the score moves along, and a run cannot be steered by pushing the correlation
down.

r92 also has the second-best validation of any run whose submission CSV exists and the worst
test of those three, which is the same validation/test disagreement recorded above -- and again
it is a comparison BETWEEN runs, which the harness never makes.

What this costs to know: three converged runs, ~90 minutes, ~830k tokens. What it buys is the
end of a line of investigation. The remaining spread across r79, r90 and r92 is 0.0006 with no
trend, which is seed noise; further runs of this kind are a coin flip rather than a method.
