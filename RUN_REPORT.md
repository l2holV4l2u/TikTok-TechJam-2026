# Slot-ladder comparison

How many solution lineages the agent should advance per turn. 5 runs across 5 slot counts, 1 to 5 lineages per turn.

## Configuration

| setting | value |
|---|---|
| model | gpt-5.6-sol |
| data contract | train-only-v3 |
| epsilon | 0.002 |
| N (patience) | 3 |
| iteration cap | 50 |
| dataset | KuaiRand-Pure |
| runs compared | 5 |

Cost columns are omitted: the run records hold token counts, not prices. Pass `--price-in=` and `--price-out=` in USD per 1M tokens to include them.

## What each rung produced

Same agent, same dataset, same data contract, same convergence rule. The only difference is `--slots`. Where a slot count has repeats, the cell shows the mean with the observed range in brackets.

| slots | runs | validation | hidden test | vs baseline | wall clock | tokens | scripts | turns |
|---|---|---|---|---|---|---|---|---|
| **1** | 1 | 0.604523 | **0.598583** | +0.003983 | 14.4 min | 137,079 | 7 | 5 |
| **2** | 1 | 0.605608 | **0.598459** | +0.003859 | 16.6 min | 163,773 | 10 | 4 |
| **3** | 1 | 0.605565 | **0.598832** | +0.004232 | 25.2 min | 236,564 | 14 | 4 |
| **4** | 1 | 0.604810 | **0.598487** | +0.003887 | 26.4 min | 303,430 | 18 | 4 |
| **5** | 1 | 0.606296 | **0.598881** | +0.004281 | 36.6 min | 456,617 | 27 | 5 |

Baseline for the `vs baseline` column is the official baseline: validation 0.6016, hidden test 0.5946. `scripts` and `turns` are listed per run rather than averaged, because they are counts of work done and an average of them describes no run that was actually executed.

## Where the score stops moving

Marginal change from adding one more lineage. The noise floor here is the baseline's published 5-seed sigma (0.0008), used because no rung has been repeated -- repeat the ladder to measure it directly. A difference below it is not evidence of anything.

| step | test | delta | reading | tokens | delta | wall | delta |
|---|---|---|---|---|---|---|---|
| 1 slot | 0.598583 | - | starting point | 137,079 | - | 14.4 min | - |
| 1 -> 2 | 0.598459 | -0.000124 | within noise | 163,773 | +26,694 | 16.6 min | +2.2 |
| 2 -> 3 | 0.598832 | +0.000373 | within noise | 236,564 | +72,791 | 25.2 min | +8.6 |
| 3 -> 4 | 0.598487 | -0.000345 | within noise | 303,430 | +66,866 | 26.4 min | +1.2 |
| 4 -> 5 | 0.598881 | +0.000394 | within noise | 456,617 | +153,187 | 36.6 min | +10.2 |

**The curve is flat from 1 slot onward.** Rungs 1, 2, 3, 4, 5 all land within 0.00080 of the best result (0.598881 at 5 slots), so on this evidence they are indistinguishable on the hidden test. Across that flat span the spend still rises from 137,079 to 456,617 tokens (3.3x) and from 14.4 to 36.6 minutes.

The honest reading is that everything past 1 slot is bought and not delivered: cost scales close to linearly in the slot count while the scored result does not move outside noise.

## Validation vs test: is the extra search buying score or noise?

`gap` is validation minus hidden test for the submitted iteration. A gap that widens with the slot count, while test does not improve, means the added lineages are winning on validation noise rather than finding transferable signal.

| slots | candidates compared | validation | hidden test | gap | gap vs 1 slot |
|---|---|---|---|---|---|
| 1 | 92 | 0.604523 | 0.598583 | 0.005941 | +0.000000 |
| 2 | 224 | 0.605608 | 0.598459 | 0.007149 | +0.001209 |
| 3 | 249 | 0.605565 | 0.598832 | 0.006733 | +0.000792 |
| 4 | 330 | 0.604810 | 0.598487 | 0.006323 | +0.000382 |
| 5 | 578 | 0.606296 | 0.598881 | 0.007415 | +0.001474 |

Rank correlation between candidates compared and the validation-test gap: **+0.60** over 5 rungs. Positive means more comparison produces more validation inflation. At n=5 this is suggestive at best.

The gap spans 0.001474 across the ladder, against a noise floor of 0.00080. That is larger than the floor, so the differences in generalisation are worth taking seriously.

## Where the gain happens inside a run

Share of each run's total validation gain reached by iteration k. The convergence rule stops these runs at turn 4-5; this shows how much was already banked by then.

| slots | run | improve iters | total gain | by #1 | by #2 | by #4 | last gain at |
|---|---|---|---|---|---|---|---|
| 1 | `s1_r1` | 4 | +0.00068 | 43% | 97% | - | #4 of 4 |
| 2 | `s2_r1` | 8 | +0.00179 | 66% | 66% | 87% | #7 of 8 |
| 3 | `s3_r1` | 12 | +0.00358 | 10% | 64% | 79% | #11 of 12 |
| 4 | `s4_r1` | 16 | +0.00136 | 1% | 2% | 90% | #7 of 16 |
| 5 | `s5_r1` | 20 | +0.00343 | 0% | 37% | 51% | #18 of 20 |

`last gain at` is the final iteration that set a new best. A run whose last gain is well before its final iteration spent the remainder finding nothing.

## Cost of the result

| slots | tokens | wall | test | tokens per 0.0001 of test delta | verdict |
|---|---|---|---|---|---|
| 1 | 137,079 | 14.4 min | 0.598583 | 3,442 | **best value** |
| 2 | 163,773 | 16.6 min | 0.598459 | 4,244 | ties the best for 1.2x the cost |
| 3 | 236,564 | 25.2 min | 0.598832 | 5,590 | ties the best for 1.7x the cost |
| 4 | 303,430 | 26.4 min | 0.598487 | 7,806 | ties the best for 2.2x the cost |
| 5 | 456,617 | 36.6 min | 0.598881 | 10,667 | ties the best for 3.3x the cost |

The `tokens per 0.0001` column divides the whole run's spend by the test gain over the official baseline. It is a blunt figure -- it charges the baseline reproduction and EDA to the improvement -- but it is the number that decides whether a rung is worth running.

## What the spend bought

| slots | tokens | tokens/slot | wall clock | script time | candidates compared | candidates/1k tokens |
|---|---|---|---|---|---|---|
| 1 | 137,079 | 137,079 | 14.4 min | 3.6 min | 92 | 0.67 |
| 2 | 163,773 | 81,886 | 16.6 min | 5.5 min | 224 | 1.37 |
| 3 | 236,564 | 78,855 | 25.2 min | 15.8 min | 249 | 1.05 |
| 4 | 303,430 | 75,858 | 26.4 min | 9.7 min | 330 | 1.09 |
| 5 | 456,617 | 91,323 | 36.6 min | 10.3 min | 578 | 1.27 |

`candidates compared` counts the alternatives evaluated INSIDE scripts, which is where most of the search happens: a script may build and compare a dozen models for one iteration of budget. It rises with the slot count, so the extra lineages are doing real work -- the question the table above answers is whether that work reaches the score.

## Where the wall-clock time goes

`script` is time executing the agent's code; the remainder is dominated by LLM latency. The split matters operationally: a run that is mostly latency parallelises almost free, while one that is mostly compute contends for cores.

| slots | wall | script time | script share | remainder (LLM etc.) |
|---|---|---|---|---|
| 1 | 14.4 min | 3.6 min | 25% | 10.8 min |
| 2 | 16.6 min | 5.5 min | 33% | 11.1 min |
| 3 | 25.2 min | 15.8 min | 63% | 9.3 min |
| 4 | 26.4 min | 9.7 min | 37% | 16.7 min |
| 5 | 36.6 min | 10.3 min | 28% | 26.2 min |

Compute share ranges from 25% (1 slots) to 63% (3 slots) -- it does not rise monotonically with the slot count, because what a script costs depends on the model the agent chose to write, not only on how many scripts run. Slots already execute concurrently inside a run (`agent/loop.py:737`) and no thread counts are pinned, so running several ladder rungs at once would contend for cores, and would contend by different amounts per rung. That is why this ladder is run sequentially: the wall-clock and token columns above are the comparison, and overlapping the runs would corrupt them.

## Did the extra lineages disagree?

Within-user rank correlation between the slots' own pre-blend models. Above ~0.95 the extra lineages rank validation identically and return one lineage's information for n lineages' spend.

| slots | run | per-turn mean | mean across turns | last turn (`mean_slot_correlation`) | reading |
|---|---|---|---|---|---|
| 2 | `s2_r1` | 0.932 -> 0.759 -> 0.756 -> 0.453 | **0.725** | 0.453 | lineages genuinely differ |
| 3 | `s3_r1` | 0.831 -> 0.780 -> 0.596 -> 0.720 | **0.732** | 0.720 | lineages genuinely differ |
| 4 | `s4_r1` | 0.767 -> 0.784 -> 0.803 -> 0.756 | **0.778** | 0.756 | lineages genuinely differ |
| 5 | `s5_r1` | 0.862 -> 0.741 -> 0.670 -> 0.535 -> 0.459 | **0.653** | 0.459 | lineages genuinely differ |

The last-turn column is what `run_meta.json` records; it is a single turn and swings widely, so the verdict above is taken from the average across turns.

## Reliability across the ladder

| slots | run | scripts | crashes | integrity rejections | manual interventions | stop reason |
|---|---|---|---|---|---|---|
| 1 | `s1_r1` | 7 | 1 | 0 | 0 | `converged` |
| 2 | `s2_r1` | 10 | 0 | 0 | 0 | `converged` |
| 3 | `s3_r1` | 14 | 0 | 0 | 0 | `converged` |
| 4 | `s4_r1` | 18 | 0 | 0 | 0 | `converged` |
| 5 | `s5_r1` | 27 | 5 | 0 | 0 | `converged` |

`crashes` counts scripts that failed to produce a scored result, separately from integrity rejections, which are results the critic refused. Both are handled in the loop; neither reaches a human.

## Score curve within each run

The run's best validation score after each scored improve iteration. This is the curve the convergence rule is measured against.

- **1 slot(s)** (`s1_r1`): 0.6038 -> 0.6041 -> 0.6045 -> 0.6045
- **2 slot(s)** (`s2_r1`): 0.6038 -> 0.6050 -> 0.6050 -> 0.6052 -> 0.6054 -> 0.6054 -> 0.6056 -> 0.6056
- **3 slot(s)** (`s3_r1`): 0.6020 -> 0.6023 -> 0.6043 -> 0.6048 -> 0.6048 -> 0.6049 -> 0.6051 -> 0.6052 -> 0.6052 -> 0.6052 -> 0.6056 -> 0.6056
- **4 slot(s)** (`s4_r1`): 0.6035 -> 0.6035 -> 0.6035 -> 0.6044 -> 0.6047 -> 0.6047 -> 0.6048 -> 0.6048 -> 0.6048 -> 0.6048 -> 0.6048 -> 0.6048 -> 0.6048 -> 0.6048 -> 0.6048 -> 0.6048
- **5 slot(s)** (`s5_r1`): 0.6029 -> 0.6029 -> 0.6041 -> 0.6042 -> 0.6046 -> 0.6046 -> 0.6050 -> 0.6050 -> 0.6050 -> 0.6062 -> 0.6062 -> 0.6062 -> 0.6062 -> 0.6062 -> 0.6062 -> 0.6062 -> 0.6062 -> 0.6063 -> 0.6063 -> 0.6063

## Caveat

Each rung is a single run. Run-to-run spread on this benchmark is roughly 0.0008 on the scored metric -- the same size as most of the differences above -- so this ladder shows the SHAPE of the cost/score curve rather than a precise value for any rung, and any plateau claim rests on several rungs agreeing rather than on any single pair. Repeating each rung would replace that assumed noise figure with a measured one.
