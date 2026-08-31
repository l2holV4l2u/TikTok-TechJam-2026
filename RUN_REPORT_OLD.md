# Slot-ladder comparison

5 runs, 1 to 5 lineages per turn. Model: gpt-5.6-sol. Data contract: train-plus-valid-v2.

Cost columns are omitted: the run records hold token counts, not prices. Pass `--price-in=` and `--price-out=` in USD per 1M tokens to include them.
## What each rung produced

Every run below is the same agent, same dataset, same data contract and the same convergence rule. The only difference is `--slots`.

| slots | run | validation | hidden test | vs baseline | wall clock | tokens | scripts | turns |
|---|---|---|---|---|---|---|---|---|
| **1** | `r85_1slot` | 0.604424 | **0.598788** | +0.004188 | 10.8 min | 128,159 | 8 | 5 |
| **2** | `r86_2slot` | 0.604452 | **0.599473** | +0.004873 | 15.4 min | 179,572 | 11 | 4 |
| **3** | `r87_3slot` | 0.605713 | **0.600410** | +0.005810 | 22.0 min | 246,618 | 14 | 4 |
| **4** | `r88_4slot` | 0.605090 | **0.600194** | +0.005594 | 27.4 min | 319,616 | 18 | 4 |
| **5** | `r89_5slot` | 0.605716 | **0.600256** | +0.005656 | 32.9 min | 429,394 | 22 | 4 |

Baseline for the `vs baseline` column is the official baseline: validation 0.6016, hidden test 0.5946.

## Where the score stops moving

Marginal change from adding one more lineage. A benchmark difference below 0.0008 is inside the baseline's own 5-seed noise and is not evidence of anything; below 0.002 it is under the organizers' convergence epsilon.

| step | test | delta | reading | tokens | delta | wall | delta |
|---|---|---|---|---|---|---|---|
| 1 slot | 0.598788 | - | starting point | 128,159 | - | 10.8 min | - |
| 1 -> 2 | 0.599473 | +0.000685 | within seed noise | 179,572 | +51,413 | 15.4 min | +4.5 |
| 2 -> 3 | 0.600410 | +0.000937 | under epsilon | 246,618 | +67,046 | 22.0 min | +6.7 |
| 3 -> 4 | 0.600194 | -0.000216 | within seed noise | 319,616 | +72,998 | 27.4 min | +5.4 |
| 4 -> 5 | 0.600256 | +0.000063 | within seed noise | 429,394 | +109,778 | 32.9 min | +5.5 |

**The curve flattens at 3 slots.** Rungs 3, 4, 5 all land within 0.0008 of the best result (0.600410 at 3 slots), so on this evidence they are indistinguishable on the hidden test. Across that flat span the spend still rises from 246,618 to 429,394 tokens (1.7x) and from 22.0 to 32.9 minutes.

The honest reading is that everything past 3 slots is bought and not delivered: cost scales close to linearly in the slot count while the scored result does not move outside noise.

## What the spend bought

| slots | tokens | tokens/slot | wall clock | script time | candidates compared | candidates/1k tokens |
|---|---|---|---|---|---|---|
| 1 | 128,159 | 128,159 | 10.8 min | 2.7 min | 24 | 0.19 |
| 2 | 179,572 | 89,786 | 15.4 min | 5.2 min | 41 | 0.23 |
| 3 | 246,618 | 82,206 | 22.0 min | 9.9 min | 282 | 1.14 |
| 4 | 319,616 | 79,904 | 27.4 min | 14.6 min | 257 | 0.80 |
| 5 | 429,394 | 85,878 | 32.9 min | 17.7 min | 696 | 1.62 |

`candidates compared` counts the alternatives evaluated INSIDE scripts, which is where most of the search happens: a script may build and compare a dozen models for one iteration of budget. It rises with the slot count, so the extra lineages are doing real work -- the question the table above answers is whether that work reaches the score.

## Did the extra lineages disagree?

Within-user rank correlation between the slots' own pre-blend models. Above ~0.95 the extra lineages rank validation identically and return one lineage's information for n lineages' spend.

| slots | per-turn mean | mean across turns | last turn (`mean_slot_correlation`) | reading |
|---|---|---|---|---|
| 2 | 0.842 -> 0.738 -> 0.924 -> 0.293 | **0.699** | 0.293 | lineages genuinely differ |
| 3 | 0.914 -> 0.635 -> 0.764 -> 0.741 | **0.763** | 0.741 | lineages genuinely differ |
| 4 | 0.900 -> 0.822 -> 0.182 -> 0.762 | **0.667** | 0.762 | lineages genuinely differ |
| 5 | 0.617 -> 0.776 -> 0.478 -> 0.343 | **0.554** | 0.343 | lineages genuinely differ |

The last-turn column is what `run_meta.json` records; it is a single turn and swings widely, so the verdict above is taken from the average across turns.

## Reliability across the ladder

| slots | scripts | crashes | integrity rejections | manual interventions | stop reason |
|---|---|---|---|---|---|
| 1 | 8 | 2 | 0 | 0 | `converged` |
| 2 | 11 | 0 | 1 | 0 | `converged` |
| 3 | 14 | 0 | 1 | 0 | `converged` |
| 4 | 18 | 0 | 1 | 0 | `converged` |
| 5 | 22 | 0 | 1 | 0 | `converged` |

`crashes` counts scripts that failed to produce a scored result, separately from integrity rejections, which are results the critic refused. Both are handled in the loop; neither reaches a human.

## Score curve within each run

The run's best validation score after each scored improve iteration. One curve per run, which is what the convergence rule is measured against.

- **1 slot(s)** (`r85_1slot`): 0.6030 -> 0.6035 -> 0.6042 -> 0.6044
- **2 slot(s)** (`r86_2slot`): 0.6036 -> 0.6038 -> 0.6042 -> 0.6042 -> 0.6043 -> 0.6044 -> 0.6044 -> 0.6045
- **3 slot(s)** (`r87_3slot`): 0.6042 -> 0.6042 -> 0.6043 -> 0.6043 -> 0.6049 -> 0.6052 -> 0.6052 -> 0.6055 -> 0.6055 -> 0.6057 -> 0.6057 -> 0.6057
- **4 slot(s)** (`r88_4slot`): 0.6041 -> 0.6041 -> 0.6045 -> 0.6045 -> 0.6046 -> 0.6046 -> 0.6047 -> 0.6048 -> 0.6048 -> 0.6048 -> 0.6048 -> 0.6048 -> 0.6048 -> 0.6048 -> 0.6048 -> 0.6051
- **5 slot(s)** (`r89_5slot`): 0.6037 -> 0.6040 -> 0.6041 -> 0.6042 -> 0.6042 -> 0.6043 -> 0.6044 -> 0.6044 -> 0.6044 -> 0.6050 -> 0.6051 -> 0.6054 -> 0.6055 -> 0.6055 -> 0.6055 -> 0.6057 -> 0.6057 -> 0.6057 -> 0.6057 -> 0.6057

## Caveat

Each rung is a single run. Run-to-run spread on this benchmark is roughly 0.0008 on the scored metric, which is the same size as most of the differences above, so the ladder shows the SHAPE of the cost/score curve rather than a precise value for any rung. The plateau claim rests on several rungs agreeing, not on any single pair.
