# Slot-count ladder

How many solution lineages (`--slots`) a run should advance per turn. One report per run,
generated with `report_run.py`. Runs are named `s<slots>_r<repeat>`; repeats of the same
slot count sort together.

All runs: data contract `train-only-v3`, cross-run memory OFF, eps 0.002 and N 3 (the
`run_agent.py` defaults, selected in `reports/convergence_sweep/`). Run **sequentially** --
slots already execute concurrently inside a run (`agent/loop.py:737`) and no thread counts
are pinned, so overlapping runs would inflate the wall-clock and token columns unevenly.

## Repeat 1

| report | slots | turns | scripts | wall | tokens | slot corr | valid | test | test delta |
|---|---|---|---|---|---|---|---|---|---|
| [s1_r1](s1_r1.md) | 1 | 5 | 7 | 14.4 min | 137,079 | - | 0.604523 | 0.598583 | +0.0040 |
| [s2_r1](s2_r1.md) | 2 | 4 | 10 | 16.6 min | 163,773 | 0.453 | 0.605608 | 0.598459 | +0.0039 |
| [s3_r1](s3_r1.md) | 3 | 4 | 14 | 25.2 min | 236,564 | 0.720 | 0.605565 | 0.598832 | +0.0042 |
| [s4_r1](s4_r1.md) | 4 | 4 | 18 | 26.4 min | 303,430 | 0.756 | 0.604810 | 0.598487 | +0.0039 |
| [s5_r1](s5_r1.md) | 5 | 5 | 27 | 36.6 min | 456,617 | 0.459 | 0.606296 | 0.598881 | +0.0043 |

All five converged on their own; the 50-iteration cap never bound.

Cost scales, score does not. Tokens run 137,079 to 456,617 (3.3x) while test
spans 0.598459 to 0.598881 -- a range of 0.00042, roughly half the
~0.0008 seed noise. The ordering is scatter, not trend: s5 has the best test and s2 the worst.

Not yet interpretable at one repeat each:

- **Slot correlation is non-monotonic** (0.453, 0.720, 0.756, 0.459). If more slots meant
  more redundancy, s5 should be highest.
- **s1 and s5 took 5 turns, the middle three took 4.** Every run in the convergence sweep
  fired at exactly N+1.

For reference, the best compliant test score on record is 0.599731, from a **one-slot** run
(`runs/convergence_sweep/curve_2`) -- better than anything in this round, which is itself a
measure of how much of the spread above is run-to-run variance.

## Reproduce

```bash
./run_slot_ladder.sh 1          # s1_r1 .. s5_r1
./run_slot_ladder.sh 2          # the next repeat
./run_slot_ladder.sh 2 3 4      # only slots 3 and 4 of repeat 2
```

`reports/slot_ladder_old/` holds the earlier ladder (r85-r89), which ran under the
non-compliant `train-plus-valid-v2` contract and is superseded by these runs.
