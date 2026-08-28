# Why every method lands within ±0.005 of the baseline

Reproduce: `python -m research.ceiling_probe` (~150s, CPU).

## The puzzle

Across this project — human hand-tuning and the agent's own runs — roughly fifteen distinct
approaches all landed within about ±0.005 of the official baseline: factorization machines,
DeepFM, DCN, NFM, LightGBM pointwise, LambdaMART, pairwise BPR, listwise softmax, target
encodings, multi-task heads, and rank blends of several of those. Deeper models did not help.
More features did not help. That pattern needs an explanation, because the three candidate
explanations imply completely different strategies:

1. models too small → build bigger ones
2. features carry little signal → engineer better ones
3. signal exists but does not transfer across the date split → attack drift, not capacity

## The measurement

Fit a deliberately over-powered LightGBM (255 leaves, 250 rounds, `min_data_in_leaf=5`) on all
37 categorical fields, and score it *both* in-sample and on validation. In-sample performance
upper-bounds what the feature set can express when memorisation is allowed.

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| high-capacity, **in-sample** (train) | 0.9456 | 0.9034 | **0.9245** |
| same model, **validation** | 0.6469 | 0.5266 | 0.5868 |
| official baseline, validation | 0.6674 | 0.5357 | 0.6016 |
| oracle (true labels), validation | 1.0000 | 0.6968 | 0.8484 |

## The finding

**Explanation 3.** The features can separate `long_view` almost perfectly on the training
window — 0.9245 primary, GAUC 0.9456 — so neither capacity nor feature expressiveness is the
limit. Essentially none of that transfers: the same model scores 0.5868 on validation, which is
*worse than the baseline it dwarfs in capacity*. The generalisation gap is **0.3377**.

The splits are defined by date: train is 9–21 Apr 2022, validation the following week, test the
ten days after that. The agent's own EDA independently measured how far the distribution moves
across that boundary — the share of users with zero positives goes 5.1% → 30.3%, and the median
rows per user 59 → 7.

## Consequences

- **A small, heavily-regularised model is near-optimal here.** The official baseline's k=16 FM
  over five fields is not a weak starting point that better architectures should beat; it is
  close to the right amount of capacity for a signal this non-stationary. That is why bigger
  models reliably lose.
- **The realistic ceiling is roughly 0.60–0.61 test primary**, not the 0.8645 oracle. The oracle
  assumes knowledge of the labels; it bounds the metric, not the achievable score.
- **Deltas in this competition will be small.** Scoring is `delta = agent − baseline` on the
  hidden test set, and the gap between "did nothing" and "did everything we know how to do" is
  a few thousandths. This is a benchmark where +0.004 is a real result.
- **The methods with a mechanism here target drift, not capacity**: recency weighting over the
  13 training days, time-based validation, features that are stationary across windows rather
  than identity-specific. This is why `Split.date` is now exposed to the agent — withholding it
  silently ruled out that entire family.

## Caveat

This is one probe with one model family, and in-sample performance is an upper bound on
expressiveness, not a proof about every possible model. It does not rule out that a method
specifically designed for temporal drift extracts more than we have. It does rule out "the
models were too small", which was the intuition we started from.

## Selection is not the bottleneck

A natural response to a noisy one-week validation split is to build a better selector -- for
example scoring candidates on a train-only chronological fold as well. We measured the ceiling
on that idea before building it, by asking what a *perfect* selector would have been worth: for
each run, compare the test score of the validation-best iteration against the test score of the
best iteration available in that run.

| run | validation-best | its test | best test in run | cost of selecting on validation |
|---|---|---|---|---|
| r33 | 0.6044 | 0.5981 | 0.5981 | 0.0000 |
| r34 | 0.6043 | 0.5979 | 0.5984 | 0.0005 |
| r35 | 0.6049 | 0.5985 | 0.5988 | 0.0003 |
| r36 | 0.6037 | 0.5982 | 0.5982 | 0.0000 |
| r37 | 0.6037 | 0.5987 | 0.5987 | 0.0000 |
| r40 | 0.6036 | 0.5976 | 0.5976 | 0.0000 |

Validation picks the test-best iteration in **4 of 6 eligible runs**, and an oracle with access
to the hidden test set would gain less than **+0.0002 primary on average**. r39 and r41 are
excluded from this analysis because their item statistics aggregate the evaluation month.

The problem is not choosing among the iterations a run produces. It is that the best iteration
a run produces is not better. Effort spent on the selection signal is therefore capped at
roughly a fiftieth of the effort spent on what gets proposed.

One thing this does **not** rule out: a temporal fold could still change *which experiments the
agent proposes*, by putting drift evidence into `FINDINGS` and the belief set. That is a
different mechanism from selection and this measurement says nothing about it.
