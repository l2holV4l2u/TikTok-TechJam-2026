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
