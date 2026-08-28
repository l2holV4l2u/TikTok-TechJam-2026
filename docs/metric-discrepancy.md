# Metric question — RESOLVED (starter kit received 26 Aug)

## Root cause: the label

`kuairand-starter-kit/data.py` line 4:

    LABEL = 'long_view'

The problem statement prose says "click = positive" in three places. **The prose is wrong.**
The scored label is `long_view` (0/1), not `is_click`. We had been evaluating a 46%-positive
label against numbers computed on a 33%-positive one, which is why our nDCG@5 ran ~0.11 high
on every rung.

## What was NOT the cause
Our earlier hypothesis -- that the organizers rank against a large candidate set, because
Recall@50 is meaningless at a median of 4 impressions per user -- was **wrong**. Their
`evaluate.py` docstring states it plainly: `不做全库检索` ("no full-catalog retrieval"),
within-user ranking over logged impressions. Our formula and protocol were correct all along.

The Recall@50 / NDCG@10 mention in §2.3 and §2.6 of the problem statement is simply stale text
left over from the AliCCP version. GAUC / nDCG@5 is authoritative.

## Verification
`pipeline/evaluate.py` vs the official `evaluate.py`, identical predictions, validation split:

| rung | metric | ours | official | diff |
|---|---|---|---|---|
| random | GAUC | 0.502845 | 0.502845 | 8.7e-15 |
| random | nDCG@5 | 0.468489 | 0.468489 | " |
| item-pop | GAUC | 0.638707 | 0.638707 | 1.7e-14 |
| item-pop | nDCG@5 | 0.522825 | 0.522825 | " |

Bit-identical. Ours is vectorized and ~7x faster (0.15s vs 1.0s), which matters because the
agent evaluates on every iteration. Keep ours in the loop; the official file is the reference.

## Official numbers (baseline_scores.json), validation
| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random | 0.4993 | 0.4675 | 0.4834 |
| item popularity (per-video positive RATE, not count) | 0.6387 | 0.5227 | 0.5807 |
| **FM baseline** | 0.6674 | 0.5357 | **0.6016** |
| oracle ceiling | 1.0 | 0.6968 | **0.8484** |

Their `baseline.py` reproduces locally: valid primary 0.6015, test 0.5953, ~70s single core.

## The number that matters
Headroom is **0.6016 -> 0.8484**, not 0.6016 -> 1.0. nDCG@5 cannot reach 1.0 because 27.1% of
test users are all-negative and score 0 for any model; another 9.2% are all-positive. GAUC is
computed on only the 63.7% of users that are discriminative. Report progress as a fraction of
that gap.

## Also settled
- Convergence rule: epsilon = 0.002, N = 3 (confirmed in baseline_scores.json).
- Official FM config: k=16, lr=0.001, batch 8192, max_epochs 40, patience 4,
  fields [user_id, video_id, author_id, tab, dur_bucket].
- We hold the public test labels, so we can self-score the test window before submitting.
