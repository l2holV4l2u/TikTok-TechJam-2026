# Autonomous ML Research Agent — KuaiRand-Pure

## How the solution addresses the problem statement

We built an LLM-driven agent that runs the full MLE iteration loop on KuaiRand-Pure without a
human in it: it reads the problem, proposes a hypothesis grounded in published work, writes the
code, trains, evaluates against the official metric, records the outcome, and decides what to try
next — until the validation score converges under the organizers' rule (epsilon = 0.002, N = 3)
or a compute budget is spent.

The agent is the product; the recommender is the sandbox it works in.

### The loop

```
propose (LLM + retrieved papers)
  -> write script -> execute sandboxed with timeout
  -> parse METRICS -> judge against incumbent
  -> append to ledger -> recover on failure
  -> repeat until converged or out of budget
```

Every iteration appends one immutable record: hypothesis, full code, metrics, wall/compute
seconds, input and output tokens, status, and any error plus how it was handled.

### Three design decisions that follow from the rubric

**Cost is scored, so the LLM stays out of the training loop.** It is called only to propose and
to react to a failure. Everything else is deterministic Python. Prompt size is bounded — history
is summarised to one line per iteration, capped at five — so token usage stays flat as a run
grows rather than compounding. A 17-iteration run cost 46.5K input / 13.7K output tokens.

**Autonomy is measured by intervention count, so failures never escalate.** A crashed script's
traceback *and its source* are fed back so the model fixes the failing line instead of rewriting
from scratch. Timeouts are reported distinctly ("too slow, not wrong") because rerunning a slow
approach unchanged just times out again. Every completed run reports 0 manual interventions.

**Ideas are retired on two separate grounds.** Crash-based blacklisting alone is not enough: an
idea that runs cleanly but keeps losing is never excluded by it, and in one run 8 of 17
iterations went to a single losing idea for exactly that reason. The agent now retires an idea
family after 3 crashes *or* 3 sub-incumbent scores. Retirement keys on the named method, so
rewording "Implement a LambdaRank loss" as "Implementing a LambdaRank objective" does not evade
it.

### Robustness

Beyond retry/blacklist: a circuit breaker halts the run with `environment_broken` after five
consecutive instant, output-less failures. We added this after a real incident — a torn-down
parent process left the runner unable to spawn children, and 32 iterations "failed" in 0 seconds
each, silently shredding the budget and discarding a genuinely good result along the way.

### What we learned about steering an agent

Most of this project's engineering went into the agent's *brief*, not its loop. Every recurring
failure traced back to something the brief failed to say, and each fix is visible in the failure
rate: ~50% -> 35% -> 27% -> 0%.

- It called `.cuda()` repeatedly because the brief claimed a GPU was available when torch was
  CPU-only. We had written a false statement into the prompt.
- It kept getting tensor shapes wrong until the skeleton spelled out what an embedding block
  flattens to, and what a numeric feature has to do differently.
- It crashed on `IndexError` five times building per-video tables, because our own helper
  applied field offsets and nothing said the raw ids were needed for lookups.
- It spent 8 of 17 iterations on one losing idea, because retirement only triggered on crashes.
  An idea that runs cleanly and scores badly was never excluded.
- Rewording "Implement a LambdaRank loss" as "Implementing a LambdaRank objective" evaded
  retirement entirely, because similarity was measured with Jaccard over the whole sentence and
  the model restates ideas at very different lengths.

The subtlest one: after we added a section telling the agent that ensembling was the proven
direction, it still proposed only single models, six times in a row. The retrieved-papers block
directly beneath that section was seeded with query terms like "click", "ranking", "ndcg" -- so
it surfaced only single-model papers, every call. Two halves of the same prompt disagreed, and
the retrieval half won. Adding ensemble terms to the retrieval query fixed it.

None of these were visible from the score. They were only visible by reading what the agent
actually chose to do, which is what the per-iteration ledger exists for.

## Results

Official baseline (organizer-provided FM, k=16): validation primary 0.6016, hidden test 0.5946.

Our submission: **test primary 0.5986, delta +0.0040** (GAUC 0.6656, nDCG@5 0.5316).

Critically, this was produced by the agent itself, not by us. Its hypothesis was *"reproduce
and extend the proven blend by adding a train-only video pos-rate numeric feature; FM +
LightGBM pointwise + lambdarank blending on ranks"* — it took the ensembling principle from its
brief and extended it with a feature combination we had never tested, beating our own
hand-built blend (+0.0038). The run required **zero manual interventions**; the agent recovered
from its own code failure via the retry path and emitted the final test predictions itself.

We report +0.0040 and +0.0038 as equivalent within noise rather than claiming a decisive win
over our own baseline attempt — the point is that the autonomous path matched the hand-tuned
one, which is what the challenge is actually asking for.

We verified `pipeline/evaluate.py` is bit-identical to the organizers' `evaluate.py`
(max abs diff 1.7e-14) and ~7x faster, and that our row order matches their loader exactly
(170,588/170,588 on user_id, video_id and label) so `row_id` alignment is correct.

### The central finding: diversity, not capacity

Single-model search is exhausted on this dataset — roughly a dozen approaches all land within
±0.005 of the baseline. The gain comes from **rank-blending models that disagree with each
other**, including models that individually lose to the baseline.

Measuring rank-correlation between components on validation shows why. Components fall into
FAMILIES, and within a family they correlate 0.98–0.99 — which is exactly why averaging five FM
seeds buys only +0.0007. Across families the correlation drops to ~0.65:

| axis | example | correlation with the rest |
|---|---|---|
| factorisation machines | fm_k16, fm_k32, fm_all37, dcn | 0.98–0.99 within family |
| gradient-boosted trees, pointwise | lgb_binary, lgb_deep | 0.98–0.99 within family |
| **ranking objective** | lgb_lambdarank | **0.645–0.72 against everything** |
| item popularity | itempop | 0.67–0.72 |

`lgb_lambdarank` is the weakest LightGBM variant standalone (0.5998) yet earns its place in every
good blend, purely because it disagrees with the others. Decorrelation alone is not sufficient
though: a naive 50/50 blend of the two *least*-correlated components actually loses. You need
decorrelation **and** comparable strength, hence weighted blending.

An equal-weight blend of six components drawn from different families reaches validation 0.6054.

### Is the gain real, or validation overfitting?

Blend weights are chosen on validation, which risks reporting a number that will not transfer.
We measured this directly with a **user-grouped** K-fold estimate — folding by user, not by row,
because GAUC and nDCG are computed per user and a row-level split would leak across folds:

| | primary |
|---|---|
| in-sample (weights fitted and scored on the same split) | 0.6047 |
| honest out-of-fold | 0.6045 |
| **optimism** | **+0.0002** |

The gain is real. This also justified letting the agent tune blend weights on validation.

### What the agent found, including the negative results

Both the agent and manual probes converged on the same conclusion: **a well-regularised FM is
very close to optimal on this data.** Roughly ten distinct approaches all landed within ±0.005:

| approach | validation primary |
|---|---|
| official FM baseline | 0.6016 |
| all 37 features instead of 5 | 0.6019 |
| DCN | 0.6024 |
| target/count encoding (train-only) | 0.6018 |
| embedding dropout | 0.6021 |
| 5-seed ensemble + early stopping | 0.6020 |
| DeepFM | 0.5897–0.6010 |
| listwise softmax CE | 0.4632–0.5990 |
| pairwise BPR | 0.5804–0.5955 |
| NFM | 0.5598 |

Two mechanisms explain this. **Capacity does not help**: 1.14M rows over 26K users is ~43 rows
per user, so deeper models overfit; FM's bilinear form is the right inductive bias. **Ranking
objectives lose to logloss**: 33% of impressions are positive, so this is not the sparse implicit
feedback regime BPR targets, and GAUC rewards calibrated pointwise scores.

We also measured that duration is a weak signal (long_view rate 0.273–0.376 by duration decile,
non-monotonic) — `long_view` is already defined relative to duration, so the normalisation is
baked into the label.

The headroom is also smaller than the oracle suggests: the ceiling is 0.8484, not 1.0, because
27.1% of test users are all-negative and score nDCG 0 for any model.

## Development tools
VS Code, Claude Code (orchestration and implementation), Python 3.12 on Windows.

## APIs used
OpenAI Chat Completions (gpt-4o) as the agent's proposer, via a stdlib `urllib` client — no SDK
dependency. The interface is a single injected `complete(prompt) -> (text, tokens_in, tokens_out)`
callable, so any provider can be swapped in; an Anthropic client ships alongside it.

## Libraries and frameworks
PyTorch (CPU), NumPy. Metrics, data loading, submission handling and the agent itself are
stdlib + NumPy only — no pandas, no scikit-learn.

## Datasets
KuaiRand-Pure (Zenodo 10439422), used under the organizers' fixed date splits. No external
training data. Post-click signals (`play_time_ms`, `is_click`, `long_view`-adjacent outcomes) are
exposed only as auxiliary targets and are asserted absent from the feature set by test;
`video_features_statistic_pure.csv` is excluded entirely because its counts are aggregated over
the whole log period, including the validation and test windows.

## Limitations
The delta over baseline is small and close to the noise floor (baseline seed std 0.0008). We
report it as measured rather than as a decisive win. The agent's per-iteration failure rate is
~25–40%, mostly tensor-shape errors in generated code; the largest remaining improvement would be
a cheap shape smoke-test before committing to a full training run.
