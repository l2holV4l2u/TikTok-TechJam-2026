# Autonomous ML Research Agent — KuaiRand-Pure

## What we built

An LLM-driven agent that runs the full MLE iteration loop on KuaiRand-Pure with no human in it.
It inspects the data, stands up an end-to-end pipeline and reproduces the official baseline, then
repeatedly proposes a hypothesis, writes the code, trains, evaluates against the official metric,
**revises what it believes**, and decides what to try next — until validation converges under the
organizers' rule (ε = 0.002, N = 3).

```
                 ┌──────────────────────────────────────────────────────┐
                 ▼                                                      │
 inspect data ─► reproduce baseline ─► select node ─► propose ─► execute ─► evaluate ─► revise
   (agent's        (Requirement 1)     (adaptive:     (LLM edits  (sandbox,   (GAUC /   beliefs
    own EDA)                            refine or      a script)   timeout)   nDCG@5)    (LLM)
                                        broaden)                                           │
                                            ▲                                              │
                                            └────── belief set guides the next choice ──────┘
```

Every iteration appends one immutable record: phase, parent node, hypothesis, full code, metrics,
seconds, tokens in and out, status, and any error plus how it was handled. The agent is the
product; the recommender is the sandbox it works in.

## The design decision the project turns on

**The agent's prompt contains no findings about this dataset.** It carries the task
specification, the pipeline API and the output contract — no model skeleton, no list of what has
been measured, no suggestion of what to try. The agent establishes all of that itself: it writes
its own EDA (the *only* dataset knowledge it ever gets), reproduces the baseline, and after every
experiment rewrites a **belief set** — claims with evidence and a status of active / qualified /
invalidated.

We did this because Innovation is judged on "what the agent identified as worth trying and why."
An earlier version wrote our findings into the prompt; it is kept in
`archive/proposer_v1_human_priors.py` as an honest record, and
`tests/test_proposer.py::test_brief_carries_no_human_findings` now fails the build if a finding
creeps back in.

**That guard was not enough.** Our own measurements were also sitting in the knowledge base's
`expected_effect` fields, injected every improve iteration — the `rank_aggregation` entry spelled
out the winning recipe and its 0.6/0.3/0.1 weights. Fifteen of twenty-eight entries were
contaminated, and uselessly so, since retrieval never surfaced them. Entries now state only what
each paper claims, guarded by a second test. *The prompt is not the only way a human's answer
reaches the agent.*

**It works.** Unprompted, the agent's EDA measured things we had hand-written into the old brief
(duration is weak and non-monotonic; `is_lowactive_period` is constant) plus a shift we had never
documented: between train and validation, users with zero positives go 5.1% → 30.3% and median
rows per user 59 → 7.

**And one of our priors was simply wrong.** The old brief asserted that per-user sequences do not
exist here, so "DIN, DIEN, BST, SASRec and GRU4Rec are NOT implementable" — closing off an entire
literature. Sequences are not a column but are constructible by ordering each user's rows by
date, and once we stopped withholding the impression date the agent built exactly that: DIN-style
candidate-aware pooling over each user's strictly-prior positives. It scored 0.6032 against
DeepFM's 0.6043 — it did not win, but it ran, which our brief called impossible. A human prior is
not merely redundant; it propagates a human's mistake into every iteration and, unlike the
agent's own beliefs, can never be revised by evidence.

## The search follows the literature, not our intuition

- **Breadth beats depth, adaptively.** FML-bench ([arXiv:2510.10472](https://arxiv.org/abs/2510.10472))
  finds broad exploration beats narrow-deep refinement, and that switching to breadth *on
  detecting stagnation* beats every fixed strategy ([arXiv:2605.17373](https://arxiv.org/abs/2605.17373)).
  We refine while gains land; the moment an iteration fails to clear ε we broaden — keeping the
  best script as the base but demanding a change of *direction*, with everything already tried
  listed so a restatement does not qualify.
- **Exhaustive search is the wrong tool at frontier strength.** Gome
  ([arXiv:2603.01692](https://arxiv.org/abs/2603.01692)) measures a crossover: weak models favour
  tree search, frontier models favour directed updates (35.1% vs 24.0% any-medal on MLE-bench).
- **Information management, not solution management, is the centre.** Iris
  ([arXiv:2608.02143](https://arxiv.org/abs/2608.02143)) reports 64.9% any-medal vs AIDE's 17.1%
  at half the budget; its small-data ablations cost −26.7 points without adaptive topology, −13.4
  without knowledge management. Hence the belief set instead of append-only reflections — our own
  logs showed a run recommending the same experiment four times because a reflection can never be
  overturned. Across 8 runs the agent formed **45 claims and revised 4**.
- **One iteration is one script, not one model.** A script may build and compare several
  candidates within its time budget and report them (`CANDIDATES`), buying comparisons the
  iteration budget cannot. It may also print `FINDINGS` — a distribution, a correlation, an
  assumption checked — which feed the belief set whatever the score was.

The rendered search tree (`runs/<id>/search_tree.txt`) and belief set (`runs/<id>/knowledge.md`)
are deliverables in their own right: they show where the budget went and what the agent concluded.

## Results

Official baseline (organizer-provided FM, k=16): validation primary 0.6016, hidden test 0.5946.

Submitted run (`runs/r87_3slot`, three parallel lineages):

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| validation, agent's best iteration (train-only fit) | 0.6727 | 0.5388 | 0.6057 |
| official baseline, validation | 0.6674 | 0.5357 | 0.6016 |
| **hidden test, this submission** | **0.6676** | **0.5332** | **0.6004** |
| official baseline, hidden test | 0.6610 | 0.5282 | 0.5946 |

**Absolute delta on hidden test: +0.0058 primary** (GAUC +0.0066, nDCG@5 +0.0050), from
**14 iterations of 50**, **22 minutes**, **246,618 tokens**, CPU only, **0 manual interventions**.
The agent wrote its own EDA, reproduced the official baseline on the first attempt, and emitted
the test predictions that became the submission.

### The result that matters most

We deleted our own best finding from the prompt — that rank-blending *decorrelated* models beats
any single model — and the agent derived it again, unaided. One improve iteration's hypothesis,
in its own words:

> "replacing two redundant DeepFM seeds with low-rank DCN-V2 cross models; their explicit
> bounded-order feature crosses should create **less-correlated** user–item ordering errors that
> heterogeneous rank aggregation can **cancel**."

It then compared fourteen aggregation schemes inside that one iteration, and its belief set
recorded the correlation measurements explaining the ceiling it hit ("rank correlations were
generally about 0.94 or higher"). We had run that same correlation study by hand in the earlier
version of this project and reached the same conclusion — the difference is that this time nobody
told it to.

### What the architecture changes did

Nine runs, one change at a time, each converged with zero manual interventions:

| | validation | hidden-test delta |
|---|---|---|
| scaffolding fixes only (r27–r30) | 0.6023 – 0.6033 | +0.0006 – +0.0024 |
| literature-driven changes (r33–r37) | **0.6037 – 0.6049** | **+0.0033 – +0.0041** |

r27–r30 are the honest negative result: **none of those four scaffolding fixes moved the score**,
and their spread is the size of the baseline's own seed noise. The worst run after beats the best
run before on both metrics, 5/5 against 4/4 — but five runs is a small sample reading a ~0.0015
effect against a 0.0008 noise floor, so we report ranges rather than a mean and a p-value.
`research/verify_claims.py` re-checks every published row against the run records.

### Parallel lineages: the slot ladder

The portfolio advances *n* lineages per turn under one convergence counter, with an archive, a
refill policy and a cross-lineage blend:

| slots | run | validation | hidden test | delta | wall-clock | tokens |
|---|---|---|---|---|---|---|
| 1 | r85 | 0.604424 | 0.598788 | +0.00419 | 10.8 min | 128,159 |
| 2 | r86 | 0.604452 | 0.599473 | +0.00487 | 15.4 min | 179,572 |
| **3** | **r87** | **0.605713** | **0.600410** | **+0.00581** | 22.0 min | 246,618 |
| 4 | r88 | 0.605090 | 0.600194 | +0.00559 | 27.4 min | 319,616 |
| 5 | r89 | 0.605716 | 0.600256 | +0.00566 | 32.9 min | 429,394 |

Three slots is the knee: one to three buys +0.0016 of hidden-test delta for 2× the wall-clock,
and five buys nothing for 1.7× the tokens. r89 ties r87 on validation to within 0.000003 — far
below the 0.0008 noise floor — so we submit the cheaper configuration.

**The gate that nearly killed this feature was broken, not the feature.** Slots are only worth
their cost if they explore differently, measured as mean pairwise rank correlation. The first
measurement read 0.94–1.00 and said "three expensive copies of one agent." Two bugs produced it:
`retain_or_blend` overwrote each slot's published scores with the winner, so discarded slots were
compared against the *same incumbent array* and correlated at exactly 1.0000; and sibling
disclosure was assigned after scoring, so every slot opened turn 1 from an identical prompt — the
slots were not merely measured as clones, they were being *made* into clones. Corrected, the
lineages start around 0.6–0.92 and **diverge** as the run proceeds. Had we trusted the first
number we would have deleted the subsystem and reported "search breadth does not help here" as a
finding. A measurement that decides whether to delete a subsystem deserves the same scrutiny as
the subsystem.

### Selection integrity

- **We select on validation only.** We hold the public test labels and therefore *can* see each
  iteration's test score, but choosing on it would be fitting the hidden set. r37 scored the best
  hidden-test delta of any eligible run (+0.0041) and we did not submit it, because its validation
  score was not the best.
- **We do not run a selection lottery.** Across r33–r37 the runs do not separate from each other
  (validation spread 0.0012, test 0.0008, both under the baseline's 0.0008 seed noise; of ten run
  pairs, 4 concordant, 5 discordant, 1 tied). Running the agent twenty times and submitting the
  peak would be sampling noise.
- **We stop where the rule says the run is over** — the first of ε/N = 0.002/3, 50 iterations, or
  6 hours. An earlier run produced its best number six iterations past its own convergence point;
  truncating it at the rule cost 0.0003 on test, and we truncated it.
- **Rejected runs.** r39/r41/r43/r44 looked better after exposing
  `video_features_statistic_pure.csv`, but that file's item statistics average over the full
  month, overlapping validation and test. Under the fixed date split that is future-window
  information; the harness now excludes it and provides equivalent train-only aggregates.
- `pipeline/evaluate.py` is bit-identical to the organizers' `evaluate.py` (max abs diff 1.7e-14)
  and our row order matches their loader exactly (170,588/170,588).

### Why the deltas on this benchmark are small — measured, not assumed

Roughly fifteen distinct approaches all landed within ±0.005 of the baseline. We measured why
(`python -m research.ceiling_probe`): fit a deliberately over-powered LightGBM on all 37 fields
and score it both in-sample and on validation.

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| high-capacity, **in-sample** | 0.9456 | 0.9034 | **0.9245** |
| same model, **validation** | 0.6469 | 0.5266 | 0.5868 |
| official baseline, validation | 0.6674 | 0.5357 | 0.6016 |

The features separate `long_view` almost perfectly on the training window and essentially none of
it transfers — a generalisation gap of 0.3377, and *worse than the baseline it dwarfs in
capacity*. Capacity is not the constraint; transfer across the date boundary is. So the
baseline's small k=16 FM is close to the right capacity, the realistic ceiling is ~0.60–0.61 test
primary, and the methods with a mechanism here target drift rather than capacity — which is why
we exposed `Split.date` to the agent. (This probe is human analysis, clearly labelled and not on
the submission path; full write-up in `docs/generalisation-ceiling.md`.)

## Bonus dataset: KuaiRand-1K

We ran the same agent, unchanged, on KuaiRand-1K — not to show our Pure model scores well there,
but to ask whether **the agent** adapts when the problem changes underneath it.

| | Pure | 1K |
|---|---|---|
| train rows | 1,141,112 | 5,055,984 |
| distinct train videos | ~7,600 | 2,119,510 |
| impressions per test user | 7.1 | 4,145 |
| **test rows on a video never seen in train** | **0.01%** | **84.94%** |
| perfect-ranking ceiling | 0.8645 | 0.9995 |

Converged in 14 iterations, 2.7 h, **0 manual interventions**, 192,900 tokens, 76 internal
candidates. Against the organizers' recipe run by us (0.6355 primary) the agent reached
**0.6777 (+0.0422)**. *These numbers are not comparable to the Pure result* — a 0.9995 ceiling, a
weaker anchor, no published baseline. What makes them trustworthy is that the same script
reproduces Pure's published 0.6016/0.5946 as 0.6022/0.5957.

The agent named the problem itself in its first improve iteration, proposing content features to
rank "the 74% of validation impressions whose video IDs were unseen in training" — it had
measured the cold-item rate in its own EDA and concluded that when item identity is untrainable,
you substitute item content. A different architecture from the rank aggregation it converged on
for Pure, and neither was suggested to it. When iteration #9 hit a hard LightGBM limit
(`Number of rows 13924 exceeds upper limit of 10000 for a query`, reachable only because 1K's
users are dense), it recovered in one attempt by chunking oversized users and added its own
invariants. Full log in `RUN_REPORT_1K.md`.

## What the agent adopted, and what it ignored

| capability | kind | adoption |
|---|---|---|
| `s.num` — continuous features | data | **63.0%** (17/27) |
| `s.time_ms` — impression order | data | **33.3%** (9/27) |
| `s.date` — impression day | data | **29.2%** (19/65) |
| `FINDINGS` — report epistemic evidence | process | 9.8% (38/387) |
| `CANDIDATES` — compare inside an iteration | process | 8.5% (33/387) |
| `RUN_ARTIFACTS` — cache between iterations | process | 1.0% (4/387) |
| `evaluate(per_user=True)` — segment diagnosis | process | 0.8% (3/387) |

**Give the agent new data and it uses it; give it new process and it mostly does not.** Every
data channel was picked up within an iteration or two; every optional protocol sat near the
floor, including two we were confident about. (Caveat: `s.num` and `s.time_ms` were measured on a
56-field channel later withdrawn as ineligible, so the behavioural claim stands but those exact
rates do not.) It cost us: `RUN_ARTIFACTS` and `per_user` were built early and did nothing for
six runs. We report adoption numbers rather than a feature list, because a capability with 0.8%
uptake is one we should not claim.

## What we learned building the harness

Every recurring failure traced to a defect in the scaffolding, not the model's reasoning, and
each fix showed in the failure rate: ~50% → 35% → 27% → 0%.

- **Two halves of the prompt can disagree.** We told the agent ensembling was the proven
  direction; it proposed single models six times running, because the retrieved-papers block
  below was seeded with "click", "ranking", "ndcg" and surfaced only single-model papers.
  Retrieval later tunnel-visioned again — the same three entries every iteration, four ensembling
  papers unreachable. The fix is not pinning a paper (that is steering) but excluding what has
  already been surfaced and handing the agent the full catalogue.
- **The analysis stage and the proposer optimised different objectives.** Every reflection asked
  for the *most informative* experiment; the run ends after three iterations without a **gain**.
- **The agent did not know how the run ends**, so it spent early iterations on cautious ablations.
  It now gets the stopping rule and its live budget state — task specification, not a dataset hint.
- **A completed run was killed by its own success message**: a hypothesis containing `×` met a
  legacy stdout codepage and raised `UnicodeEncodeError` while *printing the result*, destroying
  the run metadata after the loop had converged. Agent output is untrusted text.
- **What actually determines the score is completed iterations.** Every run that finished three
  improve iterations gained +0.0000–+0.0019 over its own baseline; the only runs above +0.0029
  are the only ones that finished more than three. The cause was mundane and ours: a 3-requests-
  per-minute tier, breached within two iterations, bleeding whole iterations to 429s. The client
  now spaces requests below the limit — ~40 seconds per iteration against losing an experiment.

**Robustness.** Failures are handled in-loop; the agent never escalates. A crash returns its
traceback *and its source* so the proposer fixes the failing line (two retries, then the idea is
retired, keyed by method name so rewording cannot evade it); timeouts are fed back differently
from crashes ("too slow, not wrong"); timed-out scripts are killed process-tree-wide; LLM outages
and rate limits are absorbed with server-hinted backoff, with a per-day cap recognised
immediately and failed over to another key; and a circuit breaker halts with `environment_broken`
after five instant, output-less failures.

**Harness fixes worth naming.** `pipeline/models.py` initialised every embedding with `N(0,1)`,
which starts an FM's interaction term orders of magnitude too large (0.5533 valid after 40 epochs
against 0.6020 in under 15) — fixed for all four architectures, and no run was affected, 0 of 333
agent scripts import it. `agent/facts.py` now measures the brief's dataset constants instead of
hard-coding them, which corrected one shipped error. `evaluate()` is 1.47× faster with
bit-identical output. `--replay` re-runs a recorded run with no network and no tokens, so loop
and parser changes can be tested in 2.9 minutes instead of 30. And the harness is CPU-only by
measurement: the GPU returned only ~1.25× end-to-end and changed results, since `torch.randperm`
draws a different permutation on CUDA.

## Three things we tested, and what the results said

1. **Selecting on a chronological window — refuted.** The test window sits after validation, so
   ranking candidates on the last validation days looks like the closer proxy. Across 49
   iterations, full 7-day validation ranked iterations *more* like the hidden test (Spearman
   0.8659) than the last 4 (0.8397), 3 (0.6637) or 2 days (0.3273), and all four selectors picked
   the same model. The proposer prompt now marks the axis refuted.
2. **Recency weighting — adopted, but only in the right place.** Standalone, uniform day weights
   score 0.4597 against 0.5518 for a 4-day half-life. Applied to a *side* component the blender
   damped it to +0.00002; directed at the main model's `sample_weight` it produced r90's winning
   recency-weighted LightGBM. The lesson is placement, not technique.
3. **A portfolio of parallel lineages — abandoned, then reinstated when the measurement was
   fixed** (see the slot ladder above).

## One thing we deliberately did not do

KuaiRand-Pure ships a randomised-exposure log that the problem statement points at for
counterfactual evaluation, and our loader ignores it entirely — because it covers 22 Apr – 8 May,
exactly the validation and hidden-test windows. The splits are date-based precisely to stop a
model learning from the evaluation period. No rule names the file, which makes it ambiguous
rather than permitted, and we would rather report a smaller honest delta. If the organizers
confirm it is in scope for training, it is the single most promising unexplored direction here
and the change is one loader function.

## Tools, APIs, libraries and data

**Development:** VS Code, Python 3.12 on Windows.
**APIs:** OpenAI Chat Completions for the proposer and belief revision, via a stdlib `urllib`
client — no SDK dependency. The submitted run used **gpt-5.6-sol** for both; later runs route
belief revision to a second model, since rate limits are per-model. Every call, with its model,
prompt, response and token counts, is recorded in `llm_calls.jsonl`. The interface is a single
injected `complete(prompt) -> (text, tokens_in, tokens_out)` callable, so any provider can be
swapped in; an Anthropic client ships alongside it.
**Libraries:** PyTorch (CPU), NumPy, LightGBM. Metrics, data loading, submission handling and the
agent itself are stdlib + NumPy only — no pandas, no scikit-learn.
**Datasets:** KuaiRand-Pure (Zenodo 10439422) under the organizers' fixed date splits, plus
KuaiRand-1K for the bonus. No external training data. Logged outcome signals are exposed only as
auxiliary targets and asserted absent from the feature set by test.

## Limitations

- The delta over baseline is small and close to the noise floor (baseline seed std 0.0008); we
  report it as measured rather than as a decisive win.
- Every script gets the same time budget regardless of what it attempts, which biases against
  methods that legitimately need longer.
- **We never validated the belief set's claims.** Nothing verifies a claim before it is carried
  into the next prompt, so a false belief would propagate silently — the same failure mode as the
  human priors we removed.
- Cross-run memory is new and nearly unproven: it reads only runs from this architecture, so it
  has little history to draw on.
- With a 6-hour ceiling against a 22-minute run, most of the compute budget is still unused.

*A longer version of this write-up, with the full per-run tables and failure logs, is in
`DEVPOST_full.md`.*
