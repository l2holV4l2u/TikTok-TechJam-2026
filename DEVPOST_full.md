# Autonomous ML Research Agent — KuaiRand-Pure

## How the solution addresses the problem statement

We built an LLM-driven agent that runs the full MLE iteration loop of Figure 1 on KuaiRand-Pure
without a human in it. It inspects the data, stands up an end-to-end pipeline and reproduces the
official baseline, then repeatedly proposes a hypothesis, writes the code, trains, evaluates
against the official metric, **revises what it believes**, and decides what to try
next — searching over a tree of solution scripts until validation converges under the
organizers' rule (ε = 0.002, N = 3).

The agent is the product; the recommender is the sandbox it works in.

### The loop

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

Every iteration appends one immutable record: phase, parent node, hypothesis, full code,
metrics, seconds, input and output tokens, status, and any error plus how it was handled.

### The design decision the whole project turns on

**The agent's prompt contains no findings about this dataset.**

It contains the task specification (the organizers' own numbers), the pipeline API, and the
output contract. It does not contain a working model skeleton, a list of what has already been
measured, or any suggestion of what to try. The agent has to establish all of that itself:

1. **It inspects the data.** The first thing it does is write an exploratory script and print
   what it finds. That output is the *only* dataset knowledge it ever gets, and it is carried
   into every subsequent prompt.
2. **It reproduces the baseline.** Requirement 1, done by the agent, checked against the
   organizers' published 0.6016. The script it writes becomes the root of the search tree.
3. **It revises what it believes.** After each experiment a separate call rewrites the agent's
   belief set — claims with evidence and a status of active / qualified / invalidated. That set,
   not a human-authored brief, is where the run's knowledge lives.

We did this because of how Innovation and Autonomy are actually scored. Innovation is judged on
"what the agent identified as worth trying and why." If we write *"ensembling is the proven
direction, here is the pattern"* into the prompt, then we identified it, not the agent — and the
run becomes an expensive way of typing up our own research.

An earlier version of this project did exactly that. Its prompt is kept in
`archive/proposer_v1_human_priors.py` as an honest record. At the time we removed it, it scored
better, and we removed it anyway — the autonomy claim was worth more than the delta.
`tests/test_proposer.py::test_brief_carries_no_human_findings` now fails the build if any finding
creeps back in.

**And that guard was not enough.** Looking for what else fed the prompt, we found our own
measurements in the knowledge base's `expected_effect` fields — injected on every improve
iteration. The `rank_aggregation` entry spelled out the winning recipe, its 0.6/0.3/0.1 weights
and the rank-transform trick; fifteen of twenty-eight entries were contaminated, and uselessly
so, because retrieval never surfaced them anyway. The entries now say what each paper claims and
nothing about what we measured (`archive/papers_v1_human_priors.json` keeps the originals), and
`test_knowledge_base_carries_no_measured_results` guards the channel the first test missed.
The lesson generalises: *the prompt is not the only way a human's answer reaches the agent.*

**It works.** Unprompted, the agent's own EDA independently measured several things we had
previously hand-written into that old brief — that duration is a weak, non-monotonic signal
(long_view rate 0.273–0.376 across buckets), that `is_lowactive_period` is constant — plus a
distribution shift we had never documented: between train and validation the share of users
with zero positives goes 5.1% → 30.3%, and the median rows per user 59 → 7.

**And one of our priors was simply wrong.** The old brief stated flatly that per-user behaviour
sequences do not exist here, so "DIN, DIEN, BST, SASRec and GRU4Rec are NOT implementable" — an
instruction that closed off an entire literature. It was false. Sequences are not a column, but
they are constructible by ordering each user's rows by date, and once we stopped withholding the
impression date the agent built exactly that: a DIN-style candidate-aware pooling over each
user's strictly-prior positive videos. It scored 0.6032 against DeepFM's 0.6043, so it did not
win — but it ran, which our brief had asserted was impossible.

That is the sharpest argument for the whole design. A human prior is not merely redundant when
the agent could find it alone; it is a hard constraint that propagates a human's mistake into
every iteration, and unlike the agent's own beliefs it can never be revised by evidence.

### The search follows the literature, not our intuition

Our first design was solution-centric tree search: every executed script a node, expand the
best, retire what stops paying. We then read what the current systems actually do, and rebuilt
around three findings that contradicted us.

**Breadth beats depth, and the switch should be adaptive.** FML-bench finds broad exploration
beats narrow-deep refinement ([arXiv:2510.10472](https://arxiv.org/abs/2510.10472)), and that an
agent switching to broader exploration *on detecting stagnation* outperforms every fixed
strategy, with breadth tracking opportunity density
([arXiv:2605.17373](https://arxiv.org/abs/2605.17373)). Our prompt had said "make ONE targeted
change" unconditionally — the losing strategy, applied always. Now: refine while gains land; the
moment one iteration fails to clear ε, broaden — keep the best script as the base but demand a
change of *direction*, with everything already tried listed so a restatement does not qualify.
Gains here are sparse (almost nothing clears 0.002), which is exactly the regime the paper says
favours breadth.

We got that detail wrong first, and a run showed us. Broaden originally sent the agent back to
the *earliest* node, to reuse working plumbing. In one of those runs it proposed recency weighting — the right
idea for this dataset's constraint, worth +0.0005 in its own sweep — but anchored to the plain
baseline instead of the leading DeepFM, the iteration scored 0.6034 against a 0.6043 leader and
was recorded as a loss. Breadth belongs in method space, not in which file you start from.

**Exhaustive search is the wrong tool at frontier model strength.** Gome
([arXiv:2603.01692](https://arxiv.org/abs/2603.01692)) measures a crossover across ten models:
with weak models tree search wins by 1.3–2.7 points, but at frontier tier directed updates win
by 2.7–7.1, reaching 35.1% vs 24.0% any-medal on MLE-bench with GPT-5. We were running
exhaustive search on a frontier model — the wrong side of that line.

**Information management, not solution management, is the centre.** Iris
([arXiv:2608.02143](https://arxiv.org/abs/2608.02143), 64.9% any-medal vs AIDE's 17.1% at half
the budget) argues that organising research around candidate solutions leaves the system's
evolving understanding secondary, and that under limited budgets this is what costs you. Its
ablations on small-data tasks — the closest published setting to ours — put numbers on it:

| component removed | any-medal | delta |
|---|---|---|
| — (full system) | 66.7% | — |
| adaptive topology | 40.0% | −26.7 |
| knowledge management | 53.3% | −13.4 |
| epistemic actions | 60.0% | −6.7 |

So the agent now keeps a **belief set** rather than a pile of reflections: claims with evidence
and a status of `active` / `qualified` / `invalidated`, rewritten after every scored iteration so
later evidence can demote an earlier conclusion. We built this because our own logs showed the
failure it prevents — one run recommended the same next experiment four times running, because
an append-only reflection can never be overturned.

How often revision fires is worth stating rather than implying: across 8 runs the agent formed
**45 claims and revised 4** (3 invalidated, 1 qualified). That is not a weak reflect stage — it
is the convergence rule. At a median of **4 improve iterations per run**, a claim formed at
iteration #2 gets one or two chances to be contradicted. The revisions are specific and
evidence-cited, e.g. *"does not improve the k=16 FM; it scored 0.5998, which is 0.0026 below the
unweighted reproduction and outside the stated seed-noise threshold"*.

Two smaller consequences of the same reading:

- **One iteration is one script, not one model.** A script may build and compare several
  candidates within its time budget and report what it compared (`CANDIDATES`). The convergence
  rule charges an iteration per experiment, so searching *inside* an iteration buys comparisons
  the iteration budget cannot.

  We checked whether searching harder inside an iteration pays. Across nine eligible Pure runs
  the candidate count ranges 0-48; the three above the mean average 0.6043 best-validation
  against 0.6032 for the other six. That +0.0011 sits inside the 0.002 noise threshold and is
  confounded with later harness revisions, so the mechanism is used but not shown to cause a
  gain.
- **Epistemic evidence rides along.** A script may print `FINDINGS` lines — a distribution, a
  correlation, an assumption checked — which feed the belief set whatever the score was. Iris
  scores diagnostic actions separately; our convergence rule cannot afford a purely diagnostic
  iteration, so we attach them to a scored one instead.

The rendered search tree (`runs/<id>/search_tree.txt`) and the belief set
(`runs/<id>/knowledge.md`) are deliverables in their own right: they show *where* the agent
spent its budget and *what it concluded*, not just what it scored.

### Robustness

Failures are handled in-loop; the agent never escalates.

- A crashed script's traceback **and its source** go back to the proposer, so it fixes the
  failing line instead of rewriting from scratch. Two retries, then the idea is retired.
- Ideas are keyed by **method name**, so rewording "Implement a LambdaRank loss" as
  "Implementing a LambdaRank objective" does not evade retirement. Sentence-level Jaccard, tried
  first, read those two as different ideas.
- **Timeouts are fed back differently from crashes** ("too slow, not wrong"), because rerunning a
  slow approach unchanged just times out again.
- Timed-out scripts are killed **process-tree-wide**. A bare `kill()` left orphaned trainers
  running for an hour, competing for CPU and corrupting the reported compute.
- LLM outages and rate limits are absorbed with server-hinted backoff; the loop survives them.
- A circuit breaker halts with `environment_broken` after five consecutive instant, output-less
  failures — added after a torn-down parent left the runner unable to spawn children and 32
  iterations "failed" in 0 seconds each, silently shredding the budget.

### What we learned about steering an agent

Most of this project's engineering went into the agent's *scaffolding*, not its reasoning. Every
recurring failure traced to a defect in what we had built around the model, and each fix showed
up in the failure rate: ~50% → 35% → 27% → 0%.

- It called `.cuda()` repeatedly because our brief claimed a GPU when torch was CPU-only — a
  false statement we had written into the prompt.
- It crashed on `IndexError` five times building per-video tables, because our own helper
  applied field offsets and nothing said raw ids were needed for lookups.
- It spent 8 of 17 iterations on one losing idea: retirement only triggered on crashes, so an
  idea that ran cleanly and scored badly was never excluded. Our fix then killed the iteration
  that scored the run's **best** result, because "did not beat the incumbent by ε" counted as an
  underperformance.
- Generated blend scripts arrived truncated with `SyntaxError: '(' was never closed` — our
  `max_tokens` was 4096, cutting the model off mid-script.

The subtlest: after we added a section telling the agent that ensembling was the proven
direction, it still proposed only single models, six times in a row. The retrieved-papers block
directly beneath was seeded with query terms like "click", "ranking", "ndcg", so it surfaced only
single-model papers, every call. Two halves of one prompt disagreed and retrieval won.

Rebuilding around search and a belief set produced more of exactly the same kind, found only by
reading the logs:

- **The analysis stage and the proposer disagreed.** Every reflection in our first run ended
  "next, run matched-seed leave-one-field-out ablations" — and the proposer never ran one, four
  times running. The reflection prompt asked for the *most informative* experiment; the run ends
  after three iterations without a **gain**. Two halves of one system optimising different
  objectives — the v1 retrieval bug in a different hat.
- **Retrieval tunnel vision came back.** The paper retriever returned the same three entries
  every iteration, because the query is built from the agent's own recent hypotheses — so a run
  that opens on factorization machines is only ever shown factorization machines, and four
  ensembling papers sat unreachable. In v1 we had "fixed" this by pinning a chosen paper, which
  is just steering. A retriever returning the same three books every visit is broken: it now
  excludes what it has already surfaced, and the agent gets the full catalogue and decides.
- **The agent did not know how the run ends.** It spent early iterations on cautious ablations,
  each burning one of the three lives the convergence rule allows. It now gets the stopping rule
  and its live budget state — task specification straight out of the organizers' document, not a
  hint about the dataset.
- **A completed run was killed by its own success message.** A hypothesis contained `×`
  (U+00D7); redirected stdout on this machine defaults to the system codepage; the harness raised
  `UnicodeEncodeError` while *printing the result*, after the loop had converged and written its
  submission, destroying the run metadata. The agent's own output is untrusted text and must
  never be able to crash the harness that reports on it.
- **A retry policy that could not tell two rate limits apart.** A per-minute limit clears if you
  wait; a per-day cap does not. Ours treated both as transient and spent twelve minutes climbing
  a backoff ladder against a wall. The client now recognises a daily cap immediately and fails
  over to another key, marking the capped one so later calls skip it.

None of these were visible from the score, only from reading what the agent actually chose to do
— which is what the per-iteration ledger exists for. The pattern across both versions is the
same: **every defect was in the scaffolding, not in the model's reasoning.**

## Results

Official baseline (organizer-provided FM, k=16): validation primary 0.6016, hidden test 0.5946.

Our submitted run (`runs/r96`, three parallel lineages; full log in `RUN_REPORT_PURE.md`):

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| validation, agent's best iteration | 0.6729 | 0.5386 | 0.6058 |
| official baseline, validation | 0.6674 | 0.5357 | 0.6016 |
| **hidden test, this submission** | **0.6668** | **0.5315** | **0.5991** |
| official baseline, hidden test | 0.6610 | 0.5282 | 0.5946 |

**Absolute delta on hidden test: GAUC +0.0058, nDCG@5 +0.0033, mean +0.0045.**

The selection rule and the score agree here: iteration 11 is the validation-best checkpoint at
the moment the convergence rule fired, and it is the one that was submitted. Nothing was chosen
by looking at the test set.

**Training data is the train split only** (20220408-20220421), per FAQ 2.9.2. No model whose
predictions are saved is fitted on validation -- not for the test predictions, not through a
refit, and not through early stopping or feature statistics chosen by watching validation.
`agent/critic.py` rejects an iteration that violates this.

Resources for the submitted run (`r96`): **14 scripts across 4 turns, of a 50-iteration cap**,
**53 minutes** of agent wall-clock, **260,967 tokens**, CPU only, **0 failures**, **0 manual
interventions**, and **316 candidate solutions compared inside those iterations**.

The agent did all of it: its own EDA script, a baseline reproduction to 0.6020 on the first
attempt, the further experiments, and the test predictions that became the submission.

### The result that matters most

We deleted our own best finding from the agent's prompt — that rank-blending *decorrelated*
models beats any single model — and the agent derived it again, unaided, and then beat us with
it.

Iteration #5's hypothesis, in its own words:

> "replacing two redundant DeepFM seeds with low-rank DCN-V2 cross models; their explicit
> bounded-order feature crosses should create **less-correlated** user–item ordering errors that
> heterogeneous rank aggregation can **cancel**."

That is the decorrelation argument from the deleted `EXHIBIT A`, reconstructed from first
principles. It compared fourteen aggregation schemes inside that one iteration — Borda versus
logit-mean, homogeneous versus heterogeneous, per-seed — and its belief set recorded the
correlation measurements that explain the ceiling it hit:

> "The tested DeepFM seeds, low-rank DCN-V2 models, per-user-weighted DeepFM models, and MMoE
> model are too prediction-correlated to yield a measurable ensemble gain; observed rank
> correlations were generally about 0.94 or higher."

We ran that same correlation study by hand in the earlier version of this project and reached
the same conclusion. The difference is that this time nobody told it to.

**The scores make the point sharply.** Our hand-tuned blend reached validation 0.6045; the
earlier agent, *with the winning recipe written into its prompt*, 0.6046; this agent, with the
recipe deleted and a test that fails the build if it creeps back, **0.6049**, at the same
hidden-test delta (+0.0039).

### What the architecture changes actually did

We ran the same architecture repeatedly, changing one thing at a time. Every run converged in 5
iterations with zero failures and zero interventions:

| # | change under test | validation best | hidden-test delta |
|---|---|---|---|
| 1 | no priors in the brief | 0.6033 | +0.0024 |
| 2 | + agent told the convergence rule | 0.6027 | +0.0013 |
| 3 | + budget-aware analysis, paper catalogue, retrieval rotation, noise-band memory | 0.6023 | +0.0013 |
| 4 | + "one iteration is one script, not one model" | 0.6030 | +0.0006 |
| **5** | **+ belief set, adaptive breadth, internal candidates, decontaminated KB** | **0.6044** | **+0.0035** |
| **6** | **+ `Split.date` exposed to the agent** | **0.6043** | **+0.0033** |
| **7** | **+ broaden anchors on the best node, not the earliest** | **0.6049** | **+0.0039** |
| **8** | **+ `evaluate(per_user=True)` for per-segment diagnosis** | **0.6037** | **+0.0036** |
| **9** | **+ `RUN_ARTIFACTS` cache directory (agent never used it)** | **0.6037** | **+0.0041** |

One of those runs lost its `run_meta.json` to an encoding crash *after* it wrote its submission (a
hypothesis containing `×` met a cp874 stdout), so its row is re-derived by scoring that
submission directly — test primary 0.5952, delta +0.0006. `research/verify_claims.py` re-checks
every row here against the run records and exits non-zero on any disagreement.

Configurations 1–4 are the honest negative result: **none of those four scaffolding fixes moved
the score**, and their spread (0.6023–0.6033) is the size of the baseline's own seed noise. We
report it rather than quietly dropping it.

Configurations 5–9 carry the literature-driven changes, and the separation is clean on both
metrics:

|  | validation | hidden-test delta |
|---|---|---|
| before (1–4) | 0.6023 – 0.6033 | +0.0006 – +0.0024 |
| after (5–9) | **0.6037 – 0.6049** | **+0.0033 – +0.0041** |

### The proposer-model comparison is inconclusive

`gpt-5.6-luna` and `gpt-5.6-terra` ran only under the later, ineligible full-month statistics:
model, harness and data contract changed at once, so their spread cannot identify a
proposer-model effect and is not used as evidence.

One failure mode was visible rather than statistical: `gpt-5.1` spent two iterations inventing
field names — `duration_ms_range`, then `duration_range`, where the real field is
`duration_bucket` — because the brief named only five of the 37 categorical features, so a model
wanting a sixth had to guess. All 37 are now listed; the weaker model found a hole the stronger
ones had been stepping around.

**The worst run after beats the best run before, on both metrics, 5/5 against 4/4.** Five runs is
a small sample and we are reading a ~0.0015 effect against a 0.0008 noise floor, so we report
ranges rather than a mean and a p-value; the direction is not in doubt.

The cost side is worth stating too: the most expensive of the later configurations took 35
minutes and 92K tokens against the cheapest early one's 8 minutes and 37K. The gain is real but not free, and a judge weighing Feasibility should see both numbers.
Both remain far inside the organizers' 6-hour ceiling and 50-iteration cap.

### Why we do not run a selection lottery

The five post-change runs separate cleanly from the four before them, but they do **not**
separate from each other. Across configurations 5–9:

| | spread | std | vs baseline seed noise (0.0008) |
|---|---|---|---|
| validation | 0.0012 | 0.00051 | below |
| hidden test | 0.0008 | 0.00032 | below |

Both are under the noise floor, and ranking the runs against each other is a coin flip: of the
ten run pairs, **4 are concordant, 5 discordant, 1 tied**. Validation carries no usable
information about which converged run will do better on the hidden test.

This matters because there is an obvious way to inflate a score here — run the agent twenty
times and submit whichever run peaks. The numbers above say that samples noise rather than
selects quality, which is the failure mode the held-out test exists to catch. We submit the
validation-best run under the organizers' rule and report the spread.

We have had runs whose hidden-test delta exceeded the submitted one while their validation score
did not. They are not submitted, because selecting them would mean choosing on the test set.
`runs/r96` is the submission on the only criterion the rules allow: it is the validation-best
checkpoint at the moment the convergence rule fired.

What is *not* ambiguous is the mechanism. In one run broaden fired for the first time at iteration
#3, and that iteration is the one that won: instead of tuning latent dimension again, the agent
changed direction to a DeepFM and swept the FM/deep mixing weight inside one iteration —

| α | 0.00 | 0.25 | 0.50 | **0.75** | 1.00 |
|---|---|---|---|---|---|
| validation primary | 0.6019 | 0.6033 | 0.6041 | **0.6044** | 0.6040 |

Five models compared for one iteration of budget. It also missed resetting the convergence
counter by 0.0001 (0.6044 against a 0.6045 threshold), a fair illustration of how tight the
ε = 0.002 rule is here.

Both mechanisms are auditable in the run logs. In that run's `llm_calls.jsonl` the search mode goes
`refine → broaden → broaden` while the belief set carries 1 → 2 → 3 claims into successive
prompts, so each proposal is conditioned on a revised reading of everything before it rather than
on a raw score history. The submitted run, r96, traces 0.6020 → 0.6046 → 0.6055 → 0.6058 across
its four turns: breadth sweeps over model families rather than refinement of one.

That last point is a measured decision, not a style. Two diagnostic runs with the convergence
stop disabled scored every iteration on *both* splits, and family sweeps moved the hidden test
+0.00224 while two tuning iterations and a 9→37 field expansion moved it −0.00001 between them.
Refinement buys validation points that do not exist on test, and because the rules force
selection onto validation, keeping it steered the harness toward the worse model. Every improve
iteration now sweeps model families; `docs/BUGS.md` carries the numbers.

The submitted run is chosen on **validation**: `runs/r96` is the validation-best checkpoint at
the point its convergence rule fired, and nothing about the hidden test entered that choice.

**We stop the run where the rule says the run is over.** The scored result is "the
validation-best checkpoint at [the convergence] point", and convergence is the *first* of
ε/N = 0.002/3, the 50-iteration cap, or 6 hours — so iterations after the ε/N rule has fired
cannot legitimately supply the submission, whatever they score. An earlier run used a looser
patience and produced its best number six iterations past its own convergence point; truncating
it at the rule cost 0.0003 on test, and we truncated it. The default `--patience 3` now makes
this correct by construction.

We verified `pipeline/evaluate.py` is bit-identical to the organizers' `evaluate.py`
(max abs diff 1.7e-14) and ~7× faster, and that our row order matches their loader exactly
(170,588/170,588 on user_id, video_id and label) so `row_id` alignment is correct.

### Why the deltas on this benchmark are small — measured, not assumed

Across this project, roughly fifteen distinct approaches all landed within ±0.005 of the
baseline: FM, DeepFM, DCN, NFM, LightGBM pointwise, LambdaMART, pairwise BPR, listwise softmax,
target encodings, multi-task heads, and rank blends of those. That needs an explanation, and the
three candidates — too little capacity, too little feature signal, or signal that does not
survive the date split — imply completely different strategies.

We measured it (`python -m research.ceiling_probe`, ~150s CPU). Fit a deliberately over-powered
LightGBM on all 37 fields and score it *both* in-sample and on validation:

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| high-capacity, **in-sample** | 0.9456 | 0.9034 | **0.9245** |
| same model, **validation** | 0.6469 | 0.5266 | 0.5868 |
| official baseline, validation | 0.6674 | 0.5357 | 0.6016 |
| oracle (true labels) | 1.0000 | 0.6968 | 0.8484 |

**The features separate `long_view` almost perfectly on the training window and essentially none
of it transfers.** The same model that scores 0.9245 in-sample scores 0.5868 on validation —
*worse than the baseline it dwarfs in capacity*. A generalisation gap of 0.3377.

So capacity is not the constraint; transfer across the date boundary is. Train is 9–21 Apr and
validation the following week — the shift the agent's own EDA measured (zero-positive users
5.1% → 30.3%, median rows per user 59 → 7).

Three consequences we act on:

1. **The baseline's small k=16 FM is not a weak starting point** — it is close to the right
   capacity for a signal this non-stationary, which is why bigger models reliably lose.
2. **The realistic ceiling is ~0.60–0.61 test primary**, not the 0.8645 oracle, which assumes
   knowledge of the labels and bounds the metric rather than the achievable score. On a benchmark
   scored by absolute delta, a few thousandths is a real result.
3. **The methods with a mechanism here target drift, not capacity.** That is why we exposed
   `Split.date` — our harness had been withholding the impression date, silently ruling out
   recency weighting and time-based validation, the methods aimed at the one thing that binds.
   Full write-up in `docs/generalisation-ceiling.md`.

This probe is human analysis, clearly labelled and **not** on the submission path — context for
reading the agent's delta, not one of the agent's findings.

## Bonus dataset: KuaiRand-1K

We ran the same agent, unchanged, on KuaiRand-1K — not to show our Pure model scores well there
(it would not) but to ask whether **the agent** adapts when the problem changes underneath it.

It does, and the problem changes a great deal. Measured before running anything:

| | Pure | 1K |
|---|---|---|
| train rows | 1,141,112 | 5,055,984 |
| distinct train videos | ~7,600 | 2,119,510 |
| test users | 23,875 | 997 |
| impressions per test user | 7.1 | 4,145 |
| **test rows on a video never seen in train** | **0.01%** | **84.94%** |
| perfect-ranking ceiling | 0.8645 | 0.9995 |
| item-popularity lift over random | +0.098 | +0.055 |

Same schema, same label, same calendar window — a different problem. Pure is dense on a small
catalogue, so item identity dominates. 1K logs 1,000 users against the full catalogue at 2.4
impressions per video, so for five test rows in six the item embedding is an untrained row, and
item popularity, the most dependable signal in the field, earns barely half as much.

**Result** (`runs/r97_1k`, three parallel lineages). Converged at turn 10 on 32 executed
scripts, 1.7 h, 3 failures, **0 manual interventions**, 659,573 tokens.

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| reference (organizers' recipe, our run) | 0.6704 | 0.6006 | 0.6355 |
| agent, iteration #30 | 0.7023 | 0.6931 | **0.6977** |
| delta | +0.0319 | +0.0924 | **+0.0622** |

Validation reached 0.7056 against the reference's 0.6422, a validation delta of +0.0634 against a
test delta of +0.0622 — the gain survives the held-out split almost intact. The winning iteration
trains four structurally different rankers on within-user impression pairs and rank-blends them,
which is what moves nDCG@5 by +0.0924: it optimises top-of-list ordering directly.

**These numbers are not comparable to the Pure result and we do not present them as a bigger
win.** 1K has a 0.9995 ceiling against Pure's 0.8645, a weaker anchor, and no published baseline
at all — the reference is our own run of the organizers' recipe. What makes it trustworthy is
that the same script reproduces Pure's published 0.6016/0.5946 as 0.6022/0.5957. Full protocol
and the leakage reasoning are in `research/transfer-1k.md`.

**What the agent actually did.** Its first improve iteration named the problem itself:

> *"Expand the FM feature stage with tag, upload type, music type, and hour; these fields vary
> within users and encode cold-video content and context, allowing user-content interactions and
> direct effects to rank the 74% of validation impressions whose video IDs were unseen in
> training."*

It measured the cold-item rate in its own EDA (74% on validation; we measured 84.9% on test) and
drew the right conclusion — when item identity is untrainable, substitute item content. That is a
different architecture from the rank aggregation over redundant crosses it converged on for Pure.
Neither was suggested to it.

**Robustness, unscripted.** Iteration #9 died on a hard LightGBM limit — `Number of rows 13924
exceeds upper limit of 10000 for a query` — a wall that exists only because 1K's users are dense
enough to overflow a lambdarank query group; on Pure, at 7 impressions per user, it is
unreachable. The agent recovered in one attempt by splitting oversized users into bounded chunks
and added its own invariants:

```python
if int(groups.sum()) != int(order.size):
    raise RuntimeError("Ranking group construction is inconsistent")
if int(groups.max(initial=0)) > MAX_QUERY_SIZE:
    raise RuntimeError("A ranking query still exceeds the size limit")
```

Iteration #11 then hit the wall-clock timeout, and the harness distinguished that from a bug —
telling it the approach was too slow rather than wrong, since re-running a slow script unchanged
only times out again. Full log in `RUN_REPORT_1K.md`.

## What the agent adopted, and what it ignored

We added capabilities over many runs and then measured which ones the agent actually used,
counting only the scripts written in runs where each capability existed:

| capability | kind | adoption |
|---|---|---|
| `s.num` — continuous features | data | **63.0%** (17/27) |
| `s.time_ms` — impression order | data | **33.3%** (9/27) |
| `s.date` — impression day | data | **29.2%** (19/65) |
| `FINDINGS` — report epistemic evidence | process | 9.8% (38/387) |
| `CANDIDATES` — compare inside an iteration | process | 8.5% (33/387) |
| `RUN_ARTIFACTS` — cache between iterations | process | 1.0% (4/387) |
| `evaluate(per_user=True)` — segment diagnosis | process | 0.8% (3/387) |

The two top rows need a caveat. `s.num` and `s.time_ms` were measured in runs that also carried
the supplied video statistics, later withdrawn as overlapping the evaluation window — so the 63%
figure describes a 56-field channel that no longer exists (the eligible channel is five fields).
What the numbers support is the **behavioural** claim, which does not depend on a capability
being eligible: the agent reaches for new data and largely ignores new protocol.

**Give the agent new data and it uses it; give it new process and it mostly does not.** Every
data channel we exposed was picked up within an iteration or two of becoming available. Every
optional protocol we invented sat near the floor, including two we were confident about: a
per-user error breakdown it had to ask nothing to obtain, and a cache directory that would have
saved it retraining the same model each iteration.

This is the most useful thing we learned about building the harness, and not what we expected.
It cost us: `RUN_ARTIFACTS` and `per_user` were built early, reported as capabilities, and did
nothing for six runs. We report adoption numbers rather than a feature list, because a capability
with 0.8% uptake is one we should not claim. One caveat on causation: `s.num` arrived alongside a
knowledge-base gap being filled (28 entries, zero on numeric features until we added five), so
its uptake is not attributable to exposure alone.

## Why this harness is CPU-only

We measured the available RTX 4050 rather than assuming it would help. A raw 4096x4096 matmul
was 16.2x faster, but the models this benchmark rewards are embedding-lookup bound: end to end
the GPU returned only ~1.25x (FM 69.3 s CPU against 55.6 s GPU, DeepFM 113.4 s against 90.3 s).
It also changes results -- `torch.randperm` draws a different permutation on CUDA, so the same
seed is not the same run, and the FM moved 0.6023 to 0.6007, inside noise but on the exact margin
the baseline gate checks. Not worth a reference that shifts with whatever device happens to be
installed, so every device path was removed.

## What actually determines the score: completed iterations

Across every eligible run, the variable that tracks the final result is not which capabilities
the harness exposed but how many improve iterations actually finished. Measured as gain over each
run's own baseline reproduction, which removes the noise in where a run happens to start:

| scored improve iterations | runs | gain over that run's baseline |
|---|---|---|
| **6** | 1 run | **+0.0029** |
| **4** | 1 run | **+0.0030** |
| 3 | 11 runs | +0.0000 … +0.0019 |

Every run that completed three improve iterations landed between +0.0000 and +0.0019, across
four harness versions and two feature contracts. The only two runs that exceeded +0.0029 are the
only two that completed more than three.

The mechanism is compounding: each iteration edits the best script so far, so a run on its sixth
attempt is refining a far stronger incumbent. The six-iteration run traces 0.6036 → 0.6040 →
0.6047 → 0.6049 —
small steps, each taken from where the last one landed, while a run cut off at two never leaves
the first plateau.

This contradicts the intuition we started with. We spent most of this project adding capability
to the harness, and capability is not what separates these runs — every one of them spent its
attempts mostly on model architecture, including the ones that scored worst. What separated them
was whether their iterations ran.

The cause was mundane and entirely ours: the account's free tier allows **3 requests per
minute**, an iteration costs two model calls plus retries, and runs breached the limit within a
couple of iterations and then bled whole iterations to 429 responses. The fix is to wait our turn
rather than retry into a wall — the client now spaces requests below the limit, costing ~40
seconds per iteration against a lost iteration costing the entire experiment.

The organizers' convergence rule (ε = 0.002 over 3 iterations) already makes iterations scarce: a
run gets three attempts and earns more only by clearing ε. Spending them on infrastructure
failures is the most expensive mistake available here, and we made it repeatedly before measuring
it.

## Harness engineering

Four changes that are not model work but decide whether the agent gets a fair run.

**The provided models were mis-initialised.** `pipeline/models.py` built every embedding table
with PyTorch's default `N(0,1)`. Summed across fields an FM's interaction term starts orders of
magnitude too large and converges to a much worse optimum: on Pure, **0.5533 valid after 40
epochs against 0.6020 in under 15**. Fixed for all four architectures, with a regression test
that asserts initial logit magnitude rather than running a slow convergence check. No run was
affected -- 0 of 333 agent scripts import that module -- and we say so rather than claim a score
we did not lose.

**The brief stated dataset facts as literals.** Row counts, the perfect-ranking ceiling and the
random and item-popularity rungs were Pure constants typed in by hand, which would have fed the
agent false premises the moment the harness pointed elsewhere. `agent/facts.py` measures them
from the cache, reproducing the organizers' published numbers -- ceiling 0.86446 vs 0.8645,
zero-positive users 27.108% vs 27.1%, item-pop 0.5709 vs 0.5715 -- and corrected one error we had
shipped: the train window starts 20220409, not the 20220408 implied by the filename.

**`evaluate()` is 1.47x faster with bit-identical output.** The agent calls it once per training
epoch, so it was ~26% of an iteration. Two exact rewrites: `lexsort` replaced by paired stable
argsorts, and a closed-form IDCG for 0/1 labels that removes an entire sort of millions of rows
-- the ideal ranking puts every positive first, so IDCG@k is a prefix sum indexed by positive
count (graded labels still take the sorting path). Verified identical on 11 cases across both
datasets, including all-tied scores, degenerate users and the submitted scores themselves.

**`--replay` re-runs a recorded run with no network and no tokens.** A full run costs ~30
minutes and real API spend, a poor way to test a change to the loop, the parsers or the
reporting. Replaying a recorded run reproduced its ledger to the last decimal at every iteration in 2.9
minutes against the original 6.3. It tests plumbing, not prompting -- a changed prompt still
receives the response recorded for the old one -- and `--replay-strict` fails the moment a prompt
diverges, so that limit cannot pass unnoticed.

## Development tools
VS Code, Python 3.12 on Windows.

## APIs used
OpenAI Chat Completions for the agent's proposer and its belief revision, via a stdlib
`urllib` client —
no SDK dependency. The submitted run used **gpt-5.6-sol** for both; later runs route belief
revision to a second model (`--revision-model`), because rate limits are per-model and revision
is ~37% of a run's requests. Every call, with its model, prompt, response and token counts, is
recorded in each run's `llm_calls.jsonl`. The interface is a single injected
`complete(prompt) -> (text, tokens_in, tokens_out)` callable, so any provider can be swapped in;
an Anthropic client ships alongside it.

## Libraries and frameworks
PyTorch (CPU), NumPy, LightGBM. Metrics, data loading, submission handling and the agent itself
are stdlib + NumPy only — no pandas, no scikit-learn.

## Datasets
KuaiRand-Pure (Zenodo 10439422), under the organizers' fixed date splits. No external training
data. Logged outcome signals (`play_time_ms`, `is_click`, `is_like`, …) are exposed only as
auxiliary targets and are asserted absent from the feature set by test;
`video_features_statistic_pure.csv` is excluded entirely because its counts are aggregated over
the whole log period, including the validation and test windows.

## Three things we tested, and what the results actually said

An agent that only reports what worked is reporting a filtered view. Each of these cost a real
measurement, and each changed what the agent does next. All three are reproducible from the run
records in `runs/`; the full working is in `docs/BUGS.md`.

### 1. Selecting on a chronological window, refuted

The hidden test window (29 Apr - 8 May) sits strictly after validation (22 - 28 Apr), so the
last validation days are the closest legal proxy for the evaluation period. The obvious move is
to rank candidates on those days instead of all seven.

We scored all 49 iterations across the diagnostic runs that saved predictions on both splits
(`research/selector_window.py`):

| selector | Spearman vs hidden test | test primary of its pick |
|---|---|---|
| full 7-day validation | **0.8659** | 0.599688 |
| last 4 days | 0.8397 | 0.599688 |
| last 3 days | 0.6637 | 0.599688 |
| last 2 days | 0.3273 | 0.599688 |

Full validation ranks iterations *more* like the hidden test than any truncation of it, and all
four selectors submit the same model. The premise had been misapplied rather than wrong: its
evidence was one run beating another on validation and losing on test — a comparison *between*
runs, where the harness only ever chooses *within* one. Shrinking the window trades a bias it
does not have for variance it cannot afford, and the proposer prompt now marks the axis refuted
so no future iteration spends a convergence life on it.

### 2. Recency weighting, adopted — but only in the right place

The binding constraint on this benchmark is drift, not capacity. Across the train-to-test gap,
users with no positive label go 5.1% -> 27.1% and the median rows per user go 31 -> 5. Measured
standalone on a boosted tree, uniform day weights score 0.4597 (random is 0.4753) against 0.5518
for a 4-day half-life.

That effect had been tried once, on a *side* component, where the blender damped it to +0.00002.
Directing the agent at the main model's `sample_weight` instead produced a later run's winning
iteration,
a recency-weighted pointwise LightGBM, and the best hidden-test score we had at that point:
**0.599904, +0.00530**. The lesson is about placement, not the technique.

### 3. A portfolio of parallel lineages — abandoned, then reinstated when the measurement was fixed

We built a portfolio search that advances *n* solution lineages per turn under one convergence
counter, with an archive, a refill policy and a cross-lineage blend. It shipped behind a go/no-go
gate: three slots are only worth their cost if they explore differently, measured as mean pairwise
rank correlation between slots. Below 0.90 proceed; above 0.95 stop.

**The first measurement said stop, and it was wrong.** Two runs read 0.94, 0.99, 0.99, 1.00 and we
concluded we had three expensive copies of one agent. Two bugs produced that number:

1. *The gate read the wrong array.* `retain_or_blend` overwrites the published scores with
   whichever of {incumbent, blend, candidate} wins on validation, so when a slot's candidate
   lost, the array recorded as "that slot's prediction" was **the incumbent**. Slots that had
   each been discarded were then compared against one another — the same array twice —
   correlating at exactly 1.0000 by construction. The blend was fed the incumbent as a candidate
   member for the same reason, which is why it declined every turn with "no member improved
   fold A".
2. *Sibling disclosure never fired.* Each slot is told what its siblings are attempting so it does
   not duplicate them. But `slot.last_hypothesis` was assigned **after** scoring, so a slot saw
   what its siblings did *last* turn — and on turn 1 saw nothing at all. Every slot opened from an
   identical prompt. The slots were not merely measured as clones; early in each run they were
   being *made* into clones.

The second bug is the causal one, and it was invisible behind the first.

**Corrected, the gate passes.** Same harness, same dataset, measured on each slot's own pre-blend
model:

Measured that way the lineages start similar — around 0.90 on the first turn, when every slot is
branching from the same baseline — and **diverge** as the run proceeds, reaching 0.2-0.4 by the
fourth turn. That is the opposite of the collapse the broken measurement showed, and the verdict
reversed: the portfolio stays. The submitted run r96 carries the same signature.

**How many lineages to run.** We swept the slot count from one to five under the current
train-only contract, one run per rung, every run converging under the organizers' rule with 0
manual interventions. Cost scales close to linearly — roughly 3x the tokens and 2.5x the
wall-clock from one slot to five — while the hidden-test spread across the whole ladder was
about 0.0004, half the baseline's own 5-seed noise of 0.0008. On that evidence the slot count
does not measurably change the score, and the run-to-run spread at a single slot count is as
large as the spread across the ladder.

We run **r96 at three slots**: it is the setting the earlier portfolio work was built and
validated around, and the ladder gives no measured reason to move off it in either direction.
What the extra lineages demonstrably do buy is decorrelation — the divergence measured above —
which is what the archive and the portfolio blend are there to exploit.

**What we would have lost.** Had we trusted the first measurement we would have deleted the
archive, the refill policy and the blend, and reported "search breadth does not help on this
benchmark" — a confident, well-evidenced, wrong conclusion drawn from an instrument nobody had
checked. The lesson is about the instrument, not the portfolio: a measurement that decides
whether to delete a subsystem deserves the same scrutiny as the subsystem.

## One thing we deliberately did not do

KuaiRand-Pure ships a randomised-exposure log (`log_random_4_22_to_5_08_pure.csv`), and the
problem statement points at it: the randomised intervention "supports counterfactual evaluation",
and the dataset notes say it "enables off-policy / counterfactual evaluation (OPE)". Debiasing
logged feedback is arguably what this dataset exists for, and our loader ignores it entirely.

We left it alone because of its date range. That file covers 22 Apr - 8 May, which is exactly
the validation and hidden-test windows. The official splits are date-based precisely to stop a
model learning from the evaluation period, and training on randomised impressions drawn from
those same days would learn item-level behaviour from the window we are scored on. No rule
names the file, which makes it ambiguous rather than permitted — and we would rather report a
smaller honest delta than a larger one that a judge might read as exploiting the split.

If the organizers confirm the random log is in scope for training, it is the single most
promising unexplored direction here, and the harness change is one loader function.

## Limitations and what we would improve

- **The delta over baseline is small and close to the noise floor** (baseline seed std 0.0008).
  We report it as measured rather than as a decisive win.
- **The search is explored one node at a time**, because scripts run sequentially on one CPU.
  Iris and Gome both run parallel traces sharing a memory; we do not, and with a 6-hour ceiling
  against a 35-minute run there is a lot of unused budget.
- **Every script gets the same time budget** regardless of what it is attempting, which biases
  the agent against methods that legitimately need longer.
- **We never validated the belief set's claims.** The agent asserts things like "these
  components correlate above 0.94"; we checked a few by hand and they held, but nothing verifies
  a claim before it is carried into the next prompt. A false belief would propagate silently —
  the same failure mode as the human priors we removed.
- **Cross-run memory is new and nearly unproven** — it deliberately reads only runs produced by
  this architecture, so at submission time it has very little history to draw on.
