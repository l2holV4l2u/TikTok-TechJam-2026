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

That trade turned out not to be a trade. The prior-laden agent reached hidden-test +0.0039; the
no-priors agent now reaches **+0.0039 as well**, from a higher validation score (0.6049 against
0.6038). Deleting our answers cost nothing in the end.

**And that guard was not enough.** After removing the findings from the brief we went looking
for what else fed the prompt, and found our own measurements sitting in the knowledge base's
`expected_effect` fields — injected on every improve iteration. The `rank_aggregation` entry
spelled out the winning recipe, its 0.6/0.3/0.1 weights, and the rank-transform trick. Fifteen
of twenty-eight entries were contaminated. We had the worst of both worlds: the claim was false
in principle, and useless in practice, because retrieval never surfaced those entries anyway.
The entries now say what each paper claims and nothing about what we measured
(`archive/papers_v1_human_priors.json` keeps the originals), and a second test,
`test_knowledge_base_carries_no_measured_results`, guards the channel the first test missed.
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
the *earliest* node, on the reasoning that it would reuse working plumbing. In r34 the agent
proposed recency weighting — the right idea for this dataset's actual constraint — and its own
internal sweep showed it was worth +0.0005 over not doing it. But because broaden had anchored
it to the plain baseline instead of the leading DeepFM, the whole iteration scored 0.6034
against a 0.6043 leader and was recorded as a loss. Breadth belongs in method space, not in
which file you start from.

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

How often revision actually fires is worth stating rather than implying: across 8 runs the agent
formed **45 claims and revised 4** (3 invalidated, 1 qualified). That is not a weak reflect
stage — it is the convergence rule. At a median of **4 improve iterations per run**, a claim
formed at iteration #2 gets one or two chances to be contradicted before the run ends. The
revisions that do happen are specific and evidence-cited, e.g. *"does not improve the k=16 FM;
it scored 0.5998, which is 0.0026 below the unweighted reproduction and outside the stated
seed-noise threshold"*, and one narrows a claim rather than deleting it.

Two smaller consequences of the same reading:

- **One iteration is one script, not one model.** A script may build and compare several
  candidates within its time budget and report what it compared (`CANDIDATES`). The convergence
  rule charges an iteration per experiment, so searching *inside* an iteration buys comparisons
  the iteration budget cannot.

  We checked whether searching harder inside an iteration actually pays, and it does not:
  across nine Pure runs the candidate count ranges 0-96, and runs above the mean average 0.6046
  best-validation against 0.6041 below it -- a 0.0005 gap on n=3 against n=6, inside the noise
  floor. The run that compared the most candidates (96) scored *worst* of the recent set
  (0.6036); the best run (0.6059) compared ten. The mechanism gives the agent room the
  iteration budget denies it, and it is used, but we have no evidence it drives the result and
  do not claim it does.
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
  "Implementing a LambdaRank objective" does not evade retirement. Sentence-level Jaccard, which
  we tried first, read those two as different ideas.
- **Timeouts are fed back differently from crashes** ("too slow, not wrong") because rerunning
  a slow approach unchanged just times out again.
- Timed-out scripts are killed **process-tree-wide**. A bare `kill()` left orphaned trainers
  running for an hour, competing for CPU and corrupting the reported compute.
- LLM outages and rate limits are absorbed with server-hinted backoff; the loop survives them.
- A circuit breaker halts with `environment_broken` after five consecutive instant, output-less
  failures. We added this after a real incident: a torn-down parent left the runner unable to
  spawn children, and 32 iterations "failed" in 0 seconds each, silently shredding the budget.

### What we learned about steering an agent

Most of this project's engineering went into the agent's *scaffolding*, not its reasoning. Every
recurring failure traced to a defect in what we had built around the model, and each fix showed
up in the failure rate: ~50% → 35% → 27% → 0%.

- It called `.cuda()` repeatedly because our brief claimed a GPU was available when torch was
  CPU-only. We had written a false statement into the prompt.
- It crashed on `IndexError` five times building per-video tables, because our own helper
  applied field offsets and nothing said raw ids were needed for lookups.
- It spent 8 of 17 iterations on one losing idea, because retirement only triggered on crashes;
  an idea that runs cleanly and scores badly was never excluded.
- Our retirement rule then killed the iteration that scored the run's **best** result, because
  "did not beat the incumbent by ε" counted as an underperformance.
- Generated blend scripts arrived truncated with `SyntaxError: '(' was never closed` — our
  `max_tokens` was 4096 and we were cutting the model off mid-script.

The subtlest one: after we added a section telling the agent that ensembling was the proven
direction, it still proposed only single models, six times in a row. The retrieved-papers block
directly beneath that section was seeded with query terms like "click", "ranking", "ndcg" — so
it surfaced only single-model papers, every call. Two halves of the same prompt disagreed and
the retrieval half won.

Rebuilding the agent around search and a belief set produced more of exactly the same
kind, and we only found them by reading the logs:

- **The analysis stage and the proposer disagreed.** Every reflection in our first run ended
  "next, run matched-seed leave-one-field-out ablations" — and the proposer never ran one, four
  times in a row. The reflection prompt asked for the *most informative* experiment; the run
  ends after three iterations without a **gain**. Two halves of the same system optimising
  different objectives, which is the v1 retrieval bug wearing a different hat.
- **Retrieval tunnel vision came back.** The paper retriever returned the same three entries on
  every single iteration, because the query is built from the agent's own recent hypotheses — so
  a run that opens on factorization machines is only ever shown factorization machines. Four
  ensembling papers sat in the knowledge base, unreachable. In v1 we had "fixed" this by pinning
  a chosen paper, which is just steering. The real fix is that a retriever returning the same
  three books every visit is broken: it now excludes what it has already surfaced, and the agent
  is given the full catalogue and left to decide.
- **The agent did not know how the run ends.** It was spending early iterations on cautious
  ablations, each of which burned one of the three lives the convergence rule allows. It now
  gets the stopping rule and its live budget state. This is task specification straight out of
  the organizers' document, not a hint about the dataset.
- **A completed run was killed by its own success message.** A hypothesis contained `×`
  (U+00D7); redirected stdout on this machine defaults to the system codepage; the harness
  raised `UnicodeEncodeError` while *printing the result*, after the loop had converged and
  written its submission, destroying the run metadata. The agent's own output is untrusted text
  and must never be able to crash the harness that reports on it.
- **A retry policy that could not tell two rate limits apart.** A per-minute limit clears if you
  wait; a per-day cap does not. Ours treated both as transient and spent twelve minutes climbing
  a backoff ladder against a wall, then failed anyway. The client now recognises a daily cap
  immediately and fails over to another key, marking the capped one so later calls skip it.

None of these were visible from the score. They were only visible by reading what the agent
actually chose to do, which is what the per-iteration ledger exists for. The pattern across both
versions is the same: **every defect was in the scaffolding, not in the model's reasoning.**

## Results

Official baseline (organizer-provided FM, k=16): validation primary 0.6016, hidden test 0.5946.

Our submitted run (`runs/r41`, full log in `RUN_REPORT.md`):

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| validation, agent's best iteration | 0.6732 | 0.5386 | 0.6059 |
| official baseline, validation | 0.6674 | 0.5357 | 0.6016 |
| **hidden test, this submission** | **0.6677** | **0.5316** | **0.5996** |
| official baseline, hidden test | 0.6610 | 0.5282 | 0.5946 |

**Absolute delta on hidden test: GAUC +0.0067, nDCG@5 +0.0034, mean +0.0050.**

r41 is validation-best of every run, which is the only criterion we select on. r39 reached a
marginally higher hidden-test score (0.5997, +0.0051) from a lower validation score (0.6053) --
we do not take it, because choosing the higher test number would be selecting on the split we
are scored against.

Both are runs with two instruments the harness previously withheld -- impression timestamps and
a continuous-feature channel -- and both winning iterations use them. Before them
the best run reached 0.6049 validation / +0.0039 test (`runs/r35`), and five runs had clustered
there. Details in **Harness engineering** below.

One thing stated plainly: r39 did not converge on its own. The machine it ran on was suspended
mid-run, which left the proposer's HTTP socket dead, and we terminated the process during
iteration #7. Iteration #6 -- the one selected -- had completed and been scored normally before
that, and the submission was written by the harness's own validation-best selector, not rebuilt
by hand. The incident is also why `agent/llm.py` now bounds the total time of a single call: a
run must not be able to hang on one dead socket.

Resources for the submitted run (`r41`): **11 iterations of 50**, **7.1 minutes** of script
time, **54,029 tokens** over 8 model calls, **0 GPU-hours** (CPU only), **0 manual
interventions**, and **0 failures of its own code**.

It did lose 6 iterations to an expired API key — an outage, not the agent failing an experiment,
and `report_run.py` now separates the two rather than reporting "6 failures" against the agent.
The best iteration (#4) was reached before the key died.

The agent did all of it: it wrote its own EDA script, reproduced the official baseline to 0.6015
on the first attempt, proposed and coded further experiments, and emitted the test
predictions that became the submission.

### The result that matters most

We deleted our own best finding from the agent's prompt — that rank-blending *decorrelated*
models beats any single model — and the agent derived it again, unaided, and then beat us with
it.

Iteration #5's hypothesis, in its own words:

> "replacing two redundant DeepFM seeds with low-rank DCN-V2 cross models; their explicit
> bounded-order feature crosses should create **less-correlated** user–item ordering errors that
> heterogeneous rank aggregation can **cancel**."

That is the decorrelation argument from the deleted `EXHIBIT A`, reconstructed from first
principles. It then compared fourteen aggregation schemes inside that one iteration — Borda
versus logit-mean, homogeneous versus heterogeneous, per-seed — and its belief set went on to
record the correlation measurements that explain the ceiling it hit:

> "The tested DeepFM seeds, low-rank DCN-V2 models, per-user-weighted DeepFM models, and MMoE
> model are too prediction-correlated to yield a measurable ensemble gain; observed rank
> correlations were generally about 0.94 or higher, including MMoE correlations of 0.9888 with
> standard DeepFM."

We ran that same correlation study by hand in the earlier version of this project and reached
the same conclusion. The difference is that this time nobody told it to.

**The scores make the point sharply.** Our hand-tuned blend reached validation 0.6045. The
earlier agent, *with the winning recipe written into its prompt*, reached 0.6046. This agent,
with the recipe deleted and a test that fails the build if it creeps back, reached **0.6049**,
and its hidden-test delta (+0.0039) matches the prior-laden version's exactly. Removing our
answers did not cost us score; it cost us nothing and produced a better one.

### What the architecture changes actually did

We ran the same architecture repeatedly, changing one thing at a time. Every run converged in 5
iterations with zero failures and zero interventions:

| run | change under test | validation best | hidden-test delta |
|---|---|---|---|
| r27 | no priors in the brief | 0.6033 | +0.0024 |
| r28 | + agent told the convergence rule | 0.6027 | +0.0013 |
| r29 | + budget-aware analysis, paper catalogue, retrieval rotation, noise-band memory | 0.6023 | +0.0013 |
| r30 | + "one iteration is one script, not one model" | 0.6030 | +0.0006 |
| **r33** | **+ belief set, adaptive breadth, internal candidates, decontaminated KB** | **0.6044** | **+0.0035** |
| **r34** | **+ `Split.date` exposed to the agent** | **0.6043** | **+0.0033** |
| **r35** | **+ broaden anchors on the best node, not the earliest** | **0.6049** | **+0.0039** |
| **r36** | **+ `evaluate(per_user=True)` for per-segment diagnosis** | **0.6037** | **+0.0036** |
| **r37** | **+ `RUN_ARTIFACTS` cache directory (agent never used it)** | **0.6037** | **+0.0041** |
| **r39** | **+ `s.time_ms` and `s.num`: impression order and 22 continuous features** | **0.6053** | **+0.0051** |
| **r41** | **+ stale-capability note in cross-run memory** | **0.6059** | **+0.0050** |
| r43 | + all 51 video statistics, GPU visible — *proposer swapped to gpt-5.6-luna* | 0.6042 | +0.0023 |
| r44 | same harness — *proposer swapped to gpt-5.6-terra* | 0.6020 | +0.0014 |

r30's `run_meta.json` was destroyed by an encoding crash *after* it had written its submission
(a hypothesis containing `×` met a cp874 stdout). Its row is therefore re-derived by scoring
that submission directly — test primary 0.5952, delta +0.0006 — rather than read from metadata.
`research/verify_claims.py` re-checks every row in this table against the run records and exits
non-zero on any disagreement.

r27–r30 are the honest negative result: **none of those four scaffolding fixes moved the score**,
and the spread across them (0.6023–0.6033) is the size of the baseline's own seed noise. We
report that rather than quietly dropping it.

r33–r37 carry the literature-driven changes, and the separation is clean on both metrics:

|  | validation | hidden-test delta |
|---|---|---|
| before (r27–r30) | 0.6023 – 0.6033 | +0.0006 – +0.0024 |
| after (r33–r37) | **0.6037 – 0.6049** | **+0.0033 – +0.0041** |
| with the new instruments (r39) | **0.6053** | **+0.0051** |

### The proposer model is the largest single effect we measured

We swapped the proposer between three models of the same family, holding the harness fixed:

| proposer | runs | hidden-test delta |
|---|---|---|
| `gpt-5.6-sol` | r37, r39, r40, r41 | +0.0041, **+0.0051**, +0.0030, **+0.0050** (mean +0.0043) |
| `gpt-5.6-luna` | r43 | +0.0023 |
| `gpt-5.6-terra` | r44 | +0.0014 |

That spread is **larger than any harness change in this table**, which is worth stating plainly
given how much of this project is harness work. Two caveats keep it from being a clean
experiment: luna and terra are single runs each, and both also carried later harness states (all
51 statistics, a visible GPU), so model and harness are not fully separated.

One concrete failure mode was visible rather than statistical. `gpt-5.1`, tried briefly, spent
two iterations inventing field names — `duration_ms_range`, then `duration_range`, where the real
field is `duration_bucket`. That exposed a genuine gap: the brief said `s.X` held 37 categorical
features and named only the five baseline ones, so a model wanting a sixth had to guess. All 37
are now listed. The weaker model found a hole the stronger ones had been stepping around.

**The worst run after beats the best run before, on both metrics, 5/5 against 4/4.** Five runs
is still a small sample and we are reading a ~0.0015 effect against a 0.0008 noise floor, so we
report the ranges rather than a mean and a p-value. What is not in doubt is the direction.

The cost side is worth stating too: r35 took 35 minutes and 92K tokens against r27's 8 minutes
and 37K. The gain is real but it is not free, and a judge weighing Feasibility should see both
numbers. Both remain far inside the organizers' 6-hour ceiling and 50-iteration cap.

### Why we do not run a selection lottery

The five post-change runs separate cleanly from the four before them, but they do **not**
separate from each other. Across r33–r37:

| | spread | std | vs baseline seed noise (0.0008) |
|---|---|---|---|
| validation | 0.0012 | 0.00051 | below |
| hidden test | 0.0008 | 0.00032 | below |

Both are under the noise floor, and ranking the runs against each other is a coin flip: of the
ten run pairs, **4 are concordant, 5 discordant, 1 tied**. Validation carries no usable
information about which converged run will do better on the hidden test.

This matters because there is an obvious way to inflate a score here — run the agent twenty
times and submit whichever run happens to peak. The numbers above say that would be sampling
noise, not selecting quality, and it is exactly the failure mode the held-out test exists to
catch. We submit the validation-best run under the organizers' rule and report the spread.

r37 is the concrete case. It scored **+0.0041 on the hidden test, the best of any run** — and we
do not submit it, because its validation score (0.6037) is not the best and selecting it would
mean choosing on the test set. r39 (validation 0.6053, test +0.0051) is the submission, selected the same way.

What is *not* ambiguous is the mechanism, and the clearest single example is r33, where broaden
fired for the first time at iteration #3 and that iteration is the one that won. Instead of
tuning latent dimension again, the agent changed direction to a DeepFM and used the candidate
contract to sweep the FM/deep mixing weight inside one iteration —

| α | 0.00 | 0.25 | 0.50 | **0.75** | 1.00 |
|---|---|---|---|---|---|
| validation primary | 0.6019 | 0.6033 | 0.6041 | **0.6044** | 0.6040 |

Five models compared for one iteration of budget. It also missed resetting the convergence
counter by 0.0001 (0.6044 against a 0.6045 threshold), a fair illustration of how tight the
ε = 0.002 rule is here.

Both mechanisms are auditable in the run logs. In r33's `llm_calls.jsonl` the search mode goes
`refine → broaden → broaden` across the three improve iterations while the belief set carries
1 → 2 → 3 claims into successive prompts, so each proposal is conditioned on a revised reading
of everything before it rather than on a raw score history. The submitted run, r41, traces
0.6015 → 0.6007 → 0.6053 → 0.6059 across its scored iterations: the baseline reproduction, one
experiment that lost ground, then two that built on what the belief set had recorded.

The submitted run is chosen on **validation** — r41 is the validation-best of all no-priors runs
(0.6059, against r39's 0.6053 and r35's 0.6049).
We never used the hidden-test column to choose between runs; it is shown only because we hold
the public test labels and would rather report it than hide it.

**We stop the run where the rule says the run is over.** The scored result is "the
validation-best checkpoint at [the convergence] point", and convergence is the *first* of
ε/N = 0.002/3, the 50-iteration cap, or 6 hours. So iterations that happen after the ε/N rule
has already fired cannot legitimately supply the submission, no matter what they score. An
earlier run of ours used a looser patience and produced its best number six iterations past its
own convergence point; truncating it at the rule cost 0.0003 on test, and we truncated it. The
agent's default `--patience 3` now makes this correct by construction, and the ledger shows
exactly where the rule fired.

Selection is on validation only. We hold the public test labels and therefore *can* see each
iteration's test score, but choosing between iterations on that basis would be fitting the
hidden set, so the submission is always the validation-best checkpoint regardless of what the
test number says. In one run this cost us: the validation-best iteration (0.6042) scored
slightly worse on test (+0.0037) than an earlier run's validation-best (0.6039 → +0.0040).
Differences that small are below the baseline's own 5-seed σ of 0.0008 and we report them as
noise, not as a result.

We verified `pipeline/evaluate.py` is bit-identical to the organizers' `evaluate.py`
(max abs diff 1.7e-14) and ~7× faster, and that our row order matches their loader exactly
(170,588/170,588 on user_id, video_id and label) so `row_id` alignment is correct.

### Why the deltas on this benchmark are small — measured, not assumed

Across this project, roughly fifteen distinct approaches all landed within ±0.005 of the
baseline: FM, DeepFM, DCN, NFM, LightGBM pointwise, LambdaMART, pairwise BPR, listwise softmax,
target encodings, multi-task heads, and rank blends of those. That needs an explanation, and
the three candidates imply completely different strategies: models too small, features too
weak, or signal that does not survive the date split.

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

So capacity is not the constraint; transfer across the date boundary is. Train is 9–21 Apr,
validation the following week, and the agent's own EDA independently measured how far the
distribution moves: zero-positive users 5.1% → 30.3%, median rows per user 59 → 7.

Three consequences we act on:

1. **The baseline's small k=16 FM is not a weak starting point** — it is close to the right
   amount of capacity for a signal this non-stationary. That is why bigger models reliably lose.
2. **The realistic ceiling is ~0.60–0.61 test primary**, not the 0.8645 oracle. The oracle
   assumes knowledge of the labels; it bounds the metric, not the achievable score. On a
   benchmark scored by absolute delta, a few thousandths is a real result.
3. **The methods with a mechanism here target drift, not capacity.** That is why we exposed
   `Split.date` to the agent — our harness had been withholding the impression date, silently
   ruling out recency weighting and time-based validation, the family of methods aimed at the
   one thing that actually binds. Full write-up in `docs/generalisation-ceiling.md`.

This probe is human analysis, clearly labelled and **not** on the submission path. It is context
for reading the agent's delta, not one of the agent's findings.

## Bonus dataset: KuaiRand-1K

We ran the same agent, unchanged, on KuaiRand-1K. The point was never to show our Pure model
scores well there — it would not — but to ask whether **the agent** adapts when the problem
changes underneath it.

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
catalogue, so item identity is the dominant signal. 1K logs 1,000 users against the full
catalogue at 2.4 impressions per video, so for five test rows in six the item embedding is an
untrained row. Item popularity, the most dependable signal in the field, earns barely half as
much.

**Result.** Converged in 14 iterations, 2.7 h, 2 failures, **0 manual interventions**, 192,900
tokens, 76 internally-evaluated candidates, 14 belief-set claims.

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| reference (organizers' recipe, our run) | 0.6704 | 0.6006 | 0.6355 |
| agent, iteration #8 | 0.6959 | 0.6595 | **0.6777** |
| delta | +0.0255 | +0.0589 | **+0.0422** |

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
drew the right conclusion — when item identity is untrainable, substitute item content. That is
a different architecture from the one it converged on for Pure, which was rank aggregation over
redundant crosses. Neither was suggested to it.

**Robustness, unscripted.** Iteration #9 died on a hard LightGBM limit — `Number of rows 13924
exceeds upper limit of 10000 for a query` — a wall that only exists because 1K's users are dense
enough to overflow a lambdarank query group; on Pure, at 7 impressions per user, it is
unreachable. The agent recovered in one attempt by splitting oversized users into bounded
chunks, and added its own invariants:

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
| `s.num` — 22 continuous features | data | **63.0%** (17/27) |
| `s.time_ms` — impression order | data | **33.3%** (9/27) |
| `s.date` — impression day | data | **29.2%** (19/65) |
| `FINDINGS` — report epistemic evidence | process | 9.8% (38/387) |
| `CANDIDATES` — compare inside an iteration | process | 8.5% (33/387) |
| `RUN_ARTIFACTS` — cache between iterations | process | 1.0% (4/387) |
| `evaluate(per_user=True)` — segment diagnosis | process | 0.8% (3/387) |

**Give the agent new data and it uses it; give it new process and it mostly does not.** Every
data channel we exposed was picked up within an iteration or two of becoming available, and the
winning iteration of the submitted run uses two of them. Every optional protocol we invented
sat near the floor, including two we were confident about: a per-user error breakdown it asked
for nothing to obtain, and a cache directory that would have saved it retraining the same model
each iteration.

This is the most useful thing we learned about building the harness, and it was not what we
expected. It also cost us: `RUN_ARTIFACTS` and `per_user` were built early, reported as
capabilities, and did nothing for six runs. We report the adoption numbers rather than the
feature list, because a capability with 0.8% uptake is a capability we should not claim.

One caveat on causation: `s.num` arrived alongside a knowledge-base gap being filled (the KB had
28 entries and zero on numeric features until we added five), so its uptake is not attributable
to exposure alone. The process mechanisms had contracts in the brief from the start and no
comparable retrieval support, which may be part of why they lagged.

## The GPU does not help this benchmark

An RTX 4050 was available and unused, so we measured what it was worth rather than assuming.

| | result |
|---|---|
| raw 4096² matmul ×20 | **16.2× faster** on GPU (0.74 s vs 11.90 s) |
| the organizers' FM recipe, end to end | **60 s on GPU vs 56 s on CPU** |
| a real agent iteration, CUDA runtime merely *present* | **131.5 s vs 108.1 s**, bit-identical output |

The benchmark is not matmul-limited. These models are small and embedding-lookup bound, and the
CUDA runtime costs about 20 seconds per script to load — which a 100-second script cannot earn
back. Simply having the CUDA build on the path made an unchanged script 22% slower while
producing exactly the same metric, because the script never moved anything to the device.

Two consequences we acted on:

- The brief now states the device and **the load cost**, so the agent can judge whether moving
  work earns it back instead of assuming a GPU is free. It does use it: the first two scored
  iterations of the following run wrote CUDA code.
- `research/baseline_reference.py` is **pinned to CPU**. The same recipe returns valid 0.6020 /
  test 0.5947 on GPU against 0.6022 / 0.5957 on CPU — a 0.0010 shift on test, at the edge of the
  0.0008 seed-noise band. A reference number that moves with whichever device happens to be
  installed is not a reference, and this one anchors both the ablation table and the entire
  KuaiRand-1K experiment.

The organizers note that compute is deliberately not the binding constraint here — their
baseline runs in about 40 s on one CPU core. Our measurements agree: there is nothing for a GPU
to win on this task.

## Harness engineering

Four changes that are not model work but decide whether the agent gets a fair run.

**The provided models were mis-initialised.** `pipeline/models.py` built every embedding table
with PyTorch's default `N(0,1)`. Summed across fields an FM's interaction term starts orders of
magnitude too large and converges to a much worse optimum: measured on Pure, **0.5533 valid
after 40 epochs against 0.6020 in under 15**. Fixed for all four architectures, with a
regression test that asserts initial logit magnitude rather than running a slow convergence
check. No run was affected -- 0 of 333 agent scripts import that module -- and we say so
rather than claiming a score we did not lose.

**The brief stated dataset facts as literals.** Row counts, the perfect-ranking ceiling, the
random and item-popularity rungs were Pure constants typed in by hand, which would have fed the
agent false premises the moment the harness pointed elsewhere. `agent/facts.py` measures them
from the cache. It reproduces the organizers' published numbers -- ceiling 0.86446 vs 0.8645,
zero-positive users 27.108% vs 27.1%, item-pop 0.5709 vs 0.5715 -- and corrected one error we
had shipped: the train window starts 20220409, not the 20220408 implied by the filename.

**`evaluate()` is 1.47x faster with bit-identical output.** The agent calls it once per training
epoch, so it was ~26% of an iteration. Two exact rewrites: `lexsort` replaced by paired stable
argsorts (equivalent by sort stability), and a closed-form IDCG for 0/1 labels that removes an
entire sort of millions of rows -- the ideal ranking puts every positive first, so IDCG@k is a
prefix sum indexed by positive count. Graded labels still take the sorting path. Verified
identical on 11 cases across both datasets including all-tied scores, graded labels, degenerate
users, and the submitted scores themselves.

**`--replay` re-runs a recorded run with no network and no tokens.** A full run costs ~30
minutes and real API spend, which makes it a poor way to find out whether a change to the loop,
the parsers or the reporting works. Replaying r30 reproduced its ledger to the last decimal at
every iteration in 2.9 minutes against the original 6.3. It tests plumbing, not prompting -- a
changed prompt still receives the response recorded for the old one -- and `--replay-strict`
fails the moment a prompt diverges so that limit cannot pass unnoticed.

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
  components correlate above 0.94"; we checked a few by hand and they held, but nothing in the
  harness verifies a claim before it is carried into the next prompt. A false belief would
  propagate silently — the same failure mode as a human prior, which is what we removed.
- **Three runs is a small sample.** The separation between the old and new architectures has no
  overlap, but we are reading a ~0.002 effect against a 0.0008 noise floor from n=3.
- **Cross-run memory is new and nearly unproven** — it deliberately reads only runs produced by
  this architecture, so at submission time it has very little history to draw on.
