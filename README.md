# Autonomous ML Research Agent — KuaiRand-Pure

TikTok TechJam 2026, Track 2.

An LLM-driven agent that runs the MLE iteration loop of the problem statement's Figure 1
unattended. It inspects the data, reproduces the official baseline, then repeatedly proposes a
hypothesis, writes the code, trains, evaluates, **revises what it believes**, and decides what to
try next — searching over a tree of solution scripts until validation converges.

Nothing in the agent's prompt tells it what works on this dataset. That is the point: what to
try, and why, has to come from the agent, or the Autonomy and Innovation claims are not real.

## Result

`submission_best.csv` is the submission. It is `runs/r35/submission.csv`, written by the harness
from that run's validation-best iteration — no human rebuilt it, and selection never looked at
the test split.

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| official baseline, hidden test | 0.6610 | 0.5282 | 0.59460 |
| **this submission, hidden test** | **0.6657** | **0.5312** | **0.59847** |
| delta | +0.0047 | +0.0030 | **+0.00387** |

Validation 0.6049 against the baseline's 0.6016. Test always runs below validation on this
benchmark — a later date window — and the baseline shows the same drop.

Some later runs scored higher, but they read the supplied `video_features_statistic` fields,
whose full-month aggregates overlap the validation and test windows. Those runs are excluded from
the submission, from cross-run memory and from every comparison in the write-up.

`python -m research.verify_claims` re-derives every score cited in DEVPOST.md from the run
records and exits non-zero on any disagreement.

Bonus benchmark: on KuaiRand-1K the same agent reached **+0.0422** over a reference we measured
ourselves, since no official baseline exists for it (`research/transfer-1k.md`).

## The loop

```
                 ┌──────────────────────────────────────────────────────┐
                 ▼                                                      │
 inspect data ─► reproduce baseline ─► select node ─► propose ─► execute ─► evaluate ─► revise
   (agent's        (Requirement 1)     (adaptive:     (LLM edits  (sandbox,   (GAUC/    beliefs
    own EDA)                            refine or      a script)   timeout)   nDCG@5)    (LLM)
                                        broaden)                                           │
                                            ▲                                              │
                                            └────── belief set guides the next choice ──────┘
```

- **inspect data** — the agent writes and runs its own EDA script. What it prints is the only
  dataset knowledge it ever has; it is carried into every later prompt (`runs/<id>/eda_report.txt`).
- **reproduce baseline** — it stands up an end-to-end FM pipeline itself and is checked against
  the organizers' published validation primary of 0.6016. That script becomes the search root.
- **select node** — **adaptive**. While the last iteration gained, the search refines the leader;
  once it stalls it *broadens* — same base script, but the proposer is told to change direction
  rather than detail, with everything already tried listed. A node whose children keep failing
  to improve on it is retired, and the next-best becomes the anchor.
- **propose** — the LLM receives the chosen node's *actual source*. One iteration is one script,
  not one model: a script may build and compare several candidates internally and report what it
  compared (`CANDIDATES`), and may report evidence that is not its score (`FINDINGS`).
- **revise beliefs** — after every scored iteration the agent rewrites its **belief set**: claims
  with evidence and a status of `active` / `qualified` / `invalidated`. Later evidence can demote
  an earlier conclusion instead of piling up beside it, and that set — not a human-authored
  brief — is what guides the next proposal (`runs/<id>/knowledge.md`).

The search policy and the belief set follow the current literature rather than our intuition:
[FML-bench](https://arxiv.org/abs/2605.17373) finds an agent that broadens on stagnation beats
every fixed strategy; [Iris](https://arxiv.org/abs/2608.02143) centres a continually revised
information state and loses the most any-medal rate of any ablation when it is removed;
[Gome](https://arxiv.org/abs/2603.01692) shows exhaustive tree search loses to directed updates
at frontier model strength.

## Layout

```
agent/       the product. dataset-agnostic.
  loop.py        controller: the three phases above, wall-clock and iteration budgets
  tree.py        search over solution scripts: adaptive refine/broaden, retirement
  proposer.py    prompt construction; the brief is task spec + API only, no findings
  critic.py      integrity gate; rejected scores cannot enter the tree or submission
  ensemble.py    controller-owned validation-selected blend against the incumbent
  hardware.py    reproducible CPU/CUDA discovery for prompts and run metadata
  knowledge.py   the revisable belief set -- the reflect+revise stage of Figure 1
  memory.py      distils this agent's own prior run ledgers into the next run's context
  llm.py         stdlib LLM client (Anthropic / OpenAI-compatible), records every call
  kb.py          keyword retrieval over kb/papers.json
  recovery.py    retry <=2 then retire the idea; failures never reach a human
  ledger.py      append-only JSONL, one record per iteration (graded deliverable)
  executor.py    sandboxed subprocess with timeout and process-tree kill
  facts.py       measures the brief's dataset facts from the cache, never hand-written
  diagnose.py    per-segment error profile fed back so proposals target a measured weakness
pipeline/    the sandbox the agent works in.
  data.py        KuaiRand-Pure loader, organizer-fixed date splits, train-only vocabs
  history.py     leakage-safe train-only item/author histories, leave-one-out on train
  evaluate.py    GAUC / nDCG@5, verified bit-identical to the official evaluate.py
  baseline_fm.py numpy Factorization Machine — the official baseline, for reference
  submit.py      submission writer + validator
  models.py      reference FM/DeepFM/DCNv2/DIN — human-side only, never named to the agent
  train.py       manual trainer for those reference models
  synth.py       synthetic cache so the tests run without the real data
kb/papers.json   methods with literature-only descriptions, retrieved to ground proposals
research/    human analysis, clearly off the submission path.
  baseline_reference.py  runs the organizers' recipe on any variant to anchor a run
  ceiling_probe.py       in-sample vs validation probe behind the generalisation ceiling
  verify_claims.py       re-checks every DEVPOST score against the run records
run_agent.py     drives a run
report_run.py    renders the Run & Iteration Logs deliverable
runs/            per-run ledger, scripts, EDA report, belief set, search tree, candidates
```

## Setup

```bash
# data (194MB, public, no account needed)
mkdir -p data/raw && cd data/raw
curl -LO https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar -xzf KuaiRand-Pure.tar.gz && cd ../..

python -c "import pipeline.data as d; d.build_cache('data/raw/KuaiRand-Pure/data','data/cache')"
export OPENAI_API_KEY=...   # and LLM_PROVIDER=openai, or ANTHROPIC_API_KEY
```

### GPU and fast prototypes

Use the CUDA-enabled Python environment when prototyping PyTorch models:

```powershell
.\.venv-gpu\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.\.venv-gpu\Scripts\python.exe run_agent.py --run-dir runs/rN --device cuda --iters 50
```

`--device cuda` opts in; `--device auto` selects CUDA when the active Python can use it. CPU is
the default because the measured reference FM was 56 s on CPU versus 60 s on this RTX 4050, and
loading CUDA costs about 20 s per generated script. Generated scripts receive the resolved choice in `AGENT_DEVICE`, and
`run_meta.json` records the PyTorch/CUDA versions and GPU. CUDA helps neural training, but data
loading, NumPy evaluation, and this LightGBM build remain CPU work, so use short dry runs and
small in-script samples for architecture smoke tests before paying for a full autonomous run.

## Reproduce our result

```bash
python -m pipeline.baseline_fm                        # the official baseline, for reference
python run_agent.py --run-dir runs/rN --iters 50 --timeout 900 --patience 3
python report_run.py runs/rN > RUN_REPORT.md          # iteration log, resources, autonomy
python -m pipeline.submit --check --in runs/rN/submission.csv --split test
```

`--iters 50` is the organizers' hard cap; the run normally stops earlier on the convergence
rule (ε = 0.002 over N = 3 iterations). A 6-hour wall-clock backstop is enforced in the loop.

`python run_agent.py --dry-run` exercises the whole loop with a canned LLM and no network.

Other flags: `--epsilon` and `--patience` set the convergence rule (organizer values are the
defaults), `--wall-clock-s` the 6-hour backstop, `--max-retries` how often a failed script is
retried before its idea is retired, `--max-misses` how many non-improving children retire a
search node, and `--revision-model` routes belief revision to a second model — rate limits are
per-model and revision is ~37% of a run's requests, so a second model both halves the pressure
on the proposer's quota and costs less for a summarising task.

`--replay runs/rN/llm_calls.jsonl` re-runs a previous run against its recorded responses: no
network, no tokens, deterministic. Replaying r30 reproduces its ledger to the last decimal at
every iteration in 2.9 minutes against the original 6.3. It tests the loop, the parsers, the
ledger and the reporting -- **not** prompts, since a changed prompt still receives the response
recorded for the old one. `--replay-strict` fails the moment a prompt diverges, so that
limitation cannot pass unnoticed.

### Running on another KuaiRand variant

The dataset facts in the task brief -- row counts, date windows, the perfect-ranking ceiling,
the random and item-popularity rungs -- are measured from the cache by `agent.facts`, not
written into the prompt by hand, so the harness can be pointed at another release without
feeding the agent false premises. `KUAIRAND_VARIANT` selects which raw files the loader reads.

```bash
export KUAIRAND_VARIANT=1k KUAIRAND_CACHE_DIR=data/cache_1k
python -c "from pipeline.data import build_cache; build_cache('data/raw/KuaiRand-1K/data','data/cache_1k')"
python -m research.baseline_reference --epochs 10 --out research/reference_1k.json
python -m agent.facts --baseline research/reference_1k.json --out research/facts_1k.json --variant KuaiRand-1K
python run_agent.py --run-dir runs/rN_1k --facts research/facts_1k.json     --baseline-valid 0.6422 --baseline-test 0.6355 --timeout 3000 --no-memory
```

Only KuaiRand-Pure has a published baseline. On any other variant the anchor is our own run of
the organizers' recipe (`research/baseline_reference.py`), which reproduces Pure's published
0.6016 / 0.5946 as 0.6022 / 0.5957 -- that agreement is what makes it usable elsewhere. Scores
are **not** comparable between variants; see `research/transfer-1k.md`. Cross-run memory is
disabled with `--no-memory`, since it ranks prior runs against a baseline on a different scale.

The agent writes the submission from its own validation-best iteration — every generated script
saves both validation and test scores. The controller recomputes the official validation metrics
from `scores_valid.npy`, rejects mismatched or invalid output, and assembles the winner. No human
rebuilds it, and selection is on validation only; the hidden test set never picks the winner.

## Tests

```bash
for m in agent.tree agent.knowledge agent.memory agent.demo \
         tests.test_proposer tests.test_llm tests.test_evaluate tests.test_submit \
         tests.test_data tests.test_models tests.test_weights tests.test_executor \
         tests.test_harness tests.test_history tests.test_ensemble tests.test_hardware; \
         do python -m $m; done
```

Two tests guard the whole design, and they cover different channels into the prompt:
`test_brief_carries_no_human_findings` fails if a measured finding is written back into the
brief, and `test_knowledge_base_carries_no_measured_results` fails if one is written into a
paper entry. We added the second only after discovering that the first had been missing a leak —
our own results were sitting in the knowledge base the whole time.

## Design notes

**Cost is a scored criterion, so the LLM stays out of the training loop.** It is called twice
per iteration — once to propose, once to revise the belief set. Everything else is deterministic Python.
History is summarised to a fixed window, so token use per iteration stays flat as a run grows.

**Feasibility is scored on agent wall-clock**, not GPU-hours (the organizers changed this on
27 Aug). `run_meta.json` records true elapsed wall-clock, not the sum of script runtimes.

**Autonomy is measured by intervention count, so failures are handled in-loop.** A crashed
script's traceback goes back to the proposer with its own source, retried at most twice, then
the idea is retired. Ideas are keyed by method name, so a reworded repeat cannot evade
retirement. Timeouts are fed back differently from crashes — a timeout is not a bug to fix.
The loop tolerates LLM outages and rate limits, and halts only if the environment itself is
broken. It never escalates to a human.

**The agent gets instruments, not answers.** Two things it was missing turned out to matter more
than any prompt wording. It could not see the impression date, so recency weighting and
time-based validation were silently impossible — the family of methods aimed at the drift that
`docs/generalisation-ceiling.md` shows is the binding constraint. `Split.date` is now exposed,
and the agent's own EDA immediately started measuring per-day label rates. It also could not
reach most of the paper knowledge base, because retrieval returned the same three entries every
iteration; it now sees the full catalogue and retrieval rotates.

**Leakage is the main correctness risk on this dataset.** `play_time_ms`, `is_click`, `is_like`
and the other logged signals are outcomes of the row being scored; they are exposed as
`Split.aux`, never as features, asserted by test. `video_features_statistic_pure.csv` is
excluded entirely — its counts are aggregated over the whole log period, including the
validation and test windows. Historical r39/r41/r43/r44 scores that used that file are retained as
an audit trail but excluded from memory, comparison, and submission; `submission_best.csv` is
restored to the strongest eligible run, r35.

## What we would do with more time

- **Parallel branch expansion.** The tree is explored one node at a time because scripts run
  sequentially on one CPU. Expanding several nodes concurrently would use the 6-hour budget far
  better than the ~5 minutes per iteration we currently spend.
- **Let the agent choose its own compute.** It gets a fixed 300s per script; a cheap idea and
  an expensive one are given the same budget, which biases it against methods that need time.
- **Cross-run memory is new and mostly unproven.** It only reads runs produced by this
  architecture, so at submission time it has very little history to work with.
- **The belief set is not evaluated.** We can see it changes what gets proposed, but we have
  not measured whether its claims are correct, only that the loop is better structured with it.

## Limitations

- The scored label is `long_view`, not `is_click`. The problem statement's Limits table still
  says "NDCG@10 / Recall@50, click = positive"; the Starter Kit, the Benchmarks table and the
  shipped `evaluate.py` all say `long_view` with GAUC / nDCG@5. We follow the Starter Kit.
- `pipeline/evaluate.py` is verified bit-identical to the official `evaluate.py`
  (max abs diff 1.7e-14) and ~7x faster. See `docs/metric-discrepancy.md`.
- `archive/` is dead code from the original AliCCP problem statement, plus
  `proposer_v1_human_priors.py` — the earlier prompt that contained hand-measured findings,
  kept as an honest record of what was removed and why. Neither is on the submission path.
- `research/` holds manual experiments from before the agent produced its own submission. They
  are **not** on the submission path; see `research/README.md`.
