"""The proposer: turns the run's accumulated state into the next script to execute.

The brief below is deliberately limited to the task specification, the pipeline API, and the
harness contract. It carries no findings about what works on this dataset. Everything the agent
knows about KuaiRand it has to establish itself -- by inspecting the data (EDA phase),
reproducing the official baseline (BASELINE phase), and then reading its own results through
revising its own belief set. That is what makes Requirement 1 and Innovation honest:
what gets tried, and why, has to come from the agent.
"""
import json
import os
import re
from pathlib import Path
from typing import Callable

from .kb import index, retrieve
from .loop import Proposal

CompleteFn = Callable[[str], tuple[str, int, int]]

_HYP_RE = re.compile(r"HYPOTHESIS:\s*(.+)")
_CODE_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)

MAX_HISTORY = 8
MAX_FEEDBACK_CHARS = 1200
MAX_KB = 3
MAX_ATTEMPTS = 2
MAX_CODE_CHARS = 14000     # a parent script is sent in full; truncating it produces bad edits
MAX_EDA_CHARS = 4000

# ---------------------------------------------------------------- task specification only

TASK_BRIEF = """You are the proposer inside an autonomous ML research agent competing on the
{variant} benchmark. You write complete Python scripts; a harness runs them and returns
the result. Nobody edits your code and nobody advises you: what to try, and why, is your call.

DATA: Kuaishou short-video feed, date-split. train {train_rows:,} rows ({train_lo}-{train_hi}),
validation {valid_rows:,} ({valid_lo}-{valid_hi}), hidden test {test_rows:,} ({test_lo}-{test_hi}).
The scored relevance label is the native column `long_view` (0/1), fixed by the organizers.
The split boundary is a date, and the distribution moves across it:
{drift_note}

METRIC: primary = mean(GAUC, nDCG@5), ranking within each user's logged impressions
(NOT full-catalog retrieval).
  GAUC: per-user AUC, counting only users with 0 < positives < impressions, weighted by
        each user's positive count.
  nDCG@5: gain = 2^rel - 1; users with zero positives score 0 and ARE included in the mean.
BASELINE ({baseline_source}, this is what you must beat):
  validation GAUC {baseline_valid_gauc:.4f} / nDCG@5 {baseline_valid_ndcg5:.4f} / primary {baseline_valid_primary:.4f}
  hidden test  GAUC {baseline_test_gauc:.4f} / nDCG@5 {baseline_test_ndcg5:.4f} / primary {baseline_test_primary:.4f}{baseline_noise_note}
  It is a Factorization Machine, k={baseline_k:.0f}, lr={baseline_lr}, over 5 categorical fields.
Harness self-check rungs, measured on this dataset: random scoring primary {random_primary:.4f},
item popularity primary {itempop_primary:.4f} (on test).
A validation difference smaller than 0.002 is inside seed noise and is NOT evidence. That
0.002 is also the organizers' convergence epsilon.
Perfect ranking on test reaches only primary {ceiling:.4f}, because {zero_pos_user_pct:.1f}% of test
users have no positive label at all. Judge headroom against {ceiling:.4f}, not 1.0.

HOW THE RUN ENDS -- read this before choosing an experiment:
  The run stops at whichever comes first: 50 iterations, 6 hours, or the organizers'
  convergence rule -- the best validation primary failing to improve by more than 0.002 for
  3 CONSECUTIVE iterations. Only the best validation score reached before that point is
  scored. Three experiments in a row that do not beat the incumbent by 0.002 end the run,
  whatever they taught you. Spend your iterations accordingly.

PIPELINE API -- import these, do not reimplement them:
  from pipeline.data import load, FEATURE_CARDINALITIES
  s = load("train")     # or "valid" or "test"
  s.X        dict[str, int64 array] -- {n_categorical} categorical features, contiguous ids,
             0 = unseen. Names, because guessing one costs an iteration: {categorical_names}
  s.y        int8 array -- long_view, the scored label
  s.user_id  int64 array      s.video_id  int64 array
  s.date     int32 array -- YYYYMMDD of each impression. train covers {train_days} days ({train_lo}-{train_hi}),
             validation the 7 days after it, test the 10 days after that. The splits are
             defined by this column, and it is an ordinary array you may use however you like.
  s.time_ms  int64 array -- epoch milliseconds of each impression. This is the only column that
             orders a user's impressions: s.date separates days and the `hour` field is
             time-of-day, neither of which can say which impression came first. Sorting by
             (user_id, time_ms) gives each user's history in order. Impressions logged in the
             same feed batch can share a timestamp, so ties are ordered by row position.
  s.num      dict[str, float32 array] -- {n_numeric} CONTINUOUS features, NaN where unknown.
             s.X holds only categorical ids, so these are the numeric quantities: the raw video
             length and the user's actual follower/following/friend counts and account age.
             Scale varies by orders of magnitude and NaN means absent, so handle both.
             Names: {numeric_names}
  from pipeline.history import historical_features
  historical_features(split_name, key="video_id" or "author_id") -> dict[str, float32 array]
             train-only counts and smoothed long_view/feedback histories for the entity. Train
             rows are leave-one-out; valid/test rows use the full train table. Available if you
             want it; nothing here says feature work beats model work, and you have only a
             handful of experiments before the run converges. (The dataset also ships full-month
             video statistics. Those overlap the evaluation window and are not exposed.)
  s.aux      dict of other logged signals (is_click, is_like, play_time_ms, ...)
  FEATURE_CARDINALITIES[name] -> int, the number of ids for that field
  from pipeline.evaluate import evaluate
  evaluate(user_ids, labels, scores) -> {{"primary","gauc","ndcg@5","ndcg@10","recall@50"}}
  evaluate(..., per_user=True) additionally returns out["per_user"], a dict of parallel arrays
  -- user_id, n_impressions, n_positives, auc (NaN where the user is excluded from GAUC),
  "ndcg@5", "ndcg@10" -- one entry per user, in ascending user_id. Both aggregates are exactly
  reconstructible from them (GAUC is the positive-count-weighted mean of auc; nDCG@5 is the
  plain mean over every user). Use it if you want to know which users a score comes from.
  This is the organizers' own scoring code, verified bit-identical to theirs.
  from agent.diagnose import segment_report
  segment_report(user_ids, labels, scores) -> a text table of YOUR model's nDCG@5 against the
  per-user ceiling, bucketed by how many impressions each user has, with each bucket's share of
  the total gap and its GAUC. It answers "where is this model losing" rather than "how much".
  Printing it costs one call and the aggregate score alone cannot tell you the same thing.

RULES:
  - The validation score you report MUST come from a model fit on "train" only. Every
    comparison and every selection is made on that number, so a model that has seen
    validation labels would make it meaningless.
  - The TEST scores you save may come from the SAME recipe refit on train + validation
    together. Validation is the week immediately before the test period and the rules
    allow developing on both splits; a model fit only on data that stops a week earlier
    is fitting staler behaviour than the one being scored. Fit train-only, evaluate and
    report validation, then refit the identical recipe on the two splits combined and
    use that model for scores_test.npy.
  - Never fit or select anything on "test", and never read test labels.
  - load("test").y DOES NOT EXIST. Touching it -- .y, .y.sum(), .y.astype(), len(s.y),
    anything -- raises RuntimeError and the whole iteration is lost. The test split gives
    you features only; you produce scores for it and the harness scores them for you.
  - Everything in s.aux is an OUTCOME of the row being scored. Using any of it as an input
    feature is label leakage and invalidates the result.
  - No external datasets, and no pretrained weights trained on this benchmark's test labels.

ENVIRONMENT:
  - Everything runs on CPU. numpy, torch, lightgbm 4.7 and scipy are installed.
  - lightgbm 4.7 removed `early_stopping_rounds` as a keyword everywhere; it raises TypeError.
    The supported form is a callback:
        m = lgb.train(params, dset, num_boost_round=600, valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
        preds = m.predict(X, num_iteration=m.best_iteration)
    Set verbosity as `"verbose": -1` inside params.
  - HARD LIMIT: {timeout:.0f} seconds per script, killed at the limit and scored as a failure.
    {train_rows_m:.2f}M rows: vectorize in numpy/torch. A Python loop over rows, users or pairs
    will time out.

WHAT ONE ITERATION IS: one script, executed once, reporting one METRICS line. Within that
script you may do as much as fits the time budget -- one iteration is one script, not one model.
The convergence rule counts ITERATIONS, not models: comparing five approaches inside one
script costs one of the three non-improving iterations the run is allowed, while comparing
them across five iterations costs five and ends the run. Search inside the script is the
cheap search; only the conclusion needs its own iteration
and not one idea. The METRICS line reports whatever you finally choose to evaluate.

OUTPUT CONTRACT -- the harness reads stdout:
  - Save the exact validation scores used for METRICS. The harness independently re-runs the
    pinned evaluator; self-reported metrics are never trusted:

        out = os.environ.get("ITER_OUT")
        if out:
            np.save(os.path.join(out, "scores_valid.npy"),
                    np.asarray(valid_scores, dtype=np.float64))

  - If the scores you report are a COMBINATION of your own model with anything you did not
    fit in this script -- the published incumbent, a stored array, another run's predictions
    -- also save the scores of your own model on its own:

        np.save(os.path.join(out, "scores_valid_raw.npy"),
                np.asarray(own_model_valid_scores, dtype=np.float64))

    Optional, never submitted, and never compared against your METRICS line: it is not a
    second result and nothing about your iteration is judged on it. Omit it if your reported
    scores already come from one model. It exists because a combined score cannot show what
    your own model contributed, and the harness has no other way to see that.
  - The FINAL stdout line must be exactly:
    METRICS {{"primary": <float>, "gauc": <float>, "ndcg@5": <float>,
              "gpu_seconds": <float>}}
    `gpu_seconds` is the script's wall time in seconds.
  - The script must ALSO score the test split and save it, so that your best iteration can be
    submitted without a human rebuilding anything. After evaluating validation:

        import os
        out = os.environ.get("ITER_OUT")
        if out:
            te = load("test")
            te_scores = <your model's scores for te, exactly as you scored valid>
            np.save(os.path.join(out, "scores_test.npy"),
                    np.asarray(te_scores, dtype=np.float64))

    Producing test SCORES is required. Fitting or selecting on test is forbidden.

REUSING WORK BETWEEN ITERATIONS: three directories, with different lifetimes.
  $ITER_OUT          this iteration only. Where scores_valid.npy, scores_test.npy and the
                     optional scores_valid_raw.npy go.
  $RUN_ARTIFACTS     yours for the whole run, and not shared with any experiment running
                     beside you. Anything you save there -- fitted predictions, arrays,
                     parameters -- is still there next iteration, and reloading is free where
                     refitting is not. Name files so you can recognise them later, and check a
                     file exists before trusting it: earlier iterations may have crashed.
  $SHARED_ARTIFACTS  read-only to you. The controller publishes the trusted incumbent's
                     verified predictions here; see the REUSABLE TRUSTED INCUMBENT note when
                     one exists. Do not write to it.
"""

# ---------------------------------------------------------------- phase instructions

_EDA_TASK = """PHASE: INSPECT DATA.

You have never seen this dataset. Before modelling anything, write a script that interrogates
it and PRINTS what you find. You get one shot, and whatever you print is the only thing you
will carry into every later iteration -- so measure what would actually change a modelling
decision, not what is merely easy to print.

Consider: label balance overall and per user; how many rows and positives a typical user has;
which of the 37 fields carry signal about `long_view` and which are near-constant; field
cardinalities against how many ids actually appear in each split; how much of validation is
users or videos unseen in train; whether s.X holds one scalar per row or any sequence; and the
shape of anything you might want to build a feature from.

Print compact, quantitative lines. Budget your output: at most ~60 lines, and it is truncated
at {max_eda} characters. Do NOT train a model and do NOT print a METRICS line in this phase."""

_BASELINE_TASK = """PHASE: REPRODUCE THE OFFICIAL BASELINE.

Requirement 1 of this challenge: stand up a working end-to-end pipeline and confirm it reaches
the official baseline's reported validation primary of {baseline:.4f}.

Write that pipeline yourself: a Factorization Machine, k=16, lr=0.001, trained on the 5
categorical fields the baseline uses (user identity, video identity, its author, the feed tab
it was shown in, and a bucketed video duration -- pick the matching names from the field list
you printed during EDA). Print the METRICS line and save test scores per the output contract.

You are reproducing a reference, not beating it. Getting within noise of {baseline:.4f} is
success; a much higher number means you have leaked something and a much lower one means the
pipeline is wrong."""

_IMPROVE_TASK = """PHASE: ITERATE.

Propose the next experiment and write the complete script for it. Improvements may target ANY
stage of the pipeline -- what features exist and how they are encoded, the model, the loss, the
training procedure, how predictions are post-processed or combined -- not only the architecture.

State in your hypothesis which stage you are targeting and the MECHANISM by which it should
raise a within-user ranking metric. An experiment that swaps in another named architecture with
no mechanism is worth little; one whose result is informative either way is worth a lot.
Remember that anything under 0.002 on validation is noise, so prefer changes big enough to
show above it.

Your script is not limited to a single alternative. It may construct and evaluate several
within its time budget and report whichever result you choose to stand behind -- the METRICS
line is what the harness scores, and how you arrive at it is yours to decide. If you do
evaluate more than one, print a line `CANDIDATES {{"name": score, ...}}` before the METRICS
line so the run log records what you compared.

The script may also measure things that are not its score -- a distribution, a correlation, a
per-group breakdown, a check of an assumption. Print each such observation on its own line
beginning `FINDINGS ` and it is carried into what the agent believes, whatever the score turns
out to be. An iteration that scores badly but establishes a fact is not a wasted iteration."""

_EDIT_PARENT = """You are editing an existing solution, not starting over.

BEST SCRIPT SO FAR -- iteration #{iid}, validation primary {score:.4f}:
hypothesis: {hyp}
```python
{code}
```

Make ONE targeted change to this script and return the COMPLETE modified script. Keep what is
working. If you believe this line of attack is exhausted, say so in your hypothesis and write
something structurally different instead."""

_BROADEN_CODE = """THE BEST SCRIPT SO FAR -- iteration #{iid}, validation primary {score:.4f}.
Keep what makes it work and build the new direction on top of it; do not restart from
something weaker just to be different. Changing direction means changing the method, not
discarding the best result:
```python
{code}
```"""

# Split from the script above so the script -- which repeats whenever the search stays on one
# node -- sits in the cacheable prefix, while the parts that change every iteration do not.
_BROADEN_INSTRUCTION = """The last {stale} experiment(s) produced no gain above 0.002. Refining
the same line of attack again is the most common way a run ends with nothing.

Change DIRECTION, not detail. Target a different stage of the pipeline, or a different family
of method, from everything listed above. A variation on an approach already in that list is
not a new direction, however it is described.

ALREADY TRIED THIS RUN:
{tried}

Return the COMPLETE script."""


_SWEEP_INSTRUCTION = """Train several structurally DIFFERENT model families on the same inputs
in this ONE script, score each on validation, and return the best.

Different family means a different way of FORMING the prediction. The same model at another
width, depth, learning rate, seed, epoch count or feature subset is NOT a different family,
and this iteration is wasted if you sweep those instead. Families to draw from -- not
exhaustive, not a recommendation, and you may propose one that is not listed:

  factorisation      FM, FFM, field-weighted FM
  deep CTR           DeepFM, xDeepFM, AutoInt, FiBiNET, DCNv2, PNN
  sequence/attention DIN, DIEN, transformer over user history
  gradient boosting  LightGBM binary, LambdaRank, setwise ranking
  multi-task         MMoE, PLE, ESMM over the other feedback signals
  latent/MF          truncated SVD, ALS, item2vec
  non-parametric     empirical Bayes, target statistics, popularity priors

A FAMILY IS NOT THE ONLY AXIS, AND ON THIS DATA IT IS NO LONGER THE BEST ONE. Seventeen
architectures have been tried across every group above; the first three complementary ones took
essentially all of the gain and the rest landed inside seed noise. The binding constraint is
DRIFT, not capacity. The evaluation window sits 8-20 days after training ends, and across that
gap users with no positive label go 5.1% -> 30.3% and the median rows per user go 31 -> 4. A
sweep over how the model is TRAINED AND SELECTED is now worth more than a sweep over what it is:

  sample weighting   weight the 13 training days by proximity to the evaluation window; sweep
                     the half-life. `split.date` is available. Measured standalone on a boosted
                     tree: uniform days score 0.4597 and a 4-day half-life scores 0.5518. That
                     has only ever been tried on a SIDE component the blender then damped to
                     +0.00002. Put it on the MAIN model's sample_weight and sweep the half-life
                     there; that is the untried version.
  selection protocol REFUTED, do not spend an iteration on it. Ranking this run's 49 scored
                     iterations by the last 2/3/4 days of validation instead of all 7 correlates
                     WORSE with hidden test (Spearman 0.33/0.66/0.84 against 0.87 for the full
                     window) and submits the same model. Full validation is the better selector.
  stationarity       prefer features whose distribution holds across windows over identity
                     embeddings. Expanding 9 -> 37 categorical fields gained validation and lost
                     test, which is what a non-stationary feature looks like.
  cold start         30% of evaluated users have no positive label at all and cannot be helped;
                     the score lives on the users who do.

WHY BREADTH, measured on this harness's own runs: a family sweep gains 0.0027-0.0031 primary
and clears the 0.002 convergence threshold on its own. An iteration that tunes an existing
model gains 0.0000-0.0004 and has never once cleared it in 15 attempts. Three sub-threshold
iterations end the run, so sweeping is what buys the budget to keep going.

The convergence rule charges per ITERATION, not per model: eight families in one script cost
exactly what one costs. Keep each fit short enough that all of them finish inside the time
limit -- a rough comparison of eight beats one polished fit.

Also score the BLEND of each new family with the trusted incumbent. On this dataset the blend
has beaten every standalone family, every time.

Print `CANDIDATES {{"family_name": score, ...}}` so the comparison is recorded, save the
winner's validation and test scores, and report its metrics.

FAMILIES AND DIRECTIONS ALREADY TRIED THIS RUN -- choose ones that are not here:
{tried}"""

_TUNE_CODE = """This experiment EXPLOITS the best architecture found so far rather than
looking for a new one.

THE ARCHITECTURE TO TUNE -- iteration #{iid}, validation primary {score:.4f}:
```python
{code}
```"""

_TUNE_INSTRUCTION = """Breadth has stopped paying: the last {stale} experiments gained less than
0.002. Do NOT introduce another model family. Take the architecture above and search its
CONFIGURATION space hard, in this one script.

What counts as configuration here: embedding width k, MLP depth and widths, dropout, L2 and
embedding regularisation, learning rate and schedule, batch size, epoch count and early
stopping, negative weighting, feature-field subsets, and the blend weights between the
components it already fuses.

Search it as SUCCESSIVE HALVING, not as a grid -- the timeout is {timeout:.0f}s and a full
grid will not fit:
  rung 1  ~16 configurations at a short budget (1 epoch, or a subsample of train)
  rung 2  the best ~6 of them at a medium budget
  rung 3  the best ~2 at the full budget
Report the rung-3 winner. This costs one iteration however many configurations it holds.

Two cautions from this run's own history. Selecting the max over many configurations on
124,909 validation rows overfits that split: prefer a configuration that is good across its
neighbours over an isolated peak. And a gain under 0.0008 is inside seed noise -- if nothing
clears that, say so and return the incumbent unchanged rather than shipping noise.

CONFIGURATIONS AND DIRECTIONS ALREADY TRIED THIS RUN:
{tried}"""


_DRAFT = """No prior solution has survived, so write this script from scratch."""

_SIBLINGS = """RUNNING IN PARALLEL WITH YOU THIS TURN -- other lines of work are writing these
scripts right now, and all of them will be scored against the same validation split:
{siblings}

Propose something in a different family, or targeting a different stage, from every line above.
A variation on one of them is not a different direction however it is described, and if two of
us return the same idea this turn the second one bought nothing.

This is not a hint about what works. It is the list of what is already covered."""

_SEED_NOTE = """WHY THIS LINE WAS RESTARTED HERE -- carried over from the archived attempt this
slot is resuming:
{note}"""

_SWEEP_CODE = """This experiment buys BREADTH across model families, not depth on one.

THE BASELINE TO BEAT -- iteration #{iid}, validation primary {score:.4f}:
```python
{code}
```"""


def _tried(history) -> str:
    """What the run has already attempted, so breadth prompts do not re-propose it."""
    return "\n".join(f"  - {e.hypothesis[:100]}" for e in history
                     if e.phase == "improve") or "  - nothing yet"


def _fmt_metrics(metrics: dict) -> str:
    if not metrics:
        return "-"
    return ",".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in metrics.items())


def _summarize_history(history, n: int) -> str:
    """Label each past iteration by what happened to the SCORE, not by the loop status.

    "kept" only means the gain was under epsilon; the score is still kept, still ranked and
    still submittable -- r59's submission came from such an iteration. Printing the raw label
    told the agent its three best results had been thrown away.
    """
    best, labels = float("-inf"), []
    for e in history:
        p = e.metrics.get("primary")
        if not isinstance(p, (int, float)):
            labels.append(e.status)
        elif p > best:
            best = p
            labels.append("BEST SO FAR")
        else:
            labels.append(f"below best {best:.4f}")
    lines = [f"#{e.iter_id} {lab} {_fmt_metrics(e.metrics)} :: {e.hypothesis[:70]}"
             for e, lab in list(zip(history, labels))[-n:]]
    return "\n".join(lines) if lines else "none"


def _kb_query(history, reflections, feedback: str | None) -> str:
    """Retrieval is driven by what the agent is actually working on. Nothing is pinned:
    a pinned entry is a human steering the search."""
    words = ["click through rate", "ranking", "ndcg", "gauc", "short video recommendation"]
    words += [e.hypothesis for e in history[-3:]]
    words += reflections[-2:]
    if feedback:
        words.append(feedback[:200])
    return " ".join(words)


class LLMProposer:
    """LLM-backed Proposer. `complete` is injected so this stays testable offline."""

    def __init__(self, complete: CompleteFn, kb_papers: list[dict] | None = None,
                 max_history: int = MAX_HISTORY, timeout: float = 300.0,
                 baseline: float = 0.6016, facts: dict | None = None):
        self.complete = complete
        self.kb_papers = kb_papers
        self.max_history = max_history
        self.timeout = timeout
        self.baseline = baseline
        # dataset facts are measured by agent.facts, not written into the brief by hand, so
        # pointing the harness at another KuaiRand variant does not feed the agent false premises
        self.facts = dict(facts) if facts is not None else json.loads(
            (Path(__file__).resolve().parent.parent / "research" / "facts_pure.json")
            .read_text(encoding="utf-8"))
        from pipeline.data import FEATURE_CARDINALITIES
        categorical = sorted(FEATURE_CARDINALITIES)
        self.facts.setdefault("n_categorical", len(categorical))
        # run_agent measures this from the splits; a proposer built without it still
        # formats, rather than dying on a missing key.
        self.facts.setdefault("drift_note", "(not measured for this dataset)")
        self.facts.setdefault("categorical_names", ", ".join(categorical))
        self._seen_papers: set[str] = set()   # so retrieval widens instead of repeating

    def propose(self, *, phase: str = "improve", history=None, blacklist=None,
                feedback: str | None = None, parent=None, context=None) -> Proposal | None:
        history = history or []
        blacklist = blacklist if blacklist is not None else set()
        context = context or {}
        note = None
        tokens_in = tokens_out = 0
        for _ in range(MAX_ATTEMPTS):
            prompt = self._build_prompt(phase, history, blacklist, feedback, parent, context, note)
            text, ti, to = self.complete(prompt)
            tokens_in += ti
            tokens_out += to
            parsed = self._parse(text)
            if parsed is None:
                note = ("Your previous reply did not parse. Reply with exactly one "
                        "'HYPOTHESIS: <one line>' followed by exactly one ```python fence.")
                continue
            hyp, code = parsed
            if phase == "improve" and hyp in blacklist:
                note = f'"{hyp}" is retired; propose a materially different experiment.'
                continue
            return Proposal(hyp, code, tokens_in, tokens_out)
        return None

    def _build_prompt(self, phase, history, blacklist, feedback, parent, context, note) -> str:
        blocks = [TASK_BRIEF.format(timeout=self.timeout, **self.facts)]

        if phase == "eda":
            blocks.append(_EDA_TASK.format(max_eda=MAX_EDA_CHARS))
        elif phase == "baseline":
            blocks.append(_BASELINE_TASK.format(baseline=self.baseline))
        else:
            blocks.append(_IMPROVE_TASK)

        # The catalogue is identical on every call of a run. Providers discount input that
        # repeats as a stable PREFIX, so it earns nothing sitting after the volatile blocks;
        # it has to sit against the brief for one cached span to cover both. Nothing is
        # dropped here -- the same text is sent, just where it can be reused.
        if phase == "improve":
            blocks.append(
                "AVAILABLE LITERATURE -- the whole catalogue you can draw on. It lists every "
                "method available, including ones that may not suit this data; deciding which "
                "are relevant is your job:\n" + index(self.kb_papers))

        eda = context.get("eda")
        if eda:
            blocks.append(f"WHAT YOU MEASURED WHEN YOU INSPECTED THE DATA:\n{eda[:MAX_EDA_CHARS]}")

        if phase == "improve":
            # Ordered for prompt caching, which matches on the longest common PREFIX. What
            # repeats across a run goes first -- memory, then the parent script, the largest
            # single block and identical whenever the search stays on the same node. Blocks
            # rewritten every iteration follow. Same text; only the order differs.
            mem = context.get("memory")
            if mem:
                blocks.append(mem)

            if parent is None:
                blocks.append(_DRAFT)
            elif context.get("mode") == "sweep":
                blocks.append(_SWEEP_CODE.format(iid=parent.iter_id, score=parent.score,
                                            code=parent.code[:MAX_CODE_CHARS]))
            elif context.get("mode") == "tune":
                blocks.append(_TUNE_CODE.format(iid=parent.iter_id, score=parent.score,
                                                code=parent.code[:MAX_CODE_CHARS]))
            elif context.get("mode") == "broaden":
                blocks.append(_BROADEN_CODE.format(iid=parent.iter_id, score=parent.score,
                                                   code=parent.code[:MAX_CODE_CHARS]))
            else:
                blocks.append(_EDIT_PARENT.format(iid=parent.iter_id, score=parent.score,
                                                  hyp=parent.hypothesis,
                                                  code=parent.code[:MAX_CODE_CHARS]))

            diagnosis = context.get("diagnosis")
            if diagnosis:
                blocks.append("WHERE THE TRUSTED INCUMBENT LOSES ON VALIDATION (computed by the "
                              "controller from its saved predictions):\n" + diagnosis)
            if context.get("incumbent_ready"):
                blocks.append(
                    "REUSABLE TRUSTED INCUMBENT: $SHARED_ARTIFACTS contains "
                    "incumbent_valid_scores.npy and incumbent_test_scores.npy plus "
                    "incumbent.json. You may load and blend these exact predictions instead "
                    "of retraining the incumbent; choose every blend weight on validation "
                    "only and apply the same fixed weight to test.")
            blocks.append("EXPERIMENTS THIS RUN (oldest first):\n"
                          f"{_summarize_history(history, self.max_history)}")
            know = context.get("knowledge") or ""
            refl = [know] if know else []
            if know and know != "nothing established yet":
                blocks.append("WHAT YOU CURRENTLY BELIEVE ABOUT THIS TASK, established by your "
                              "own results. A claim marked (invalidated) was contradicted by "
                              "later evidence -- do not act on it again without a new "
                              "mechanism:\n" + know)
            bl = ", ".join(sorted(blacklist))
            if bl:
                blocks.append(f"RETIRED -- do not propose again: {bl}")
            # What the other slots are attempting this turn. A negative constraint only: naming
            # architectures to build would be a prior on method space, which the brief refuses
            # to carry. This is the broaden instruction's "already tried" list applied across
            # slots instead of across time.
            siblings = context.get("siblings")
            if siblings:
                blocks.append(_SIBLINGS.format(siblings=siblings))
            seed_note = context.get("seed_note")
            if seed_note:
                blocks.append(_SEED_NOTE.format(note=seed_note))
            if parent is not None and context.get("mode") == "tune":
                blocks.append(_TUNE_INSTRUCTION.format(
                    stale=context.get("stale", 2), timeout=self.timeout,
                    tried=_tried(history)))
            if parent is not None and context.get("mode") == "sweep":
                blocks.append(_SWEEP_INSTRUCTION.format(tried=_tried(history)))
            if parent is not None and context.get("mode") == "broaden":
                tried = "\n".join(f"  - {e.hypothesis[:100]}" for e in history
                                  if e.phase == "improve") or "  - nothing yet"
                blocks.append(_BROADEN_INSTRUCTION.format(
                    stale=context.get("stale", 1), tried=tried))
            kb = retrieve(_kb_query(history, refl, feedback), k=MAX_KB, papers=self.kb_papers,
                          seen=self._seen_papers)
            if kb:
                self._seen_papers.update(p["id"] for p in kb)
                blocks.append("DETAIL ON A FEW OF THEM (a retrieval hit is not a recommendation; "
                              "judge for yourself whether the mechanism applies here):\n"
                              + "\n".join(f"- {p['id']}: {p['title']} ({p['year']}) - "
                                          f"{p['expected_effect']}" for p in kb))

        if feedback:
            fb = ("YOUR PREVIOUS ATTEMPT FAILED. Diagnose it from the output below, fix it, and "
                  f"keep the same hypothesis.\n{feedback[:MAX_FEEDBACK_CHARS]}")
            last_code = next((e.diff for e in reversed(history)
                              if e.status in ("failed", "blacklisted") and e.diff), "")
            if last_code and parent is None:
                fb += f"\n\nThe script that failed:\n```python\n{last_code[:MAX_CODE_CHARS]}\n```"
            blocks.append(fb)

        if note:
            blocks.append(f"NOTE: {note}")

        blocks.append(
            "Respond EXACTLY as:\n"
            "HYPOTHESIS: <one line: what you are testing, which stage, and the mechanism>\n"
            "```python\n"
            "<complete runnable script>\n"
            "```"
        )
        return "\n\n".join(blocks)

    def _parse(self, text: str) -> tuple[str, str] | None:
        hm = _HYP_RE.search(text)
        cm = _CODE_RE.search(text)
        if not hm or not cm:
            return None
        hyp, code = hm.group(1).strip(), cm.group(1).strip()
        return (hyp, code) if hyp and code else None
