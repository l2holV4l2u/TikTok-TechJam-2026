"""The controller: runs the MLE loop of the problem statement's Figure 1.

Three phases, all executed by the agent itself:
  eda       -- inspect the data and print findings, which persist into every later prompt
  baseline  -- stand up an end-to-end pipeline and reproduce the official baseline (Req 1)
  improve   -- guided search over solution scripts, with a knowledge revision between
               evaluate and propose so later evidence can overturn earlier conclusions

Wall-clock, not summed script time, is what Feasibility scores, so it is measured here and
enforced as a stop condition.
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .executor import RunResult, run_script
from .ensemble import retain_or_blend
from .ledger import Entry, Ledger
from .llm import LLMDailyLimit
from .recovery import RETRY, Recovery
from .critic import review as critic_review
from .knowledge import Knowledge
from .tree import Node, Tree

METRIC_PREFIX = "METRICS "
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIX_HOURS = 6 * 3600.0
BASELINE_TOLERANCE = 0.002  # organizer epsilon ~= 2.5 times Pure's reported 5-seed std
MAX_BASELINE_ATTEMPTS = 4
MAX_EDA_ATTEMPTS = 2


@dataclass
class Proposal:
    hypothesis: str
    code: str
    tokens_in: int
    tokens_out: int


@dataclass
class LoopResult:
    stop_reason: str            # converged|budget|wall_clock|max_iters|exhausted|...
    iterations: int = 0
    wall_clock_s: float = 0.0
    script_seconds: float = 0.0
    baseline_primary: float | None = None
    eda_ok: bool = False
    knowledge: Knowledge | None = None
    tree: Tree | None = None
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    candidates_evaluated: int = 0   # alternatives compared inside iterations, not just between
    # the iteration at which the stricter per-iteration reading of the convergence rule
    # would have stopped, recorded so a judge can check the run against either reading
    strict_converged_iter: int | None = None


class Proposer(Protocol):
    def propose(self, *, phase: str, history: list[Entry], blacklist: set[str],
                feedback: str | None, parent, context: dict) -> Proposal | None: ...


class Evaluator(Protocol):
    def evaluate(self, result: RunResult, iter_out: Path | None = None) -> dict | None: ...


class StdoutJsonEvaluator:
    """Reads the last `METRICS {...}` line the generated script printed."""

    def evaluate(self, result: RunResult, iter_out: Path | None = None) -> dict | None:
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(METRIC_PREFIX):
                try:
                    return json.loads(line[len(METRIC_PREFIX):])
                except json.JSONDecodeError:
                    return None
        return None


class SavedScoresEvaluator(StdoutJsonEvaluator):
    """Recompute metrics from saved predictions instead of trusting generated stdout.

    The generated script is part of the search space, not part of the trusted evaluator.  Its
    METRICS line is checked for drift, while the arrays retained here are also used for
    automatic diagnosis and incumbent reuse by the controller.
    """

    # The check catches a FABRICATED metric, and a fabrication worth making differs by far
    # more than display rounding. At 1e-9 an honest script that printed six significant
    # figures was rejected -- r62 lost a baseline iteration to 0.590473 against
    # 0.590472506127, a gap of 5e-7. 1e-6 is still 2000x tighter than the epsilon that
    # decides convergence.
    def __init__(self, tolerance: float = 1e-6, require_test: bool = True):
        self.tolerance = tolerance
        self.require_test = require_test
        self.last_error: str | None = None
        self.last_scores = None
        self.last_test_scores = None

    @staticmethod
    def _load_scores(path: Path, expected: int, name: str):
        import numpy as np

        if not path.exists():
            raise ValueError(f"Missing {name}. Save it under $ITER_OUT before METRICS.")
        scores = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        if scores.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {scores.shape}.")
        if len(scores) != expected:
            raise ValueError(f"{name} has {len(scores)} rows; expected {expected}.")
        if not np.isfinite(scores).all():
            raise ValueError(f"{name} contains NaN or Inf.")
        return scores

    def evaluate(self, result: RunResult, iter_out: Path | None = None) -> dict | None:
        import math

        self.last_error = None
        self.last_scores = self.last_test_scores = None
        reported = super().evaluate(result, iter_out)
        if reported is None:
            self.last_error = "The METRICS line is missing or is not valid JSON."
            return None
        if iter_out is None:
            self.last_error = "The harness did not provide the iteration output directory."
            return None

        try:
            from pipeline.data import load
            from pipeline.evaluate import evaluate

            valid = load("valid")
            self.last_scores = self._load_scores(
                Path(iter_out) / "scores_valid.npy", len(valid.y), "scores_valid.npy")
            if self.require_test:
                test = load("test")
                self.last_test_scores = self._load_scores(
                    Path(iter_out) / "scores_test.npy", len(test.y), "scores_test.npy")
            verified = evaluate(valid.user_id, valid.y, self.last_scores)
        except (OSError, ValueError, TypeError) as exc:
            self.last_error = str(exc)
            self.last_scores = self.last_test_scores = None
            return None

        for key in ("primary", "gauc", "ndcg@5"):
            claimed = reported.get(key)
            actual = verified[key]
            if (not isinstance(claimed, (int, float)) or not math.isfinite(float(claimed))
                    or abs(float(claimed) - actual) > self.tolerance):
                self.last_error = (f"Self-reported {key}={claimed!r} does not match the trusted "
                                   f"evaluator's {actual:.12g}. Save and evaluate the same "
                                   "validation score array.")
                self.last_scores = self.last_test_scores = None
                return None
        if isinstance(reported.get("gpu_seconds"), (int, float)):
            verified["gpu_seconds"] = float(reported["gpu_seconds"])
        return verified


CANDIDATE_PREFIX = "CANDIDATES "


def parse_candidates(stdout: str) -> dict | None:
    """A script may evaluate several alternatives internally and report only its choice. This
    recovers what it compared, so the run log shows the search that happened inside one
    iteration rather than just its winner."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(CANDIDATE_PREFIX):
            try:
                got = json.loads(line[len(CANDIDATE_PREFIX):])
            except json.JSONDecodeError:
                return None
            return got if isinstance(got, dict) and got else None
    return None


FINDINGS_PREFIX = "FINDINGS "
MAX_FINDINGS_CHARS = 1500


def parse_findings(stdout: str) -> str:
    """Evidence a script gathered that is not its score.

    Iris (arXiv:2608.02143) separates epistemic actions -- gathering discriminating evidence --
    from interventional ones that change the retained solution, and loses 6.7 points of
    any-medal rate without them. Our convergence rule charges an iteration for any experiment,
    so a purely diagnostic iteration is unaffordable; letting a scored script also report what
    it observed buys the same evidence without spending a life on it.
    """
    lines = [ln[len(FINDINGS_PREFIX):].strip() for ln in (stdout or "").splitlines()
             if ln.startswith(FINDINGS_PREFIX)]
    return "\n".join(lines)[:MAX_FINDINGS_CHARS]


def _failure_text(res: RunResult, timeout: float, primary: str) -> str:
    if res.timed_out:
        # a timeout is not a bug to fix -- rerunning the same approach times out again
        return (f"KILLED at the {timeout:.0f}s limit. The approach is too slow, not wrong. "
                f"Do not resubmit it unchanged: either vectorize the hot loop or propose a "
                f"different, cheaper idea.\n{res.stderr}")
    if not res.ok:
        return res.stderr
    return f"The script ran but printed no `{primary}` in a METRICS line."


def _publish_incumbent(evaluator, artifacts: Path, run_dir: Path, iter_id: int,
                       score: float) -> str:
    """Persist trusted predictions and derive the diagnosis the proposer rarely requests.

    Concrete, fixed filenames turn RUN_ARTIFACTS from an optional convention into a usable
    controller guarantee: the next iteration can blend against the incumbent without
    retraining it.  Only arrays verified by SavedScoresEvaluator reach this path.
    """
    import numpy as np

    valid_scores = getattr(evaluator, "last_scores", None)
    test_scores = getattr(evaluator, "last_test_scores", None)
    if valid_scores is None or test_scores is None:
        return ""
    np.save(artifacts / "incumbent_valid_scores.npy", valid_scores)
    np.save(artifacts / "incumbent_test_scores.npy", test_scores)
    (artifacts / "incumbent.json").write_text(json.dumps({
        "iter_id": iter_id, "valid_primary": score,
        "valid_scores": "incumbent_valid_scores.npy",
        "test_scores": "incumbent_test_scores.npy",
    }, indent=2), encoding="utf-8")

    from pipeline.data import load
    from .diagnose import segment_report
    valid = load("valid")
    diagnosis = segment_report(valid.user_id, valid.y, valid_scores)
    with (run_dir / "diagnostics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"iter_id": iter_id, "valid_primary": score,
                            "report": diagnosis}) + "\n")
    return diagnosis


def converged(best_curve: list[float], patience: int, epsilon: float) -> bool:
    """The organizers' rule, read as written: has the validation best improved by more than
    epsilon ACROSS the last N iterations -- not whether each single step beat it by epsilon.
    """
    return (len(best_curve) > patience
            and best_curve[-1] - best_curve[-1 - patience] <= epsilon)


def run_loop(proposer: Proposer, evaluator: Evaluator, ledger: Ledger, *,
             workdir, primary: str = "primary", epsilon: float = 0.002,
             patience: int = 3, max_iters: int = 50, timeout: float = 300,
             wall_clock_limit_s: float = SIX_HOURS,
             tree: Tree | None = None, recovery: Recovery | None = None,
             knowledge: Knowledge | None = None, revise_fn=None,
             memory: str = "", baseline: float = 0.6016, ceiling: float | None = None,
             baseline_tolerance: float = BASELINE_TOLERANCE,
             max_instant_failures: int = 5, max_proposer_errors: int = 6,
             force_mode: str = "") -> LoopResult:
    tree = tree if tree is not None else Tree(epsilon=epsilon)
    recovery = recovery or Recovery()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    out = LoopResult("max_iters", tree=tree)
    spent = 0.0
    stale = 0
    best_curve: list[float] = []   # validation best after each scored improve iteration
    best = float("-inf")
    selection_best = float("-inf")
    feedback: str | None = None
    knowledge = knowledge if knowledge is not None else Knowledge()
    out.knowledge = knowledge
    context: dict = {"eda": None, "knowledge": knowledge.render(), "memory": memory}
    eda_attempts = baseline_attempts = 0
    instant_failures = proposer_errors = 0

    def elapsed() -> float:
        return time.perf_counter() - t_start

    def finish(reason: str) -> LoopResult:
        out.stop_reason = reason
        out.wall_clock_s = elapsed()
        out.script_seconds = spent
        return out

    for i in range(max_iters):
        if elapsed() >= wall_clock_limit_s:
            return finish("wall_clock")

        # ---- which stage of Figure 1 are we in
        if context["eda"] is None and eda_attempts < MAX_EDA_ATTEMPTS:
            phase = "eda"
        elif out.baseline_primary is None and baseline_attempts < MAX_BASELINE_ATTEMPTS:
            phase = "baseline"
        else:
            phase = "improve"
            # The organizers' rule: "converged when validation score has not improved by more
            # than eps over the last N consecutive iterations". We read that as written -- the
            # gain ACROSS the window, not each single step beating the incumbent by eps. The
            # per-step reading stops a run climbing +0.0008 an iteration, which under the spec
            # has improved by 0.0024 over three and is not converged. r59 ended at iteration 6
            # of 50 still setting a record every step. Both stop points are recorded.
            if stale >= patience and out.strict_converged_iter is None:
                out.strict_converged_iter = i
            if converged(best_curve, patience, epsilon):
                return finish("converged")

        # live budget state: how close the convergence rule is to ending the run
        context["stale"] = stale
        context["patience"] = patience
        context["iters_left"] = max_iters - i
        # Adaptive search policy, after FML-bench (arXiv:2605.17373): an agent that switches to
        # broader exploration on detecting stagnation outperformed every fixed strategy, and
        # breadth should track opportunity density -- greedy while gains are dense, broad when
        # they are sparse. Gains are sparse here: almost nothing clears epsilon. One
        # non-improving iteration is our switch signal, because with only `patience` of them
        # before the run ends, waiting for a longer stagnation streak wastes the whole budget.
        # The first improve iteration is spent on breadth, not depth. Measured on this
        # dataset, distinct model families differ by 0.0019 primary while two seeds of the
        # same family differ by 0.0002 -- the choice of family is worth more than anything
        # tuning it recovers, and the convergence rule charges the same one iteration either
        # way. Later iterations refine whatever the sweep found.
        # Sweep again on the FIRST miss, not just at the start. Measured over r70-r74: a
        # family sweep gains 0.0027-0.0031 and clears epsilon alone, while all 15 refine
        # iterations gained 0.0000-0.0004 and none did. Three sub-epsilon iterations end the
        # run, so the runs converged at 6 of 50 iterations having spent 16 min of the 6h
        # ceiling. Breadth is what buys the budget; a second miss falls through to broaden,
        # which can leave the model stage entirely.
        # Explore while breadth pays, then exploit into the convergence tail. A run must
        # spend `patience` sub-epsilon iterations before it can stop, so that tail is going to
        # be spent either way -- r70-r74 spent it wandering and banked 0.0005 total. The last
        # rung tunes the best architecture instead, as a config search inside one script.
        if force_mode and phase == "improve":
            mode = force_mode
        elif phase == "improve" and (not best_curve or stale == 1):
            mode = "sweep"
        elif phase == "improve" and stale >= 2:
            mode = "tune"
        else:
            mode = "refine" if stale == 0 else "broaden"
        context["mode"] = mode
        parent = tree.select(mode) if phase == "improve" else None

        try:
            p = proposer.propose(phase=phase, history=ledger.read(),
                                 blacklist=recovery.blacklist, feedback=feedback,
                                 parent=parent, context=context)
        except Exception as exc:
            # an LLM outage or rate limit must not kill a long unattended run
            proposer_errors += 1
            ledger.append(Entry(i, None, 0, "(proposer unavailable)", "", {}, 0.0, 0, 0,
                                "failed", f"{type(exc).__name__}: {exc}"[:2000], phase))
            # Every key being out of daily quota is not an outage that clears on a retry.
            # Retrying it burned 50 minutes of r57 rediscovering the same cap six times.
            if isinstance(exc, LLMDailyLimit):
                return finish("llm_daily_limit")
            if proposer_errors >= max_proposer_errors:
                return finish("proposer_unavailable")
            time.sleep(min(60.0, 5.0 * proposer_errors))
            continue
        proposer_errors = 0
        if p is None:
            return finish("exhausted")
        out.llm_tokens_in += p.tokens_in
        out.llm_tokens_out += p.tokens_out

        script = (workdir / f"iter_{i}.py").resolve()
        script.write_text(p.code, encoding="utf-8")
        iter_out = (workdir / f"iter_{i}_out").resolve()
        iter_out.mkdir(parents=True, exist_ok=True)
        # a scratch directory shared by every iteration of this run. Without it each script
        # retrains from cold: in one run the per-iteration time went 84s -> 368s purely because
        # later iterations rebuilt the components earlier ones had already fitted, which caps
        # how much an iteration can attempt before it hits the wall-clock limit.
        artifacts = (workdir.parent / "artifacts").resolve()
        artifacts.mkdir(parents=True, exist_ok=True)
        res = run_script(script, timeout=timeout, cwd=None, pythonpath=_PROJECT_ROOT,
                         extra_env={"ITER_OUT": str(iter_out),
                                    "RUN_ARTIFACTS": str(artifacts),
                                    # Generated code never needs hidden-test labels. The loader
                                    # still exposes len/shape for allocation, but array access
                                    # fails instead of silently enabling test selection.
                                    "AGENT_HIDE_TEST_LABELS": "1"})
        spent += res.seconds
        parent_id = parent.iter_id if parent is not None else None
        findings = parse_findings(res.stdout)

        # ---- EDA produces prose, not a score; judge it on whether it ran and said anything
        if phase == "eda":
            eda_attempts += 1
            if res.ok and res.stdout.strip():
                context["eda"] = res.stdout.strip()
                (workdir.parent / "eda_report.txt").write_text(context["eda"], encoding="utf-8")
                feedback = None
                ledger.append(Entry(i, None, 0, p.hypothesis, p.code, {}, res.seconds,
                                    p.tokens_in, p.tokens_out, "ok", None, "eda"))
                out.eda_ok = True
            else:
                feedback = _failure_text(res, timeout, primary)
                ledger.append(Entry(i, None, 0, p.hypothesis, p.code, {}, res.seconds,
                                    p.tokens_in, p.tokens_out, "failed", feedback[:2000], "eda"))
            continue

        metrics = evaluator.evaluate(res, iter_out) if res.ok else None
        scored = metrics is not None and primary in metrics

        # ---- a run that produced nothing at all: retry, or declare the box broken
        if not scored:
            why = getattr(evaluator, "last_error", None) or _failure_text(res, timeout, primary)
            action, feedback = recovery.on_failure(p.hypothesis, why)
            if phase == "baseline":
                baseline_attempts += 1
            ledger.append(Entry(i, parent_id, 0, p.hypothesis, p.code, {}, res.seconds,
                                p.tokens_in, p.tokens_out,
                                "failed" if action == RETRY else "blacklisted", feedback, phase))
            # a child that crashed is evidence against its parent, not a free retry. Without
            # this, only scored children ever counted toward retirement, so a node producing
            # nothing but crashes stayed selectable forever -- r38_1k #8 spent two children and
            # 78 minutes that way, one of them timing out at 4,373s.
            tree.record_failure(parent_id, res.seconds)
            # a script that dies instantly with no output never even started: the interpreter or
            # the machine is broken, not the code. Grinding on shreds the budget for nothing.
            if res.seconds < 1.0 and not res.stdout and not res.stderr.strip():
                instant_failures += 1
                if instant_failures >= max_instant_failures:
                    return finish("environment_broken")
            else:
                instant_failures = 0
            continue

        instant_failures = 0
        if phase == "improve":
            metrics = retain_or_blend(metrics, evaluator, artifacts, workdir.parent, i)
        score = metrics[primary]
        cands = parse_candidates(res.stdout)

        # ---- baseline reproduction is checked against the organizers' published number
        if phase == "baseline":
            baseline_attempts += 1
            if abs(score - baseline) <= baseline_tolerance:
                out.baseline_primary = score
                feedback = None
                status = "ok"
            else:
                direction = "far above" if score > baseline else "far below"
                feedback = (f"Your pipeline scored {score:.4f}, {direction} the official "
                            f"baseline's {baseline:.4f}. "
                            + ("A score this high on a reproduction usually means leakage -- "
                               "check that no column from s.aux reached the features and that "
                               "nothing was fit on validation."
                               if score > baseline else
                               "Check the field choice, the training length, and that the label "
                               "is long_view.")
                            + " Fix the pipeline and reproduce it again.")
                status = "reverted"
            ledger.append(Entry(i, None, 0, p.hypothesis, p.code, metrics, res.seconds,
                                p.tokens_in, p.tokens_out, status, feedback, "baseline"))
            if out.baseline_primary is not None:
                tree.add(Node(i, None, p.hypothesis, p.code, score))
                best, stale = score, 0
                if status == "ok" and score > selection_best:
                    selection_best = score
                    context["diagnosis"] = _publish_incumbent(
                        evaluator, artifacts, workdir.parent, i, score)
                    context["incumbent_ready"] = getattr(evaluator, "last_scores", None) is not None
            if out.baseline_primary is None and baseline_attempts >= MAX_BASELINE_ATTEMPTS:
                _revise(revise_fn, ledger, out, context, knowledge, findings, stale, patience)
                return finish("baseline_not_reproduced")
            _revise(revise_fn, ledger, out, context, knowledge, findings, stale, patience)
            continue

        # ---- improve
        # Arbor (arXiv:2606.12563) ablates its critic and reports the largest drop of any
        # component. Ours had none for improve iterations: the leakage warning fires only during
        # baseline reproduction, so an iteration that scored 0.95 by reading the label would have
        # been accepted as the leader. Rejection is recoverable: the proposer receives the exact
        # reason and may re-run a genuine breakthrough without the unsafe access pattern.
        # Judge the generated candidate itself, before a safe incumbent blend can damp an
        # implausible or leaky raw score enough to hide it from the integrity thresholds.
        critic_score = metrics.get("raw_candidate_primary", score)
        flags = critic_review(p.code, critic_score, best, ceiling)
        if flags:
            feedback = ("This result was not accepted as-is. " + " ".join(flags)
                        + " Re-run the check yourself and either show the result survives it or "
                          "propose something else.")
            ledger.append(Entry(i, parent_id, 0, p.hypothesis, p.code, metrics, res.seconds,
                                p.tokens_in, p.tokens_out, "rejected", feedback, "improve"))
            # A rejected score is neither a search node nor a submission candidate. Count the
            # attempt against its parent so integrity failures cannot create an immortal branch.
            tree.record_child(parent_id, float("-inf"), res.seconds)
            stale += 1
            _revise(revise_fn, ledger, out, context, knowledge, "", stale, patience)
            continue

        recovery.on_success(p.hypothesis)

        if cands:
            out.candidates_evaluated += len(cands)
            with (workdir.parent / "candidates.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"iter_id": i, "phase": phase, "candidates": cands}) + "\n")

        improved = score > best + epsilon
        ledger.append(Entry(i, parent_id, 0, p.hypothesis, p.code, metrics, res.seconds,
                            p.tokens_in, p.tokens_out, "ok" if improved else "kept",
                            None, "improve"))
        feedback = None
        tree.add(Node(i, parent_id, p.hypothesis, p.code, score))
        tree.record_child(parent_id, score, res.seconds)

        if score > selection_best:
            selection_best = score
            context["diagnosis"] = _publish_incumbent(
                evaluator, artifacts, workdir.parent, i, score)
            context["incumbent_ready"] = getattr(evaluator, "last_scores", None) is not None

        # retire an idea only when it is clearly worse; a result that ties or sets a record is
        # not evidence the idea is a dead end
        idea_score = metrics.get("raw_candidate_primary", score)
        if idea_score < best - 2 * epsilon:
            recovery.on_underperform(p.hypothesis)

        if improved:
            best, stale = score, 0
        else:
            stale += 1
        best_curve.append(selection_best)

        _revise(revise_fn, ledger, out, context, knowledge, findings, stale, patience)

    return finish("max_iters")


def _revise(revise_fn, ledger: Ledger, out: LoopResult, context: dict,
            knowledge: Knowledge, findings: str, stale: int, patience: int) -> None:
    """Revise the belief set, then republish it so the next proposal sees the current state."""
    if revise_fn is None:
        return
    entries = ledger.read()
    if not entries:
        return
    ti, to = revise_fn(entries, entries[-1], findings, stale, patience)
    out.llm_tokens_in += ti
    out.llm_tokens_out += to
    context["knowledge"] = knowledge.render()
