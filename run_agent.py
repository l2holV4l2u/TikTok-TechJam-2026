"""Drive the autonomous ML research agent on KuaiRand-Pure.

python run_agent.py --run-dir runs/r27 --iters 50
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from agent.hardware import prompt_hardware_note, resolve_device
from agent.kb import load_papers
from agent.ledger import Ledger
from agent.llm import FakeComplete, RecordingComplete, ReplayComplete, make_complete
from agent.loop import SavedScoresEvaluator, run_loop
from agent.memory import distil
from agent.proposer import LLMProposer
from agent.recovery import Recovery
from agent.knowledge import Knowledge
from agent.tree import Tree

# KuaiRand-Pure's published baseline. Other variants have no published number, so a run on one
# passes --baseline-valid/--baseline-test measured by research/baseline_reference.py.
def _load_probe() -> list[str]:
    """Names an agent script can reach on a Split, for the stale-capability diff in memory."""
    from pipeline.data import load
    try:
        s = load("train")
    except FileNotFoundError:
        return []
    names = ["s.X", "s.y", "s.user_id", "s.video_id", "s.aux"]
    if s.date is not None:
        names.append("s.date")
    if s.time_ms is not None:
        names.append("s.time_ms")
    names += [f"s.num[{k}]" for k in (s.num or {})]
    names.append("pipeline.history.historical_features")
    return names


BASELINE_VALID = 0.6016
BASELINE_TEST = 0.5946


def _dry_run_baseline() -> float:
    """Use the canned item's real validation score so dry-run tests plumbing, not the FM."""
    import numpy as np
    from pipeline.data import load
    from pipeline.evaluate import evaluate

    train, valid = load("train"), load("valid")
    positives = np.bincount(train.X["video_id"], weights=np.asarray(train.y, dtype=float))
    counts = np.maximum(np.bincount(train.X["video_id"]), 1)
    rates = positives / counts
    video = np.minimum(valid.X["video_id"], len(rates) - 1)
    return float(evaluate(valid.user_id, valid.y, rates[video])["primary"])


def _write_submission(run_dir: Path, ledger: Ledger, baseline_test: float) -> dict:
    """Build the submission from the agent's own best iteration -- no human rebuild.

    The organizers score the validation-best checkpoint. We hold the public test labels, so we
    also report what that submission scores -- but selection is on validation only; picking a
    different iteration because its test score is higher would be fitting the hidden set.
    """
    import numpy as np
    scored = [e for e in ledger.read()
              if e.status in ("ok", "reverted") and "primary" in e.metrics]
    if not scored:
        print("no scored iteration; nothing to submit")
        return {}
    best = max(scored, key=lambda e: e.metrics["primary"])
    path = run_dir / "scripts" / f"iter_{best.iter_id}_out" / "scores_test.npy"
    if not path.exists():
        print(f"best iteration #{best.iter_id} left no scores_test.npy; cannot build submission")
        return {}

    from pipeline.data import load
    te = load("test")
    scores = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if scores.ndim != 1 or len(scores) != len(te.y):
        print(f"score shape {scores.shape} != ({len(te.y)},); refusing to write")
        return {}
    if not np.isfinite(scores).all():
        print("test scores contain NaN or Inf; refusing to write")
        return {}

    out = run_dir / "submission.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        f.write("row_id,user_id,video_id,score\n")
        for i, (u, v, sc) in enumerate(zip(te.user_id, te.video_id, scores)):
            f.write(f"{i},{u},{v},{sc}\n")

    from pipeline.evaluate import evaluate
    r = evaluate(te.user_id, te.y, scores)
    print(f"\nsubmission from iteration #{best.iter_id} "
          f"(validation primary {best.metrics['primary']:.4f}) -> {out}")
    print(f"  hypothesis: {best.hypothesis[:90]}")
    print(f"  test primary {r['primary']:.4f}  gauc {r['gauc']:.4f}  ndcg@5 {r['ndcg@5']:.4f}"
          f"   delta vs baseline {r['primary'] - baseline_test:+.4f}")
    return {"iter_id": best.iter_id, "valid_primary": best.metrics["primary"],
            "test_primary": r["primary"], "test_gauc": r["gauc"], "test_ndcg@5": r["ndcg@5"],
            "test_delta": r["primary"] - baseline_test, "hypothesis": best.hypothesis}


def main() -> None:
    # a hypothesis is model-written text and can contain any character. Redirected stdout takes
    # the system codepage, so one 'x' printed as U+00D7 killed a completed run before it could
    # write its metadata. The agent's own output must never be able to crash the harness.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/latest")
    ap.add_argument("--iters", type=int, default=50, help="organizer hard cap is 50")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-script timeout; one iteration is one script and may search internally, so this is generous — the 6h ceiling binds first")
    ap.add_argument("--wall-clock-s", type=float, default=6 * 3600.0,
                    help="organizer backstop is 6 hours")
    ap.add_argument("--patience", type=int, default=3, help="N in the convergence rule")
    ap.add_argument("--epsilon", type=float, default=0.002, help="organizer-fixed eps")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--max-misses", type=int, default=3,
                    help="non-improving children before a search node is retired")
    ap.add_argument("--baseline-valid", type=float, default=BASELINE_VALID,
                    help="validation score the agent must reproduce; measure it with research.baseline_reference on a non-Pure variant")
    ap.add_argument("--baseline-test", type=float, default=BASELINE_TEST,
                    help="test score the reported delta is taken against")
    ap.add_argument("--baseline-tolerance", type=float, default=0.002,
                    help="maximum absolute baseline reproduction error")
    ap.add_argument("--revision-model", default=None,
                    help="model for belief revision. Rate limits are per-model, and revision is "
                         "~37%% of a run's requests, so pointing it at a second model both halves "
                         "pressure on the proposer's quota and costs less for a summarising task")
    ap.add_argument("--facts", default="research/facts_pure.json",
                    help="dataset facts for the brief, produced by agent.facts")
    ap.add_argument("--no-memory", action="store_true", help="ignore this agent's prior runs")
    ap.add_argument("--dry-run", action="store_true", help="use a canned LLM, no network")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu",
                    help="execution device exposed to generated scripts (default: cpu)")
    ap.add_argument("--replay", default=None,
                    help="replay a previous run's llm_calls.jsonl: no network, no cost. Tests loop/parsing/reporting changes, NOT prompt changes")
    ap.add_argument("--replay-strict", action="store_true",
                    help="with --replay, fail if a prompt differs from the recording")
    args = ap.parse_args()

    try:
        hardware = resolve_device("cpu" if args.dry_run else args.device)
    except RuntimeError as exc:
        ap.error(str(exc))
    os.environ["AGENT_DEVICE"] = hardware["device"]
    print("execution device: " + (
        f"cuda ({hardware['gpu_name']}, {hardware['gpu_memory_gb']:.2f} GiB)"
        if hardware["device"] == "cuda" else "cpu"
    ))

    if args.dry_run:
        args.baseline_valid = _dry_run_baseline()
        print(f"dry-run control baseline: {args.baseline_valid:.4f} (item popularity)")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        complete = FakeComplete([(
            "HYPOTHESIS: item-popularity prior as a scoring baseline\n"
            "```python\n"
            "import numpy as np, json, time\n"
            "from pipeline.data import load\n"
            "from pipeline.evaluate import evaluate\n"
            "t0=time.perf_counter()\n"
            "tr=load('train'); va=load('valid')\n"
            "pop=np.bincount(tr.X['video_id'], weights=tr.y.astype(float))\n"
            "cnt=np.maximum(np.bincount(tr.X['video_id']),1)\n"
            "rate=pop/cnt\n"
            "vid=np.minimum(va.X['video_id'], len(rate)-1)\n"
            "valid_scores=rate[vid]\n"
            "m=evaluate(va.user_id, va.y, valid_scores)\n"
            "import os\n"
            "out=os.environ.get('ITER_OUT')\n"
            "if out:\n"
            " np.save(os.path.join(out,'scores_valid.npy'),valid_scores.astype(float))\n"
            " te=load('test'); tvid=np.minimum(te.X['video_id'],len(rate)-1)\n"
            " np.save(os.path.join(out,'scores_test.npy'),rate[tvid].astype(float))\n"
            "print('METRICS', json.dumps({'primary':m['primary'],'gauc':m['gauc'],"
            "'ndcg@5':m['ndcg@5'],'gpu_seconds':time.perf_counter()-t0,'device':'cpu'}))\n"
            "```", 900, 300)])
    elif args.replay:
        complete = ReplayComplete(args.replay, strict=args.replay_strict)
        print(f"REPLAY from {args.replay}: {len(complete.records)} recorded calls, no network")
    else:
        complete = make_complete()

    # every prompt/response lands in the run log -- it is a graded deliverable
    complete = RecordingComplete(complete, run_dir / "llm_calls.jsonl")

    # belief revision rewrites a claim list; it does not have to be the model that writes the
    # experiments. Keeping it on a separate model spends a separate per-model quota.
    if args.revision_model and not args.dry_run and not args.replay:
        import os as _os
        from agent.llm import OpenAICompatComplete
        revise_complete = RecordingComplete(
            OpenAICompatComplete(model=args.revision_model), run_dir / "llm_calls.jsonl")
        print(f"belief revision routed to {args.revision_model} "
              f"(proposer stays on {getattr(complete, 'model', '?')})")
    else:
        revise_complete = complete

    # what a run could reach. Recorded so a later run can tell which capabilities post-date the
    # experiments in its memory, instead of reading their absence as evidence against them.
    from pipeline.data import NUMERIC_FEATURES
    _probe = _load_probe()
    api_surface = sorted(_probe)
    memory = "" if args.no_memory else distil("runs", exclude=run_dir.name,
                                              baseline=args.baseline_valid,
                                              api_surface=api_surface)
    if memory:
        print(f"cross-run memory: {len(memory.splitlines())} lines from this agent's prior runs")

    facts = json.loads(Path(args.facts).read_text(encoding="utf-8"))
    facts["hardware_note"] = prompt_hardware_note(hardware)
    print(f"dataset: {facts.get('variant', '?')}  "
          f"train {facts['train_rows']:,} / valid {facts['valid_rows']:,} / test {facts['test_rows']:,}")
    proposer = LLMProposer(complete, kb_papers=load_papers(), timeout=args.timeout,
                           baseline=args.baseline_valid, facts=facts)
    ledger = Ledger(run_dir / "ledger.jsonl")
    knowledge = Knowledge()

    t0 = time.perf_counter()
    r = run_loop(
        proposer, SavedScoresEvaluator(), ledger,
        workdir=run_dir / "scripts",
        primary="primary",
        epsilon=args.epsilon,
        patience=args.patience,
        max_iters=args.iters,
        timeout=args.timeout,
        wall_clock_limit_s=args.wall_clock_s,
        tree=Tree(max_misses=args.max_misses, epsilon=args.epsilon),
        recovery=Recovery(max_retries=args.max_retries),
        knowledge=knowledge,
        revise_fn=lambda entries, last, findings, stale, patience: knowledge.revise(
            revise_complete, entries, last, findings, stale, patience),
        memory=memory,
        baseline=args.baseline_valid,
        baseline_tolerance=args.baseline_tolerance,
        # the perfect-ranking ceiling, measured by agent.facts. The critic needs it to
        # tell an extraordinary result from an impossible one.
        ceiling=facts.get("ceiling"),
    )
    wall = time.perf_counter() - t0

    sub = _write_submission(run_dir, ledger, args.baseline_test)
    entries = ledger.read()
    if r.tree is not None:
        (run_dir / "search_tree.txt").write_text(r.tree.render(), encoding="utf-8")
    if r.knowledge is not None and r.knowledge.claims:
        (run_dir / "knowledge.json").write_text(r.knowledge.to_json(), encoding="utf-8")
        (run_dir / "knowledge.md").write_text(
            "# What the agent established\n\n" + r.knowledge.render() + "\n", encoding="utf-8")

    t = ledger.totals()
    meta = {
        "model": getattr(complete, "model", "unknown"),
        "dataset": facts.get("variant", "unknown"),
        "api_surface": api_surface,
        "data_contract": "train-only-v1",
        "provider": ("replay" if args.replay else "dry-run" if args.dry_run
                     else __import__("os").environ.get("LLM_PROVIDER", "anthropic")),
        "stop_reason": r.stop_reason,
        "iterations": len(entries),
        "iteration_cap": args.iters,
        "wall_clock_s": wall,
        "wall_clock_h": wall / 3600.0,
        "script_seconds": r.script_seconds,
        # Conservative allocation-time upper bound. A CUDA script may spend some of its wall
        # time in CPU preprocessing, but this never understates allocated GPU time.
        "gpu_hours": r.script_seconds / 3600.0 if hardware["device"] == "cuda" else 0.0,
        "hardware": hardware,
        # the ledger counts proposer calls; r.llm_tokens_* also include knowledge revision
        "proposer_tokens_in": t["tokens_in"],
        "proposer_tokens_out": t["tokens_out"],
        "tokens_in": r.llm_tokens_in,
        "tokens_out": r.llm_tokens_out,
        "tokens_total": r.llm_tokens_in + r.llm_tokens_out,
        "eda_completed": r.eda_ok,
        "candidates_evaluated": r.candidates_evaluated,
        "claims_established": len(r.knowledge.claims) if r.knowledge else 0,
        "baseline_reproduced": r.baseline_primary,
        "baseline_target": args.baseline_valid,
        "manual_interventions": 0,
        "failures": sum(1 for e in entries if e.status in ("failed", "blacklisted", "rejected")),
        "integrity_rejections": sum(1 for e in entries if e.status == "rejected"),
        "submission": sub,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nstopped: {r.stop_reason}")
    print(f"iterations: {len(entries)} of {args.iters}   "
          f"failures: {meta['failures']}")
    print(f"agent wall-clock: {wall / 60:.1f} min   script time: {r.script_seconds / 60:.1f} min")
    print(f"tokens (incl. knowledge revision): {r.llm_tokens_in + r.llm_tokens_out:,}")
    print(f"EDA completed: {r.eda_ok}   baseline reproduced: {r.baseline_primary}")
    print(f"manual interventions: 0")
    print(f"run log: {run_dir}")


if __name__ == "__main__":
    main()
