"""Cross-run memory: what earlier runs of this agent already established.

Each run used to start blind, so the same dead ends were re-explored from scratch and the only
way knowledge accumulated was a human editing the prompt. This reads the agent's own ledgers
instead. It reads ONLY ledgers, never hand-written notes, so nothing here is a human finding
laundered through the agent.
"""
import json
from pathlib import Path

MAX_WINS = 8
MAX_DEAD = 8
MAX_CRASH = 4
MAX_FLAT = 10
NOISE = 0.002  # organizer epsilon; below this a difference is not a finding
UNSAFE_API_SENTINEL = "s.num[counts]"  # supplied full-month video outcome aggregates


def _entries(runs_dir: Path, exclude: str | None, require_marker: bool,
             excluded_runs: set[str] | None = None):
    for led in sorted(runs_dir.glob("*/ledger.jsonl")):
        if exclude and led.parent.name == exclude:
            continue
        if excluded_runs and led.parent.name in excluded_runs:
            continue
        # only runs produced by the current agent count. Earlier runs were driven by a brief
        # containing human-authored findings; distilling those would re-import a human's
        # research through the agent's own ledger and make the autonomy claim false.
        if require_marker and not (led.parent / "run_meta.json").exists():
            continue
        for line in led.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                try:
                    yield led.parent.name, json.loads(line)
                except json.JSONDecodeError:
                    continue


def _run_models(runs_dir: Path) -> dict:
    """Which model wrote each run.

    Results from three models sit in these ledgers (sol, luna, mini). A weak score from a
    weak proposer is evidence about the proposer, not about the method it named, and without
    the label the next run reads them as the same kind of fact.
    """
    out = {}
    for meta in runs_dir.glob("*/run_meta.json"):
        try:
            out[meta.parent.name] = json.loads(meta.read_text(encoding="utf-8")).get("model")
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _unsafe_runs(runs_dir: Path, exclude: str | None) -> set[str]:
    """Runs exposed to the dataset's full-month video aggregates cannot seed future search."""
    out = set()
    for meta in runs_dir.glob("*/run_meta.json"):
        if exclude and meta.parent.name == exclude:
            continue
        try:
            api = json.loads(meta.read_text(encoding="utf-8")).get("api_surface") or []
        except (json.JSONDecodeError, OSError):
            continue
        if UNSAFE_API_SENTINEL in api:
            out.add(meta.parent.name)
    return out


def _stale_capability_note(runs_dir: Path, exclude: str | None, current: list[str],
                           excluded_runs: set[str] | None = None) -> str:
    """Name the API the prior runs could not reach, so their silence on it is not read as a verdict.

    Memory ranks prior hypotheses by score, so a run that starts after the harness gains a new
    column sees a leaderboard of ideas built without it and re-proposes the old winner. That is
    exactly what happened the first time Split.num and Split.time_ms were added: the agent
    inventoried both in EDA, then spent its first experiment re-deriving the previous run's best
    idea. Absence of a capability from prior runs is not evidence against it.
    """
    seen: set[str] = set()
    for meta in sorted(runs_dir.glob("*/run_meta.json")):
        if exclude and meta.parent.name == exclude:
            continue
        if excluded_runs and meta.parent.name in excluded_runs:
            continue
        try:
            seen |= set(json.loads(meta.read_text(encoding="utf-8")).get("api_surface") or [])
        except (json.JSONDecodeError, OSError):
            continue
    added = sorted(set(current) - seen) if seen else sorted(current)
    if not added or not seen:
        return ""
    return ("\nNot available in any run above, so their results say nothing about it: "
            + ", ".join(added) + ".")


def distil(runs_dir="runs", exclude: str | None = None, baseline: float = 0.6016,
           require_marker: bool = True, api_surface: list[str] | None = None) -> str:
    """A compact record of prior runs: what scored, what lost, how things broke."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return ""
    scored, crashes = [], {}
    unsafe_runs = _unsafe_runs(runs_dir, exclude)
    models = _run_models(runs_dir)
    for run, e in _entries(runs_dir, exclude, require_marker, unsafe_runs):
        if e.get("status") == "rejected":
            continue
        p = (e.get("metrics") or {}).get("primary")
        if isinstance(p, (int, float)):
            # Depth matters as much as the score. r35's best hypothesis was its SIXTH
            # iteration, standing on a DeepFM + bagging + cross-model ensemble. Reported as a
            # bare line it read as "this works", and five later runs opened with it against a
            # plain five-field FM, gained nothing, and spent the attempt that decides whether a
            # run earns more. A hypothesis without what it was built on is not a finding.
            # depth counts EXPERIMENTS, not ledger rows: the baseline reproduction is step 0,
            # so the first real experiment of a run is step 1.
            is_improve = e.get("phase", "improve") == "improve"
            depth = (sum(1 for x in scored if x[2] == run and x[4]) + 1) if is_improve else 0
            scored.append((p, e.get("hypothesis", "")[:100], run, depth, is_improve,
                           models.get(run) or "unknown"))
        elif e.get("status") in ("failed", "blacklisted"):
            first = (e.get("error") or "").strip().splitlines()
            key = next((ln.strip()[:90] for ln in reversed(first) if ln.strip()), "")
            if key:
                crashes[key] = crashes.get(key, 0) + 1
    if not scored:
        return ""
    _note = _stale_capability_note(runs_dir, exclude, api_surface or [], unsafe_runs)
    if unsafe_runs:
        _note += (f"\nExcluded {len(unsafe_runs)} prior run(s) whose API exposed full-month "
                  "video outcome aggregates overlapping validation/test.")

    scored.sort(reverse=True)
    wins = [s for s in scored if s[0] > baseline + NOISE][:MAX_WINS]
    dead = [s for s in scored if s[0] < baseline - NOISE][-MAX_DEAD:]
    # an experiment that landed inside the noise band is not a win or a loss, but "this was
    # tried and moved nothing" is exactly what stops the next run spending an iteration on it
    flat = [s for s in scored if baseline - NOISE <= s[0] <= baseline + NOISE][:MAX_FLAT]
    out = [f"PRIOR RUNS OF THIS AGENT ({len(scored)} scored experiments across "
           f"{len({s[2] for s in scored})} runs). Baseline is {baseline:.4f}; a gap under "
           f"{NOISE} is noise."]
    if wins:
        out.append("\nBeat baseline by more than noise. Each line names the step it was reached at and the model that proposed it; a weak score from a weak proposer is evidence about the proposer, not about the method. A result from "
                   "step 5 was reached by a script that already carried four earlier changes, so "
                   "repeating its hypothesis against a fresh baseline is a different experiment "
                   "and usually a much weaker one.")
        out += [f"  {p:.4f}  (step {d}, {mo})  {h}" for p, h, _, d, _, mo in wins]
    # The first experiment decides how long a run lives. Under the convergence rule a run gets
    # three attempts and earns more only by clearing epsilon, so an opening that gains nothing
    # costs the run. What earlier openings actually returned is a fact from their own ledgers --
    # reported so the agent can weigh it, not a recommendation about what to try.
    openers = sorted((x for x in scored if x[4] and x[3] == 1), reverse=True)
    if len(openers) >= 3:
        out.append("\nWhat the FIRST experiment of a run has returned, across every prior run. "
                   "This is the move that decides whether a run gets three attempts or six:")
        out += [f"  {p:.4f}  ({mo})  {h}" for p, h, _, _, _, mo in openers[:MAX_WINS]]

    if flat:
        out.append("\nTried and landed INSIDE noise -- these moved nothing, so repeating them "
                   "costs an iteration and returns no information:")
        out += [f"  {p:.4f}  (step {d}, {mo})  {h}" for p, h, _, d, _, mo in flat]
    if dead:
        out.append("\nLost to baseline by more than noise (do not simply repeat these):")
        out += [f"  {p:.4f}  (step {d}, {mo})  {h}" for p, h, _, d, _, mo in reversed(dead)]
    if crashes:
        top = sorted(crashes.items(), key=lambda kv: -kv[1])[:MAX_CRASH]
        out.append("\nMost frequent crashes in earlier runs:")
        out += [f"  x{n}  {k}" for k, n in top]
    return "\n".join(out) + _note


def demo() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for run, rows in {
            "r1": [{"hypothesis": "rank blend of FM and GBDT", "metrics": {"primary": 0.6045},
                    "status": "ok", "error": None},
                   {"hypothesis": "pairwise BPR loss", "metrics": {"primary": 0.5935},
                    "status": "ok", "error": None},
                   {"hypothesis": "deepfm", "metrics": {}, "status": "failed",
                    "error": "Traceback\nValueError: shapes do not align"}],
            "r2": [{"hypothesis": "noise-level tweak", "metrics": {"primary": 0.6020},
                    "status": "ok", "error": None},
                   {"hypothesis": "deepfm again", "metrics": {}, "status": "failed",
                    "error": "Traceback\nValueError: shapes do not align"}],
        }.items():
            p = root / run / "ledger.jsonl"
            p.parent.mkdir(parents=True)
            p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        txt = distil(root, require_marker=False)
        assert "rank blend" in txt, "a real win must be carried forward"
        assert "pairwise BPR" in txt, "a real loss must be carried forward"
        assert "INSIDE noise" in txt and "noise-level tweak" in txt, (
            "an inside-noise result is not a win or a loss, but it must still be carried "
            "forward so the next run does not spend an iteration rediscovering it")
        assert "x2  ValueError: shapes do not align" in txt, "repeat crashes must be counted"
        assert "3 scored experiments across 2 runs" in txt

        assert "rank blend" not in distil(root, exclude="r1", require_marker=False)
        assert distil(root / "nope", require_marker=False) == "", "a missing runs dir is not an error"
        assert distil(root) == "", "a run with no run_meta.json marker is not this agent's"
        (root / "r1" / "run_meta.json").write_text("{}", encoding="utf-8")
        assert "rank blend" in distil(root), "a marked run does count"
        (root / "r1" / "run_meta.json").write_text(
            json.dumps({"model": "some-small-model"}), encoding="utf-8")
        assert "some-small-model" in distil(root), (
            "a remembered result must name the model that produced it: a weak score from "
            "a weak proposer is evidence about the proposer, not about the method")
        assert "noise-level" not in distil(root), "r2 has no marker, so it is excluded"

        p = root / "r3" / "ledger.jsonl"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"hypothesis": "future aggregate winner",
                                 "metrics": {"primary": 0.9}, "status": "ok"}),
                     encoding="utf-8")
        (p.parent / "run_meta.json").write_text(
            json.dumps({"api_surface": [UNSAFE_API_SENTINEL]}), encoding="utf-8")
        txt = distil(root)
        assert "future aggregate winner" not in txt
        assert "Excluded 1 prior run" in txt
    print("ok")


if __name__ == "__main__":
    demo()
