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


def _entries(runs_dir: Path, exclude: str | None, require_marker: bool):
    for led in sorted(runs_dir.glob("*/ledger.jsonl")):
        if exclude and led.parent.name == exclude:
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


def distil(runs_dir="runs", exclude: str | None = None, baseline: float = 0.6016,
           require_marker: bool = True) -> str:
    """A compact record of prior runs: what scored, what lost, how things broke."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return ""
    scored, crashes = [], {}
    for run, e in _entries(runs_dir, exclude, require_marker):
        p = (e.get("metrics") or {}).get("primary")
        if isinstance(p, (int, float)):
            scored.append((p, e.get("hypothesis", "")[:100], run))
        elif e.get("status") in ("failed", "blacklisted"):
            first = (e.get("error") or "").strip().splitlines()
            key = next((ln.strip()[:90] for ln in reversed(first) if ln.strip()), "")
            if key:
                crashes[key] = crashes.get(key, 0) + 1
    if not scored:
        return ""

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
        out.append("\nBeat baseline by more than noise:")
        out += [f"  {p:.4f}  {h}" for p, h, _ in wins]
    if flat:
        out.append("\nTried and landed INSIDE noise -- these moved nothing, so repeating them "
                   "costs an iteration and returns no information:")
        out += [f"  {p:.4f}  {h}" for p, h, _ in flat]
    if dead:
        out.append("\nLost to baseline by more than noise (do not simply repeat these):")
        out += [f"  {p:.4f}  {h}" for p, h, _ in reversed(dead)]
    if crashes:
        top = sorted(crashes.items(), key=lambda kv: -kv[1])[:MAX_CRASH]
        out.append("\nMost frequent crashes in earlier runs:")
        out += [f"  x{n}  {k}" for k, n in top]
    return "\n".join(out)


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
        assert "noise-level" not in distil(root), "r2 has no marker, so it is excluded"
    print("ok")


if __name__ == "__main__":
    demo()
