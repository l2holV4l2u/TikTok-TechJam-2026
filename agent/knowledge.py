"""The agent's information state: what it currently believes, and how sure it is.

This replaces an append-only pile of reflections. The problem with append-only is visible in
our own logs: one run recommended the same next experiment four times running, because nothing
could ever overturn it. Iris (arXiv:2608.02143) makes exactly this the centre of the design --
research organised around a continually revised representation of what has been learned, rather
than around a tree of candidate solutions -- and reports the largest single ablation drop from
removing it (66.7% -> 53.3% any-medal on small-data tasks). Gome (arXiv:2603.01692) reaches for
the same thing as "success memory".

A claim carries a status, so evidence can demote it instead of silently piling up beside it:
  active       believed, and acted on
  qualified    true only in some scope, which the text must state
  invalidated  contradicted by later evidence; kept so it is not rediscovered
"""
import json
import re
from dataclasses import dataclass, field

MAX_CLAIMS = 14
MAX_CLAIM_CHARS = 340  # a claim is a deliverable judges read; 220 cut them mid-sentence
STATUSES = ("active", "qualified", "invalidated")
_JSON_RE = re.compile(r"\[.*\]", re.DOTALL)

_PROMPT = """You maintain the information state of an autonomous ML research agent on
KuaiRand-Pure. An experiment just finished. Revise what the agent believes.

Metric: primary = mean(GAUC, nDCG@5), within-user ranking. Official baseline validation 0.6016.
A difference below 0.002 is inside seed noise and is not evidence of anything.
{budget}

LATEST EXPERIMENT
hypothesis: {hypothesis}
outcome:    {outcome}
{findings}
RUN SO FAR (oldest first)
{history}

WHAT YOU CURRENTLY BELIEVE
{claims}

Return the COMPLETE revised belief set as a JSON array, nothing else. Each element:
  {{"text": "<one specific, falsifiable claim about this task>",
    "status": "active" | "qualified" | "invalidated",
    "evidence": [<iteration numbers that support or contradict it>]}}

Rules:
- Carry forward every claim that still stands. Do not silently drop one.
- If the new result contradicts a claim, set it to "invalidated" and rewrite the text to say
  what is now believed instead. If it holds only in a narrower case, mark it "qualified" and
  state the scope in the text.
- Add claims only for things the evidence actually supports. A result inside noise supports
  "X changes nothing measurable", not "X helps" or "X hurts".
- Prefer claims that would change what to try next. Drop restatements of the metric or the
  task. At most {max_claims} claims; merge overlapping ones."""


@dataclass
class Claim:
    text: str
    status: str = "active"
    evidence: list = field(default_factory=list)

    def line(self) -> str:
        ev = f" [iters {','.join(str(e) for e in self.evidence)}]" if self.evidence else ""
        return f"- ({self.status}) {self.text}{ev}"


def _coerce(raw) -> list[Claim]:
    """Never trust the model's shape. A malformed belief set must not kill the run."""
    out: list[Claim] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()[:MAX_CLAIM_CHARS]
        if not text:
            continue
        status = str(item.get("status", "active")).strip().lower()
        if status not in STATUSES:
            status = "active"
        ev = item.get("evidence")
        ev = [int(x) for x in ev if isinstance(x, (int, float))] if isinstance(ev, list) else []
        out.append(Claim(text, status, ev))
    return out[:MAX_CLAIMS]


def _outcome(entry) -> str:
    if entry.status in ("failed", "blacklisted"):
        return f"CRASHED ({entry.status}): {(entry.error or '')[:300]}"
    parts = [f"{k}={v:.4f}" for k, v in (entry.metrics or {}).items() if isinstance(v, float)]
    return f"scored {', '.join(parts)} in {entry.gpu_seconds:.0f}s"


def _history(entries, n: int = 12) -> str:
    return "\n".join(f"#{e.iter_id} {e.status} {_outcome(e)} :: {e.hypothesis[:70]}"
                     for e in entries[-n:]) or "nothing yet"


def _budget(stale, patience) -> str:
    if stale is None or patience is None:
        return ""
    return (f"The run ends after {patience} consecutive iterations without a gain above 0.002; "
            f"{stale} are already used.")


@dataclass
class Knowledge:
    claims: list[Claim] = field(default_factory=list)

    def render(self) -> str:
        if not self.claims:
            return "nothing established yet"
        live = [c for c in self.claims if c.status != "invalidated"]
        dead = [c for c in self.claims if c.status == "invalidated"]
        out = [c.line() for c in live]
        if dead:
            out.append("Ruled out by evidence (do not revisit without a new mechanism):")
            out += [c.line() for c in dead]
        return "\n".join(out)

    def to_json(self) -> str:
        return json.dumps([{"text": c.text, "status": c.status, "evidence": c.evidence}
                           for c in self.claims], indent=2)

    def revise(self, complete, entries, last, findings: str = "",
               stale=None, patience=None) -> tuple[int, int]:
        """Returns (tokens_in, tokens_out). Never raises: an LLM outage or a malformed reply
        leaves the previous belief set standing rather than ending the run."""
        prompt = _PROMPT.format(
            budget=_budget(stale, patience),
            hypothesis=last.hypothesis,
            outcome=_outcome(last),
            findings=f"\nWHAT THE SCRIPT REPORTED WHILE RUNNING\n{findings[:1500]}\n" if findings
                     else "",
            history=_history(entries),
            claims=self.render(),
            max_claims=MAX_CLAIMS,
        )
        try:
            text, ti, to = complete(prompt)
        except Exception:
            return 0, 0
        m = _JSON_RE.search(text or "")
        if not m:
            return ti, to
        try:
            revised = _coerce(json.loads(m.group(0)))
        except json.JSONDecodeError:
            return ti, to
        if revised:
            self.claims = revised
        return ti, to


def demo() -> None:
    from dataclasses import dataclass as dc

    @dc
    class E:
        iter_id: int
        status: str
        hypothesis: str
        metrics: dict
        gpu_seconds: float
        error: str = None

    ok = E(2, "reverted", "all 37 fields", {"primary": 0.6018}, 90.0)
    seen: list[str] = []

    def reply(payload):
        def fn(prompt):
            seen.append(prompt)
            return (payload, 100, 40)
        return fn

    k = Knowledge()
    ti, to = k.revise(reply(
        'Sure, here you go:\n[{"text": "Adding all 37 fields changes nothing measurable",'
        ' "status": "active", "evidence": [2]}]'), [ok], ok, stale=1, patience=3)
    assert (ti, to) == (100, 40)
    assert len(k.claims) == 1 and k.claims[0].status == "active"
    assert "1 are already used" in seen[0]
    assert "nothing established yet" in seen[0], "the first revision starts from an empty state"

    # a later result overturns it; the state must demote, not accumulate a contradiction
    k.revise(reply('[{"text": "37 fields helps only when author_id is dropped",'
                   ' "status": "qualified", "evidence": [2,5]}]'), [ok], ok)
    assert len(k.claims) == 1 and k.claims[0].status == "qualified"
    assert "Adding all 37 fields" in seen[1], "prior beliefs must be shown for revision"

    k.revise(reply('[{"text": "capacity is not the bottleneck", "status": "invalidated",'
                   ' "evidence": [7]}]'), [ok], ok)
    assert "Ruled out by evidence" in k.render()

    before = list(k.claims)
    k.revise(reply("no json here at all"), [ok], ok)
    assert k.claims == before, "an unparseable reply must leave beliefs untouched"
    k.revise(reply('[{"nope": 1}, "junk", {"text": ""}]'), [ok], ok)
    assert k.claims == before, "malformed elements must not wipe the state"

    def boom(prompt):
        raise RuntimeError("429")
    assert k.revise(boom, [ok], ok) == (0, 0) and k.claims == before

    k2 = Knowledge()
    k2.revise(reply('[{"text": "t", "status": "bogus", "evidence": "x"}]'), [ok], ok)
    assert k2.claims[0].status == "active" and k2.claims[0].evidence == []

    k3 = Knowledge()
    k3.revise(reply(json.dumps([{"text": f"c{i}", "status": "active"} for i in range(40)])),
              [ok], ok)
    assert len(k3.claims) == MAX_CLAIMS, "the state stays bounded"

    k4 = Knowledge()
    k4.revise(reply('[{"text": "x", "status": "active"}]'), [ok], ok, findings="rows=1141112")
    assert "rows=1141112" in seen[-1], "script findings must reach the revision step"
    print("ok")


if __name__ == "__main__":
    demo()
