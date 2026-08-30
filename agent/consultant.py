"""One synthesis per turn across every slot and the archive.

`knowledge.py` revises one belief set from one trajectory. A portfolio has several running at
once plus a pile of retired ones, and the thing its slots most lack is not literature -- they
already receive the whole 28-paper catalogue every turn -- but knowledge of EACH OTHER. Three
lines that independently rediscover the same idea have spent three slots on one result.

So this is not a literature oracle. An adviser seeded with what works on this dataset would be
a human prior wearing a robe, and the no-priors guarantee is what the Autonomy claim rests on.
It reads only what the agent's own experiments produced: the turn's results, the live slots,
the archive, and the measured correlation between them. It emits the shared belief set and one
short note per slot saying what is already covered.

One call per turn rather than one per slot. Belief revision is already ~37% of a run's
requests; n of them would triple that for a summarising task, and Feasibility is scored.
"""
from __future__ import annotations

import json
import re

from .knowledge import MAX_CLAIMS, Knowledge, _coerce

MAX_NOTE_CHARS = 300
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

_PROMPT = """You maintain the shared information state of an autonomous ML research agent that
is advancing several solution lineages at once against the same validation split.

Metric: primary = mean(GAUC, nDCG@5), within-user ranking. A difference below {epsilon} is
inside seed noise and is not evidence of anything.
{budget}

THIS TURN'S RESULTS (one line per slot)
{results}

LIVE SLOTS
{live}

{archive}

HOW MUCH THE LIVE SLOTS AGREE
{correlation}

WHAT YOU CURRENTLY BELIEVE
{claims}

Return ONE JSON object, nothing else:
{{"claims": [{{"text": "<one specific, falsifiable claim about this task>",
              "status": "active" | "qualified" | "invalidated",
              "evidence": [<iteration numbers>]}}],
  "notes": {{"<slot_id>": "<at most {max_note} characters, for that slot's next proposal>"}}}}

Rules for claims:
- Carry forward every claim that still stands. Do not silently drop one.
- If a result contradicts a claim, set it to "invalidated" and rewrite the text to say what is
  believed instead. If it holds only in a narrower case, mark it "qualified" and state the scope.
- A result inside noise supports "X changes nothing measurable", not "X helps" or "X hurts".
- At most {max_claims} claims; merge overlapping ones.

Rules for notes:
- One note per live slot id. A note says what its siblings already cover, what the archive has
  ruled out, or which catalogue method composes with a stored result.
- A note may not assert a fact about this dataset that no iteration measured.
- If the slots are ranking validation almost identically, say so and tell them to separate."""


def _slot_lines(slots) -> str:
    rows = []
    for s in slots:
        best = f"{s.best:.4f}" if s.best > float("-inf") else "nothing scored yet"
        rows.append(f"  slot {s.slot_id} ({s.origin}): best {best}, {s.stale} turn(s) without a "
                    f"gain, {len(s.lineage)} experiment(s). Last: {s.last_hypothesis[:110]}")
    return "\n".join(rows) or "  none"


def _result_lines(results) -> str:
    rows = []
    for r in results or []:
        score = r.get("primary")
        got = f"{score:.4f}" if isinstance(score, (int, float)) else r.get("status", "no score")
        rows.append(f"  slot {r.get('slot_id')} (#{r.get('iter_id')}): {got} :: "
                    f"{str(r.get('hypothesis', ''))[:110]}")
    return "\n".join(rows) or "  nothing scored this turn"


def _correlation_line(correlation) -> str:
    if not correlation or correlation.get("mean") is None:
        return "  not measurable this turn (fewer than two slots produced predictions)"
    pairs = ", ".join(f"{k} {v:.3f}" for k, v in sorted(correlation["pairs"].items()))
    return (f"  mean within-user rank correlation {correlation['mean']:.3f} "
            f"(max {correlation['max']:.3f}); pairs: {pairs}")


def _budget(stale, patience) -> str:
    if stale is None or patience is None:
        return ""
    return (f"The run ends after {patience} consecutive turns without a gain above 0.002; "
            f"{stale} are already used.")


def revise(complete, knowledge: Knowledge, slots, results, archive=None, correlation=None,
           stale=None, patience=None, epsilon: float = 0.002) -> tuple[int, int, dict]:
    """Returns (tokens_in, tokens_out, {slot_id: note}).

    Never raises. An LLM outage or a malformed reply leaves the previous belief set and the
    previous notes standing rather than ending the run -- the same contract knowledge.revise
    has, for the same reason.
    """
    prompt = _PROMPT.format(
        epsilon=epsilon,
        budget=_budget(stale, patience),
        results=_result_lines(results),
        live=_slot_lines(slots),
        archive=(archive.summary() if archive else "") or "NOTHING RETIRED YET",
        correlation=_correlation_line(correlation),
        claims=knowledge.render(),
        max_claims=MAX_CLAIMS,
        max_note=MAX_NOTE_CHARS,
    )
    try:
        text, ti, to = complete(prompt)
    except Exception:
        return 0, 0, {}
    m = _JSON_OBJ_RE.search(text or "")
    if not m:
        return ti, to, {}
    try:
        payload = json.loads(m.group(0))
    except json.JSONDecodeError:
        return ti, to, {}
    if not isinstance(payload, dict):
        return ti, to, {}

    revised = _coerce(payload.get("claims"))
    if revised:
        knowledge.claims = revised

    notes: dict[int, str] = {}
    raw = payload.get("notes")
    if isinstance(raw, dict):
        live = {s.slot_id for s in slots}
        for key, value in raw.items():
            try:
                slot_id = int(key)
            except (TypeError, ValueError):
                continue
            if slot_id in live and isinstance(value, str) and value.strip():
                notes[slot_id] = value.strip()[:MAX_NOTE_CHARS]
    return ti, to, notes


def demo() -> None:
    from .portfolio import Slot

    slots = [Slot(slot_id=0, best=0.604, last_hypothesis="deep cross network"),
             Slot(slot_id=1, best=0.601, last_hypothesis="recency weighting")]
    results = [{"slot_id": 0, "iter_id": 4, "primary": 0.604, "hypothesis": "deep cross"},
               {"slot_id": 1, "iter_id": 5, "primary": 0.601, "hypothesis": "recency"}]
    corr = {"mean": 0.97, "max": 0.97, "pairs": {"0-1": 0.97}}
    seen = []

    def reply(payload):
        def fn(prompt):
            seen.append(prompt)
            return (payload, 120, 60)
        return fn

    k = Knowledge()
    ti, to, notes = revise(reply(json.dumps({
        "claims": [{"text": "both live lines rank validation nearly identically",
                    "status": "active", "evidence": [4, 5]}],
        "notes": {"0": "slot 1 already covers recency; take a different stage",
                  "1": "slot 0 covers explicit crosses"},
    })), k, slots, results, correlation=corr, stale=1, patience=3)
    assert (ti, to) == (120, 60)
    assert len(k.claims) == 1 and notes == {
        0: "slot 1 already covers recency; take a different stage",
        1: "slot 0 covers explicit crosses"}, notes
    assert "0.970" in seen[0] and "1 are already used" in seen[0]
    assert "deep cross network" in seen[0] and "recency weighting" in seen[0]

    before = list(k.claims)
    _, _, n2 = revise(reply("no json at all"), k, slots, results)
    assert k.claims == before and n2 == {}, "an unparseable reply changes nothing"

    _, _, n3 = revise(reply('{"claims": [], "notes": {"7": "not a live slot"}}'),
                      k, slots, results)
    assert n3 == {} and k.claims == before, "a note for a dead slot is dropped"

    long_note = "x" * 900
    _, _, n4 = revise(reply(json.dumps({"claims": [], "notes": {"0": long_note}})),
                      k, slots, results)
    assert len(n4[0]) == MAX_NOTE_CHARS, len(n4[0])

    def boom(_):
        raise RuntimeError("429")
    assert revise(boom, k, slots, results) == (0, 0, {}) and k.claims == before
    print("ok  consultant.revise")


if __name__ == "__main__":
    demo()
