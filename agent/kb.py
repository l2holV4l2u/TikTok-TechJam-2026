import json
import re
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "kb" / "papers.json"

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def load_papers(path=None) -> list[dict]:
    p = Path(path) if path else DEFAULT_PATH
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def _score(query_tokens: set[str], paper: dict) -> int:
    tag_tokens = _tokens(" ".join(paper.get("tags", [])))
    text_tokens = _tokens(
        paper.get("title", "") + " " + paper.get("summary", "") + " " + paper.get("applies_to", "")
    )
    return 3 * len(query_tokens & tag_tokens) + len(query_tokens & text_tokens)


def retrieve(query: str, k: int = 3, papers: list[dict] | None = None,
             seen: set[str] | None = None) -> list[dict]:
    """Top-k papers by keyword/tag overlap, preferring ones not shown before.

    Without the `seen` filter this returns the same handful every iteration, because the query
    is dominated by the agent's own recent hypotheses -- so a run that starts on factorisation
    machines is never shown anything but factorisation machines. Excluding what has already
    been surfaced widens coverage without expressing any preference about which paper is right.
    """
    papers = papers if papers is not None else load_papers()
    q = _tokens(query)
    scored = [(_score(q, p), i, p) for i, p in enumerate(papers)]
    hits = [t for t in scored if t[0] > 0]
    hits.sort(key=lambda t: (-t[0], t[1]))
    ranked = [p for _, _, p in hits] or list(papers)
    if seen:
        fresh = [p for p in ranked if p["id"] not in seen]
        # once the corpus is exhausted, repeats are better than showing nothing
        ranked = fresh + [p for p in ranked if p["id"] in seen]
    return ranked[:k]


def index(papers: list[dict] | None = None) -> str:
    """One line per paper: the catalogue the agent can ask for, not a recommendation.

    It lists everything, including methods that do not apply to this data, so it steers
    nothing -- it only stops the agent being unable to know what literature is available.
    """
    papers = papers if papers is not None else load_papers()
    return "\n".join(f"  {p['id']}: {p['title'][:64]} ({p['year']})" for p in papers)
