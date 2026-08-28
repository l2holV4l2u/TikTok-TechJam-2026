import re
from dataclasses import dataclass, field

RETRY = "retry"
BLACKLIST = "blacklist"

# a hypothesis names a method; that name, not the sentence, identifies the idea
_ALIASES = {
    "lambdarank": "lambdarank", "lambdamart": "lambdarank", "listwise": "lambdarank",
    "ranknet": "lambdarank", "bpr": "bpr", "pairwise": "bpr",
    "deepfm": "deepfm", "xdeepfm": "xdeepfm", "dcn": "dcn", "dcnv2": "dcn",
    "autoint": "autoint", "fibinet": "fibinet", "pnn": "pnn", "nfm": "nfm",
    "din": "din", "dien": "dien", "bst": "bst", "sasrec": "sasrec", "gru4rec": "gru4rec",
    "mmoe": "mmoe", "ple": "ple", "esmm": "esmm", "multitask": "mmoe",
    "lightgbm": "gbdt", "xgboost": "gbdt", "gbdt": "gbdt",
    "focal": "focal", "softmax": "sampled_softmax", "negative": "negative_sampling",
    "encoding": "target_encoding", "encodings": "target_encoding",
    "dropout": "regularisation", "decay": "regularisation", "regularization": "regularisation",
    "regularisation": "regularisation", "l2": "regularisation",
    "calibration": "calibration", "calibrating": "calibration", "centering": "calibration",
    "senet": "senet", "se": "senet",
    "attention": "attention", "transformer": "attention",
}
_STOPWORDS = {"a", "an", "the", "to", "for", "of", "on", "in", "with", "and", "using", "use",
              "implement", "implementing", "add", "adding", "apply", "applying", "directly",
              "improve", "improving", "based", "style", "inspired", "loss", "function", "model"}
_WORD_RE = re.compile(r"[a-z0-9]+")
SIMILARITY = 0.6


def tail(s: str, n: int = 2000) -> str:
    return s[-n:]


def idea_key(hypothesis: str) -> frozenset[str]:
    """Prefer the named method. Falls back to content words when no known method is mentioned."""
    words = _WORD_RE.findall(hypothesis.lower())
    methods = {_ALIASES[w] for w in words if w in _ALIASES}
    if methods:
        return frozenset(methods)
    return frozenset(w for w in words if w not in _STOPWORDS)


def _overlap(a: frozenset[str], b: frozenset[str]) -> float:
    """Overlap coefficient, not Jaccard: the model restates one idea at very different lengths,
    and Jaccard reads a wordier rephrasing of the same hypothesis as a different idea."""
    if not a or not b:
        return 1.0 if a == b else 0.0
    return len(a & b) / min(len(a), len(b))


class IdeaSet(set):
    """A set of hypothesis strings whose membership test matches by named method, not exact text.

    The proposer already does `hyp in blacklist`; overriding __contains__ makes that check
    similarity-aware without changing the proposer or the loop.
    """

    def __init__(self, similarity: float = SIMILARITY):
        super().__init__()
        self.similarity = similarity
        self.keys: list[frozenset[str]] = []

    def add(self, hypothesis: str) -> None:
        super().add(hypothesis)
        k = idea_key(hypothesis)
        if k and k not in self.keys:
            self.keys.append(k)

    def __contains__(self, hypothesis: object) -> bool:
        if not isinstance(hypothesis, str):
            return False
        t = idea_key(hypothesis)
        return any(_overlap(t, k) >= self.similarity for k in self.keys)


@dataclass
class Recovery:
    """Keeps failures in-loop. Every escalation to a human costs us Autonomy points."""
    max_retries: int = 2
    max_underperforms: int = 3   # an idea that runs fine but keeps losing is still a dead end
    proven_retries: int = 6      # a proven idea gets more room to be debugged, not infinite
    similarity: float = SIMILARITY
    _attempts: dict[frozenset[str], int] = field(default_factory=dict)
    _underperforms: dict[frozenset[str], int] = field(default_factory=dict)
    blacklist: IdeaSet = field(default_factory=IdeaSet)
    _blacklisted: list[frozenset[str]] = field(default_factory=list)
    _proven: list[frozenset[str]] = field(default_factory=list)

    def _match(self, t: frozenset[str], keys) -> frozenset[str] | None:
        best, score = None, self.similarity
        for k in keys:
            s = _overlap(t, k)
            if s >= score:
                best, score = k, s
        return best

    def on_failure(self, key: str, stderr: str) -> tuple[str, str]:
        t = idea_key(key)
        k = self._match(t, self._attempts) or t
        n = self._attempts.get(k, 0) + 1
        self._attempts[k] = n
        if n <= self.max_retries:
            return RETRY, tail(stderr)
        # a family that has already scored well is failing on code bugs, not on being wrong, so
        # give it a wider budget -- but not an unlimited one, or a bug the model cannot fix
        # loops until the run dies
        if self._match(t, self._proven) is not None and n <= self.proven_retries:
            return RETRY, tail(stderr)
        self.blacklist.add(key)
        self._blacklisted.append(k)
        return BLACKLIST, tail(stderr)

    def on_underperform(self, key: str) -> bool:
        """Scored, but below the incumbent. Retiring these is what stops a run grinding on
        one losing idea -- crash-based blacklisting never fires for code that works."""
        t = idea_key(key)
        k = self._match(t, self._underperforms) or t
        n = self._underperforms.get(k, 0) + 1
        self._underperforms[k] = n
        if n >= self.max_underperforms:
            self.blacklist.add(key)
            self._blacklisted.append(k)
            return True
        return False

    def is_blacklisted(self, hypothesis: str) -> bool:
        return hypothesis in self.blacklist

    def on_success(self, key: str) -> None:
        t = idea_key(key)
        k = self._match(t, self._attempts)
        if k is not None:
            self._attempts.pop(k, None)
        if t and self._match(t, self._proven) is None:
            self._proven.append(t)
