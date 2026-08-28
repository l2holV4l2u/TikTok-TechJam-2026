import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class Entry:
    iter_id: int
    parent_iter_id: int | None
    tier: int
    hypothesis: str
    diff: str
    metrics: dict
    gpu_seconds: float
    tokens_in: int
    tokens_out: int
    status: str  # ok | failed | reverted | blacklisted
    error: str | None = None
    phase: str = "improve"  # eda | baseline | improve; default keeps old ledgers readable
    timestamp: float = field(default_factory=time.time)


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: Entry) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def read(self) -> list[Entry]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [Entry(**json.loads(ln)) for ln in f if ln.strip()]

    def totals(self) -> dict:
        e = self.read()
        return {
            "iterations": len(e),
            "gpu_seconds": sum(x.gpu_seconds for x in e),
            "tokens_in": sum(x.tokens_in for x in e),
            "tokens_out": sum(x.tokens_out for x in e),
        }
