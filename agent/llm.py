"""Real LLM client: complete(prompt) -> (text, tokens_in, tokens_out), stdlib only."""
import json
import re
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MAX_TOKENS = 8192  # blend scripts overflow 4096 and arrive truncated
DEFAULT_TIMEOUT_S = 120
MAX_RETRIES = 8
BASE_DELAY_S = 1.0
MAX_DELAY_S = 90.0
# A ceiling on ALL attempts for one call, not just on each one. Eight retries at up to 90s of
# backoff plus a 120s socket timeout each is ~28 minutes before the caller hears anything, and a
# socket left dead by a suspended machine can outlast its own timeout: an observed run sat 42
# minutes with no child process and no progress. The loop cannot recover from a call that never
# returns, so the call has to give up on its own.
TOTAL_DEADLINE_S = 420.0
# Free-tier accounts allow 3 requests per minute. An iteration costs two calls (propose, then
# revise beliefs) and every retry counts, so a run breaches the limit within a couple of
# iterations and then loses whole iterations to 429s -- measured across r45/r49/r51, which each
# lost 2-5 iterations that way. Waiting our turn is strictly cheaper than retrying into a wall:
# a 21s spacing costs ~40s per iteration, while one lost iteration costs the whole experiment.
MIN_REQUEST_INTERVAL_S = float(os.environ.get("LLM_MIN_INTERVAL_S", "21"))
_last_request_at = 0.0


def _wait_for_slot() -> None:
    """Space requests so we stay under the account's per-minute limit."""
    global _last_request_at
    if MIN_REQUEST_INTERVAL_S <= 0:
        return
    gap = time.monotonic() - _last_request_at
    if gap < MIN_REQUEST_INTERVAL_S:
        time.sleep(MIN_REQUEST_INTERVAL_S - gap)
    _last_request_at = time.monotonic()
CHARS_PER_TOKEN = 4  # fallback estimate (chars/4) used only when the API omits usage


class LLMError(Exception):
    """Non-retryable API error: 4xx other than 429, missing key, bad response shape."""


class LLMRetryExhausted(Exception):
    """429/5xx/connection errors persisted past MAX_RETRIES."""


class LLMDailyLimit(Exception):
    """A per-day request cap. Backing off cannot clear it; only another key or tomorrow can."""


class LLMKeyRejected(Exception):
    """The key itself was refused: revoked, disabled, or out of quota. Retrying cannot help."""


def _is_daily_cap(body: str) -> bool:
    low = (body or "").lower()
    return "per day" in low or "requests per day" in low or "rpd" in low


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _retry_delay(err, body: str, attempt: int) -> float:
    """Prefer the server's own hint: a 429 states how long to wait, guessing is worse."""
    hdr = getattr(err, "headers", None)
    if hdr is not None:
        for key in ("retry-after", "x-ratelimit-reset-tokens"):
            try:
                v = hdr.get(key)
            except Exception:
                v = None
            if v:
                try:
                    return min(MAX_DELAY_S, float(str(v).rstrip("s")) + 1.0)
                except ValueError:
                    pass
    m = re.search(r"try again in ([0-9.]+)(ms|s)", body or "")
    if m is not None:
        secs = float(m.group(1))
        if m.group(2) == "ms":
            secs /= 1000.0
        return min(MAX_DELAY_S, secs + 1.0)
    return min(MAX_DELAY_S, BASE_DELAY_S * (2 ** (attempt - 1)))


def _post_json(url: str, headers: dict, body: dict, timeout: float = DEFAULT_TIMEOUT_S,
               total_deadline_s: float = TOTAL_DEADLINE_S) -> dict:
    """POST JSON via urllib, retrying 429/5xx/connection errors with capped exponential backoff.

    Gives up once total_deadline_s has elapsed across all attempts, so one call can never wedge
    a run indefinitely.
    """
    data = json.dumps(body).encode("utf-8")
    attempt = 0
    started = time.monotonic()

    def _out_of_time() -> bool:
        return time.monotonic() - started >= total_deadline_s

    while True:
        _wait_for_slot()
        if _out_of_time():
            raise LLMRetryExhausted(
                f"gave up after {time.monotonic() - started:.0f}s across {attempt} attempts "
                f"(deadline {total_deadline_s:.0f}s)")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            if e.code == 429 and _is_daily_cap(err_body):
                # a per-day cap will not clear inside the backoff ladder -- waiting out eight
                # retries here costs ~12 minutes and still fails. Surface it at once so the
                # caller can switch to another key instead.
                raise LLMDailyLimit(err_body) from e
            if e.code in (401, 403):
                # a revoked or disabled key. Failover already exists for daily caps; without
                # this a key cancelled mid-run raises a plain LLMError and burns the loop's
                # proposer-error budget instead of moving to the next key. This is not
                # hypothetical -- one of the keys in rotation was revoked after a leak.
                raise LLMKeyRejected(f"HTTP {e.code}: {err_body[:200]}") from e
            if e.code == 429 or e.code >= 500:
                attempt += 1
                if attempt > MAX_RETRIES:
                    raise LLMRetryExhausted(f"HTTP {e.code} after {MAX_RETRIES} retries: {err_body}") from e
                delay = _retry_delay(e, err_body, attempt)
                if time.monotonic() - started + delay >= total_deadline_s:
                    raise LLMRetryExhausted(
                        f"HTTP {e.code}; next backoff would pass the {total_deadline_s:.0f}s "
                        f"deadline") from e
                time.sleep(delay)
                continue
            raise LLMError(f"HTTP {e.code}: {err_body}") from e
        except urllib.error.URLError as e:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise LLMRetryExhausted(f"connection error after {MAX_RETRIES} retries: {e}") from e
            delay = min(MAX_DELAY_S, BASE_DELAY_S * (2 ** (attempt - 1)))
            if time.monotonic() - started + delay >= total_deadline_s:
                raise LLMRetryExhausted(
                    f"connection error; next backoff would pass the "
                    f"{total_deadline_s:.0f}s deadline") from e
            time.sleep(delay)


class AnthropicComplete:
    """complete() over the Anthropic Messages API; model/key are env-overridable, key never logged."""

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY not set")
        self.max_tokens = max_tokens

    def __call__(self, prompt: str) -> tuple[str, int, int]:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = _post_json(ANTHROPIC_URL, headers, body)
        text = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
        usage = resp.get("usage") or {}
        tokens_in = usage.get("input_tokens")
        tokens_out = usage.get("output_tokens")
        if tokens_in is None:
            tokens_in = _estimate_tokens(prompt)
        if tokens_out is None:
            tokens_out = _estimate_tokens(text)
        return text, tokens_in, tokens_out


class OpenAICompatComplete:
    """complete() over an OpenAI-compatible /chat/completions endpoint; key never logged."""

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, max_tokens: int = DEFAULT_MAX_TOKENS,
                 api_keys: list[str] | None = None):
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        # OPENAI_API_KEYS holds one or more comma-separated keys. Each key is a separate
        # organization with its own per-day request cap, so failing over across them is the
        # difference between three runs a day and enough to finish an experiment.
        keys = api_keys or [k.strip() for k in
                            os.environ.get("OPENAI_API_KEYS", "").split(",") if k.strip()]
        if not keys:
            single = api_key or os.environ.get("OPENAI_API_KEY")
            keys = [single] if single else []
        if not keys:
            raise LLMError("no OPENAI_API_KEY or OPENAI_API_KEYS set")
        self.api_keys = keys
        self.key_index = 0
        self.exhausted: set[int] = set()
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)).rstrip("/")
        self.max_tokens = max_tokens

    @property
    def api_key(self) -> str:
        return self.api_keys[self.key_index]

    def _next_key(self) -> bool:
        """Move to a key that is still usable. False when none are.

        Retires the current key for the rest of the process, whether it hit a daily cap or was
        rejected outright -- in both cases retrying it costs a request and cannot succeed.
        """
        self.exhausted.add(self.key_index)
        for offset in range(1, len(self.api_keys) + 1):
            cand = (self.key_index + offset) % len(self.api_keys)
            if cand not in self.exhausted:
                self.key_index = cand
                return True
        return False

    def __call__(self, prompt: str) -> tuple[str, int, int]:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        # gpt-5.x rejects max_tokens and requires max_completion_tokens
        key = "max_completion_tokens" if self.model.startswith(("gpt-5", "o1", "o3", "o4")) else "max_tokens"
        body[key] = self.max_tokens
        while True:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            try:
                resp = _post_json(f"{self.base_url}/chat/completions", headers, body)
                break
            except (LLMDailyLimit, LLMKeyRejected):
                if not self._next_key():
                    raise
            except LLMRetryExhausted:
                # A per-minute rate limit exhausts the backoff ladder without ever raising
                # LLMDailyLimit, so failover never fired and a run died while other keys sat
                # idle with quota. Observed on the free tier: RPM is 3, and two concurrent runs
                # exceed it. Try the remaining keys before giving up; if none is left, the
                # original exhaustion is the honest error.
                if not self._next_key():
                    raise
        choice = (resp.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = resp.get("usage") or {}
        tokens_in = usage.get("prompt_tokens")
        tokens_out = usage.get("completion_tokens")
        if tokens_in is None:
            tokens_in = _estimate_tokens(prompt)
        if tokens_out is None:
            tokens_out = _estimate_tokens(text)
        return text, tokens_in, tokens_out


def make_complete(provider: str | None = None):
    """Builds a complete() callable per LLM_PROVIDER env var ('anthropic' default, or 'openai')."""
    provider = (provider or os.environ.get("LLM_PROVIDER", "anthropic")).lower()
    if provider == "anthropic":
        return AnthropicComplete()
    if provider == "openai":
        return OpenAICompatComplete()
    raise LLMError(f"unknown LLM_PROVIDER: {provider}")


class RecordingComplete:
    """Wraps a complete callable, appending every prompt/response pair to a JSONL run log."""

    def __init__(self, inner, log_path):
        self.inner = inner
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def model(self) -> str:
        return getattr(self.inner, "model", "unknown")

    def __call__(self, prompt: str) -> tuple[str, int, int]:
        text, tokens_in, tokens_out = self.inner(prompt)
        record = {
            "ts": time.time(),
            # which model produced this is part of the deliverable: the run log is graded, and
            # "APIs used" is a required field. Without it nobody can verify the claim later.
            "model": self.model,
            "prompt": prompt,
            "response": text,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return text, tokens_in, tokens_out


class ReplayComplete:
    """Replays a previous run's recorded responses in order: no network, no cost, no waiting.

    A full run costs ~30 minutes and real tokens, which makes it a terrible way to find out
    whether a change to the loop, the parsers, the ledger or the reporting works. Replay turns
    that into seconds, deterministically, against responses a real model actually produced.

    It validates PLUMBING, not prompting. A changed prompt still receives the response recorded
    for the old one, so replay can never tell you whether a prompt edit helps -- only that the
    machinery around it still runs. `strict=True` refuses to hide that, failing as soon as the
    prompt sent differs from the prompt recorded.
    """

    model = "replay"

    def __init__(self, log_path, strict: bool = False):
        self.records = [json.loads(line) for line in
                        Path(log_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.records:
            raise LLMError(f"no recorded calls in {log_path}")
        self.strict = strict
        self.n = 0

    def __call__(self, prompt: str) -> tuple[str, int, int]:
        if self.n >= len(self.records):
            raise LLMError(f"replay log exhausted after {len(self.records)} calls; "
                           "the run under replay asks for more than the recording holds")
        rec = self.records[self.n]
        self.n += 1
        if self.strict and rec.get("prompt") != prompt:
            raise LLMError(f"prompt diverged from the recording at call {self.n}")
        return rec.get("response", ""), rec.get("tokens_in", 0), rec.get("tokens_out", 0)


class FakeComplete:
    """Offline stand-in for tests: cycles canned (text, tokens_in, tokens_out) replies, records prompts seen."""

    model = "fake-offline"

    def __init__(self, replies: list[tuple[str, int, int]]):
        self.replies = list(replies)
        self.n = 0
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> tuple[str, int, int]:
        self.prompts.append(prompt)
        reply = self.replies[min(self.n, len(self.replies) - 1)]
        self.n += 1
        return reply
