"""Offline tests for the LLM client: python -m tests.test_llm

No network. The daily-cap failover is exercised by substituting the transport, because the
behaviour that matters -- not burning twelve minutes of backoff on a cap that cannot clear --
is invisible from the outside until it costs a run.
"""
import os
from contextlib import contextmanager

import agent.llm as llm


@contextmanager
def _no_ambient_keys():
    """These tests construct clients explicitly; a real OPENAI_API_KEYS in the shell would be
    picked up instead and make the result depend on how the suite was invoked."""
    saved = {k: os.environ.pop(k, None) for k in ("OPENAI_API_KEYS", "OPENAI_API_KEY", "OPENAI_MODEL")}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v

DAILY = ('{"error": {"message": "Rate limit reached for gpt-5.6-sol in organization org-x on '
         'requests per day (RPD): Limit 50, Used 50, Requested 1.", "code": "rate_limit_exceeded"}}')
TPM = ('{"error": {"message": "Rate limit reached on tokens per min (TPM). '
       'Please try again in 1.2s.", "code": "rate_limit_exceeded"}}')


def _reply(text="hi", pin=11, pout=7):
    return {"choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": pin, "completion_tokens": pout}}


def _client(monkey_posts, keys=("k1", "k2")):
    """Returns (client, calls) where monkey_posts is a list of results or exceptions."""
    calls = []
    seq = list(monkey_posts)

    def fake_post(url, headers, body, timeout=None):
        calls.append(headers["Authorization"])
        got = seq.pop(0)
        if isinstance(got, Exception):
            raise got
        return got

    llm._post_json = fake_post
    return llm.OpenAICompatComplete(model="gpt-5.6-sol", api_keys=list(keys)), calls


def test_daily_cap_body_is_recognised():
    assert llm._is_daily_cap(DAILY)
    assert not llm._is_daily_cap(TPM)
    assert not llm._is_daily_cap("")


def test_failover_to_the_second_key_on_a_daily_cap():
    real = llm._post_json
    try:
        c, calls = _client([llm.LLMDailyLimit(DAILY), _reply("ok")])
        text, ti, to = c("prompt")
        assert (text, ti, to) == ("ok", 11, 7)
        assert calls == ["Bearer k1", "Bearer k2"], calls
        assert c.api_key == "k2", "the working key stays selected for later calls"
    finally:
        llm._post_json = real


def test_all_keys_exhausted_raises_rather_than_looping():
    real = llm._post_json
    try:
        c, calls = _client([llm.LLMDailyLimit(DAILY)] * 2)
        try:
            c("prompt")
        except llm.LLMDailyLimit:
            pass
        else:
            raise AssertionError("should raise once every key is capped")
        assert calls == ["Bearer k1", "Bearer k2"], calls
    finally:
        llm._post_json = real


def test_a_capped_key_is_not_retried_on_the_next_call():
    real = llm._post_json
    try:
        c, calls = _client([llm.LLMDailyLimit(DAILY), _reply("a"), _reply("b")])
        c("one")
        c("two")
        assert calls == ["Bearer k1", "Bearer k2", "Bearer k2"], calls
    finally:
        llm._post_json = real


def test_single_key_still_works_from_the_old_env_var():
    with _no_ambient_keys():
        c = llm.OpenAICompatComplete(model="gpt-4o", api_keys=None, api_key="solo")
        assert c.api_keys == ["solo"] and c.api_key == "solo"


def test_no_key_anywhere_is_an_explicit_error():
    with _no_ambient_keys():
        try:
            llm.OpenAICompatComplete(model="gpt-4o")
        except llm.LLMError as e:
            assert "OPENAI_API_KEY" in str(e)
        else:
            raise AssertionError("a missing key must fail loudly, not silently")


def test_recording_wrapper_exposes_the_model_for_the_run_log():
    """Which model produced a run is a graded deliverable; the log must be able to say."""
    import tempfile, os.path
    inner = llm.FakeComplete([("hi", 1, 1)])
    w = llm.RecordingComplete(inner, os.path.join(tempfile.mkdtemp(), "calls.jsonl"))
    assert w.model == "fake-offline"
    w("prompt")
    import json
    rec = json.loads(open(w.log_path, encoding="utf-8").read().strip())
    assert rec["model"] == "fake-offline", rec


def test_key_is_never_placed_in_the_body():
    real = llm._post_json
    try:
        seen = {}

        def fake_post(url, headers, body, timeout=None):
            seen["body"] = body
            return _reply()

        llm._post_json = fake_post
        llm.OpenAICompatComplete(model="gpt-5.6-sol", api_keys=["secret"])("p")
        assert "secret" not in str(seen["body"]), "a key must never reach the request body"
        assert seen["body"]["max_completion_tokens"], "gpt-5.x needs max_completion_tokens"
    finally:
        llm._post_json = real


def test_replay_returns_recorded_responses_and_refuses_to_overrun():
    import json, tempfile, os
    from agent.llm import ReplayComplete, LLMError
    recs = [{"prompt": "P1", "response": "R1", "tokens_in": 10, "tokens_out": 3},
            {"prompt": "P2", "response": "R2", "tokens_in": 20, "tokens_out": 4}]
    fd, path = tempfile.mkstemp(suffix=".jsonl"); os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

        c = ReplayComplete(path)
        assert c("P1") == ("R1", 10, 3)
        assert c("anything at all") == ("R2", 20, 4)   # lenient: plumbing test, not a prompt test
        try:
            c("P3"); assert False, "should refuse to invent a third response"
        except LLMError:
            pass

        # strict mode exists precisely so a changed prompt cannot pass silently
        strict = ReplayComplete(path, strict=True)
        assert strict("P1") == ("R1", 10, 3)
        try:
            strict("CHANGED"); assert False, "strict mode must reject a diverged prompt"
        except LLMError:
            pass
    finally:
        os.unlink(path)


def test_a_call_gives_up_instead_of_wedging_the_run():
    """One LLM call must not be able to hang a run indefinitely.

    Eight retries at up to 90s of backoff, each behind a 120s socket timeout, is ~28 minutes
    before the caller hears anything -- and a socket left dead by a suspended machine outlasted
    even that: an observed run sat 42 minutes with no child process and no progress. The loop
    has no way to recover from a call that never returns, so the call enforces its own ceiling.
    """
    import time as _time
    import urllib.error
    from agent import llm

    calls = {"n": 0}

    def always_down(*a, **k):
        calls["n"] += 1
        raise urllib.error.URLError("connection refused")

    real = llm.urllib.request.urlopen
    llm.urllib.request.urlopen = always_down
    try:
        t0 = _time.monotonic()
        try:
            llm._post_json("https://example.invalid", {}, {"x": 1}, total_deadline_s=1.0)
            raise AssertionError("should have given up")
        except llm.LLMRetryExhausted as e:
            elapsed = _time.monotonic() - t0
            assert elapsed < 8.0, f"took {elapsed:.1f}s despite a 1s deadline"
            assert "deadline" in str(e).lower(), str(e)
        # and it must not have burned the full retry ladder to get there
        assert calls["n"] <= llm.MAX_RETRIES, calls["n"]
    finally:
        llm.urllib.request.urlopen = real


def test_a_revoked_key_fails_over_instead_of_killing_the_run():
    """A key can be cancelled mid-run; that must cost one request, not the whole run.

    Failover previously triggered only on a daily-cap 429. A revoked key answers 401, which fell
    through to a plain LLMError, spending the loop's proposer-error budget while a perfectly good
    second key sat unused. One of the keys in rotation here was revoked after a leak, which is
    how this was found.
    """
    real = llm._post_json
    try:
        c, calls = _client([llm.LLMKeyRejected("HTTP 401: revoked"), _reply("ok")])
        assert c("prompt") == ("ok", 11, 7)
        assert calls == ["Bearer k1", "Bearer k2"], calls
        assert c.api_key == "k2", "the working key stays selected"

        # and when every key is refused it must raise, not spin
        c2, calls2 = _client([llm.LLMKeyRejected("401")] * 2)
        try:
            c2("prompt")
        except llm.LLMKeyRejected:
            pass
        else:
            raise AssertionError("all keys refused must raise")
        assert calls2 == ["Bearer k1", "Bearer k2"], calls2
    finally:
        llm._post_json = real


def test_rate_limit_exhaustion_fails_over_to_another_key():
    """A per-minute cap must not kill a run while other keys sit idle with quota.

    RPM exhaustion raises LLMRetryExhausted, not LLMDailyLimit, so the failover path never fired.
    Observed live: the free tier allows 3 requests per minute, two concurrent runs exceed it, and
    a run died with two of three keys still usable.
    """
    real = llm._post_json
    try:
        c, calls = _client([llm.LLMRetryExhausted("HTTP 429 after 8 retries: RPM"), _reply("ok")])
        assert c("prompt") == ("ok", 11, 7)
        assert calls == ["Bearer k1", "Bearer k2"], calls

        # A transient limit on every key is waited out, not given up on -- but only for a
        # bounded number of passes, after which the original exhaustion is the honest error.
        saved_waits, saved_interval = llm.MAX_TRANSIENT_WAITS, llm.MIN_REQUEST_INTERVAL_S
        llm.MAX_TRANSIENT_WAITS, llm.MIN_REQUEST_INTERVAL_S = 1, 0.0
        try:
            c2, _ = _client([llm.LLMRetryExhausted("429")] * 12)
            try:
                c2("prompt")
            except llm.LLMRetryExhausted:
                pass
            else:
                raise AssertionError("must eventually raise if every key stays limited")
        finally:
            llm.MAX_TRANSIENT_WAITS, llm.MIN_REQUEST_INTERVAL_S = saved_waits, saved_interval
    finally:
        llm._post_json = real


def test_requests_are_spaced_to_stay_under_the_per_minute_limit():
    """Losing an iteration to a 429 costs far more than waiting for a slot.

    The free tier allows 3 requests/minute; an iteration costs two calls plus any retries, so
    runs breached it within a couple of iterations. Measured across three runs, 2-5 iterations
    each were lost that way -- and iteration count is what tracks the final score.
    """
    import time as _time
    saved = llm.MIN_REQUEST_INTERVAL_S
    llm.MIN_REQUEST_INTERVAL_S = 0.3
    real = llm.urllib.request.urlopen
    stamps = []

    def fake_open(req, timeout=None):
        stamps.append(_time.monotonic())

        class R:
            def read(self):
                return (b'{"choices":[{"message":{"content":"ok"}}],'
                        b'"usage":{"prompt_tokens":1,"completion_tokens":1}}')
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return R()

    llm.urllib.request.urlopen = fake_open
    try:
        for _ in range(3):
            llm._post_json("https://example.invalid", {}, {})
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        assert gaps and all(g >= 0.29 for g in gaps), gaps
    finally:
        llm.urllib.request.urlopen = real
        llm.MIN_REQUEST_INTERVAL_S = saved


def test_a_transient_rate_limit_does_not_retire_a_good_key():
    """RPM clears in seconds; a daily cap does not. Conflating them kills runs.

    The first failover fix marked a key exhausted on any LLMRetryExhausted. One momentary
    per-minute 429 then retired the only key with quota, sent the call to keys that were
    genuinely capped for the day, and ended the run -- observed twice, with the good key idle.
    """
    real = llm._post_json
    try:
        c, calls = _client([llm.LLMRetryExhausted("HTTP 429: RPM"), _reply("ok")])
        assert c("prompt") == ("ok", 11, 7)
        assert calls == ["Bearer k1", "Bearer k2"], calls
        assert not c.exhausted, f"a transient limit must not retire a key: {c.exhausted}"

        # a daily cap, by contrast, must retire it
        c2, _ = _client([llm.LLMDailyLimit(DAILY), _reply("ok")])
        c2("prompt")
        assert c2.exhausted, "a daily cap must retire the key for the process"
    finally:
        llm._post_json = real


if __name__ == "__main__":
    for t in (test_daily_cap_body_is_recognised,
              test_failover_to_the_second_key_on_a_daily_cap,
              test_all_keys_exhausted_raises_rather_than_looping,
              test_a_capped_key_is_not_retried_on_the_next_call,
              test_single_key_still_works_from_the_old_env_var,
              test_no_key_anywhere_is_an_explicit_error,
              test_recording_wrapper_exposes_the_model_for_the_run_log,
              test_key_is_never_placed_in_the_body,
              test_replay_returns_recorded_responses_and_refuses_to_overrun,
              test_a_call_gives_up_instead_of_wedging_the_run,
              test_a_revoked_key_fails_over_instead_of_killing_the_run,
              test_rate_limit_exhaustion_fails_over_to_another_key,
              test_requests_are_spaced_to_stay_under_the_per_minute_limit,
              test_a_transient_rate_limit_does_not_retire_a_good_key):
        t()
        print(f"ok  {t.__name__}")
    print("all passed")
