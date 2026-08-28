"""Executor safety: python -m tests.test_executor

Generated code is written by a model and run unsandboxed. Anything the controller can read from
its environment, that script can read too -- and send anywhere.
"""
import os
import tempfile
from pathlib import Path

from agent.executor import _strip_credentials, run_script


def test_credentials_never_reach_a_generated_script():
    """Measured on this harness before the fix: a generated script could read OPENAI_API_KEYS,
    OPENAI_API_KEY, GITHUB_PACKAGE_TOKEN and CLAUDE_CODE_MESSAGING_TOKEN. None of them are
    needed to train a model, and a model-written script is exactly the wrong thing to trust
    with a GitHub token."""
    env = {
        "OPENAI_API_KEY": "sk-proj-secret", "OPENAI_API_KEYS": "sk-a,sk-b",
        "GITHUB_PACKAGE_TOKEN": "ghp_secret", "CLAUDE_CODE_MESSAGING_TOKEN": "tok",
        "AWS_SECRET_ACCESS_KEY": "x", "DB_PASSWORD": "y", "MY_AUTH_HEADER": "z",
        "PATH": "/usr/bin", "SystemRoot": r"C:\Windows", "TEMP": "/tmp",
        "PYTHONPATH": "/proj", "OPENAI_MODEL": "gpt-5.6-sol", "HOME": "/home/u",
    }
    clean = _strip_credentials(env)
    for leaked in ("OPENAI_API_KEY", "OPENAI_API_KEYS", "GITHUB_PACKAGE_TOKEN",
                   "CLAUDE_CODE_MESSAGING_TOKEN", "AWS_SECRET_ACCESS_KEY",
                   "DB_PASSWORD", "MY_AUTH_HEADER"):
        assert leaked not in clean, leaked
    # and the machinery a training script genuinely needs must survive
    for kept in ("PATH", "SystemRoot", "TEMP", "PYTHONPATH", "HOME", "OPENAI_MODEL"):
        assert kept in clean, kept


def test_a_script_really_cannot_see_them_end_to_end():
    """The unit check above tests the filter; this tests the path a real iteration takes."""
    os.environ["OPENAI_API_KEYS"] = "sk-proj-canary-value"
    try:
        d = Path(tempfile.mkdtemp())
        s = d / "probe.py"
        s.write_text("import os\nprint('SEEN', 'sk-proj-canary-value' in "
                     "''.join(os.environ.values()))\n", encoding="utf-8")
        r = run_script(s, timeout=60)
        assert "SEEN False" in r.stdout, r.stdout + r.stderr
    finally:
        os.environ.pop("OPENAI_API_KEYS", None)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok: {t.__name__}")
    print(f"{len(tests)} tests passed")
