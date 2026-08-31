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
        "OPENAI_API_KEY": "FAKE-KEY-FOR-TEST", "OPENAI_API_KEYS": "FAKE-A,FAKE-B",
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
    os.environ["OPENAI_API_KEYS"] = "FAKE-CANARY-NOT-A-KEY"
    try:
        d = Path(tempfile.mkdtemp())
        s = d / "probe.py"
        s.write_text("import os\nprint('SEEN', 'FAKE-CANARY-NOT-A-KEY' in "
                     "''.join(os.environ.values()))\n", encoding="utf-8")
        r = run_script(s, timeout=60)
        assert "SEEN False" in r.stdout, r.stdout + r.stderr
    finally:
        os.environ.pop("OPENAI_API_KEYS", None)



def test_openmp_preload_only_for_scripts_that_import_torch():
    """The childenv sitecustomize imports LightGBM in every child to win the OpenMP race.

    It costs +3.8s per subprocess (0.16s -> 3.96s measured) and a script that never touches
    torch cannot hit the clash it guards against. Unconditional, it took agent.demo from
    seconds to nine minutes.
    """
    import tempfile
    from pathlib import Path as _P
    from agent.executor import _CHILD_ENV_DIR, run_script

    d = _P(tempfile.mkdtemp())
    show = 'import os' + chr(10) + "print(os.environ.get('PYTHONPATH', ''))" + chr(10)
    plain = d / 'plain.py'
    plain.write_text(show, encoding='utf-8')
    torchy = d / 'torchy.py'
    torchy.write_text('# torch' + chr(10) + show, encoding='utf-8')

    assert str(_CHILD_ENV_DIR) not in run_script(plain, timeout=120).stdout, \
        'a script with no torch must not pay for the preload'
    assert str(_CHILD_ENV_DIR) in run_script(torchy, timeout=120).stdout, \
        'a script mentioning torch must still get the import-order fix'
    print('ok: test_openmp_preload_only_for_scripts_that_import_torch')

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok: {t.__name__}")
    print(f"{len(tests)} tests passed")
