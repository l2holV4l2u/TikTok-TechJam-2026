import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunResult:
    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool


def _text(v) -> str:
    if v is None:
        return ""
    return v.decode(errors="replace") if isinstance(v, bytes) else v


# Substrings that mark an environment variable as a credential. Generated code is written by a
# model and run unsandboxed, so anything the controller can read, the script can exfiltrate. A
# measured run of the harness exposed OPENAI_API_KEYS, OPENAI_API_KEY, GITHUB_PACKAGE_TOKEN and
# CLAUDE_CODE_MESSAGING_TOKEN to the script. None of them are needed to train a model.
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH",
                   "SESSION", "COOKIE", "PRIVATE")
# ...except these, which are ordinary configuration that happens to match a marker.
_SECRET_ALLOW = {"KEYBOARD", "SSH_AUTH_SOCK_DISABLED"}


def _strip_credentials(env: dict) -> dict:
    """Remove credentials from the environment a generated script inherits.

    Deny by pattern rather than allow by name: an allowlist of what a training script needs is
    impossible to keep correct across platforms, and getting it wrong breaks every run. Getting
    the denylist slightly wide only costs a variable nothing needed.
    """
    out = {}
    for k, v in env.items():
        up = k.upper()
        if up in _SECRET_ALLOW:
            out[k] = v
            continue
        if any(m in up for m in _SECRET_MARKERS):
            continue
        out[k] = v
    return out


def run_script(path: Path, timeout: float = 1800, cwd=None, pythonpath=None, extra_env=None) -> RunResult:
    """Never raises on script failure; the loop decides what a failure means."""
    # always build the child environment explicitly, even with no extras: inheriting the parent's
    # wholesale hands the script every credential the controller holds
    env = _strip_credentials(dict(os.environ))
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join([str(pythonpath), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.update(extra_env or {})
    t0 = time.perf_counter()
    process_group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                     if sys.platform == "win32" else {"start_new_session": True})
    proc = subprocess.Popen(
        [sys.executable, str(path)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=cwd, env=env, **process_group,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        # normalise here too, not only on the timeout path: communicate can hand back None
        # for a stream, and a None stdout killed a whole run inside parse_findings.
        out, err = _text(out), _text(err)
        code, timed_out = proc.returncode, False
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        out, err = proc.communicate()
        out, err = _text(out), _text(err) + "\nTIMEOUT"
        code, timed_out = -1, True
    return RunResult(code == 0, code, out, err, time.perf_counter() - t0, timed_out)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the whole tree. A bare kill() leaves the script's own children running, and a stray
    trainer both burns CPU and corrupts the GPU-seconds we report."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, check=False)
    else:
        import signal
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
