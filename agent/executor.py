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


def run_script(path: Path, timeout: float = 1800, cwd=None, pythonpath=None, extra_env=None) -> RunResult:
    """Never raises on script failure; the loop decides what a failure means."""
    env = None
    if pythonpath or extra_env:
        # a generated script lives outside the project, so the root must be injected explicitly
        env = dict(os.environ)
        if pythonpath:
            env["PYTHONPATH"] = os.pathsep.join([str(pythonpath), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        env.update(extra_env or {})
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd, env=env,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
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
