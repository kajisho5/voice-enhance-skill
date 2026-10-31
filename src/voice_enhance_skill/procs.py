"""Child processes: the only place this skill starts one. Always a list argv (never a shell), stdin closed, a
minimal environment, its own process group, and a timeout that kills the whole group."""
from __future__ import annotations

import os
import signal
import subprocess
from typing import Any, Dict, Optional, Sequence, Tuple

from .errors import VoiceEnhanceError

TIMEOUT_EXIT = 124
# Environment kept for children. VOICE_ENHANCE_* carries engine locations into the worker; nothing else
# (no tokens, no proxies) is passed on.
_ENV_KEEP = ("PATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL", "TERM", "SYSTEMROOT", "SYSTEMDRIVE", "PATHEXT",
             "COMSPEC", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "PROGRAMDATA", "PYTHONPATH", "VIRTUAL_ENV")
_ENV_PREFIX = "VOICE_ENHANCE_"


def child_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k.upper() in _ENV_KEEP or k.upper().startswith(_ENV_PREFIX)}
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if extra:
        env.update(extra)
    return env


def _group_kwargs() -> Dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined]
    return {"start_new_session": True}


def kill_tree(proc: "subprocess.Popen[str]") -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.kill()
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
    try:
        proc.communicate(timeout=5)
    except Exception:
        pass


def run_process_group(argv: Sequence[str], timeout: float, cwd: Optional[str] = None,
                      env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    """Run argv to completion. Returns (exit code, stdout, stderr); exit code TIMEOUT_EXIT when the timeout killed
    the process group. Raises VoiceEnhanceError CANCELLED on Ctrl-C (after killing the group)."""
    argv = list(argv)
    if not argv or any(not isinstance(a, str) or "\x00" in a for a in argv):
        raise VoiceEnhanceError("INTERNAL_ERROR", "argv must be a non-empty list of clean strings")
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                errors="replace", cwd=cwd, env=child_env() if env is None else env, **_group_kwargs())
    except OSError as e:
        return 127, "", f"cannot start {os.path.basename(argv[0])}: {e}"
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        return TIMEOUT_EXIT, "", f"timed out after {timeout}s"
    except KeyboardInterrupt:
        kill_tree(proc)
        raise VoiceEnhanceError("CANCELLED", "interrupted while a child process was running", {"reason": "signal"})
    return proc.returncode, out or "", err or ""
