"""HTTP client for the TypeScript RLM orchestrator."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from assistant.settings import get_settings

log = logging.getLogger(__name__)

_proc: subprocess.Popen | None = None


def _state_dir(s) -> Path:
    base = Path(getattr(s, "nestling_data_dir", "") or (Path(s.nestling_root) / "data"))
    return base / "rlm_state"


def ts_base_url() -> str:
    s = get_settings()
    return (getattr(s, "rlm_ts_url", None) or os.environ.get("RLM_TS_URL") or "http://127.0.0.1:8120").rstrip("/")


def health() -> bool:
    try:
        with urllib.request.urlopen(f"{ts_base_url()}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_ts_orchestrator() -> None:
    global _proc
    if health():
        return
    s = get_settings()
    if getattr(s, "rlm_ts_autostart", True) is False:
        raise RuntimeError(f"TypeScript RLM orchestrator is not running at {ts_base_url()}")
    root = Path(s.nestling_root).resolve()
    pkg = root / "vendor" / "rlm-ide"
    env = os.environ.copy()
    env.setdefault("NESTLING_ROOT", str(root))
    env.setdefault("RLM_TS_PORT", ts_base_url().rsplit(":", 1)[-1])
    env.setdefault("NESTLING_LLM_URL", s.nestling_llm_url or "http://127.0.0.1:8100")
    env.setdefault("NESTLING_LLM_MODEL", s.nestling_llm_model)
    env.setdefault("NESTLING_KNOWLEDGE_DIR", str(root / "knowledge"))
    # Conversations live in the orchestrator. Kept on disk beside the app's
    # other data, a restart of that process does not forget every chat.
    env.setdefault("RLM_STATE_DIR", str(_state_dir(s)))
    # The interpreter running this app is known to exist and to have the
    # worker's dependencies. Leaving the choice to a PATH search for "python3"
    # finds the Microsoft Store stub on Windows, which exists and cannot run.
    env.setdefault("RLM_PYTHON", s.rlm_python or sys.executable)
    # Resolve the launcher the way the OS does. A bare "npm" only works where
    # the executable is literally named npm; on Windows it is npm.cmd, which
    # CreateProcess does not find by its stem, so every chat turn died with
    # "[WinError 2] The system cannot find the file specified".
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("TypeScript RLM orchestrator cannot start: npm is not on PATH")
    # Orchestrator output goes to a file, never to an undrained pipe: a
    # long-lived server that logs more than the OS pipe buffer (~64 KiB) would
    # block on its next write and stall every turn. The file is also what a
    # failed start is diagnosed from.
    log_dir = Path(s.nestling_data_dir) if s.nestling_data_dir else Path(tempfile.gettempdir())
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "rlm_ts_orchestrator.log"
    log.info("starting TypeScript RLM orchestrator from %s (log: %s)", pkg, log_path)
    with open(log_path, "ab") as log_file:
        _proc = subprocess.Popen(
            [npm, "run", "nestling"],
            cwd=str(pkg),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    for _ in range(40):
        if health():
            return
        if _proc.poll() is not None:
            break  # it died; waiting longer cannot help
        time.sleep(0.25)
    # Read the log rather than the process's stream: reading a pipe to EOF
    # blocks for as long as a live-but-unhealthy server keeps it open.
    try:
        err = log_path.read_bytes()[-800:].decode("utf-8", errors="replace")
    except OSError:
        err = ""
    raise RuntimeError(f"TypeScript RLM orchestrator failed to start: {err}")


def run_turn(payload: dict) -> dict:
    ensure_ts_orchestrator()
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{ts_base_url()}/turn",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    stall = float(get_settings().rlm_model_stall_sec) + 60
    with urllib.request.urlopen(req, timeout=stall) as r:
        return json.loads(r.read().decode("utf-8"))
