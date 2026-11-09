"""HTTP client for the TypeScript RLM orchestrator."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from assistant.settings import get_settings

log = logging.getLogger(__name__)

_proc: subprocess.Popen | None = None


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
    env.setdefault("NESTLING_LLM_MODEL", s.nestling_llm_model or "openbmb/MiniCPM5-1B")
    env.setdefault("NESTLING_KNOWLEDGE_DIR", str(root / "knowledge"))
    log.info("starting TypeScript RLM orchestrator from %s", pkg)
    _proc = subprocess.Popen(
        ["npm", "run", "nestling"],
        cwd=str(pkg),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    for _ in range(40):
        if health():
            return
        time.sleep(0.25)
    err = ""
    if _proc.stderr:
        err = _proc.stderr.read().decode("utf-8", errors="replace")[:800]
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
