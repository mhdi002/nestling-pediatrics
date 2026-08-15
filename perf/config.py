"""Env-driven configuration for the Nestling performance harness.

Nothing here is hardcoded: every knob reads a ``PERF_*`` environment variable
with a documented default. See ``perf/README.md`` for the full list.
"""

from __future__ import annotations

import os
from pathlib import Path

PERF_DIR = Path(__file__).resolve().parent
REPO_ROOT = PERF_DIR.parent


def env_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    return default if val is None or val == "" else val


def env_int(name: str, default: int) -> int:
    try:
        return int(env_str(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, str(default)))
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    return env_str(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


# --- target ---
HOST = env_str("PERF_HOST", "http://127.0.0.1:8080")
API_KEY = env_str("PERF_API_KEY", "")
API_PREFIX = env_str("PERF_API_PREFIX", "/api")

# --- load shape ---
USERS = env_int("PERF_USERS", 50)
SPAWN_RATE = env_float("PERF_SPAWN_RATE", 10.0)
DURATION = env_str("PERF_DURATION", "60s")
STAGES = env_str("PERF_STAGES", "50,200,500,1000")

# --- think time (seconds) between a simulated parent's requests ---
THINK_MIN = env_float("PERF_THINK_MIN", 3.0)
THINK_MAX = env_float("PERF_THINK_MAX", 12.0)

# --- per-request client timeouts (seconds) ---
TIMEOUT = env_float("PERF_TIMEOUT", 60.0)
STREAM_TIMEOUT = env_float("PERF_STREAM_TIMEOUT", 120.0)

# --- task weights: relative frequency of each parent action ---
W_CHAT = env_int("PERF_W_CHAT", 30)
W_CHAT_STREAM = env_int("PERF_W_CHAT_STREAM", 20)
W_GROWTH = env_int("PERF_W_GROWTH", 12)
W_GROWTH_CURVES = env_int("PERF_W_GROWTH_CURVES", 8)
W_DOSSIER = env_int("PERF_W_DOSSIER", 10)
W_CHILD_LIST = env_int("PERF_W_CHILD_LIST", 5)
W_SCREENING = env_int("PERF_W_SCREENING", 8)
W_SESSION_HISTORY = env_int("PERF_W_SESSION_HISTORY", 5)
W_STATIC = env_int("PERF_W_STATIC", 10)
W_HEALTH = env_int("PERF_W_HEALTH", 4)

# --- workload realism ---
SEED_POOL = PERF_DIR / env_str("PERF_SEED_FILE", "results/seed_children.json")
# 0 = every simulated user creates its own child (registration burst).
SEED_CHILDREN = env_int("PERF_SEED_CHILDREN", 0)
# Distinct chat session per simulated user so session/memory tables get real contention.
SESSION_PER_USER = env_bool("PERF_SESSION_PER_USER", True)

# --- monitoring ---
MONITOR_INTERVAL = env_float("PERF_MONITOR_INTERVAL", 1.0)
SERVER_PROC_MATCH = env_str("PERF_SERVER_PROC_MATCH", "uvicorn")
RESULTS_DIR = Path(env_str("PERF_RESULTS_DIR", str(PERF_DIR / "results")))
