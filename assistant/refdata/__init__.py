"""Load clinical reference JSON from config/."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def _load(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing clinical config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache
def who_lms() -> dict[str, Any]:
    return _load("who_lms.json")


@lru_cache
def clinical_bounds() -> dict[str, Any]:
    return _load("clinical_bounds.json")


@lru_cache
def asq_scoring() -> dict[str, Any]:
    return _load("asq_scoring.json")


@lru_cache
def mchat_config() -> dict[str, Any]:
    return _load("mchat.json")


def weeks_per_month() -> float:
    b = clinical_bounds()
    if b.get("use_legacy_weeks_per_month", True):
        return float(b["weeks_per_month_legacy"])
    return float(b["weeks_per_month"])


def clear_refdata_cache() -> None:
    who_lms.cache_clear()
    clinical_bounds.cache_clear()
    asq_scoring.cache_clear()
    mchat_config.cache_clear()
