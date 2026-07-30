"""Declarative intent rules loaded from config/intent_rules.yaml."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "intent_rules.yaml"


@lru_cache
def load_intent_rules() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return {"intents": {}, "slots": {}, "thresholds": {}}
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


def clear_rules_cache() -> None:
    load_intent_rules.cache_clear()


def match_intents_from_rules(message: str) -> set[str]:
    """Lightweight YAML-driven matcher (fallback / future primary path)."""
    rules = load_intent_rules()
    msg = (message or "").strip()
    if not msg:
        return {"help"}
    hits: list[tuple[int, str]] = []
    for name, spec in (rules.get("intents") or {}).items():
        patterns = list(spec.get("patterns_en") or []) + list(spec.get("patterns_fa") or [])
        for pat in patterns:
            try:
                if re.search(pat, msg, re.I):
                    hits.append((int(spec.get("priority", 0)), name))
                    break
            except re.error:
                continue
    if not hits:
        return set()
    hits.sort(reverse=True)
    # Return top priority intent plus any within 20 points
    top = hits[0][0]
    return {name for pri, name in hits if pri >= top - 20}


def extract_slots_from_rules(message: str) -> dict[str, Any]:
    rules = load_intent_rules()
    msg = (message or "").lower()
    out: dict[str, Any] = {}
    for slot, mapping in (rules.get("slots") or {}).items():
        if not isinstance(mapping, dict):
            continue
        for value, synonyms in mapping.items():
            for syn in synonyms or []:
                if syn.lower() in msg:
                    out[slot] = value
                    break
            if slot in out:
                break
    return out
