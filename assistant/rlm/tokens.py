"""Token / character estimates — ratios come from settings, not literals in callers."""

from __future__ import annotations

import math

from assistant.settings import get_settings


def chars_per_token() -> float:
    return float(get_settings().rlm_chars_per_token)


def message_overhead_tokens() -> int:
    return int(get_settings().rlm_message_overhead_tokens)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return int(math.ceil(len(text) / chars_per_token()))


def chars_for_tokens(tokens: float) -> int:
    return max(0, int(tokens * chars_per_token()))


def estimate_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        total += estimate_tokens(str(m.get("content") or "")) + message_overhead_tokens()
    return total


def clip_chars(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n[truncated {omitted} chars]"
