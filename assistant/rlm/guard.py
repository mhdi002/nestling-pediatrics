"""Cut off looping / over-thinking model replies."""

from __future__ import annotations

import time
from typing import Literal

from assistant.settings import get_settings

CutReason = Literal["thinking_budget", "repeating"]


def _window() -> int:
    return int(get_settings().rlm_guard_window_chars)


def _check_every() -> int:
    return int(get_settings().rlm_guard_check_every_chars)


def _needle() -> int:
    return int(get_settings().rlm_guard_needle_chars)


class ReplyGuard:
    def __init__(
        self,
        *,
        thinking_max_chars: int,
        thinking_max_ms: int | None,
        repeat_limit: int,
    ) -> None:
        self.thinking_max_chars = thinking_max_chars
        self.thinking_max_ms = thinking_max_ms
        self.repeat_limit = repeat_limit
        self.thinking_chars = 0
        self.thinking_since = 0.0
        self.buf = {"thinking": "", "text": ""}
        self.unchecked = {"thinking": 0, "text": 0}

    def push(self, kind: Literal["thinking", "text"], text: str) -> CutReason | None:
        if kind == "thinking":
            now = time.time() * 1000
            if not self.thinking_chars:
                self.thinking_since = now
            self.thinking_chars += len(text)
            if self.thinking_chars > self.thinking_max_chars:
                return "thinking_budget"
            if self.thinking_max_ms is not None and now - self.thinking_since > self.thinking_max_ms:
                return "thinking_budget"
        window = _window()
        b = (self.buf[kind] + text)[-window:]
        self.buf[kind] = b
        self.unchecked[kind] += len(text)
        if self.unchecked[kind] < _check_every():
            return None
        self.unchecked[kind] = 0
        if repeating_tail(b, self.repeat_limit):
            return "repeating"
        if kind == "thinking" and repeated_sentence(b, self.repeat_limit):
            return "repeating"
        return None


def repeating_tail(s: str, times: int) -> bool:
    needle = _needle()
    if len(s) < needle * times:
        return False
    frag = s[-needle:]
    from_idx = len(s) - needle - 1
    for _ in range(64):
        if from_idx < 0:
            return False
        at = s.rfind(frag, 0, from_idx + 1)
        if at < 0:
            return False
        period = len(s) - needle - at
        from_idx = at - 1
        if period < 16 or period * times > len(s):
            continue
        block = s[-period:]
        if len(set(block)) < 8 or period * times < 400:
            continue
        ok = True
        for k in range(2, times + 1):
            if s[len(s) - period * k : len(s) - period * (k - 1)] != block:
                ok = False
                break
        if ok:
            return True
    return False


def repeated_sentence(s: str, times: int) -> bool:
    counts: dict[str, int] = {}
    import re

    for raw in re.split(r"(?<=[.!?])\s+|\n+", s):
        k = re.sub(r"\d+", "#", raw.lower())
        k = re.sub(r"\s+", " ", k).strip()
        if len(k) < 30:
            continue
        n = counts.get(k, 0) + 1
        if n > times:
            return True
        counts[k] = n
    return False
