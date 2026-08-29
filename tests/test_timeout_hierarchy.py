"""
The app must give up on generation before the proxy gives up on the app.

With NESTLING_LLM_TIMEOUT at 180s and the load balancer's chat timeout at
120s, nginx always cut the connection first: the request appeared to hang and
then returned 504, and the extractive answer the app had already retrieved was
never sent. The app can only degrade gracefully if its own deadline fires
first.

Both numbers are read from their real sources so this fails if either drifts.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from assistant.settings import get_settings

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "docker-compose.yml"


def _lb_chat_timeout_seconds() -> float:
    """Default value of NESTLING_LB_CHAT_TIMEOUT from docker-compose.yml."""
    text = COMPOSE.read_text(encoding="utf-8")
    m = re.search(r"NESTLING_LB_CHAT_TIMEOUT=\$\{NESTLING_LB_CHAT_TIMEOUT:-(\d+)s\}", text)
    assert m, "could not read NESTLING_LB_CHAT_TIMEOUT default from docker-compose.yml"
    return float(m.group(1))


def test_generation_deadline_is_below_the_proxy_deadline():
    llm = get_settings().nestling_llm_timeout
    proxy = _lb_chat_timeout_seconds()
    assert llm < proxy, (
        f"LLM timeout {llm}s must be under the proxy's {proxy}s or a slow "
        "generation returns 504 instead of the extractive fallback"
    )


def test_there_is_real_headroom_for_the_rest_of_the_turn():
    """
    Generation is not the only work in a turn: retrieval, tools, translation
    and an optional web search all run inside the same request. Leave room.
    """
    llm = get_settings().nestling_llm_timeout
    proxy = _lb_chat_timeout_seconds()
    assert proxy - llm >= 30, (
        f"only {proxy - llm}s left for retrieval, tools and translation"
    )


def test_probe_is_far_shorter_than_generation():
    """The readiness probe must fail fast; it runs on every turn."""
    s = get_settings()
    assert s.nestling_llm_probe_timeout < s.nestling_llm_timeout / 10


def test_web_search_timeout_fits_inside_the_turn():
    s = get_settings()
    proxy = _lb_chat_timeout_seconds()
    assert s.nestling_websearch_timeout < proxy - s.nestling_llm_timeout, (
        "a web search must not consume the headroom generation needs"
    )
