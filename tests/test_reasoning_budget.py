"""A reasoning model that never answers must be diagnosable, not silent.

Found by running the app against a real local qwen3-vl:4b: it reasoned for
3500-4000 characters and returned EMPTY content with finish_reason "length",
so every grounded reply silently became the extractive fallback.
"""

from __future__ import annotations

import assistant.llm.qwen_client as qc
from assistant.settings import reset_settings


def _choice(content="", reasoning="", finish="stop"):
    return {"message": {"content": content, "reasoning": reasoning}, "finish_reason": finish}


def test_an_answer_is_returned_unchanged():
    assert qc._content_or_explain(_choice(content="Paris."), 600) == "Paris."


def test_an_empty_answer_mid_reasoning_is_recognised():
    assert qc._ran_out_thinking(_choice(reasoning="thinking…", finish="length"))


def test_an_empty_answer_the_model_simply_chose_is_not_blamed_on_budget():
    """finish_reason "stop" means it finished; the budget was not the problem."""
    assert not qc._ran_out_thinking(_choice(reasoning="thinking…", finish="stop"))
    assert not qc._ran_out_thinking(_choice(content="Paris.", finish="length"))


def test_the_failure_is_logged_loudly(caplog):
    """Silent permanent degradation is the thing being prevented."""
    import logging

    with caplog.at_level(logging.WARNING):
        qc._content_or_explain(_choice(reasoning="x" * 3500, finish="length"), 900)
    assert any("budget reasoning" in r.message for r in caplog.records)


def test_the_retry_is_off_by_default():
    """Measured: escalating 900 -> 2700 -> 4096 still produced no answer.

    All it bought was three times the latency on a turn a parent waits for.
    """
    reset_settings()
    assert qc._retry_budget(900) is None


def test_the_retry_can_be_enabled_and_is_bounded(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_RETRY_MULTIPLIER", "3")
    monkeypatch.setenv("LLM_REASONING_RETRY_MAX_TOKENS", "4096")
    reset_settings()
    try:
        assert qc._retry_budget(900) == 2700
        # Never past the ceiling, and never a second time from it.
        assert qc._retry_budget(2700) == 4096
        assert qc._retry_budget(4096) is None
    finally:
        reset_settings()


def test_thinking_is_disabled_by_default():
    """Why the deployed vLLM answers at all: it honours this field."""
    reset_settings()
    assert qc._chat_template_kwargs() == {"chat_template_kwargs": {"enable_thinking": False}}


def test_the_field_can_be_cleared_for_a_server_that_rejects_it(monkeypatch):
    """Hardcoded, it would fail every request with no way to turn it off."""
    for value in ("", "{}"):
        monkeypatch.setenv("NESTLING_LLM_CHAT_TEMPLATE_KWARGS", value)
        reset_settings()
        assert qc._chat_template_kwargs() == {}
    reset_settings()


def test_malformed_configuration_is_ignored_not_fatal(monkeypatch, caplog):
    import logging

    monkeypatch.setenv("NESTLING_LLM_CHAT_TEMPLATE_KWARGS", "{not json")
    reset_settings()
    with caplog.at_level(logging.WARNING):
        assert qc._chat_template_kwargs() == {}
    assert any("malformed" in r.message for r in caplog.records)
    reset_settings()
