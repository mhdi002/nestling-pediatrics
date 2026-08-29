"""
Regression tests for multi-turn slot filling.

The bug these cover: when the assistant asked for a missing slot, a bare
answer ("boy", "term", "37") carries no intent of its own, so the router
classified it as small talk and the original growth request was silently
dropped -- the user had to restate everything in one message.
"""

from __future__ import annotations

import pytest

from assistant.agent.orchestrator import ParentAssistant


@pytest.fixture()
def assistant(tmp_path, monkeypatch):
    monkeypatch.setenv("NESTLING_USE_LLM", "0")
    monkeypatch.setenv("NESTLING_LOAD_MODELS", "0")
    monkeypatch.setenv("NESTLING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NESTLING_CHILD_DB", str(tmp_path / "children.db"))
    monkeypatch.setenv("NESTLING_CHAT_DB", str(tmp_path / "chat.db"))
    from assistant.settings import get_settings

    get_settings.cache_clear()
    return ParentAssistant()


def _reply(assistant, session_id, text):
    return assistant.chat(session_id, text, ui_lang="en")


def test_bare_sex_answer_resumes_growth_request(assistant):
    """'boy' after being asked for sex should complete the plot, not reset."""
    session_id = assistant.chat_memory.create_session()

    first = _reply(assistant, session_id, "show my kid chart age 12 weight 5")
    # The assistant should be waiting on something at this point.
    assert first.get("missing_slots"), "expected the first turn to ask for a slot"

    second = _reply(assistant, session_id, "boy")
    slots = assistant.chat_memory.get_slots(session_id)
    # The answer must be retained *and* the original request resumed.
    assert slots.get("sex") in {"male", "boy"}, slots
    assert "growth" in set(second.get("intents") or []), second.get("intents")


def test_slots_accumulate_across_turns(assistant):
    """Each turn contributes a slot; none of the earlier ones are lost."""
    session_id = assistant.chat_memory.create_session()

    _reply(assistant, session_id, "plot weight for my baby")
    _reply(assistant, session_id, "5 kg")
    _reply(assistant, session_id, "12 weeks")
    _reply(assistant, session_id, "boy")

    slots = assistant.chat_memory.get_slots(session_id)
    assert slots.get("sex") in {"male", "boy"}, slots
    assert slots.get("measure") == "weight", slots


def test_pending_intent_cleared_once_satisfied(assistant):
    """A completed request must not hijack a later unrelated turn."""
    session_id = assistant.chat_memory.create_session()

    # Gestational age is also required before a chart can be chosen, so the
    # request is only complete once term/preterm is known.
    _reply(assistant, session_id, "plot weight 5 kg at 12 weeks for my boy")
    done = _reply(assistant, session_id, "term")
    assert not done.get("missing_slots"), done.get("missing_slots")

    slots = assistant.chat_memory.get_slots(session_id)
    assert not slots.get("pending_intent"), "pending_intent should be cleared"

    later = _reply(assistant, session_id, "he has a scar on his knee")
    intents = set(later.get("intents") or [])
    assert "growth" not in intents, f"unrelated turn hijacked by growth: {intents}"


def test_clear_slots_removes_key(assistant):
    """merge_slots cannot unset; clear_slots must."""
    session_id = assistant.chat_memory.create_session()
    assistant.chat_memory.merge_slots(session_id, {"pending_intent": "growth"})
    assert assistant.chat_memory.get_slots(session_id).get("pending_intent") == "growth"

    assistant.chat_memory.clear_slots(session_id, ["pending_intent"])
    assert "pending_intent" not in assistant.chat_memory.get_slots(session_id)


def test_age_in_years_is_parsed(assistant):
    """
    "2 years old" is how parents state a toddler's age. Before this was
    handled the extractor returned no age at all, so the assistant re-asked
    for an age the parent had just given, in English and Persian alike.
    """
    from assistant.agent.slots import extract_growth_slots

    for text in ("girl 2 years old 3 kg", "2 years", "2yo", "دختر ۲ ساله ۳ کیلو", "۲ سال"):
        slots = extract_growth_slots(text)
        assert slots.get("age_months") == 24.0, f"{text!r} -> {slots}"

    # Weeks and months must keep working unchanged.
    assert extract_growth_slots("18 months").get("age_months") == 18.0
    assert extract_growth_slots("40 weeks").get("weeks") == 40.0
    assert "age_months" not in extract_growth_slots("32 weeks at birth")


def test_year_age_answer_resumes_growth_request(assistant):
    """A follow-up given in years must complete the plot, not restart it."""
    session_id = assistant.chat_memory.create_session()
    _reply(assistant, session_id, "plot weight for my girl, 3 kg")
    out = _reply(assistant, session_id, "2 years")
    slots = assistant.chat_memory.get_slots(session_id)
    assert slots.get("age_months") == 24.0, slots
    assert "growth" in set(out.get("intents") or []), out.get("intents")
