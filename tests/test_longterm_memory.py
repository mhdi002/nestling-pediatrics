"""
Long-term memory: a child's story must survive the session it was told in.

Chat sessions are disposable; a child's developmental thread is not. These
cover the three ways memory used to be lost: a new session started blank, a
long session silently dropped its oldest turns once the window slid past them,
and nothing tied remembered facts to the CHILD rather than the session.
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


def _reply(assistant, session_id, text, child_id=None):
    return assistant.chat(session_id, text, child_id=child_id, ui_lang="en")


def _child(assistant, name, owner=None):
    return assistant.db.create_child(
        name=name,
        sex="male",
        date_of_birth="2024-01-01",
        gestational_age_weeks=39,
        owner_user_id=owner,
    )


def test_child_facts_carry_into_a_new_session(assistant):
    """Session B about child 1 must start knowing what session A established."""
    child = _child(assistant, "Ali")

    s_a = assistant.chat_memory.create_session(child_id=child)
    _reply(assistant, s_a, "he has had a persistent cough for three days", child_id=child)

    # A brand-new session, no shared session_id, no shared slots.
    s_b = assistant.chat_memory.create_session(child_id=child)
    out = _reply(assistant, s_b, "any update on that?", child_id=child)

    ctx = out["memory"]["child_context"]
    assert "cough" in ctx.lower(), ctx


def test_memory_does_not_bleed_between_children(assistant):
    """Two kids, two threads — child 2's session must not see child 1's history."""
    one = _child(assistant, "Ali")
    two = _child(assistant, "Sara")

    s_one = assistant.chat_memory.create_session(child_id=one)
    _reply(assistant, s_one, "he has had a persistent cough for three days", child_id=one)

    s_two = assistant.chat_memory.create_session(child_id=two)
    _reply(assistant, s_two, "she has a rash on her arm", child_id=two)

    ctx_two = assistant.db.child_context_text(two)
    assert "rash" in ctx_two.lower(), ctx_two
    assert "cough" not in ctx_two.lower(), ctx_two

    ctx_one = assistant.db.child_context_text(one)
    assert "rash" not in ctx_one.lower(), ctx_one


def test_child_memory_is_scoped_per_account(assistant):
    """A session owned by one account must never load another account's child."""
    owned = _child(assistant, "Ali", owner="user-a")

    # Same child id, read as the wrong account: nothing at all.
    assistant.db.remember_note(owned, "cough for three days", owner_user_id="user-a")
    assert "cough" in assistant.db.child_context_text(owned, owner_user_id="user-a").lower()
    assert assistant.db.child_context_text(owned, owner_user_id="user-b") == ""

    # And writes from the wrong account are refused outright.
    assert assistant.db.remember_note(owned, "rash on arm", owner_user_id="user-b") is None
    assert "rash" not in assistant.db.child_context_text(owned, owner_user_id="user-a").lower()


def test_long_session_keeps_early_turns_in_the_summary(assistant):
    """
    Turns that slide out of the window must land in the rolling summary.
    This runs with NESTLING_USE_LLM=0 — the fold is deterministic, so a
    CPU-only host with the sidecar down still keeps its memory.
    """
    from assistant.settings import get_settings

    settings = get_settings()
    session_id = assistant.chat_memory.create_session()

    marker = "zulubadger"
    assistant.chat_memory.add_message(session_id, "user", f"remember the word {marker}")
    assistant.chat_memory.add_message(session_id, "assistant", "noted")
    turns = settings.nestling_summary_trigger_turns + settings.nestling_history_window + 4
    for i in range(turns):
        assistant.chat_memory.add_message(session_id, "user", f"filler question {i}")
        assistant.chat_memory.add_message(session_id, "assistant", f"filler answer {i}")
        # Rebuild each turn, as chat() does, so folding happens incrementally.
        assistant.chat_memory.build_context(session_id)

    ctx = assistant.chat_memory.build_context(session_id)
    assert marker not in ctx["recent_text"], "marker should have fallen out of the window"
    assert marker in ctx["summary"], ctx["summary"][:400]


def test_summary_does_not_duplicate_folded_turns(assistant):
    """The fold watermark must stop the same turns being folded repeatedly."""
    session_id = assistant.chat_memory.create_session()
    for i in range(60):
        assistant.chat_memory.add_message(session_id, "user", f"unique-turn-{i}")
        assistant.chat_memory.build_context(session_id)

    summary = assistant.chat_memory.build_context(session_id)["summary"]
    assert summary.count("unique-turn-3 ") <= 1, summary[:400]


def test_growth_and_screenings_appear_in_child_context(assistant):
    """Durable clinical data is part of the cross-session digest, not just notes."""
    child = _child(assistant, "Ali")
    assistant.db.add_growth(child, weeks=52.0, measure="weight", value=8.1, centile=45.0)
    ctx = assistant.db.child_context_text(child)
    assert "weight" in ctx.lower() and "8.1" in ctx, ctx


def test_child_memory_can_be_disabled_by_settings(assistant, monkeypatch):
    """Every tunable is env-driven; the whole feature is switchable."""
    from assistant.settings import get_settings

    child = _child(assistant, "Ali")
    monkeypatch.setenv("NESTLING_CHILD_MEMORY_ENABLED", "0")
    get_settings.cache_clear()
    try:
        assert assistant.db.remember_note(child, "cough for three days") is None
    finally:
        monkeypatch.delenv("NESTLING_CHILD_MEMORY_ENABLED", raising=False)
        get_settings.cache_clear()
