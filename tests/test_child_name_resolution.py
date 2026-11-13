"""
Resolving a child by name mid-conversation.

The bug these cover: "How is Monica doing?" fell through to generic small talk
because nothing mapped a spoken name to a child record, so the child summary
tool (which needs a UUID) never ran.
"""

from __future__ import annotations

import pytest

from assistant.agent.orchestrator import ParentAssistant
from assistant.memory.chat_memory import ChatMemory
from assistant.memory.child_db import ChildMemoryDB


@pytest.fixture()
def assistant(tmp_path, monkeypatch):
    monkeypatch.setenv("NESTLING_USE_LLM", "0")
    monkeypatch.setenv("NESTLING_LOAD_MODELS", "0")
    monkeypatch.setenv("NESTLING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NESTLING_CHILD_DB", str(tmp_path / "children.db"))
    monkeypatch.setenv("NESTLING_CHAT_DB", str(tmp_path / "chat.db"))
    from assistant.settings import get_settings

    get_settings.cache_clear()
    # Explicit paths: the DB modules resolve their default path at import time,
    # so env vars alone would leave every test sharing one database.
    svc = ParentAssistant(
        db=ChildMemoryDB(tmp_path / "children.db"),
        chat_memory=ChatMemory(tmp_path / "chat.db"),
    )
    yield svc
    svc.close()


OWNER = "owner-1"
OTHER_OWNER = "owner-2"


def _add_child(assistant, name, owner=OWNER, **kw):
    return assistant.db.create_child(
        name=name,
        sex=kw.get("sex", "female"),
        gestational_age_weeks=kw.get("gestational_age_weeks", 39.0),
        owner_user_id=owner,
    )


def _session(assistant, owner=OWNER):
    return assistant.chat_memory.create_session(owner_user_id=owner)


def test_name_resolves_child_and_pulls_history(assistant):
    child_id = _add_child(assistant, "Monica")
    assistant.db.add_growth(
        child_id, weeks=52.0, measure="weight", value=9.4, centile=55.0, age_months=12.0
    )

    out = assistant.chat(
        _session(assistant), "How is Monica doing?", ui_lang="en", owner_user_id=OWNER
    )

    assert out["child_id"] == child_id, out["child_id"]
    # The turn was given this child's record, growth history included.
    assert "9.4" in (out["memory"].get("child_context") or ""), out["memory"]


def test_persian_name_question_resolves_child(assistant):
    child_id = _add_child(assistant, "مونیکا")
    assistant.db.add_growth(
        child_id, weeks=52.0, measure="weight", value=9.4, centile=55.0, age_months=12.0
    )

    out = assistant.chat(
        _session(assistant), "وضعیت مونیکا چجوریه ؟", ui_lang="fa", owner_user_id=OWNER
    )

    assert out["child_id"] == child_id
    assert "9.4" in (out["memory"].get("child_context") or ""), out["memory"]


def test_two_children_with_the_same_name_are_not_guessed(assistant):
    _add_child(assistant, "Sara")
    _add_child(assistant, "Sara")

    out = assistant.chat(
        _session(assistant), "How is Sara doing?", ui_lang="en", owner_user_id=OWNER
    )

    assert out["child_id"] is None, out["child_id"]
    assert len(out["ambiguous_children"]) == 2, out["ambiguous_children"]
    assert "which child" in out["reply"].lower(), out["reply"]


def test_another_users_child_is_never_resolved(assistant):
    _add_child(assistant, "Monica", owner=OTHER_OWNER)

    out = assistant.chat(
        _session(assistant), "How is Monica doing?", ui_lang="en", owner_user_id=OWNER
    )

    assert out["child_id"] is None, out["child_id"]
    assert not out["ambiguous_children"]
    # Nothing about that child reached this turn, through the record or a tool.
    assert not (out["memory"].get("child_context") or "")
    summaries = [
        tc for tc in out["tools"]["tool_calls"]
        if tc.get("name") == "get_child_summary" and tc.get("ok") and (tc.get("result") or {}).get("ok")
    ]
    assert not summaries, summaries


def test_an_only_child_is_meant_when_none_is_named(assistant):
    """No child picked in the UI and no name in the message: with one child on
    the account, what the parent says is about that child."""
    child_id = _add_child(assistant, "Monica")
    out = assistant.chat(
        _session(assistant), "she has had a runny nose since yesterday", ui_lang="en", owner_user_id=OWNER
    )
    assert out["child_id"] == child_id, out["child_id"]


def test_no_child_is_assumed_when_there_are_several(assistant):
    _add_child(assistant, "Monica")
    _add_child(assistant, "Sara")
    out = assistant.chat(
        _session(assistant), "she has had a runny nose since yesterday", ui_lang="en", owner_user_id=OWNER
    )
    assert out["child_id"] is None, out["child_id"]

