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
    assert "history" in set(out.get("intents") or []), out.get("intents")
    names = [tc.get("name") for tc in out["tools"]["tool_calls"]]
    assert "get_child_summary" in names, names
    summary = next(
        tc["result"]["summary"]
        for tc in out["tools"]["tool_calls"]
        if tc["name"] == "get_child_summary"
    )
    assert "9.4" in summary, summary


def test_persian_name_question_resolves_child(assistant):
    child_id = _add_child(assistant, "مونیکا")
    assistant.db.add_growth(
        child_id, weeks=52.0, measure="weight", value=9.4, centile=55.0, age_months=12.0
    )

    out = assistant.chat(
        _session(assistant), "وضعیت مونیکا چجوریه ؟", ui_lang="fa", owner_user_id=OWNER
    )

    assert out["child_id"] == child_id
    names = [tc.get("name") for tc in out["tools"]["tool_calls"]]
    assert "get_child_summary" in names, names


def test_two_children_with_the_same_name_are_not_guessed(assistant):
    _add_child(assistant, "Sara")
    _add_child(assistant, "Sara")

    out = assistant.chat(
        _session(assistant), "How is Sara doing?", ui_lang="en", owner_user_id=OWNER
    )

    assert out["child_id"] is None, out["child_id"]
    assert len(out["ambiguous_children"]) == 2, out["ambiguous_children"]
    assert "child (which one?)" in out["missing_slots"], out["missing_slots"]
    assert "which one" in out["reply"].lower(), out["reply"]


def test_another_users_child_is_never_resolved(assistant):
    _add_child(assistant, "Monica", owner=OTHER_OWNER)

    out = assistant.chat(
        _session(assistant), "How is Monica doing?", ui_lang="en", owner_user_id=OWNER
    )

    assert out["child_id"] is None, out["child_id"]
    assert not out["ambiguous_children"]
    names = [tc.get("name") for tc in out["tools"]["tool_calls"]]
    assert "get_child_summary" not in names, names
