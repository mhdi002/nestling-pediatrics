"""
Clearing chat history must remove only the caller's conversations.

A "clear my history" button that deletes unscoped would take every family's
conversations with it, so the ownership check lives in the store as well as
the route — no future caller can bypass it by using the store directly.
Medical records (children, growth, screenings) are deliberately untouched:
this clears conversation history, not the child's chart.
"""

from __future__ import annotations

import pytest

from assistant.memory.chat_memory import ChatMemory


@pytest.fixture()
def mem(tmp_path):
    m = ChatMemory(tmp_path / "chat.db")
    try:
        yield m
    finally:
        m.close()


def _session_with_messages(mem, owner, text="hello"):
    sid = mem.create_session(owner_user_id=owner)
    mem.add_message(sid, "user", text)
    mem.add_message(sid, "assistant", "hi")
    return sid


def test_clears_only_the_callers_sessions(mem):
    mine = _session_with_messages(mem, "user-a")
    theirs = _session_with_messages(mem, "user-b")

    removed = mem.delete_all_sessions("user-a")
    assert removed == 1
    assert mem.get_session(mine) is None
    assert mem.get_session(theirs) is not None, "another account's history was deleted"


def test_messages_and_facts_go_with_the_session(mem):
    sid = _session_with_messages(mem, "user-a")
    mem.delete_all_sessions("user-a")
    # No orphaned rows left behind pointing at a session that no longer exists.
    rows = mem.conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=?", (sid,)
    ).fetchone()[0]
    assert rows == 0


def test_refuses_to_run_without_an_owner(mem):
    """Without a scope this would wipe the whole table."""
    _session_with_messages(mem, "user-a")
    with pytest.raises(ValueError):
        mem.delete_all_sessions("")
    assert mem.list_sessions(owner_user_id="user-a"), "sessions were destroyed anyway"


def test_single_delete_is_ownership_checked(mem):
    theirs = _session_with_messages(mem, "user-b")
    assert mem.delete_session(theirs, owner_user_id="user-a") is False
    assert mem.get_session(theirs) is not None
    assert mem.delete_session(theirs, owner_user_id="user-b") is True
    assert mem.get_session(theirs) is None


def test_deleting_a_missing_session_reports_false(mem):
    assert mem.delete_session("no-such-session", owner_user_id="user-a") is False


def test_children_survive_clearing_chat_history(tmp_path):
    """Clearing conversations must not touch the medical record."""
    from assistant.memory.child_db import ChildMemoryDB

    db = ChildMemoryDB(tmp_path / "children.db")
    mem = ChatMemory(tmp_path / "chat.db")
    try:
        cid = db.create_child("Ada", "female", owner_user_id="user-a")
        db.add_growth(cid, 40.0, "weight", 3.4)
        _session_with_messages(mem, "user-a")

        mem.delete_all_sessions("user-a")

        assert db.get_child(cid, owner_user_id="user-a") is not None
        assert db.growth_history(cid), "growth measurements were lost"
    finally:
        mem.close()
        db.close()
