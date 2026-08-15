"""Regression tests for backend hardening bug fixes."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from assistant.agent.orchestrator import parse_tool_calls, resolve_known_age_months
from assistant.memory.chat_memory import ChatMemory
from assistant.memory.child_db import ChildMemoryDB
from assistant.rag.stores import VectorStore, _trim_extract
from assistant.refdata import weeks_per_month


# --- age resolution -----------------------------------------------------------


def test_postnatal_life_weeks_are_not_read_as_postmenstrual_age():
    """
    A 13.5-month-old born at 28w must not be reported as ~7 months.

    59 weeks of life stored on a WHO save looks like a plausible INTERGROWTH PMA;
    subtracting the gestational age was the original bug.
    """
    slots = {"weeks": 59.0, "gestational_age_weeks": 28.0}
    age = resolve_known_age_months(slots, None, None)
    assert age is not None
    assert 12.5 <= age <= 14.5, f"expected ~13.5 months, got {age}"


def test_intergrowth_pma_still_converts_to_chronological_age():
    """The preterm chart genuinely stores PMA — that path must keep subtracting GA."""
    slots = {
        "weeks": 40.0,
        "gestational_age_weeks": 30.0,
        "last_chart_standard": "intergrowth_preterm",
    }
    age = resolve_known_age_months(slots, None, None)
    expected = (40.0 - 30.0) / weeks_per_month()
    assert age == pytest.approx(expected, abs=0.05)


def test_explicit_age_months_always_wins():
    slots = {"age_months": 13.5, "weeks": 59.0, "gestational_age_weeks": 28.0}
    assert resolve_known_age_months(slots, None, None) == pytest.approx(13.5)


def test_future_date_of_birth_does_not_produce_a_negative_age():
    from assistant.agent.orchestrator import _age_months_from_dob

    assert _age_months_from_dob("2999-01-01") is None
    assert _age_months_from_dob("not-a-date") is None
    assert _age_months_from_dob(None) is None


def test_growth_route_ignores_derived_weeks_for_preterm_child():
    """weeks == age_months * weeks_per_month is postnatal, never a PMA."""
    from assistant.tools.clinical import resolve_chart_route

    age_months = 13.5
    out = resolve_chart_route(
        age_months=age_months,
        weeks=age_months * weeks_per_month(),
        gestational_age_weeks=28.0,
    )
    assert out["ok"] is True
    assert out.get("age_months") == pytest.approx(age_months, abs=0.05)


# --- parsing / validation -----------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "not json", "[1, 2, 3]", '"a string"', "null", '{"tool_calls": "nope"}'],
)
def test_parse_tool_calls_never_raises_on_junk(raw):
    assert parse_tool_calls(raw) == []


def test_parse_tool_calls_reads_an_embedded_object():
    raw = 'thinking...\n{"tool_calls": [{"name": "growth_percentile"}]}\ndone'
    assert parse_tool_calls(raw) == [{"name": "growth_percentile"}]


# --- RAG index robustness -----------------------------------------------------


def test_corrupt_index_is_ignored_instead_of_crashing(tmp_path):
    """A truncated docs.json must mean 'rebuild', not a 500 on every request."""
    (tmp_path / "docs.json").write_text("", encoding="utf-8")
    store = VectorStore("test", tmp_path)
    assert store.load() is False

    (tmp_path / "docs.json").write_text('{"not": "a list"}', encoding="utf-8")
    assert store.load() is False


def test_index_save_is_atomic_and_reloadable(tmp_path):
    store = VectorStore("test", tmp_path)
    store.add([{"id": "a", "title": "Iron", "text": "iron drops for breastfed infants"}])
    store.save()
    assert json.loads((tmp_path / "docs.json").read_text(encoding="utf-8"))
    assert not list(tmp_path.glob("*.tmp"))

    other = VectorStore("test", tmp_path)
    assert other.load() is True
    assert other.docs[0]["id"] == "a"


def test_extract_trim_keeps_the_configured_budget():
    """Snapping to a very early sentence end used to discard 40% of the answer."""
    text = "Short intro. " + "word " * 300
    trimmed = _trim_extract(text, max_chars=420, min_sentence_chars=160, keep_ratio=0.75)
    assert len(trimmed) > 300
    assert len(trimmed) <= 421


def test_extract_trim_prefers_a_late_sentence_end():
    body = "a" * 380 + ". " + "b" * 200
    trimmed = _trim_extract(body, max_chars=420, min_sentence_chars=160, keep_ratio=0.75)
    assert trimmed.endswith(".")


# --- concurrency --------------------------------------------------------------


def test_chat_memory_survives_concurrent_writers(tmp_path):
    """One SQLite connection is shared by every request thread."""
    mem = ChatMemory(path=tmp_path / "chat.db")
    try:
        sid = mem.create_session()
        errors: list[Exception] = []

        def writer(n: int) -> None:
            try:
                for i in range(20):
                    mem.add_message(sid, "user", f"worker {n} message {i}")
                    mem.merge_slots(sid, {"last_topic": f"topic-{n}-{i}"})
            except Exception as exc:  # pragma: no cover - only on a locking regression
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"concurrent writes failed: {errors}"
        assert len(mem.get_history(sid)) == 8 * 20
    finally:
        mem.close()


def test_child_db_survives_concurrent_writers(tmp_path):
    db = ChildMemoryDB(path=tmp_path / "children.db")
    try:
        cid = db.create_child("Concurrent", "female", gestational_age_weeks=39)
        errors: list[Exception] = []

        def writer(n: int) -> None:
            try:
                for i in range(15):
                    db.add_growth(cid, weeks=40.0 + i, measure="weight", value=3.0 + i / 10)
            except Exception as exc:  # pragma: no cover - only on a locking regression
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"concurrent writes failed: {errors}"
        assert len(db.growth_history(cid)) == 6 * 15
    finally:
        db.close()


def test_history_limit_is_applied_in_sql(tmp_path):
    """A long session must not load every row to return the newest few."""
    mem = ChatMemory(path=tmp_path / "chat.db")
    try:
        sid = mem.create_session()
        for i in range(50):
            mem.add_message(sid, "user", f"message {i}")
        recent = mem.get_history(sid, limit=5)
        assert len(recent) == 5
        assert recent[-1]["content"] == "message 49"
        assert mem.get_history(sid, limit=0) == []
    finally:
        mem.close()


def test_services_singleton_is_built_once_under_concurrency(monkeypatch):
    import app.services as services

    built: list[int] = []

    class FakeServices:
        def __init__(self) -> None:
            built.append(1)

    monkeypatch.setattr(services, "_services", None)
    monkeypatch.setattr(services, "create_services", FakeServices)
    try:
        results = []
        threads = [
            threading.Thread(target=lambda: results.append(services.get_services()))
            for _ in range(12)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(built) == 1
        assert len({id(r) for r in results}) == 1
    finally:
        services.set_services(None)


# --- resource handling --------------------------------------------------------


def test_dense_cache_is_bounded():
    from assistant.rag import dense

    dense._cache.clear()
    for i in range(50):
        dense._cache_put(f"k{i}", i, max_size=10)
    assert len(dense._cache) == 10
    # Oldest keys are evicted first.
    assert "k0" not in dense._cache
    assert "k49" in dense._cache


def test_closing_services_releases_every_handle(tmp_path):
    from app.services import Services

    class Boom:
        def close(self):
            raise RuntimeError("assistant failed to close")

    db = ChildMemoryDB(path=tmp_path / "children.db")
    chat = ChatMemory(path=tmp_path / "chat.db")
    # A failure in one closer must not leak the other two connections.
    Services(db=db, chat=chat, assistant=Boom()).close()

    for closed in (db, chat):
        assert closed.conn is None or _is_closed(closed.conn)


def _is_closed(conn) -> bool:
    try:
        conn.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return True
    return False
