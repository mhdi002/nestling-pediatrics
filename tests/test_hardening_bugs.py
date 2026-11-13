"""Regression tests for backend hardening bug fixes."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from assistant.memory.chat_memory import ChatMemory
from assistant.memory.child_db import ChildMemoryDB
from assistant.refdata import weeks_per_month


# --- age resolution -----------------------------------------------------------


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


# --- RAG index robustness -----------------------------------------------------


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


def test_future_date_of_birth_does_not_produce_a_negative_age():
    from datetime import date, timedelta

    from assistant.rlm.clinical_dispatch import _age_months

    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    assert _age_months(tomorrow) is None
    assert _age_months((date.today() - timedelta(days=61)).isoformat()) > 1.9
