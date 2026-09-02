"""The admission gate as the HTTP contract sees it.

A saturated app must answer 503 with a Retry-After, not hang the caller until
a 504 or fall over with a 500, and /api/health must report the occupancy an
operator sizes a fleet from. Driven through the real app with two concurrent
requests and a deliberately slow turn, so the behaviour is the wire behaviour.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

import app.api.routes as routes
from app.concurrency import ChatGate
from app.main import app
from app.services import create_services, set_services


@pytest.fixture()
def client(tmp_path):
    svc = create_services(
        child_db_path=tmp_path / "children.db", chat_db_path=tmp_path / "chat.db"
    )
    set_services(svc)
    with TestClient(app) as c:
        # After lifespan startup (which installs its own gate), inject ours.
        set_services(svc)
        yield c
    set_services(None)
    svc.close()


def test_health_reports_capacity(client):
    routes.set_chat_gate(ChatGate(4, 2, 5.0))
    body = client.get("/api/health").json()
    assert "capacity" in body
    cap = body["capacity"]
    assert cap["chat_gate_enabled"] is True
    assert cap["max_inflight"] == 4
    assert cap["max_waiting"] == 2
    assert cap["inflight"] == 0


def test_a_saturated_app_returns_503_not_a_hang(client, monkeypatch):
    """One slot, no queue: while it is held, the next chat gets a clean 503."""
    routes.set_chat_gate(ChatGate(max_inflight=1, max_waiting=0, acquire_timeout=2.0))

    release = threading.Event()
    holding = threading.Event()
    real_turn = routes._run_chat_turn

    def slow_turn(body, owner_user_id=None):
        holding.set()
        release.wait(10)
        return real_turn(body, owner_user_id)

    monkeypatch.setattr(routes, "_run_chat_turn", slow_turn)

    got = {}

    def hold():
        got["hold"] = client.post("/api/chat", json={"message": "hello"})

    t = threading.Thread(target=hold)
    t.start()
    assert holding.wait(10), "the first request never entered the turn"

    # Second request, while the slot is held, is shed immediately.
    r = client.post("/api/chat", json={"message": "second"})
    assert r.status_code == 503, r.text
    body = r.json()
    assert body["error"] == "busy"
    assert "retry_after" in body
    assert r.headers.get("Retry-After")

    # And /api/health shows the one in flight.
    cap = client.get("/api/health").json()["capacity"]
    assert cap["inflight"] == 1

    release.set()
    t.join(10)
    assert got["hold"].status_code == 200


def test_a_freed_slot_admits_the_next_request(client, monkeypatch):
    routes.set_chat_gate(ChatGate(max_inflight=1, max_waiting=0, acquire_timeout=2.0))

    # First request occupies and frees the slot.
    r1 = client.post("/api/chat", json={"message": "first"})
    assert r1.status_code == 200, r1.text
    # Second request, now that the slot is free, is admitted (not 503).
    r2 = client.post("/api/chat", json={"message": "second"})
    assert r2.status_code == 200, r2.text
    assert client.get("/api/health").json()["capacity"]["inflight"] == 0


def test_a_disabled_gate_never_sheds(client):
    """App-only / un-sized: the gate must not cap anything."""
    routes.set_chat_gate(ChatGate(0, 0, 0.0))
    # Several sequential turns all succeed; the gate is transparent.
    for _ in range(3):
        r = client.post("/api/chat", json={"message": "hi"})
        assert r.status_code == 200, r.text
    assert client.get("/api/health").json()["capacity"]["chat_gate_enabled"] is False
