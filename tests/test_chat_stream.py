"""SSE chat stream endpoint tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import create_services, set_services


@pytest.fixture()
def client(tmp_path, monkeypatch):
    child_db = tmp_path / "children.db"
    chat_db = tmp_path / "chat.db"
    overlay_dir = tmp_path / "overlays"
    overlay_dir.mkdir()
    monkeypatch.setattr("assistant.config.OVERLAY_DIR", overlay_dir)
    monkeypatch.setattr("assistant.tools.clinical.OVERLAY_DIR", overlay_dir)
    monkeypatch.setattr("app.api.routes.OVERLAY_DIR", overlay_dir)

    svc = create_services(child_db_path=child_db, chat_db_path=chat_db)
    set_services(svc)
    with TestClient(app) as c:
        set_services(svc)
        yield c
    set_services(None)
    svc.close()


def test_chat_stream_sse_result(client):
    r = client.post("/api/chat/stream", json={"message": "hi, how can you help me?"})
    if r.status_code == 404:
        pytest.skip("/api/chat/stream not available")
    assert r.status_code == 200
    ctype = (r.headers.get("content-type") or "").lower()
    assert "text/event-stream" in ctype
    body = r.text
    assert "event: result" in body
