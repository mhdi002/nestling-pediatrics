"""Regression tests for API input validation and error handling."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import create_services, set_services


@pytest.fixture()
def client(tmp_path, monkeypatch):
    overlay_dir = tmp_path / "overlays"
    overlay_dir.mkdir()
    monkeypatch.setattr("assistant.config.OVERLAY_DIR", overlay_dir)
    monkeypatch.setattr("assistant.tools.clinical.OVERLAY_DIR", overlay_dir)
    monkeypatch.setattr("app.api.routes.OVERLAY_DIR", overlay_dir)

    svc = create_services(
        child_db_path=tmp_path / "children.db", chat_db_path=tmp_path / "chat.db"
    )
    set_services(svc)
    with TestClient(app) as c:
        set_services(svc)
        yield c
    set_services(None)
    svc.close()


def _png_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


# --- numeric bounds -----------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"sex": "female", "measure": "weight", "weeks": 40, "value": -5},
        {"sex": "female", "measure": "weight", "weeks": -1, "value": 3.0},
        {"sex": "female", "measure": "weight", "weeks": 1e9, "value": 3.0},
        {"sex": "female", "measure": "weight", "age_months": -3, "value": 3.0},
    ],
)
def test_growth_rejects_out_of_range_numbers(client, payload):
    r = client.post("/api/growth", json=payload)
    assert r.status_code in (400, 422), r.text


def test_growth_accepts_a_valid_measurement(client):
    r = client.post(
        "/api/growth",
        json={
            "sex": "female",
            "measure": "weight",
            "weeks": 40,
            "value": 3.0,
            "gestational_age_weeks": 30,
        },
    )
    assert r.status_code == 200, r.text


def test_child_rejects_an_implausible_gestational_age(client):
    r = client.post(
        "/api/children",
        json={"name": "GA Kid", "sex": "male", "gestational_age_weeks": 900},
    )
    assert r.status_code == 422


def test_asq_rejects_an_out_of_range_age(client):
    r = client.post(
        "/api/asq/score",
        json={"age_months": 5000, "domain_answers": {"communication": ["yes"]}},
    )
    assert r.status_code == 422


def test_mchat_rejects_non_numeric_question_keys(client):
    r = client.post("/api/mchat/score", json={"answers": {"not-a-number": "yes"}})
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_answers"


def test_empty_chat_message_is_rejected(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


# --- upload validation --------------------------------------------------------


def test_vision_rejects_a_non_image_content_type(client):
    r = client.post(
        "/api/chat/vision",
        files={"image": ("notes.txt", b"hello", "text/plain")},
        data={"message": "look"},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "unsupported_media_type"


def test_vision_rejects_an_oversized_upload(client, monkeypatch):
    from assistant.settings import Settings, get_settings

    small = Settings(**{**get_settings().model_dump(), "nestling_max_upload_bytes": 64})
    monkeypatch.setattr("app.api.routes.get_settings", lambda: small)
    r = client.post(
        "/api/chat/vision",
        files={"image": ("big.png", b"\x89PNG" + b"0" * 5000, "image/png")},
        data={"message": "look"},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "image_too_large"


def test_vision_rejects_bytes_that_are_not_a_real_image(client):
    r = client.post(
        "/api/chat/vision",
        files={"image": ("fake.png", b"definitely not a png", "image/png")},
        data={"message": "look"},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_image"


# --- path traversal -----------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["../secret.png", "..%2Fsecret.png", "sub/dir.png", "C:secret.png"]
)
def test_overlay_download_stays_inside_the_overlay_dir(client, name):
    r = client.get(f"/api/overlays/{name}")
    assert r.status_code in (400, 404), r.text


# --- response bounds ----------------------------------------------------------


def test_session_history_is_bounded(client):
    from assistant.settings import get_settings

    sid = client.post("/api/sessions", json={}).json()["session_id"]

    from app.services import get_services

    chat = get_services().chat
    limit = get_settings().nestling_history_response_limit
    for i in range(limit + 10):
        chat.add_message(sid, "user", f"m{i}")

    body = client.get(f"/api/sessions/{sid}").json()
    assert len(body["history"]) == limit
    assert body["history"][-1]["content"] == f"m{limit + 9}"


def test_session_list_limit_is_validated(client):
    assert client.get("/api/sessions?limit=0").status_code == 422
    assert client.get("/api/sessions?limit=5").status_code == 200


def test_unhandled_errors_do_not_leak_internals(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr("app.api.routes._run_chat_turn", boom)
    r = client.post("/api/chat", json={"message": "hello"})
    assert r.status_code == 500
    assert r.json()["error"] == "chat_failed"


def test_health_needs_no_llm_sidecar(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm"]["configured"] is False
