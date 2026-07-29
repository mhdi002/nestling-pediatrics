"""API tests for Nestling FastAPI backend (TestClient)."""

from __future__ import annotations

from pathlib import Path

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
        # lifespan may overwrite; pin our svc again
        set_services(svc)
        yield c
    set_services(None)
    svc.close()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_children_crud(client):
    r = client.post(
        "/api/children",
        json={"name": "Ali", "sex": "male", "gestational_age_weeks": 32},
    )
    assert r.status_code == 200
    data = r.json()
    assert "child_id" in data
    cid = data["child_id"]

    r2 = client.get("/api/children")
    assert r2.status_code == 200
    assert any(c["child_id"] == cid for c in r2.json()["children"])

    r3 = client.get(f"/api/children/{cid}")
    assert r3.status_code == 200
    assert r3.json()["child"]["name"] == "Ali"

    r4 = client.get("/api/children/does-not-exist")
    assert r4.status_code == 404
    body = r4.json()
    assert "error" in body


def test_session_and_chat(client):
    s = client.post("/api/sessions", json={})
    assert s.status_code == 200
    sid = s.json()["session_id"]

    r = client.post("/api/chat", json={"session_id": sid, "message": "hello"})
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == sid
    assert data.get("reply") or data.get("message")


def test_chat_auto_creates_session(client):
    """UI may omit session_id on first message — API must not 422."""
    r = client.post("/api/chat", json={"message": "hi, how can you help me?"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("session_id")
    assert data.get("reply")
    assert "help" in (data.get("intents") or []) or "Nestling" in data["reply"]


def test_child_dossier_endpoint(client):
    child = client.post(
        "/api/children",
        json={"name": "DossierKid", "sex": "female", "gestational_age_weeks": 32},
    ).json()
    cid = child["child_id"]
    client.post(
        "/api/growth",
        json={"child_id": cid, "sex": "female", "measure": "weight", "weeks": 40, "value": 2.9},
    )
    r = client.get(f"/api/children/{cid}/dossier")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["profile"]["name"] == "DossierKid"
    assert data["maturity"] == "preterm"
    assert len(data["growth"]) >= 1


def test_growth(client, tmp_path):
    child = client.post(
        "/api/children",
        json={"name": "Sara", "sex": "female", "gestational_age_weeks": 30},
    ).json()
    cid = child["child_id"]

    r = client.post(
        "/api/growth",
        json={
            "child_id": cid,
            "sex": "female",
            "measure": "weight",
            "weeks": 40,
            "value": 3.0,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "centile" in data
    assert "z_score" in data
    # Overlay PNG is best-effort (matplotlib); numeric assessment is required.
    assert data.get("ok") is not False
    if not (data.get("overlay") or data.get("overlay_path")):
        assert "plot_error" in data or data.get("centile") is not None


def test_asq_questions_and_score(client):
    # Prefer an age that exists in data/en/asq
    ages = [12, 10, 6, 4, 18]
    age = None
    for a in ages:
        r = client.get(f"/api/asq/{a}/questions")
        if r.status_code == 200:
            age = a
            break
    if age is None:
        pytest.skip("no ASQ data available")

    q = client.get(f"/api/asq/{age}/questions")
    assert q.status_code == 200
    assert "domains" in q.json() or "age_months" in q.json()

    child = client.post(
        "/api/children",
        json={"name": "ASQ Kid", "sex": "male", "gestational_age_weeks": 38},
    ).json()["child_id"]

    domain_answers = {
        "communication": ["yes"] * 6,
        "gross_motor": ["sometimes"] * 6,
        "fine_motor": ["yes"] * 6,
        "problem_solving": ["yes"] * 6,
        "personal_social": ["yes"] * 6,
    }
    r = client.post(
        "/api/asq/score",
        json={"child_id": child, "age_months": age, "domain_answers": domain_answers},
    )
    assert r.status_code == 200
    body = r.json()
    result = body.get("result") or body
    assert result.get("needs_referral") is False


def test_mchat_questions_and_score(client):
    r = client.get("/api/mchat/questions")
    if r.status_code == 404:
        pytest.skip("M-CHAT data missing")
    assert r.status_code == 200
    assert "questions" in r.json()

    answers = {str(i): "yes" for i in range(1, 21)}
    for i in (2, 5, 12):
        answers[str(i)] = "no"
    scored = client.post("/api/mchat/score", json={"answers": answers})
    assert scored.status_code == 200
    assert scored.json()["result"]["risk"] == "low"


def test_overlay_serve(client, tmp_path):
    import app.api.routes as routes

    overlay_dir = Path(routes.OVERLAY_DIR)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    name = "overlay_test.png"
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    (overlay_dir / name).write_bytes(png)
    r = client.get(f"/api/overlays/{name}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/png")

    bad = client.get("/api/overlays/../secret.png")
    assert bad.status_code in (400, 404)
