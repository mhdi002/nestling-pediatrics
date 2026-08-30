"""
Cross-account (tenant) isolation at the API layer.

A signed-in account must never read or mutate another account's child record,
dossier, growth chart, screenings, or chat session by passing the other
account's ids. These are regression tests for a set of IDOR holes where routes
reached a child_id / session_id without scoping it to the caller:

  * GET  /api/children/{id}/dossier  returned the full medical record unscoped
  * POST /api/growth                 wrote a growth row into any child
  * POST /api/mchat/score, /asq/score wrote a screening into any child
  * POST /api/chat (+ /chat/stream)  reused any session id, appending the
                                     caller's turn to it and echoing its slots

Every request here is authenticated with a real bearer token, so the scoping —
not the open-API bypass — is what is under test.
"""

from __future__ import annotations

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

    # The login/register throttle is process-global; clear it so repeated
    # registrations across tests from the single TestClient IP don't trip it.
    import app.api.routes as _routes

    _routes._LOGIN_ATTEMPTS.clear()

    svc = create_services(
        child_db_path=tmp_path / "children.db", chat_db_path=tmp_path / "chat.db"
    )
    set_services(svc)
    with TestClient(app) as c:
        set_services(svc)
        yield c
    set_services(None)
    svc.close()


def _register(client: TestClient, username: str) -> str:
    r = client.post(
        "/api/auth/register", json={"username": username, "password": "Password123!"}
    )
    assert r.status_code == 200, r.text
    return r.json()["token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def two_accounts(client):
    """Attacker A and victim B, where B owns a child + a session with history."""
    a = _register(client, "attacker")
    b = _register(client, "victimacct")
    child = client.post(
        "/api/children",
        headers=_auth(b),
        json={
            "name": "VictimBaby",
            "sex": "female",
            "gestational_age_weeks": 32,
            "notes": "confidential clinical note",
        },
    ).json()["child_id"]
    sess = client.post(
        "/api/sessions",
        headers=_auth(b),
        json={"child_id": child, "title": "B private chat"},
    ).json()["session_id"]
    return {"a": a, "b": b, "child": child, "session": sess}


def test_dossier_is_not_readable_across_accounts(client, two_accounts):
    a, child = two_accounts["a"], two_accounts["child"]
    r = client.get(f"/api/children/{child}/dossier", headers=_auth(a))
    assert r.status_code == 404, r.text
    assert "confidential" not in r.text
    # And the legitimate owner still gets it.
    ok = client.get(
        f"/api/children/{child}/dossier", headers=_auth(two_accounts["b"])
    )
    assert ok.status_code == 200
    assert ok.json()["profile"]["name"] == "VictimBaby"


def test_growth_cannot_be_written_to_another_accounts_child(client, two_accounts):
    a, b, child = two_accounts["a"], two_accounts["b"], two_accounts["child"]
    r = client.post(
        "/api/growth",
        headers=_auth(a),
        json={"child_id": child, "sex": "female", "measure": "weight", "weeks": 40, "value": 3.5},
    )
    assert r.status_code == 404, r.text
    # The victim's chart stayed empty — nothing landed.
    dossier = client.get(f"/api/children/{child}/dossier", headers=_auth(b)).json()
    assert dossier["growth"] == []


def test_mchat_cannot_be_scored_on_another_accounts_child(client, two_accounts):
    a, b, child = two_accounts["a"], two_accounts["b"], two_accounts["child"]
    r = client.post(
        "/api/mchat/score",
        headers=_auth(a),
        json={"child_id": child, "answers": {"1": "no", "2": "no"}},
    )
    assert r.status_code == 404, r.text
    dossier = client.get(f"/api/children/{child}/dossier", headers=_auth(b)).json()
    assert dossier["screenings"] == []


def test_asq_cannot_be_scored_on_another_accounts_child(client, two_accounts):
    a, child = two_accounts["a"], two_accounts["child"]
    r = client.post(
        "/api/asq/score",
        headers=_auth(a),
        json={"child_id": child, "age_months": 12, "domain_answers": {"communication": ["yes"]}},
    )
    assert r.status_code == 404, r.text


def test_chat_cannot_hijack_another_accounts_session(client, two_accounts):
    a, b, child, sess = (
        two_accounts["a"],
        two_accounts["b"],
        two_accounts["child"],
        two_accounts["session"],
    )
    # B records some private history in the session.
    client.post(
        "/api/chat",
        headers=_auth(b),
        json={"session_id": sess, "child_id": child, "message": "my private detail"},
    )
    before = client.get(f"/api/sessions/{sess}", headers=_auth(b)).json()
    before_count = len(before["history"])

    # A posts into B's session id. It must be redirected to a fresh, A-owned
    # session rather than appended to B's, and must not echo B's child slots.
    r = client.post(
        "/api/chat", headers=_auth(a), json={"session_id": sess, "message": "hello"}
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["session_id"] != sess, "attacker turn landed in the victim's session"
    assert data.get("child_id") != child
    assert (data.get("slots") or {}).get("child_id") != child

    # B's session is unchanged: A's turn is not in it, and A cannot read it.
    after = client.get(f"/api/sessions/{sess}", headers=_auth(b)).json()
    assert len(after["history"]) == before_count
    assert client.get(f"/api/sessions/{sess}", headers=_auth(a)).status_code == 404


def test_chat_rejects_another_accounts_child_id(client, two_accounts):
    a, child = two_accounts["a"], two_accounts["child"]
    r = client.post(
        "/api/chat", headers=_auth(a), json={"message": "hi", "child_id": child}
    )
    assert r.status_code == 404, r.text
