"""Growth curves endpoint for client SVG charts."""

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


def test_growth_curves_defaults(client):
    r = client.get("/api/growth/curves")
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert data.get("chart_standard") == "who_term"
    assert data.get("age_unit") == "months"
    assert data.get("percentiles") == [3, 10, 50, 90, 97]
    assert "50" in data.get("curves", {})
    assert len(data.get("ages") or []) > 0
    assert len(data["curves"]["50"]) == len(data["ages"])


def test_growth_curves_intergrowth(client):
    r = client.get(
        "/api/growth/curves",
        params={
            "sex": "female",
            "measure": "length",
            "gestational_age_weeks": 32,
            "age_max": 40,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["chart_standard"] == "intergrowth_preterm"
    assert data["age_unit"] == "weeks"
    assert data["ages"][0] >= 27.0
    assert data["ages"][-1] <= 40.0
