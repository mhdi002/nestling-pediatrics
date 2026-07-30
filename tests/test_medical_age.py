"""Regression: medical feeding age must use chronological months, not PMA−GA ≈ 7."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.agent.orchestrator import resolve_known_age_months
from assistant.memory.child_db import ChildMemoryDB
from assistant.refdata import weeks_per_month
from assistant.tools.clinical import growth_percentile


def test_resolve_prefers_last_age_months_over_pma_weeks():
    db = ChildMemoryDB(":memory:")
    slots = {
        "weeks": 59.0,
        "gestational_age_weeks": 28.0,
        "last_chart_standard": "intergrowth_preterm",
        "last_age_months": 13.5,
    }
    assert abs(resolve_known_age_months(slots, None, db) - 13.5) < 0.01


def test_resolve_life_weeks_not_ga_subtracted_when_who_stored():
    """WHO save stores weeks=age_months*wpm; must not become ~7m via GA subtract."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = ChildMemoryDB(Path(td) / "c.db")
        cid = db.create_child("Maya", "female", gestational_age_weeks=28)
        life_w = 13.5 * weeks_per_month()
        db.add_growth(
            cid,
            weeks=life_w,
            measure="weight",
            value=9.0,
            centile=50,
            age_months=13.5,
        )
        age = resolve_known_age_months({}, cid, db)
        assert age is not None and abs(age - 13.5) < 0.05


def test_resolve_heuristic_life_weeks_without_age_months_column():
    """Legacy row: weeks≈59 life-weeks + GA 28 must not resolve to ~7 months."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = ChildMemoryDB(Path(td) / "c.db")
        cid = db.create_child("Maya", "female", gestational_age_weeks=28)
        db.add_growth(cid, weeks=59.0, measure="weight", value=1.0, centile=0.1)
        age = resolve_known_age_months({}, cid, db)
        assert age is not None
        # Prefer life-weeks reading (~13.6m) over PMA−GA (~7.1m)
        assert age >= 12.0
        assert abs(age - 59.0 / weeks_per_month()) < 0.2


def test_growth_13_5m_preterm_keeps_chronological_for_medical():
    r = growth_percentile(
        "female",
        "weight",
        age_months=13.5,
        weeks=13.5 * weeks_per_month(),
        value=9.0,
        gestational_age_weeks=28,
        chart_standard="intergrowth_preterm",
    )
    assert r["ok"]
    assert abs(float(r["age_months"]) - 13.5) < 0.05
    slots = {
        "last_age_months": r["age_months"],
        "last_chart_standard": r["chart_standard"],
        "weeks": r.get("weeks") or 13.5 * weeks_per_month(),
        "gestational_age_weeks": 28,
    }
    db = ChildMemoryDB(":memory:")
    known = resolve_known_age_months(slots, None, db)
    assert known is not None and abs(known - 13.5) < 0.05
