"""Robustness tests for clinical tools — validation & edge cases."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.memory.child_db import ChildMemoryDB
from assistant.tools.clinical import (
    TOOL_SPECS,
    WEEKS_MAX,
    WEEKS_MIN,
    dispatch_tool,
    get_child_summary,
    growth_percentile,
    list_asq_questions,
    list_mchat_questions,
    overlay_growth_on_chart,
    set_child_db,
)


def test_tool_specs_cover_all_tools():
    names = {t["name"] for t in TOOL_SPECS}
    assert {
        "growth_percentile",
        "overlay_growth_on_chart",
        "score_asq_questionnaire",
        "score_mchat",
        "list_asq_questions",
        "list_mchat_questions",
        "get_child_summary",
    } <= names
    for t in TOOL_SPECS:
        assert "parameters" in t
        assert "properties" in t["parameters"]


def test_weeks_bounds_rejected():
    low = growth_percentile(
        "male", "weight", WEEKS_MIN - 1, value=3.0, chart_standard="intergrowth_preterm"
    )
    assert low["ok"] is False
    assert "27" in low["error"] or "weeks" in low["error"].lower()

    high = growth_percentile(
        "female", "weight", WEEKS_MAX + 0.1, value=5.0, chart_standard="intergrowth_preterm"
    )
    assert high["ok"] is False

    ok = growth_percentile(
        "male", "weight", 40, value=3.2, chart_standard="intergrowth_preterm"
    )
    assert ok["ok"] is True
    assert ok["centile"] is not None


def test_invalid_sex_and_measure():
    assert growth_percentile(
        "unknown", "weight", 40, chart_standard="intergrowth_preterm"
    )["ok"] is False
    assert growth_percentile(
        "male", "bmi", 40, chart_standard="intergrowth_preterm"
    )["ok"] is False


def test_value_range_rejected():
    huge = growth_percentile(
        "male", "weight", 40, value=999, chart_standard="intergrowth_preterm"
    )
    assert huge["ok"] is False
    assert "value" in huge["error"].lower() or "between" in huge["error"].lower()


def test_dispatch_unknown_tool_error_dict():
    out = dispatch_tool("not_a_real_tool", {})
    assert out["ok"] is False
    assert out["error"] == "unknown_tool"


def test_overlay_missing_matplotlib_graceful():
    with mock.patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
        # Force import failure inside overlay by patching the import path
        real_import = __import__

        def _blocked(name, *args, **kwargs):
            if name.startswith("matplotlib"):
                raise ImportError("mocked missing matplotlib")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=_blocked):
            out = overlay_growth_on_chart(
                "male", "weight", 40, 3.2, chart_standard="intergrowth_preterm"
            )
    assert out.get("ok") is True  # numeric assessment still present
    assert out.get("overlay_path") is None
    assert out.get("centile") is not None
    assert "matplotlib" in (out.get("plot_error") or "").lower() or out.get("note")


def test_list_asq_and_mchat_questions():
    asq = list_asq_questions(4)
    assert asq["ok"] is True
    assert asq["question_count"] > 0
    assert asq["domains"]

    missing = list_asq_questions(999)
    assert missing["ok"] is False

    mchat = list_mchat_questions()
    assert mchat["ok"] is True
    assert mchat["question_count"] >= 20


def test_get_child_summary_tool():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = ChildMemoryDB(Path(td) / "c.db")
        try:
            set_child_db(db)
            cid = db.create_child("Lia", "female", gestational_age_weeks=32)
            db.add_growth(cid, weeks=40, measure="weight", value=3.0, centile=40.0)
            out = get_child_summary(cid)
            assert out["ok"] is True
            assert out["profile"]["name"] == "Lia"
            assert out["growth_count"] == 1
            bad = get_child_summary("no-such-id")
            assert bad["ok"] is False
        finally:
            set_child_db(None)
            db.close()


def test_boundary_weeks_27_and_64():
    a = growth_percentile(
        "male", "weight", 27, value=0.9, chart_standard="intergrowth_preterm"
    )
    b = growth_percentile(
        "female", "length", 64, value=65.0, chart_standard="intergrowth_preterm"
    )
    assert a["ok"] is True
    assert b["ok"] is True


def test_score_mchat_invalid_returns_error_dict():
    out = dispatch_tool("score_mchat", {"answers": {"1": "maybe"}})
    assert out["ok"] is False


def test_missing_ga_requires_clarification():
    out = growth_percentile("male", "weight", 40, value=3.2)
    assert out["ok"] is False
    assert out.get("needs_gestational_age") is True
