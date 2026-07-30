"""Regression tests for clinical guards (age bounds, unknown GA, corrected age)."""

from __future__ import annotations

import pytest

from assistant.tools.clinical import (
    corrected_age_months,
    growth_percentile,
    list_asq_questions,
    resolve_chart_route,
)
from assistant.tools.who_term_equations import AgeOutOfRangeError, z_score


def test_unknown_ga_refuses_silent_preterm_default():
    out = growth_percentile("male", "weight", weeks=40, value=3.2)
    assert out["ok"] is False
    assert out.get("needs_gestational_age") or "gestational" in (out.get("error") or "").lower()


def test_age_months_without_ga_assumes_term():
    out = growth_percentile("male", "weight", age_months=6, value=7.9)
    assert out["ok"] is True
    assert out["chart_standard"] == "who_term"
    assert out.get("assumed_term") is True
    assert out.get("assumption_note")


def test_explicit_chart_standard_bypasses_ga():
    out = growth_percentile(
        "male", "weight", weeks=40, value=3.2, chart_standard="intergrowth_preterm"
    )
    assert out["ok"] is True
    assert out["chart_standard"] == "intergrowth_preterm"


def test_who_age_out_of_range_errors():
    out = growth_percentile(
        "male", "weight", age_months=36, value=14.0, gestational_age_weeks=40
    )
    assert out["ok"] is False
    assert "24" in (out.get("error") or "") or "months" in (out.get("error") or "").lower()


def test_who_direct_z_score_rejects_out_of_range():
    with pytest.raises(AgeOutOfRangeError):
        z_score("male", "weight", 48.0, 15.0)


def test_corrected_age_preterm():
    info = corrected_age_months(12.0, gestational_age_weeks=32)
    assert info["used_corrected"] is True
    # 8 weeks early ≈ 1.84 months → corrected ≈ 10.16
    assert 9.5 < info["corrected_age_months"] < 10.5
    assert info["age_for_questionnaire"] == info["corrected_age_months"]


def test_corrected_age_term_unchanged():
    info = corrected_age_months(12.0, gestational_age_weeks=39)
    assert info["used_corrected"] is False
    assert info["age_for_questionnaire"] == 12.0


def test_asq_list_uses_corrected_age_for_preterm():
    # Chronological 12m, GA 28w → ~3 months early → corrected ~9m → nearest form
    out = list_asq_questions(12, gestational_age_weeks=28)
    assert out["ok"] is True
    assert out["used_corrected"] is True
    assert out["age_months"] != 12 or out["corrected_age_months"] is not None
    # Form age should be closer to corrected than to chronological 12
    assert abs(out["age_months"] - out["corrected_age_months"]) <= abs(out["age_months"] - 12) + 0.01


def test_who_precision_note_present_when_interpolated():
    out = growth_percentile(
        "male", "length", age_months=7.5, value=70.0, gestational_age_weeks=40
    )
    assert out["ok"] is True
    # 7.5 sits between 6 and 9 anchors
    assert out.get("precision_note") or out.get("lms_anchors")
