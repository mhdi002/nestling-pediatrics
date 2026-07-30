"""Property / edge-case tests for the clinical layer only."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import intergrowth_preterm_equations as ig
from assistant.tools.clinical import (
    corrected_age_months,
    resolve_chart_route,
    score_asq_domain,
)
from assistant.tools.who_term_equations import AgeOutOfRangeError
from assistant.tools.who_term_equations import z_score as who_z_score

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:  # pragma: no cover
    HAS_HYPOTHESIS = False

SEXES = ("male", "female")
MEASURES = ("weight", "length", "head_circumference")
WEEKS = (27, 40, 64)
CHART_PS = (3, 10, 50, 90, 97)


def _assert_ig_percentile_monotonic(sex: str, measure: str, weeks: float) -> None:
    vals = [ig.percentile(sex, measure, weeks, p) for p in CHART_PS]
    assert all(math.isfinite(v) and v > 0 for v in vals)
    assert vals == sorted(vals)
    # strict increase across distinct chart percentiles
    assert all(a < b for a, b in zip(vals, vals[1:]))


def _assert_ig_z_centile_roundtrip(sex: str, measure: str, weeks: float, p: float) -> None:
    value = ig.percentile(sex, measure, weeks, p)
    z = ig.z_score(sex, measure, weeks, value)
    recovered = ig.centile_from_measurement(sex, measure, weeks, value)
    # z at percentile p should match NormalDist inverse CDF (within float noise)
    expected_z = ig.z_for_percentile(p)
    assert abs(z - expected_z) < 1e-9
    assert abs(recovered - p) < 0.05


def test_intergrowth_percentile_monotonic_over_grid():
    for sex in SEXES:
        for measure in MEASURES:
            for weeks in WEEKS:
                _assert_ig_percentile_monotonic(sex, measure, weeks)


def test_intergrowth_z_centile_roundtrip_over_grid():
    for sex in SEXES:
        for measure in MEASURES:
            for weeks in WEEKS:
                for p in CHART_PS:
                    _assert_ig_z_centile_roundtrip(sex, measure, weeks, p)


if HAS_HYPOTHESIS:

    @given(
        sex=st.sampled_from(SEXES),
        measure=st.sampled_from(MEASURES),
        weeks=st.sampled_from(WEEKS),
        p=st.sampled_from(CHART_PS),
    )
    @settings(max_examples=40, deadline=None)
    def test_intergrowth_properties_hypothesis(sex, measure, weeks, p):
        _assert_ig_percentile_monotonic(sex, measure, weeks)
        _assert_ig_z_centile_roundtrip(sex, measure, weeks, p)


def test_who_z_score_age_out_of_range_raises():
    with pytest.raises(AgeOutOfRangeError):
        who_z_score("male", "weight", 48.0, 15.0)
    with pytest.raises(AgeOutOfRangeError):
        who_z_score("female", "length", -1.0, 50.0)


def test_corrected_age_ga32_chrono12_uses_corrected():
    info = corrected_age_months(12.0, gestational_age_weeks=32)
    assert info["used_corrected"] is True


def test_resolve_chart_route_weeks40_without_ga_needs_ga():
    out = resolve_chart_route(weeks=40)
    assert out["ok"] is False
    assert out.get("needs_gestational_age") is True


def test_resolve_chart_route_age_months6_without_ga_assumes_term():
    out = resolve_chart_route(age_months=6)
    assert out["ok"] is True
    assert out["chart_standard"] == "who_term"
    assert out.get("assumed_term") is True


def test_score_asq_domain_includes_cutoff_source():
    out = score_asq_domain(["yes"] * 6)
    assert out["ok"] is True
    assert "cutoff_source" in out
    assert isinstance(out["cutoff_source"], str)
    assert out["cutoff_source"]
