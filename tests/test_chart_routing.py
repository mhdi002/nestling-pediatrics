"""Chart routing + WHO term + INTERGROWTH preterm accuracy."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.tools.clinical import growth_percentile, score_asq_questionnaire, score_mchat
from assistant.tools.who_term_equations import classify_maturity, percentile as who_pct
import intergrowth_preterm_equations as ig


def test_maturity_classifier():
    assert classify_maturity(32) == "preterm"
    assert classify_maturity(37) == "term"
    assert classify_maturity(40) == "term"


def test_preterm_routes_to_intergrowth():
    r = growth_percentile(
        "male",
        "weight",
        weeks=40,
        value=3.2,
        gestational_age_weeks=32,
    )
    assert r["ok"] is True
    assert r["chart_standard"] == "intergrowth_preterm"
    assert abs(r["centile"] - 30.85) < 0.2


def test_term_routes_to_who():
    r = growth_percentile(
        "male",
        "weight",
        age_months=2,
        value=5.6,
        gestational_age_weeks=39,
    )
    assert r["ok"] is True
    assert r["chart_standard"] == "who_term"
    assert r["maturity_label_fa"] == "طبیعی"
    assert r["maturity_label_en"] == "term"
    assert 1 < r["centile"] < 99


def test_term_40w_is_near_birth_not_nine_months():
    """Regression: '40 weeks' for a term baby must not plot at ~9 months on WHO."""
    r = growth_percentile(
        "male",
        "weight",
        weeks=40,
        value=3.2,
        gestational_age_weeks=39,
    )
    assert r["ok"] is True
    assert r["chart_standard"] == "who_term"
    # 40w PMA − 39w GA ≈ 1 week ≈ 0.23 months — not 40/4.345≈9.2
    assert r["age_months"] is not None and r["age_months"] < 1.0
    assert r["centile"] > 10  # newborn-range weight, not extreme low


def test_who_p50_near_median_at_birth():
    # Male birth weight median ~3.35 kg → centile ~50
    r = growth_percentile(
        "male",
        "weight",
        age_months=0,
        value=3.3464,
        gestational_age_weeks=40,
    )
    assert r["ok"]
    assert abs(r["centile"] - 50) < 2


def test_intergrowth_published_still_holds():
    c = ig.centile_from_measurement("male", "weight", 27.0, 0.99)
    assert abs(c - 96.89) < 0.05


def test_asq_all_domain_referral_logic():
    domains = {
        "communication": ["not_yet"] * 6,
        "gross_motor": ["yes"] * 6,
        "fine_motor": ["yes"] * 6,
        "problem_solving": ["yes"] * 6,
        "personal_social": ["yes"] * 6,
    }
    out = score_asq_questionnaire(domains)
    assert out["ok"]
    assert out["needs_referral"] is True
    assert "communication" in out["referral_domains"]


def test_mchat_complete_tiers():
    low = {i: ("no" if i in (2, 5, 12) else "yes") for i in range(1, 21)}
    assert score_mchat(low)["risk"] == "low"
    med = {i: "yes" for i in range(1, 21)}
    assert score_mchat(med)["risk"] == "medium"
    high = {i: "no" for i in range(1, 21)}
    assert score_mchat(high)["risk"] == "high"


def test_who_curves_monotonic():
    for sex in ("male", "female"):
        for measure in ("weight", "length", "head_circumference"):
            p3 = who_pct(sex, measure, 6, 3)
            p50 = who_pct(sex, measure, 6, 50)
            p97 = who_pct(sex, measure, 6, 97)
            assert p3 < p50 < p97
