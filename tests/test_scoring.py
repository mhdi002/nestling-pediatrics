"""Scoring accuracy tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.tools.clinical import score_asq_domain, score_asq_questionnaire, score_mchat


def test_asq_points():
    r = score_asq_domain(["yes", "sometimes", "not_yet", "بله", "گاهی", "هنوز نه"])
    assert r["item_scores"] == [10, 5, 0, 10, 5, 0]
    assert r["total"] == 30


def test_asq_referral_flag():
    low = ["not_yet"] * 6
    high = ["yes"] * 6
    out = score_asq_questionnaire({"communication": low, "gross_motor": high})
    assert out["ok"] is True
    assert out["needs_referral"] is True
    assert "communication" in out["referral_domains"]
    assert "gross_motor" not in out["referral_domains"]


def test_mchat_risk_tiers():
    # All yes on non-reverse items = mostly pass; reverse items yes = fail
    answers = {i: "yes" for i in range(1, 21)}
    # reverse 2,5,12 fail when yes
    r = score_mchat(answers)
    assert r["ok"] is True
    assert r["total_failed"] == 3
    assert r["risk"] == "medium"

    answers2 = {i: "no" for i in range(1, 21)}
    # non-reverse fail on no → 17 fails + reverse items pass on no → 17 fails = high
    r2 = score_mchat(answers2)
    assert r2["total_failed"] == 17
    assert r2["risk"] == "high"

    answers3 = {i: "yes" for i in range(1, 21)}
    for i in (2, 5, 12):
        answers3[i] = "no"
    r3 = score_mchat(answers3)
    assert r3["total_failed"] == 0
    assert r3["risk"] == "low"
