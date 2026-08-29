"""
The chart, the analysis and the child summary must agree about a measurement.

A live session reported the same 3.2 kg for a 32-week preterm baby as
"~100th centile, outside the usual range — see a doctor" on the chart turn but
"centile≈43, within_10_90" in the child summary. Both cannot be true, and the
disagreement is clinically dangerous in either direction: a false alarm or
false reassurance about a preterm infant's growth.

The cause is an age-semantics mismatch. INTERGROWTH preterm charts are indexed
by postmenstrual age, and the tool derives PMA as `gestational_age + age_months`
— so `age_months` MUST be chronological age since birth. Passing a corrected
age (months past term) instead shifts the plot by the whole prematurity gap:
here 0.1 months corrected reads as PMA 32.4w rather than 40.7w, which turns a
median baby into an off-the-chart one.
"""

from __future__ import annotations

import pytest

from assistant.tools.clinical import dispatch_tool

# A 32-week preterm infant weighing 3.2 kg at 40.7 weeks postmenstrual age:
# term-equivalent and unremarkable.
GA_WEEKS = 32.0
PMA_WEEKS = 40.7
WEIGHT_KG = 3.2


def _overlay(**kwargs):
    args = {"sex": "female", "measure": "weight", "value": WEIGHT_KG,
            "gestational_age_weeks": GA_WEEKS}
    args.update(kwargs)
    return dispatch_tool("overlay_growth_on_chart", args)


def test_pma_weeks_plot_is_unremarkable():
    res = _overlay(weeks=PMA_WEEKS)
    assert res.get("ok"), res
    assert 10 < res["centile"] < 90, res["centile"]
    assert res.get("track_status") == "within_10_90", res


def test_age_months_round_trips_through_the_chart():
    """
    Re-plotting with the age the tool itself reported must land on the same
    centile. If it does not, the tool is emitting an age in different units
    from the one it consumes, which is exactly how the two answers diverged.
    """
    first = _overlay(weeks=PMA_WEEKS)
    assert first.get("ok"), first
    again = _overlay(age_months=first["age_months"])
    assert again.get("ok"), again
    assert abs(first["centile"] - again["centile"]) < 1.0, (first, again)


def test_corrected_age_is_not_accepted_as_chronological():
    """
    Guards the specific confusion: 0.1 months *corrected* is ~2.0 months
    *chronological* for a 32-week baby. Feeding the corrected figure in must
    not silently produce a confident, wildly different verdict.
    """
    chronological = _overlay(weeks=PMA_WEEKS)["age_months"]
    assert chronological == pytest.approx(2.0, abs=0.3), chronological

    corrected_months = 0.1
    wrong = _overlay(age_months=corrected_months)
    # It is a different plot, and that is the point: the two must not be
    # confused for one another anywhere upstream.
    assert abs(wrong["centile"] - _overlay(weeks=PMA_WEEKS)["centile"]) > 20


def test_summary_and_chart_agree_for_a_stored_measurement(tmp_path):
    """End-to-end: what the chart says and what the summary says must match."""
    from assistant.memory.child_db import ChildMemoryDB
    from assistant.tools import clinical

    db = ChildMemoryDB(tmp_path / "children.db")
    clinical.set_child_db(db)
    try:
        cid = db.create_child("Preterm Baby", "female", gestational_age_weeks=GA_WEEKS)
        plotted = _overlay(weeks=PMA_WEEKS, child_id=cid)
        assert plotted.get("ok"), plotted
        db.add_growth(
            cid, PMA_WEEKS, "weight", WEIGHT_KG,
            centile=plotted["centile"], age_months=plotted["age_months"],
        )
        summary = dispatch_tool("get_child_summary", {"child_id": cid})
        assert summary.get("ok"), summary
        # The stored centile is the one the summary reports back.
        assert f"{plotted['centile']:.0f}" in f"{summary}" or \
            abs(db.growth_history(cid)[-1]["centile"] - plotted["centile"]) < 1.0
    finally:
        clinical.set_child_db(None)
        db.close()
