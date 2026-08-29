"""
The analysis must describe the measurement that was actually plotted.

A live session showed a point at PMA ~32.5 weeks — visibly above the 97th
centile on the INTERGROWTH chart — while the follow-up analysis said
"about the 43th centile ... in the usual range (10th-90th)". Both cannot
describe the same dot.

The cause was `latest_growth_snapshot` mixing sources: `centile` and
`age_months` came from the last plotted result (`last_*`), but `weeks` and
`value` came from the live session slots, which can belong to a different
measurement entirely.
"""

from __future__ import annotations

import pytest

from assistant.agent.orchestrator import latest_growth_snapshot


class _NoDB:
    def growth_history(self, child_id):  # pragma: no cover - not reached
        raise AssertionError("session slots should satisfy the snapshot")


def test_snapshot_never_mixes_measurements():
    """A stale `weeks` slot must not attach itself to a fresh centile."""
    slots = {
        # The plotted result:
        "last_measure": "weight",
        "last_value": 3.2,
        "last_centile": 42.78,
        "last_track_status": "within_10_90",
        "last_age_months": 2.0,
        "last_weeks": 40.7,
        "last_chart_standard": "intergrowth_preterm",
        # Stale live slots from an earlier turn about a different age:
        "weeks": 32.57,
        "value": 9.9,
        "measure": "length",
    }
    snap = latest_growth_snapshot(_NoDB(), None, slots)
    assert snap["weeks"] == 40.7, "took the stale live `weeks` slot"
    assert snap["value"] == 3.2, "took the stale live `value` slot"
    assert snap["measure"] == "weight", "took the stale live `measure` slot"
    assert snap["centile"] == pytest.approx(42.78)


def test_snapshot_age_and_centile_are_consistent():
    """
    The reported age must reproduce the reported centile on the same chart.
    This is the invariant that failed: a 3.2 kg point is ~43rd centile at
    40.7w PMA but above the 97th at 32.5w, so age and centile must travel
    together or the narrative contradicts the picture.
    """
    from assistant.tools.clinical import dispatch_tool

    plotted = dispatch_tool(
        "overlay_growth_on_chart",
        {"sex": "female", "measure": "weight", "value": 3.2,
         "weeks": 40.7, "gestational_age_weeks": 32.0},
    )
    assert plotted.get("ok"), plotted

    slots = {
        "last_measure": plotted["measure"],
        "last_value": plotted["value"],
        "last_centile": plotted["centile"],
        "last_track_status": plotted["track_status"],
        "last_age_months": plotted["age_months"],
        "last_weeks": plotted["weeks"],
        "last_chart_standard": plotted["chart_standard"],
        "gestational_age_weeks": 32.0,
        "sex": "female",
    }
    snap = latest_growth_snapshot(_NoDB(), None, slots)

    # Recompute from the snapshot's own age; it must land on its own centile.
    recomputed = dispatch_tool(
        "overlay_growth_on_chart",
        {"sex": "female", "measure": snap["measure"], "value": snap["value"],
         "weeks": snap["weeks"], "gestational_age_weeks": 32.0},
    )
    assert abs(recomputed["centile"] - snap["centile"]) < 1.0, (snap, recomputed)


def test_above_97th_is_not_described_as_usual_range():
    """A point genuinely above the 97th must not be narrated as typical."""
    from assistant.parent_voice import growth_plot_chat
    from assistant.tools.clinical import dispatch_tool

    high = dispatch_tool(
        "overlay_growth_on_chart",
        {"sex": "female", "measure": "weight", "value": 3.2,
         "age_months": 0.1, "gestational_age_weeks": 32.0},
    )
    assert high.get("ok"), high
    assert high["centile"] > 97, high["centile"]

    for fa in (False, True):
        text = growth_plot_chat(high, fa=fa)
        lowered = text.lower()
        assert "usual peer range" not in lowered
        assert "بازه معمول همسالان" not in text
