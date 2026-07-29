"""Accuracy tests for INTERGROWTH equation tools."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import intergrowth_preterm_equations as ig
from assistant.tools.clinical import dispatch_tool, growth_percentile


def test_published_examples():
    c = ig.centile_from_measurement("male", "weight", 27, 0.99)
    assert abs(c - 96.89486) < 0.05
    c = ig.centile_from_measurement("female", "weight", 27, 0.91)
    assert abs(c - 97.12034) < 0.05
    z = ig.z_score("female", "length", 64, 64.68)
    assert abs(z) < 0.05


def test_tool_matches_module():
    out = growth_percentile("male", "weight", 40, value=3.5)
    assert out["tool"] == "growth_percentile"
    assert out["ok"] is True
    assert abs(out["centile"] - ig.centile_from_measurement("male", "weight", 40, 3.5)) < 1e-9
    assert abs(out["z_score"] - ig.z_score("male", "weight", 40, 3.5)) < 1e-9


def test_dispatch_unknown_raises():
    out = dispatch_tool("not_a_tool", {})
    assert out["ok"] is False
    assert out["error"] == "unknown_tool" or out.get("detail", "").startswith("Unknown tool")


def test_percentiles_monotonic():
    vals = [ig.percentile("male", "weight", 40, p) for p in (3, 10, 50, 90, 97)]
    assert vals == sorted(vals)


def test_no_nan():
    for sex in ("male", "female"):
        for meas in ("weight", "length", "head_circumference"):
            for w in (27, 40, 64):
                y = ig.percentile(sex, meas, w, 50)
                assert math.isfinite(y) and y > 0
