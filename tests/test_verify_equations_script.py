"""Keep verify_equations.py green under CI.

verify_equations.py is a standalone script (self-consistency + published
value->centile anchors for the INTERGROWTH-21st preterm equations). It was not
covered by pytest, so a regression in the equations or the harness would only
surface if someone ran the script by hand. This wrapper runs it in-process and
asserts it reports zero failures, and guards the published external anchors
directly so a coefficient drift cannot pass silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import verify_equations as ve


def test_verify_equations_reports_no_failures(monkeypatch, tmp_path):
    # Keep the optional matplotlib plot out of the repo tree during tests.
    monkeypatch.chdir(tmp_path)
    assert ve.main() == 0


def test_published_anchors_are_actually_checked():
    """The external anchors must exist and be concrete (no None placeholders)."""
    assert ve.PUBLISHED_EXAMPLES, "published anchors were removed"
    for ex in ve.PUBLISHED_EXAMPLES:
        assert "expected_centile" in ex or "expected_z" in ex
        val = ex.get("expected_centile", ex.get("expected_z"))
        assert val is not None, f"anchor {ex.get('name')!r} has a None expected value"


def test_published_weight_anchor_reproduces():
    """male 0.99 kg at 27w PMA ~ 96.9th centile (growthstandards docs)."""
    import intergrowth_preterm_equations as ig

    c = ig.centile_from_measurement("male", "weight", 27, 0.99)
    assert abs(c - 96.89486) < 0.05
