"""ASQ extraction structure tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXTRACTED = ROOT / "extracted" / "asq"
REQUIRED = {
    "communication",
    "gross_motor",
    "fine_motor",
    "problem_solving",
    "personal_social",
    "overall",
}


def test_all_asq_have_six_domains():
    files = sorted(EXTRACTED.glob("*.json"))
    assert files, "no ASQ json extracted"
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        ids = {d["id"] for d in data["domains"]}
        assert REQUIRED.issubset(ids), f"{path.name} missing domains: {REQUIRED - ids}"
        total_q = sum(len(d["questions"]) for d in data["domains"])
        assert total_q >= 30, f"{path.name} only {total_q} questions"


def test_questions_have_options():
    data = json.loads((EXTRACTED / "4m.json").read_text(encoding="utf-8"))
    for dom in data["domains"]:
        for q in dom["questions"]:
            assert q["text"].strip()
            assert len(q["options"]) >= 2
