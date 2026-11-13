"""Functional end-to-end assistant workflow tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.agent.orchestrator import ParentAssistant
from assistant.memory.chat_memory import ChatMemory
from assistant.memory.child_db import ChildMemoryDB


def test_asq_session_report():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = ChildMemoryDB(Path(td) / "c.db")
        try:
            asst = ParentAssistant(db=db, use_xlam=False, use_pleias=False)
            cid = db.create_child("Sara", "female")
            answers = {
                "communication": ["yes"] * 6,
                "gross_motor": ["not_yet"] * 6,
                "fine_motor": ["sometimes"] * 6,
                "problem_solving": ["yes"] * 6,
                "personal_social": ["yes"] * 6,
            }
            out = asst.run_asq_session(cid, 4, answers)
            assert out["result"]["needs_referral"] is True
            assert "gross_motor" in out["result"]["referral_domains"]
            assert "BELOW cutoff" in out["parent_report"]
        finally:
            asst.chat_memory.close()
            db.close()


def test_mchat_session():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = ChildMemoryDB(Path(td) / "c.db")
        try:
            asst = ParentAssistant(db=db, use_xlam=False, use_pleias=False)
            cid = db.create_child("Omar", "male")
            answers = {i: "yes" for i in range(1, 21)}
            for i in (2, 5, 12):
                answers[i] = "no"
            out = asst.run_mchat_session(cid, answers)
            assert out["result"]["risk"] == "low"
        finally:
            asst.chat_memory.close()
            db.close()


def test_handle_does_not_invent_without_params():
    asst = ParentAssistant(use_xlam=False, use_pleias=False)
    try:
        out = asst.handle("What is the weight percentile?")  # no sex, age or value given
        computed = [
            c for c in out["tools"]["tool_calls"]
            if c.get("name") in {"growth_percentile", "overlay_growth_on_chart"} and (c.get("result") or {}).get("ok")
        ]
        assert not computed, computed
    finally:
        asst.chat_memory.close()
        asst.db.close()


def test_growth_measurement_in_chat_is_computed_by_the_tool():
    """A measurement given in chat is assessed by the growth tool, with the
    child's sex and age taken from the record rather than guessed."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        child_db = ChildMemoryDB(Path(td) / "c.db")
        chat_db = ChatMemory(Path(td) / "chat.db")
        try:
            asst = ParentAssistant(
                db=child_db, chat_memory=chat_db, use_xlam=False, use_pleias=False
            )
            cid = child_db.create_child("Ali", "male", gestational_age_weeks=30)
            sid = asst.start_session(cid)
            out = asst.chat(sid, "his weight at 40 weeks is 3.2 kg, please put it on the chart", child_id=cid)
            calls = [
                c for c in out["tools"]["tool_calls"]
                if c["name"] in {"growth_percentile", "overlay_growth_on_chart"}
            ]
            assert calls, out["tools"]["tool_calls"]
            ok = [c for c in calls if (c.get("result") or {}).get("ok")]
            assert ok, calls
            assert any(float((c.get("result") or {}).get("value") or 0) == 3.2 for c in ok), ok
            assert out.get("reply")
        finally:
            chat_db.close()
            child_db.close()


def test_full_growth_overlay_workflow():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = ChildMemoryDB(Path(td) / "c.db")
        try:
            asst = ParentAssistant(db=db, use_xlam=False, use_pleias=False)
            cid = db.create_child("Ali", "male", gestational_age_weeks=30)
            out = asst.record_growth_and_overlay(cid, "male", "weight", 40, 3.2)
            assert out.get("ok") is not False
            assert out["centile"] is not None
            if out.get("overlay_path"):
                assert Path(out["overlay_path"]).exists()
            hist = db.growth_history(cid, "weight")
            assert len(hist) == 1
            # The recorded point is what the child summary reports back.
            assert "weight" in db.child_context_text(cid).lower()
        finally:
            asst.chat_memory.close()
            db.close()
