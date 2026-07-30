"""Functional end-to-end assistant workflow tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.agent.orchestrator import ParentAssistant, parse_tool_calls, rule_based_tool_calls
from assistant.memory.chat_memory import ChatMemory
from assistant.memory.child_db import ChildMemoryDB


def test_parse_tool_calls_json():
    text = '{"tool_calls": [{"name": "growth_percentile", "arguments": {"sex": "male", "measure": "weight", "weeks": 40}}]}'
    calls = parse_tool_calls(text)
    assert calls[0]["name"] == "growth_percentile"


def test_rule_based_growth_router():
    calls = rule_based_tool_calls("Compute growth percentile for male weight at 40 weeks value: 3.5 kg")
    assert calls and calls[0]["name"] == "overlay_growth_on_chart"
    assert calls[0]["arguments"]["value"] == 3.5


def test_full_growth_overlay_workflow():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = ChildMemoryDB(Path(td) / "c.db")
        try:
            asst = ParentAssistant(db=db, use_xlam=False, use_pleias=False)
            asst.medical.load() or asst.refresh_medical_index()
            cid = db.create_child("Ali", "male", gestational_age_weeks=30)
            out = asst.record_growth_and_overlay(cid, "male", "weight", 40, 3.2)
            assert out.get("ok") is not False
            assert out["centile"] is not None
            if out.get("overlay_path"):
                assert Path(out["overlay_path"]).exists()
            hist = db.growth_history(cid, "weight")
            assert len(hist) == 1
            mem = asst.ask_child(cid, "latest weight")
            assert "weight" in mem["answer"].lower()
        finally:
            asst.chat_memory.close()
            db.close()


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
        out = asst.handle("What is the weight percentile?")  # missing sex/weeks/value
        assert out["tools"]["tool_calls"] == []
    finally:
        asst.chat_memory.close()
        asst.db.close()


def test_multi_turn_growth_chat_boy_then_overlay():
    """Parent says sex first, then measure/weeks/value/overlay — agent uses prior context."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        child_db = ChildMemoryDB(Path(td) / "c.db")
        chat_db = ChatMemory(Path(td) / "chat.db")
        try:
            asst = ParentAssistant(
                db=child_db, chat_memory=chat_db, use_xlam=False, use_pleias=False
            )
            sid = asst.start_session()

            turn1 = asst.chat(sid, "boy")
            assert turn1["tools"]["tool_calls"] == []
            assert turn1["slots"].get("sex") == "male"

            turn2 = asst.chat(sid, "preterm weight 40 weeks 3.2 kg overlay")
            calls = turn2["tools"]["tool_calls"]
            assert len(calls) >= 1
            overlay = next(c for c in calls if c["name"] == "overlay_growth_on_chart")
            args = overlay["arguments"]
            assert args["sex"] == "male"
            assert args["measure"] == "weight"
            assert args["weeks"] == 40
            assert args["value"] == 3.2
            assert overlay["result"].get("ok") is True
            assert overlay["result"].get("centile") is not None
            assert "reply" in turn2
        finally:
            chat_db.close()
            child_db.close()
