"""Intent routing: help / medical / growth / history isolation."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.agent.orchestrator import (
    ParentAssistant,
    classify_intent,
    extract_growth_slots,
    rule_based_tool_calls,
)
from assistant.memory.chat_memory import ChatMemory
from assistant.memory.child_db import ChildMemoryDB


def test_rule_based_growth_still_routes_compute_phrase():
    """Regression: growth intent required, but classic compute phrase still works."""
    msg = "Compute growth percentile for male weight at 40 weeks value: 3.5 kg"
    assert "growth" in classify_intent(msg)
    calls = rule_based_tool_calls(msg)
    assert calls and calls[0]["name"] == "overlay_growth_on_chart"
    assert calls[0]["arguments"]["value"] == 3.5
    assert calls[0]["arguments"]["sex"] == "male"
    assert calls[0]["arguments"]["weeks"] == 40.0


def test_help_greeting_no_medical_rag_mentions_nestling():
    asst = ParentAssistant(use_xlam=False, use_pleias=False)
    try:
        sid = asst.start_session()
        out = asst.chat(sid, "hi, how can you help me?")
        assert "help" in out["intents"]
        assert "medical" not in out["intents"]
        assert "medical_rag" not in out
        assert "Nestling" in (out.get("reply") or "")
        assert out["tools"]["tool_calls"] == []
    finally:
        asst.close()


def test_multi_turn_growth_then_history_no_overlay_refire():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        child_db = ChildMemoryDB(Path(td) / "c.db")
        chat_db = ChatMemory(Path(td) / "chat.db")
        try:
            asst = ParentAssistant(
                db=child_db, chat_memory=chat_db, use_xlam=False, use_pleias=False
            )
            cid = child_db.create_child("Ali", "male", gestational_age_weeks=32)
            sid = asst.start_session(cid)

            t1 = asst.chat(sid, "boy", child_id=cid)
            assert t1["tools"]["tool_calls"] == []
            assert t1["slots"].get("sex") == "male"

            t2 = asst.chat(sid, "weight 40 weeks 3.2 kg overlay", child_id=cid)
            names2 = [c["name"] for c in t2["tools"]["tool_calls"]]
            assert "overlay_growth_on_chart" in names2
            assert "growth" in t2["intents"]

            t3 = asst.chat(sid, "what was my child's last growth result?", child_id=cid)
            assert "history" in t3["intents"]
            assert "growth" not in t3["intents"]
            names3 = [c["name"] for c in t3["tools"]["tool_calls"]]
            assert "overlay_growth_on_chart" not in names3
            assert "growth_percentile" not in names3
        finally:
            asst.close()


def test_tell_me_about_iron_is_medical():
    assert "medical" in classify_intent("tell me about iron")
    asst = ParentAssistant(use_xlam=False, use_pleias=False)
    try:
        sid = asst.start_session()
        out = asst.chat(sid, "tell me about iron")
        assert "medical" in out["intents"]
        assert "medical_rag" in out
        assert out["tools"]["tool_calls"] == []
    finally:
        asst.close()


def test_speech_concern_not_help_dump():
    assert "medical" in classify_intent("my child cant talk")
    assert "help" not in classify_intent("my child cant talk")
    assert "medical" in classify_intent("من پسرم سه ماهشه و حرف نمیزنه مشکل چیه ؟")
    asst = ParentAssistant(use_xlam=False, use_pleias=False)
    try:
        sid = asst.start_session()
        out = asst.chat(sid, "my child cant talk", ui_lang="en")
        assert "medical" in out["intents"]
        assert "help" not in out["intents"]
        reply = out.get("reply") or ""
        assert "speech" in reply.lower() or "3-month" in reply.lower() or "words" in reply.lower() or "talk" in reply.lower()
        assert "I can:\n" not in reply  # not the capability dump
    finally:
        asst.close()


def test_show_chart_asks_for_details_not_vague_chat():
    assert "growth" in classify_intent("show the chart")
    assert extract_growth_slots("age 32").get("weeks") == 32.0
    asst = ParentAssistant(use_xlam=False, use_pleias=False)
    try:
        sid = asst.start_session()
        t1 = asst.chat(sid, "show the chart", ui_lang="en")
        assert "growth" in t1["intents"]
        assert "I hear you" not in (t1.get("reply") or "")
        assert "measure" in (t1.get("reply") or "").lower() or t1.get("missing_slots")

        t2 = asst.chat(sid, "age 32 and by the mesure you mean what ?", ui_lang="en")
        assert "growth" in t2["intents"]
        assert t2["slots"].get("weeks") == 32.0
        assert "weight" in (t2.get("reply") or "").lower()
        assert "I hear you" not in (t2.get("reply") or "")

        t3 = asst.chat(sid, "boy weight 3.2 kg", ui_lang="en")
        assert "growth" in t3["intents"]
        names = [c["name"] for c in t3["tools"]["tool_calls"]]
        assert "overlay_growth_on_chart" in names or "growth_percentile" in names
    finally:
        asst.close()


def test_analyze_and_on_track_after_chart():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        child_db = ChildMemoryDB(Path(td) / "c.db")
        chat_db = ChatMemory(Path(td) / "chat.db")
        try:
            asst = ParentAssistant(
                db=child_db, chat_memory=chat_db, use_xlam=False, use_pleias=False
            )
            cid = child_db.create_child("TermBaby", "male", gestational_age_weeks=39)
            child_db.add_growth(
                cid,
                weeks=1.0,
                measure="weight",
                value=3.2,
                centile=20.2,
                z_score=-0.83,
                track_status="within_10_90",
            )
            sid = asst.start_session(cid)
            chart = asst.chat(sid, "show my child chart", child_id=cid, ui_lang="en")
            assert "growth" in chart["intents"]

            a1 = asst.chat(sid, "can you analyze that ?", child_id=cid, ui_lang="en")
            assert "growth_analysis" in a1["intents"]
            assert "I hear you" not in (a1.get("reply") or "")
            assert "usual range" in (a1.get("reply") or "").lower() or "typical" in (
                a1.get("reply") or ""
            ).lower()
            assert "overlay_growth_on_chart" not in [
                c["name"] for c in a1["tools"]["tool_calls"]
            ]

            a2 = asst.chat(sid, "is my baby in a good track ?", child_id=cid, ui_lang="en")
            assert "growth_analysis" in a2["intents"]
            assert "I hear you" not in (a2.get("reply") or "")
            assert "typical" in (a2.get("reply") or "").lower() or "usual" in (
                a2.get("reply") or ""
            ).lower()
        finally:
            asst.close()


def test_speech_does_not_reuse_chart_tools():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        child_db = ChildMemoryDB(Path(td) / "c.db")
        chat_db = ChatMemory(Path(td) / "chat.db")
        try:
            asst = ParentAssistant(
                db=child_db, chat_memory=chat_db, use_xlam=False, use_pleias=False
            )
            cid = child_db.create_child("TermBaby", "male", gestational_age_weeks=39)
            child_db.add_growth(cid, weeks=40.0, measure="weight", value=3.2)
            sid = asst.start_session(cid)
            asst.chat(sid, "show my child chart", child_id=cid, ui_lang="en")
            out = asst.chat(sid, "پسرم کی حرف میزنه؟", child_id=cid, ui_lang="fa")
            assert "medical" in out["intents"]
            assert "growth" not in out["intents"]
            names = [c["name"] for c in out["tools"]["tool_calls"]]
            assert "overlay_growth_on_chart" not in names
            assert "I plotted" not in (out.get("reply") or "")
            assert "رسم کردم" not in (out.get("reply") or "")
        finally:
            asst.close()


def test_show_my_child_chart_replots_saved_growth():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        child_db = ChildMemoryDB(Path(td) / "c.db")
        chat_db = ChatMemory(Path(td) / "chat.db")
        try:
            asst = ParentAssistant(
                db=child_db, chat_memory=chat_db, use_xlam=False, use_pleias=False
            )
            cid = child_db.create_child("TermBaby", "male", gestational_age_weeks=39)
            # Save a measurement the parent already entered
            child_db.add_growth(
                cid, weeks=40.0, measure="weight", value=3.2, centile=0.0, z_score=-8.0
            )
            sid = asst.start_session(cid)
            out = asst.chat(sid, "show my child chart", child_id=cid, ui_lang="en")
            assert "growth" in out["intents"]
            assert "history" not in out["intents"]
            names = [c["name"] for c in out["tools"]["tool_calls"]]
            assert "overlay_growth_on_chart" in names
            res = out["tools"]["tool_calls"][0]["result"]
            assert res.get("ok")
            assert res.get("age_months") is not None and res["age_months"] < 1.0
            assert res.get("centile", 0) > 10
            assert "By measure I mean" not in (out.get("reply") or "")
            assert len(child_db.growth_history(cid)) == 1

            show = asst.chat(sid, "show", child_id=cid, ui_lang="en")
            assert "growth" in show["intents"]
            assert "I hear you" not in (show.get("reply") or "")

            ok = asst.chat(sid, "so its okey now", ui_lang="en")
            assert "reassure" in ok["intents"]
        finally:
            asst.close()
