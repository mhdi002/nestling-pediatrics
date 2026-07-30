"""Regression tests from live multi-turn chat verification."""

from __future__ import annotations

import tempfile
from pathlib import Path

from assistant.agent.orchestrator import ParentAssistant, extract_growth_slots
from assistant.memory.chat_memory import ChatMemory
from assistant.memory.child_db import ChildMemoryDB
from assistant.rag.stores import MedicalRAG


def test_extract_born_at_sets_ga_not_measurement_weeks():
    s = extract_growth_slots("he was preterm, born at 32 weeks")
    assert s.get("gestational_age_weeks") == 32
    assert s.get("chart_standard") == "intergrowth_preterm"
    assert "weeks" not in s  # must not clobber prior measurement age


def test_extract_sex_not_from_medical_concern():
    s = extract_growth_slots("من پسرم سه ماهشه و حرف نمیزنه مشکل چیه ؟")
    assert "sex" not in s


def test_iron_rag_not_speech_after_chat_context():
    """Reproduce the bug: prior 'developmental/talk' memory must not steal iron retrieval."""
    rag = MedicalRAG()
    assert rag.store.load()
    poisoned = (
        "[RECENT_CHAT]\nASSISTANT: We can talk through growth or developmental worries\n\n"
        "[CURRENT_USER]\ntell me about iron for breastfed babies"
    )
    # Topic detection should still treat this as iron
    out = rag.answer(poisoned, use_pleias=False)
    blob = ((out.get("answer") or "") + " " + " ".join(
        (h.get("text") or "")[:200] for h in (out.get("hits") or [])
    )).lower()
    assert "iron" in blob
    assert "cooing" not in (out.get("answer") or "").lower()


def test_live_memory_and_tools_preterm_child():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = ChildMemoryDB(Path(td) / "c.db")
        chat = ChatMemory(Path(td) / "chat.db")
        asst = ParentAssistant(db=db, chat_memory=chat, use_xlam=False, use_pleias=False)
        try:
            asst.medical.load() or asst.refresh_medical_index()
            cid = db.create_child("Aria", "male", gestational_age_weeks=32)
            sid = asst.start_session(cid)
            asst.chat(sid, "Our baby is a boy", child_id=cid)
            t = asst.chat(
                sid,
                "weight at 40 weeks value 3.2 kg please show the chart",
                child_id=cid,
            )
            tools = (t.get("tools") or {}).get("tool_calls") or []
            ok = next(
                (
                    x
                    for x in tools
                    if x.get("name") in {"overlay_growth_on_chart", "growth_percentile"}
                    and (x.get("result") or {}).get("ok")
                ),
                None,
            )
            assert ok is not None
            assert abs(float(ok["result"]["centile"]) - 30.85) < 1.0
            assert chat.list_facts(sid).get("sex", {}).get("value") == "male"

            t2 = asst.chat(sid, "is he on track?", child_id=cid)
            assert "growth_analysis" in (t2.get("intents") or [])
            assert t2["slots"].get("last_centile") is not None

            t3 = asst.chat(sid, "tell me about iron for breastfed babies", child_id=cid)
            assert "medical" in (t3.get("intents") or [])
            assert "iron" in (t3.get("reply") or "").lower()
            assert "cooing" not in (t3.get("reply") or "").lower()
        finally:
            chat.close()
            db.close()


def test_weeks_without_ga_asks_then_clarifies():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = ChildMemoryDB(Path(td) / "c.db")
        chat = ChatMemory(Path(td) / "chat.db")
        asst = ParentAssistant(db=db, chat_memory=chat, use_xlam=False, use_pleias=False)
        try:
            sid = asst.start_session()
            t1 = asst.chat(sid, "boy weight 40 weeks 3.2 kg overlay")
            assert t1.get("needs_gestational_age") or "gestational" in (t1.get("reply") or "").lower()
            tools_ok = [
                x
                for x in ((t1.get("tools") or {}).get("tool_calls") or [])
                if (x.get("result") or {}).get("ok")
                and x.get("name") in {"overlay_growth_on_chart", "growth_percentile"}
            ]
            assert tools_ok == []

            asst.chat(sid, "he was preterm, born at 32 weeks")
            t3 = asst.chat(sid, "show the chart for weight 40 weeks 3.2 kg")
            ok = next(
                (
                    x
                    for x in ((t3.get("tools") or {}).get("tool_calls") or [])
                    if (x.get("result") or {}).get("ok")
                    and x.get("name") in {"overlay_growth_on_chart", "growth_percentile"}
                ),
                None,
            )
            assert ok is not None
            assert abs(float(ok["result"]["centile"]) - 30.85) < 1.0
        finally:
            chat.close()
            db.close()
