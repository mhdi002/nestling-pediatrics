"""Chat memory + multi-turn slot filling tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.agent.orchestrator import ParentAssistant, extract_growth_slots
from assistant.memory.chat_memory import ChatMemory
from assistant.memory.child_db import ChildMemoryDB


def test_chat_memory_persists_turns():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        mem = ChatMemory(Path(td) / "chat.db")
        try:
            sid = mem.create_session()
            mem.add_message(sid, "user", "hello")
            mem.add_message(sid, "assistant", "hi parent")
            hist = mem.get_history(sid)
            assert len(hist) == 2
            assert hist[0]["role"] == "user"
            assert hist[1]["content"] == "hi parent"
        finally:
            mem.close()


def test_multi_turn_growth_slots():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = ChildMemoryDB(Path(td) / "c.db")
        chat = ChatMemory(Path(td) / "chat.db")
        try:
            asst = ParentAssistant(db=db, chat_memory=chat, use_xlam=False, use_pleias=False)
            cid = db.create_child("Ali", "male", gestational_age_weeks=32)
            sid = asst.start_session(cid)
            r1 = asst.chat(sid, "Our baby is a boy", child_id=cid)
            assert r1["slots"].get("sex") == "male"
            assert isinstance(r1.get("missing_slots"), list)
            r2 = asst.chat(sid, "weight at 40 weeks value: 3.2 kg please overlay chart", child_id=cid)
            assert r2["slots"]["sex"] == "male"
            assert r2["slots"]["measure"] == "weight"
            assert r2["slots"]["weeks"] == 40
            assert r2["slots"]["value"] == 3.2
            names = [t["name"] for t in r2["tools"]["tool_calls"]]
            assert any(n in {"growth_percentile", "overlay_growth_on_chart"} for n in names)
            assert len(r2["history"]) >= 4
        finally:
            chat.close()
            db.close()


def test_only_allowed_models_advertised():
    asst = ParentAssistant(use_xlam=False, use_pleias=False)
    sid = asst.start_session()
    out = asst.chat(sid, "hello")
    assert out["models"]["tool_calling"] == "Salesforce/xLAM-1b-fc-r"
    assert out["models"]["rag"] == "PleIAs/Pleias-RAG-1B"
    asst.chat_memory.close()
    asst.db.close()


def test_extract_slots():
    s = extract_growth_slots("girl length 50 weeks 55 cm")
    assert s["sex"] == "female"
    assert s["measure"] == "length"
    assert s["weeks"] == 50
    assert s["value"] == 55
