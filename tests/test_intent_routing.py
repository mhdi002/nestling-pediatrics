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


def test_feeding_followup_after_growth_analysis_routes_medical_not_fallback():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        child_db = ChildMemoryDB(Path(td) / "c.db")
        chat_db = ChatMemory(Path(td) / "chat.db")
        try:
            asst = ParentAssistant(
                db=child_db, chat_memory=chat_db, use_xlam=False, use_pleias=False
            )
            cid = child_db.create_child("TermBaby", "female", gestational_age_weeks=39)
            child_db.add_growth(
                cid,
                weeks=40.0,
                measure="weight",
                value=3.2,
                centile=35.0,
                z_score=-0.4,
                track_status="within_10_90",
            )
            sid = asst.start_session(cid)
            t1 = asst.chat(sid, "show my child chart", child_id=cid, ui_lang="en")
            assert "growth" in t1["intents"]
            t2 = asst.chat(sid, "is my child growth okey?", child_id=cid, ui_lang="en")
            assert "growth_analysis" in t2["intents"]
            t3 = asst.chat(
                sid,
                "tell about the foods that should be eaten in its year",
                child_id=cid,
                ui_lang="en",
            )
            assert "medical" in t3["intents"]
            assert "medical_rag" in t3
            assert "chat" not in t3["intents"]
            reply = (t3.get("reply") or "").lower()
            assert "i'm here and listening" not in reply
            assert "feeding" in reply or "food" in reply or "nutrition" in reply
        finally:
            asst.close()


def test_persian_feeding_routes_medical_and_returns_feeding_not_centiles():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        child_db = ChildMemoryDB(Path(td) / "c.db")
        chat_db = ChatMemory(Path(td) / "chat.db")
        try:
            asst = ParentAssistant(
                db=child_db, chat_memory=chat_db, use_xlam=False, use_pleias=False
            )
            cid = child_db.create_child("Sara", "female", gestational_age_weeks=39)
            child_db.add_growth(
                cid,
                weeks=8.0,
                measure="weight",
                value=4.5,
                centile=40.0,
                z_score=-0.2,
                track_status="within_10_90",
            )
            sid = asst.start_session(cid)
            out = asst.chat(sid, "بچم غذا باید چی بخوره ؟", child_id=cid, ui_lang="fa")
            assert "medical" in out["intents"]
            assert "growth" not in out["intents"]
            assert "medical_rag" in out
            cites = [c.get("id") or "" for c in (out["medical_rag"].get("citations") or [])]
            assert any("feeding" in c for c in cites), cites
            assert not any("centile" in c for c in cites), cites
            reply = (out.get("reply") or "").lower()
            assert "centile" not in reply
            assert out.get("reply_lang") == "fa"
            # No growth chart tools on a feeding turn
            names = [tc.get("name") for tc in (out.get("tool_results") or [])]
            assert "overlay_growth_on_chart" not in names
        finally:
            asst.close()


def test_persian_growth_and_feeding_reply_in_persian_even_if_ui_english():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        child_db = ChildMemoryDB(Path(td) / "c.db")
        chat_db = ChatMemory(Path(td) / "chat.db")
        try:
            asst = ParentAssistant(
                db=child_db, chat_memory=chat_db, use_xlam=False, use_pleias=False
            )
            sid = asst.start_session()
            g = asst.chat(sid, "رشد بچم چطوره ؟", ui_lang="en")
            assert g.get("reply_lang") == "fa"
            assert any("\u0600" <= ch <= "\u06FF" for ch in (g.get("reply") or ""))
            f = asst.chat(sid, "چی بهش بدم بخوره که رشدش بهتر بشه ؟", ui_lang="en")
            assert "medical" in f["intents"]
            assert f.get("reply_lang") == "fa"
            assert any("\u0600" <= ch <= "\u06FF" for ch in (f.get("reply") or ""))
        finally:
            asst.close()


def test_scar_after_growth_is_medical_not_chat_fallback():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        child_db = ChildMemoryDB(Path(td) / "c.db")
        chat_db = ChatMemory(Path(td) / "chat.db")
        try:
            asst = ParentAssistant(
                db=child_db, chat_memory=chat_db, use_xlam=False, use_pleias=False
            )
            cid = child_db.create_child("Lina", "female", gestational_age_weeks=39)
            child_db.add_growth(
                cid,
                weeks=8.0,
                measure="weight",
                value=4.5,
                centile=40.0,
                z_score=-0.2,
                track_status="within_10_90",
            )
            sid = asst.start_session(cid)
            asst.chat(sid, "show my child chart", child_id=cid, ui_lang="en")
            out = asst.chat(
                sid,
                "she has a small scar in her hand what should i do ?",
                child_id=cid,
                ui_lang="en",
            )
            assert "medical" in out["intents"]
            assert "chat" not in out["intents"]
            assert "growth_analysis" not in out["intents"]
            assert "medical_rag" in out
            reply = (out.get("reply") or "").lower()
            assert "analyze that" not in reply
            assert "i'm here and listening" not in reply
            assert any(
                k in reply
                for k in ("wound", "scar", "cut", "rinse", "bandage", "clean", "redness", "clinician")
            ), reply[:400]
        finally:
            asst.close()


def test_speech_followup_dada_stays_medical_not_listening_menu():
    """After a speech/medical turn, 'yes she says dada…' must continue care — not open_chat."""
    assert "medical" in classify_intent("yes she says dada is that good so ?")
    prior = {
        "last_topic": "talk",
        "last_intents": ["medical", "screening"],
        "last_medical_query": "she cant talk well is that okey?",
    }
    assert "medical" in classify_intent(
        "yes she says dada is that good so ?", prior_slots=prior
    )
    assert "medical" in classify_intent("is that good", prior_slots=prior)
    # Bare reassurance after growth must not become medical
    assert "medical" not in classify_intent(
        "is that good",
        prior_slots={"last_topic": "growth_analysis", "last_intents": ["growth_analysis"]},
    )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        child_db = ChildMemoryDB(Path(td) / "c.db")
        chat_db = ChatMemory(Path(td) / "chat.db")
        try:
            asst = ParentAssistant(
                db=child_db, chat_memory=chat_db, use_xlam=False, use_pleias=False
            )
            cid = child_db.create_child("Maya", "female", gestational_age_weeks=39)
            # ~13.5 months chronological stored as weeks since birth
            child_db.add_growth(
                cid,
                weeks=58.7,
                measure="weight",
                value=9.5,
                centile=45.0,
                z_score=-0.1,
                track_status="within_10_90",
            )
            sid = asst.start_session(cid)
            asst.chat(sid, "show my child chart", child_id=cid, ui_lang="en")
            speech = asst.chat(
                sid, "she cant talk well is that okey?", child_id=cid, ui_lang="en"
            )
            assert "medical" in speech["intents"]
            assert "medical_rag" in speech
            speech_q = (speech["slots"].get("last_medical_query") or "").lower()
            assert "talk" in speech_q
            assert speech["slots"].get("last_topic") not in {
                "growth",
                "growth_analysis",
                "help",
                "chat",
                "",
            }

            follow = asst.chat(
                sid,
                "yes she says dada is that good so ?",
                child_id=cid,
                ui_lang="en",
            )
            assert "medical" in follow["intents"]
            assert "chat" not in follow["intents"]
            assert "medical_rag" in follow
            assert follow["slots"].get("last_medical_query") == speech["slots"].get(
                "last_medical_query"
            )
            reply = (follow.get("reply") or "").lower()
            assert "i'm listening" not in reply
            assert "revisit the measurement" not in reply
            assert any(
                k in reply
                for k in (
                    "dada",
                    "mama",
                    "speech",
                    "talk",
                    "language",
                    "word",
                    "babbl",
                    "milestone",
                    "encourag",
                    "good sign",
                    "normal",
                    "develop",
                )
            ), reply[:500]
        finally:
            asst.close()


def test_topic_switch_and_soft_followup_same_session():
    """Speech follow-up, iron switch, scar switch, then soft follow-up stays on scar."""
    prior_speech = {
        "last_topic": "talk",
        "last_intents": ["medical", "screening"],
        "last_medical_query": "she cant talk well",
    }
    prior_skin = {
        "last_topic": "scar-hand",
        "last_intents": ["medical"],
        "last_medical_query": "she has a small scar",
    }
    assert "medical" in classify_intent(
        "yes she says dada is that good so ?", prior_slots=prior_speech
    )
    assert "medical" in classify_intent("how about her iron?", prior_slots=prior_speech)
    assert "medical" in classify_intent(
        "she has a small scar", prior_slots=prior_speech
    )
    assert "medical" in classify_intent("is that okay?", prior_slots=prior_skin)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        child_db = ChildMemoryDB(Path(td) / "c.db")
        chat_db = ChatMemory(Path(td) / "chat.db")
        try:
            asst = ParentAssistant(
                db=child_db, chat_memory=chat_db, use_xlam=False, use_pleias=False
            )
            cid = child_db.create_child("Maya", "female", gestational_age_weeks=39)
            child_db.add_growth(
                cid,
                weeks=58.7,
                measure="weight",
                value=9.5,
                centile=45.0,
                z_score=-0.1,
                track_status="within_10_90",
            )
            sid = asst.start_session(cid)
            asst.chat(sid, "show my child chart", child_id=cid, ui_lang="en")
            food = asst.chat(sid, "what food does she need", child_id=cid, ui_lang="en")
            assert "medical" in food["intents"]
            food_q = (food["slots"].get("last_medical_query") or "").lower()
            assert "food" in food_q

            speech = asst.chat(
                sid, "she cant talk well is that okey?", child_id=cid, ui_lang="en"
            )
            speech_q = speech["slots"].get("last_medical_query") or ""
            assert "talk" in speech_q.lower()
            assert speech_q != food["slots"].get("last_medical_query")

            # A — same-topic follow-up
            follow = asst.chat(
                sid, "yes she says dada is that good so ?", child_id=cid, ui_lang="en"
            )
            assert "medical" in follow["intents"]
            assert follow["slots"].get("last_medical_query") == speech_q
            fr = (follow.get("reply") or "").lower()
            assert "i'm listening" not in fr
            assert "revisit the measurement" not in fr

            # B — switch to iron
            iron = asst.chat(sid, "how about her iron?", child_id=cid, ui_lang="en")
            assert "medical" in iron["intents"]
            iron_q = (iron["slots"].get("last_medical_query") or "").lower()
            assert "iron" in iron_q
            assert iron_q != speech_q.lower()
            assert "medical_rag" in iron
            ir = (iron.get("reply") or "").lower()
            assert "i'm listening" not in ir
            assert "iron" in ir or "آهن" in ir

            # C — switch to skin/scar
            scar = asst.chat(
                sid, "she has a small scar in her hand", child_id=cid, ui_lang="en"
            )
            assert "medical" in scar["intents"]
            scar_q = scar["slots"].get("last_medical_query") or ""
            assert "scar" in scar_q.lower()
            assert scar_q.lower() != iron_q
            sr = (scar.get("reply") or "").lower()
            assert "i'm listening" not in sr
            assert any(k in sr for k in ("scar", "wound", "cut", "clean", "bandage", "redness"))

            # D — soft follow-up stays on scar query
            ok = asst.chat(sid, "is that okay?", child_id=cid, ui_lang="en")
            assert "medical" in ok["intents"]
            assert ok["slots"].get("last_medical_query") == scar_q
            assert "medical_rag" in ok
            okr = (ok.get("reply") or "").lower()
            assert "i'm listening" not in okr
            assert "revisit the measurement" not in okr
            assert any(
                k in okr for k in ("scar", "wound", "cut", "clean", "bandage", "redness", "skin", "heal")
            ), okr[:500]
        finally:
            asst.close()


def test_dynamic_topics_fever_teething_soft_followups():
    """Arbitrary care topics (not iron/speech/skin) continue and switch dynamically."""
    prior_fever = {
        "last_topic": "fever",
        "last_intents": ["medical"],
        "last_medical_query": "my baby has a fever of 38.5",
    }
    assert "medical" in classify_intent("is that okay?", prior_slots=prior_fever)
    assert "medical" in classify_intent(
        "she seems to have colic at night", prior_slots=prior_fever
    )
    assert "medical" in classify_intent(
        "when is her next vaccine?", prior_slots=prior_fever
    )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        child_db = ChildMemoryDB(Path(td) / "c.db")
        chat_db = ChatMemory(Path(td) / "chat.db")
        try:
            asst = ParentAssistant(
                db=child_db, chat_memory=chat_db, use_xlam=False, use_pleias=False
            )
            cid = child_db.create_child("Maya", "female", gestational_age_weeks=39)
            sid = asst.start_session(cid)

            fever = asst.chat(
                sid, "my baby has a fever of 38.5 what should i do", child_id=cid, ui_lang="en"
            )
            assert "medical" in fever["intents"]
            fever_q = fever["slots"].get("last_medical_query") or ""
            assert "fever" in fever_q.lower()
            fr = (fever.get("reply") or "").lower()
            assert "i'm listening" not in fr

            soft = asst.chat(sid, "is that okay?", child_id=cid, ui_lang="en")
            assert "medical" in soft["intents"]
            assert soft["slots"].get("last_medical_query") == fever_q
            assert "i'm listening" not in (soft.get("reply") or "").lower()

            teeth = asst.chat(
                sid, "she is teething and drooling a lot", child_id=cid, ui_lang="en"
            )
            assert "medical" in teeth["intents"]
            teeth_q = teeth["slots"].get("last_medical_query") or ""
            assert "teething" in teeth_q.lower()
            assert teeth_q != fever_q
            assert "i'm listening" not in (teeth.get("reply") or "").lower()

            soft2 = asst.chat(sid, "yes how long?", child_id=cid, ui_lang="en")
            assert "medical" in soft2["intents"]
            assert soft2["slots"].get("last_medical_query") == teeth_q
            assert "i'm listening" not in (soft2.get("reply") or "").lower()
            # Age/sex facts must survive topic switches (child profile still bound).
            assert soft2.get("child_id") == cid
        finally:
            asst.close()


_FEEDING_BLEED = (
    "16–24",
    "16-24",
    "480",
    "finger food",
    "family meal",
    "toddler portion",
    "iron-rich",
    "whole milk",
    "oz (",
    "ounces",
)


def test_hard_topic_switch_after_feeding_walk_and_scar():
    """After feeding, walk and scar must replace the medical thread — no feeding bleed."""
    from assistant.agent.intents import (
        _is_soft_followup,
        care_topic_family,
        classify_intent,
        MEDICAL_FOLLOWUP_RE,
    )

    prior_feed = {
        "last_topic": "foods-good",
        "last_intents": ["medical"],
        "last_medical_query": "what foods are good for her ?",
    }
    walk_msg = "she cant walk is that okey?"
    scar_msg = "she has a little scar on her hand"

    assert care_topic_family(prior_feed["last_medical_query"]) == "feeding"
    assert care_topic_family(walk_msg) == "motor"
    assert care_topic_family(scar_msg) == "skin"
    assert not _is_soft_followup(
        walk_msg,
        followup_hit=bool(MEDICAL_FOLLOWUP_RE.search(walk_msg)),
        affirm=False,
        analyze=False,
        prior_query=prior_feed["last_medical_query"],
    )
    assert "medical" in classify_intent(walk_msg, prior_slots=prior_feed)
    assert "medical" in classify_intent(scar_msg, prior_slots=prior_feed)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        child_db = ChildMemoryDB(Path(td) / "c.db")
        chat_db = ChatMemory(Path(td) / "chat.db")
        try:
            asst = ParentAssistant(
                db=child_db, chat_memory=chat_db, use_xlam=False, use_pleias=False
            )
            cid = child_db.create_child("Maya", "female", gestational_age_weeks=39)
            child_db.add_growth(
                cid,
                weeks=58.7,
                measure="weight",
                value=9.5,
                centile=45.0,
                z_score=-0.1,
                track_status="within_10_90",
                age_months=13.5,
            )
            sid = asst.start_session(cid)

            food = asst.chat(
                sid, "what foods are good for her ?", child_id=cid, ui_lang="en"
            )
            assert "medical" in food["intents"]
            food_q = food["slots"].get("last_medical_query") or ""
            assert "food" in food_q.lower()

            walk = asst.chat(sid, walk_msg, child_id=cid, ui_lang="en")
            assert "medical" in walk["intents"]
            walk_q = (walk["slots"].get("last_medical_query") or "").lower()
            assert "walk" in walk_q
            assert walk_q != food_q.lower()
            wr = (walk.get("reply") or "").lower()
            assert "i'm listening" not in wr
            assert any(
                k in wr for k in ("walk", "motor", "cruise", "stand", "milestone", "step")
            ), wr[:500]
            assert not any(k in wr for k in _FEEDING_BLEED), wr[:500]
            # Must not lead with toddler meal guidance
            lead = wr[:180]
            assert "family meal" not in lead and "finger food" not in lead
            assert "16" not in lead or "month" in lead

            scar = asst.chat(sid, scar_msg, child_id=cid, ui_lang="en")
            assert "medical" in scar["intents"]
            scar_q = (scar["slots"].get("last_medical_query") or "").lower()
            assert "scar" in scar_q
            assert scar_q != walk_q
            sr = (scar.get("reply") or "").lower()
            assert "i'm listening" not in sr
            assert any(
                k in sr for k in ("scar", "wound", "cut", "clean", "bandage", "redness", "heal")
            ), sr[:500]
            assert not any(k in sr for k in _FEEDING_BLEED), sr[:500]
            assert "milk" not in sr or "wound" in sr[:80]
        finally:
            asst.close()
