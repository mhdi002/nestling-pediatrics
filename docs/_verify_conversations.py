"""Multi-scenario agent verification + write docs/VERIFY_CONVERSATIONS.json."""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["NESTLING_LOAD_MODELS"] = "0"
os.environ["MPLBACKEND"] = "Agg"

from assistant.agent.orchestrator import ParentAssistant, classify_intent
from assistant.memory.chat_memory import ChatMemory
from assistant.memory.child_db import ChildMemoryDB
from assistant.runtime_translate import ensure_pure_lang
import intergrowth_preterm_equations as ig


def check(name, ok, detail=None):
    row = {"name": name, "ok": bool(ok), "detail": detail}
    print(("PASS" if ok else "FAIL"), name, detail or "")
    return row


def main():
    rows = []
    tmp = Path(tempfile.mkdtemp(prefix="nest_verify_"))
    db = ChildMemoryDB(tmp / "c.db")
    chat = ChatMemory(tmp / "chat.db")
    asst = ParentAssistant(db=db, chat_memory=chat, use_xlam=False, use_pleias=False)

    # Equations
    c = ig.centile_from_measurement("male", "weight", 40.0, 3.2)
    rows.append(check("eq_male_40w_3.2", abs(c - 30.85) < 0.2, {"centile": c}))

    # Create preterm + term children
    preterm = db.create_child("PretermMaya", "female", gestational_age_weeks=32)
    term = db.create_child("TermOmar", "male", gestational_age_weeks=39)

    # Growth persist + overlay
    g1 = asst.record_growth_and_overlay(preterm, "female", "weight", 40.0, 2.9)
    rows.append(check("db_growth_preterm", g1.get("ok") and g1.get("centile") is not None, g1.get("summary")))
    rows.append(check("db_growth_stored", len(db.growth_history(preterm)) >= 1))

    g2 = asst.record_growth_and_overlay(term, "male", "weight", None, 5.6, age_months=2.0)
    rows.append(check("db_growth_term_who", g2.get("ok") and g2.get("chart_standard") == "who_term", g2.get("summary")))

    # Conversation scenarios
    sid = asst.start_session(child_id=preterm)

    r_help = asst.chat(sid, "hi, how can you help me?", child_id=preterm, ui_lang="en")
    rows.append(check("conv_help", "help" in r_help["intents"] and "Nestling" in r_help["reply"]))
    rows.append(check("conv_help_no_fa", not re.search(r"[\u0600-\u06FF]", r_help["reply"])))

    r_iron = asst.chat(sid, "tell me about iron", child_id=preterm, ui_lang="en")
    rows.append(check("conv_iron", "medical" in r_iron["intents"] and "iron" in r_iron["reply"].lower()))

    r_simple = asst.chat(sid, "weight 3.0 kg at 42 weeks", child_id=preterm, ui_lang="en")
    tools = [t["name"] for t in (r_simple.get("tool_results") or [])]
    rows.append(check("conv_soft_growth_tools", "overlay_growth_on_chart" in tools or any("overlay" in (t or "") for t in tools), tools))
    rows.append(check("conv_soft_growth_no_fa", not re.search(r"[\u0600-\u06FF]", r_simple["reply"])))
    rows.append(check("conv_soft_growth_centile", "centile" in r_simple["reply"].lower()))

    r_hist = asst.chat(sid, "show my child profile and growth", child_id=preterm, ui_lang="en")
    hist_tools = [t["name"] for t in (r_hist.get("tool_results") or [])]
    rows.append(check("conv_history_intent", "history" in r_hist["intents"], r_hist["intents"]))
    rows.append(check("conv_history_tool", "get_child_summary" in hist_tools, hist_tools))
    rows.append(check("conv_history_has_name", "PretermMaya" in r_hist["reply"], r_hist["reply"][:300]))
    rows.append(check("conv_history_no_overlay_refire", "overlay_growth_on_chart" not in hist_tools, hist_tools))

    # Switch child mid-session
    sid2 = asst.start_session(child_id=term)
    r_term = asst.chat(sid2, "show my child", child_id=term, ui_lang="en")
    rows.append(check("conv_term_fetch", "TermOmar" in r_term["reply"] and "term" in r_term["reply"].lower(), r_term["reply"][:350]))

    # FA conversation purity
    sid3 = asst.start_session(child_id=preterm)
    r_fa = asst.chat(sid3, "سلام کمک کن", child_id=preterm, ui_lang="fa")
    rows.append(check("conv_fa_help", r_fa.get("reply_lang") == "fa" and "نستلینگ" in r_fa["reply"], r_fa["reply"][:200]))

    # Slots memory across turns
    sid4 = asst.start_session(child_id=preterm)
    asst.chat(sid4, "girl", child_id=preterm, ui_lang="en")
    r_slot = asst.chat(sid4, "weight 2.8 kg at 38 weeks", child_id=preterm, ui_lang="en")
    rows.append(check("memory_slots_sex", (r_slot.get("slots") or {}).get("sex") == "female", r_slot.get("slots")))
    rows.append(check("memory_history_len", len(r_slot.get("history") or []) >= 4))

    # Dossier API shape via db tools
    from assistant.tools.clinical import get_child_summary, set_child_db

    set_child_db(db)
    summ = get_child_summary(preterm, db=db)
    rows.append(check("tool_child_summary_ok", summ.get("ok") and summ.get("growth_count", 0) >= 1, summ.get("summary")))

    # Pure lang helper
    rows.append(check("pure_lang_strips_fa", "نارس" not in ensure_pure_lang("preterm نارس child", "en")))

    out = {
        "all_ok": all(r["ok"] for r in rows),
        "passed": sum(1 for r in rows if r["ok"]),
        "total": len(rows),
        "checks": rows,
    }
    path = ROOT / "docs" / "VERIFY_CONVERSATIONS.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"all_ok": out["all_ok"], "passed": out["passed"], "total": out["total"]}, indent=2))
    return 0 if out["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
