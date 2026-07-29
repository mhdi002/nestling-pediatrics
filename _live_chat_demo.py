#!/usr/bin/env python3
"""Live end-to-end ParentAssistant chat demo (deterministic tools + BM25/extractive RAG)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["NESTLING_LOAD_MODELS"] = "0"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from assistant.agent.orchestrator import ParentAssistant
from assistant.memory.chat_memory import ChatMemory
from assistant.memory.child_db import ChildMemoryDB
from assistant.tools.clinical import MCHAT_REVERSE


def _tool_names(out: dict) -> list[str]:
    tools = (out.get("tools") or {}).get("tool_calls") or []
    return [t.get("name") for t in tools if isinstance(t, dict)]


def _overlay_path(out: dict) -> str | None:
    for tc in (out.get("tools") or {}).get("tool_calls") or []:
        res = tc.get("result") or {}
        if res.get("overlay_path"):
            return res["overlay_path"]
        if res.get("overlay_filename"):
            return res["overlay_filename"]
    return None


def _print_turn(n: int, msg: str, out: dict) -> None:
    print("\n" + "=" * 72)
    print(f"TURN {n}")
    print("=" * 72)
    print(f"USER: {msg}")
    print(f"TOOLS: {_tool_names(out)}")
    print(f"SLOTS: {json.dumps(out.get('slots') or {}, ensure_ascii=False)}")
    ov = _overlay_path(out)
    print(f"OVERLAY: {ov}")
    reply = out.get("reply") or ""
    print(f"ASSISTANT:\n{reply}")
    if out.get("medical_rag"):
        mr = out["medical_rag"]
        print(f"MEDICAL_RAG keys: {list(mr.keys())}")
        if mr.get("pending") or "FA" in str(mr) or "pending" in str(mr).lower():
            print(f"MEDICAL_RAG note: {json.dumps(mr, ensure_ascii=False)[:500]}")
    if out.get("child_rag"):
        print(f"CHILD_RAG answer excerpt: {(out['child_rag'].get('answer') or '')[:400]}")
    if out.get("missing_slots"):
        print(f"MISSING_SLOTS: {out['missing_slots']}")
    models = out.get("models") or {}
    print(f"MODELS declared: {models}")


def main() -> int:
    print("NESTLING_LOAD_MODELS =", os.environ.get("NESTLING_LOAD_MODELS"))
    demo_dir = ROOT / "data" / "children" / "_live_demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    db_path = demo_dir / "children_live.sqlite"
    chat_path = demo_dir / "chat_live.sqlite"
    # fresh each run
    for p in (db_path, chat_path):
        if p.exists():
            p.unlink()

    db = ChildMemoryDB(db_path)
    chat = ChatMemory(chat_path)
    asst = ParentAssistant(db=db, chat_memory=chat, use_xlam=False, use_pleias=False)
    print(f"xLAM enabled: {asst.tool_caller.enabled if hasattr(asst.tool_caller, 'enabled') else 'n/a'}")
    print(f"use_pleias: {asst.use_pleias}")

    cid = db.create_child("Demo Baby", "male", gestational_age_weeks=32)
    print(f"Created child_id={cid}")
    sid = asst.start_session(child_id=cid)
    print(f"session_id={sid}")

    turns = [
        "hi, how can you help me?",
        "Tell me about iron for breastfed infants",
        "our baby is a boy",
        "weight at 40 weeks value: 3.2 kg please overlay the chart",
        "what was my child's last growth result?",
    ]
    for i, msg in enumerate(turns, 1):
        out = asst.chat(sid, msg, child_id=cid)
        _print_turn(i, msg, out)

    print("\n" + "=" * 72)
    print("ASQ SESSION")
    print("=" * 72)
    asq_answers = {
        "communication": ["yes"] * 6,
        "gross_motor": ["sometimes"] * 6,
        "fine_motor": ["yes"] * 6,
        "problem_solving": ["yes"] * 6,
        "personal_social": ["sometimes"] * 6,
    }
    asq = asst.run_asq_session(cid, 4, asq_answers)
    print("questionnaire_available:", asq.get("questionnaire_available"))
    print("result.ok:", (asq.get("result") or {}).get("ok"))
    print("parent_report:\n", asq.get("parent_report"))
    print("needs_referral:", (asq.get("result") or {}).get("needs_referral"))

    print("\n" + "=" * 72)
    print("M-CHAT SESSION")
    print("=" * 72)
    answers = {i: "yes" for i in range(1, 21)}
    for i in MCHAT_REVERSE:
        answers[i] = "no"
    print(f"MCHAT_REVERSE={sorted(MCHAT_REVERSE)}; answers for reverse=no, rest=yes")
    mchat = asst.run_mchat_session(cid, answers)
    res = mchat.get("result") or {}
    print("ok:", res.get("ok"))
    print("risk:", res.get("risk") or res.get("risk_level") or res.get("level"))
    print("score/total:", res.get("score"), res.get("total"), res.get("failed"))
    print("summary/parent_report:", mchat.get("parent_report") or res.get("summary"))
    print("full result keys:", list(res.keys()))
    print(json.dumps({k: res.get(k) for k in ("ok", "risk", "risk_level", "score", "total_score", "failed", "summary", "detail") if k in res or True}, ensure_ascii=False, default=str)[:800])
    print("FULL_RESULT:", json.dumps(res, ensure_ascii=False, default=str)[:1200])

    asst.close()
    print("\nDemo complete. DB paths:", db_path, chat_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
