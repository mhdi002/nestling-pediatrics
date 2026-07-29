#!/usr/bin/env python3
"""Proof: load Salesforce/xLAM-1b-fc-r + PleIAs/Pleias-RAG-1B and exercise chat."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "huggingface"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ["NESTLING_LOAD_MODELS"] = "1"

sys.path.insert(0, str(ROOT))

DOCS = ROOT / "docs"
STATUS_PATH = DOCS / "PROOF_MODELS_STATUS.json"
CHAT_PATH = DOCS / "PROOF_MODELS_CHAT.txt"


def write_status(payload: dict) -> None:
    STATUS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("STATUS:", json.dumps(payload, ensure_ascii=False)[:800])


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("HF_HOME =", os.environ.get("HF_HOME"))
    print("NESTLING_LOAD_MODELS =", os.environ.get("NESTLING_LOAD_MODELS"))
    try:
        import torch

        print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
    except Exception as exc:
        tb = traceback.format_exc()
        print(tb)
        write_status({"loaded": False, "error": f"torch import failed: {exc!s}", "traceback": tb})
        return 1

    try:
        from assistant.agent.orchestrator import ParentAssistant
        from assistant.memory.chat_memory import ChatMemory
        from assistant.memory.child_db import ChildMemoryDB
        from assistant.rag.stores import _PLEIAS

        print("Constructing ParentAssistant(use_xlam=True, use_pleias=True) ...")
        demo_dir = ROOT / "data" / "children" / f"_proof_models_{os.getpid()}"
        demo_dir.mkdir(parents=True, exist_ok=True)
        db_path = demo_dir / "children_proof.sqlite"
        chat_path = demo_dir / "chat_proof.sqlite"
        print(f"proof sqlite dir: {demo_dir}")

        db = ChildMemoryDB(db_path)
        chat = ChatMemory(chat_path)
        asst = ParentAssistant(db=db, chat_memory=chat, use_xlam=True, use_pleias=True)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc!s}"
        tb = traceback.format_exc()
        print(tb)
        write_status({"loaded": False, "error": err, "traceback": tb})
        return 1

    tool_loaded = bool(getattr(asst.tool_caller, "enabled", False) and asst.tool_caller._model is not None)
    print("xLAM enabled/loaded:", tool_loaded, "use_pleias flag:", asst.use_pleias)

    lines: list[str] = []
    lines.append("NESTLING model proof chat")
    lines.append(f"HF_HOME={os.environ.get('HF_HOME')}")
    lines.append(f"proof_dir={demo_dir}")
    lines.append(f"tool_calling_loaded={tool_loaded}")
    lines.append(f"use_pleias={asst.use_pleias}")
    lines.append("")

    iron_reply = ""
    growth_reply = ""
    iron_mode = None
    pleias_ready = False
    growth_tool_model = None
    errors: list[str] = []
    tracebacks: list[str] = []

    try:
        sid = asst.start_session()
        iron_q = "Tell me about iron supplementation for breastfed infants / newborns"
        print("CHAT 1 (iron / Pleias RAG):", iron_q)
        out1 = asst.chat(sid, iron_q)
        iron_reply = out1.get("reply") or ""
        mr = out1.get("medical_rag") or {}
        iron_mode = mr.get("mode")
        pleias_ready = bool(_PLEIAS.ready)
        lines.append("=" * 72)
        lines.append("CHAT 1 — iron (Pleias RAG)")
        lines.append(f"USER: {iron_q}")
        lines.append(f"medical_rag.mode={iron_mode}")
        lines.append(f"medical_rag.model={mr.get('model')}")
        lines.append(f"pleias_ready={pleias_ready}")
        lines.append("ASSISTANT:")
        lines.append(iron_reply)
        if mr.get("answer"):
            lines.append("--- medical_rag.answer ---")
            lines.append(str(mr.get("answer"))[:4000])
        lines.append("")
    except Exception as exc:
        err = f"iron chat failed: {type(exc).__name__}: {exc!s}"
        tb = traceback.format_exc()
        errors.append(err)
        tracebacks.append(tb)
        print(err)
        print(tb)
        lines.append(f"ERROR iron chat: {err}")
        lines.append(tb)

    try:
        growth_q = (
            "Please compute growth percentile for a boy, weight 3.2 kg at 40 weeks "
            "corrected age (use tools)."
        )
        print("CHAT 2 (growth / xLAM tools):", growth_q)
        out2 = asst.chat(sid, growth_q)
        growth_reply = out2.get("reply") or ""
        tools = out2.get("tools") or {}
        growth_tool_model = tools.get("tool_model")
        lines.append("=" * 72)
        lines.append("CHAT 2 - growth tool-calling (xLAM)")
        lines.append(f"USER: {growth_q}")
        lines.append(f"tool_model={growth_tool_model}")
        lines.append(f"tool_calls={json.dumps(tools.get('tool_calls'), ensure_ascii=False, default=str)[:3000]}")
        lines.append(f"models={json.dumps(out2.get('models'), ensure_ascii=False)}")
        lines.append("ASSISTANT:")
        lines.append(growth_reply)
        lines.append("")
    except Exception as exc:
        err = f"growth chat failed: {type(exc).__name__}: {exc!s}"
        tb = traceback.format_exc()
        errors.append(err)
        tracebacks.append(tb)
        print(err)
        print(tb)
        lines.append(f"ERROR growth chat: {err}")
        lines.append(tb)

    CHAT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", CHAT_PATH)

    # Honest status: both models must actually be in memory for loaded=true
    rag_really = bool(pleias_ready and iron_mode and str(iron_mode).startswith("pleias"))
    loaded = bool(tool_loaded and rag_really)
    status = {
        "loaded": loaded,
        "tool_calling_loaded": tool_loaded,
        "rag_loaded": rag_really,
        "use_pleias_flag": bool(asst.use_pleias),
        "iron_rag_mode": iron_mode,
        "growth_tool_model": growth_tool_model,
        "sample_replies": {
            "iron": (iron_reply or "")[:800],
            "growth": (growth_reply or "")[:800],
        },
    }
    if errors:
        status["errors"] = errors
    if tracebacks:
        status["tracebacks"] = tracebacks
    if not loaded:
        missing = []
        if not tool_loaded:
            missing.append("xLAM not loaded")
        if not rag_really:
            missing.append(f"Pleias not confirmed (mode={iron_mode}, ready={pleias_ready})")
        status["error"] = "; ".join(missing) if missing else "partial failure"
    write_status(status)
    return 0 if loaded else 2


if __name__ == "__main__":
    raise SystemExit(main())


