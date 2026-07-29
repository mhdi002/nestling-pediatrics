#!/usr/bin/env python3
"""Proof chat demo: clean transcript for docs/PROOF_CHAT_TRANSCRIPT.txt."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ["NESTLING_LOAD_MODELS"] = "0"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.agent.orchestrator import ParentAssistant
from assistant.memory.chat_memory import ChatMemory
from assistant.memory.child_db import ChildMemoryDB

DOCS = Path(__file__).resolve().parent
OUT = DOCS / "PROOF_CHAT_TRANSCRIPT.txt"
IMAGES = DOCS / "images"


def _tool_names(out: dict) -> list[str]:
    return [t.get("name") for t in (out.get("tools") or {}).get("tool_calls") or []]


def _overlay_path(out: dict) -> str | None:
    for tc in (out.get("tools") or {}).get("tool_calls") or []:
        res = tc.get("result") or {}
        if res.get("overlay_path"):
            return str(res["overlay_path"])
        if res.get("overlay_filename"):
            return str(res["overlay_filename"])
    return None


def _write_turn(lines: list[str], n: int, msg: str, out: dict) -> None:
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"TURN {n}")
    lines.append("=" * 72)
    lines.append(f"USER: {msg}")
    lines.append(f"INTENTS: {out.get('intents')}")
    lines.append(f"TOOLS: {_tool_names(out)}")
    lines.append(f"SLOTS: {out.get('slots')}")
    ov = _overlay_path(out)
    if ov:
        lines.append(f"OVERLAY: {ov}")
    reply = (out.get("reply") or "").strip()
    lines.append("ASSISTANT:")
    lines.append(reply)
    if "medical_rag" in out:
        lines.append("(medical_rag present)")
    else:
        lines.append("(no medical_rag)")


def main() -> int:
    IMAGES.mkdir(parents=True, exist_ok=True)
    demo_dir = ROOT / "data" / "children" / "_proof_demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    db_path = demo_dir / "children.sqlite"
    chat_path = demo_dir / "chat.sqlite"
    for p in (db_path, chat_path):
        if p.exists():
            p.unlink()

    db = ChildMemoryDB(db_path)
    chat = ChatMemory(chat_path)
    asst = ParentAssistant(db=db, chat_memory=chat, use_xlam=False, use_pleias=False)

    lines: list[str] = []
    lines.append("Nestling proof chat transcript")
    lines.append(f"NESTLING_LOAD_MODELS={os.environ.get('NESTLING_LOAD_MODELS')}")
    lines.append(f"xLAM loaded: {asst.tool_caller.enabled}")
    lines.append(f"Pleias RAG: {asst.use_pleias}")

    cid = db.create_child("Demo Baby", "male", gestational_age_weeks=32)
    sid = asst.start_session(child_id=cid)
    lines.append(f"child_id={cid}")
    lines.append(f"session_id={sid}")

    turns = [
        "hi, how can you help me?",
        "tell me about iron",
        "boy",
        "weight 40 weeks 3.2 kg overlay",
        "what was my child's last growth result?",
    ]
    copied_overlay = False
    for i, msg in enumerate(turns, 1):
        out = asst.chat(sid, msg, child_id=cid)
        _write_turn(lines, i, msg, out)
        ov = _overlay_path(out)
        if ov and not copied_overlay:
            src = Path(ov)
            if not src.is_absolute():
                # try common overlay dirs
                candidates = [
                    ROOT / "data" / "overlays" / src.name,
                    ROOT / ov,
                    Path(ov),
                ]
                for c in candidates:
                    if c.exists():
                        src = c
                        break
            if src.exists():
                dest = IMAGES / "growth_overlay_demo.png"
                shutil.copy2(src, dest)
                lines.append(f"COPIED_OVERLAY -> {dest}")
                copied_overlay = True

    if not copied_overlay:
        # fallback: copy newest overlay from data/overlays
        overlay_dir = ROOT / "data" / "overlays"
        pngs = sorted(overlay_dir.glob("overlay_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pngs:
            dest = IMAGES / "growth_overlay_demo.png"
            shutil.copy2(pngs[0], dest)
            lines.append(f"FALLBACK_OVERLAY {pngs[0].name} -> {dest}")

    asst.close()
    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

