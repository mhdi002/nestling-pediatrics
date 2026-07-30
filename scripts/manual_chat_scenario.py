#!/usr/bin/env python3
"""MANUAL-style single-session parent chat scenario via HTTP API."""

from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
import os

BASE = os.getenv("NESTLING_API_BASE", "http://127.0.0.1:8000/api")
REPORT_PATH = Path(__file__).resolve().parent.parent / "_manual_chat_report.txt"


def post(client: httpx.Client, path: str, body: dict) -> dict:
    r = client.post(f"{BASE}{path}", json=body, timeout=180.0)
    r.raise_for_status()
    return r.json()


def turn(client: httpx.Client, session_id: str, child_id: str, message: str, ui_lang: str | None = None) -> dict:
    body = {"session_id": session_id, "child_id": child_id, "message": message}
    if ui_lang:
        body["ui_lang"] = ui_lang
    return post(client, "/chat", body)


def check(label: str, ok: bool, detail: str) -> str:
    status = "PASS" if ok else "FAIL"
    return f"[{status}] {label}: {detail}"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    lines: list[str] = []
    verdicts: list[str] = []

    with httpx.Client() as client:
        # 1. Create child
        dob = (date.today() - timedelta(days=6)).isoformat()
        child = post(
            client,
            "/children",
            {
                "name": "Aria",
                "sex": "female",
                "date_of_birth": dob,
                "gestational_age_weeks": 39,
                "notes": "manual scenario test",
            },
        )
        child_id = child["child_id"]
        lines.append(f"Created child {child_id}: {child.get('child')}")

        # 2. Session
        sess = post(client, "/sessions", {"child_id": child_id, "title": "Manual growth+feeding test"})
        session_id = sess["session_id"]

        # 3. Growth measurement
        growth = post(
            client,
            "/growth",
            {
                "child_id": child_id,
                "sex": "female",
                "measure": "weight",
                "age_months": 0.2,
                "value": 3.2,
                "gestational_age_weeks": 39,
            },
        )
        lines.append(f"Growth recorded: centile={growth.get('centile')} overlay={growth.get('overlay_url') or growth.get('overlay')}")

        turns = [
            ("show my child chart", lambda o: "growth" in o.get("intents", []) and any(
                tc.get("name") == "overlay_growth_on_chart" for tc in (o.get("tools") or {}).get("tool_calls", [])
            ), "chart overlay intent+tool"),
            ("is my child growth okay?", lambda o: "growth_analysis" in o.get("intents", []), "growth_analysis intent"),
            ("is he okey?", lambda o: "growth_analysis" in o.get("intents", []) or "reassure" in o.get("intents", []), "follow-up growth okay"),
            (
                "what food should my baby eat at this age?",
                lambda o: "medical" in o.get("intents", [])
                and any(
                    k in (o.get("medical_rag") or {}).get("answer", "").lower()
                    for k in ("breast", "formula", "milk", "exclusive", "solid")
                )
                and "solid" not in (o.get("medical_rag") or {}).get("answer", "").lower()[:120]
                or "breast" in (o.get("medical_rag") or {}).get("answer", "").lower(),
                "age-appropriate feeding (milk not solids for 0.2mo)",
            ),
            (
                "what did we just look at?",
                lambda o: any(
                    w in (o.get("reply") or "").lower()
                    for w in ("chart", "growth", "weight", "centile", "curve", "overlay")
                ),
                "memory: chart recall",
            ),
            (
                "remind me about the chart",
                lambda o: any(w in (o.get("reply") or "").lower() for w in ("chart", "growth", "weight", "centile")),
                "memory: chart reminder",
            ),
            (
                "you said growth was okay — what about iron?",
                lambda o: "medical" in o.get("intents", [])
                and "iron" in ((o.get("medical_rag") or {}).get("answer") or o.get("reply") or "").lower(),
                "memory + iron topic",
            ),
            (
                "نوزاد من چقدر باید بخوابد؟",
                lambda o: "medical" in o.get("intents", [])
                or any(w in (o.get("reply") or "").lower() for w in ("sleep", "hour", "خواب")),
                "Persian sleep question",
            ),
            (
                "should my baby be talking words yet?",
                lambda o: "medical" in o.get("intents", [])
                and "growth" not in o.get("intents", [])
                and not any(
                    tc.get("name") == "overlay_growth_on_chart"
                    for tc in (o.get("tools") or {}).get("tool_calls", [])
                ),
                "speech without chart replot",
            ),
        ]

        for i, (msg, pred, label) in enumerate(turns, 1):
            ui = "fa" if "نوزاد" in msg else None
            t0 = time.time()
            out = turn(client, session_id, child_id, msg, ui_lang=ui)
            elapsed = time.time() - t0
            reply = (out.get("reply") or "")[:300]
            intents = out.get("intents", [])
            slots = {k: out.get("slots", {}).get(k) for k in ("child_id", "last_centile", "last_measure", "last_value")}
            mem = out.get("memory", {})
            ok = bool(pred(out))
            verdict = check(f"Turn {i}", ok, label)
            verdicts.append(verdict)
            block = [
                f"\n--- Turn {i} ---",
                f"USER: {msg}",
                f"INTENTS: {intents}",
                f"REPLY_HEAD: {reply}",
                f"SLOTS: {slots}",
                f"MEMORY: summary_len={len(mem.get('summary') or '')} facts={mem.get('facts')}",
                f"ELAPSED: {elapsed:.1f}s",
                verdict,
            ]
            if out.get("medical_rag"):
                cites = [c.get("id") for c in (out["medical_rag"].get("citations") or [])[:3]]
                block.append(f"MEDICAL_CITES: {cites}")
            if out.get("growth_analysis"):
                block.append(f"GROWTH_ANALYSIS: {out['growth_analysis']}")
            tools = (out.get("tools") or {}).get("tool_calls") or []
            if tools:
                block.append(f"TOOLS: {[t.get('name') for t in tools]}")
            lines.extend(block)
            print("\n".join(block))

    report = "\n".join(lines) + "\n\n=== VERDICTS ===\n" + "\n".join(verdicts)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")
    fails = sum(1 for v in verdicts if v.startswith("[FAIL]"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
