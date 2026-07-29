"""Multi-pattern live chat matching the parent's reported failure transcript."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = "http://127.0.0.1:8015"


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode("utf-8"))


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def chat(sid, cid, msg, lang="en"):
    return post(
        "/api/chat",
        {"session_id": sid, "child_id": cid, "message": msg, "ui_lang": lang},
    )


def main() -> int:
    print("HEALTH", get("/api/health"))
    child = post(
        "/api/children",
        {"name": "TermBabyMP", "sex": "male", "gestational_age_weeks": 39},
    )
    cid = child.get("child_id") or child.get("id")
    sid = (post("/api/sessions", {"child_id": cid})).get("session_id")
    print("CHILD", cid, "SESSION", sid)

    # Seed growth via API (term baby, 40w language — must plot near birth)
    seeded = post(
        "/api/growth",
        {
            "child_id": cid,
            "sex": "male",
            "measure": "weight",
            "weeks": 40,
            "value": 3.2,
            "gestational_age_weeks": 39,
        },
    )
    print(
        "SEEDED centile",
        seeded.get("centile"),
        "age_m",
        seeded.get("age_months"),
        "overlay",
        seeded.get("overlay_filename"),
    )

    turns = [
        "show my child chart",
        "show",
        "why my chald cant talk",
        "so its okey now",
        "the talking ability im worried about",
        "tell me about iron",
        "what was my child's last growth result?",
    ]

    failed = 0
    for msg in turns:
        out = chat(sid, cid, msg)
        reply = out.get("reply") or ""
        tools = [t.get("name") for t in (out.get("tool_results") or [])]
        overlay = any(t.get("overlay_filename") for t in (out.get("tool_results") or []))
        # also check nested tools
        for tc in (out.get("tools") or {}).get("tool_calls") or []:
            tools.append(tc.get("name"))
            res = tc.get("result") or {}
            if res.get("overlay_filename"):
                overlay = True
            if res.get("centile") is not None:
                centile = res.get("centile")
                age_m = res.get("age_months")
            else:
                centile = age_m = None
        centile = None
        age_m = None
        for tc in (out.get("tools") or {}).get("tool_calls") or []:
            res = tc.get("result") or {}
            if res.get("centile") is not None:
                centile = res.get("centile")
            if res.get("age_months") is not None:
                age_m = res.get("age_months")
            if res.get("overlay_filename"):
                overlay = True

        print("\n====", msg)
        print("intents", out.get("intents"))
        print("tools", tools, "overlay", overlay, "centile", centile, "age_m", age_m)
        print("reply:", reply[:320].replace("\n", " | "))

        checks = []
        if msg == "show my child chart":
            checks.append(("growth", "growth" in (out.get("intents") or [])))
            checks.append(("no_history", "history" not in (out.get("intents") or [])))
            checks.append(("overlay", overlay or "overlay_growth_on_chart" in tools))
            checks.append(("age_near_birth", age_m is not None and age_m < 1.0))
            checks.append(("centile_ok", centile is not None and centile > 10))
            checks.append(("no_measure_lecture", "By measure I mean" not in reply))
            checks.append(("no_history_dump", "Growth points:" not in reply))
        elif msg == "show":
            checks.append(("growth_or_overlay", "growth" in (out.get("intents") or []) or overlay))
            checks.append(("not_vague_chat", "I hear you" not in reply))
        elif msg == "why my chald cant talk":
            checks.append(("medical", "medical" in (out.get("intents") or [])))
            checks.append(("speech", "speech" in reply.lower() or "3-month" in reply.lower()))
            checks.append(("no_vitamin", "Vitamin A" not in reply))
            checks.append(("shortish", reply.count("Child growth & parenting") <= 1))
        elif msg == "so its okey now":
            checks.append(("reassure", "reassure" in (out.get("intents") or []) or "okay for now" in reply.lower() or "sounds okay" in reply.lower()))
            checks.append(("not_vague", "I hear you" not in reply))
        elif msg == "the talking ability im worried about":
            checks.append(("medical", "medical" in (out.get("intents") or [])))
            checks.append(("speech", "speech" in reply.lower()))
            checks.append(("no_4to5", "4 to 5 years" not in reply.lower()))
        elif msg.startswith("tell me about iron"):
            checks.append(("iron", "iron" in reply.lower()))
        elif "last growth" in msg:
            checks.append(("history", "history" in (out.get("intents") or [])))
            checks.append(("mentions_weight", "3.2" in reply or "weight" in reply.lower()))

        for name, ok in checks:
            print(("PASS" if ok else "FAIL"), name)
            if not ok:
                failed += 1

    print("\nFAILED", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
