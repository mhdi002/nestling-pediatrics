"""Full multi-turn EN + FA story verification for Nestling."""
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
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def chat(sid, cid, msg, lang):
    return post(
        "/api/chat",
        {"session_id": sid, "child_id": cid, "message": msg, "ui_lang": lang},
    )


def tool_names(out):
    names = []
    for t in out.get("tool_results") or []:
        if t.get("name"):
            names.append(t["name"])
    for tc in (out.get("tools") or {}).get("tool_calls") or []:
        if tc.get("name"):
            names.append(tc["name"])
    return names


def has_overlay(out):
    for t in out.get("tool_results") or []:
        if t.get("overlay_filename"):
            return True
    for tc in (out.get("tools") or {}).get("tool_calls") or []:
        if (tc.get("result") or {}).get("overlay_filename"):
            return True
    return False


def main() -> int:
    print("HEALTH", get("/api/health"))
    failed = 0

    # —— EN story ——
    child = post(
        "/api/children",
        {"name": "StoryEN", "sex": "male", "gestational_age_weeks": 39},
    )
    cid = child.get("child_id") or child.get("id")
    sid = post("/api/sessions", {"child_id": cid, "title": "EN story"}).get("session_id")
    en_turns = [
        ("hi", lambda o: "help" in o["intents"] and "Nestling" in (o.get("reply") or "")),
        (
            "show my child profile",
            lambda o: "history" in o["intents"]
            and "overlay_growth_on_chart" not in tool_names(o)
            and "Based on retrieved sources" not in (o.get("reply") or ""),
        ),
        (
            "boy weight 40 weeks 3.2 kg",
            lambda o: "growth" in o["intents"]
            and has_overlay(o)
            and any(
                ((tc.get("result") or {}).get("age_months") or 99) < 1
                for tc in (o.get("tools") or {}).get("tool_calls") or []
            ),
        ),
        (
            "show my child chart",
            lambda o: "growth" in o["intents"]
            and has_overlay(o)
            and "history" not in o["intents"],
        ),
        (
            "when will my son talk?",
            lambda o: "medical" in o["intents"]
            and "growth" not in o["intents"]
            and "overlay_growth_on_chart" not in tool_names(o)
            and "speech" in (o.get("reply") or "").lower(),
        ),
        (
            "so its okey now",
            lambda o: "reassure" in o["intents"] and "I hear you" not in (o.get("reply") or ""),
        ),
        (
            "tell me about iron",
            lambda o: "medical" in o["intents"]
            and "overlay_growth_on_chart" not in tool_names(o)
            and "iron" in (o.get("reply") or "").lower(),
        ),
        (
            "what was my child's last growth result?",
            lambda o: "history" in o["intents"] and "3.2" in (o.get("reply") or ""),
        ),
    ]

    print("\n=== EN STORY ===")
    for msg, check in en_turns:
        out = chat(sid, cid, msg, "en")
        ok = False
        try:
            ok = bool(check(out))
        except Exception as exc:
            print("CHECK ERR", msg, exc)
        print(("PASS" if ok else "FAIL"), msg, "intents", out.get("intents"), "tools", tool_names(out))
        print(" ", (out.get("reply") or "")[:180].replace("\n", " | "))
        if not ok:
            failed += 1

    # —— FA story ——
    child_fa = post(
        "/api/children",
        {"name": "داستان‌فا", "sex": "male", "gestational_age_weeks": 39},
    )
    cid_fa = child_fa.get("child_id") or child_fa.get("id")
    sid_fa = post("/api/sessions", {"child_id": cid_fa, "title": "FA story"}).get("session_id")
    # seed growth
    post(
        "/api/growth",
        {
            "child_id": cid_fa,
            "sex": "male",
            "measure": "weight",
            "weeks": 40,
            "value": 3.2,
            "gestational_age_weeks": 39,
        },
    )
    fa_turns = [
        ("سلام", lambda o: "help" in o["intents"] and "نستلینگ" in (o.get("reply") or "")),
        (
            "پروفیل بچمو نشون میدی ؟",
            lambda o: "history" in o["intents"]
            and "overlay_growth_on_chart" not in tool_names(o)
            and "بر اساس منابع بازیابی شده" not in (o.get("reply") or ""),
        ),
        (
            "چارتشو نشون بده",
            lambda o: "growth" in o["intents"]
            and has_overlay(o)
            and "history" not in o["intents"],
        ),
        (
            "پسرم کی حرف میزنه؟",
            lambda o: (
                "medical" in o["intents"]
                and "growth" not in o["intents"]
                and "overlay_growth_on_chart" not in tool_names(o)
                and (
                    "گفتار" in (o.get("reply") or "")
                    or "حرف" in (o.get("reply") or "")
                    or "کلمه" in (o.get("reply") or "")
                    or "speech" in (o.get("reply") or "").lower()
                )
            ),
        ),
        (
            "پس خوبه",
            lambda o: "reassure" in o["intents"]
            or "نگران" in (o.get("reply") or "")
            or "خوب" in (o.get("reply") or "")
            or "okay" in (o.get("reply") or "").lower(),
        ),
        (
            "درباره آهن بگو",
            lambda o: "medical" in o["intents"] and "overlay_growth_on_chart" not in tool_names(o),
        ),
    ]

    print("\n=== FA STORY ===")
    for msg, check in fa_turns:
        out = chat(sid_fa, cid_fa, msg, "fa")
        ok = False
        try:
            ok = bool(check(out))
        except Exception as exc:
            print("CHECK ERR", msg, exc)
            ok = False
        print(("PASS" if ok else "FAIL"), msg, "intents", out.get("intents"), "tools", tool_names(out))
        print(" ", (out.get("reply") or "")[:200].replace("\n", " | "))
        if not ok:
            failed += 1

    # sessions API
    sessions = get("/api/sessions?limit=10")
    print("\nSESSIONS", len(sessions.get("sessions") or []))
    if not sessions.get("sessions"):
        failed += 1
        print("FAIL sessions list empty")

    print("\nFAILED", failed)
    Path("docs/E2E_STORY_REPORT.json").write_text(
        json.dumps({"failed": failed}, indent=2), encoding="utf-8"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
