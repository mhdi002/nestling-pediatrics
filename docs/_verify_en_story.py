"""English multi-turn story verification for Nestling (port 8015)."""
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


def chat(sid: str, cid: str, msg: str) -> dict:
    return post(
        "/api/chat",
        {"session_id": sid, "child_id": cid, "message": msg, "ui_lang": "en"},
    )


def tool_names(out: dict) -> list[str]:
    names: list[str] = []
    for t in out.get("tool_results") or []:
        if t.get("name"):
            names.append(t["name"])
    if names:
        return names
    for tc in (out.get("tools") or {}).get("tool_calls") or []:
        if tc.get("name"):
            names.append(tc["name"])
    return names


def has_overlay(out: dict) -> bool:
    for t in out.get("tool_results") or []:
        if t.get("overlay_filename"):
            return True
    for tc in (out.get("tools") or {}).get("tool_calls") or []:
        if (tc.get("result") or {}).get("overlay_filename"):
            return True
    return False


def growth_result(out: dict) -> dict:
    for tc in (out.get("tools") or {}).get("tool_calls") or []:
        if tc.get("name") in ("plot_growth", "overlay_growth_on_chart", "score_growth"):
            res = dict(tc.get("result") or {})
            for t in out.get("tool_results") or []:
                if t.get("name") == tc.get("name"):
                    res.setdefault("overlay_filename", t.get("overlay_filename"))
                    res.setdefault("chart_standard", t.get("chart_standard"))
                    res.setdefault("centile", t.get("centile"))
                    res.setdefault("z_score", t.get("z_score"))
            return res
    for t in out.get("tool_results") or []:
        if t.get("name") in ("plot_growth", "overlay_growth_on_chart", "score_growth"):
            return t
    return {}


def check_turn(msg: str, out: dict) -> tuple[bool, str]:
    intents = out.get("intents") or []
    tools = tool_names(out)
    reply = out.get("reply") or ""
    gr = growth_result(out)

    if msg == "hi":
        ok = "help" in intents and "Nestling" in reply
        return ok, "help intent + Nestling greeting" if ok else f"intents={intents}, Nestling in reply={('Nestling' in reply)}"

    if msg == "show my child profile":
        ok = (
            "history" in intents
            and "overlay_growth_on_chart" not in tools
            and "Based on retrieved sources" not in reply
        )
        return ok, "history profile, no chart overlay" if ok else f"intents={intents}, tools={tools}"

    if msg == "boy weight 40 weeks 3.2 kg":
        age_m = gr.get("age_months")
        centile = gr.get("centile")
        chart_std = gr.get("chart_standard") or ""
        reply_has_who = "WHO" in reply
        ok = (
            "growth" in intents
            and has_overlay(out)
            and age_m is not None
            and float(age_m) < 1
            and centile is not None
            and float(centile) > 10
            and (chart_std == "who_term" or reply_has_who)
        )
        detail = f"growth+overlay, age_months={age_m:.2f}, centile={centile:.1f}, chart_standard={chart_std or 'WHO in reply'}"
        return ok, detail

    if msg == "show my child chart":
        ok = "growth" in intents and has_overlay(out) and "history" not in intents
        return ok, f"growth overlay, no history dump; intents={intents}, tools={tools}"

    if msg == "when will my son talk?":
        ok = (
            "medical" in intents
            and "growth" not in intents
            and "overlay_growth_on_chart" not in tools
            and "speech" in reply.lower()
        )
        return ok, f"medical only, speech in reply; intents={intents}, tools={tools}"

    if msg == "so its okey now":
        ok = "reassure" in intents and "I hear you" not in reply
        return ok, f"reassure intent, not vague; intents={intents}"

    if msg == "tell me about iron":
        ok = (
            "medical" in intents
            and "overlay_growth_on_chart" not in tools
            and "iron" in reply.lower()
        )
        return ok, f"medical iron answer; intents={intents}, tools={tools}"

    if msg == "what was my child's last growth result?":
        ok = "history" in intents and "3.2" in reply
        return ok, f"history recalls 3.2; intents={intents}, 3.2 in reply={'3.2' in reply}"

    return False, "unknown turn"


def main() -> int:
    print("HEALTH", get("/api/health"))
    child = post(
        "/api/children",
        {"name": "VerifyEN", "sex": "male", "gestational_age_weeks": 39},
    )
    cid = child.get("child_id") or child.get("id")
    sid = post("/api/sessions", {"child_id": cid, "title": "EN verify story"}).get("session_id")
    print("child_id", cid, "session_id", sid)

    turns = [
        "hi",
        "show my child profile",
        "boy weight 40 weeks 3.2 kg",
        "show my child chart",
        "when will my son talk?",
        "so its okey now",
        "tell me about iron",
        "what was my child's last growth result?",
    ]

    report_turns = []
    failed = 0
    for msg in turns:
        out = chat(sid, cid, msg)
        ok, detail = check_turn(msg, out)
        gr = growth_result(out)
        entry = {
            "message": msg,
            "pass": ok,
            "detail": detail,
            "intents": out.get("intents"),
            "tools": tool_names(out),
            "has_overlay": has_overlay(out),
            "growth": {k: gr.get(k) for k in ("age_months", "centile", "z_score", "chart_standard", "overlay_filename") if gr.get(k) is not None},
            "reply_preview": (out.get("reply") or "")[:300],
        }
        report_turns.append(entry)
        status = "PASS" if ok else "FAIL"
        print(f"{status} | {msg}")
        print(f"  {detail}")
        print(f"  intents={out.get('intents')} tools={tool_names(out)}")
        if not ok:
            failed += 1

    sessions = get("/api/sessions?limit=10")
    sess_count = len(sessions.get("sessions") or [])
    sess_ok = sess_count > 0
    print(f"\nGET /api/sessions: {'PASS' if sess_ok else 'FAIL'} ({sess_count} sessions)")
    if not sess_ok:
        failed += 1

    raw = {
        "child_id": cid,
        "session_id": sid,
        "turns": report_turns,
        "sessions_count": sess_count,
        "failed": failed,
    }
    Path("docs/_verify_en_story_raw.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nTOTAL FAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
