"""FA story verification for Nestling — writes docs/VERIFY_FA_STORY.md."""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = "http://127.0.0.1:8015"
CHILD_NAME = "داستانفا"


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


def check_turn(msg, out):
    """Return (pass: bool, reason: str)."""
    intents = out.get("intents") or []
    tools = tool_names(out)
    reply = out.get("reply") or ""

    if msg == "سلام":
        if "help" in intents and "نستلینگ" in reply:
            return True, "help intent + Nestling greeting in FA"
        return False, f"expected help + نستلینگ; got intents={intents}"

    if msg == "پروفیل بچمو نشون میدی ؟":
        if "history" not in intents:
            return False, f"expected history intent; got {intents}"
        if "overlay_growth_on_chart" in tools:
            return False, f"chart tool should not run; tools={tools}"
        if "بر اساس منابع بازیابی شده" in reply:
            return False, "RAG event dump in reply"
        return True, "history only, no chart, no RAG dump"

    if msg == "چارتشو نشون بده":
        if "growth" not in intents:
            return False, f"expected growth intent; got {intents}"
        if "history" in intents:
            return False, "history should not mash with chart"
        if not has_overlay(out):
            return False, f"expected overlay; tools={tools}"
        return True, "growth overlay, no history mash"

    if msg == "پسرم کی حرف میزنه؟":
        if "medical" not in intents:
            return False, f"expected medical intent; got {intents}"
        if "growth" in intents:
            return False, "growth must not appear on speech question"
        if "overlay_growth_on_chart" in tools:
            return False, f"MUST NOT reuse chart tools; tools={tools}"
        speech_kw = any(
            k in reply or k in reply.lower()
            for k in ("گفتار", "حرف", "کلمه", "speech", "talk", "language")
        )
        if not speech_kw:
            return False, "reply should mention speech/talk"
        if "رسم کردم" in reply or "I plotted" in reply:
            return False, "chart reply leaked into speech turn"
        return True, "medical only, no chart reuse"

    if msg == "پس خوبه":
        if "reassure" in intents:
            return True, "reassure intent"
        if any(k in reply for k in ("نگران", "خوب")) or "okay" in reply.lower():
            return True, "reassuring reply"
        return False, f"expected reassure; intents={intents}"

    if msg == "درباره آهن بگو":
        if "medical" not in intents:
            return False, f"expected medical; got {intents}"
        if "overlay_growth_on_chart" in tools:
            return False, f"no chart on iron question; tools={tools}"
        return True, "medical, no chart"

    return False, "unknown turn"


def main() -> int:
    health = get("/api/health")
    if health.get("status") != "ok":
        print("Server not healthy", health)
        return 2

    child = post(
        "/api/children",
        {"name": CHILD_NAME, "sex": "male", "gestational_age_weeks": 39},
    )
    cid = child.get("child_id") or child.get("id")
    sid = post("/api/sessions", {"child_id": cid, "title": "FA story verify"}).get(
        "session_id"
    )
    post(
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

    turns = [
        "سلام",
        "پروفیل بچمو نشون میدی ؟",
        "چارتشو نشون بده",
        "پسرم کی حرف میزنه؟",
        "پس خوبه",
        "درباره آهن بگو",
    ]

    results = []
    failed = 0
    for msg in turns:
        try:
            out = chat(sid, cid, msg, "fa")
            ok, reason = check_turn(msg, out)
        except urllib.error.URLError as exc:
            out = {"error": str(exc)}
            ok, reason = False, str(exc)
        except Exception as exc:
            out = {"error": str(exc)}
            ok, reason = False, str(exc)

        entry = {
            "message": msg,
            "pass": ok,
            "reason": reason,
            "intents": out.get("intents"),
            "tools": tool_names(out) if isinstance(out, dict) else [],
            "reply_preview": (out.get("reply") or "")[:300] if isinstance(out, dict) else "",
            "has_overlay": has_overlay(out) if isinstance(out, dict) else False,
        }
        results.append(entry)
        if not ok:
            failed += 1

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# VERIFY_FA_STORY",
        "",
        f"**Run:** {ts}  ",
        f"**API:** {BASE}  ",
        f"**Child:** {CHILD_NAME} (term, GA 39, male)  ",
        f"**Session:** `{sid}`  ",
        f"**Growth seed:** weight 3.2 kg @ 40 weeks  ",
        f"**NESTLING_LOAD_MODELS:** 0  ",
        "",
        "## Summary",
        "",
        f"- **Overall:** {'PASS' if failed == 0 else 'FAIL'} ({len(turns) - failed}/{len(turns)} turns passed)",
        "",
        "## Turns",
        "",
        "| # | Message | Result | Intents | Tools | Notes |",
        "|---|---------|--------|---------|-------|-------|",
    ]
    for i, r in enumerate(results, 1):
        status = "PASS" if r["pass"] else "FAIL"
        intents = ", ".join(r["intents"] or [])
        tools = ", ".join(r["tools"]) or "—"
        msg_esc = r["message"].replace("|", "\\|")
        reason_esc = r["reason"].replace("|", "\\|")
        lines.append(
            f"| {i} | {msg_esc} | **{status}** | {intents} | {tools} | {reason_esc} |"
        )

    lines.extend(["", "## Reply previews", ""])
    for i, r in enumerate(results, 1):
        lines.append(f"### Turn {i}: {r['message']}")
        lines.append("")
        lines.append("```")
        lines.append(r["reply_preview"] or "(no reply)")
        lines.append("```")
        lines.append("")

    out_path = ROOT / "docs" / "VERIFY_FA_STORY.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = ROOT / "docs" / "VERIFY_FA_STORY.json"
    json_path.write_text(
        json.dumps(
            {
                "failed": failed,
                "child_id": cid,
                "session_id": sid,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"FAILED={failed}")
    for r in results:
        print(f"{'PASS' if r['pass'] else 'FAIL'} | {r['message']} | {r['reason']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
