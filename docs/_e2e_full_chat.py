"""Full multi-turn chat + memory smoke test against live Nestling API."""
from __future__ import annotations

import json
import sys
import urllib.error
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


def main() -> int:
    print("HEALTH", get("/api/health"))

    child = post(
        "/api/children",
        {"name": "E2ETerm", "sex": "male", "gestational_age_weeks": 39},
    )
    cid = child.get("child_id") or child.get("id")
    if not cid and isinstance(child.get("child"), dict):
        cid = child["child"].get("id")
    print("CHILD", cid)

    session = post("/api/sessions", {"child_id": cid})
    sid = session.get("session_id") or session.get("id")
    print("SESSION", sid)

    turns = [
        ("fa", "سلام"),
        ("fa", "من پسرم سه ماهشه و حرف نمیزنه مشکل چیه ؟"),
        ("en", "show the chart"),
        ("en", "by the mesure you mean what?"),
        ("en", "boy, weight, 40 weeks, 3.2 kg"),
        ("en", "what was my child's last growth result?"),
        ("en", "tell me about iron"),
    ]

    results = []
    for lang, msg in turns:
        out = post(
            "/api/chat",
            {"session_id": sid, "child_id": cid, "message": msg, "ui_lang": lang},
        )
        reply = out.get("reply") or ""
        row = {
            "msg": msg,
            "lang": lang,
            "intents": out.get("intents"),
            "slots": {
                k: (out.get("slots") or {}).get(k)
                for k in ("sex", "measure", "weeks", "age_months", "value", "want_overlay")
            },
            "missing": out.get("missing_slots"),
            "tools": [],
            "centile": None,
            "age_months": None,
            "weeks": None,
            "reply_head": reply[:260].replace("\n", " | "),
        }
        for tc in (out.get("tools") or {}).get("tool_calls") or []:
            row["tools"].append(tc.get("name"))
            res = tc.get("result") or {}
            if res.get("centile") is not None:
                row["centile"] = res.get("centile")
            if res.get("age_months") is not None:
                row["age_months"] = res.get("age_months")
            if res.get("weeks") is not None:
                row["weeks"] = res.get("weeks")
        for t in out.get("tool_results") or []:
            if t.get("name") and t["name"] not in row["tools"]:
                row["tools"].append(t.get("name"))
            if t.get("centile") is not None:
                row["centile"] = t.get("centile")
        results.append(row)

    mem = post(
        "/api/chat",
        {
            "session_id": sid,
            "child_id": cid,
            "message": "remind me the weight we just plotted",
            "ui_lang": "en",
        },
    )

    try:
        dossier = get(f"/api/children/{cid}/dossier")
    except urllib.error.HTTPError as exc:
        dossier = {"error": str(exc)}

    print("\n=== TURN RESULTS ===")
    for i, r in enumerate(results, 1):
        print(f"\nT{i} [{r['lang']}] {r['msg']}")
        print(" intents", r["intents"])
        print(" slots", r["slots"], "missing", r["missing"])
        print(
            " tools",
            r["tools"],
            "centile",
            r["centile"],
            "age_m",
            r["age_months"],
            "weeks",
            r["weeks"],
        )
        print(" reply", r["reply_head"])

    print("\n=== MEMORY TURN ===")
    print("intents", mem.get("intents"))
    print("reply", (mem.get("reply") or "")[:400].replace("\n", " | "))
    print("history_len", len(mem.get("history") or []))
    print("session_slots", mem.get("slots"))

    print("\n=== DOSSIER ===")
    print(json.dumps(dossier, ensure_ascii=False)[:800])

    from assistant.tools.clinical import growth_percentile

    term = growth_percentile(
        "male", "weight", weeks=40, value=3.2, gestational_age_weeks=39
    )
    pre = growth_percentile(
        "male", "weight", weeks=40, value=3.2, gestational_age_weeks=32
    )

    checks = []
    checks.append(
        (
            "greeting_fa",
            results[0]["intents"] == ["help"] and "نستلینگ" in results[0]["reply_head"],
        )
    )
    checks.append(
        (
            "speech_medical",
            "medical" in (results[1]["intents"] or [])
            and "help" not in (results[1]["intents"] or []),
        )
    )
    checks.append(
        (
            "speech_not_help_dump",
            "می‌توانم:" not in results[1]["reply_head"]
            and "I can:" not in results[1]["reply_head"],
        )
    )
    checks.append(("show_chart_growth", "growth" in (results[2]["intents"] or [])))
    checks.append(("measure_explained", "weight" in results[3]["reply_head"].lower()))
    checks.append(
        (
            "term_40w_plot",
            "overlay_growth_on_chart" in results[4]["tools"]
            and results[4]["age_months"] is not None
            and results[4]["age_months"] < 1.0
            and (results[4]["centile"] or 0) > 10,
        )
    )
    checks.append(("history_intent", "history" in (results[5]["intents"] or [])))
    checks.append(("iron_medical", "medical" in (results[6]["intents"] or [])))
    mem_reply = (mem.get("reply") or "").lower()
    checks.append(
        (
            "memory_recalls_weight",
            "3.2" in mem_reply or "weight" in mem_reply or "centile" in mem_reply,
        )
    )
    checks.append(
        (
            "term_route_math",
            term.get("age_months", 99) < 1 and term.get("centile", 0) > 10,
        )
    )
    checks.append(("preterm_still_intergrowth", pre.get("chart_standard") == "intergrowth_preterm"))

    print("\n=== CHECKS ===")
    failed = 0
    for name, ok in checks:
        print(("PASS" if ok else "FAIL"), name)
        if not ok:
            failed += 1
    print(
        "TERM40 age_m",
        round(term.get("age_months") or -1, 3),
        "centile",
        round(term.get("centile") or -1, 1),
    )
    print("PRE40 centile", round(pre.get("centile") or -1, 1), pre.get("chart_standard"))
    print("FAILED", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
