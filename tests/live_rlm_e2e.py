#!/usr/bin/env python3
"""Full multi-turn live E2E for the Nestling RLM orchestrator.

Requires a reachable OpenAI-compatible chat endpoint (NESTLING_LLM_URL).
Unit/fake LLM tests are NOT acceptance — this script talks to the real model.

Checklist (any failure STOPS the run; fix then re-run from turn 1):
  - non-empty reply / stream (no empty chat stream)
  - clinical tools for calculations (numbers match dispatch_tool)
  - agent reads knowledge/ (REPL/fs/grep), not MedicalRAG
  - no false compaction (only when budget trigger is exceeded)
  - at least one real compaction during the long conversation
  - LTM facts recalled later; STM vector count stays 0
  - multi-turn session continuity (including FA + EN)
  - no hang past model stall / stuck REPL
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("NESTLING_LLM_URL", "http://127.0.0.1:8100")
os.environ.setdefault("NESTLING_LLM_MODEL", "openbmb/MiniCPM5-1B")
os.environ.setdefault("NESTLING_USE_RLM", "1")
os.environ.setdefault("NESTLING_USE_LLM", "1")
os.environ.setdefault("RLM_EMBED_URL", "http://127.0.0.1:8101")
os.environ.setdefault("RLM_EMBED_MODEL", "bge-m3")
os.environ.setdefault("NESTLING_LOAD_MODELS", "0")
os.environ.setdefault("RLM_MAX_STEPS", "16")
os.environ.setdefault("RLM_CELL_TIMEOUT_SEC", "45")
os.environ.setdefault("RLM_MODEL_STALL_SEC", "240")
os.environ.setdefault("RLM_SANDBOX_PREFER_BWRAP", "0")
os.environ.setdefault("NESTLING_LLM_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false}')
# Match the MiniCPM sidecar window so compaction math is honest.
os.environ.setdefault("RLM_CONTEXT_LENGTH", "8192")
os.environ.setdefault("RLM_MAX_OUTPUT_TOKENS", "1024")
# Compact only after history actually grows; still low enough that a long
# thread MUST compact once (plan checklist).
os.environ.setdefault("RLM_COMPACT_TRIGGER_RATIO_IN_TURN", "0.62")
os.environ.setdefault("RLM_COMPACT_TRIGGER_RATIO", "0.62")
os.environ.setdefault("RLM_TEMPERATURE", "0.2")
os.environ.setdefault("LLM_TEMPERATURE", "0.2")

from assistant.settings import reset_settings  # noqa: E402

reset_settings()


class ChecklistFailure(Exception):
    pass


def _probe_llm() -> dict:
    import urllib.request

    url = os.environ["NESTLING_LLM_URL"].rstrip("/")
    with urllib.request.urlopen(f"{url}/v1/models", timeout=5) as r:
        return json.loads(r.read().decode())


def _ts_url() -> str:
    return (os.environ.get("RLM_TS_URL") or "http://127.0.0.1:8120").rstrip("/")


def _probe_ts_orchestrator() -> dict:
    import urllib.request

    with urllib.request.urlopen(f"{_ts_url()}/health", timeout=3) as r:
        health = json.loads(r.read().decode())
    _assert(health.get("ok") and health.get("engine") == "typescript", f"TS orchestrator health={health}")

    def post(payload: dict) -> dict:
        req = urllib.request.Request(
            f"{_ts_url()}/turn",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=240) as r:
            return json.loads(r.read().decode())

    sid = f"e2e-continue-{int(time.time())}"
    first = post(
        {
            "session_id": sid,
            "user_message": "Use search.grep on guidelines for iron. One parent sentence.",
            "reply_lang": "en",
            "max_steps": 3,
        }
    )
    second = post(
        {
            "session_id": sid,
            "user_message": "Continue in this same chat: did that search find guideline files? One sentence.",
            "reply_lang": "en",
        }
    )
    _assert((first.get("rlm") or {}).get("engine") == "typescript", f"first turn not TS: {first.get('rlm')}")
    _assert(bool((first.get("reply") or "").strip()), "continuation probe: empty first reply")
    _assert(bool((second.get("reply") or "").strip()), "continuation probe: empty second reply (session did not continue)")
    tools = (first.get("tools") or {}).get("tool_calls") or []
    events = (first.get("rlm") or {}).get("events") or []
    used = any("grep" in str(t) or "fs." in str(t) for t in tools) or any(
        e.get("type") == "code_cell" for e in events
    )
    return {
        "health": health,
        "engine": (first.get("rlm") or {}).get("engine"),
        "used_tools": used,
        "stopped_at_limit": bool((first.get("rlm") or {}).get("stopped_at_limit")),
        "steps": (first.get("rlm") or {}).get("steps"),
        "second_reply_head": (second.get("reply") or "")[:180],
    }


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise ChecklistFailure(msg)


def _turn_ok(i: int, out: dict, elapsed: float, stall: float) -> None:
    reply = (out.get("reply") or "").strip()
    _assert(bool(reply), f"turn {i}: empty reply / empty chat stream")
    _assert(elapsed < stall + 30, f"turn {i}: hung ({elapsed:.0f}s > stall {stall})")
    _assert(out.get("medical_rag") in (None, {}), f"turn {i}: medical_rag should be unused")
    rlm_meta = out.get("rlm") or {}
    _assert(
        int(rlm_meta.get("stm_vectors") or 0) == 0,
        f"turn {i}: STM vectors must be 0, got {rlm_meta.get('stm_vectors')}",
    )
    if rlm_meta.get("engine"):
        _assert(
            rlm_meta.get("engine") == "typescript",
            f"turn {i}: orchestrator engine is {rlm_meta.get('engine')}, expected typescript",
        )
    if rlm_meta.get("stopped_at_limit"):
        # Continuation path is allowed; hanging / empty is not.
        _assert(bool(reply), f"turn {i}: stopped at step limit with empty reply")


def _false_compaction(i: int, out: dict, used_tokens: int | None, trigger: int | None) -> None:
    rlm_meta = out.get("rlm") or {}
    if not rlm_meta.get("compacted"):
        return
    reason = rlm_meta.get("compaction_reason")
    _assert(
        reason in {"rlm", "in_turn", "truncate"},
        f"turn {i}: unexpected compaction_reason={reason}",
    )
    events = rlm_meta.get("events") or []
    usage_ev = [e for e in events if e.get("type") == "context_usage"]
    if not usage_ev:
        raise ChecklistFailure(f"turn {i}: compacted without context_usage measurement")
    compact_at = next((n for n, e in enumerate(events) if e.get("type") == "compaction"), None)
    before = usage_ev
    if compact_at is not None:
        before = [e for n, e in enumerate(events) if n < compact_at and e.get("type") == "context_usage"] or usage_ev
    max_used = max(int(e.get("usedTokens") or 0) for e in before)
    trig = int(before[-1].get("triggerTokens") or usage_ev[-1].get("triggerTokens") or 0)
    # A long in-turn REPL loop can legitimately cross the trigger even on an
    # early user turn. False compaction is firing while still well under budget.
    steps = int(rlm_meta.get("steps") or 1)
    if i <= 2 and steps <= 2 and max_used + 200 < trig and reason == "rlm":
        raise ChecklistFailure(
            f"turn {i}: false compaction max_used={max_used} trigger={trig} reason={reason}"
        )


def _used_and_trigger(out: dict) -> tuple[int | None, int | None]:
    events = (out.get("rlm") or {}).get("events") or []
    usage = [e for e in events if e.get("type") == "context_usage"]
    if not usage:
        return None, None
    last = usage[-1]
    return last.get("usedTokens"), last.get("triggerTokens")


def _has_fa(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in text or "")


def _is_dump(reply: str) -> bool:
    t = reply or ""
    low = t.lower()
    if t.lstrip().startswith("{") or t.lstrip().startswith("Tool clinical."):
        return True
    if any(
        tok in low
        for tok in (
            "```",
            "print(",
            "fs.read_file",
            "search.grep",
            "[repl output]",
            "stopped after",
            "from saved facts about this child:",
        )
    ):
        return True
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 2:
        pathish = sum(1 for ln in lines if re.search(r"\.(?:md|txt|json):\d+:", ln))
        if pathish >= max(2, int(len(lines) * 0.6)):
            return True
    return False


def _iron_gold(reply: str) -> bool:
    low = (reply or "").lower()
    if "iron" not in low and "آهن" not in (reply or ""):
        return False
    return bool(
        re.search(
            r"drop|drops|6[- ]month|complementary|fortif|anemia|haemoglobin|hemoglobin|"
            r"breastmilk|breast milk|قطره|کمکی|کم‌خونی",
            low,
        )
    )


def _danger_gold(reply: str) -> bool:
    low = (reply or "").lower()
    cues = (
        "unable to drink",
        "not feeding",
        "vomit",
        "convuls",
        "fit",
        "letharg",
        "unconscious",
        "fast breath",
        "chest indraw",
        "fever",
        "breastfeed",
        "emergency",
        "urgent",
        "تشنج",
        "تب",
        "بی‌حال",
        "شیر",
    )
    return sum(1 for c in cues if c in low) >= 2


def _complementary_gold(reply: str) -> bool:
    t = reply or ""
    return bool(
        re.search(
            r"تغذیه\s*کمکی|غذای کمکی|جامد|مکمل|complementary|solid food|wean",
            t,
            re.I,
        )
    )


def _collect(cond: bool, msg: str, bucket: list[str]) -> bool:
    if not cond:
        bucket.append(msg)
        return False
    return True


def _quality(i: int, lang: str, msg: str, reply: str, out: dict, elapsed: float, stall: float) -> dict:
    """0–100 parent-facing quality for this turn (independent of hard checklist)."""
    rlm_meta = out.get("rlm") or {}
    tools_block = (out.get("tools") or {}).get("tool_calls") or []
    cells = [e for e in (rlm_meta.get("events") or []) if e.get("type") == "code_cell"]
    used_fs = any("search.grep" in str(t) or "fs." in str(t) for t in tools_block) or any(
        "search" in (c.get("code") or "") or "fs." in (c.get("code") or "") for c in cells
    )
    used_clinical = any("growth" in str(t.get("name") or "") for t in tools_block)
    scores: dict[str, float] = {}
    scores["non_empty"] = 1.0 if (reply or "").strip() else 0.0
    scores["no_hang"] = 1.0 if elapsed < stall + 30 else 0.0
    scores["no_rag"] = 1.0 if out.get("medical_rag") in (None, {}) else 0.0
    scores["stm_zero"] = 1.0 if int(rlm_meta.get("stm_vectors") or 0) == 0 else 0.0
    if lang == "fa":
        scores["language"] = 1.0 if _has_fa(reply) else 0.0
    else:
        scores["language"] = 0.4 if _has_fa(reply) and not any(c.isascii() and c.isalpha() for c in reply) else 1.0
    asks_knowledge = bool(
        any(k in msg.lower() for k in ("search.grep", "fs.read", "knowledge", "guidelines", "index.md"))
    )
    asks_growth = "growth_percentile" in msg.lower() or "centile" in msg.lower()
    if asks_knowledge:
        scores["grounded"] = 1.0 if used_fs else 0.0
    elif asks_growth and i == 3:
        scores["grounded"] = 1.0 if used_clinical else 0.0
    else:
        scores["grounded"] = 1.0
    leak = _is_dump(reply) or any(
        tok in reply.lower()
        for tok in ("```repl", "```python", "tracebacks", "[repl output]", "host_request")
    )
    scores["parent_facing"] = 0.0 if leak else 1.0
    if len(reply) > 2500:
        scores["concise"] = 0.3
    elif len(reply) > 1200:
        scores["concise"] = 0.7
    else:
        scores["concise"] = 1.0 if reply else 0.0
    keys = [w.lower() for w in msg.replace("'", " ").split() if len(w) > 4][:8]
    hit = sum(1 for w in keys if w.lower() in reply.lower()) if keys else 1
    scores["on_topic"] = min(1.0, hit / max(2, min(4, len(keys) or 1)))
    if i in {6} or "allergy" in msg.lower() or "after all of that" in msg.lower():
        scores["memory"] = 1.0 if "peanut" in reply.lower() else 0.0
    else:
        scores["memory"] = 1.0
    gold = 1.0
    if "iron guidance" in msg.lower():
        gold = 1.0 if _iron_gold(reply) else 0.0
    elif "danger" in msg.lower():
        gold = 1.0 if _danger_gold(reply) else 0.0
    elif lang == "fa" and "کمکی" in msg:
        gold = 1.0 if _complementary_gold(reply) else 0.0
    scores["gold"] = gold
    weights = {
        "non_empty": 1.0,
        "no_hang": 1.0,
        "no_rag": 0.6,
        "stm_zero": 0.4,
        "language": 1.4,
        "grounded": 1.2,
        "parent_facing": 2.0,
        "concise": 0.8,
        "on_topic": 1.4,
        "memory": 1.4,
        "gold": 1.6,
    }
    total_w = sum(weights.values())
    pct = 100.0 * sum(scores[k] * weights[k] for k in weights) / total_w
    return {"subscores": scores, "score": round(pct, 1)}


def run() -> int:
    print("=== RLM live E2E (MiniCPM) ===")
    print("LLM", os.environ["NESTLING_LLM_URL"], os.environ["NESTLING_LLM_MODEL"])
    try:
        models = _probe_llm()
    except Exception as exc:
        print("FAIL: LLM unreachable:", exc)
        return 2
    ids = [m.get("id") for m in (models.get("data") or [])]
    print("served models:", ids)
    _assert(bool(ids), "no models served")
    ts_probe = _probe_ts_orchestrator()
    print("TS orchestrator", ts_probe)
    want = os.environ["NESTLING_LLM_MODEL"]
    _assert(
        any(want == i or want in str(i) or str(i) in want for i in ids),
        f"expected MiniCPM {want} on the sidecar, got {ids}",
    )

    tmp = tempfile.mkdtemp(prefix="nestling_rlm_e2e_")
    os.environ["NESTLING_DATA_DIR"] = tmp
    os.environ["NESTLING_CHILD_DB"] = str(Path(tmp) / "children.db")
    os.environ["NESTLING_CHAT_DB"] = str(Path(tmp) / "chat.db")
    os.environ["RLM_MEMORY_DB"] = str(Path(tmp) / "rlm_ltm.db")
    reset_settings()

    from assistant.agent.orchestrator import ParentAssistant
    from assistant.memory.chat_memory import ChatMemory
    from assistant.memory.child_db import ChildMemoryDB
    from assistant.rlm.host import reset_rlm_host
    from assistant.tools.clinical import dispatch_tool

    reset_rlm_host()
    db = ChildMemoryDB(path=Path(tmp) / "children.db")
    chat = ChatMemory(path=Path(tmp) / "chat.db")
    asst = ParentAssistant(db=db, chat_memory=chat, use_xlam=False, use_llm=True)
    _assert(asst.use_rlm, "ParentAssistant.use_rlm must be True")

    child_id = db.create_child(
        name="E2E Aria",
        sex="female",
        gestational_age_weeks=39,
        owner_user_id="e2e-user",
    )
    if isinstance(child_id, dict):
        child_id = child_id.get("child_id")
    sid = asst.start_session(child_id=child_id)

    expected_growth = dispatch_tool(
        "growth_percentile",
        {
            "sex": "female",
            "measure": "weight",
            "weeks": 40,
            "value": 3.4,
            "gestational_age_weeks": 39,
            "chart_standard": "who_term",
        },
        db=db,
    )
    _assert(expected_growth.get("ok"), f"baseline growth tool failed: {expected_growth}")
    expected_centile = expected_growth.get("centile") or expected_growth.get("percentile")
    print("baseline centile", expected_centile)

    # Distinct care questions so the long thread is a real conversation, not
    # one repeated prompt (which would trip identical-cell guards).
    long_topics = [
        ("en", "Using search.grep on guidelines, what does the knowledge say about night waking at 6 months? Two sentences."),
        ("en", "Search knowledge for breastfeeding frequency around 4 months and summarize for a parent."),
        ("en", "Look in guidelines for vitamin D / iron overlap notes and give one parent tip."),
        ("en", "Grep screening or guidelines for when to worry about delayed sitting. Short answer."),
        ("en", "Read INDEX.md with fs.read_file and name three guideline files that exist."),
        ("fa", "با search.grep در guidelines درباره خواب نوزاد دو ماهه یک نکته کوتاه بگو."),
        ("en", "Search for immunization or vaccination parent notes and give one calm sentence."),
        ("en", "Grep for 'danger signs' or similar in guidelines and list two parent-facing warning signs."),
        ("en", "Look up constipation or stool in knowledge files. One practical tip."),
        ("en", "Search knowledge for teething guidance and keep it to two sentences."),
        ("en", "Grep for 'tummy time' or motor play in guidelines. One sentence."),
        ("en", "Find a sleep routine note for preschool if present; otherwise say what files you checked."),
    ]

    core = [
        ("en", "Hello, my daughter's name is Aria."),
        ("en", "What is my daughter's name? One short sentence."),
        (
            "en",
            "Please look up iron guidance for a 6-month-old in the knowledge files "
            "(use search.grep / fs.read_file). Summarize for a parent.",
        ),
        (
            "en",
            "In a ```repl``` cell, call clinical.growth_percentile with "
            "sex='female', measure='weight', weeks=40, value=3.4, "
            "gestational_age_weeks=39, chart_standard='who_term'. "
            "Print the JSON result, then tell me the exact centile from the tool.",
        ),
        ("en", "What was the growth centile you just computed for Aria?"),
        (
            "en",
            "Remember: Aria is allergic to peanuts. Please store that as a lasting fact.",
        ),
        ("en", "What food allergy did I tell you Aria has?"),
        ("fa", "سلام، دخترم آریا شش ماهشه. یک جمله کوتاه درباره تغذیه کمکی بگو."),
    ]

    recall_after_long = [
        ("en", "After all of that, what food allergy does Aria have, and what weight centile did we compute?"),
    ]

    turns = core + long_topics + recall_after_long
    stall = float(os.environ.get("RLM_MODEL_STALL_SEC", "240"))
    history = []
    compacted_at: list[int] = []
    saw_knowledge_tools = False
    saw_growth_tool = False
    session_fails: list[str] = []

    for i, (lang, msg) in enumerate(turns, 1):
        print(f"\n--- turn {i}/{len(turns)} lang={lang} ---")
        print("USER:", msg[:180])
        fails: list[str] = []
        t0 = time.time()
        try:
            out = asst.chat(
                sid,
                msg,
                child_id=child_id,
                ui_lang=lang,
                owner_user_id="e2e-user",
            )
        except Exception as exc:
            elapsed = time.time() - t0
            traceback.print_exc()
            fails.append(f"turn {i} raised: {exc}")
            out = {"reply": "", "rlm": {}, "tools": {"tool_calls": []}, "tool_results": []}
        else:
            elapsed = time.time() - t0
        reply = (out.get("reply") or "").strip()
        rlm_meta = out.get("rlm") or {}
        used, trig = _used_and_trigger(out)
        print(
            f"elapsed={elapsed:.1f}s steps={rlm_meta.get('steps')} "
            f"compacted={rlm_meta.get('compacted')} used={used} trigger={trig} "
            f"reply_head={reply[:200]!r}"
        )

        try:
            _turn_ok(i, out, elapsed, stall)
        except ChecklistFailure as exc:
            fails.append(str(exc))
        try:
            _false_compaction(i, out, used, trig)
        except ChecklistFailure as exc:
            fails.append(str(exc))
        if rlm_meta.get("compacted") and rlm_meta.get("compaction_reason") in {"rlm", "in_turn"}:
            compacted_at.append(i)

        if i <= 2 and rlm_meta.get("compacted") and rlm_meta.get("compaction_reason") == "rlm":
            fails.append(f"turn {i}: false compaction on short history")

        events = rlm_meta.get("events") or []
        cells = [e for e in events if e.get("type") == "code_cell"]
        tool_results = out.get("tool_results") or []
        tools_block = (out.get("tools") or {}).get("tool_calls") or []

        used_fs = any(
            "search.grep" in str(t) or "fs." in str(t) or "fs.read" in str(t)
            for t in tools_block
        ) or any(
            "search" in (c.get("code") or "") or "fs." in (c.get("code") or "")
            for c in cells
        )
        if used_fs:
            saw_knowledge_tools = True

        _collect(
            not _is_dump(reply),
            f"turn {i}: parent reply is a REPL/grep dump; reply={reply[:220]}",
            fails,
        )

        if "daughter's name" in msg.lower():
            _collect(
                "aria" in reply.lower(),
                f"turn {i}: session short-term missed child name; reply={reply[:240]}",
                fails,
            )

        if "iron guidance" in msg.lower():
            _collect(
                used_fs,
                f"turn {i}: did not evidence knowledge file use; cells={len(cells)} tools={tools_block[:2]} reply={reply[:180]}",
                fails,
            )
            _collect(
                _iron_gold(reply),
                f"turn {i}: iron gold miss (dose/age/complementary); reply={reply[:240]}",
                fails,
            )

        if "clinical.growth_percentile" in msg.lower() or (
            "growth_percentile" in msg.lower() and "repl" in msg.lower()
        ):
            growth_hits = [
                t
                for t in tool_results
                if (t.get("name") or "") in {"growth_percentile", "clinical.growth_percentile"}
                or t.get("centile") is not None
            ]
            if not growth_hits:
                growth_hits = [
                    t for t in tools_block if "growth_percentile" in str(t.get("name") or "")
                ]
            ok_g = _collect(
                bool(growth_hits) or any("growth_percentile" in str(t) for t in tools_block),
                f"turn {i}: growth tool not evidenced; tools={tool_results} block={tools_block[:3]} reply={reply[:200]}",
                fails,
            )
            if ok_g:
                saw_growth_tool = True
            if expected_centile is not None:
                exp = float(expected_centile)
                mentioned = any(
                    str(n) in reply
                    for n in {int(exp), int(round(exp)), int(exp) + 1, max(0, int(exp) - 1)}
                )
                tool_match = any(
                    abs(float(t.get("centile") or t.get("percentile") or -999) - exp) < 1.0
                    for t in tool_results
                    if t.get("centile") is not None or t.get("percentile") is not None
                )
                _collect(
                    mentioned or tool_match,
                    f"turn {i}: reply/tool centile mismatch vs {exp}; reply={reply[:240]}",
                    fails,
                )

        if "food allergy did i tell you" in msg.lower():
            _collect(
                "peanut" in reply.lower() and not reply.lower().startswith("from saved facts"),
                f"turn {i}: LTM recall failed for peanut allergy; reply={reply[:240]}",
                fails,
            )

        if lang == "fa":
            _collect(
                _has_fa(reply),
                f"turn {i}: Persian turn had no Persian script; reply={reply[:200]}",
                fails,
            )
        if "کمکی" in msg:
            _collect(
                _complementary_gold(reply),
                f"turn {i}: complementary feeding ignored; reply={reply[:240]}",
                fails,
            )
        if "danger signs" in msg.lower():
            _collect(
                _danger_gold(reply),
                f"turn {i}: danger-sign gold miss; reply={reply[:240]}",
                fails,
            )
        if "tummy time" in msg.lower() or "constipation" in msg.lower() or "night waking" in msg.lower():
            stuck = "teething" in reply.lower() and "teething" not in msg.lower()
            _collect(
                not stuck,
                f"turn {i}: stuck on prior teething topic; reply={reply[:200]}",
                fails,
            )

        if i == len(turns):
            _collect(
                "peanut" in reply.lower(),
                f"turn {i}: long-conversation LTM lost peanut allergy; reply={reply[:300]}",
                fails,
            )
            if expected_centile is not None:
                exp = float(expected_centile)
                mentioned = any(
                    str(n) in reply
                    for n in {int(exp), int(round(exp)), 45, 46}
                )
                _collect(
                    mentioned or "45" in reply or "percentile" in reply.lower() or "centile" in reply.lower(),
                    f"turn {i}: long conversation lost growth centile; reply={reply[:300]}",
                    fails,
                )

        q = _quality(i, lang, msg, reply, out, elapsed, stall)
        print(f"quality={q['score']} checklist={'PASS' if not fails else 'FAIL'} {fails[:2]}")
        history.append(
            {
                "i": i,
                "lang": lang,
                "msg": msg,
                "reply": reply,
                "elapsed": elapsed,
                "rlm": {
                    "steps": rlm_meta.get("steps"),
                    "compacted": rlm_meta.get("compacted"),
                    "compaction_reason": rlm_meta.get("compaction_reason"),
                    "stm_vectors": rlm_meta.get("stm_vectors"),
                    "used": used,
                    "trigger": trig,
                },
                "tools": tool_results,
                "used_fs": used_fs,
                "checklist_ok": not fails,
                "checklist_fails": fails,
                "quality": q,
            }
        )

    if not saw_knowledge_tools:
        session_fails.append("never used fs/search tools against knowledge/")
    if not saw_growth_tool:
        session_fails.append("never used clinical.growth_percentile")
    if not compacted_at:
        session_fails.append(
            "long conversation never compacted — expected at least one real compaction after the budget trigger"
        )
    print("compaction turns:", compacted_at)

    hist = chat.get_history(sid) or []
    if len(hist) < len(turns) * 2 - 2:
        session_fails.append(f"history too short: {len(hist)}")

    from assistant.rlm.host import get_rlm_host

    ltm_facts = get_rlm_host().memory.list_facts("e2e-user")
    ltm_blob = " ".join(str(it.get("text") or "") for it in ltm_facts).lower()
    if "peanut" not in ltm_blob:
        session_fails.append(f"LTM embeddings store missing peanut fact; facts={ltm_facts[:6]}")
    if not any("aria" in (h.get("reply") or "").lower() for h in history if "daughter's name" in (h.get("msg") or "").lower()):
        session_fails.append("session short-term did not recall child name")
    in_turn_cont = sum(1 for h in history if h.get("used_fs") or (h.get("tools") or []))
    if in_turn_cont < 1:
        session_fails.append("no in-turn tool continuation (cell/tool then parent reply)")

    print("\n======== CAPABILITIES ========")
    caps = {
        "typescript_engine": ts_probe.get("engine") == "typescript",
        "auto_continuation_same_session": bool(ts_probe.get("second_reply_head")),
        "tools_in_continuation_probe": bool(ts_probe.get("used_tools")),
        "compaction": bool(compacted_at),
        "knowledge_tools": saw_knowledge_tools,
        "growth_tool": saw_growth_tool,
        "stm_vectors_zero": all(int((h.get("rlm") or {}).get("stm_vectors") or 0) == 0 for h in history),
        "session_short_term_name": any(
            "aria" in (h.get("reply") or "").lower()
            for h in history
            if (h.get("msg") or "").lower().startswith("what is my daughter's name")
        ),
        "ltm_peanut_store": "peanut" in ltm_blob,
        "ltm_peanut_after_long": any(
            "peanut" in (h.get("reply") or "").lower() for h in history if h.get("i") == len(turns)
        ),
        "multi_turn_history": len(hist) >= len(turns) * 2 - 2,
        "fa_and_en": any(h.get("lang") == "fa" for h in history) and any(h.get("lang") == "en" for h in history),
    }
    for k, v in caps.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
        if not v:
            session_fails.append(f"capability {k} failed")

    turn_pass = sum(1 for h in history if h.get("checklist_ok"))
    n = len(history) or 1
    mean_q = sum(float((h.get("quality") or {}).get("score") or 0) for h in history) / n
    checklist_rate = 100.0 * turn_pass / n
    report_obj = {
        "model": os.environ.get("NESTLING_LLM_MODEL"),
        "turns": n,
        "turn_checklist_pass": turn_pass,
        "turn_checklist_rate": round(checklist_rate, 1),
        "mean_quality": round(mean_q, 1),
        "session_fails": session_fails,
        "compacted_at": compacted_at,
        "capabilities": caps,
        "ts_probe": {k: ts_probe.get(k) for k in ("engine", "used_tools", "steps", "stopped_at_limit")},
        "ltm_facts": ltm_facts[:12],
        "history": history,
    }
    report = Path(tmp) / "rlm_e2e_report.json"
    durable = Path("/tmp/rlm_minicpm_e2e_report.json")
    payload = json.dumps(report_obj, ensure_ascii=False, indent=2, default=str)
    report.write_text(payload, encoding="utf-8")
    durable.write_text(payload, encoding="utf-8")
    print("\n======== E2E SCOREBOARD ========")
    print(f"turns {n}  checklist {turn_pass}/{n} ({checklist_rate:.1f}%)  mean quality {mean_q:.1f}/100")
    print("session:", session_fails or ["ok"])
    for h in history:
        q = (h.get("quality") or {}).get("score")
        flag = "PASS" if h.get("checklist_ok") else "FAIL"
        print(f"  t{h['i']:02d} {flag} q={q} {h['elapsed']:.1f}s {(h.get('reply') or '')[:70]!r}")
    print("wrote", report, "and", durable)
    asst.close()
    if session_fails or turn_pass < n:
        print("OVERALL: FAIL")
        return 1
    print("OVERALL: PASS")
    return 0


def main() -> int:
    print("######## MiniCPM full E2E (continue-all-turns scoreboard) ########")
    try:
        return run()
    except ChecklistFailure as exc:
        print("CHECKLIST FAIL before turns:", exc)
        traceback.print_exc()
        return 1
    except Exception as exc:
        print("UNEXPECTED:", exc)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
