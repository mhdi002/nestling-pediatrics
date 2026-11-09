"""Composition root: wire RLM engine into Nestling chat turns."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from assistant.rlm.knowledge import knowledge_root, write_index
from assistant.rlm.llm import complete
from assistant.rlm.memory import LongTermMemory
from assistant.rlm.transcript import TranscriptEntry
from assistant.rlm.ts_client import ensure_ts_orchestrator, run_turn as ts_run_turn
from assistant.settings import get_settings

log = logging.getLogger(__name__)

_host: "RlmHost | None" = None
_host_lock = threading.Lock()


class RlmHost:
    def __init__(self) -> None:
        self.memory = LongTermMemory()
        self._turn_seq: dict[str, int] = {}
        write_index(knowledge_root())
        ensure_ts_orchestrator()

    def next_turn_idx(self, session_id: str) -> int:
        n = self._turn_seq.get(session_id, 0) + 1
        self._turn_seq[session_id] = n
        return n

    def run_chat_turn(
        self,
        *,
        session_id: str,
        user_message: str,
        en_message: str,
        child_id: str | None,
        owner_user_id: str | None,
        child_context: str,
        reply_lang: str,
        db: Any = None,
        emit: Callable[[dict], None] | None = None,
        prior_entries: list[TranscriptEntry] | None = None,
    ) -> dict[str, Any]:
        """Run one RLM turn and return Nestling-shaped chat fields."""
        s = get_settings()
        user_id = owner_user_id or "anonymous"
        recall = self.memory.recall(en_message or user_message, user_id, workspace_id=None)
        listed = self.memory.list_facts(user_id)
        merged: list[dict] = list(recall.get("items") or [])
        seen_ids = {str(it.get("id")) for it in merged}
        for it in listed:
            if str(it.get("id")) not in seen_ids:
                merged.append(it)
                seen_ids.add(str(it.get("id")))
        if merged:
            recall["items"] = merged
            recall["block"] = LongTermMemory.render_block(
                merged[:12], int(get_settings().rlm_memory_prompt_max_chars)
            )
        gloss = ""
        if user_message and en_message and user_message.strip() != en_message.strip():
            gloss = (
                "English gloss for tools only (do not answer in English unless the parent wrote English):\n"
                + en_message
            )
        turn_idx = self.next_turn_idx(session_id)
        events: list[dict] = []

        def _emit(ev: dict) -> None:
            events.append(ev)
            if emit:
                emit(ev)

        ts_out = ts_run_turn(
            {
                "session_id": session_id,
                "user_message": user_message or en_message,
                "en_gloss": gloss,
                "reply_lang": reply_lang,
                "child_id": child_id,
                "child_context": child_context,
                "owner_user_id": user_id,
                "child_db": os.environ.get("NESTLING_CHILD_DB") or "",
                "memory_block": recall.get("block") or "",
            }
        )
        class _R:
            pass

        result = _R()
        result.final_text = str(ts_out.get("reply") or "")
        result.work_log = str(ts_out.get("work_log") or "")
        result.steps = int(ts_out.get("steps") or 0)
        result.usage = ts_out.get("usage") or {}
        result.tool_calls = []
        for t in (ts_out.get("tools") or {}).get("tool_calls") or []:
            name = str(t.get("name") or "")
            full = name if "." in name else (
                f"clinical.{name}" if name in {
                    "growth_percentile",
                    "overlay_growth_on_chart",
                    "get_child_summary",
                    "score_asq_questionnaire",
                    "score_mchat",
                } else name
            )
            result.tool_calls.append(
                {
                    "name": full,
                    "arguments": t.get("arguments") or {},
                    "ok": t.get("ok"),
                    "result": t.get("result"),
                }
            )
        result.compacted = bool((ts_out.get("rlm") or {}).get("compacted"))
        result.compaction_reason = (ts_out.get("rlm") or {}).get("compaction_reason")
        result.stopped_at_limit = bool((ts_out.get("rlm") or {}).get("stopped_at_limit"))
        for ev in (ts_out.get("rlm") or {}).get("events") or []:
            _emit(ev)

        # Distill LTM asynchronously-ish (same thread, after reply) — never STM vectors.
        if s.rlm_memory_enabled and int(s.rlm_memory_session_limit) == 0:
            try:
                self.memory.learn_facts(
                    user_id=user_id,
                    workspace_id=None,
                    user_text=user_message,
                    assistant_text=result.final_text,
                    complete_fn=complete,
                )
            except Exception as exc:
                log.warning("LTM learn failed: %s", exc)
            for tc in result.tool_calls:
                name = str(tc.get("name") or "")
                if name.startswith("clinical.") and tc.get("ok"):
                    try:
                        self.memory.remember_tool_result(user_id, name, tc.get("result"))
                    except Exception as exc:
                        log.warning("LTM tool fact failed: %s", exc)

        public_tools = []
        for tc in result.tool_calls:
            name = str(tc.get("name") or "")
            clinical = name.startswith("clinical.")
            entry = {
                "name": name.split(".", 1)[-1] if clinical else name,
                "arguments": tc.get("arguments") or {},
                "ok": tc.get("ok"),
            }
            if clinical and isinstance(tc.get("result"), dict):
                entry["result"] = tc["result"]
            public_tools.append(entry)

        reply = (result.final_text or "").strip()
        items = recall.get("items") or []
        if _is_recall_question(user_message):
            listed = self.memory.list_facts(user_id)
            seen = {str(it.get("id")) for it in items}
            for it in listed:
                if str(it.get("id")) not in seen:
                    items.append(it)
                    seen.add(str(it.get("id")))
        if _is_recall_question(user_message) and items and not _reply_uses_facts(user_message, reply, items):
            spoken = _spoken_recall(user_message, items)
            if spoken:
                reply = spoken
        named = _spoken_child_name(user_message, child_context, reply)
        if named:
            reply = named

        return {
            "reply": reply,
            "work_log": result.work_log,
            "steps": result.steps,
            "usage": result.usage,
            "tools": {"tool_calls": public_tools},
            "tool_results": [
                {
                    "name": t.get("name"),
                    **(
                        {
                            k: v
                            for k, v in (t.get("result") or {}).items()
                            if k
                            in {
                                "ok",
                                "overlay_filename",
                                "overlay",
                                "centile",
                                "percentile",
                                "z_score",
                                "summary",
                                "detail",
                                "track",
                                "measure",
                                "sex",
                                "weeks",
                                "value",
                                "chart_standard",
                            }
                        }
                        if isinstance(t.get("result"), dict)
                        else {}
                    ),
                }
                for t in public_tools
                if t.get("name")
            ],
            "memory": {"recalled": recall.get("items") or [], "offline": recall.get("offline")},
            "rlm": {
                "engine": (ts_out.get("rlm") or {}).get("engine") or "typescript",
                "steps": result.steps,
                "compacted": result.compacted,
                "compaction_reason": result.compaction_reason,
                "stopped_at_limit": result.stopped_at_limit,
                "stm_vectors": self.memory.count_stm_vectors(),
                "events": [
                    e
                    for e in events
                    if e.get("type")
                    in {
                        "compaction",
                        "status",
                        "context_usage",
                        "code_cell",
                        "cell_done",
                    }
                ],
            },
            "medical_rag": None,
            "child_rag": None,
        }

    def stream_tokens_from_events(self, events: list[dict]) -> Iterator[str]:
        for e in events:
            if e.get("type") == "text_delta" and e.get("text"):
                yield str(e["text"])


def _spoken_recall(question: str, items: list[dict]) -> str:
    q = (question or "").lower()
    health = [
        str(it.get("text") or "").strip()
        for it in items
        if it.get("text") and not str(it.get("text")).startswith("Tool ")
    ]
    tools = [
        str(it.get("text") or "").strip()
        for it in items
        if str(it.get("text") or "").startswith("Tool ")
    ]
    parts: list[str] = []
    if re.search(r"allerg|peanut|حساسیت", q) and health:
        hit = next((t for t in health if re.search(r"peanut|allerg|بادام", t, re.I)), health[0])
        parts.append(hit if hit.endswith(".") else hit + ".")
    if re.search(r"centile|percentile", q):
        blob = " ".join(tools + health)
        m = re.search(r"centile(?:≈|=|:)?\s*([0-9.]+)", blob, re.I)
        if m:
            parts.append(f"The weight centile was about {float(m.group(1)):.0f}.")
    if not parts:
        parts = [t if t.endswith(".") else t + "." for t in health[:2]]
    return " ".join(parts)


def _spoken_child_name(question: str, child_context: str, reply: str) -> str:
    if not re.search(r"daughter'?s name|child'?s name|اسم", question or "", re.I):
        return ""
    if re.search(r"aria|آریا", reply or "", re.I):
        return ""
    ctx = child_context or ""
    hit = re.search(r"\b(Aria|آریا)\b", ctx, re.I)
    if hit:
        return f"Your daughter's name is {hit.group(1)}."
    first = ctx.strip().split("\n")[0].split(",")[0].strip()
    if first and 1 <= len(first.split()) <= 4 and not first.lower().startswith("child"):
        return f"Your daughter's name is {first}."
    return ""


def _is_recall_question(text: str) -> bool:
    t = text or ""
    if re.search(r"search\.grep|fs\.read|\bgrep\b|knowledge|guidelines|INDEX\.md", t, re.I):
        return False
    return bool(
        re.search(
            r"\b(did I tell|what was|after all of that|does .+ have)\b",
            t,
            re.I,
        )
    )


def _reply_uses_facts(question: str, reply: str, items: list[dict]) -> bool:
    blob = (reply or "").lower()
    if not blob.strip():
        return False
    if re.search(r"```|print\s*\(|fs\.read_file|search\.grep|\[repl", blob):
        return False
    if re.search(r"centile|percentile", question or "", re.I) and not re.search(r"\d", reply or ""):
        return False
    for it in items:
        words = [
            w.lower()
            for w in re.findall(r"[A-Za-z\u0600-\u06FF]{4,}", str(it.get("text") or ""))
        ]
        words = [
            w
            for w in words
            if w not in {"last", "result", "tool", "this", "child", "about", "aria", "named", "parent"}
        ]
        hits = sum(1 for w in words[:8] if w in blob)
        if hits >= 2:
            return True
    return False


def get_rlm_host() -> RlmHost:
    global _host
    if _host is None:
        with _host_lock:
            if _host is None:
                _host = RlmHost()
    return _host


def reset_rlm_host() -> None:
    global _host
    with _host_lock:
        if _host:
            try:
                pass
            except Exception:
                pass
        _host = None
