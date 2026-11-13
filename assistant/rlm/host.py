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
from assistant.rlm.prompts import get_prompt
from assistant.rlm.transcript import TranscriptEntry
from assistant.rlm.ts_client import ensure_ts_orchestrator, run_turn as ts_run_turn
from assistant.settings import get_settings

log = logging.getLogger(__name__)


def _mostly_latin(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return True
    return sum(1 for ch in letters if ch.isascii()) / len(letters) >= 0.7


def english_gloss(user_message: str, fallback: str = "") -> str:
    """The parent's message in English, for searching an English library.

    Written by the model that is already serving the chat. The translation
    this replaces went to an online service (unreachable from many of the
    places this runs, and the parent's words left the machine) and fell back
    to a word-for-word glossary that produced half-translated text; a route
    chosen from that text is a route chosen from noise.
    """
    text = (user_message or "").strip()
    if not text or _mostly_latin(text):
        return text
    try:
        out = complete(
            [
                {"role": "system", "content": get_prompt("rlm/english_gloss.md")},
                {"role": "user", "content": text},
            ],
            max_tokens=max(64, min(600, len(text))),
            temperature=0.0,
        ).strip()
    except Exception as exc:
        log.warning("english gloss failed: %s", exc)
        out = ""
    if out and _mostly_latin(out):
        return out
    # No usable translation: a fallback that is not English either is not a gloss.
    return fallback if fallback and _mostly_latin(fallback) else ""


def _is_finding(tool_id: str) -> bool:
    from assistant.tools.clinical import TOOL_SPECS

    short = tool_id.split(".", 1)[-1]
    return any(spec.get("name") == short and spec.get("finding") for spec in TOOL_SPECS)

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
        en_message = english_gloss(user_message, en_message) or ""
        # What is remembered belongs to one child of one account. Keyed on the
        # account alone, a centile computed for one child was recalled in a
        # chat about a sibling; with sign-in off every family shared one key.
        memory_key = f"{user_id}|{child_id}" if child_id else user_id
        recall = self.memory.recall(en_message or user_message, memory_key, workspace_id=None)
        listed = self.memory.list_facts(memory_key)
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
            gloss = en_message
        # The child's age today, worked out from the record rather than read
        # back out of prose, so the library can say which age sections apply.
        child_age_months = None
        if db is not None and child_id:
            try:
                from assistant.rlm.clinical_dispatch import _age_months

                rec = db.get_child(child_id) or {}
                child_age_months = _age_months(rec.get("date_of_birth"))
            except Exception as exc:  # the turn still runs without it
                log.warning("child age unavailable: %s", exc)
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
                "child_age_months": child_age_months,
                "owner_user_id": user_id,
                # The database the app itself has open, so the clinical tools
                # read the same child record the parent is looking at. It used
                # to come from an environment variable that is only ever set
                # in tests, which left the tools without a database.
                "child_db": str(getattr(db, "path", "") or os.environ.get("NESTLING_CHILD_DB") or ""),
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
                    user_id=memory_key,
                    workspace_id=None,
                    # Both forms of what the parent wrote: a fact distilled in
                    # English from a Persian message is still the parent's.
                    user_text=(
                        "\n".join([user_message, en_message])
                        if en_message and en_message.strip() != (user_message or "").strip()
                        else user_message
                    ),
                    assistant_text=result.final_text,
                    complete_fn=complete,
                )
            except Exception as exc:
                log.warning("LTM learn failed: %s", exc)
            for tc in result.tool_calls:
                name = str(tc.get("name") or "")
                # Which results are findings is the tool's own declaration.
                if name.startswith("clinical.") and tc.get("ok") and _is_finding(name):
                    try:
                        self.memory.remember_tool_result(memory_key, name, tc.get("result"))
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

        # The reply is the orchestrator's. Saved facts and the child record
        # reach the model as context (memory_block, child_context); nothing
        # here rewrites an answer by matching the question's wording.
        reply = (result.final_text or "").strip()

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
            "memory": {
                "child_context": child_context,
                "recalled": recall.get("items") or [],
                "offline": recall.get("offline"),
            },
            "rlm": {
                "engine": (ts_out.get("rlm") or {}).get("engine") or "typescript",
                "turns": (ts_out.get("rlm") or {}).get("turns") or 1,
                "reply_via": ts_out.get("reply_via"),
                "failed": bool(ts_out.get("failed")),
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
            "task_board": ts_out.get("task_board"),
        }

    def stream_tokens_from_events(self, events: list[dict]) -> Iterator[str]:
        for e in events:
            if e.get("type") == "text_delta" and e.get("text"):
                yield str(e["text"])


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
