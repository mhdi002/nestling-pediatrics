#!/usr/bin/env python3
"""Full conversational session memory for multi-turn parent chats."""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from assistant.config import CHAT_DB_PATH
from assistant.settings import get_settings


# Storage-level caps for the session list UI.
SESSION_TITLE_CHARS = 80
SESSION_PREVIEW_CHARS = 120
SESSION_PREVIEW_MESSAGES = 2


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _synchronized(fn):
    """Serialize a method against the shared SQLite connection."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)

    return wrapper


class ChatMemory:
    """
    Persistent chat sessions with full turn history, tool-call logs,
    and extracted slots (sex, measure, weeks, value, child_id) for multi-turn filling.
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path or CHAT_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # WAL + NORMAL: measured ~3× write throughput vs delete/FULL under load.
        # One connection is shared by every request thread (check_same_thread=False),
        # so all statements must be serialized. Reentrant: public methods call each other.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.RLock()
        self._init()

    def _init(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
              session_id TEXT PRIMARY KEY,
              child_id TEXT,
              slots_json TEXT,
              title TEXT,
              summary TEXT,
              created_at TEXT,
              updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
              id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL,
              role TEXT NOT NULL,
              content TEXT NOT NULL,
              tool_calls_json TEXT,
              meta_json TEXT,
              created_at TEXT,
              FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );
            CREATE TABLE IF NOT EXISTS session_facts (
              id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL,
              key TEXT NOT NULL,
              value_json TEXT NOT NULL,
              provenance TEXT,
              source_message_id TEXT,
              confidence REAL,
              created_at TEXT,
              updated_at TEXT,
              FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
              ON messages(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_facts_session
              ON session_facts(session_id, key);
            """
        )
        # Migrate older DBs missing summary column
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(sessions)").fetchall()}
        if "summary" not in cols:
            self.conn.execute("ALTER TABLE sessions ADD COLUMN summary TEXT")
        self.conn.commit()

    @_synchronized
    def create_session(self, child_id: str | None = None, title: str | None = None) -> str:
        sid = str(uuid.uuid4())
        now = _utc()
        self.conn.execute(
            "INSERT INTO sessions(session_id, child_id, slots_json, title, summary, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (sid, child_id, "{}", title or "", "", now, now),
        )
        self.conn.commit()
        return sid

    @_synchronized
    def get_session(self, session_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["slots"] = json.loads(d.pop("slots_json") or "{}")
        return d

    @_synchronized
    def set_child(self, session_id: str, child_id: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET child_id=?, updated_at=? WHERE session_id=?",
            (child_id, _utc(), session_id),
        )
        self.conn.commit()
        self.merge_slots(session_id, {"child_id": child_id})

    @_synchronized
    def get_slots(self, session_id: str) -> dict:
        s = self.get_session(session_id)
        return dict(s["slots"]) if s else {}

    @_synchronized
    def merge_slots(self, session_id: str, updates: dict) -> dict:
        slots = self.get_slots(session_id)
        for k, v in updates.items():
            if v is not None and v != "":
                slots[k] = v
        self.conn.execute(
            "UPDATE sessions SET slots_json=?, updated_at=? WHERE session_id=?",
            (json.dumps(slots, ensure_ascii=False), _utc(), session_id),
        )
        self.conn.commit()
        return slots

    @_synchronized
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: list | dict | None = None,
        meta: dict | None = None,
    ) -> str:
        if not self.get_session(session_id):
            raise ValueError(f"Unknown session_id: {session_id}")
        mid = str(uuid.uuid4())
        self.conn.execute(
            "INSERT INTO messages(id, session_id, role, content, tool_calls_json, meta_json, created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (
                mid,
                session_id,
                role,
                content,
                json.dumps(tool_calls, ensure_ascii=False, default=str)
                if tool_calls is not None
                else None,
                json.dumps(meta or {}, ensure_ascii=False),
                _utc(),
            ),
        )
        self.conn.execute(
            "UPDATE sessions SET updated_at=? WHERE session_id=?",
            (_utc(), session_id),
        )
        self.conn.commit()
        return mid

    @_synchronized
    def get_history(self, session_id: str, limit: int | None = None) -> list[dict]:
        """Oldest-first turns. `limit` keeps the newest N and is applied in SQL."""
        if limit is not None and limit <= 0:
            return []
        if limit is None:
            rows = self.conn.execute(
                "SELECT * FROM messages WHERE session_id=? ORDER BY created_at ASC, rowid ASC",
                (session_id,),
            ).fetchall()
        else:
            # Newest N in the DB, then flipped back to chronological order.
            rows = list(
                reversed(
                    self.conn.execute(
                        "SELECT * FROM messages WHERE session_id=? "
                        "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                        (session_id, int(limit)),
                    ).fetchall()
                )
            )
        out = []
        for r in rows:
            d = dict(r)
            d["tool_calls"] = json.loads(d.pop("tool_calls_json") or "null")
            d["meta"] = json.loads(d.pop("meta_json") or "{}")
            out.append(d)
        return out

    @_synchronized
    def history_text(
        self,
        session_id: str,
        limit: int = 20,
        roles: tuple[str, ...] | None = None,
    ) -> str:
        """Flat text for slot-filling / tool routing context."""
        lines = []
        for m in self.get_history(session_id, limit=limit):
            if roles and m.get("role") not in roles:
                continue
            lines.append(f"{m['role'].upper()}: {m['content']}")
        return "\n".join(lines)

    @_synchronized
    def get_summary(self, session_id: str) -> str:
        s = self.get_session(session_id)
        return (s.get("summary") or "") if s else ""

    @_synchronized
    def set_summary(self, session_id: str, summary: str) -> None:
        cap = get_settings().nestling_summary_max_chars
        self.conn.execute(
            "UPDATE sessions SET summary=?, updated_at=? WHERE session_id=?",
            ((summary or "")[:cap], _utc(), session_id),
        )
        self.conn.commit()

    @_synchronized
    def build_context(
        self,
        session_id: str,
        *,
        window: int | None = None,
        summary_trigger: int | None = None,
    ) -> dict[str, Any]:
        """
        Conversation context for the agent: rolling summary + recent turns + slots.
        When history exceeds summary_trigger, older turns are folded into summary.
        """
        settings = get_settings()
        if window is None:
            window = settings.nestling_history_window
        if summary_trigger is None:
            summary_trigger = settings.nestling_summary_trigger_turns
        session = self.get_session(session_id) or {}
        all_msgs = self.get_history(session_id)
        summary = (session.get("summary") or "").strip()
        if len(all_msgs) > summary_trigger:
            older = all_msgs[:-window] if window else all_msgs
            # Exclude the latest user turn (added before context is built)
            fold = older[:-1] if older and older[-1].get("role") == "user" else older
            if fold:
                bits = []
                turn_cap = settings.nestling_summary_turn_chars
                for m in fold[-settings.nestling_summary_fold_turns :]:
                    role = (m.get("role") or "?").upper()
                    content = (m.get("content") or "").strip().replace("\n", " ")
                    if content:
                        bits.append(f"{role}: {content[:turn_cap]}")
                folded = " | ".join(bits)
                if folded:
                    summary = (summary + " | " + folded).strip(" |") if summary else folded
                    summary = summary[-settings.nestling_summary_fold_max_chars :]
                    self.set_summary(session_id, summary)
        recent = self.get_history(session_id, limit=window)
        # Drop the trailing user message from "recent" display — it's the current turn
        if recent and recent[-1].get("role") == "user":
            recent = recent[:-1]
        lines = [f"{m['role'].upper()}: {(m.get('content') or '').strip()}" for m in recent if (m.get("content") or "").strip()]
        facts = self.list_facts(session_id)
        return {
            "summary": summary,
            "recent_text": "\n".join(lines),
            "slots": dict(session.get("slots") or {}),
            "facts": facts,
            "message_count": len(all_msgs),
        }

    @_synchronized
    def upsert_fact(
        self,
        session_id: str,
        key: str,
        value: Any,
        *,
        provenance: str = "slot",
        source_message_id: str | None = None,
        confidence: float = 1.0,
    ) -> str:
        now = _utc()
        row = self.conn.execute(
            "SELECT id FROM session_facts WHERE session_id=? AND key=?",
            (session_id, key),
        ).fetchone()
        payload = json.dumps(value, ensure_ascii=False, default=str)
        if row:
            self.conn.execute(
                "UPDATE session_facts SET value_json=?, provenance=?, source_message_id=?, "
                "confidence=?, updated_at=? WHERE id=?",
                (payload, provenance, source_message_id, confidence, now, row["id"]),
            )
            self.conn.commit()
            return row["id"]
        fid = str(uuid.uuid4())
        self.conn.execute(
            "INSERT INTO session_facts(id, session_id, key, value_json, provenance, "
            "source_message_id, confidence, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (fid, session_id, key, payload, provenance, source_message_id, confidence, now, now),
        )
        self.conn.commit()
        return fid

    @_synchronized
    def list_facts(self, session_id: str) -> dict[str, Any]:
        rows = self.conn.execute(
            "SELECT key, value_json, provenance, confidence FROM session_facts WHERE session_id=?",
            (session_id,),
        ).fetchall()
        out: dict[str, Any] = {}
        for r in rows:
            try:
                out[r["key"]] = {
                    "value": json.loads(r["value_json"]),
                    "provenance": r["provenance"],
                    "confidence": r["confidence"],
                }
            except (json.JSONDecodeError, TypeError):
                # Legacy rows stored raw strings; surface them as-is.
                out[r["key"]] = {"value": r["value_json"], "provenance": r["provenance"]}
        return out

    @_synchronized
    def search_session(self, session_id: str, query: str, top_k: int = 5) -> list[dict]:
        q = (query or "").lower()
        hits = []
        for m in self.get_history(session_id):
            if q in (m.get("content") or "").lower():
                hits.append(m)
            if len(hits) >= top_k:
                break
        return hits

    @_synchronized
    def clear_session(self, session_id: str) -> int:
        cur = self.conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        self.conn.execute(
            "UPDATE sessions SET slots_json=?, updated_at=? WHERE session_id=?",
            ("{}", _utc(), session_id),
        )
        self.conn.commit()
        return cur.rowcount

    @_synchronized
    def list_sessions(self, child_id: str | None = None, limit: int | None = None) -> list[dict]:
        settings = get_settings()
        if limit is None:
            limit = settings.nestling_session_list_limit
        limit = max(0, min(int(limit), settings.nestling_session_list_max_limit))
        if child_id:
            rows = self.conn.execute(
                "SELECT * FROM sessions WHERE child_id=? ORDER BY updated_at DESC LIMIT ?",
                (child_id, int(limit)),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        sids = [r["session_id"] for r in rows]
        counts: dict[str, int] = {sid: 0 for sid in sids}
        if sids:
            placeholders = ",".join("?" * len(sids))
            for cr in self.conn.execute(
                f"SELECT session_id, COUNT(*) AS n FROM messages "
                f"WHERE session_id IN ({placeholders}) GROUP BY session_id",
                sids,
            ):
                counts[cr["session_id"]] = int(cr["n"])
        out = []
        for r in rows:
            d = dict(r)
            d["slots"] = json.loads(d.pop("slots_json") or "{}")
            sid = d["session_id"]
            # Last 2 messages only (DESC + reverse) — avoid loading full history
            preview_rows = self.conn.execute(
                "SELECT role, content FROM messages WHERE session_id=? "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (sid, SESSION_PREVIEW_MESSAGES),
            ).fetchall()
            msgs = list(reversed(preview_rows))
            preview = ""
            for m in msgs:
                if m["role"] == "user" and (m["content"] or "").strip():
                    preview = (m["content"] or "").strip()
                    break
            if not preview and msgs:
                preview = (msgs[0]["content"] or "").strip()
            d["preview"] = preview[:SESSION_PREVIEW_CHARS]
            d["message_count"] = counts.get(sid, 0)
            out.append(d)
        return out

    @_synchronized
    def set_title(self, session_id: str, title: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET title=?, updated_at=? WHERE session_id=?",
            ((title or "")[:SESSION_TITLE_CHARS], _utc(), session_id),
        )
        self.conn.commit()

    @_synchronized
    def close(self) -> None:
        conn = getattr(self, "conn", None)
        if conn is not None:
            conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# Alias for callers that use the ChatMemoryDB name
ChatMemoryDB = ChatMemory
