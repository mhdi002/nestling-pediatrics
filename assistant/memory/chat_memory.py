#!/usr/bin/env python3
"""Full conversational session memory for multi-turn parent chats."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from assistant.config import CHAT_DB_PATH


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        self._init()

    def _init(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
              session_id TEXT PRIMARY KEY,
              child_id TEXT,
              slots_json TEXT,
              title TEXT,
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
            CREATE INDEX IF NOT EXISTS idx_messages_session
              ON messages(session_id, created_at);
            """
        )
        self.conn.commit()

    def create_session(self, child_id: str | None = None, title: str | None = None) -> str:
        sid = str(uuid.uuid4())
        now = _utc()
        self.conn.execute(
            "INSERT INTO sessions(session_id, child_id, slots_json, title, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?)",
            (sid, child_id, "{}", title or "", now, now),
        )
        self.conn.commit()
        return sid

    def get_session(self, session_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["slots"] = json.loads(d.pop("slots_json") or "{}")
        return d

    def set_child(self, session_id: str, child_id: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET child_id=?, updated_at=? WHERE session_id=?",
            (child_id, _utc(), session_id),
        )
        self.conn.commit()
        self.merge_slots(session_id, {"child_id": child_id})

    def get_slots(self, session_id: str) -> dict:
        s = self.get_session(session_id)
        return dict(s["slots"]) if s else {}

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

    def get_history(self, session_id: str, limit: int | None = None) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY created_at ASC, rowid ASC",
            (session_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["tool_calls"] = json.loads(d.pop("tool_calls_json") or "null")
            d["meta"] = json.loads(d.pop("meta_json") or "{}")
            out.append(d)
        if limit is not None and limit >= 0:
            return out[-limit:] if limit else []
        return out

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

    def search_session(self, session_id: str, query: str, top_k: int = 5) -> list[dict]:
        q = (query or "").lower()
        hits = []
        for m in self.get_history(session_id):
            if q in (m.get("content") or "").lower():
                hits.append(m)
            if len(hits) >= top_k:
                break
        return hits

    def clear_session(self, session_id: str) -> int:
        cur = self.conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        self.conn.execute(
            "UPDATE sessions SET slots_json=?, updated_at=? WHERE session_id=?",
            ("{}", _utc(), session_id),
        )
        self.conn.commit()
        return cur.rowcount

    def list_sessions(self, child_id: str | None = None, limit: int = 40) -> list[dict]:
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
        out = []
        for r in rows:
            d = dict(r)
            d["slots"] = json.loads(d.pop("slots_json") or "{}")
            msgs = self.get_history(d["session_id"], limit=2)
            preview = ""
            for m in msgs:
                if m.get("role") == "user" and (m.get("content") or "").strip():
                    preview = (m["content"] or "").strip()
                    break
            if not preview and msgs:
                preview = (msgs[0].get("content") or "").strip()
            d["preview"] = preview[:120]
            d["message_count"] = len(self.get_history(d["session_id"]))
            out.append(d)
        return out

    def set_title(self, session_id: str, title: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET title=?, updated_at=? WHERE session_id=?",
            ((title or "")[:80], _utc(), session_id),
        )
        self.conn.commit()

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
