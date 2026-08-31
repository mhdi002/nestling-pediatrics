"""Memory kept in SQLite, ranked with the BM25 index this project already has.

The default backend, and the one that must never fail: it needs no graph
database, no embedding model and no network. That matters here because the
deployment host has blocked huggingface.co, pypi.org and deb.debian.org at
various points, and a memory layer that stops working when a download fails
is worse than a simpler one that always works.

Relevance uses the same BM25 implementation as the care-notes index, so a
question about an ulcer finds the turn where the ulcer was mentioned rather
than merely the most recent turns.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from assistant.memory.types import Episode, MemoryRecord, utc_now
from assistant.settings import get_settings

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_facts (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    owner_user_id TEXT,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    source TEXT,
    confidence REAL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    attributes TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON memory_facts(subject, owner_user_id);
CREATE INDEX IF NOT EXISTS idx_facts_current ON memory_facts(subject, valid_to);

CREATE TABLE IF NOT EXISTS memory_episodes (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    subject TEXT,
    owner_user_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    attributes TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON memory_episodes(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_episodes_subject ON memory_episodes(subject, owner_user_id);
"""


def _synchronized(fn):
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


class NativeMemoryBackend:
    """SQLite + BM25. No external services."""

    name = "native"

    def __init__(self, path: Path | str | None = None):
        settings = get_settings()
        if path is None:
            path = settings.nestling_memory_db or (
                Path(settings.nestling_chat_db).parent / "memory.db"
                if settings.nestling_chat_db
                else Path(settings.nestling_root) / "data" / "memory.db"
            )
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # WAL so a read during a write does not block a parent's chat turn.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def available(self) -> bool:
        try:
            self.conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error as exc:  # pragma: no cover - a broken file
            log.warning("Native memory unavailable: %s", exc)
            return False

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _owner_clause(owner_user_id: str | None, column: str = "owner_user_id") -> tuple[str, list]:
        """Scope to one account.

        A NULL owner is legacy data from before per-account scoping, and an
        unauthenticated (API-key) caller matches it. An authenticated caller
        sees only their own rows.
        """
        if owner_user_id is None:
            return "", []
        return f" AND ({column} = ? OR {column} IS NULL)", [owner_user_id]

    @staticmethod
    def _rank(rows: list[dict], query: str, text_key: str) -> list[dict]:
        """Order by BM25 relevance to `query`, best first."""
        from assistant.rag.embeddings import BM25Index

        texts = [r.get(text_key) or "" for r in rows]
        if not texts:
            return []
        index = BM25Index()
        index.fit(texts)
        scores = index.scores(query)
        # Nothing matched at all. That is not "there is nothing worth
        # recalling" -- it is a question with no distinctive terms, which is
        # what a follow-up like "how often?" looks like. Dropping everything
        # there emptied the child's memory out of the prompt at exactly the
        # moment the conversation depended on it, so fall back to the order
        # the caller already had, which is recency.
        if not any(float(s) > 0.0 for s in scores):
            return list(rows)
        ranked = sorted(range(len(rows)), key=lambda i: float(scores[i]), reverse=True)
        out = []
        for i in ranked:
            score = float(scores[i])
            # Some rows did match, so a zero here means this row is about
            # something else. Keeping it would pad the context with an
            # unrelated topic, which is how an ulcer question came back about
            # vaccination sites.
            if score <= 0.0:
                continue
            row = dict(rows[i])
            row["_score"] = score
            out.append(row)
        return out

    # -- semantic ---------------------------------------------------------
    @_synchronized
    def remember_fact(self, record: MemoryRecord) -> MemoryRecord:
        self.conn.execute(
            "INSERT OR REPLACE INTO memory_facts"
            "(id,subject,owner_user_id,kind,text,source,confidence,created_at,valid_from,valid_to,attributes)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                record.id,
                record.subject,
                record.owner_user_id,
                record.kind,
                record.text,
                record.source,
                float(record.confidence),
                record.created_at,
                record.valid_from,
                record.valid_to,
                json.dumps(record.attributes or {}),
            ),
        )
        self.conn.commit()
        return record

    @_synchronized
    def recall_facts(
        self,
        subject: str,
        *,
        owner_user_id: str | None = None,
        query: str = "",
        limit: int = 10,
        include_superseded: bool = False,
    ) -> list[MemoryRecord]:
        sql = "SELECT * FROM memory_facts WHERE subject = ?"
        params: list[Any] = [subject]
        clause, extra = self._owner_clause(owner_user_id)
        sql += clause
        params += extra
        if not include_superseded:
            sql += " AND (valid_to IS NULL OR valid_to > ?)"
            params.append(utc_now())
        sql += " ORDER BY created_at DESC"
        rows = [dict(r) for r in self.conn.execute(sql, params).fetchall()]
        for r in rows:
            try:
                r["attributes"] = json.loads(r.get("attributes") or "{}")
            except (TypeError, ValueError):
                r["attributes"] = {}
        if query.strip():
            rows = self._rank(rows, query, "text")
        return [MemoryRecord.from_dict(r) for r in rows[:limit]]

    @_synchronized
    def supersede_fact(self, record_id: str, *, at: str | None = None) -> bool:
        cur = self.conn.execute(
            "UPDATE memory_facts SET valid_to=? WHERE id=? AND valid_to IS NULL",
            (at or utc_now(), record_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    # -- episodic ---------------------------------------------------------
    @_synchronized
    def add_episode(self, episode: Episode) -> Episode:
        self.conn.execute(
            "INSERT OR REPLACE INTO memory_episodes"
            "(id,session_id,subject,owner_user_id,role,content,created_at,attributes)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (
                episode.id,
                episode.session_id,
                episode.subject,
                episode.owner_user_id,
                episode.role,
                episode.content,
                episode.created_at,
                json.dumps(episode.attributes or {}),
            ),
        )
        self.conn.commit()
        return episode

    @_synchronized
    def recall_episodes(
        self,
        *,
        session_id: str | None = None,
        subject: str = "",
        owner_user_id: str | None = None,
        query: str = "",
        limit: int = 12,
    ) -> list[Episode]:
        sql = "SELECT * FROM memory_episodes WHERE 1=1"
        params: list[Any] = []
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        if subject:
            sql += " AND subject = ?"
            params.append(subject)
        clause, extra = self._owner_clause(owner_user_id)
        sql += clause
        params += extra
        sql += " ORDER BY created_at DESC"
        rows = [dict(r) for r in self.conn.execute(sql, params).fetchall()]
        for r in rows:
            try:
                r["attributes"] = json.loads(r.get("attributes") or "{}")
            except (TypeError, ValueError):
                r["attributes"] = {}
        if query.strip():
            rows = self._rank(rows, query, "content")
        else:
            # No query: the most recent turns, back in reading order.
            rows = list(reversed(rows[:limit]))
        return [
            Episode(
                id=r["id"],
                session_id=r["session_id"],
                subject=r.get("subject") or "",
                owner_user_id=r.get("owner_user_id"),
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
                attributes=r.get("attributes") or {},
            )
            for r in rows[:limit]
        ]

    @_synchronized
    def count_episodes(self, *, session_id: str, owner_user_id: str | None = None) -> int:
        sql = "SELECT COUNT(*) AS n FROM memory_episodes WHERE session_id = ?"
        params: list[Any] = [session_id]
        clause, extra = self._owner_clause(owner_user_id)
        sql += clause
        params += extra
        row = self.conn.execute(sql, params).fetchone()
        return int(row["n"] if row else 0)

    @_synchronized
    def forget(
        self, *, subject: str = "", session_id: str = "", owner_user_id: str | None = None
    ) -> int:
        if not subject and not session_id:
            raise ValueError("forget() needs a subject or a session_id")
        removed = 0
        for table, column in (("memory_facts", "subject"), ("memory_episodes", "subject")):
            if subject:
                sql = f"DELETE FROM {table} WHERE {column} = ?"
                params: list[Any] = [subject]
                clause, extra = self._owner_clause(owner_user_id)
                sql += clause
                params += extra
                removed += self.conn.execute(sql, params).rowcount
        if session_id:
            sql = "DELETE FROM memory_episodes WHERE session_id = ?"
            params = [session_id]
            clause, extra = self._owner_clause(owner_user_id)
            sql += clause
            params += extra
            removed += self.conn.execute(sql, params).rowcount
        self.conn.commit()
        return removed

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error:
            pass
