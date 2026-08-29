#!/usr/bin/env python3
"""Idempotent SQLite schema helper for chat.db and children.db.

Ensures chat.db has sessions.summary + session_facts, and children.db
has the full ChildMemoryDB schema. Safe to run on every container start.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHAT = ROOT / "data" / "children" / "chat.db"
DEFAULT_CHILDREN = ROOT / "data" / "children" / "children.db"

CHAT_SCHEMA = """
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

CHILDREN_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  user_id TEXT PRIMARY KEY,
  username TEXT NOT NULL UNIQUE COLLATE NOCASE,
  password_hash TEXT NOT NULL,
  created_at TEXT
);
CREATE TABLE IF NOT EXISTS children (
  child_id TEXT PRIMARY KEY,
  name TEXT,
  sex TEXT,
  date_of_birth TEXT,
  gestational_age_weeks REAL,
  notes TEXT,
  created_at TEXT,
  updated_at TEXT,
  owner_user_id TEXT
);
CREATE TABLE IF NOT EXISTS growth_measurements (
  id TEXT PRIMARY KEY,
  child_id TEXT,
  weeks REAL,
  measure TEXT,
  value REAL,
  z_score REAL,
  centile REAL,
  track_status TEXT,
  recorded_at TEXT,
  FOREIGN KEY(child_id) REFERENCES children(child_id)
);
CREATE TABLE IF NOT EXISTS screening_sessions (
  id TEXT PRIMARY KEY,
  child_id TEXT,
  instrument TEXT,
  age_months INTEGER,
  answers_json TEXT,
  result_json TEXT,
  recorded_at TEXT,
  FOREIGN KEY(child_id) REFERENCES children(child_id)
);
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  child_id TEXT,
  kind TEXT,
  summary TEXT,
  payload_json TEXT,
  recorded_at TEXT,
  FOREIGN KEY(child_id) REFERENCES children(child_id)
);
"""


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, decl: str
) -> bool:
    cols = _table_columns(conn, table)
    if column in cols:
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    return True


def migrate_chat_db(path: Path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    actions: list[str] = []
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(CHAT_SCHEMA)
        actions.append("chat schema ensured")
        if _add_column_if_missing(conn, "sessions", "summary", "TEXT"):
            actions.append("sessions.summary added")
        conn.commit()
    finally:
        conn.close()
    return actions


def migrate_children_db(path: Path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    actions: list[str] = []
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(CHILDREN_SCHEMA)
        actions.append("children schema ensured")
        # Upgrade path for databases created before multi-user support.
        if _add_column_if_missing(conn, "children", "owner_user_id", "TEXT"):
            actions.append("children.owner_user_id added")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_children_owner ON children(owner_user_id)"
        )
        conn.commit()
    finally:
        conn.close()
    return actions


def migrate(
    chat_path: Path | None = None,
    children_path: Path | None = None,
) -> list[str]:
    actions: list[str] = []
    actions.extend(migrate_chat_db(Path(chat_path or DEFAULT_CHAT)))
    actions.extend(migrate_children_db(Path(children_path or DEFAULT_CHILDREN)))
    return actions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat-db", type=Path, default=DEFAULT_CHAT)
    parser.add_argument("--children-db", type=Path, default=DEFAULT_CHILDREN)
    args = parser.parse_args(argv)
    for line in migrate(args.chat_db, args.children_db):
        print(f"[migrate_db] {line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
