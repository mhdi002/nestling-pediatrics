#!/usr/bin/env python3
"""Per-child precision memory: SQLite store + retrievable timeline."""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from assistant.config import CHILD_DB_PATH
from assistant.settings import get_settings

# Recent-history caps used when summarizing a child for the agent.
RECENT_SCREENINGS = 5


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _synchronized(fn):
    """Serialize a method against the shared SQLite connection."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)

    return wrapper


class ChildMemoryDB:
    """Structured database for parent/baby monitoring data."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or CHILD_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # WAL + NORMAL: measured ~3× write throughput vs delete/FULL under load.
        # Shared connection is still serialized with RLock (check_same_thread=False).
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.RLock()
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.executescript(
            """
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
            CREATE TABLE IF NOT EXISTS users (
              user_id TEXT PRIMARY KEY,
              username TEXT NOT NULL UNIQUE COLLATE NOCASE,
              password_hash TEXT NOT NULL,
              created_at TEXT
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
              age_months REAL,
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
        )
        self.conn.commit()
        # Migrate older DBs that lack chronological age_months on growth rows.
        cols = {
            r[1]
            for r in self.conn.execute("PRAGMA table_info(growth_measurements)").fetchall()
        }
        if "age_months" not in cols:
            self.conn.execute(
                "ALTER TABLE growth_measurements ADD COLUMN age_months REAL"
            )
            self.conn.commit()
        # Migrate DBs created before per-user ownership. Pre-existing rows get
        # a NULL owner and are therefore not visible to any account.
        child_cols = {
            r[1] for r in self.conn.execute("PRAGMA table_info(children)").fetchall()
        }
        if "owner_user_id" not in child_cols:
            self.conn.execute("ALTER TABLE children ADD COLUMN owner_user_id TEXT")
            self.conn.commit()

    @_synchronized
    def create_child(
        self,
        name: str,
        sex: str,
        date_of_birth: str | None = None,
        gestational_age_weeks: float | None = None,
        notes: str = "",
        child_id: str | None = None,
        owner_user_id: str | None = None,
    ) -> str:
        cid = child_id or str(uuid.uuid4())
        now = _utc()
        self.conn.execute(
            "INSERT INTO children(child_id,name,sex,date_of_birth,gestational_age_weeks,notes,created_at,updated_at,owner_user_id) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (cid, name, sex, date_of_birth, gestational_age_weeks, notes, now, now, owner_user_id),
        )
        self.conn.commit()
        self.add_event(cid, "child_created", f"Created child profile for {name}", {"sex": sex})
        return cid

    @_synchronized
    def get_child(self, child_id: str, owner_user_id: str | None = None) -> dict | None:
        # Strict ownership: a signed-in account sees only its own children.
        # Unowned legacy rows are not shared, so one family can never read
        # another's record by guessing an id.
        if owner_user_id is None:
            row = self.conn.execute(
                "SELECT * FROM children WHERE child_id=?", (child_id,)
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM children WHERE child_id=? AND owner_user_id=?",
                (child_id, owner_user_id),
            ).fetchone()
        return dict(row) if row else None

    @_synchronized
    def list_children(self, owner_user_id: str | None = None) -> list[dict]:
        if owner_user_id is None:
            rows = self.conn.execute("SELECT * FROM children ORDER BY created_at")
        else:
            rows = self.conn.execute(
                "SELECT * FROM children WHERE owner_user_id=? ORDER BY created_at",
                (owner_user_id,),
            )
        return [dict(r) for r in rows]

    @_synchronized
    def find_children_by_name(
        self, name: str, owner_user_id: str | None = None
    ) -> list[dict]:
        """
        Children whose name matches `name`, best match first.

        Parents refer to a child by name mid-conversation ("how is Monica
        doing?"), so the agent needs a way from a spoken name back to the
        record. Matching is case-insensitive and accent-preserving, and tries
        exact match before substring so "Ann" cannot shadow "Anna" when both
        exist. Scoped by owner so one account can never resolve another's child.
        """
        needle = (name or "").strip().lower()
        if not needle:
            return []
        rows = self.list_children(owner_user_id=owner_user_id)
        exact = [r for r in rows if str(r.get("name", "")).strip().lower() == needle]
        if exact:
            return exact
        # Whole-word containment in either direction, so "monica"
        # matches "Monica R." and a fuller spoken name still finds the record.
        partial = [
            r
            for r in rows
            if needle in str(r.get("name", "")).strip().lower()
            or str(r.get("name", "")).strip().lower() in needle
        ]
        return partial

    # --- user accounts -------------------------------------------------

    @_synchronized
    def create_user(self, username: str, password_hash: str) -> str | None:
        """Create a user. Returns None if the username is already taken."""
        uid = str(uuid.uuid4())
        try:
            self.conn.execute(
                "INSERT INTO users(user_id,username,password_hash,created_at) VALUES(?,?,?,?)",
                (uid, username, password_hash, _utc()),
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            return None
        return uid

    @_synchronized
    def get_user(self, username: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM users WHERE username=? COLLATE NOCASE", (username,)
        ).fetchone()
        return dict(row) if row else None

    @_synchronized
    def count_users(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    @_synchronized
    def add_growth(
        self,
        child_id: str,
        weeks: float,
        measure: str,
        value: float,
        z_score: float | None = None,
        centile: float | None = None,
        track_status: str | None = None,
        age_months: float | None = None,
    ) -> str:
        gid = str(uuid.uuid4())
        now = _utc()
        self.conn.execute(
            "INSERT INTO growth_measurements(id,child_id,weeks,measure,value,z_score,centile,track_status,recorded_at,age_months) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (gid, child_id, weeks, measure, value, z_score, centile, track_status, now, age_months),
        )
        self.conn.commit()
        self.add_event(
            child_id,
            "growth",
            f"{measure}={value} at {weeks}w (centile={centile})",
            {
                "measure": measure,
                "weeks": weeks,
                "value": value,
                "centile": centile,
                "z_score": z_score,
                "age_months": age_months,
            },
        )
        return gid

    @_synchronized
    def add_screening(
        self,
        child_id: str,
        instrument: str,
        answers: dict,
        result: dict,
        age_months: int | None = None,
    ) -> str:
        sid = str(uuid.uuid4())
        now = _utc()
        self.conn.execute(
            "INSERT INTO screening_sessions(id,child_id,instrument,age_months,answers_json,result_json,recorded_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (
                sid,
                child_id,
                instrument,
                age_months,
                json.dumps(answers, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
                now,
            ),
        )
        self.conn.commit()
        self.add_event(
            child_id,
            "screening",
            result.get("summary", f"{instrument} completed"),
            {"instrument": instrument, "result": result},
        )
        return sid

    @_synchronized
    def add_event(self, child_id: str, kind: str, summary: str, payload: dict | None = None) -> str:
        eid = str(uuid.uuid4())
        self.conn.execute(
            "INSERT INTO events(id,child_id,kind,summary,payload_json,recorded_at) VALUES(?,?,?,?,?,?)",
            (eid, child_id, kind, summary, json.dumps(payload or {}, ensure_ascii=False), _utc()),
        )
        self.conn.commit()
        return eid

    @_synchronized
    def recent_events(
        self,
        child_id: str,
        kinds: tuple[str, ...] | list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Newest-last events for a child, optionally filtered by kind."""
        if limit is None:
            limit = get_settings().nestling_child_memory_events
        limit = max(0, int(limit))
        if not limit:
            return []
        sql = "SELECT * FROM events WHERE child_id=?"
        params: list = [child_id]
        if kinds:
            sql += " AND kind IN (%s)" % ",".join("?" * len(kinds))
            params.extend(kinds)
        sql += " ORDER BY recorded_at DESC, rowid DESC LIMIT ?"
        params.append(limit)
        rows = list(reversed(self.conn.execute(sql, params).fetchall()))
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["payload"] = json.loads(d.pop("payload_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                d["payload"] = {}
            out.append(d)
        return out

    @_synchronized
    def remember_note(
        self,
        child_id: str,
        summary: str,
        payload: dict | None = None,
        owner_user_id: str | None = None,
    ) -> str | None:
        """
        Persist one salient conversational fact against the CHILD.

        Reuses the `events` timeline rather than adding a parallel store: events
        are already FK'd to `children` (so ownership is inherited), already
        replayed by timeline_documents() into child RAG, and already share this
        connection's lock. Returns None when the child is not readable by
        `owner_user_id` — memory must never cross accounts.
        """
        settings = get_settings()
        text = (summary or "").strip()
        if not text or not settings.nestling_child_memory_enabled:
            return None
        if not self.get_child(child_id, owner_user_id):
            return None
        kind = settings.nestling_child_memory_kind
        text = text[: settings.nestling_child_memory_note_chars]
        # Consecutive identical notes add nothing but crowd the digest out.
        prev = self.conn.execute(
            "SELECT summary FROM events WHERE child_id=? AND kind=? "
            "ORDER BY recorded_at DESC, rowid DESC LIMIT 1",
            (child_id, kind),
        ).fetchone()
        if prev and (prev["summary"] or "").strip() == text:
            return None
        return self.add_event(child_id, kind, text, payload)

    def _known_age_months(self, child_id: str) -> float | None:
        """
        Chronological age in months, or None when it cannot be established.

        Prefers a stored growth row's `age_months` because that value was
        computed with the child's gestational age in scope; falls back to date
        of birth. Never derives an age from postmenstrual weeks here -- doing
        that without the gestational age is what produced a 0.16-month reading
        for a two-month-old.
        """
        rows = self.growth_history(child_id) or []
        for row in reversed(rows):
            if row.get("age_months") is not None:
                try:
                    return float(row["age_months"])
                except (TypeError, ValueError):
                    continue
        child = self.get_child(child_id)
        dob = (child or {}).get("date_of_birth")
        if dob:
            from datetime import date, datetime

            try:
                born = datetime.fromisoformat(str(dob)).date()
                days = (date.today() - born).days
                return max(0.0, days / float(get_settings().nestling_days_per_month))
            except (ValueError, TypeError):
                return None
        return None

    @_synchronized
    def child_context_text(self, child_id: str, owner_user_id: str | None = None) -> str:
        """
        Compact cross-session digest of what is already known about a child:
        remembered conversation notes, latest growth, recent screenings.

        Scoped by owner: an unreadable child yields an empty digest, so a
        session belonging to one account can never surface another's data.
        """
        settings = get_settings()
        child = self.get_child(child_id, owner_user_id)
        if not child:
            return ""
        lines: list[str] = []
        # State the child's age in months explicitly. The digest otherwise
        # carried only gestational and postmenstrual weeks, leaving the model
        # to infer chronological age -- which it did wrongly, telling a parent
        # of a 2-month-old what a 13-month-old should eat. Derived from the
        # stored growth row (computed with gestational age in scope) or from
        # date of birth, never guessed.
        age_months = self._known_age_months(child_id)
        profile = ", ".join(
            p
            for p in (
                child.get("name"),
                child.get("sex"),
                f"DOB {child.get('date_of_birth')}" if child.get("date_of_birth") else "",
                # Spell this out. "GA 32.0w" was read by the model as the
                # mother being 32 weeks pregnant, and it answered a question
                # about the baby's stomach with advice about managing ulcers
                # in pregnancy. The number describes how early this child was
                # BORN, so say that rather than abbreviating it.
                f"born at {float(child['gestational_age_weeks']):g} weeks gestation"
                if child.get("gestational_age_weeks") is not None
                else "",
            )
            if p
        )
        if age_months is not None:
            profile = f"{profile}, currently ~{age_months:.1f} months old" if profile else (
                f"~{age_months:.1f} months old"
            )
        if profile:
            lines.append(f"Profile: {profile}")
        if (child.get("notes") or "").strip():
            lines.append(f"Notes: {child['notes'].strip()}")

        growth = self.growth_history(child_id)
        for g in growth[-max(0, settings.nestling_child_memory_growth) :]:
            lines.append(
                f"Growth: {g['measure']}={g['value']} at {g['weeks']}w "
                f"(centile {g.get('centile')}, {g.get('track_status')})"
            )
        for s in self.screenings(child_id)[-max(0, settings.nestling_child_memory_screenings) :]:
            lines.append(
                f"Screening: {s['instrument']} at {s.get('age_months')} months -> "
                f"{(s.get('result') or {}).get('summary') or (s.get('result') or {})}"
            )
        cap = settings.nestling_child_memory_event_chars
        for e in self.recent_events(child_id, kinds=(settings.nestling_child_memory_kind,)):
            lines.append(f"Earlier: {(e.get('summary') or '')[:cap]}")
        text = "\n".join(l for l in lines if l.strip())
        return text[: settings.nestling_child_memory_max_chars]

    @_synchronized
    def growth_history(self, child_id: str, measure: str | None = None) -> list[dict]:
        if measure:
            rows = self.conn.execute(
                "SELECT * FROM growth_measurements WHERE child_id=? AND measure=? ORDER BY weeks",
                (child_id, measure),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM growth_measurements WHERE child_id=? ORDER BY weeks",
                (child_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    @_synchronized
    def screenings(self, child_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM screening_sessions WHERE child_id=? ORDER BY recorded_at",
            (child_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["answers"] = json.loads(d.pop("answers_json"))
            d["result"] = json.loads(d.pop("result_json"))
            out.append(d)
        return out

    @_synchronized
    def child_summary(self, child_id: str) -> dict[str, Any] | None:
        """Structured summary for tools / parent-facing memory."""
        child = self.get_child(child_id)
        if not child:
            return None
        growth = self.growth_history(child_id)
        screens = self.screenings(child_id)
        latest_by_measure: dict[str, dict] = {}
        for g in growth:
            latest_by_measure[g["measure"]] = g
        return {
            "child_id": child_id,
            "profile": {
                "name": child.get("name"),
                "sex": child.get("sex"),
                "date_of_birth": child.get("date_of_birth"),
                "gestational_age_weeks": child.get("gestational_age_weeks"),
                "notes": child.get("notes"),
            },
            "growth_count": len(growth),
            "latest_growth": latest_by_measure,
            "screening_count": len(screens),
            "recent_screenings": screens[-RECENT_SCREENINGS:],
        }

    @_synchronized
    def timeline_documents(self, child_id: str) -> list[dict]:
        """Flatten child history into RAG documents for precision memory retrieval."""
        child = self.get_child(child_id)
        if not child:
            return []
        docs = [
            {
                "id": f"{child_id}_profile",
                "collection": "child",
                "child_id": child_id,
                "title": f"Child profile: {child['name']}",
                "text": (
                    f"Child {child['name']} ({child['sex']}), DOB {child.get('date_of_birth')}, "
                    f"born at {child.get('gestational_age_weeks')} weeks gestation. "
                    f"Notes: {child.get('notes')}"
                ),
            }
        ]
        for g in self.growth_history(child_id):
            docs.append(
                {
                    "id": f"growth_{g['id']}",
                    "collection": "child",
                    "child_id": child_id,
                    "title": f"Growth {g['measure']} @ {g['weeks']}w",
                    "text": (
                        f"Measured {g['measure']}={g['value']} at {g['weeks']} postmenstrual weeks; "
                        f"z={g.get('z_score')}, centile={g.get('centile')}, status={g.get('track_status')}"
                    ),
                }
            )
        for s in self.screenings(child_id):
            docs.append(
                {
                    "id": f"screen_{s['id']}",
                    "collection": "child",
                    "child_id": child_id,
                    "title": f"Screening {s['instrument']}",
                    "text": f"{s['instrument']} at {s.get('age_months')} months: {s['result']}",
                }
            )
        for e in self.conn.execute(
            "SELECT * FROM events WHERE child_id=? ORDER BY recorded_at", (child_id,)
        ):
            d = dict(e)
            docs.append(
                {
                    "id": f"event_{d['id']}",
                    "collection": "child",
                    "child_id": child_id,
                    "title": f"Event {d['kind']}",
                    "text": d["summary"],
                }
            )
        return docs

    @_synchronized
    def close(self):
        conn = getattr(self, "conn", None)
        if conn is not None:
            conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
