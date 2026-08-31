"""The child's profile as a graph.

Semantic memory kept as flat lines answers "what do I know about this child?"
well enough, and answers "which hospital treated her ulcer?" only by hoping
the two words landed in the same sentence. A graph answers it by walking
child -> has_condition -> ulcer -> treated_at -> Mehr hospital, which is what
the parent actually asked.

Deliberately not Microsoft GraphRAG. That is a batch pipeline for a static
corpus: it costs tens of dollars per million tokens because indexing is
dominated by LLM calls, and Microsoft describe the project as in maintenance
mode. A profile graph is the opposite shape -- a few dozen nodes per child,
changing a little on every turn -- so it is built incrementally here, in the
SQLite this project already runs, at the cost of the extraction call that
consolidation was making anyway.

Nodes and edges are typed by data, not by an enumeration in code: the types
come from config/memory.yaml so a new kind of relationship can be recognised
without a release.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from assistant.memory.types import new_id, utc_now

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_nodes (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    owner_user_id TEXT,
    label TEXT NOT NULL,
    type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    attributes TEXT DEFAULT '{}',
    UNIQUE(subject, owner_user_id, type, label)
);
CREATE INDEX IF NOT EXISTS idx_nodes_subject ON graph_nodes(subject, owner_user_id);

CREATE TABLE IF NOT EXISTS graph_edges (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    owner_user_id TEXT,
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    relation TEXT NOT NULL,
    created_at TEXT NOT NULL,
    valid_to TEXT,
    fact_id TEXT,
    attributes TEXT DEFAULT '{}',
    UNIQUE(subject, owner_user_id, src, relation, dst)
);
CREATE INDEX IF NOT EXISTS idx_edges_subject ON graph_edges(subject, owner_user_id);
CREATE INDEX IF NOT EXISTS idx_edges_src ON graph_edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON graph_edges(dst);
"""


def _synchronized(fn):
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _norm(text: str) -> str:
    return " ".join((text or "").split()).strip()


class ProfileGraph:
    """Entities and relationships for one child, per account."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # -- writing ----------------------------------------------------------
    @_synchronized
    def upsert_node(
        self,
        *,
        subject: str,
        label: str,
        type: str,
        owner_user_id: str | None = None,
        attributes: dict | None = None,
    ) -> str:
        """Add an entity, or return the existing one.

        Labels are matched case-insensitively so "Mehr hospital" and "Mehr
        Hospital" are one clinic rather than two.
        """
        label = _norm(label)
        if not label or not subject:
            return ""
        key = label.lower()
        row = self.conn.execute(
            "SELECT id FROM graph_nodes WHERE subject=? AND type=? AND lower(label)=?"
            " AND (owner_user_id IS ? OR owner_user_id = ?)",
            (subject, type, key, owner_user_id, owner_user_id),
        ).fetchone()
        if row:
            return row["id"]
        node_id = new_id()
        self.conn.execute(
            "INSERT INTO graph_nodes(id,subject,owner_user_id,label,type,created_at,attributes)"
            " VALUES(?,?,?,?,?,?,?)",
            (
                node_id,
                subject,
                owner_user_id,
                label,
                type,
                utc_now(),
                json.dumps(attributes or {}),
            ),
        )
        self.conn.commit()
        return node_id

    @_synchronized
    def add_edge(
        self,
        *,
        subject: str,
        src: str,
        relation: str,
        dst: str,
        owner_user_id: str | None = None,
        fact_id: str | None = None,
        attributes: dict | None = None,
    ) -> str:
        if not (src and dst and relation):
            return ""
        existing = self.conn.execute(
            "SELECT id FROM graph_edges WHERE subject=? AND src=? AND relation=? AND dst=?",
            (subject, src, relation, dst),
        ).fetchone()
        if existing:
            return existing["id"]
        edge_id = new_id()
        self.conn.execute(
            "INSERT INTO graph_edges"
            "(id,subject,owner_user_id,src,dst,relation,created_at,valid_to,fact_id,attributes)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                edge_id,
                subject,
                owner_user_id,
                src,
                dst,
                relation,
                utc_now(),
                None,
                fact_id,
                json.dumps(attributes or {}),
            ),
        )
        self.conn.commit()
        return edge_id

    @_synchronized
    def retract_edge(self, edge_id: str, *, at: str | None = None) -> bool:
        """Close an edge in time rather than deleting it.

        A child who changes clinic has not un-attended the first one, so the
        edge stops being current while the history stays walkable.
        """
        cur = self.conn.execute(
            "UPDATE graph_edges SET valid_to=? WHERE id=? AND valid_to IS NULL",
            (at or utc_now(), edge_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    # -- reading ----------------------------------------------------------
    @_synchronized
    def nodes(
        self, *, subject: str, owner_user_id: str | None = None, type: str | None = None
    ) -> list[dict]:
        sql = "SELECT * FROM graph_nodes WHERE subject=?"
        params: list[Any] = [subject]
        if owner_user_id is not None:
            sql += " AND (owner_user_id = ? OR owner_user_id IS NULL)"
            params.append(owner_user_id)
        if type:
            sql += " AND type=?"
            params.append(type)
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    @_synchronized
    def edges(
        self,
        *,
        subject: str,
        owner_user_id: str | None = None,
        include_retracted: bool = False,
    ) -> list[dict]:
        sql = "SELECT * FROM graph_edges WHERE subject=?"
        params: list[Any] = [subject]
        if owner_user_id is not None:
            sql += " AND (owner_user_id = ? OR owner_user_id IS NULL)"
            params.append(owner_user_id)
        if not include_retracted:
            sql += " AND valid_to IS NULL"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def neighbourhood(
        self,
        *,
        subject: str,
        seeds: Iterable[str],
        owner_user_id: str | None = None,
        hops: int = 2,
    ) -> list[dict]:
        """Walk out from the nodes a question mentioned.

        Two hops is what makes "which hospital treated her ulcer?" work: one
        hop reaches the ulcer, the second reaches the clinic linked to it. The
        limit exists because a third hop on a small profile graph reaches
        everything and stops being a filter at all.
        """
        node_rows = {n["id"]: n for n in self.nodes(subject=subject, owner_user_id=owner_user_id)}
        edge_rows = self.edges(subject=subject, owner_user_id=owner_user_id)
        frontier = {n for n in seeds if n in node_rows}
        seen = set(frontier)
        walked: list[dict] = []
        for _ in range(max(0, hops)):
            nxt: set[str] = set()
            for edge in edge_rows:
                for a, b in ((edge["src"], edge["dst"]), (edge["dst"], edge["src"])):
                    if a in frontier and b not in seen:
                        nxt.add(b)
                    if a in frontier and edge not in walked:
                        walked.append(edge)
            if not nxt:
                break
            seen |= nxt
            frontier = nxt
        out = []
        for edge in walked:
            src = node_rows.get(edge["src"])
            dst = node_rows.get(edge["dst"])
            if src and dst:
                out.append(
                    {
                        "src": src["label"],
                        "src_type": src["type"],
                        "relation": edge["relation"],
                        "dst": dst["label"],
                        "dst_type": dst["type"],
                        "fact_id": edge["fact_id"],
                    }
                )
        return out

    def match_nodes(
        self, *, subject: str, text: str, owner_user_id: str | None = None
    ) -> list[str]:
        """Node ids whose label appears in the text.

        Substring matching on the label, not a vocabulary of known conditions:
        the graph learns its own entities, so the matcher must not be limited
        to ones written down in advance.
        """
        low = _norm(text).lower()
        if not low:
            return []
        hits = []
        for node in self.nodes(subject=subject, owner_user_id=owner_user_id):
            label = (node["label"] or "").lower()
            if len(label) >= 3 and label in low:
                hits.append(node["id"])
        return hits

    def render(
        self,
        *,
        subject: str,
        text: str,
        owner_user_id: str | None = None,
        budget_chars: int | None = None,
        hops: int = 2,
    ) -> str:
        """The relevant part of the profile graph, as readable lines."""
        seeds = self.match_nodes(subject=subject, text=text, owner_user_id=owner_user_id)
        if not seeds:
            return ""
        lines = []
        used = 0
        for rel in self.neighbourhood(
            subject=subject, seeds=seeds, owner_user_id=owner_user_id, hops=hops
        ):
            line = f"- {rel['src']} {rel['relation'].replace('_', ' ')} {rel['dst']}"
            if budget_chars is not None and used + len(line) + 1 > budget_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    @_synchronized
    def forget(self, *, subject: str, owner_user_id: str | None = None) -> int:
        removed = 0
        for table in ("graph_edges", "graph_nodes"):
            sql = f"DELETE FROM {table} WHERE subject=?"
            params: list[Any] = [subject]
            if owner_user_id is not None:
                sql += " AND (owner_user_id = ? OR owner_user_id IS NULL)"
                params.append(owner_user_id)
            removed += self.conn.execute(sql, params).rowcount
        self.conn.commit()
        return removed

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error:
            pass
