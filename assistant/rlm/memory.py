"""User long-term memory with embeddings. Short-term vectors are intentionally disabled."""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from assistant.rlm.prompts import get_prompt, render
from assistant.rlm.tokens import clip_chars
from assistant.settings import get_settings

log = logging.getLogger(__name__)

_embedder = None
_embedder_lock = threading.Lock()


def _db_path() -> Path:
    s = get_settings()
    if s.rlm_memory_db:
        return Path(s.rlm_memory_db)
    data = Path(s.nestling_data_dir) if s.nestling_data_dir else Path(s.nestling_root) / "data"
    return data / "children" / "rlm_ltm.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ltm_facts (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            workspace_id TEXT,
            category TEXT,
            text TEXT NOT NULL,
            embedding BLOB,
            created_at REAL,
            updated_at REAL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ltm_user ON ltm_facts(user_id)"
    )
    conn.commit()
    return conn


def _get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    with _embedder_lock:
        if _embedder is not None:
            return _embedder
        s = get_settings()
        model_id = s.nestling_embedding_model
        # Prefer remote OpenAI-compatible embeddings when an embed URL is set
        # (e.g. the local bge-m3 vLLM on :8101). Fall back to sentence-transformers.
        if s.rlm_embed_url:
            _embedder = ("remote", s.rlm_embed_url.rstrip("/"), s.rlm_embed_model or model_id)
            return _embedder
        try:
            from sentence_transformers import SentenceTransformer

            _embedder = ("local", SentenceTransformer(model_id), model_id)
            return _embedder
        except Exception as exc:
            log.warning("LTM embedder unavailable: %s", exc)
            _embedder = ("none", None, model_id)
            return _embedder


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    kind, obj, model = _get_embedder()
    if kind == "none" or not texts:
        return None
    if kind == "local":
        vecs = obj.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]
    # remote OpenAI-compatible /v1/embeddings
    import urllib.request

    s = get_settings()
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        f"{obj}/v1/embeddings",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {s.nestling_llm_api_key or 'EMPTY'}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(s.nestling_dense_load_timeout)) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        data = sorted(body.get("data") or [], key=lambda d: d.get("index", 0))
        return [list(map(float, d["embedding"])) for d in data]
    except Exception as exc:
        log.warning("remote embed failed: %s", exc)
        return None


def _pack(vec: list[float]) -> bytes:
    return json.dumps(vec).encode("utf-8")


def _unpack(blob: bytes | None) -> list[float] | None:
    if not blob:
        return None
    try:
        return list(map(float, json.loads(blob.decode("utf-8"))))
    except Exception:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


class LongTermMemory:
    """Embeddings only for long-term user facts. No short-term vector chunks."""

    def __init__(self) -> None:
        self._conn = _connect()
        self._lock = threading.Lock()

    def recall(self, query: str, user_id: str, workspace_id: str | None = None) -> dict[str, Any]:
        s = get_settings()
        if not s.rlm_memory_enabled or not user_id:
            return {"block": "", "items": [], "offline": False}
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT id, text, category, workspace_id, embedding FROM ltm_facts WHERE user_id = ? ORDER BY updated_at DESC",
                    (user_id,),
                ).fetchall()
            always = int(s.rlm_memory_always_recall_facts)
            limit = int(s.rlm_memory_user_limit)
            min_score = float(s.rlm_memory_min_score)
            q_words = {w.lower() for w in re.findall(r"[A-Za-z\u0600-\u06FF]{4,}", query or "")}
            items: list[dict] = []
            seen: set[str] = set()

            def add(row, score: float) -> None:
                rid = str(row["id"])
                if rid in seen:
                    return
                seen.add(rid)
                items.append({"id": rid, "text": row["text"], "score": score})

            vecs = embed_texts([query]) if query else None
            if vecs:
                qv = vecs[0]
                scored: list[tuple[float, sqlite3.Row]] = []
                for row in rows:
                    ws = row["workspace_id"]
                    if s.rlm_memory_long_term_scope == "workspace" and ws and workspace_id and ws != workspace_id:
                        continue
                    ev = _unpack(row["embedding"])
                    if not ev:
                        continue
                    scored.append((_cosine(qv, ev), row))
                scored.sort(key=lambda x: x[0], reverse=True)
                for score, row in scored:
                    if score < min_score:
                        continue
                    add(row, score)
                    if len(items) >= limit:
                        break
            for row in rows:
                if len(items) >= limit:
                    break
                text = str(row["text"] or "")
                words = {w.lower() for w in re.findall(r"[A-Za-z\u0600-\u06FF]{4,}", text)}
                if q_words and words and len(q_words & words) >= 1:
                    add(row, 0.5)
            if not items or len(rows) <= always:
                for row in rows:
                    add(row, 1.0)
                    if len(items) >= max(limit, always if len(rows) <= always else limit):
                        break
            block = self.render_block(items[:limit], int(s.rlm_memory_prompt_max_chars))
            return {"block": block, "items": items[:limit], "offline": not bool(vecs)}
        except Exception as exc:
            log.warning("LTM recall offline: %s", exc)
            return {"block": "", "items": [], "offline": True}

    @staticmethod
    def render_block(items: list[dict], max_chars: int) -> str:
        if not items:
            return ""
        lines = ["Known about the user (long-term memory):"]
        for it in items:
            lines.append(f"- {it['text']}")
        return clip_chars("\n".join(lines), max_chars)

    def learn_facts(
        self,
        *,
        user_id: str,
        workspace_id: str | None,
        user_text: str,
        assistant_text: str,
        complete_fn,
    ) -> None:
        s = get_settings()
        if not s.rlm_memory_enabled or not user_id:
            return
        if len((user_text or "").strip()) < int(s.rlm_memory_fact_min_user_chars):
            return
        recalled = self.recall(user_text, user_id, workspace_id)
        known = recalled.get("items") or []
        known_text = (
            "\n".join(f"[{k['id']}] (user) {k['text']}" for k in known) if known else "(none)"
        )
        prompt = render(
            get_prompt("memory/fact_distill.md"),
            known_facts=known_text,
            project=workspace_id or "(no project folder open)",
            user_message=clip_chars(user_text, 6000),
            assistant_message=clip_chars(assistant_text, 1500),
            require_grounding="yes" if s.rlm_memory_fact_require_user_grounding else "no",
        )
        out = ""
        schema = _fact_ops_schema(s, [k["id"] for k in known])
        try:
            try:
                out = complete_fn(
                    [{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0.0,
                    json_schema=schema,
                )
            except TypeError:
                # A completion function that cannot constrain its output.
                out = complete_fn(
                    [{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0.0,
                )
        except Exception as exc:
            log.warning("fact distill failed: %s", exc)
        ops = _parse_fact_ops(out)
        known_ids = {k["id"] for k in known}
        added = 0
        for op in ops:
            if op.get("op") == "delete" and op.get("id") in known_ids:
                self._delete(op["id"], user_id)
            elif op.get("op") == "update" and op.get("id") in known_ids and _fact_ok(op.get("text") or "", s):
                self._update(op["id"], op["text"], user_id)
            elif op.get("op") == "add":
                if added >= int(s.rlm_memory_fact_max_adds_per_turn):
                    continue
                text = (op.get("text") or "").strip()
                if not _fact_ok(text, s):
                    continue
                if text.strip() == (user_text or "").strip():
                    continue
                if s.rlm_memory_fact_require_user_grounding and not _grounded(text, user_text):
                    continue
                about = op.get("about") or "user"
                ws = workspace_id if about == "project" else None
                if about == "project" and not workspace_id:
                    continue
                self._add(user_id, ws, text, op.get("category") or "general")
                added += 1

    def _add(self, user_id: str, workspace_id: str | None, text: str, category: str) -> None:
        vecs = embed_texts([text])
        blob = _pack(vecs[0]) if vecs else None
        now = time.time()
        fid = f"fact_{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._conn.execute(
                "INSERT INTO ltm_facts(id,user_id,workspace_id,category,text,embedding,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (fid, user_id, workspace_id, category, text, blob, now, now),
            )
            self._conn.commit()

    def _update(self, fid: str, text: str, user_id: str) -> None:
        vecs = embed_texts([text])
        blob = _pack(vecs[0]) if vecs else None
        with self._lock:
            if blob is not None:
                self._conn.execute(
                    "UPDATE ltm_facts SET text=?, embedding=?, updated_at=? WHERE id=? AND user_id=?",
                    (text, blob, time.time(), fid, user_id),
                )
            else:
                self._conn.execute(
                    "UPDATE ltm_facts SET text=?, updated_at=? WHERE id=? AND user_id=?",
                    (text, time.time(), fid, user_id),
                )
            self._conn.commit()

    def _delete(self, fid: str, user_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM ltm_facts WHERE id=? AND user_id=?",
                (fid, user_id),
            )
            self._conn.commit()

    def list_facts(self, user_id: str) -> list[dict]:
        if not user_id:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, text, category FROM ltm_facts WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [{"id": r["id"], "text": r["text"], "category": r["category"]} for r in rows]

    def remember_tool_result(self, user_id: str, tool_id: str, result: Any) -> None:
        """Persist scalar clinical tool results so compaction cannot drop them."""
        s = get_settings()
        if not s.rlm_memory_enabled or not user_id or not isinstance(result, dict):
            return
        # A call that failed found nothing out; its error text is not a fact.
        if result.get("ok") is False or result.get("error"):
            return
        skip = {"ok", "overlay", "overlay_filename", "error"}
        prefer = ("centile", "percentile", "z_score", "summary", "measure", "sex", "weeks", "value")
        parts = []
        for key in prefer:
            val = result.get(key)
            if key in skip or val in (None, "", [], {}):
                continue
            if isinstance(val, (int, float, str, bool)):
                parts.append(f"{key}={val}")
        for key, val in result.items():
            if key in prefer or key in skip or val in (None, "", [], {}):
                continue
            if isinstance(val, (int, float, str, bool)):
                parts.append(f"{key}={val}")
            if len(parts) >= 8:
                break
        if not parts:
            return
        cent = result.get("centile")
        if cent is None:
            cent = result.get("percentile")
        if isinstance(cent, (int, float)):
            text = f"The child's last {result.get('measure') or 'growth'} centile was {float(cent):.0f}."
        else:
            text = clip_chars(f"Tool {tool_id} last result: " + ", ".join(parts), int(s.rlm_memory_fact_max_chars))
        existing = self.recall(text, user_id).get("items") or []
        if any(text.lower() in str(it.get("text") or "").lower() or str(it.get("text") or "").lower() in text.lower() for it in existing[:6]):
            return
        self._add(user_id, None, text, "clinical")

    def count_stm_vectors(self) -> int:
        """Always 0 — short-term vectors are not stored."""
        return 0


def _fact_ok(text: str, s) -> bool:
    t = (text or "").strip()
    return bool(t) and len(t) <= int(s.rlm_memory_fact_max_chars)


def _fact_ops_schema(s, known_ids: list[str]) -> dict:
    """The shape a distilled memory decision must take.

    What is worth keeping is the model's judgement; the shape is not. Before
    this the reply was free text hunted for a JSON array, and when the model
    wrote none a regex kept whatever followed the word "remember", which
    stored the tail of a question ("about Darya from before?") as a fact.
    """
    op = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["add", "update", "delete"]},
            "id": {"type": "string", "enum": [*known_ids, ""]},
            "text": {"type": "string", "maxLength": int(s.rlm_memory_fact_max_chars)},
            "category": {"type": "string", "enum": ["identity", "health", "feeding", "sleep", "development", "routine", "preference", "general"]},
        },
        "required": ["op", "id", "text", "category"],
    }
    return {
        "type": "object",
        "properties": {
            "facts": {"type": "array", "items": op, "maxItems": int(s.rlm_memory_fact_max_adds_per_turn) + 2}
        },
        "required": ["facts"],
    }


def _grounded(fact: str, user_text: str) -> bool:
    # Light grounding: at least one content word from the fact appears in the user text.
    words = [w.lower() for w in re.findall(r"[A-Za-z\u0600-\u06FF]{4,}", fact)]
    hay = (user_text or "").lower()
    if not words:
        return True
    hits = sum(1 for w in words[:8] if w in hay)
    return hits >= min(2, len(words[:8]))


def _parse_fact_ops(raw: str) -> list[dict]:
    text = (raw or "").strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    data = None
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        m = re.search(pattern, text)
        if not m:
            continue
        try:
            data = json.loads(m.group(0))
            break
        except json.JSONDecodeError:
            continue
    if isinstance(data, dict):
        data = data.get("facts")
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]
