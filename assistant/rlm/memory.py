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
        """Save what the parent's latest message states, for later conversations.

        The model is asked one plain thing: list the facts this message states,
        and for each, which saved fact (if any) it makes out of date. Asked
        instead to edit the store (add/update/delete against facts shown first),
        a small model copied the saved facts back as new ones and left out the
        one fact the message added. Duplicates and corrections are handled here.
        """
        s = get_settings()
        if not s.rlm_memory_enabled or not user_id:
            return
        if len((user_text or "").strip()) < int(s.rlm_memory_fact_min_user_chars):
            return
        saved = self.list_facts(user_id)
        shown = saved[: int(s.rlm_memory_always_recall_facts)]
        prompt = render(
            get_prompt("memory/fact_distill.md"),
            known_facts="\n".join(f"[{k['id']}] {k['text']}" for k in shown) or "(none)",
            user_message=clip_chars(user_text, 6000),
        )
        out = ""
        schema = _fact_list_schema(s, [k["id"] for k in shown])
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
        _ = assistant_text  # the assistant's words are never a source of facts
        shown_ids = {k["id"] for k in shown}
        # Words the saved facts already establish (the child's name, the family)
        # say nothing about whether a new fact came from this message.
        established = set().union(*(_content_words(k["text"]) for k in saved)) if saved else set()
        added = 0
        for item in _parse_fact_list(out):
            text = str(item.get("text") or "").strip()
            if not _fact_ok(text, s) or text == (user_text or "").strip():
                continue
            if s.rlm_memory_fact_require_user_grounding and not _grounded(text, user_text, established):
                continue
            changes = str(item.get("changes") or "")
            target = next((k for k in saved if k.get("id") == changes), None)
            # A correction replaces a saved fact only when the two are about the
            # same thing. Linked to an unrelated fact ("now stands" said to make
            # "started crawling" out of date), the old fact would be lost; the
            # new one is kept beside it instead.
            if changes in shown_ids and target and _same_subject(text, target["text"], saved):
                self._update(changes, text, user_id)
                for k in saved:
                    if k["id"] == changes:
                        k["text"] = text
                continue
            if added >= int(s.rlm_memory_fact_max_adds_per_turn) or _already_known(text, saved):
                continue
            self._add(user_id, None, text, _category_of(text))
            saved.append({"id": "", "text": text})
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


def _fact_list_schema(s, known_ids: list[str]) -> dict:
    """The shape of a distilled memory decision: the facts a message states,
    each with the saved fact it makes out of date, if any."""
    item = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "maxLength": int(s.rlm_memory_fact_max_chars)},
            "changes": {"type": "string", "enum": [*known_ids, ""]},
        },
        "required": ["text", "changes"],
    }
    return {
        "type": "object",
        "properties": {"facts": {"type": "array", "items": item, "maxItems": int(s.rlm_memory_fact_max_adds_per_turn) + 3}},
        "required": ["facts"],
    }


def _category_of(text: str) -> str:
    """A coarse label for listing; the fact text is what recall uses."""
    return "general"


def _grounded(fact: str, user_text: str, established: set[str] | None = None) -> bool:
    """Whether a fact plausibly came from the parent's own words.

    Words the saved facts already establish (the child's name above all) are
    set aside: a parent says "she", not the name, and the check falls on what
    is new. At least one of the remaining words must appear in the message,
    or two when nothing is set aside.
    """
    words = _content_words(fact) - (established or set())
    if not words:
        return True
    hay = (user_text or "").lower()
    hits = sum(1 for w in words if w in hay)
    return hits >= (1 if established else min(2, len(words)))


def _content_words(text: str) -> set[str]:
    from assistant.memory.bm25 import tokenize

    return {w for w in tokenize(text) if len(w) > 2}


def _subject_words(text: str, saved: list[dict]) -> set[str]:
    """Content words of `text` that say what it is about: words shared by most
    saved facts (the child's name, "has") are set aside."""
    words = _content_words(text)
    if len(saved) < 2:
        return words
    common = {
        w for w in words
        if sum(1 for k in saved if w in _content_words(str(k.get("text") or ""))) > len(saved) / 2
    }
    return words - common


def _same_subject(new: str, old: str, saved: list[dict]) -> bool:
    return bool(_subject_words(new, saved) & _subject_words(old, saved))


def _already_known(text: str, saved: list[dict], overlap: float = 0.8) -> bool:
    """Whether `text` says what a saved fact already says.

    Measured on content words, so "Darya started crawling last week" and
    "Darya started crawling last week." match, as does a reordering, while a
    fact that adds something new ("... and pulls to stand") does not.
    """
    new = _content_words(text)
    if not new:
        return True
    for row in saved:
        old = _content_words(str(row.get("text") or ""))
        # Containment, not overlap: "Darya still has mild eczema" is already in
        # "Darya has mild eczema and is breastfed ...", though the two differ in length.
        if old and len(new & old) / len(new) >= overlap:
            return True
    return False


def _parse_fact_list(raw: str) -> list[dict]:
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
