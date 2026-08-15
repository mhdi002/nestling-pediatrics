#!/usr/bin/env python3
"""
Dual RAG stores.

Retrieval: BM25 (no extra neural models).
Generation: local OpenAI-compatible LLM sidecar (or extractive fallback).
Tool calling elsewhere: Salesforce/xLAM-1b-fc-r (optional) or deterministic router.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from assistant.config import (
    CHILD_INDEX_DIR,
    KNOWLEDGE_DIR,
    MEDICAL_INDEX_DIR,
)
from assistant.rag.embeddings import BM25Index
from assistant.refdata import care_topics
from assistant.settings import get_settings

log = logging.getLogger(__name__)

# All care-topic keywords, patterns, age bands and ranking weights live in
# config/care_topics.yaml so clinical routing data is editable without code changes.
_CFG = care_topics()
_TOPICS: dict[str, dict] = _CFG.get("topics") or {}
_RANKING: dict = _CFG.get("ranking") or {}
_BOOSTS: dict = _RANKING.get("boosts") or {}
_PENALTIES: dict = _RANKING.get("penalties") or {}
_MATCH_TERMS: dict = _RANKING.get("match_terms") or {}
_SELECTION: dict = _CFG.get("selection") or {}
_SCAN_CHARS: dict = _SELECTION.get("scan_chars") or {}
_MESSAGES: dict = _CFG.get("messages") or {}

# Parent-facing boilerplate for the non-LLM extractive path. Topic-neutral: the
# previous wording was skin-specific and bled into speech/feeding answers.
SAFETY_TAIL = _MESSAGES["safety_tail"]
CITATION_TAIL = _MESSAGES["citation_tail"]
NO_MATCH_REPLY = _MESSAGES["no_match"]


def _kw(topic: str) -> tuple[str, ...]:
    return tuple((_TOPICS.get(topic) or {}).get("keywords") or ())


def _expansion(topic: str) -> str:
    return " ".join(((_TOPICS.get(topic) or {}).get("query_expansion") or "").split())


def _terms(name: str) -> tuple[str, ...]:
    return tuple(_MATCH_TERMS.get(name) or ())


def _selected(name: str) -> tuple[str, ...]:
    return tuple(_SELECTION.get(name) or ())


_FEEDING_KEYWORDS = _kw("feeding")
_IRON_KEYWORDS = _kw("iron")
_SLEEP_KEYWORDS = _kw("sleep")
_RASH_KEYWORDS = _kw("rash")
_MOTOR_KEYWORDS = _kw("motor")
_SPEECH_KEYWORDS = _kw("speech")
_GROWTH_EXPLAIN_KEYWORDS = _kw("growth_explain")
_SCREENING_KEYWORDS = _kw("screening")

_FEEDING_RE = re.compile((_TOPICS.get("feeding") or {})["pattern"], re.I)
_GROWTH_CENTILE_DOC_RE = re.compile(_CFG["growth_centile_doc_pattern"], re.I)
# Metadata the orchestrator appends — must not drive topic classification.
_TOPIC_META_LINE_RE = re.compile(_CFG["meta_line_pattern"])

# How much of a chunk body each filter reads. Full bodies are long enough that
# scanning everything pulls in incidental mentions of other topics.
_DOC_HEAD_CHARS = 240


def _scan(topic: str) -> int:
    return int(_SCAN_CHARS.get(topic, _DOC_HEAD_CHARS))


_CARE_ID_PREFIXES = tuple(_RANKING.get("care_id_prefixes") or ())
_CURATED_ID_PREFIXES = tuple(_RANKING.get("curated_id_prefixes") or ())
_QUESTIONNAIRE_ID_PREFIXES = tuple(_RANKING.get("questionnaire_id_prefixes") or ())
_CARE_TITLES = tuple(_RANKING.get("care_titles") or ())


def _hit_matches(hit: dict, terms: tuple[str, ...], scan_chars: int) -> bool:
    """True when any term appears in the chunk's title, id, or leading body text."""
    title = (hit.get("title") or "").lower()
    text = (hit.get("text") or "").lower()[:scan_chars]
    hid = str(hit.get("id") or "").lower()
    return any(t in title or t in text or t in hid for t in terms)


def _topic_text_for_detection(text: str) -> str:
    """Strip orchestrator age/sex metadata; soft-follow domain stays on the prior query."""
    raw = text or ""
    # Soft follow-ups: "prior\nFollow-up: is that okay?" — classify on prior domain.
    if re.search(r"Follow-up(?: \(original\))?:", raw):
        prior = re.split(r"\n?Follow-up(?: \(original\))?:\s*", raw, maxsplit=1)[0]
        raw = prior.strip() or raw
    raw = _TOPIC_META_LINE_RE.sub(" ", raw)
    return raw.strip()


def _is_feeding_query(text: str) -> bool:
    raw = _topic_text_for_detection(text)
    qlow = raw.lower()
    if any(k in qlow for k in _FEEDING_KEYWORDS):
        return True
    return bool(_FEEDING_RE.search(raw))


def _is_motor_query(text: str) -> bool:
    qlow = _topic_text_for_detection(text).lower()
    return any(k in qlow for k in _MOTOR_KEYWORDS)


def _feeding_ids_for_age(age_m: float | None) -> tuple[str, ...]:
    """Prefer age-banded curated feeding chunk ids (bands from config/care_topics.yaml)."""
    if age_m is None:
        return tuple(_CFG.get("feeding_ids_unknown_age") or ())
    for band in _CFG.get("feeding_age_bands") or []:
        upper = band.get("max_age_months")
        if upper is None or age_m < float(upper):
            return tuple(band.get("ids") or ())
    return ()


def _is_feeding_doc(doc: dict) -> bool:
    hid = str(doc.get("id") or "").lower()
    title = (doc.get("title") or "").lower()
    return "feeding" in hid or "feeding" in title or hid.startswith("info_iron_breastfed")


def _is_growth_centile_doc(doc: dict) -> bool:
    hid = str(doc.get("id") or "")
    title = doc.get("title") or ""
    text_head = (doc.get("text") or "")[:_DOC_HEAD_CHARS]
    if "feeding" in hid.lower():
        return False
    if any(t in hid.lower() for t in _terms("growth_explain_id")):
        return True
    return bool(_GROWTH_CENTILE_DOC_RE.search(f"{hid} {title} {text_head}"))


def _select_feeding_hits(
    docs: list[dict],
    retrieved: list[dict],
    *,
    age_m: float | None,
    top_k: int | None = None,
) -> list[dict]:
    """Force feeding guidance chunks; never return growth-centile explainers for feeding asks."""
    if top_k is None:
        top_k = get_settings().nestling_rag_topic_hits
    preferred_ids = _feeding_ids_for_age(age_m)
    by_id = {str(d.get("id")): d for d in docs}
    picked: list[dict] = []
    for pid in preferred_ids:
        d = by_id.get(pid)
        if d is None:
            # prefix match (e.g. info_feeding_0_3m_extra)
            matches = [x for x in docs if str(x.get("id", "")).startswith(pid)]
            d = matches[0] if matches else None
        if d is not None:
            item = dict(d)
            item.setdefault("score", 100.0)
            picked.append(item)
        if len(picked) >= top_k:
            break
    # When age is known, stick to the age band — do not pad with solids/toddler docs.
    if age_m is not None and picked:
        return picked[:top_k]
    if len(picked) < top_k:
        for h in retrieved:
            if _is_growth_centile_doc(h):
                continue
            if not _is_feeding_doc(h) and "nutrition" not in (h.get("title") or "").lower():
                continue
            hid = str(h.get("id"))
            if any(str(p.get("id")) == hid for p in picked):
                continue
            picked.append(dict(h))
            if len(picked) >= top_k:
                break
    if not picked:
        # Last resort: any curated feeding doc in the store
        for d in docs:
            if _is_feeding_doc(d):
                item = dict(d)
                item.setdefault("score", 1.0)
                picked.append(item)
            if len(picked) >= top_k:
                break
    return picked[:top_k]


class VectorStore:
    def __init__(self, collection: str, index_dir: Path):
        self.collection = collection
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.docs: list[dict] = []
        self.bm25 = BM25Index()

    def add(self, docs: list[dict]):
        for d in docs:
            d = dict(d)
            d["collection"] = self.collection
            self.docs.append(d)
        self._rebuild()

    def clear(self):
        self.docs = []
        self.bm25 = BM25Index()

    def _rebuild(self):
        texts = [f"{d.get('title', '')}\n{d.get('text', '')}" for d in self.docs]
        self.bm25.fit(texts)

    def save(self):
        """Write the index atomically so a reader never sees a half-written file."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        docs_path = self.index_dir / "docs.json"
        tmp_path = docs_path.with_suffix(f".{os.getpid()}.tmp")
        tmp_path.write_text(
            json.dumps(self.docs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp_path, docs_path)

    def load(self) -> bool:
        """Load a persisted index. A missing or corrupt file means 'rebuild me'."""
        docs_path = self.index_dir / "docs.json"
        if not docs_path.exists():
            return False
        try:
            docs = json.loads(docs_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            log.warning("Ignoring unreadable RAG index %s: %s", docs_path, exc)
            return False
        if not isinstance(docs, list):
            log.warning("Ignoring malformed RAG index %s: expected a list of docs", docs_path)
            return False
        self.docs = docs
        self._rebuild()
        return True

    def search(self, query: str, top_k: int | None = None, filters: dict | None = None) -> list[dict]:
        settings = get_settings()
        if top_k is None:
            top_k = settings.nestling_rag_top_k
        if not self.docs:
            return []
        sims = self.bm25.scores(query)
        order = np.argsort(-sims)
        # Pull a wider BM25 pool so dense fusion (when enabled) can re-rank.
        pool_n = max(top_k * settings.nestling_rag_pool_multiplier, settings.nestling_rag_pool_min)
        candidates = []
        for i in order:
            doc = dict(self.docs[int(i)])
            if filters:
                if any(doc.get(k) != v for k, v in filters.items()):
                    continue
            doc["score"] = float(sims[int(i)])
            candidates.append(doc)
            if len(candidates) >= pool_n:
                break
        try:
            from assistant.rag.dense import dense_enabled, hybrid_order

            if dense_enabled() and candidates:
                id_to_text = {
                    str(d.get("id")): f"{d.get('title') or ''}\n{d.get('text') or ''}"
                    for d in candidates
                }
                bm25_ids = [str(d.get("id")) for d in candidates]
                fused_ids = hybrid_order(bm25_ids, query, id_to_text, top_k=top_k)
                by_id = {str(d.get("id")): d for d in candidates}
                return [by_id[i] for i in fused_ids if i in by_id]
        except Exception as exc:
            log.warning("Dense re-ranking failed, falling back to BM25: %s", exc)
        return candidates[:top_k]


class MedicalRAG:
    def __init__(self):
        self.store = VectorStore("medical", MEDICAL_INDEX_DIR)

    def build_from_chunks(self, chunks_path: Path | None = None):
        path = chunks_path or (KNOWLEDGE_DIR / "chunks.json")
        chunks = json.loads(path.read_text(encoding="utf-8"))
        self.store.clear()
        self.store.add(chunks)
        self.store.save()
        return len(chunks)

    def load(self) -> bool:
        return self.store.load()

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        return self.store.search(query, top_k=top_k)

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        use_llm: bool = True,
        *,
        use_pleias: bool | None = None,
    ) -> dict:
        if use_pleias is not None:
            use_llm = use_pleias
        settings = get_settings()
        if top_k is None:
            top_k = settings.nestling_rag_top_k
        # Topic detection must ignore injected chat memory — only the current user turn.
        topic_src = query or ""
        if "[CURRENT_USER]" in topic_src:
            topic_src = topic_src.split("[CURRENT_USER]", 1)[-1]
        topic_src = re.split(
            r"\[SESSION_SUMMARY\]|\[RECENT_CHAT\]|\[SESSION_SLOTS\]",
            topic_src,
            maxsplit=1,
        )[0]
        detect_src = _topic_text_for_detection(topic_src)
        qlow = detect_src.lower()
        iron_q = any(k in qlow for k in _IRON_KEYWORDS)
        # Iron wins over generic feeding/breast keywords ("iron for breastfed…")
        feeding_q = _is_feeding_query(detect_src) and not iron_q
        growth_explain_q = (not feeding_q) and any(
            k in qlow for k in _GROWTH_EXPLAIN_KEYWORDS
        )
        age_m = None
        # Age may live on metadata lines stripped from detect_src — read full topic_src.
        m_age = re.search(
            r"known (?:chronological )?child age:\s*([0-9]+(?:\.[0-9]+)?)\s*months|"
            r"known chronological age:\s*([0-9]+(?:\.[0-9]+)?)\s*months",
            (topic_src or "").lower(),
        )
        if m_age:
            try:
                age_m = float(m_age.group(1) or m_age.group(2))
            except (TypeError, ValueError):
                age_m = None
        screening_q = any(k in qlow for k in _SCREENING_KEYWORDS)
        sleep_q = any(k in qlow for k in _SLEEP_KEYWORDS)
        rash_q = any(k in qlow for k in _RASH_KEYWORDS)
        motor_q = (not feeding_q and not iron_q and not sleep_q and not rash_q) and _is_motor_query(
            detect_src
        )
        # Speech only when not a more specific care topic (feeding/iron/sleep/rash/motor win).
        speech_q = (
            not feeding_q
            and not iron_q
            and not sleep_q
            and not rash_q
            and not motor_q
        ) and any(k in qlow for k in _SPEECH_KEYWORDS)
        # Expand short parent concerns so lexical search finds the right guidance chunk
        # Retrieval query = current turn only (memory context pollutes BM25).
        search_q = detect_src.strip() or topic_src.strip() or query
        for flag, topic in (
            (feeding_q, "feeding"),
            (iron_q, "iron"),
            (sleep_q, "sleep"),
            (rash_q, "rash"),
            (motor_q, "motor"),
            (speech_q, "speech"),
        ):
            if flag:
                search_q = f"{search_q} {_expansion(topic)}".strip()
                break

        hits = self.retrieve(
            search_q,
            top_k=max(top_k * settings.nestling_rag_query_multiplier, settings.nestling_rag_query_min),
        )
        topic_hits = settings.nestling_rag_topic_hits
        if feeding_q:
            hits = _select_feeding_hits(
                self.store.docs, hits, age_m=age_m, top_k=max(top_k, topic_hits)
            )
        elif not screening_q:
            # Prefer topic-matched care/guidance; drop raw questionnaire dumps.
            def _topic_rank(h: dict) -> tuple:
                title = (h.get("title") or "").lower()
                text = (h.get("text") or "").lower()
                hid = str(h.get("id", ""))
                score = float(h.get("score") or 0)
                boost = 0.0
                if iron_q and any(t in title or t in text for t in _IRON_KEYWORDS):
                    boost += _BOOSTS["topic_text_match"]
                if sleep_q and any(t in title for t in _SLEEP_KEYWORDS):
                    boost += _BOOSTS["topic_text_match"]
                if motor_q and any(
                    t in title or t in text or t in hid for t in _terms("motor_text")
                ):
                    boost += _BOOSTS["motor_text_match"]
                if motor_q and any(t in hid for t in _terms("motor_id")):
                    boost += _BOOSTS["motor_id_match"]
                if speech_q and any(t in title or t in text for t in _terms("speech_text")):
                    boost += _BOOSTS["topic_text_match"]
                if speech_q and any(t in hid for t in _terms("speech_id")):
                    boost += _BOOSTS["speech_id_match"]
                if rash_q and any(t in title or t in text for t in _terms("rash_text")):
                    boost += _BOOSTS["rash_text_match"]
                if rash_q and any(t in hid for t in _terms("rash_id")):
                    boost += _BOOSTS["rash_id_match"]
                if growth_explain_q and (
                    any(t in hid for t in _terms("growth_explain_id"))
                    or any(t in title for t in _terms("growth_explain_title"))
                ):
                    boost += _BOOSTS["growth_explain_match"]
                if hid.startswith(_CURATED_ID_PREFIXES):
                    boost += _BOOSTS["curated_prefix"]
                if hid.startswith(_QUESTIONNAIRE_ID_PREFIXES) and not screening_q:
                    boost -= _PENALTIES["questionnaire_dump"]
                # Keep feeding docs out of non-feeding medical answers.
                if "feeding" in hid or "feeding" in title:
                    boost -= _PENALTIES["off_topic_feeding"]
                return (boost + score, score)

            care = [
                h
                for h in hits
                if str(h.get("id", "")).startswith(_CARE_ID_PREFIXES)
                or "guidance" in (h.get("title") or "").lower()
                or any(k in (h.get("title") or "").lower() for k in _CARE_TITLES)
                or "intergrowth" in (h.get("text") or "").lower()
            ]
            # Drop feeding chunks entirely unless this is a feeding/iron ask.
            if not feeding_q and not iron_q:
                care = [
                    h
                    for h in care
                    if "feeding" not in str(h.get("id", "")).lower()
                    and "feeding" not in (h.get("title") or "").lower()
                ]
                hits = [
                    h
                    for h in hits
                    if "feeding" not in str(h.get("id", "")).lower()
                    and "feeding" not in (h.get("title") or "").lower()
                ]
            pool = care or [
                h
                for h in hits
                if not str(h.get("id", "")).startswith(_QUESTIONNAIRE_ID_PREFIXES)
            ] or hits
            ranked = sorted(pool, key=_topic_rank, reverse=True)
            if iron_q:
                iron_hits = [
                    h for h in ranked if _hit_matches(h, _selected("iron"), _scan("iron"))
                ]
                hits = (iron_hits or ranked)[:topic_hits]
            elif sleep_q:
                sleep_hits = [
                    h for h in ranked if _hit_matches(h, _selected("sleep"), _scan("sleep"))
                ]
                hits = (sleep_hits or ranked)[:topic_hits]
            elif rash_q:
                rash_hits = [
                    h for h in ranked if _hit_matches(h, _selected("rash"), _scan("rash"))
                ]
                if not rash_hits:
                    rash_hits = [
                        d
                        for d in self.store.docs
                        if any(
                            k in str(d.get("id") or "").lower()
                            or k in (d.get("title") or "").lower()
                            for k in _selected("rash_fallback_ids")
                        )
                    ]
                hits = (rash_hits or ranked)[:topic_hits]
            elif motor_q:
                motor_hits = [
                    h for h in ranked if _hit_matches(h, _selected("motor"), _scan("motor"))
                ]
                if not motor_hits:
                    motor_hits = [
                        d
                        for d in self.store.docs
                        if any(
                            k in str(d.get("id") or "").lower()
                            for k in _selected("motor_fallback_ids")
                        )
                        or any(
                            k in (d.get("title") or "").lower()
                            for k in _selected("motor_fallback_titles")
                        )
                    ]
                hits = (motor_hits or ranked)[:topic_hits]
            elif speech_q:
                speech_hits = [
                    h for h in ranked if _hit_matches(h, _selected("speech"), _scan("speech"))
                ]
                if not speech_hits:
                    speech_hits = [
                        d
                        for d in self.store.docs
                        if any(
                            k in str(d.get("id") or "").lower()
                            for k in _selected("speech_fallback_ids")
                        )
                        or any(
                            k in (d.get("title") or "").lower()
                            for k in _selected("speech_fallback_titles")
                        )
                    ]
                hits = (speech_hits or ranked)[:settings.nestling_rag_speech_hits]
            else:
                hits = ranked[:top_k]
        else:
            hits = hits[:top_k]
        context = "\n\n".join(f"[{h['id']}] {h['title']}: {h['text']}" for h in hits)
        text = ""
        mode = "extractive"
        model_id = None
        llm_err = ""
        model_id = None
        if use_llm:
            try:
                from assistant.llm.qwen_client import get_qwen, llm_enabled

                if llm_enabled():
                    client = get_qwen()
                    if client.ready:
                        # Generation sees the full current-user block (incl. Follow-up);
                        # topic detection already used detect_src / prior domain.
                        gen_raw = _TOPIC_META_LINE_RE.sub(" ", topic_src).strip()
                        gen_q = gen_raw or detect_src.strip() or query
                        text = client.answer_with_context(gen_q, context)
                        mode = "openai-compatible-llm"
                        model_id = client.model
            except Exception as exc:
                text = ""
                llm_err = str(exc)

        if not text:
            text = _extractive(hits, conversational=True)
            if llm_err:
                text += f"\n[LLM unavailable: {llm_err}]"
                mode = "extractive_fallback"
            else:
                mode = "extractive"
        return {
            "collection": "medical",
            "query": query,
            "mode": mode,
            "model": model_id,
            "answer": text,
            "citations": [{"id": h["id"], "title": h["title"], "score": h["score"]} for h in hits],
            "context": context,
        }


class ChildRAG:
    def __init__(self):
        self.store = VectorStore("child", CHILD_INDEX_DIR)

    def reindex_child(self, child_docs: list[dict]):
        if not child_docs:
            return
        child_id = child_docs[0]["child_id"]
        remaining = [d for d in self.store.docs if d.get("child_id") != child_id]
        self.store.docs = remaining
        self.store.add(child_docs)
        self.store.save()

    def load(self) -> bool:
        return self.store.load()

    def retrieve(self, query: str, child_id: str, top_k: int | None = None) -> list[dict]:
        return self.store.search(query, top_k=top_k, filters={"child_id": child_id})

    def answer(
        self,
        query: str,
        child_id: str,
        top_k: int | None = None,
        use_llm: bool = True,
        *,
        use_pleias: bool | None = None,
    ) -> dict:
        if use_pleias is not None:
            use_llm = use_pleias
        hits = self.retrieve(query, child_id, top_k=top_k)
        if not hits:
            return {
                "collection": "child",
                "child_id": child_id,
                "answer": "No stored records for this child matched the query.",
                "citations": [],
                "model": None,
            }
        context = "\n\n".join(f"[{h['id']}] {h['text']}" for h in hits)
        text = ""
        mode = "extractive"
        llm_err = ""
        model_id = None
        if use_llm:
            try:
                from assistant.llm.qwen_client import get_qwen, llm_enabled

                if llm_enabled():
                    client = get_qwen()
                    if client.ready:
                        text = client.answer_with_context(query, context)
                        mode = "openai-compatible-llm"
                        model_id = client.model
            except Exception as exc:
                llm_err = str(exc)
        if not text:
            text = _extractive(hits)
            if llm_err:
                text += f"\n[LLM unavailable: {llm_err}]"
                mode = "extractive_fallback"
        return {
            "collection": "child",
            "child_id": child_id,
            "mode": mode,
            "model": model_id if mode == "openai-compatible-llm" else None,
            "answer": text,
            "citations": [{"id": h["id"], "title": h["title"], "score": h["score"]} for h in hits],
        }


def _trim_extract(
    text: str, *, max_chars: int, min_sentence_chars: int, keep_ratio: float
) -> str:
    """
    Shorten a care note to `max_chars`, preferring a sentence end.

    A sentence end is only used when it keeps at least `keep_ratio` of the budget;
    otherwise the note is cut on a word boundary. Curated chunks are written as
    long bulleted age bands, so always snapping to the last full stop could drop
    the very age band the parent asked about.
    """
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    sentence_end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if sentence_end > min_sentence_chars and sentence_end >= max_chars * keep_ratio:
        return cut[: sentence_end + 1]
    word_end = cut.rfind(" ")
    if word_end > min_sentence_chars:
        cut = cut[:word_end]
    return cut.rstrip(" ,;:") + "…"


def _extractive(hits: list[dict], *, conversational: bool = True) -> str:
    if not hits:
        return NO_MATCH_REPLY
    # Prefer one best guidance chunk, spoken as chat (not a citation dump)
    h = hits[0]
    title = (h.get("title") or "").strip()
    text = (h.get("text") or "").strip()
    if text.startswith("## "):
        _, _, rest = text.partition("\n")
        text = rest.strip() or text
    # Soft trim long dumps — last-resort fallback only (prefer LLM when ready)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\*\*", "", text)
    settings = get_settings()
    text = _trim_extract(
        text,
        max_chars=settings.nestling_rag_extract_chars,
        min_sentence_chars=settings.nestling_rag_extract_min_sentence_chars,
        keep_ratio=settings.nestling_rag_extract_keep_ratio,
    )
    if conversational:
        topic = title or "this"
        return f"Here’s the short version from our notes on {topic}: {text} {SAFETY_TAIL}"
    parts = ["Based on retrieved sources:", f"- {title or h.get('id')}: {text}"]
    parts.append(CITATION_TAIL)
    return "\n".join(parts)
