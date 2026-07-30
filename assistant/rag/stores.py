#!/usr/bin/env python3
"""
Dual RAG stores.

Retrieval: BM25 (no extra neural models).
Generation: local OpenAI-compatible LLM sidecar (or extractive fallback).
Tool calling elsewhere: Salesforce/xLAM-1b-fc-r (optional) or deterministic router.
"""

from __future__ import annotations

import json
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

# EN + FA feeding / nutrition cues for topic isolation (never confuse with growth charts).
_FEEDING_KEYWORDS = (
    "feed",
    "feeding",
    "food",
    "foods",
    "eat",
    "eaten",
    "eating",
    "milk",
    "breast",
    "breastfeed",
    "breastfeeding",
    "formula",
    "solid",
    "solids",
    "weaning",
    "complementary",
    "nutrition",
    "تغذیه",
    "غذا",
    "خوراک",
    "شیر",
    "بخوره",
    "بدم",
)
_FEEDING_RE = re.compile(
    r"چی\s*(?:به(?:ش|ش)?\s*)?بدم|چه\s*غذایی|غذا\s*باید|بچم\s*غذا|به(?:ش|ش)?\s*چی\s*بدم",
    re.I,
)
_GROWTH_CENTILE_DOC_RE = re.compile(
    r"(centile|percentile|growth.?chart|intergrowth|صدک|نمودار\s*رشد)",
    re.I,
)
# Metadata the orchestrator appends — must not drive topic classification.
_TOPIC_META_LINE_RE = re.compile(
    r"(?im)^(?:known chronological age:|known child (?:sex|age):|born preterm at).*$"
)
_MOTOR_KEYWORDS = (
    "walk",
    "walking",
    "crawl",
    "crawling",
    "cruise",
    "cruising",
    "standing",
    "gross motor",
    "fine motor",
    "motor skill",
    "pull to stand",
    "pulling to stand",
    "راه رفتن",
    "خزیدن",
)


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
    """Prefer age-banded curated feeding chunk ids."""
    if age_m is None:
        return (
            "info_feeding_newborn",
            "info_feeding_0_3m",
            "info_feeding_4_5m",
            "info_feeding_solids_6m",
            "info_feeding_7_9m",
            "info_feeding_10_12m",
            "info_feeding_12_24m",
            "info_iron_breastfed",
        )
    if age_m < 4:
        return ("info_feeding_newborn", "info_feeding_0_3m", "info_iron_breastfed")
    if age_m < 6:
        return ("info_feeding_4_5m", "info_feeding_0_3m", "info_iron_breastfed")
    if age_m < 7:
        return ("info_feeding_solids_6m", "info_feeding_4_5m", "info_iron_breastfed")
    if age_m < 10:
        return ("info_feeding_7_9m", "info_feeding_solids_6m")
    if age_m < 12:
        return ("info_feeding_10_12m", "info_feeding_7_9m")
    return ("info_feeding_12_24m", "info_feeding_10_12m")


def _is_feeding_doc(doc: dict) -> bool:
    hid = str(doc.get("id") or "").lower()
    title = (doc.get("title") or "").lower()
    return "feeding" in hid or "feeding" in title or hid.startswith("info_iron_breastfed")


def _is_growth_centile_doc(doc: dict) -> bool:
    hid = str(doc.get("id") or "")
    title = doc.get("title") or ""
    text_head = (doc.get("text") or "")[:240]
    if "feeding" in hid.lower():
        return False
    if "centile" in hid.lower() or "intergrowth" in hid.lower():
        return True
    return bool(_GROWTH_CENTILE_DOC_RE.search(f"{hid} {title} {text_head}"))


def _select_feeding_hits(
    docs: list[dict],
    retrieved: list[dict],
    *,
    age_m: float | None,
    top_k: int = 3,
) -> list[dict]:
    """Force feeding guidance chunks; never return growth-centile explainers for feeding asks."""
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
        self.index_dir.mkdir(parents=True, exist_ok=True)
        (self.index_dir / "docs.json").write_text(
            json.dumps(self.docs, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self) -> bool:
        docs_path = self.index_dir / "docs.json"
        if not docs_path.exists():
            return False
        self.docs = json.loads(docs_path.read_text(encoding="utf-8"))
        self._rebuild()
        return True

    def search(self, query: str, top_k: int = 5, filters: dict | None = None) -> list[dict]:
        if not self.docs:
            return []
        sims = self.bm25.scores(query)
        order = np.argsort(-sims)
        # Pull a wider BM25 pool so dense fusion (when enabled) can re-rank.
        pool_n = max(top_k * 5, 25)
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
        except Exception:
            pass
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

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return self.store.search(query, top_k=top_k)

    def answer(self, query: str, top_k: int = 5, use_llm: bool = True, *, use_pleias: bool | None = None) -> dict:
        if use_pleias is not None:
            use_llm = use_pleias
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
        iron_q = "iron" in qlow or "آهن" in qlow
        # Iron wins over generic feeding/breast keywords ("iron for breastfed…")
        feeding_q = _is_feeding_query(detect_src) and not iron_q
        growth_explain_q = (not feeding_q) and any(
            k in qlow
            for k in (
                "centile",
                "percentile",
                "z-score",
                "z score",
                "growth chart",
                "intergrowth",
                "who chart",
                "نمودار",
                "چارت",
                "صدک",
            )
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
            except Exception:
                age_m = None
        screening_q = any(k in qlow for k in ("asq", "m-chat", "mchat", "autism", "screening item"))
        sleep_q = "sleep" in qlow or "خواب" in qlow
        rash_q = any(
            k in qlow
            for k in (
                "rash",
                "blister",
                "vesicle",
                "palm",
                "sole",
                "wound",
                "scar",
                "cut",
                "scrape",
                "injury",
                "bruise",
                "burn",
                "lesion",
                "redness",
                "hfmd",
                "hand foot",
                "hand-foot",
                "eczema",
                "spot",
                "bandage",
                "جوش",
                "راش",
                "زخم",
                "جراحت",
                "خراش",
                "کبودی",
                "سوختگی",
            )
        )
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
        ) and any(
            k in qlow
            for k in (
                "talk",
                "speech",
                "language",
                "words",
                "babbl",
                "dada",
                "mama",
                "baba",
                "papa",
                "says",
                "حرف",
                "گفتار",
                "صحبت",
                "تاخیر",
                "ماما",
                "بابا",
                "دادا",
            )
        )
        # Expand short parent concerns so lexical search finds the right guidance chunk
        # Retrieval query = current turn only (memory context pollutes BM25).
        search_q = detect_src.strip() or topic_src.strip() or query
        if feeding_q:
            search_q = (
                f"{search_q} infant feeding breastmilk formula complementary foods "
                "nutrition what should baby eat age-appropriate"
            )
        elif iron_q:
            search_q = f"{search_q} iron supplementation breastfed infant"
        elif sleep_q:
            search_q = f"{search_q} infant sleep safe sleep hours"
        elif rash_q:
            search_q = (
                f"{search_q} pediatric skin wound scar cut scrape bruise burn "
                "rash palm blister vesicle hand foot mouth redness first aid urgent care"
            )
        elif motor_q:
            search_q = (
                f"{search_q} gross motor walking crawling cruising standing milestones "
                "toddler development pull to stand delayed walk"
            )
        elif speech_q:
            search_q = f"{search_q} speech language development milestones infant cooing babbling"

        hits = self.retrieve(search_q, top_k=max(top_k * 3, 15))
        if feeding_q:
            hits = _select_feeding_hits(self.store.docs, hits, age_m=age_m, top_k=max(top_k, 3))
        elif not screening_q:
            # Prefer topic-matched care/guidance; drop raw questionnaire dumps.
            def _topic_rank(h: dict) -> tuple:
                title = (h.get("title") or "").lower()
                text = (h.get("text") or "").lower()
                hid = str(h.get("id", ""))
                score = float(h.get("score") or 0)
                boost = 0.0
                if iron_q and ("iron" in title or "iron" in text or "آهن" in text):
                    boost += 50
                if sleep_q and "sleep" in title:
                    boost += 50
                if motor_q and any(
                    t in title or t in text or t in hid
                    for t in (
                        "motor",
                        "walk",
                        "crawl",
                        "cruise",
                        "milestone",
                        "development",
                        "gross",
                    )
                ):
                    boost += 55
                if motor_q and (
                    "development" in hid
                    or "motor" in hid
                    or "milestone" in hid
                    or hid.startswith("guidance_curated_development")
                ):
                    boost += 80
                if speech_q and any(
                    t in title or t in text
                    for t in ("speech", "language", "communication", "milestone", "talk", "babbl")
                ):
                    boost += 50
                if speech_q and "speech" in hid:
                    boost += 80
                if rash_q and any(
                    t in title or t in text
                    for t in (
                        "rash",
                        "wound",
                        "scar",
                        "cut",
                        "scrape",
                        "bruise",
                        "burn",
                        "hand",
                        "foot",
                        "mouth",
                        "blister",
                        "skin",
                        "fever",
                        "vision",
                        "hfmd",
                        "first aid",
                    )
                ):
                    boost += 55
                if rash_q and any(
                    k in hid
                    for k in ("rash", "wound", "hfmd", "fever_rash", "photo", "scar", "skin")
                ):
                    boost += 80
                if growth_explain_q and (
                    "centile" in hid
                    or "percentile" in title
                    or "growth chart" in title
                    or hid.startswith("intergrowth")
                ):
                    boost += 60
                if hid.startswith(("info_", "intergrowth")):
                    boost += 5
                if hid.startswith(("mchat_", "asq_")) and not screening_q:
                    boost -= 20
                # Keep feeding docs out of non-feeding medical answers.
                if "feeding" in hid or "feeding" in title:
                    boost -= 100
                return (boost + score, score)

            care = [
                h
                for h in hits
                if str(h.get("id", "")).startswith(("info_", "intergrowth", "guidance_"))
                or "guidance" in (h.get("title") or "").lower()
                or any(
                    k in (h.get("title") or "").lower()
                    for k in (
                        "iron",
                        "sleep",
                        "vitamin",
                        "speech",
                        "language",
                        "growth",
                        "rash",
                        "wound",
                        "hand",
                        "skin",
                        "fever",
                        "motor",
                        "milestone",
                        "development",
                    )
                )
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
                if not str(h.get("id", "")).startswith(("mchat_", "asq_"))
            ] or hits
            ranked = sorted(pool, key=_topic_rank, reverse=True)
            if iron_q:
                iron_hits = [
                    h
                    for h in ranked
                    if "iron" in (h.get("title") or "").lower()
                    or "iron" in (h.get("text") or "").lower()[:400]
                    or "آهن" in (h.get("text") or "")
                ]
                hits = (iron_hits or ranked)[:3]
            elif sleep_q:
                sleep_hits = [
                    h
                    for h in ranked
                    if "sleep" in (h.get("title") or "").lower()
                    or "sleep" in (h.get("text") or "").lower()[:200]
                ]
                hits = (sleep_hits or ranked)[:3]
            elif rash_q:
                rash_hits = [
                    h
                    for h in ranked
                    if any(
                        k in (h.get("title") or "").lower()
                        or k in (h.get("text") or "").lower()[:500]
                        or k in str(h.get("id") or "").lower()
                        for k in (
                            "rash",
                            "wound",
                            "scar",
                            "cut",
                            "scrape",
                            "bruise",
                            "burn",
                            "hand-foot",
                            "hand foot",
                            "blister",
                            "skin",
                            "fever and rash",
                            "photo",
                            "hfmd",
                            "first aid",
                        )
                    )
                ]
                if not rash_hits:
                    rash_hits = [
                        d
                        for d in self.store.docs
                        if any(
                            k in str(d.get("id") or "").lower()
                            or k in (d.get("title") or "").lower()
                            for k in ("rash", "wound", "hfmd", "fever_rash", "photo_skin", "scar")
                        )
                    ]
                hits = (rash_hits or ranked)[:3]
            elif motor_q:
                motor_hits = [
                    h
                    for h in ranked
                    if any(
                        k in (h.get("title") or "").lower()
                        or k in (h.get("text") or "").lower()[:500]
                        or k in str(h.get("id") or "").lower()
                        for k in (
                            "motor",
                            "walk",
                            "crawl",
                            "cruise",
                            "milestone",
                            "development",
                            "gross",
                            "toddler",
                        )
                    )
                ]
                if not motor_hits:
                    motor_hits = [
                        d
                        for d in self.store.docs
                        if "development" in str(d.get("id") or "").lower()
                        or "motor" in str(d.get("id") or "").lower()
                        or "milestone" in (d.get("title") or "").lower()
                    ]
                hits = (motor_hits or ranked)[:3]
            elif speech_q:
                speech_hits = [
                    h
                    for h in ranked
                    if "speech" in (h.get("title") or "").lower()
                    or "language" in (h.get("title") or "").lower()
                    or "speech" in str(h.get("id") or "").lower()
                    or "communication" in (h.get("title") or "").lower()
                    or "talk" in (h.get("text") or "").lower()[:300]
                ]
                if not speech_hits:
                    speech_hits = [
                        d
                        for d in self.store.docs
                        if "speech" in str(d.get("id") or "").lower()
                        or "speech" in (d.get("title") or "").lower()
                        or "language" in (d.get("title") or "").lower()
                    ]
                hits = (speech_hits or ranked)[:2]
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

    def retrieve(self, query: str, child_id: str, top_k: int = 5) -> list[dict]:
        return self.store.search(query, top_k=top_k, filters={"child_id": child_id})

    def answer(
        self,
        query: str,
        child_id: str,
        top_k: int = 5,
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


def _extractive(hits: list[dict], *, conversational: bool = True) -> str:
    if not hits:
        return "I couldn't find a matching care note for that — try asking another way, or tell me more detail."
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
    if len(text) > 420:
        cut = text[:420]
        sp = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
        text = cut[: sp + 1] if sp > 160 else cut.rstrip(",;:") + "…"
    if conversational:
        topic = title or "this"
        return (
            f"Here’s the short version from our notes on {topic}: {text} "
            "If something looks worse (spreading redness, fever, or you’re unsure), call your clinician."
        )
    parts = ["Based on retrieved sources:", f"- {title or h.get('id')}: {text}"]
    parts.append("For diagnosis or treatment decisions, consult a pediatric clinician.")
    return "\n".join(parts)
