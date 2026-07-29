#!/usr/bin/env python3
"""
Dual RAG stores.

Retrieval: BM25 (no extra neural models).
Generation: PleIAs/Pleias-RAG-1B ONLY.
Tool calling elsewhere: Salesforce/xLAM-1b-fc-r ONLY.
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
    PLEIAS_RAG_MODEL_ID,
)
from assistant.rag.embeddings import BM25Index


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
        out = []
        for i in order:
            doc = dict(self.docs[int(i)])
            if filters:
                if any(doc.get(k) != v for k, v in filters.items()):
                    continue
            score = float(sims[int(i)])
            if score <= 0 and filters is None:
                # still allow weak matches ranked last
                pass
            doc["score"] = score
            out.append(doc)
            if len(out) >= top_k:
                break
        return out


class PleiasRAGGenerator:
    """Lazy loader for PleIAs/Pleias-RAG-1B — the only generative RAG model allowed."""

    def __init__(self):
        self._tok = None
        self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(PLEIAS_RAG_MODEL_ID, trust_remote_code=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
        kwargs["device_map"] = "auto" if device == "cuda" else "cpu"
        self._model = AutoModelForCausalLM.from_pretrained(PLEIAS_RAG_MODEL_ID, **kwargs)

    def generate(self, query: str, context: str, max_new_tokens: int = 256) -> str:
        if not self.ready:
            self.load()
        prompt = (
            "You are a careful pediatric education assistant. Use ONLY the provided sources. "
            "Quote sources. If insufficient, say you do not know. Never invent growth numbers "
            "or screening scores — those come from tools.\n\n"
            f"SOURCES:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:"
        )
        enc = self._tok(prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        # Avoid BatchEncoding.device (raises empty AttributeError on transformers 5.x)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        out = self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return self._tok.decode(out[0][input_ids.shape[1] :], skip_special_tokens=True).strip()


_PLEIAS = PleiasRAGGenerator()


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

    def answer(self, query: str, top_k: int = 5, use_pleias: bool = True) -> dict:
        qlow = (query or "").lower()
        screening_q = any(k in qlow for k in ("asq", "m-chat", "mchat", "autism", "screening item"))
        speech_q = any(
            k in qlow
            for k in (
                "talk",
                "speech",
                "language",
                "words",
                "babbl",
                "milestone",
                "development",
                "حرف",
                "گفتار",
                "صحبت",
                "تاخیر",
            )
        )
        iron_q = "iron" in qlow or "آهن" in qlow
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
                "redness",
                "hfmd",
                "hand foot",
                "hand-foot",
                "eczema",
                "spot",
                "جوش",
                "راش",
                "زخم",
            )
        )
        # Expand short parent concerns so lexical search finds the right guidance chunk
        search_q = query
        if speech_q:
            search_q = f"{query} speech language development milestones infant cooing babbling"
        elif iron_q:
            search_q = f"{query} iron supplementation breastfed infant"
        elif sleep_q:
            search_q = f"{query} infant sleep safe sleep hours"
        elif rash_q:
            search_q = (
                f"{query} pediatric rash palm blister vesicle hand foot mouth "
                "wound redness fever urgent care"
            )

        hits = self.retrieve(search_q, top_k=max(top_k * 3, 15))
        if not screening_q:
            # Prefer topic-matched care/guidance; drop raw questionnaire dumps.
            def _topic_rank(h: dict) -> tuple:
                title = (h.get("title") or "").lower()
                text = (h.get("text") or "").lower()
                hid = str(h.get("id", ""))
                score = float(h.get("score") or 0)
                boost = 0.0
                if speech_q and any(
                    t in title or t in text
                    for t in ("speech", "language", "communication", "milestone", "development")
                ):
                    boost += 50
                if iron_q and "iron" in title:
                    boost += 50
                if sleep_q and "sleep" in title:
                    boost += 50
                if rash_q and any(
                    t in title or t in text
                    for t in (
                        "rash",
                        "wound",
                        "hand",
                        "foot",
                        "mouth",
                        "blister",
                        "skin",
                        "fever",
                        "vision",
                    )
                ):
                    boost += 55
                if hid.startswith(("info_", "intergrowth")):
                    boost += 5
                if hid.startswith(("mchat_", "asq_")) and not screening_q:
                    boost -= 20
                return (boost + score, score)

            care = [
                h
                for h in hits
                if str(h.get("id", "")).startswith(("info_", "intergrowth"))
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
                    )
                )
                or "intergrowth" in (h.get("text") or "").lower()
            ]
            pool = care or [
                h
                for h in hits
                if not str(h.get("id", "")).startswith(("mchat_", "asq_"))
            ] or hits
            ranked = sorted(pool, key=_topic_rank, reverse=True)
            if speech_q:
                speech_hits = [
                    h
                    for h in ranked
                    if "speech" in (h.get("title") or "").lower()
                    or "language" in (h.get("title") or "").lower()
                    or "speech" in str(h.get("id") or "").lower()
                    or "communication" in (h.get("title") or "").lower()
                ]
                hits = (speech_hits or ranked)[:2]
            elif iron_q:
                iron_hits = [h for h in ranked if "iron" in (h.get("title") or "").lower() or "iron" in (h.get("text") or "").lower()[:200]]
                hits = (iron_hits or ranked)[:3]
            elif rash_q:
                rash_hits = [
                    h
                    for h in ranked
                    if any(
                        k in (h.get("title") or "").lower()
                        or k in (h.get("text") or "").lower()[:400]
                        for k in (
                            "rash",
                            "wound",
                            "hand-foot",
                            "hand foot",
                            "blister",
                            "skin",
                            "fever and rash",
                            "photo",
                        )
                    )
                ]
                hits = (rash_hits or ranked)[:3]
            else:
                hits = ranked[:top_k]
        else:
            hits = hits[:top_k]
        context = "\n\n".join(f"[{h['id']}] {h['title']}: {h['text']}" for h in hits)
        text = ""
        mode = "extractive"
        model_id = None
        # Prefer Bonsai-27B (llama-server) when configured
        try:
            from assistant.llm.bonsai_client import bonsai_enabled, get_bonsai

            if bonsai_enabled():
                client = get_bonsai()
                if client.ready:
                    text = client.answer_with_context(query, context)
                    mode = "bonsai-27b-q1"
                    model_id = "prism-ml/Bonsai-27B-gguf"
        except Exception as exc:
            text = ""
            bonsai_err = str(exc)
        else:
            bonsai_err = ""

        if not text and use_pleias:
            try:
                text = _PLEIAS.generate(query, context)
                mode = "pleias-rag-1b"
                model_id = PLEIAS_RAG_MODEL_ID
            except Exception as exc:
                text = _extractive(hits)
                if bonsai_err:
                    text += f"\n[Bonsai unavailable: {bonsai_err}]"
                text += f"\n[Pleias-RAG-1B unavailable: {exc}]"
                mode = "extractive_fallback"
        elif not text:
            text = _extractive(hits)
            if bonsai_err:
                text += f"\n[Bonsai unavailable: {bonsai_err}]"
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

    def answer(self, query: str, child_id: str, top_k: int = 5, use_pleias: bool = True) -> dict:
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
        if use_pleias:
            try:
                text = _PLEIAS.generate(query, context)
                mode = "pleias-rag-1b"
            except Exception as exc:
                text = _extractive(hits) + f"\n[Pleias-RAG-1B unavailable: {exc}]"
                mode = "extractive_fallback"
        else:
            text = _extractive(hits)
            mode = "extractive"
        return {
            "collection": "child",
            "child_id": child_id,
            "mode": mode,
            "model": PLEIAS_RAG_MODEL_ID if mode.startswith("pleias") else None,
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
    # Soft trim long dumps
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 700:
        cut = text[:700]
        # end on a sentence if possible
        sp = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
        text = cut[: sp + 1] if sp > 200 else cut + "…"
    if conversational:
        return text
    parts = ["Based on retrieved sources:", f"- {title or h.get('id')}: {text}"]
    parts.append("For diagnosis or treatment decisions, consult a pediatric clinician.")
    return "\n".join(parts)
