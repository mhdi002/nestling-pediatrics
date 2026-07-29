"""Dual RAG functional + accuracy tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.memory.child_db import ChildMemoryDB
from assistant.rag.stores import ChildRAG, MedicalRAG


def test_medical_rag_retrieves_iron():
    rag = MedicalRAG()
    assert rag.load() or rag.build_from_chunks() > 0
    hits = rag.retrieve("iron supplements for breastfed infants", top_k=5)
    assert hits
    joined = " ".join(h["text"].lower() for h in hits)
    assert "iron" in joined
    ans = rag.answer("What about iron for newborns?", use_pleias=False)
    assert "iron" in ans["answer"].lower()
    assert ans["citations"]


def test_medical_rag_speech_concern():
    rag = MedicalRAG()
    rag.load() or rag.build_from_chunks()
    ans = rag.answer("my child cant talk", use_pleias=False)
    joined_cites = " ".join(
        ((c.get("id") or "") + " " + (c.get("title") or "") + " " + (c.get("text") or "")).lower()
        for c in ans["citations"]
    )
    answer = ans["answer"].lower()
    assert (
        "speech" in joined_cites
        or "talk" in joined_cites
        or "speech" in answer
        or "talk" in answer
    )
    assert "3-month" in answer or "words" in answer or "talk" in answer


def test_medical_rag_intergrowth():
    rag = MedicalRAG()
    rag.load() or rag.build_from_chunks()
    hits = rag.retrieve("preterm growth chart percentiles INTERGROWTH", top_k=3)
    assert any("intergrowth" in (h["text"] + h["title"]).lower() for h in hits)


def test_child_rag_precision_memory():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = ChildMemoryDB(Path(td) / "c.db")
        try:
            cid = db.create_child("Test Baby", "male", gestational_age_weeks=32)
            db.add_growth(cid, weeks=40, measure="weight", value=3.2, centile=45, z_score=-0.1, track_status="within_10_90")
            db.add_screening(
                cid,
                "ASQ",
                {"communication": ["yes"] * 6},
                {"summary": "ASQ communication total 60", "needs_referral": False},
                age_months=4,
            )
            rag = ChildRAG()
            rag.store.index_dir = Path(td) / "idx"
            rag.store.index_dir.mkdir(parents=True, exist_ok=True)
            rag.store.docs = []
            rag.reindex_child(db.timeline_documents(cid))
            hits = rag.retrieve("weight measurement history", child_id=cid, top_k=5)
            assert hits
            assert any("weight" in h["text"].lower() for h in hits)
            ans = rag.answer("What were the ASQ results?", child_id=cid, use_pleias=False)
            assert "ASQ" in ans["answer"] or "asq" in ans["answer"].lower()
            empty = rag.retrieve("weight", child_id="no-such-child", top_k=3)
            assert empty == []
        finally:
            db.close()
