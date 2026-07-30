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


def test_medical_rag_feeding_not_growth_centiles():
    """Feeding questions must retrieve feeding chunks, never growth-centile explainers."""
    rag = MedicalRAG()
    assert rag.load() or rag.build_from_chunks() > 0
    for q in (
        "بچم غذا باید چی بخوره ؟",
        "what should my baby eat",
        "tell about the foods that should be eaten in its year",
        "what should my baby eat\nKnown child age: 2 months.",
        "what foods are good for her?\nKnown chronological age: 13.5 months. Use ONLY this age.",
    ):
        ans = rag.answer(q, use_llm=False)
        ids = [c.get("id") or "" for c in ans.get("citations") or []]
        assert ids, f"no citations for {q!r}"
        assert any("feeding" in i or "iron_breastfed" in i for i in ids), ids
        assert not any("centile" in i or "intergrowth" in i for i in ids), ids
        joined = " ".join(ids).lower() + " " + (ans.get("answer") or "").lower()
        assert "centile" not in joined or "feeding" in joined
        assert "breast" in joined or "milk" in joined or "formula" in joined or "complementary" in joined or "feeding" in joined
        if "13.5" in q:
            assert any("12_24" in i or "10_12" in i for i in ids), ids
            assert not any("7_9" in i for i in ids), ids


def test_medical_rag_scar_and_walk_not_feeding_with_age_meta():
    """Age metadata must not force feeding RAG; scar/walk stay on-topic."""
    rag = MedicalRAG()
    assert rag.load() or rag.build_from_chunks() > 0
    age_meta = (
        "\nKnown chronological age: 13.5 months. "
        "Use ONLY this age for care guidance."
    )
    scar = rag.answer(f"she has a little scar on her hand{age_meta}", use_llm=False)
    scar_ids = [c.get("id") or "" for c in scar.get("citations") or []]
    assert scar_ids
    assert not any("feeding" in i for i in scar_ids), scar_ids
    scar_ans = (scar.get("answer") or "").lower()
    assert any(k in scar_ans for k in ("scar", "wound", "cut", "bandage", "clean", "redness"))
    assert "16" not in scar_ans and "finger food" not in scar_ans

    walk = rag.answer(f"she cant walk is that okey?{age_meta}", use_llm=False)
    walk_ids = [c.get("id") or "" for c in walk.get("citations") or []]
    assert walk_ids
    assert not any("feeding" in i for i in walk_ids), walk_ids
    walk_ans = (walk.get("answer") or "").lower()
    assert any(
        k in walk_ans or any(k in i for i in walk_ids)
        for k in ("walk", "motor", "cruise", "stand", "milestone", "development")
    ), (walk_ids, walk_ans[:300])
    assert "finger food" not in walk_ans and "16–24" not in walk_ans


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
