"""Translation verification tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.config import EN_DIR, KNOWLEDGE_DIR
from assistant.translate import GLOSSARY, translate_text, verify_translation


def test_glossary_exact():
    assert translate_text("برقراری ارتباط") == "Communication"
    assert translate_text("بله") == "Yes"


def test_en_asq_exists_and_verified():
    files = list((EN_DIR / "asq").glob("*.json"))
    assert files, "English ASQ files missing — run assistant.translate"
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["domains"]
        for dom in data["domains"]:
            assert dom["title_en"]
            for q in dom["questions"]:
                assert q["text_en"]


def test_knowledge_chunks():
    path = KNOWLEDGE_DIR / "chunks.json"
    assert path.exists()
    chunks = json.loads(path.read_text(encoding="utf-8"))
    assert len(chunks) > 10
    assert any(c["id"] == "intergrowth_docs" for c in chunks)


def test_curated_medical_present():
    p = EN_DIR / "curated_medical_en.md"
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "INTERGROWTH" in text
    assert "Iron" in text or "iron" in text
