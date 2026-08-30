"""Translation verification tests."""

from __future__ import annotations

import json
import re
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


# --- ensure_pure_lang: mixed English body + Persian disclaimer (regression) ---
# The LLM answer generation and the appended disclaimer can leave a reply whose
# body is English but whose trailing disclaimer is Persian. ensure_pure_lang used
# to delegate the whole string to translate_en_to_fa(), which no-ops on any text
# that already contains Persian, so the English body slipped through untranslated.

from assistant import runtime_translate as RT  # noqa: E402


def _stub_translator(monkeypatch):
    """Deterministic offline stand-in for the MT call."""
    calls: list[str] = []

    def fake(text: str) -> str:
        calls.append(text)
        return "«ترجمه‌شده»"

    monkeypatch.setattr(RT, "translate_en_to_fa", fake)
    return calls


def test_ensure_pure_lang_translates_leaked_english_body(monkeypatch):
    calls = _stub_translator(monkeypatch)
    disclaimer = "برای تشخیص یا درمان حتماً با متخصص کودکان مشورت کنید."
    mixed = f"Keep the child hydrated and monitor the temperature closely.\n\n{disclaimer}"

    out = RT.ensure_pure_lang(mixed, "fa")

    # English block was handed to the translator; Persian disclaimer was not.
    assert calls == ["Keep the child hydrated and monitor the temperature closely."]
    assert "«ترجمه‌شده»" in out
    assert disclaimer in out
    # No English prose survives in the final reply.
    assert not re.search(r"[A-Za-z]{4,}", out)


def test_ensure_pure_lang_keeps_persian_with_english_term(monkeypatch):
    calls = _stub_translator(monkeypatch)
    text = "دوز acetaminophen را با پزشک چک کنید."

    out = RT.ensure_pure_lang(text, "fa")

    # A Persian block carrying an English medical term is left intact.
    assert out == text
    assert calls == []


def test_ensure_pure_lang_pure_persian_untouched(monkeypatch):
    calls = _stub_translator(monkeypatch)
    text = "سلام، حال فرزند شما چطور است؟"

    assert RT.ensure_pure_lang(text, "fa") == text
    assert calls == []
