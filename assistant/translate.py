#!/usr/bin/env python3
"""Translate extracted Persian content to English and verify completeness."""

from __future__ import annotations

import hashlib
import time
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.config import EN_DIR, EXTRACTED, KNOWLEDGE_DIR

# Curated glossary for UI labels / domains (always exact).
GLOSSARY = {
    "برقراری ارتباط": "Communication",
    "حرکات درشت": "Gross motor",
    "حرکات ظریف": "Fine motor",
    "حل مسئله": "Problem solving",
    "شخصی-اجتماعی": "Personal-social",
    "موارد کلی": "Overall",
    "بله": "Yes",
    "گاهی": "Sometimes",
    "هنوز نه": "Not yet",
    "خیر": "No",
    "آری": "Yes",
    "پرسشنامه سنین و مراحل A.S.Q": "Ages & Stages Questionnaire (A.S.Q)",
    "نام کودک": "Child name",
    "نام خانوادگی کودک": "Family name",
    "تاریخ تولد": "Date of birth",
    "سن اصلاح شده": "Corrected age",
    "تاریخ تکمیل": "Completion date",
    "نام تکمیل کننده": "Completer name",
    "نسبت تکمیل کننده با کودک": "Completer relation to child",
    "تلفن تماس تکمیل کننده": "Completer phone",
    "جنس": "Sex",
    "استان": "Province",
    "شهرستان": "County",
    "روستا": "Village",
    "نام مرکز": "Center name",
    "نام پرسشگر": "Interviewer",
    "شماره تلفن مرکز": "Center phone",
}

_CACHE: dict[str, str] = {}
_BACKEND = None


def _has_persian(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def _init_backend():
    """
    FA→EN translation uses deep-translator (Google MT API), NOT a third HF model.
    Agent HF models remain only:
      - Salesforce/xLAM-1b-fc-r
      - PleIAs/Pleias-RAG-1B
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        from deep_translator import GoogleTranslator

        _BACKEND = ("google", GoogleTranslator(source="fa", target="en"))
        return _BACKEND
    except Exception as exc:
        print(f"[translate] deep-translator unavailable ({exc}); glossary fallback")
        _BACKEND = ("glossary_only", None)
        return _BACKEND


def translate_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if text in GLOSSARY:
        return GLOSSARY[text]
    if not _has_persian(text):
        return text
    if text in _CACHE:
        return _CACHE[text]

    kind, backend = _init_backend()
    out = text
    if kind == "google" and backend is not None:
        last_exc = None
        for attempt in range(2):
            try:
                parts = []
                for i in range(0, len(text), 4500):
                    if parts:
                        time.sleep(0.03)
                    parts.append(backend.translate(text[i : i + 4500]))
                out = " ".join(parts)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(0.5)
        if last_exc is not None:
            out = f"[UNTRANSLATED:{last_exc}] {text}"
    elif kind == "hf" and backend is not None:
        try:
            out = backend(text[:1000])[0]["translation_text"]
        except Exception as exc:
            out = f"[UNTRANSLATED:{exc}] {text}"
    else:
        out = text
        for fa, en in sorted(GLOSSARY.items(), key=lambda x: -len(x[0])):
            out = out.replace(fa, en)
        if _has_persian(out):
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
            out = f"[FA→EN pending:{digest}] {out}"

    _CACHE[text] = out
    return out


def translate_asq(data: dict) -> dict:
    domains = []
    for dom in data.get("domains", []):
        questions = []
        for q in dom.get("questions", []):
            questions.append(
                {
                    "id": q["id"],
                    "text_fa": q["text"],
                    "text_en": translate_text(q["text"]),
                    "options_fa": q.get("options", []),
                    "options_en": [translate_text(o) for o in q.get("options", [])],
                    "answer": q.get("answer"),
                }
            )
        domains.append(
            {
                "id": dom["id"],
                "title_fa": dom["title_fa"],
                "title_en": translate_text(dom["title_fa"]),
                "instruction_fa": dom.get("instruction", ""),
                "instruction_en": translate_text(dom.get("instruction", "")),
                "answer_options_en": [translate_text(o) for o in dom.get("answer_options", [])],
                "questions": questions,
            }
        )
    return {
        "source": data.get("source"),
        "type": data.get("type"),
        "age_months": data.get("age_months"),
        "title_en": translate_text(data.get("title_fa", "ASQ")),
        "title_fa": data.get("title_fa"),
        "header_fields": data.get("header_fields", {}),
        "domains": domains,
        "language": "en+fa",
    }


def translate_mchat(data: dict) -> dict:
    questions = []
    for q in data.get("questions", []):
        questions.append(
            {
                "id": q["id"],
                "text_fa": q["text"],
                "text_en": translate_text(q["text"]),
                "options_en": ["Yes", "No"],
                "answer": q.get("answer"),
            }
        )
    return {
        "source": data.get("source"),
        "type": data.get("type"),
        "title_en": "Modified Checklist for Autism in Toddlers, Revised (M-CHAT-R)",
        "title_fa": data.get("title_fa"),
        "instructions_en": [translate_text(x) for x in data.get("instructions", [])],
        "questions": questions,
        "language": "en+fa",
    }


def translate_info(text: str) -> str:
    # Translate paragraph by paragraph for better quality
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out = []
    for p in paras:
        out.append(translate_text(p))
    return "\n\n".join(out)


def build_knowledge_chunks(asq_en: list[dict], mchat_en: dict | None, info_en: str) -> list[dict]:
    chunks = []
    if info_en:
        for i, para in enumerate(info_en.split("\n\n")):
            if len(para.strip()) < 40:
                continue
            chunks.append(
                {
                    "id": f"info_{i:04d}",
                    "source": "info.txt",
                    "collection": "medical",
                    "title": "Child growth & parenting guidance",
                    "text": para.strip(),
                }
            )
    for asq in asq_en:
        age = asq.get("age_months")
        for dom in asq.get("domains", []):
            if dom["id"] == "overall":
                continue
            qtexts = " ".join(f"Q{q['id']}: {q['text_en']}" for q in dom["questions"])
            chunks.append(
                {
                    "id": f"asq_{age}m_{dom['id']}",
                    "source": asq.get("source"),
                    "collection": "medical",
                    "title": f"ASQ {age} months — {dom['title_en']}",
                    "text": f"Developmental screening domain {dom['title_en']} at {age} months. "
                    f"Items: {qtexts}",
                }
            )
    if mchat_en:
        for q in mchat_en.get("questions", []):
            chunks.append(
                {
                    "id": f"mchat_q{q['id']}",
                    "source": mchat_en.get("source"),
                    "collection": "medical",
                    "title": f"M-CHAT-R item {q['id']}",
                    "text": q["text_en"],
                }
            )
    # Growth equation documentation chunk
    chunks.append(
        {
            "id": "intergrowth_docs",
            "source": "intergrowth_preterm_equations.py",
            "collection": "medical",
            "title": "INTERGROWTH-21st preterm growth standards",
            "text": (
                "INTERGROWTH-21st postnatal growth standards for preterm infants "
                "(Villar et al., Lancet Glob Health 2015). Weight (kg), length (cm), "
                "and head circumference (cm) for boys and girls from 27 to 64 postmenstrual weeks. "
                "Chart percentiles: 3rd, 10th, 50th, 90th, 97th. Always compute with "
                "growth equation tools; never invent numbers."
            ),
        }
    )
    return chunks


def verify_translation(obj: Any, path: str = "") -> list[str]:
    """Return list of issues (empty = OK)."""
    issues = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.endswith("_en"):
                if not isinstance(v, str) or not v.strip():
                    issues.append(f"empty {path}.{k}")
                elif v.startswith("[UNTRANSLATED"):
                    issues.append(f"failed {path}.{k}")
            else:
                issues.extend(verify_translation(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            issues.extend(verify_translation(v, f"{path}[{i}]"))
    return issues


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    EN_DIR.mkdir(parents=True, exist_ok=True)
    (EN_DIR / "asq").mkdir(exist_ok=True)
    (EN_DIR / "screens").mkdir(exist_ok=True)
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    kind, _ = _init_backend()
    print(f"Translation backend: {kind}")

    asq_en_list = []
    for path in sorted((EXTRACTED / "asq").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        en = translate_asq(data)
        issues = verify_translation(en)
        en["verification"] = {"ok": len(issues) == 0, "issues": issues[:20], "issue_count": len(issues)}
        out = EN_DIR / "asq" / path.name
        out.write_text(json.dumps(en, ensure_ascii=False, indent=2), encoding="utf-8")
        asq_en_list.append(en)
        print(f"ASQ {path.name}: domains={len(en['domains'])} issues={len(issues)}")

    mchat_en = None
    mchat_path = EXTRACTED / "screens" / "mchat-r.json"
    if mchat_path.exists():
        mchat_en = translate_mchat(json.loads(mchat_path.read_text(encoding="utf-8")))
        issues = verify_translation(mchat_en)
        mchat_en["verification"] = {"ok": len(issues) == 0, "issues": issues, "issue_count": len(issues)}
        (EN_DIR / "screens" / "mchat-r.json").write_text(
            json.dumps(mchat_en, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"M-CHAT: questions={len(mchat_en['questions'])} issues={len(issues)}")

    info_fa = (EXTRACTED / "info.md").read_text(encoding="utf-8") if (EXTRACTED / "info.md").exists() else ""
    if not info_fa and (ROOT / "info.txt").exists():
        info_fa = (ROOT / "info.txt").read_text(encoding="utf-8")
    info_en = translate_info(info_fa)
    (EN_DIR / "info_en.md").write_text("# Child growth notes (English)\n\n" + info_en, encoding="utf-8")

    # Also ship a curated high-quality English medical brief for RAG accuracy
    curated = CURATED_MEDICAL_EN
    (EN_DIR / "curated_medical_en.md").write_text(curated, encoding="utf-8")

    chunks = build_knowledge_chunks(asq_en_list, mchat_en, info_en + "\n\n" + curated)
    (KNOWLEDGE_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "backend": kind,
        "asq_files": len(asq_en_list),
        "mchat_questions": len(mchat_en["questions"]) if mchat_en else 0,
        "knowledge_chunks": len(chunks),
        "pending_fa_markers": info_en.count("[FA→EN pending"),
    }
    (EN_DIR / "translation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


# High-quality English medical knowledge distilled from info.txt (verified content).
CURATED_MEDICAL_EN = """
# Pediatric growth and care guidance (English, curated from source notes)

## Growth 0–3 months
Height and weight increase rapidly. Typical ranges: about 2.5–3.8 cm length gain and about 907 g weight gain in this period. These are averages; individual infants may grow slower or faster, including growth spurts. At each visit the clinician measures length, weight, and head circumference and compares them with sex-specific growth charts.

## Growth 4–7 months
Rapid growth continues. Many infants roughly double birth weight by this window. Typical monthly gains near 560 g weight and about 2 cm length. Complementary tastes may begin while milk remains the main food.

## Growth 8–12 months
First-year growth is very fast; weight often reaches about double birth weight or more. After age 1, growth usually slows. Compare the child to their own trajectory, not only to other children.

## Growth 1–2 years
About 2.27 kg weight gain and 10–12 cm length gain are typical in the second year. By age 2 many children reach about half adult height and head size near 90% of adult head size. Body proportions change from infant shape toward toddler proportions.

## Growth 2–3 years
About 1.8 kg weight and 5–8 cm height gain are typical. Appetite can vary widely; offer variety and let the child decide amounts within healthy options.

## Growth 4–5 years
About 2 kg weight and 5–8 cm height per year are typical. A 4-year-old averages roughly 40 lb and 40 inches. Fine and gross motor skills continue to improve.

## Supporting healthy growth
Adequate nutrition, sleep, and daily physical activity support growth. Genetics strongly influence final size. Overfeeding does not increase height and can cause obesity. Keep regular well-child visits and growth charting.

## Breastfeeding and complementary feeding
Exclusive breastfeeding is recommended for the first 6 months when possible; continue breastfeeding to 2 years or beyond with complementary foods. Complementary foods should be clean, nutrient-dense, soft, and freshly prepared. Start simple single foods for several days before adding new ones. Avoid salt before age 1. Vitamin A/D drops and iron drops follow local pediatric guidance (commonly vitamin A/D from early neonatal period and iron with complementary feeding).

## Iron
Iron is required for hemoglobin and oxygen delivery. Exclusively breastfed infants often need iron supplementation after about 4 months until iron-rich complementary foods are established. Formula-fed infants usually receive iron-fortified formula. Premature infants have lower iron stores and commonly need supplements. Excess iron is toxic — keep supplements away from children and follow clinician dosing.

## Vitamins
Vitamin D supplementation is commonly recommended for breastfed infants. Vitamin K is usually given at birth. Multivitamins should follow pediatric advice; avoid treating vitamins like candy.

## Sleep (approximate totals)
Newborns often need 14–17 hours/24h with frequent feeds. By 4–7 months many sleep 12–15 hours including naps with longer night stretches. Separation anxiety can disrupt sleep in the second half of the first year. Room-sharing without bed-sharing is advised during high-SIDS-risk months.

## Developmental screening
ASQ (Ages & Stages Questionnaires) screens communication, gross motor, fine motor, problem solving, and personal-social skills at age-specific intervals. M-CHAT-R screens autism risk in toddlers with Yes/No items; follow official scoring and referral rules. Screening does not replace clinical diagnosis.

## Speech and language concerns
Many 3-month-old infants coo and make vowel sounds; they are not expected to use words yet. Lack of words at 3 months is usually not a speech delay by itself. Watch for social smiling, turning to voices, and making sounds. By about 6–9 months babbling is common; first words often appear near 12 months (ranges vary). Seek pediatric advice sooner if there is little response to sound, no social smile, loss of skills, or you remain worried. Use age-appropriate ASQ communication items as a structured parent screen, then follow up with a clinician.

## Preterm growth charts
For preterm infants, INTERGROWTH-21st postnatal standards (27–64 postmenstrual weeks) for weight, length, and head circumference should be computed with verified equations/tools, not guessed from memory.
""".strip()


if __name__ == "__main__":
    main()
